from __future__ import annotations

import argparse
import math
import signal
import sys
from pathlib import Path
from typing import List

import numpy as np
from scipy.signal import firwin, lfilter_zi

from recorder.dsp import PolyphaseFftChannelizer, compute_monitor_db, extract_narrowband_channel, power_db
from recorder.models import BandPlan, ChannelState, RuntimeConfig
from recorder.sink import close_resources, open_wav, write_audio_to_wav
from recorder.source import CompatRtlSdr, configure_library_environment
from recorder.squelch import channel_should_record, coarse_should_process_channel

running = True


def install_signal_handlers() -> None:
    def _handler(signum: int, _frame) -> None:
        global running
        if running:
            print(f"\\nReceived signal {signum}; stopping and closing WAV files...", flush=True)
        running = False

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def channel_frequencies(plan: BandPlan) -> List[float]:
    return [plan.first_channel_hz + i * plan.channel_spacing_hz for i in range(plan.channel_count)]


def build_runtime_config(args: argparse.Namespace, plan: BandPlan) -> RuntimeConfig:
    # Validate rate relationships and derive all fixed DSP constants once at startup.
    if args.sample_rate % args.audio_rate != 0:
        raise ValueError("sample-rate must be an integer multiple of audio-rate")

    decimation = args.sample_rate // args.audio_rate
    if decimation < 2:
        raise ValueError("sample-rate/audio-rate decimation must be >= 2")
    if decimation % 2 != 0:
        raise ValueError("sample-rate/audio-rate decimation must be an even integer for the channelizer")

    freqs = channel_frequencies(plan)
    center_hz = (freqs[0] + freqs[-1]) / 2.0
    offsets = [f - center_hz for f in freqs]
    squelch_offset = plan.squelch_monitor_hz - center_hz
    max_offset = max(abs(o) for o in offsets)
    max_capture_offset = max(max_offset, abs(squelch_offset))
    if max_capture_offset + (plan.channel_width_hz / 2.0) > (args.sample_rate / 2.0):
        raise ValueError("sample-rate too low to cover all channels and squelch monitor simultaneously; increase --sample-rate")

    if decimation % 8 == 0:
        coarse_decimation = 8
    elif decimation % 4 == 0:
        coarse_decimation = 4
    else:
        coarse_decimation = 2
    coarse_sample_rate = args.sample_rate // coarse_decimation
    post_decimation = decimation // coarse_decimation
    if post_decimation < 1:
        raise ValueError("internal channelizer post-decimation became invalid")
    if post_decimation % 4 == 0:
        post_decimation_stage1 = 4
    elif post_decimation % 2 == 0:
        post_decimation_stage1 = 2
    else:
        post_decimation_stage1 = 1
    post_decimation_stage2 = post_decimation // post_decimation_stage1

    bin_offsets = np.fft.fftfreq(coarse_decimation, d=1.0 / args.sample_rate).astype(np.float32)

    monitor_bin_index = int(np.argmin(np.abs(bin_offsets - squelch_offset)))
    monitor_bin_center = float(bin_offsets[monitor_bin_index])
    monitor_residual_hz = squelch_offset - monitor_bin_center
    monitor_residual_step = np.complex64(np.exp(-1j * (2.0 * np.pi * monitor_residual_hz / coarse_sample_rate)))

    lpf_cutoff = min(6000.0, plan.channel_width_hz * 0.48)
    post1_lpf_taps = firwin(numtaps=33, cutoff=lpf_cutoff, fs=coarse_sample_rate).astype(np.float32)
    post2_sample_rate = coarse_sample_rate // post_decimation_stage1
    post2_lpf_taps = firwin(numtaps=33, cutoff=lpf_cutoff, fs=post2_sample_rate).astype(np.float32)
    squelch_hold_samples = max(0, int(args.audio_rate * (args.squelch_hold_ms / 1000.0)))
    squelch_cal_samples = max(0, int(args.audio_rate * args.squelch_cal_seconds))

    return RuntimeConfig(
        freqs=freqs,
        center_hz=center_hz,
        decimation=decimation,
        coarse_decimation=coarse_decimation,
        coarse_sample_rate=coarse_sample_rate,
        post_decimation=post_decimation,
        post_decimation_stage1=post_decimation_stage1,
        post_decimation_stage2=post_decimation_stage2,
        offsets=offsets,
        bin_offsets=bin_offsets,
        monitor_bin_index=monitor_bin_index,
        monitor_residual_step=monitor_residual_step,
        post1_lpf_taps=post1_lpf_taps,
        post2_lpf_taps=post2_lpf_taps,
        squelch_hold_samples=squelch_hold_samples,
        squelch_cal_samples=squelch_cal_samples,
    )


def open_configured_sdr(device_index: int, sample_rate: int, center_hz: float, gain_db: float) -> CompatRtlSdr:
    # Enumerate first so device-index errors are explicit before opening hardware.
    detected = CompatRtlSdr.list_devices()
    print(f"Detected RTL-SDR devices: {len(detected)}")
    for i, name in enumerate(detected):
        print(f"  [{i}] {name}")
    if not detected:
        raise RuntimeError("No RTL-SDR devices detected. Check USB connection, cable/hub power, and that no other SDR app is using the dongle.")
    if device_index < 0 or device_index >= len(detected):
        raise RuntimeError(f"Invalid --device-index {device_index}; detected indices are 0..{len(detected)-1}")

    sdr = CompatRtlSdr(device_index)
    sdr.sample_rate = float(sample_rate)
    sdr.center_freq = float(center_hz)
    sdr.gain = float(gain_db)
    return sdr


def print_runtime_summary(args: argparse.Namespace, plan: BandPlan, cfg: RuntimeConfig) -> None:
    print(f"Band plan:          {plan.name}")
    print(f"RTL-SDR device:     {args.device_index}")
    print(f"Center frequency:   {cfg.center_hz / 1e6:.6f} MHz")
    print(f"Sample rate:        {args.sample_rate} sps")
    print(f"Audio/WAV rate:     {args.audio_rate} Hz")
    print(f"Decimation:         {cfg.decimation}x")
    print(
        f"Channelizer:        {cfg.coarse_decimation}-way PFB + FFT, then "
        f"{cfg.post_decimation_stage1}x/{cfg.post_decimation_stage2}x post-decimation"
    )
    if args.no_squelch:
        print("Squelch:            disabled (continuous recording)")
    else:
        print(
            f"Squelch monitor:    {plan.squelch_monitor_hz / 1e6:.6f} MHz "
            f"(coarse pre-open +{args.squelch_coarse_open_db:.1f} dB, fine open +{args.squelch_open_db:.1f} dB, close +{args.squelch_close_db:.1f} dB)"
        )
    print("Recording channels:")


def init_channel_states(args: argparse.Namespace, cfg: RuntimeConfig) -> List[ChannelState]:
    # Each channel keeps independent DSP/squelch state so recording gates are per-channel.
    post1_zi_template = lfilter_zi(cfg.post1_lpf_taps, [1.0]).astype(np.complex64)
    post2_zi_template = lfilter_zi(cfg.post2_lpf_taps, [1.0]).astype(np.complex64)
    states: List[ChannelState] = []
    for i, off in enumerate(cfg.offsets, start=1):
        coarse_bin_index = int(np.argmin(np.abs(cfg.bin_offsets - off)))
        coarse_bin_center = float(cfg.bin_offsets[coarse_bin_index])
        residual_hz = off - coarse_bin_center
        out = args.output_dir / f"{i}.wav"
        st = ChannelState(
            number=i,
            offset_hz=off,
            coarse_bin_index=coarse_bin_index,
            residual_osc=np.complex64(1.0 + 0.0j),
            residual_step=np.complex64(np.exp(-1j * (2.0 * np.pi * residual_hz / cfg.coarse_sample_rate))),
            post1_zi=post1_zi_template.copy(),
            post2_zi=post2_zi_template.copy(),
            prev_sample=0.0 + 0.0j,
            deemp_last=0.0,
            wav=open_wav(out, args.audio_rate),
            squelch_open=False,
            squelch_hold_samples=0,
            squelch_delta_floor_db=None,
            coarse_delta_floor_db=None,
            squelch_cal_remaining_samples=cfg.squelch_cal_samples,
        )
        states.append(st)
        print(f"  CH {i:02d}: {cfg.freqs[i - 1] / 1e6:.5f} MHz -> {out}")
    return states


def build_parser(plan: BandPlan) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Record {plan.name} channels into per-channel WAV files")
    p.add_argument("--device-index", "--device", dest="device_index", type=int, default=0, help="RTL-SDR device index (default: 0)")
    p.add_argument("--output-dir", type=Path, default=Path(plan.default_output_dir), help="Output directory")
    p.add_argument("--gain", type=float, default=30.0, help="Tuner gain in dB (default: 30)")
    p.add_argument("--sample-rate", type=int, default=plan.default_sample_rate, help=f"SDR sample rate (default: {plan.default_sample_rate})")
    p.add_argument("--audio-rate", type=int, default=16_000, help="Per-channel WAV sample rate (default: 16000)")
    p.add_argument("--chunk-size", type=int, default=65_536, help="IQ samples per read (default: 65536)")
    p.add_argument("--audio-gain", type=float, default=3.0, help="Post-demod audio gain multiplier before WAV conversion (default: 3.0)")
    p.add_argument("--no-squelch", action="store_true", help="Disable squelch logic and record continuously")
    p.add_argument(
        "--squelch-coarse-open-db",
        type=float,
        default=2.0,
        help="First-stage gate: run full per-channel DSP only when coarse channel power exceeds baseline by this many dB (default: 2.0)",
    )
    p.add_argument("--squelch-open-db", type=float, default=4.0, help="Per-channel: open when channel power rises this many dB above monitor channel power (default: 4)")
    p.add_argument("--squelch-close-db", type=float, default=2.0, help="Per-channel: close when channel power falls below this many dB above monitor channel power (default: 2)")
    p.add_argument("--squelch-hold-ms", type=int, default=300, help="Hold squelch open this long after signal drops (default: 300 ms)")
    p.add_argument("--squelch-cal-seconds", type=float, default=2.0, help="Initial squelch calibration duration while forced closed (default: 2.0 s)")
    return p


def run_recorder(args: argparse.Namespace, plan: BandPlan) -> int:
    global running
    running = True

    try:
        # Precompute immutable runtime DSP config before touching hardware.
        cfg = build_runtime_config(args, plan)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sdr = None
    states: List[ChannelState] = []
    install_signal_handlers()

    try:
        # 1) Open/configure hardware and print session summary.
        sdr = open_configured_sdr(args.device_index, args.sample_rate, cfg.center_hz, args.gain)
        print_runtime_summary(args, plan, cfg)
        states = init_channel_states(args, cfg)

        # 2) Initialize monitor DSP state used as squelch reference.
        channelizer = PolyphaseFftChannelizer(sample_rate=args.sample_rate, decimation=cfg.coarse_decimation)
        monitor_post1_zi = lfilter_zi(cfg.post1_lpf_taps, [1.0]).astype(np.complex64)
        monitor_post2_zi = lfilter_zi(cfg.post2_lpf_taps, [1.0]).astype(np.complex64)
        monitor_mixer_osc = np.complex64(1.0 + 0.0j)

        # 3) Stream IQ chunks, evaluate squelch, and write per-channel WAV audio.
        while running:
            # Everything below is hot-path: avoid per-iteration recomputation where possible.
            iq = sdr.read_samples(args.chunk_size).astype(np.complex64, copy=False)
            n = iq.size
            if n == 0:
                continue
            coarse_bins = channelizer.process(iq)
            if coarse_bins.shape[1] == 0:
                continue

            monitor_db = 0.0
            monitor_coarse_db = 0.0
            if not args.no_squelch:
                monitor_coarse_db = power_db(coarse_bins[cfg.monitor_bin_index])
                monitor_db, monitor_mixer_osc, monitor_post1_zi, monitor_post2_zi = compute_monitor_db(
                    coarse_bins=coarse_bins,
                    monitor_bin_index=cfg.monitor_bin_index,
                    mixer_osc=monitor_mixer_osc,
                    mixer_step=cfg.monitor_residual_step,
                    post1_zi=monitor_post1_zi,
                    post2_zi=monitor_post2_zi,
                    post1_lpf_taps=cfg.post1_lpf_taps,
                    post2_lpf_taps=cfg.post2_lpf_taps,
                    decimation_stage1=cfg.post_decimation_stage1,
                    decimation_stage2=cfg.post_decimation_stage2,
                )

            for st in states:
                if not args.no_squelch:
                    coarse_stream = coarse_bins[st.coarse_bin_index]
                    if not coarse_should_process_channel(
                        st=st,
                        coarse_stream=coarse_stream,
                        monitor_coarse_db=monitor_coarse_db,
                        args=args,
                    ):
                        continue

                narrow = extract_narrowband_channel(
                    coarse_bins=coarse_bins,
                    st=st,
                    post1_lpf_taps=cfg.post1_lpf_taps,
                    post2_lpf_taps=cfg.post2_lpf_taps,
                    decimation_stage1=cfg.post_decimation_stage1,
                    decimation_stage2=cfg.post_decimation_stage2,
                )

                if not args.no_squelch:
                    if not channel_should_record(
                        st=st,
                        narrow=narrow,
                        monitor_db=monitor_db,
                        args=args,
                        squelch_hold_samples=cfg.squelch_hold_samples,
                    ):
                        continue

                write_audio_to_wav(st, narrow, args.audio_rate, args.audio_gain)

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"fatal error: {exc}", file=sys.stderr)
        return 1
    finally:
        close_resources(states, sdr)
        print("Shutdown complete. WAV files closed.")

    return 0


def main_for_plan(plan: BandPlan) -> int:
    configure_library_environment()
    parser = build_parser(plan)
    args = parser.parse_args()
    return run_recorder(args, plan)
