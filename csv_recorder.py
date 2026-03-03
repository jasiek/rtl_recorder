#!/usr/bin/env python3
"""Record custom channels from CSV with per-channel squelch and label-based output names."""

from __future__ import annotations

import argparse
import csv
import math
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi

from recorder.dsp import PolyphaseFftChannelizer, compute_monitor_db, mix_with_oscillator, multistage_decimate, power_db
from recorder.models import ChannelState
from recorder.pipeline import open_configured_sdr
from recorder.sink import close_resources, open_wav, write_audio_to_wav
from recorder.source import configure_library_environment
from recorder.squelch import channel_should_record, coarse_should_process_channel

MAX_FREQUENCY_SPAN_HZ = 2_000_000.0


@dataclass(frozen=True)
class ChannelSpec:
    label: str
    frequency_hz: float
    width_hz: float


@dataclass(frozen=True)
class ChannelRuntime:
    spec: ChannelSpec
    state: ChannelState
    output_path: Path
    post1_lpf_taps: np.ndarray
    post2_lpf_taps: np.ndarray


@dataclass(frozen=True)
class RuntimeConfig:
    freqs: List[float]
    center_hz: float
    decimation: int
    coarse_decimation: int
    coarse_sample_rate: int
    post_decimation: int
    post_decimation_stage1: int
    post_decimation_stage2: int
    offsets: List[float]
    bin_offsets: np.ndarray
    monitor_bin_index: int
    monitor_residual_step: complex
    squelch_hold_samples: int
    squelch_cal_samples: int


def install_signal_handlers() -> callable:
    running = True

    def _handler(signum: int, _frame) -> None:
        nonlocal running
        if running:
            print(f"\nReceived signal {signum}; stopping and closing WAV files...", flush=True)
        running = False

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return lambda: running


def parse_frequency_hz(raw: str) -> float:
    text = raw.strip().replace("_", "")
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kKmMgG]?[hH]?[zZ]?)?", text)
    if not m:
        raise ValueError(f"invalid frequency '{raw}'")
    value = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if suffix in {"", "hz"}:
        return value if value >= 1_000_000.0 else value * 1_000_000.0
    if suffix in {"k", "khz"}:
        return value * 1_000.0
    if suffix in {"m", "mhz"}:
        return value * 1_000_000.0
    if suffix in {"g", "ghz"}:
        return value * 1_000_000_000.0
    raise ValueError(f"invalid frequency suffix in '{raw}'")


def parse_width_hz(raw: str) -> float:
    text = raw.strip().replace("_", "")
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kKmM]?[hH]?[zZ]?)?", text)
    if not m:
        raise ValueError(f"invalid width '{raw}'")
    value = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if suffix in {"", "hz"}:
        width = value if value >= 1_000.0 else value * 1_000.0
    elif suffix in {"k", "khz"}:
        width = value * 1_000.0
    elif suffix in {"m", "mhz"}:
        width = value * 1_000_000.0
    else:
        raise ValueError(f"invalid width suffix in '{raw}'")
    if width <= 0:
        raise ValueError(f"width must be > 0, got '{raw}'")
    return width


def load_channels(csv_path: Path) -> List[ChannelSpec]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"label", "frequency", "width"}
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header row")
        got = {h.strip().lower() for h in reader.fieldnames}
        if got != expected:
            raise ValueError("CSV header must be exactly: label,frequency,width")
        channels: List[ChannelSpec] = []
        for row_idx, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip()
            if not label:
                raise ValueError(f"row {row_idx}: label is required")
            try:
                frequency_hz = parse_frequency_hz(row["frequency"])
                width_hz = parse_width_hz(row["width"])
            except ValueError as exc:
                raise ValueError(f"row {row_idx}: {exc}") from exc
            channels.append(ChannelSpec(label=label, frequency_hz=frequency_hz, width_hz=width_hz))
    if not channels:
        raise ValueError("CSV contains no channels")
    return channels


def validate_frequency_span(channels: List[ChannelSpec]) -> None:
    freqs = sorted(ch.frequency_hz for ch in channels)
    span_hz = freqs[-1] - freqs[0]
    if span_hz <= MAX_FREQUENCY_SPAN_HZ:
        return
    raise ValueError(
        f"frequency span is {span_hz / 1e6:.6f} MHz (> 2.000000 MHz); "
        "all channels must be within 2 MHz of each other"
    )


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()).strip("._-")
    return cleaned or "channel"


def build_runtime_config(args: argparse.Namespace, channels: List[ChannelSpec]) -> RuntimeConfig:
    if args.sample_rate % args.audio_rate != 0:
        raise ValueError("sample-rate must be an integer multiple of audio-rate")

    decimation = args.sample_rate // args.audio_rate
    if decimation < 2:
        raise ValueError("sample-rate/audio-rate decimation must be >= 2")
    if decimation % 2 != 0:
        raise ValueError("sample-rate/audio-rate decimation must be an even integer for the channelizer")

    freqs = [ch.frequency_hz for ch in channels]
    min_freq = min(freqs)
    max_freq = max(freqs)
    center_hz = (min_freq + max_freq) / 2.0
    offsets = [f - center_hz for f in freqs]

    max_width_hz = max(ch.width_hz for ch in channels)
    monitor_hz = args.squelch_monitor_hz
    if monitor_hz is None:
        monitor_hz = min_freq - max(25_000.0, max_width_hz * 1.5)
    squelch_offset = monitor_hz - center_hz

    max_capture_offset = 0.0
    for off, ch in zip(offsets, channels):
        max_capture_offset = max(max_capture_offset, abs(off) + (ch.width_hz / 2.0))
    max_capture_offset = max(max_capture_offset, abs(squelch_offset) + (max_width_hz / 2.0))
    if max_capture_offset > (args.sample_rate / 2.0):
        raise ValueError("sample-rate too low to cover channels and squelch monitor simultaneously; increase --sample-rate")

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
    squelch_hold_samples = max(0, int(args.audio_rate * (args.squelch_hold_ms / 1000.0)))
    squelch_cal_samples = max(0, int(args.audio_rate * args.squelch_cal_seconds))

    args.squelch_monitor_hz = monitor_hz

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
        squelch_hold_samples=squelch_hold_samples,
        squelch_cal_samples=squelch_cal_samples,
    )


def init_channel_states(args: argparse.Namespace, cfg: RuntimeConfig, channels: List[ChannelSpec]) -> List[ChannelRuntime]:
    runtimes: List[ChannelRuntime] = []
    used_names: dict[str, int] = {}

    for idx, (off, ch) in enumerate(zip(cfg.offsets, channels), start=1):
        coarse_bin_index = int(np.argmin(np.abs(cfg.bin_offsets - off)))
        coarse_bin_center = float(cfg.bin_offsets[coarse_bin_index])
        residual_hz = off - coarse_bin_center

        post1_lpf_taps = firwin(
            numtaps=33,
            cutoff=min(6000.0, ch.width_hz * 0.48),
            fs=cfg.coarse_sample_rate,
        ).astype(np.float32)
        post2_sample_rate = cfg.coarse_sample_rate // cfg.post_decimation_stage1
        post2_lpf_taps = firwin(
            numtaps=33,
            cutoff=min(6000.0, ch.width_hz * 0.48),
            fs=post2_sample_rate,
        ).astype(np.float32)
        post1_zi = lfilter_zi(post1_lpf_taps, [1.0]).astype(np.complex64)
        post2_zi = lfilter_zi(post2_lpf_taps, [1.0]).astype(np.complex64)

        base_name = sanitize_label(ch.label)
        count = used_names.get(base_name, 0) + 1
        used_names[base_name] = count
        file_name = f"{base_name}.wav" if count == 1 else f"{base_name}_{count}.wav"
        out = args.output_dir / file_name

        st = ChannelState(
            number=idx,
            offset_hz=off,
            coarse_bin_index=coarse_bin_index,
            residual_osc=np.complex64(1.0 + 0.0j),
            residual_step=np.complex64(np.exp(-1j * (2.0 * np.pi * residual_hz / cfg.coarse_sample_rate))),
            post1_zi=post1_zi,
            post2_zi=post2_zi,
            prev_sample=0.0 + 0.0j,
            deemp_last=0.0,
            wav=open_wav(out, args.audio_rate),
            squelch_open=False,
            squelch_hold_samples=0,
            squelch_delta_floor_db=None,
            coarse_delta_floor_db=None,
            squelch_cal_remaining_samples=cfg.squelch_cal_samples,
        )
        runtimes.append(
            ChannelRuntime(
                spec=ch,
                state=st,
                output_path=out,
                post1_lpf_taps=post1_lpf_taps,
                post2_lpf_taps=post2_lpf_taps,
            )
        )
    return runtimes


def print_runtime_summary(args: argparse.Namespace, cfg: RuntimeConfig, runtimes: List[ChannelRuntime]) -> None:
    print("Band plan:          CSV custom")
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
            f"Squelch monitor:    {args.squelch_monitor_hz / 1e6:.6f} MHz "
            f"(coarse pre-open +{args.squelch_coarse_open_db:.1f} dB, fine open +{args.squelch_open_db:.1f} dB, close +{args.squelch_close_db:.1f} dB)"
        )
    print("Recording channels:")
    for rt in runtimes:
        print(
            f"  CH {rt.state.number:02d}: {rt.spec.label} "
            f"({rt.spec.frequency_hz / 1e6:.6f} MHz, {rt.spec.width_hz / 1e3:.2f} kHz) -> {rt.output_path}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record channels from CSV (label,frequency,width) into per-label WAV files")
    p.add_argument("csv_file", type=Path, help="CSV file with header: label,frequency,width")
    p.add_argument("--device-index", "--device", dest="device_index", type=int, default=0, help="RTL-SDR device index (default: 0)")
    p.add_argument("--output-dir", type=Path, default=Path("recordings-csv"), help="Output directory")
    p.add_argument("--gain", type=float, default=30.0, help="Tuner gain in dB (default: 30)")
    p.add_argument("--sample-rate", type=int, default=256_000, help="SDR sample rate (default: 256000)")
    p.add_argument("--audio-rate", type=int, default=16_000, help="Per-channel WAV sample rate (default: 16000)")
    p.add_argument("--chunk-size", type=int, default=65_536, help="IQ samples per read (default: 65536)")
    p.add_argument("--audio-gain", type=float, default=3.0, help="Post-demod audio gain multiplier before WAV conversion (default: 3.0)")
    p.add_argument("--squelch-monitor-hz", type=float, default=None, help="Override monitor frequency in Hz (default: auto below the minimum channel)")
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


def run(args: argparse.Namespace) -> int:
    try:
        channels = load_channels(args.csv_file)
        validate_frequency_span(channels)
        cfg = build_runtime_config(args, channels)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    should_continue = install_signal_handlers()

    sdr = None
    runtimes: List[ChannelRuntime] = []
    states: List[ChannelState] = []

    try:
        sdr = open_configured_sdr(args.device_index, args.sample_rate, cfg.center_hz, args.gain)
        runtimes = init_channel_states(args, cfg, channels)
        states = [rt.state for rt in runtimes]
        print_runtime_summary(args, cfg, runtimes)

        channelizer = PolyphaseFftChannelizer(sample_rate=args.sample_rate, decimation=cfg.coarse_decimation)

        monitor_cutoff = min(6000.0, min(ch.width_hz for ch in channels) * 0.48)
        monitor_post1_lpf_taps = firwin(numtaps=33, cutoff=monitor_cutoff, fs=cfg.coarse_sample_rate).astype(np.float32)
        monitor_post2_sample_rate = cfg.coarse_sample_rate // cfg.post_decimation_stage1
        monitor_post2_lpf_taps = firwin(numtaps=33, cutoff=monitor_cutoff, fs=monitor_post2_sample_rate).astype(np.float32)
        monitor_post1_zi = lfilter_zi(monitor_post1_lpf_taps, [1.0]).astype(np.complex64)
        monitor_post2_zi = lfilter_zi(monitor_post2_lpf_taps, [1.0]).astype(np.complex64)
        monitor_mixer_osc = np.complex64(1.0 + 0.0j)

        while should_continue():
            iq = sdr.read_samples(args.chunk_size).astype(np.complex64, copy=False)
            if iq.size == 0:
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
                    post1_lpf_taps=monitor_post1_lpf_taps,
                    post2_lpf_taps=monitor_post2_lpf_taps,
                    decimation_stage1=cfg.post_decimation_stage1,
                    decimation_stage2=cfg.post_decimation_stage2,
                )

            for rt in runtimes:
                st = rt.state
                if not args.no_squelch:
                    coarse_stream = coarse_bins[st.coarse_bin_index]
                    if not coarse_should_process_channel(st=st, coarse_stream=coarse_stream, monitor_coarse_db=monitor_coarse_db, args=args):
                        continue

                coarse = coarse_bins[st.coarse_bin_index]
                shifted, st.residual_osc = mix_with_oscillator(
                    coarse,
                    np.complex64(st.residual_osc),
                    np.complex64(st.residual_step),
                )
                narrow = multistage_decimate(
                    shifted=shifted,
                    post1_lpf_taps=rt.post1_lpf_taps,
                    post2_lpf_taps=rt.post2_lpf_taps,
                    post1_zi=st.post1_zi,
                    post2_zi=st.post2_zi,
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


def main() -> int:
    configure_library_environment()
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
