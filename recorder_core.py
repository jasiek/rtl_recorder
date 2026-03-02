#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from functools import lru_cache
import math
import os
import signal
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi


@dataclass(frozen=True)
class BandPlan:
    name: str
    channel_count: int
    first_channel_hz: float
    channel_spacing_hz: float
    channel_width_hz: float
    squelch_monitor_hz: float
    default_output_dir: str
    default_sample_rate: int


@dataclass
class ChannelState:
    number: int
    offset_hz: float
    mixer_osc: complex
    mixer_step: complex
    lpf_zi: np.ndarray
    prev_sample: complex
    deemp_last: float
    wav: wave.Wave_write
    squelch_open: bool
    squelch_hold_samples: int
    squelch_delta_floor_db: float | None
    squelch_cal_remaining_samples: int


running = True


def _candidate_rtlsdr_paths() -> List[Path]:
    candidates: List[Path] = []
    env = os.environ.get("RTLSDR_LIB_PATH")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/opt/homebrew/lib/librtlsdr.2.0.1.dylib"),
            Path("/opt/homebrew/lib/librtlsdr.dylib"),
            Path("/opt/homebrew/lib/librtlsdr.0.dylib"),
            Path("/usr/local/lib/librtlsdr.dylib"),
            Path("/usr/local/lib/librtlsdr.0.dylib"),
            Path("/usr/lib/librtlsdr.so"),
            Path("/usr/lib/x86_64-linux-gnu/librtlsdr.so"),
        ]
    )
    return [p for p in candidates if p.exists()]


def _prepend_env_path(var_name: str, new_path: str) -> None:
    current = os.environ.get(var_name, "")
    if not current:
        os.environ[var_name] = new_path
        return
    parts = current.split(":")
    if new_path in parts:
        return
    os.environ[var_name] = f"{new_path}:{current}"


def configure_library_environment() -> None:
    # Best-effort setup so scripts can run without external wrapper logic.
    candidates = _candidate_rtlsdr_paths()
    if not candidates:
        return
    lib_path = str(candidates[0])
    os.environ["RTLSDR_LIB_PATH"] = lib_path
    lib_dir = str(Path(lib_path).parent)
    _prepend_env_path("DYLD_LIBRARY_PATH", lib_dir)
    _prepend_env_path("LD_LIBRARY_PATH", lib_dir)


def _load_librtlsdr() -> ctypes.CDLL:
    last_error = None
    for lib in _candidate_rtlsdr_paths():
        try:
            return ctypes.CDLL(str(lib))
        except OSError as exc:
            last_error = exc

    for name in ("librtlsdr.dylib", "librtlsdr.so", "librtlsdr.dll"):
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            last_error = exc

    hint = ""
    libs = _candidate_rtlsdr_paths()
    if libs:
        hint = f" Detected candidate: {libs[0]}"
    raise RuntimeError(
        "Could not load librtlsdr. Install librtlsdr and/or set RTLSDR_LIB_PATH."
        + hint
        + (f" Last error: {last_error}" if last_error else "")
    )


class CompatRtlSdr:
    def __init__(self, device_index: int = 0) -> None:
        self.lib = _load_librtlsdr()
        self._setup_api()
        self.dev = ctypes.c_void_p()
        rc = self.lib.rtlsdr_open(ctypes.byref(self.dev), ctypes.c_uint32(device_index))
        if rc != 0:
            raise RuntimeError(f"rtlsdr_open failed with code {rc}")
        self._closed = False
        self._read_buf = None
        self._read_buf_len = 0
        self.lib.rtlsdr_reset_buffer(self.dev)

    def _setup_api(self) -> None:
        self.lib.rtlsdr_get_device_count.argtypes = []
        self.lib.rtlsdr_get_device_count.restype = ctypes.c_uint32
        self.lib.rtlsdr_get_device_name.argtypes = [ctypes.c_uint32]
        self.lib.rtlsdr_get_device_name.restype = ctypes.c_char_p

        self.lib.rtlsdr_open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
        self.lib.rtlsdr_open.restype = ctypes.c_int
        self.lib.rtlsdr_close.argtypes = [ctypes.c_void_p]
        self.lib.rtlsdr_close.restype = ctypes.c_int

        self.lib.rtlsdr_set_center_freq.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.rtlsdr_set_center_freq.restype = ctypes.c_int
        self.lib.rtlsdr_set_sample_rate.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.rtlsdr_set_sample_rate.restype = ctypes.c_int

        self.lib.rtlsdr_set_tuner_gain_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.rtlsdr_set_tuner_gain_mode.restype = ctypes.c_int
        self.lib.rtlsdr_set_tuner_gain.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.rtlsdr_set_tuner_gain.restype = ctypes.c_int

        self.lib.rtlsdr_reset_buffer.argtypes = [ctypes.c_void_p]
        self.lib.rtlsdr_reset_buffer.restype = ctypes.c_int
        self.lib.rtlsdr_read_sync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.rtlsdr_read_sync.restype = ctypes.c_int

    @classmethod
    def list_devices(cls) -> List[str]:
        lib = _load_librtlsdr()
        lib.rtlsdr_get_device_count.argtypes = []
        lib.rtlsdr_get_device_count.restype = ctypes.c_uint32
        lib.rtlsdr_get_device_name.argtypes = [ctypes.c_uint32]
        lib.rtlsdr_get_device_name.restype = ctypes.c_char_p
        count = int(lib.rtlsdr_get_device_count())
        names: List[str] = []
        for i in range(count):
            raw = lib.rtlsdr_get_device_name(ctypes.c_uint32(i))
            names.append(raw.decode("utf-8", errors="replace") if raw else f"device-{i}")
        return names

    @property
    def sample_rate(self) -> float:
        raise AttributeError("sample_rate getter not implemented")

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        rc = self.lib.rtlsdr_set_sample_rate(self.dev, ctypes.c_uint32(int(value)))
        if rc != 0:
            raise RuntimeError(f"rtlsdr_set_sample_rate failed with code {rc}")

    @property
    def center_freq(self) -> float:
        raise AttributeError("center_freq getter not implemented")

    @center_freq.setter
    def center_freq(self, value: float) -> None:
        rc = self.lib.rtlsdr_set_center_freq(self.dev, ctypes.c_uint32(int(value)))
        if rc != 0:
            raise RuntimeError(f"rtlsdr_set_center_freq failed with code {rc}")

    @property
    def gain(self) -> float:
        raise AttributeError("gain getter not implemented")

    @gain.setter
    def gain(self, value: float) -> None:
        rc = self.lib.rtlsdr_set_tuner_gain_mode(self.dev, 1)
        if rc != 0:
            raise RuntimeError(f"rtlsdr_set_tuner_gain_mode failed with code {rc}")
        rc = self.lib.rtlsdr_set_tuner_gain(self.dev, int(round(value * 10.0)))
        if rc != 0:
            raise RuntimeError(f"rtlsdr_set_tuner_gain failed with code {rc}")

    def read_samples(self, count: int) -> np.ndarray:
        byte_count = int(count) * 2
        if self._read_buf is None or self._read_buf_len != byte_count:
            self._read_buf = (ctypes.c_ubyte * byte_count)()
            self._read_buf_len = byte_count
        buf = self._read_buf
        n_read = ctypes.c_int(0)
        rc = self.lib.rtlsdr_read_sync(self.dev, buf, byte_count, ctypes.byref(n_read))
        if rc != 0:
            raise RuntimeError(f"rtlsdr_read_sync failed with code {rc}")
        got = int(n_read.value)
        if got <= 1:
            return np.zeros(0, dtype=np.complex64)
        got -= got % 2
        raw = np.frombuffer(buf, dtype=np.uint8, count=got).astype(np.float32)
        i = (raw[0::2] - 127.5) / 127.5
        q = (raw[1::2] - 127.5) / 127.5
        return (i + 1j * q).astype(np.complex64)

    def close(self) -> None:
        if not self._closed:
            self.lib.rtlsdr_close(self.dev)
            self._closed = True


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


def open_wav(path: Path, sample_rate: int) -> wave.Wave_write:
    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    return w


def fm_demod(samples: np.ndarray, prev: complex) -> tuple[np.ndarray, complex]:
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32), prev
    ref = np.empty_like(samples)
    ref[0] = prev
    ref[1:] = samples[:-1]
    demod = np.angle(samples * np.conj(ref)).astype(np.float32)
    return demod, samples[-1]


def deemphasis(x: np.ndarray, fs: float, tau_s: float, y_last: float) -> tuple[np.ndarray, float]:
    if x.size == 0:
        return x, y_last
    b, a = _deemphasis_coeffs(float(fs), float(tau_s))
    y, zf = lfilter(b, a, x.astype(np.float32, copy=False), zi=np.array([y_last], dtype=np.float32))
    return y.astype(np.float32, copy=False), float(zf[0])


@lru_cache(maxsize=8)
def _deemphasis_coeffs(fs: float, tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = math.exp(-1.0 / (fs * tau_s))
    b = np.array([1.0 - alpha], dtype=np.float32)
    a = np.array([1.0, -alpha], dtype=np.float32)
    return b, a


@dataclass(frozen=True)
class RuntimeConfig:
    freqs: List[float]
    center_hz: float
    decimation: int
    offsets: List[float]
    squelch_offset: float
    squelch_step: complex
    lpf_taps: np.ndarray
    squelch_hold_samples: int
    squelch_cal_samples: int


def _build_runtime_config(args: argparse.Namespace, plan: BandPlan) -> RuntimeConfig:
    if args.sample_rate % args.audio_rate != 0:
        raise ValueError("sample-rate must be an integer multiple of audio-rate")

    decimation = args.sample_rate // args.audio_rate
    if decimation < 2:
        raise ValueError("sample-rate/audio-rate decimation must be >= 2")

    freqs = channel_frequencies(plan)
    center_hz = (freqs[0] + freqs[-1]) / 2.0
    offsets = [f - center_hz for f in freqs]
    squelch_offset = plan.squelch_monitor_hz - center_hz
    squelch_step = np.complex64(np.exp(-1j * (2.0 * np.pi * squelch_offset / args.sample_rate)))
    max_offset = max(abs(o) for o in offsets)
    max_capture_offset = max(max_offset, abs(squelch_offset))
    if max_capture_offset + (plan.channel_width_hz / 2.0) > (args.sample_rate / 2.0):
        raise ValueError("sample-rate too low to cover all channels and squelch monitor simultaneously; increase --sample-rate")

    lpf_cutoff = min(6000.0, plan.channel_width_hz * 0.48)
    lpf_taps = firwin(numtaps=129, cutoff=lpf_cutoff, fs=args.sample_rate).astype(np.float32)
    squelch_hold_samples = max(0, int(args.audio_rate * (args.squelch_hold_ms / 1000.0)))
    squelch_cal_samples = max(0, int(args.audio_rate * args.squelch_cal_seconds))

    return RuntimeConfig(
        freqs=freqs,
        center_hz=center_hz,
        decimation=decimation,
        offsets=offsets,
        squelch_offset=squelch_offset,
        squelch_step=squelch_step,
        lpf_taps=lpf_taps,
        squelch_hold_samples=squelch_hold_samples,
        squelch_cal_samples=squelch_cal_samples,
    )


def _open_configured_sdr(device_index: int, sample_rate: int, center_hz: float, gain_db: float) -> CompatRtlSdr:
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


def _print_runtime_summary(args: argparse.Namespace, plan: BandPlan, cfg: RuntimeConfig) -> None:
    print(f"Band plan:          {plan.name}")
    print(f"RTL-SDR device:     {args.device_index}")
    print(f"Center frequency:   {cfg.center_hz / 1e6:.6f} MHz")
    print(f"Sample rate:        {args.sample_rate} sps")
    print(f"Audio/WAV rate:     {args.audio_rate} Hz")
    print(f"Decimation:         {cfg.decimation}x")
    if args.no_squelch:
        print("Squelch:            disabled (continuous recording)")
    else:
        print(
            f"Squelch monitor:    {plan.squelch_monitor_hz / 1e6:.6f} MHz "
            f"(per-channel open +{args.squelch_open_db:.1f} dB, close +{args.squelch_close_db:.1f} dB)"
        )
    print("Recording channels:")


def _init_channel_states(args: argparse.Namespace, cfg: RuntimeConfig) -> List[ChannelState]:
    # Each channel keeps independent DSP/squelch state so recording gates are per-channel.
    zi_template = lfilter_zi(cfg.lpf_taps, [1.0]).astype(np.complex64)
    states: List[ChannelState] = []
    for i, off in enumerate(cfg.offsets, start=1):
        out = args.output_dir / f"{i}.wav"
        st = ChannelState(
            number=i,
            offset_hz=off,
            mixer_osc=np.complex64(1.0 + 0.0j),
            mixer_step=np.complex64(np.exp(-1j * (2.0 * np.pi * off / args.sample_rate))),
            lpf_zi=zi_template.copy(),
            prev_sample=0.0 + 0.0j,
            deemp_last=0.0,
            wav=open_wav(out, args.audio_rate),
            squelch_open=False,
            squelch_hold_samples=0,
            squelch_delta_floor_db=None,
            squelch_cal_remaining_samples=cfg.squelch_cal_samples,
        )
        states.append(st)
        print(f"  CH {i:02d}: {cfg.freqs[i - 1] / 1e6:.5f} MHz -> {out}")
    return states


def _compute_monitor_db(
    iq: np.ndarray,
    mixer_osc: complex,
    mixer_step: complex,
    lpf_zi: np.ndarray,
    lpf_taps: np.ndarray,
    decimation: int,
) -> tuple[float, complex, np.ndarray]:
    # The monitor channel is treated as a local noise reference for squelch.
    shifted_sq, next_osc = _mix_with_oscillator(iq, np.complex64(mixer_osc), np.complex64(mixer_step))
    filtered_sq, next_zi = lfilter(lpf_taps, [1.0], shifted_sq, zi=lpf_zi)
    narrow_sq = filtered_sq[::decimation]
    monitor_power = float(np.mean(np.abs(narrow_sq) ** 2)) if narrow_sq.size else 0.0
    monitor_db = 10.0 * math.log10(max(monitor_power, 1e-12))
    return monitor_db, next_osc, next_zi


def _extract_narrowband_channel(
    iq: np.ndarray,
    st: ChannelState,
    lpf_taps: np.ndarray,
    decimation: int,
) -> np.ndarray:
    shifted, st.mixer_osc = _mix_with_oscillator(iq, np.complex64(st.mixer_osc), np.complex64(st.mixer_step))
    filtered, st.lpf_zi = lfilter(lpf_taps, [1.0], shifted, zi=st.lpf_zi)
    return filtered[::decimation]


def _mix_with_oscillator(iq: np.ndarray, mixer_osc: np.complex64, mixer_step: np.complex64) -> tuple[np.ndarray, np.complex64]:
    n = iq.size
    if n == 0:
        return iq, mixer_osc
    mixer = np.empty(n, dtype=np.complex64)
    mixer[0] = mixer_osc
    if n > 1:
        mixer[1:] = mixer_step
        np.cumprod(mixer, out=mixer)
    shifted = iq * mixer
    next_osc = np.complex64(mixer[-1] * mixer_step)
    return shifted, next_osc


def _channel_should_record(
    st: ChannelState,
    narrow: np.ndarray,
    monitor_db: float,
    args: argparse.Namespace,
    squelch_hold_samples: int,
) -> bool:
    chunk_audio_samples = len(narrow)
    ch_power = float(np.mean(np.abs(narrow) ** 2)) if narrow.size else 0.0
    ch_db = 10.0 * math.log10(max(ch_power, 1e-12))
    delta_db = ch_db - monitor_db

    if st.squelch_delta_floor_db is None:
        st.squelch_delta_floor_db = delta_db

    # Initial calibration: learn baseline, keep channel muted.
    if st.squelch_cal_remaining_samples > 0:
        st.squelch_cal_remaining_samples -= chunk_audio_samples
        st.squelch_delta_floor_db = 0.98 * st.squelch_delta_floor_db + 0.02 * delta_db
        st.squelch_open = False
        if narrow.size:
            st.prev_sample = narrow[-1]
        return False

    if not st.squelch_open:
        st.squelch_delta_floor_db = 0.98 * st.squelch_delta_floor_db + 0.02 * delta_db

    open_threshold = st.squelch_delta_floor_db + args.squelch_open_db
    close_threshold = st.squelch_delta_floor_db + args.squelch_close_db

    if not st.squelch_open and delta_db >= open_threshold:
        st.squelch_open = True
        st.squelch_hold_samples = squelch_hold_samples
        print(f"Squelch OPEN  (CH {st.number:02d}, delta {delta_db:.1f} dB)")
    elif st.squelch_open:
        if delta_db < close_threshold:
            st.squelch_hold_samples -= chunk_audio_samples
            if st.squelch_hold_samples <= 0:
                st.squelch_open = False
                print(f"Squelch CLOSE (CH {st.number:02d}, delta {delta_db:.1f} dB)")
        else:
            st.squelch_hold_samples = squelch_hold_samples

    if not st.squelch_open:
        if narrow.size:
            st.prev_sample = narrow[-1]
        return False
    return True


def _write_audio_to_wav(st: ChannelState, narrow: np.ndarray, audio_rate: int, audio_gain: float) -> None:
    demod, st.prev_sample = fm_demod(narrow, st.prev_sample)
    audio, st.deemp_last = deemphasis(demod, audio_rate, 50e-6, st.deemp_last)
    if audio.size:
        audio = audio - float(np.mean(audio))
    audio *= 9000.0 * audio_gain
    pcm = np.clip(audio, -32768, 32767).astype(np.int16)
    st.wav.writeframes(pcm.tobytes())


def _close_resources(states: List[ChannelState], sdr: CompatRtlSdr | None) -> None:
    for st in states:
        try:
            st.wav.close()
        except Exception:
            pass
    if sdr is not None:
        try:
            sdr.close()
        except Exception:
            pass


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
    p.add_argument("--squelch-open-db", type=float, default=4.0, help="Per-channel: open when channel power rises this many dB above monitor channel power (default: 4)")
    p.add_argument("--squelch-close-db", type=float, default=2.0, help="Per-channel: close when channel power falls below this many dB above monitor channel power (default: 2)")
    p.add_argument("--squelch-hold-ms", type=int, default=300, help="Hold squelch open this long after signal drops (default: 300 ms)")
    p.add_argument("--squelch-cal-seconds", type=float, default=2.0, help="Initial squelch calibration duration while forced closed (default: 2.0 s)")
    return p


def run_recorder(args: argparse.Namespace, plan: BandPlan) -> int:
    global running
    running = True

    try:
        cfg = _build_runtime_config(args, plan)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sdr = None
    states: List[ChannelState] = []
    install_signal_handlers()

    try:
        # 1) Open/configure hardware and print session summary.
        sdr = _open_configured_sdr(args.device_index, args.sample_rate, cfg.center_hz, args.gain)
        _print_runtime_summary(args, plan, cfg)
        states = _init_channel_states(args, cfg)

        # 2) Initialize monitor DSP state used as squelch reference.
        zi_template = lfilter_zi(cfg.lpf_taps, [1.0]).astype(np.complex64)
        monitor_mixer_osc = np.complex64(1.0 + 0.0j)
        monitor_lpf_zi = zi_template.copy()

        # 3) Stream IQ chunks, evaluate squelch, and write per-channel WAV audio.
        while running:
            iq = sdr.read_samples(args.chunk_size).astype(np.complex64, copy=False)
            n = iq.size
            if n == 0:
                continue

            monitor_db = 0.0
            if not args.no_squelch:
                monitor_db, monitor_mixer_osc, monitor_lpf_zi = _compute_monitor_db(
                    iq=iq,
                    mixer_osc=monitor_mixer_osc,
                    mixer_step=cfg.squelch_step,
                    lpf_zi=monitor_lpf_zi,
                    lpf_taps=cfg.lpf_taps,
                    decimation=cfg.decimation,
                )

            for st in states:
                narrow = _extract_narrowband_channel(
                    iq=iq,
                    st=st,
                    lpf_taps=cfg.lpf_taps,
                    decimation=cfg.decimation,
                )

                if not args.no_squelch:
                    if not _channel_should_record(
                        st=st,
                        narrow=narrow,
                        monitor_db=monitor_db,
                        args=args,
                        squelch_hold_samples=cfg.squelch_hold_samples,
                    ):
                        continue

                _write_audio_to_wav(st, narrow, args.audio_rate, args.audio_gain)

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"fatal error: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_resources(states, sdr)
        print("Shutdown complete. WAV files closed.")

    return 0


def main_for_plan(plan: BandPlan) -> int:
    configure_library_environment()
    parser = build_parser(plan)
    args = parser.parse_args()
    return run_recorder(args, plan)
