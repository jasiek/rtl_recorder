#!/usr/bin/env python3
"""Record all PMR446 channels (1-16) from a USB RTL-SDR into per-channel WAV files.

Requirements:
  pip install numpy scipy
"""

from __future__ import annotations

import argparse
import ctypes
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

# PMR446 channel plan
CHANNEL_COUNT = 16
FIRST_CHANNEL_HZ = 446_006_250.0
CHANNEL_SPACING_HZ = 12_500.0
CHANNEL_WIDTH_HZ = 12_500.0


@dataclass
class ChannelState:
    number: int
    offset_hz: float
    mixer_phase: float
    lpf_zi: np.ndarray
    prev_sample: complex
    deemp_last: float
    wav: wave.Wave_write


running = True


def _candidate_rtlsdr_paths() -> List[Path]:
    # Prioritize explicit user override.
    candidates: List[Path] = []
    env = os.environ.get("RTLSDR_LIB_PATH")
    if env:
        candidates.append(Path(env))

    # Typical locations on macOS Homebrew (Apple Silicon + Intel) and Linux.
    candidates.extend(
        [
            Path("/opt/homebrew/lib/librtlsdr.dylib"),
            Path("/opt/homebrew/lib/librtlsdr.0.dylib"),
            Path("/usr/local/lib/librtlsdr.dylib"),
            Path("/usr/local/lib/librtlsdr.0.dylib"),
            Path("/usr/lib/librtlsdr.so"),
            Path("/usr/lib/x86_64-linux-gnu/librtlsdr.so"),
        ]
    )
    return [p for p in candidates if p.exists()]


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
    """Minimal RTL-SDR wrapper via ctypes to avoid Python binding ABI mismatches."""

    def __init__(self, device_index: int = 0) -> None:
        self.lib = _load_librtlsdr()
        self._setup_api()
        self.dev = ctypes.c_void_p()
        rc = self.lib.rtlsdr_open(ctypes.byref(self.dev), ctypes.c_uint32(device_index))
        if rc != 0:
            raise RuntimeError(f"rtlsdr_open failed with code {rc}")
        self._closed = False
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

        self.lib.rtlsdr_read_sync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
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
        # librtlsdr gain units are tenths of dB.
        rc = self.lib.rtlsdr_set_tuner_gain_mode(self.dev, 1)
        if rc != 0:
            raise RuntimeError(f"rtlsdr_set_tuner_gain_mode failed with code {rc}")
        rc = self.lib.rtlsdr_set_tuner_gain(self.dev, int(round(value * 10.0)))
        if rc != 0:
            raise RuntimeError(f"rtlsdr_set_tuner_gain failed with code {rc}")

    def read_samples(self, count: int) -> np.ndarray:
        byte_count = int(count) * 2
        buf = (ctypes.c_ubyte * byte_count)()
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


def channel_frequencies() -> List[float]:
    return [FIRST_CHANNEL_HZ + i * CHANNEL_SPACING_HZ for i in range(CHANNEL_COUNT)]


def open_wav(path: Path, sample_rate: int) -> wave.Wave_write:
    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)  # int16
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

    alpha = math.exp(-1.0 / (fs * tau_s))
    y = np.empty_like(x, dtype=np.float32)
    prev = y_last
    one_minus = 1.0 - alpha
    for i in range(x.size):
        prev = alpha * prev + one_minus * float(x[i])
        y[i] = prev
    return y, prev


def main() -> int:
    parser = argparse.ArgumentParser(description="Record PMR446 channels 1-16 into per-channel WAV files")
    parser.add_argument(
        "--device-index",
        "--device",
        dest="device_index",
        type=int,
        default=0,
        help="RTL-SDR device index (default: 0)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"), help="Output directory")
    parser.add_argument("--gain", type=float, default=30.0, help="Tuner gain in dB (default: 30)")
    parser.add_argument("--sample-rate", type=int, default=256_000, help="SDR sample rate (default: 256000)")
    parser.add_argument("--audio-rate", type=int, default=16_000, help="Per-channel WAV sample rate (default: 16000)")
    parser.add_argument("--chunk-size", type=int, default=65_536, help="IQ samples per read (default: 65536)")
    args = parser.parse_args()

    if args.sample_rate % args.audio_rate != 0:
        print("sample-rate must be an integer multiple of audio-rate", file=sys.stderr)
        return 2

    decimation = args.sample_rate // args.audio_rate
    if decimation < 2:
        print("sample-rate/audio-rate decimation must be >= 2", file=sys.stderr)
        return 2

    # Fit all PMR channels inside one RTL-SDR capture by centering on the PMR block midpoint.
    freqs = channel_frequencies()
    center_hz = (freqs[0] + freqs[-1]) / 2.0
    offsets = [f - center_hz for f in freqs]
    max_offset = max(abs(o) for o in offsets)
    if max_offset + (CHANNEL_WIDTH_HZ / 2.0) > (args.sample_rate / 2.0):
        print(
            "sample-rate too low to cover all 16 channels simultaneously; "
            "increase --sample-rate",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Channel extraction low-pass before decimation (12.5 kHz channel; keep ~5 kHz useful audio/deviation).
    lpf_taps = firwin(numtaps=129, cutoff=6_000, fs=args.sample_rate).astype(np.float32)

    sdr = None
    states: List[ChannelState] = []
    install_signal_handlers()

    try:
        detected = CompatRtlSdr.list_devices()
        print(f"Detected RTL-SDR devices: {len(detected)}")
        for i, name in enumerate(detected):
            print(f"  [{i}] {name}")
        if not detected:
            raise RuntimeError(
                "No RTL-SDR devices detected. Check USB connection, cable/hub power, and that no other SDR app is using the dongle."
            )
        if args.device_index < 0 or args.device_index >= len(detected):
            raise RuntimeError(
                f"Invalid --device-index {args.device_index}; detected indices are 0..{len(detected)-1}"
            )

        sdr = CompatRtlSdr(args.device_index)
        sdr.sample_rate = float(args.sample_rate)
        sdr.center_freq = float(center_hz)
        sdr.gain = float(args.gain)

        print(f"RTL-SDR device index: {args.device_index}")
        print(f"Center frequency:   {center_hz / 1e6:.6f} MHz")
        print(f"Sample rate:        {args.sample_rate} sps")
        print(f"Audio/WAV rate:     {args.audio_rate} Hz")
        print(f"Decimation:         {decimation}x")
        print("Recording channels:")

        zi_template = lfilter_zi(lpf_taps, [1.0]).astype(np.complex64)
        for i, off in enumerate(offsets, start=1):
            wav = open_wav(args.output_dir / f"{i}.wav", args.audio_rate)
            st = ChannelState(
                number=i,
                offset_hz=off,
                mixer_phase=0.0,
                lpf_zi=zi_template.copy(),
                prev_sample=0.0 + 0.0j,
                deemp_last=0.0,
                wav=wav,
            )
            states.append(st)
            print(f"  CH {i:02d}: {freqs[i - 1] / 1e6:.5f} MHz -> {(args.output_dir / f'{i}.wav')}")

        while running:
            iq = sdr.read_samples(args.chunk_size).astype(np.complex64, copy=False)
            n = iq.size
            if n == 0:
                continue

            idx = np.arange(n, dtype=np.float32)
            for st in states:
                w = 2.0 * np.pi * st.offset_hz / args.sample_rate
                phases = st.mixer_phase + w * idx
                mixer = np.exp(-1j * phases).astype(np.complex64)
                shifted = iq * mixer
                st.mixer_phase = float((phases[-1] + w) % (2.0 * np.pi))

                filtered, st.lpf_zi = lfilter(lpf_taps, [1.0], shifted, zi=st.lpf_zi)
                narrow = filtered[::decimation]

                demod, st.prev_sample = fm_demod(narrow, st.prev_sample)
                audio, st.deemp_last = deemphasis(demod, args.audio_rate, 50e-6, st.deemp_last)

                audio *= 9000.0
                pcm = np.clip(audio, -32768, 32767).astype(np.int16)
                st.wav.writeframes(pcm.tobytes())

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"fatal error: {exc}", file=sys.stderr)
        return 1
    finally:
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
        print("Shutdown complete. WAV files closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
