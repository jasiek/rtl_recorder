#!/usr/bin/env python3
"""Record all PMR446 channels (1-16) from a USB RTL-SDR into per-channel WAV files.

Requirements:
  pip install pyrtlsdr numpy scipy
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from rtlsdr import RtlSdr
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
    parser.add_argument("--device-index", type=int, default=0, help="RTL-SDR device index (default: 0)")
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
        sdr = RtlSdr(args.device_index)
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
