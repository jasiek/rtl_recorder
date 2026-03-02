from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi

from recorder.models import ChannelState


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
    # Keep deemphasis state chunk-to-chunk by seeding lfilter with the previous output sample.
    b, a = deemphasis_coeffs(float(fs), float(tau_s))
    y, zf = lfilter(b, a, x.astype(np.float32, copy=False), zi=np.array([y_last], dtype=np.float32))
    return y.astype(np.float32, copy=False), float(zf[0])


@lru_cache(maxsize=8)
def deemphasis_coeffs(fs: float, tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    # 1-pole IIR matching analog FM deemphasis: y[n] = (1-a)x[n] + a*y[n-1].
    alpha = math.exp(-1.0 / (fs * tau_s))
    b = np.array([1.0 - alpha], dtype=np.float32)
    a = np.array([1.0, -alpha], dtype=np.float32)
    return b, a


class PolyphaseFftChannelizer:
    def __init__(self, sample_rate: int, decimation: int, taps_per_phase: int = 16) -> None:
        # Build a decimation-way analysis filter bank: polyphase FIR branches + FFT recombination.
        self.decimation = decimation
        self.sample_rate = sample_rate
        prototype_len = decimation * taps_per_phase
        cutoff = (sample_rate / (2.0 * decimation)) * 0.90
        prototype = firwin(numtaps=prototype_len, cutoff=cutoff, fs=sample_rate).astype(np.float32)
        self.phase_taps = [prototype[p::decimation].astype(np.float32, copy=False) for p in range(decimation)]
        self.phase_zis = [lfilter_zi(t, [1.0]).astype(np.complex64) for t in self.phase_taps]

    def process(self, iq: np.ndarray) -> np.ndarray:
        # Output shape: [coarse_bin, time] at sample_rate/decimation.
        n_out = iq.size // self.decimation
        if n_out == 0:
            return np.zeros((self.decimation, 0), dtype=np.complex64)
        used = iq[: n_out * self.decimation]
        branches = np.empty((self.decimation, n_out), dtype=np.complex64)
        for phase in range(self.decimation):
            phase_input = used[phase:]
            y, self.phase_zis[phase] = lfilter(self.phase_taps[phase], [1.0], phase_input, zi=self.phase_zis[phase])
            branches[phase, :] = y[: n_out * self.decimation : self.decimation]
        return np.fft.fft(branches, axis=0).astype(np.complex64, copy=False)


def extract_narrowband_channel(
    coarse_bins: np.ndarray,
    st: ChannelState,
    post1_lpf_taps: np.ndarray,
    post2_lpf_taps: np.ndarray,
    decimation_stage1: int,
    decimation_stage2: int,
) -> np.ndarray:
    # Fine-tune selected coarse bin to exact channel center, then run staged LPF+decimation to audio rate.
    coarse = coarse_bins[st.coarse_bin_index]
    shifted, st.residual_osc = mix_with_oscillator(coarse, np.complex64(st.residual_osc), np.complex64(st.residual_step))
    return multistage_decimate(
        shifted=shifted,
        post1_lpf_taps=post1_lpf_taps,
        post2_lpf_taps=post2_lpf_taps,
        post1_zi=st.post1_zi,
        post2_zi=st.post2_zi,
        decimation_stage1=decimation_stage1,
        decimation_stage2=decimation_stage2,
    )


def compute_monitor_db(
    coarse_bins: np.ndarray,
    monitor_bin_index: int,
    mixer_osc: complex,
    mixer_step: complex,
    post1_zi: np.ndarray,
    post2_zi: np.ndarray,
    post1_lpf_taps: np.ndarray,
    post2_lpf_taps: np.ndarray,
    decimation_stage1: int,
    decimation_stage2: int,
) -> tuple[float, complex, np.ndarray, np.ndarray]:
    # The monitor channel is treated as a local noise reference for squelch.
    monitor_stream = coarse_bins[monitor_bin_index]
    shifted_sq, next_osc = mix_with_oscillator(monitor_stream, np.complex64(mixer_osc), np.complex64(mixer_step))
    narrow_sq = multistage_decimate(
        shifted=shifted_sq,
        post1_lpf_taps=post1_lpf_taps,
        post2_lpf_taps=post2_lpf_taps,
        post1_zi=post1_zi,
        post2_zi=post2_zi,
        decimation_stage1=decimation_stage1,
        decimation_stage2=decimation_stage2,
    )
    monitor_db = power_db(narrow_sq)
    return monitor_db, next_osc, post1_zi, post2_zi


def multistage_decimate(
    shifted: np.ndarray,
    post1_lpf_taps: np.ndarray,
    post2_lpf_taps: np.ndarray,
    post1_zi: np.ndarray,
    post2_zi: np.ndarray,
    decimation_stage1: int,
    decimation_stage2: int,
) -> np.ndarray:
    # Decimate in two cheaper stages to reduce FIR cost at higher rates.
    stage1, post1_zi[:] = lfilter(post1_lpf_taps, [1.0], shifted, zi=post1_zi)
    out = stage1[::decimation_stage1]
    if decimation_stage2 > 1:
        stage2, post2_zi[:] = lfilter(post2_lpf_taps, [1.0], out, zi=post2_zi)
        return stage2[::decimation_stage2]
    return out


def mix_with_oscillator(iq: np.ndarray, mixer_osc: np.complex64, mixer_step: np.complex64) -> tuple[np.ndarray, np.complex64]:
    # Generate chunk-local NCO by recurrence (cumprod) to avoid per-sample trig calls.
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


def power_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    power = float(np.vdot(samples, samples).real / samples.size)
    return 10.0 * math.log10(max(power, 1e-12))
