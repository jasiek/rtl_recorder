from __future__ import annotations

from pathlib import Path
from typing import List
import wave

import numpy as np

from recorder.dsp import deemphasis, fm_demod
from recorder.models import ChannelState


def open_wav(path: Path, sample_rate: int) -> wave.Wave_write:
    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    return w


def write_audio_to_wav(st: ChannelState, narrow: np.ndarray, audio_rate: int, audio_gain: float) -> None:
    # FM demod -> deemphasis -> level/clip -> 16-bit PCM.
    demod, st.prev_sample = fm_demod(narrow, st.prev_sample)
    audio, st.deemp_last = deemphasis(demod, audio_rate, 50e-6, st.deemp_last)
    if audio.size:
        audio = audio - float(np.mean(audio))
    audio *= 9000.0 * audio_gain
    pcm = np.clip(audio, -32768, 32767).astype(np.int16)
    st.wav.writeframes(pcm.tobytes())


def close_resources(states: List[ChannelState], sdr) -> None:
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
