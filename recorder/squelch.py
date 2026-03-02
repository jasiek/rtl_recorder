from __future__ import annotations

import argparse

import numpy as np

from recorder.dsp import power_db
from recorder.models import ChannelState


def coarse_should_process_channel(
    st: ChannelState,
    coarse_stream: np.ndarray,
    monitor_coarse_db: float,
    args: argparse.Namespace,
) -> bool:
    # Stage-1 gate: skip expensive fine DSP unless coarse-bin energy rises above learned baseline.
    coarse_delta_db = power_db(coarse_stream) - monitor_coarse_db
    if st.coarse_delta_floor_db is None:
        st.coarse_delta_floor_db = coarse_delta_db

    if st.squelch_open or st.squelch_cal_remaining_samples > 0:
        return True

    st.coarse_delta_floor_db = 0.98 * st.coarse_delta_floor_db + 0.02 * coarse_delta_db
    coarse_open_threshold = st.coarse_delta_floor_db + args.squelch_coarse_open_db
    return coarse_delta_db >= coarse_open_threshold


def channel_should_record(
    st: ChannelState,
    narrow: np.ndarray,
    monitor_db: float,
    args: argparse.Namespace,
    squelch_hold_samples: int,
) -> bool:
    # Stage-2 gate with hysteresis and hold-time on the fully narrowed channel stream.
    chunk_audio_samples = len(narrow)
    ch_db = power_db(narrow)
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
