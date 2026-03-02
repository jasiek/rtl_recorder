from __future__ import annotations

import wave
from dataclasses import dataclass
from typing import List

import numpy as np


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
    coarse_bin_index: int
    residual_osc: complex
    residual_step: complex
    post1_zi: np.ndarray
    post2_zi: np.ndarray
    prev_sample: complex
    deemp_last: float
    wav: wave.Wave_write
    squelch_open: bool
    squelch_hold_samples: int
    squelch_delta_floor_db: float | None
    coarse_delta_floor_db: float | None
    squelch_cal_remaining_samples: int


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
    post1_lpf_taps: np.ndarray
    post2_lpf_taps: np.ndarray
    squelch_hold_samples: int
    squelch_cal_samples: int
