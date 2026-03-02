#!/usr/bin/env python3
"""Record CB channels from a USB RTL-SDR into per-channel WAV files.

Default band plan: CEPT 40 channels (26.965 MHz to 27.405 MHz, 10 kHz spacing).
"""

from __future__ import annotations

from recorder_core import BandPlan, main_for_plan

CB_PLAN = BandPlan(
    name="CB (CEPT 40)",
    channel_count=40,
    first_channel_hz=26_965_000.0,
    channel_spacing_hz=10_000.0,
    channel_width_hz=10_000.0,
    squelch_monitor_hz=26_955_000.0,
    default_output_dir="recordings-cb",
    default_sample_rate=1_024_000,
)


if __name__ == "__main__":
    raise SystemExit(main_for_plan(CB_PLAN))
