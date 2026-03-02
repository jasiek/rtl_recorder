#!/usr/bin/env python3
"""Record PMR446 channels from a USB RTL-SDR into per-channel WAV files."""

from __future__ import annotations

from recorder_core import BandPlan, main_for_plan

PMR446_PLAN = BandPlan(
    name="PMR446",
    channel_count=16,
    first_channel_hz=446_006_250.0,
    channel_spacing_hz=12_500.0,
    channel_width_hz=12_500.0,
    squelch_monitor_hz=445_993_750.0,
    default_output_dir="recordings",
    default_sample_rate=256_000,
)


if __name__ == "__main__":
    raise SystemExit(main_for_plan(PMR446_PLAN))
