#!/usr/bin/env python3
from __future__ import annotations

"""Backward-compatible exports for the modular recorder package.

This module keeps the original import surface used by entry scripts and ad-hoc
benchmarks while delegating implementation to composable modules under
`recorder/`.
"""

from recorder.dsp import (  # noqa: F401
    PolyphaseFftChannelizer,
    compute_monitor_db as _compute_monitor_db,
    deemphasis,
    deemphasis_coeffs as _deemphasis_coeffs,
    extract_narrowband_channel as _extract_narrowband_channel,
    fm_demod,
    mix_with_oscillator as _mix_with_oscillator,
    multistage_decimate as _multistage_decimate,
    power_db as _power_db,
)
from recorder.models import BandPlan, ChannelState, RuntimeConfig  # noqa: F401
from recorder.pipeline import (  # noqa: F401
    build_parser,
    build_runtime_config as _build_runtime_config,
    channel_frequencies,
    init_channel_states as _init_channel_states,
    install_signal_handlers,
    main_for_plan,
    open_configured_sdr as _open_configured_sdr,
    print_runtime_summary as _print_runtime_summary,
    run_recorder,
)
from recorder.sink import close_resources as _close_resources, open_wav, write_audio_to_wav as _write_audio_to_wav  # noqa: F401,E501
from recorder.source import CompatRtlSdr, configure_library_environment  # noqa: F401
from recorder.squelch import channel_should_record as _channel_should_record, coarse_should_process_channel as _coarse_should_process_channel  # noqa: F401,E501

__all__ = [
    "BandPlan",
    "ChannelState",
    "RuntimeConfig",
    "CompatRtlSdr",
    "PolyphaseFftChannelizer",
    "configure_library_environment",
    "install_signal_handlers",
    "channel_frequencies",
    "open_wav",
    "fm_demod",
    "deemphasis",
    "build_parser",
    "run_recorder",
    "main_for_plan",
    "_build_runtime_config",
    "_open_configured_sdr",
    "_print_runtime_summary",
    "_init_channel_states",
    "_compute_monitor_db",
    "_extract_narrowband_channel",
    "_mix_with_oscillator",
    "_multistage_decimate",
    "_power_db",
    "_coarse_should_process_channel",
    "_channel_should_record",
    "_write_audio_to_wav",
    "_close_resources",
    "_deemphasis_coeffs",
]
