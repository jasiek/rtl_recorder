from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List

import numpy as np


def candidate_rtlsdr_paths() -> List[Path]:
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


def prepend_env_path(var_name: str, new_path: str) -> None:
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
    candidates = candidate_rtlsdr_paths()
    if not candidates:
        return
    lib_path = str(candidates[0])
    os.environ["RTLSDR_LIB_PATH"] = lib_path
    lib_dir = str(Path(lib_path).parent)
    prepend_env_path("DYLD_LIBRARY_PATH", lib_dir)
    prepend_env_path("LD_LIBRARY_PATH", lib_dir)


def load_librtlsdr() -> ctypes.CDLL:
    # Try explicit candidate paths first so local/Homebrew installs work without system linker changes.
    last_error = None
    for lib in candidate_rtlsdr_paths():
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
    libs = candidate_rtlsdr_paths()
    if libs:
        hint = f" Detected candidate: {libs[0]}"
    raise RuntimeError(
        "Could not load librtlsdr. Install librtlsdr and/or set RTLSDR_LIB_PATH."
        + hint
        + (f" Last error: {last_error}" if last_error else "")
    )


class CompatRtlSdr:
    def __init__(self, device_index: int = 0) -> None:
        self.lib = load_librtlsdr()
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
        # Declare ctypes signatures once so per-call marshalling stays predictable and fast.
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
        lib = load_librtlsdr()
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
        # Read interleaved uint8 IQ from librtlsdr and normalize to complex64 [-1, 1].
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
