#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_lib() {
  local candidates=(
    "${RTLSDR_LIB_PATH:-}"
    "/opt/homebrew/lib/librtlsdr.2.0.1.dylib"
    "/opt/homebrew/lib/librtlsdr.dylib"
    "/opt/homebrew/lib/librtlsdr.0.dylib"
    "/usr/local/lib/librtlsdr.dylib"
    "/usr/local/lib/librtlsdr.0.dylib"
    "/usr/lib/librtlsdr.so"
    "/usr/lib/x86_64-linux-gnu/librtlsdr.so"
  )

  local c
  for c in "${candidates[@]}"; do
    if [[ -n "$c" && -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

if LIBRTLSDR_PATH="$(find_lib)"; then
  export RTLSDR_LIB_PATH="$LIBRTLSDR_PATH"
  LIB_DIR="$(dirname "$LIBRTLSDR_PATH")"

  if [[ -n "${DYLD_LIBRARY_PATH:-}" ]]; then
    export DYLD_LIBRARY_PATH="$LIB_DIR:$DYLD_LIBRARY_PATH"
  else
    export DYLD_LIBRARY_PATH="$LIB_DIR"
  fi

  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"
  else
    export LD_LIBRARY_PATH="$LIB_DIR"
  fi
fi

exec python "$SCRIPT_DIR/pmr446_recorder.py" "$@"
