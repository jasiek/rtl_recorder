# Radio Recorder (RTL-SDR)

Generic USB RTL-SDR recorder with per-channel WAV output and per-channel squelch.

Scripts:
- `pmr446_recorder.py`: PMR446 plan (16 channels)
- `cb_recorder.py`: CB CEPT plan (40 channels)

## Install
```bash
python -m pip install -r requirements.txt
```

## Run
PMR446:
```bash
python pmr446_recorder.py --device 0 --output-dir recordings
```

CB:
```bash
python cb_recorder.py --device 0 --output-dir recordings-cb
```

## Features
- Simultaneous channel recording into `1.wav ... N.wav`
- Per-channel squelch (channels open/close independently)
- Reference monitor channel below the band for squelch baseline
- Clean shutdown (`Ctrl+C` / `SIGTERM`) closes all WAV files

## Common options
- `--gain 30`
- `--sample-rate ...`
- `--audio-rate 16000`
- `--audio-gain 3.0`
- `--chunk-size 65536`
- `--no-squelch`
- `--squelch-open-db 4`
- `--squelch-close-db 2`
- `--squelch-hold-ms 300`
- `--squelch-cal-seconds 2.0`

Use `--help` on either script for all options.

## macOS/Homebrew `librtlsdr`
If needed:
```bash
export RTLSDR_LIB_PATH=/opt/homebrew/lib/librtlsdr.dylib
```
