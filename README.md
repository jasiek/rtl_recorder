# Radio Recorder (RTL-SDR)

Generic USB RTL-SDR recorder with per-channel WAV output and per-channel squelch.

Scripts:
- `pmr446_recorder.py`: PMR446 plan (16 channels)
- `cb_recorder.py`: CB CEPT plan (40 channels)
- `csv_recorder.py`: custom channels from CSV (`label,frequency,width`)

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

CSV custom:
```bash
python csv_recorder.py channels.csv --device 0 --output-dir recordings-csv
```

`channels.csv` format:
```csv
label,frequency,width
Dispatch,446.00625,12.5
Ops 2,446.01875,12.5
```

`csv_recorder.py` will refuse to run if channel frequencies span more than 2 MHz.

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
