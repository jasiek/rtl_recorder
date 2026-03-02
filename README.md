# PMR446 USB Recorder

Python implementation for recording PMR446 channels 1-16 from a single USB RTL-SDR.

## Why Python
Python is the most practical choice for this first version because:
- USB RTL-SDR access is straightforward (`pyrtlsdr`)
- DSP/channelization is fast to iterate (`numpy`, `scipy`)
- Easy to adapt while validating RF settings on real hardware

## Features implemented
- Listens to all 16 PMR446 channels (12.5 kHz spacing)
- Records all channels simultaneously into separate WAV files (`1.wav` ... `16.wav`)
- Closes WAV files and SDR device cleanly on termination (`Ctrl+C` / `SIGTERM`)
- Channel 1 starts at 446.00625 MHz
- Narrowband FM channelization (12.5 kHz channels)

## Install
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python pmr446_recorder.py --device-index 0 --output-dir recordings
```

Optional tuning:
- `--gain 30`
- `--sample-rate 256000`
- `--audio-rate 16000`
- `--chunk-size 65536`

## Notes
- An RTL-SDR with stable TCXO and decent front-end filtering is recommended.
- If audio is distorted, lower `--gain`.
- If you use a different SDR, this can be adapted to SoapySDR in a follow-up.
