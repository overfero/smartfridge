# SmartFridge

Object tracking pipeline for smart fridge — detects and counts items taken/returned using YOLO11 (ONNX) + HybridSORT + virtual line crossing.

## Requirements

- Python 3.11+
- OpenVINO runtime (for hardware-accelerated inference)

## Setup

```bash
uv sync
```

Or with conda (existing env):

```bash
conda activate glair
```

## Configuration

All parameters are in `smartfridge/cfg/default.yaml`.  
Model-specific params: `smartfridge/cfg/models/<name>.yaml`  
Tracker params: `smartfridge/cfg/trackers/<name>.yaml`

Key settings:

| Key | Description |
|-----|-------------|
| `model.name` | Model to use (`nano_v2`, `small_v1`, etc.) |
| `model.conf` | Confidence threshold |
| `video.source` | Path to input video |
| `video.save` | Save tracked output video |
| `outputs.enabled` | Save cropped images + CSV |

## Run

```bash
python run_tracking.py
```

Profiling output is printed after each run.

## Project Structure

```
smartfridge/
├── cfg/
│   ├── default.yaml          # main config
│   ├── models/               # per-model params
│   └── trackers/             # tracker params
├── counter/                  # line crossing logic
├── renderer/                 # drawing utilities
├── trackers/                 # HybridSORT
├── inference.py              # ONNX Runtime wrapper
├── yolo.py                   # main pipeline
├── predict.py                # post-tracking logic
├── profiler.py               # pipeline profiler
└── outputs.py                # CSV + image saver
models/                       # ONNX model files (not tracked)
```
