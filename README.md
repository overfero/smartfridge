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
├── run_tracking.py           # single-camera entry point
├── run_tracking_parallel.py  # dual-camera parallel entry point (multiprocessing)
├── run_predict.py            # pure inference benchmark (no tracking logic)
├── models/                   # ONNX model files (not tracked)
└── smartfridge/
    ├── cfg/
    │   ├── default.yaml      # main config
    │   ├── models/           # per-model params (conf, imgsz, …)
    │   └── trackers/         # tracker params (hybridsort.yaml)
    ├── core/
    │   ├── inference.py      # ONNX Runtime + OpenVINO EP wrapper (IO binding, rect inference)
    │   ├── config.py         # config loader
    │   ├── profiler.py       # per-stage timing
    │   └── types.py          # SimpleResult, SimpleBoxes
    ├── counter/
    │   ├── multi_line_crossing.py  # taken/returned detection via line history
    │   ├── line_crossing.py        # single-line variant
    │   ├── product.py              # Product state (bbox, trail, counts)
    │   └── geometry.py             # segment intersection math
    ├── renderer/
    │   ├── multi_line_crossing.py  # draws lines + count overlay
    │   ├── line_crossing.py        # single-line variant
    │   └── overlay.py              # frame number, box drawing
    ├── trackers/
    │   ├── hybrid_sort_tracker.py  # HybridSORT entry point
    │   ├── hybrid_sort/            # core tracker (Kalman, association, ReID)
    │   └── utils/                  # IoU, class ReID, spatial ReID
    ├── mediapipe/
    │   └── hand_detector.py        # hand detection + box drawing helper
    ├── pipeline.py           # SmartFridgeYOLO: infer → track → logic loop
    ├── frame_processor.py    # DetectionPredictor: crossing, capture, on_capture callback
    └── outputs.py            # file saver (full image + product crop + CSV)
```
