"""
run_tracking_parallel.py — parallel inference 2 kamera (atas + bawah).

Untuk produksi: ganti SOURCE_TOP dan SOURCE_BOTTOM dengan jalur kamera/video
masing-masing. Saat ini keduanya memakai video yang sama (simulasi).

Mengapa multiprocessing dan bukan threading:
  Threading         → 2 OpenVINO session bersaing untuk semua core → throttle
                      infer ~60ms/frame, ~15 FPS
  Multiprocessing   → tiap process mendapat core dedicated (CPU affinity)
                      infer ~49ms/frame, ~17 FPS — setara `taskset -c X-Y`

Tiap worker process:
  - Pin ke setengah jumlah core CPU (os.sched_setaffinity)
  - Inisialisasi model + tracker + predictor sendiri (tidak ada shared state)
  - Jalan independen — tidak perlu sinkronisasi antar kamera
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np

from smartfridge.core.config import load_config
from smartfridge.core.inference import SmartFridgeModel
from smartfridge.core.profiler import Profiler
from smartfridge.core.types import SimpleBoxes, SimpleResult
from smartfridge.frame_processor import DetectionPredictor
from smartfridge.trackers.hybrid_sort_tracker import HybridSORT

# ── Sumber video ───────────────────────────────────────────────────────────────
SOURCE_TOP    = "smart_fridge_atas.mp4"
SOURCE_BOTTOM = "smart_fridge_bawah.mp4"   # produksi: ganti ke kamera bawah

# ── CPU affinity per process ───────────────────────────────────────────────────
# Ubah sesuai kebutuhan. Pastikan kedua set tidak overlap.
# Contoh 8-core: CORES_TOP=[0,1,2,3]  CORES_BOTTOM=[4,5,6,7]
CORES_TOP    = [0, 1]
CORES_BOTTOM = [2, 3]


# ── Helper ─────────────────────────────────────────────────────────────────────

def _build_result(tracks: np.ndarray, frame: np.ndarray, names: dict) -> SimpleResult:
    data = tracks[:, :7].astype(np.float32) if len(tracks) else np.empty((0, 7), dtype=np.float32)
    return SimpleResult(orig_img=frame, boxes=SimpleBoxes(data), names=names)


def _tracker_args(cfg: SimpleNamespace) -> SimpleNamespace:
    d = vars(cfg.tracker).copy()
    d.pop("name", None)
    return SimpleNamespace(**d)


# ── Pipeline per kamera ────────────────────────────────────────────────────────

class CameraPipeline:
    """Satu kamera: model ONNX + tracker + predictor + capture."""

    def __init__(self, name: str, source: str, cfg: SimpleNamespace, cores: list[int] | None = None) -> None:
        self.name = name
        self.model = SmartFridgeModel(cfg.model.path, num_threads=len(cores) if cores else None)

        tracker_args = _tracker_args(cfg)
        self.tracker = HybridSORT(tracker_args, frame_rate=cfg.video.default_fps)
        self.tracker.names = self.model.names

        self.predictor = DetectionPredictor(
            tracker_args=tracker_args,
            names=self.model.names,
            tracker=self.tracker,
            cfg=cfg,
        )

        self.cap   = cv2.VideoCapture(source)
        self._conf = cfg.model.conf
        self.prof  = Profiler()

    def step(self, frame_id: int) -> SimpleResult | None:
        with self.prof.measure("decode"):
            ret, frame = self.cap.read()

        if not ret:
            return None

        dets = self.model.infer(frame, conf_thresh=self._conf, profiler=self.prof)

        with self.prof.measure("track"):
            tracks = self.tracker.update(dets, frame)

        result = _build_result(tracks, frame, self.model.names)

        with self.prof.measure("logic"):
            self.predictor._on_tracking_complete(result, frame_id)

        return result

    def release(self) -> None:
        self.cap.release()


# ── Worker process ─────────────────────────────────────────────────────────────

def _camera_worker(
    name: str,
    source: str,
    cores: list[int],
    result_queue: mp.Queue,
) -> None:
    """Dijalankan di process terpisah. Pin ke `cores`, jalankan pipeline penuh."""
    # Pin process ke core yang dialokasikan — cegah kontestasi dengan process lain
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cores))

    cfg = load_config()
    cam = CameraPipeline(name, source, cfg, cores=cores)

    frame_id = 0
    t_start  = time.perf_counter()
    try:
        while True:
            frame_id += 1
            if cam.step(frame_id) is None:
                frame_id -= 1
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()

    t_elapsed = time.perf_counter() - t_start
    result_queue.put((name, cam.prof, frame_id, t_elapsed))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    cores_top    = CORES_TOP
    cores_bottom = CORES_BOTTOM

    print(f"[parallel] top={cores_top}  bottom={cores_bottom}")

    result_queue: mp.Queue = mp.Queue()

    p_top = mp.Process(
        target=_camera_worker,
        args=("top", SOURCE_TOP, cores_top, result_queue),
    )
    p_bottom = mp.Process(
        target=_camera_worker,
        args=("bottom", SOURCE_BOTTOM, cores_bottom, result_queue),
    )

    t_wall = time.perf_counter()
    p_top.start()
    p_bottom.start()

    try:
        p_top.join()
        p_bottom.join()
    except KeyboardInterrupt:
        p_top.terminate()
        p_bottom.terminate()
        p_top.join()
        p_bottom.join()

    t_wall = time.perf_counter() - t_wall
    print(f"\n[parallel] wall-clock total: {t_wall:.2f}s\n")

    # Cetak profiling dari tiap process
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    # Urutkan: top dulu, bottom kedua
    results.sort(key=lambda x: x[0])
    for name, prof, n_frames, t_elapsed in results:
        fps = n_frames / t_elapsed if t_elapsed > 0 else 0
        print(f"[{name}] {n_frames} frames  {t_elapsed:.2f}s  {fps:.1f} FPS")
        label = f"{name.upper():<6} PROFILING"
        print(prof.report(n_frames).replace("PIPELINE PROFILING", label))
        print()


if __name__ == "__main__":
    mp.set_start_method("fork")   # Linux default, eksplisit untuk kejelasan
    main()
