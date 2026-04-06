"""
SmartFridgeYOLO — wrapper utama pipeline inferensi + tracking.

Tidak bergantung pada ultralytics sama sekali. Menggunakan:
  - SmartFridgeModel  (onnxruntime + OpenVINO EP)
  - HybridSORT        (tracker standalone)
  - DetectionPredictor (business logic: line crossing, captures, CSV)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from smartfridge.core.config import load_config
from smartfridge.core.inference import SmartFridgeModel
from smartfridge.core.profiler import Profiler, measure_or_null
from smartfridge.core.types import SimpleBoxes, SimpleResult
from smartfridge.frame_processor import DetectionPredictor
from smartfridge.trackers.hybrid_sort_tracker import HybridSORT


def _build_result(
    tracks: np.ndarray,
    frame: np.ndarray,
    names: dict,
) -> SimpleResult:
    if len(tracks) == 0:
        data = np.empty((0, 7), dtype=np.float32)
    else:
        # tracks (N, 8): [x1, y1, x2, y2, track_id, score, cls, idx]
        # SimpleBoxes  (N, 7): [x1, y1, x2, y2, track_id, conf, cls]
        data = tracks[:, :7].astype(np.float32)
    return SimpleResult(orig_img=frame, boxes=SimpleBoxes(data), names=names)


def _tracker_args(cfg: SimpleNamespace) -> SimpleNamespace:
    """Ekstrak tracker config sebagai SimpleNamespace flat (kompatibel HybridSORT)."""
    d = vars(cfg.tracker).copy()
    d.pop("name", None)
    return SimpleNamespace(**d)


class SmartFridgeYOLO:
    """Entry-point utama SmartFridge: load model + config, jalankan tracking."""

    def __init__(
        self,
        model_path: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self.cfg = load_config(config_path)

        # Override model path jika diberikan eksplisit
        if model_path:
            self.cfg.model.path = model_path

        self.model = SmartFridgeModel(self.cfg.model.path)

    def track(
        self,
        source: str | None = None,
        save: bool | None = None,
        verbose: bool = False,
        profiling: bool = False,
        on_capture=None,
        **_kwargs,  # terima kwargs lama (half, device, persist, …) tanpa error
    ):
        """Generator: yield SimpleResult satu per frame.

        Args:
            source     : Path ke video. Default: cfg.video.source.
            save       : Override cfg.video.save jika diberikan.
            verbose    : Tidak dipakai (kompatibilitas).
            profiling  : Jika True, cetak ringkasan waktu per-stage setelah selesai.
            on_capture : Callback dipanggil setiap crossing event.
                         Signature: fn(cross_frame, best_frame, orig_img, track_id,
                                       class_name, bbox, line_name, direction)
                         Jika di-set, menggantikan output_saver.
        """
        video_source = source or self.cfg.video.source
        do_save      = save if save is not None else self.cfg.video.save

        tracker_args = _tracker_args(self.cfg)
        tracker      = HybridSORT(tracker_args, frame_rate=self.cfg.video.default_fps)
        tracker.names = self.model.names

        predictor = DetectionPredictor(
            tracker_args=tracker_args,
            names=self.model.names,
            tracker=tracker,
            cfg=self.cfg,
        )
        if on_capture is not None:
            predictor.on_capture = on_capture

        cap = cv2.VideoCapture(video_source)
        fps = cap.get(cv2.CAP_PROP_FPS) or float(self.cfg.video.default_fps)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer: cv2.VideoWriter | None = None
        if do_save:
            out_path = str(Path(video_source).stem) + "_tracked.mp4"
            writer   = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        # Render hanya kalau dibutuhkan untuk video output
        needs_render = do_save

        prof    = Profiler() if profiling else None
        _m      = lambda s: measure_or_null(prof, s)  # noqa: E731

        frame_id = 0
        try:
            while True:
                with _m("decode"):
                    ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                dets = self.model.infer(frame, conf_thresh=self.cfg.model.conf, profiler=prof)

                with _m("track"):
                    tracks = tracker.update(dets, frame)

                result = _build_result(tracks, frame, self.model.names)

                with _m("logic"):
                    predictor._on_tracking_complete(result, frame_id)

                if needs_render:
                    with _m("render"):
                        rendered = predictor._render_frame(result, frame_id)
                    with _m("write"):
                        if writer is not None:
                            writer.write(rendered)

                yield result
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if prof is not None:
                print(prof.report(frame_id))
