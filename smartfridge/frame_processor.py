from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from smartfridge.counter import MultiLineCrossingCounter
from smartfridge.counter.product import Product
from smartfridge.mediapipe.hand_detector import draw_boxes
from smartfridge.outputs import OutputSaver
from smartfridge.renderer import MultiLineCrossingRenderer
from smartfridge.renderer.overlay import draw_frame_number
from smartfridge.core.types import SimpleResult


class DetectionPredictor:
    """Prediction & tracking logic untuk SmartFridge — standalone, tanpa ultralytics."""

    def __init__(
        self,
        tracker_args: SimpleNamespace,
        names: dict[int, str],
        tracker,
        cfg: SimpleNamespace,
    ) -> None:
        self._tracker_args = tracker_args
        self.names = names
        self.tracker = tracker

        lines  = cfg.lines.crossings
        colors = cfg.lines.colors

        self.counter = MultiLineCrossingCounter(
            lines=lines,
            debug=cfg.counter.debug,
            debug_file=cfg.counter.debug_file,
        )
        self.counter.on_first_cross = self._on_first_cross

        self.renderer = MultiLineCrossingRenderer(
            lines=lines,
            colors=colors,
            line_thickness=cfg.lines.thickness,
            ui=cfg.ui,
        )

        self.output_saver: OutputSaver | None = (
            OutputSaver(output_dir=Path(cfg.outputs.dir))
            if cfg.outputs.enabled
            else None
        )

        self._capture_buffer_size: int = cfg.capture.buffer_size
        self.capture_mode: str = cfg.capture.mode

        self.tracked_products: dict[int, Product] = {}
        self.current_taken_result: list[dict] = []

        self._track_frame_info: dict[int, deque] = {}
        self._rendered_window: deque = deque(maxlen=self._capture_buffer_size)
        self._captures: dict[int, dict[int, tuple]] = {}
        self._last_seen_frame: dict[int, int] = {}
        self._current_frame: int = 0

    # ── Tracking logic ────────────────────────────────────────────────────────

    def _on_tracking_complete(self, result: SimpleResult, frame_id: int) -> None:
        if not (hasattr(result, "boxes") and result.boxes.id is not None):
            return

        bbox_xyxy = result.boxes.xyxy
        track_ids = result.boxes.id.astype(int)
        class_ids = result.boxes.cls.astype(int)
        confs     = result.boxes.conf

        low_thresh = float(getattr(self._tracker_args, "low_thresh", 0.2))
        max_age    = int(getattr(self._tracker_args, "max_age", 30))

        valid_mask = confs >= low_thresh
        valid_ids  = track_ids[valid_mask]

        for bbox, track_id, class_id in zip(bbox_xyxy, track_ids, class_ids):
            self._upsert_product(track_id, bbox, class_id, frame_id)

        self._current_frame = frame_id

        for i, track_id in enumerate(track_ids):
            if not valid_mask[i]:
                continue
            buf = self._track_frame_info.setdefault(
                track_id, deque(maxlen=self._capture_buffer_size)
            )
            buf.append((frame_id, bbox_xyxy[i].tolist(), float(confs[i])))

        self.counter.update(self.tracked_products, valid_ids, frame_id)

        for track_id in valid_ids:
            self._last_seen_frame[track_id] = frame_id

        current_ids = set(track_ids)
        for tid in list(self._last_seen_frame):
            if tid not in current_ids and (frame_id - self._last_seen_frame[tid]) > max_age:
                self._on_track_lost(tid)
                self._last_seen_frame.pop(tid, None)

        self.current_taken_result = [
            {"product": p.class_name, "trail_len": len(p.trail_points)}
            for p in self.tracked_products.values()
            if p.taken_counted and not p.is_complete
        ]

    # ── Capture callbacks ─────────────────────────────────────────────────────

    def _on_first_cross(self, track_id: int, line_idx: int, frame: int) -> None:
        buf = self._track_frame_info.get(track_id)
        if not buf:
            return

        if self.capture_mode == "size":
            best = max(buf, key=lambda e: (e[1][2] - e[1][0]) * (e[1][3] - e[1][1]))
        else:
            best = max(buf, key=lambda e: e[2])

        best_frame_id, bbox, _ = best

        rendered = None
        for fid, img in self._rendered_window:
            if fid == best_frame_id:
                rendered = img
                break
        if rendered is None and self._rendered_window:
            _, rendered = self._rendered_window[-1]

        self._captures.setdefault(track_id, {})[line_idx] = (
            frame, best_frame_id, rendered, bbox
        )

    def _on_track_lost(self, track_id: int) -> None:
        product = self.tracked_products.get(track_id)
        event   = self.counter.finalize(track_id)

        if event and self.output_saver:
            class_name = product.class_name if product else "unknown"
            direction  = event["direction"]

            if direction == "taken":
                self.counter.taken_counts[class_name] = (
                    self.counter.taken_counts.get(class_name, 0) + 1
                )
                if product:
                    product.mark_taken()
            else:
                self.counter.returned_counts[class_name] = (
                    self.counter.returned_counts.get(class_name, 0) + 1
                )
                if product:
                    product.mark_returned()

            captures = self._captures.pop(track_id, {})
            for line_idx, (cross_frame, best_frame, rendered, bbox) in sorted(captures.items()):
                if rendered is not None:
                    self.output_saver.save(
                        cross_frame=cross_frame,
                        best_frame=best_frame,
                        rendered_frame=rendered,
                        track_id=track_id,
                        class_name=class_name,
                        bbox=bbox,
                        line_idx=line_idx,
                        direction=direction,
                    )
        else:
            self._captures.pop(track_id, None)

        self._track_frame_info.pop(track_id, None)
        self.tracked_products.pop(track_id, None)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _render_frame(self, result: SimpleResult, frame_id: int) -> np.ndarray:
        im0 = result.orig_img.copy()
        track_ids = result.boxes.id.astype(int) if result.boxes.id is not None else []
        self.renderer.draw(im0, frame_id, self.counter.taken_counts, self.counter.returned_counts)
        draw_boxes(im0, self.tracked_products, track_ids)
        draw_frame_number(im0, frame_id)
        self._rendered_window.append((self._current_frame, im0.copy()))
        return im0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _upsert_product(
        self,
        track_id: int,
        bbox: np.ndarray,
        class_id: int,
        frame: int,
    ) -> Product:
        bbox_list  = bbox.tolist()
        class_name = self.names[int(class_id)]

        product = self.tracked_products.get(track_id)
        if product is not None:
            product.update(bbox_list, int(class_id), class_name, frame)
        else:
            cx = int((bbox_list[0] + bbox_list[2]) / 2)
            cy = int((bbox_list[1] + bbox_list[3]) / 2)
            product = Product(
                id=int(track_id),
                class_id=int(class_id),
                class_name=class_name,
                current_position=(cx, cy),
                bbox=bbox_list,
                last_seen_frame=frame,
            )
            self.tracked_products[track_id] = product
        return product
