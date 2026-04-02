"""OutputSaver — saves crossing events to outputs/ folder.

Folder structure:
  outputs/
    full_image/  frame_{n}.jpg   — full rendered frame at crossing moment
    product/     frame_{n}.jpg   — cropped bbox of the product that crossed
    cloud.csv                    — one row per crossing event
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


OUTPUT_DIR = Path("outputs")
CSV_COLUMNS = ["cross_frame", "best_frame", "track_id", "class", "x1", "y1", "x2", "y2", "line", "direction"]


class OutputSaver:
    """Menyimpan crossing events ke folder outputs/.

    Struktur folder:
        outputs/
            full_image/   frame_{n}.jpg   — frame lengkap saat crossing
            product/      frame_{n}.jpg   — crop bbox produk yang crossing
            cloud.csv                     — satu baris per event

    CSV columns: cross_frame, best_frame, track_id, class,
                 x1, y1, x2, y2, line, direction
    CSV di-reset setiap sesi baru.
    """

    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        self._full = output_dir / "full_image"
        self._product = output_dir / "product"
        for d in (self._full, self._product):
            d.mkdir(parents=True, exist_ok=True)

        self._csv_path = output_dir / "cloud.csv"
        # Always reset CSV at session start
        with open(self._csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)

    def save(
        self,
        cross_frame: int,
        best_frame: int,
        rendered_frame: np.ndarray,
        track_id: int,
        class_name: str,
        bbox: list,          # [x1, y1, x2, y2]
        line_idx: int,       # 0-based, so L1 = 0
        direction: str,      # "taken" | "returned"
    ) -> None:
        stem = f"frame_{cross_frame}_t{best_frame}"

        x1, y1, x2, y2 = [int(v) for v in bbox]
        line_name = f"L{line_idx + 1}"

        # ── full image ────────────────────────────────────────────────────────
        cv2.imwrite(str(self._full / f"{stem}.jpg"), rendered_frame)

        # ── product crop ──────────────────────────────────────────────────────
        h, w = rendered_frame.shape[:2]
        crop = rendered_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size > 0:
            cv2.imwrite(str(self._product / f"{stem}.jpg"), crop)

        # ── append row to cloud.csv ───────────────────────────────────────────
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([cross_frame, best_frame, track_id, class_name, x1, y1, x2, y2, line_name, direction])
