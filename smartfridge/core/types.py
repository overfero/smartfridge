"""
Tipe data standalone pengganti ultralytics.engine.results.Results dan Boxes.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimpleDetections:
    """Output SmartFridgeModel.infer() — diterima oleh HybridSORT.update().

    Attributes:
        xyxy: Bounding boxes (N, 4) float32 dalam koordinat gambar asli.
        conf: Confidence scores (N,) float32.
        cls:  Class indices (N,) float32.
    """

    xyxy: np.ndarray
    conf: np.ndarray
    cls:  np.ndarray

    def __len__(self) -> int:
        return len(self.conf)


class SimpleBoxes:
    """Pengganti ultralytics.engine.results.Boxes.

    Data layout: (N, 7) → [x1, y1, x2, y2, track_id, conf, cls]
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = data  # shape (N, 7)

    @property
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4]

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 5]

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 6]

    @property
    def id(self) -> np.ndarray | None:
        return self.data[:, 4] if len(self.data) > 0 else None


@dataclass
class SimpleResult:
    """Pengganti ultralytics.engine.results.Results — satu frame."""

    orig_img: np.ndarray    # BGR numpy (H, W, 3)
    boxes: SimpleBoxes
    names: dict             # {int_id: str_name}
    speed: dict = field(default_factory=lambda: {"inference": 0.0})
