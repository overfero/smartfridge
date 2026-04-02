"""MultiLineCrossingRenderer — draws 5-line overlay for MultiLineCrossingCounter."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from smartfridge.renderer.overlay import draw_taken_counts_panel


class MultiLineCrossingRenderer:
    """Renders virtual crossing lines and the counts panel onto a frame."""

    def __init__(
        self,
        lines: list[tuple],
        colors: list[tuple],
        line_thickness: int,
        ui: SimpleNamespace,
    ) -> None:
        self.lines = lines
        self.colors = colors
        self.line_thickness = line_thickness
        self._ui = ui

    def draw(
        self,
        im0: np.ndarray,
        frame: int,
        taken_counts: dict,
        returned_counts: dict,
    ) -> None:
        """Draw all overlays onto im0 in-place."""
        self._draw_crossing_lines(im0)
        draw_taken_counts_panel(im0, taken_counts, returned_counts, self._ui)

    def _draw_crossing_lines(self, im0: np.ndarray) -> None:
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.lines):
            color = self.colors[idx]
            cv2.line(im0, (x1, y1), (x2, y2), color, self.line_thickness)
            cv2.putText(
                im0, f"L{idx + 1}",
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )
