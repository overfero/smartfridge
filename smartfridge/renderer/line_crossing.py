"""LineCrossingRenderer — draws virtual line overlay for the LineCrossingCounter."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from smartfridge.renderer.overlay import draw_taken_counts_panel, draw_frame_number

_DEFAULT_LINE_COLOR = (46, 162, 112)  # BGR — hijau


class LineCrossingRenderer:
    """Renders visual overlays for the single-line crossing counting strategy."""

    def __init__(
        self,
        virtual_line: tuple,
        ui: SimpleNamespace,
        line_color: tuple = _DEFAULT_LINE_COLOR,
        line_thickness: int = 3,
    ) -> None:
        self.virtual_line = virtual_line
        self._ui = ui
        self._line_color = line_color
        self._line_thickness = line_thickness

    def draw(
        self,
        im0: np.ndarray,
        frame: int,
        taken_counts: dict,
        returned_counts: dict,
    ) -> None:
        cv2.line(im0, self.virtual_line[0], self.virtual_line[1],
                 self._line_color, self._line_thickness)
        draw_taken_counts_panel(im0, taken_counts, returned_counts, self._ui)
        draw_frame_number(im0, frame)
