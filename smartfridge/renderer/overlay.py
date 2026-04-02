"""Common overlay drawing utilities."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np


def draw_taken_counts_panel(
    im0: np.ndarray,
    taken_counts: dict,
    returned_counts: dict,
    ui: SimpleNamespace,
) -> None:
    """Gambar panel 'Products Taken' dengan net count (taken - returned)."""
    net_counts = {k: v - returned_counts.get(k, 0) for k, v in taken_counts.items()}
    for k, v in returned_counts.items():
        if k not in net_counts:
            net_counts[k] = -v
    displayed = [(k, cnt) for k, cnt in net_counts.items() if cnt != 0]

    header_y = ui.top_margin + 15
    if displayed:
        cv2.line(im0,
                 (ui.left_margin, ui.top_margin),
                 (ui.box_width, ui.top_margin),
                 ui.box_color, ui.line_height)
        cv2.putText(im0, "Products Taken",
                    (ui.left_margin - 9, header_y),
                    0, 1, ui.text_color,
                    thickness=ui.text_thickness, lineType=cv2.LINE_AA)

    row_start_y = ui.top_margin + ui.line_height
    for idx, (key, value) in enumerate(displayed):
        row_y = row_start_y + idx * ui.line_height
        cv2.line(im0,
                 (ui.left_margin, row_y),
                 (ui.box_width, row_y),
                 ui.box_color, ui.line_height - 10)
        cv2.putText(im0, f"{key}: {value}",
                    (ui.left_margin - 9, row_y + 10),
                    0, 1, ui.text_color,
                    thickness=ui.text_thickness, lineType=cv2.LINE_AA)


def draw_frame_number(im0: np.ndarray, frame: int | None) -> None:
    """Gambar nomor frame di pojok kanan bawah."""
    height, width = im0.shape[:2]
    text = f"Frame: {frame if frame is not None else 0}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness, margin = 0.8, 2, 15
    (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.putText(im0, text,
                (width - text_w - margin, height - margin),
                font, font_scale, (255, 255, 255),
                thickness=thickness, lineType=cv2.LINE_AA)
