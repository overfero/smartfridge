"""Drawing utilities: MediaPipe hand landmarks + bounding box rendering."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from smartfridge.counter.product import Product

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# MediaPipe hand landmark constants
_HAND_MARGIN: int = 10
_HAND_FONT_SCALE: float = 1.0
_HAND_FONT_THICKNESS: int = 1
_HAND_TEXT_COLOR: tuple = (88, 205, 54)  # BGR — vibrant green


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list     = detection_result.handedness
    annotated_image     = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness     = handedness_list[idx]

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        height, width, _ = annotated_image.shape
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - _HAND_MARGIN

        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            _HAND_FONT_SCALE,
            _HAND_TEXT_COLOR,
            _HAND_FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def compute_color_for_labels(label: int) -> tuple:
    color_map = {
        0: (85,  45,  255),   # person
        2: (222, 82,  175),   # car
        3: (0,   204, 255),   # motorbike
        5: (0,   149, 255),   # bus
    }
    if label in color_map:
        return color_map[label]
    return tuple(int((p * (label ** 2 - label + 1)) % 255) for p in palette)


def draw_border(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    color: tuple,
    thickness: int,
    r: int,
    d: int,
) -> np.ndarray:
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)
    return img


def _draw_labeled_bbox(
    x,
    img: np.ndarray,
    color: tuple | None = None,
    label: str | None = None,
    line_thickness: int = 2,
) -> None:
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    if color is None:
        color = tuple(np.random.randint(0, 255, 3).tolist())
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf     = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(
            img,
            (c1[0], c1[1] - t_size[1] - 3),
            (c1[0] + t_size[0], c1[1] + 3),
            color, 1, 8, 2,
        )
        cv2.putText(
            img, label, (c1[0], c1[1] - 2), 0, tl / 3,
            [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA,
        )


def draw_boxes(
    img: np.ndarray,
    stored_object: dict,
    identities,
) -> np.ndarray:
    """Render bounding boxes untuk semua tracked products."""
    for track_id in identities:
        if track_id not in stored_object:
            continue
        prod  = stored_object[track_id]
        color = compute_color_for_labels(prod.class_id)
        _draw_labeled_bbox(prod.bbox, img, label=f"#{track_id} {prod.class_name}", color=color)
    return img
