"""Product — tracked object dengan informasi posisi dan state counting."""

from __future__ import annotations

from collections import deque

_TRAIL_BUFFER_SIZE = 64  # jumlah titik trail yang disimpan per produk


class Product:
    """Merepresentasikan satu produk yang sedang di-track.

    Attributes:
        id:                 Track ID unik.
        class_id:           Class ID dari model deteksi.
        class_name:         Nama kelas yang mudah dibaca.
        current_position:   Posisi pusat bbox saat ini (x, y).
        bbox:               Bounding box [x1, y1, x2, y2].
        trail_points:       Riwayat posisi (maxlen=_TRAIL_BUFFER_SIZE).
        is_below_line:      Apakah produk berada di bawah virtual line.
        movement_direction: Arah gerakan (North/South/East/West).
        taken_counted:      Sudah dihitung sebagai diambil.
        return_counted:     Sudah dihitung sebagai dikembalikan.
        last_seen_frame:    Nomor frame terakhir terdeteksi.
    """

    def __init__(
        self,
        id: int,
        class_id: int,
        class_name: str,
        current_position: tuple[int, int],
        bbox: list[int],
        trail_points: list[tuple[int, int]] | None = None,
        is_below_line: bool = False,
        movement_direction: str = "",
        taken_counted: bool = False,
        return_counted: bool = False,
        last_seen_frame: int = 0,
    ) -> None:
        self.id                 = id
        self.class_id           = class_id
        self.class_name         = class_name
        self.current_position   = current_position
        self.bbox               = bbox
        self.trail_points       = deque(trail_points or [], maxlen=_TRAIL_BUFFER_SIZE)
        self.is_below_line      = is_below_line
        self.movement_direction = movement_direction
        self.taken_counted      = taken_counted
        self.return_counted     = return_counted
        self.last_seen_frame    = last_seen_frame

    # ── Frame update ──────────────────────────────────────────────────────────

    def update(self, bbox: list[int], class_id: int, class_name: str, frame: int) -> None:
        """Update state dari deteksi baru dan tambahkan pusat bbox ke trail."""
        self.bbox         = bbox
        self.class_id     = class_id
        self.class_name   = class_name
        self.last_seen_frame = frame
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.current_position = (cx, cy)
        self.trail_points.append(self.current_position)

    def append_anchor(self, camera_from_top: bool) -> tuple[int, int]:
        """Prepend titik anchor berbasis kamera untuk deteksi crossing.

        Kamera atas  → anchor di bawah-tengah bbox (dekat tepi rak).
        Kamera bawah → anchor di dekat atas bbox.
        """
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        if camera_from_top:
            anchor = (int((x1 + x2) / 2), int(y2))
        else:
            # 0.8 * y1: sedikit di bawah tepi atas bbox untuk kamera bawah
            anchor = (int((x1 + x2) / 2), int(y1 * 0.8))
        self.trail_points.appendleft(anchor)
        return anchor

    # ── Counting state ────────────────────────────────────────────────────────

    def mark_taken(self) -> None:
        """Tandai produk sebagai diambil dari rak."""
        self.taken_counted = True

    def mark_returned(self) -> None:
        """Tandai produk sebagai dikembalikan ke rak."""
        self.return_counted = True

    @property
    def is_complete(self) -> bool:
        """True setelah produk dikembalikan."""
        return self.return_counted

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id":                 self.id,
            "class_id":           self.class_id,
            "class_name":         self.class_name,
            "current_position":   self.current_position,
            "trail_points":       list(self.trail_points),
            "is_below_line":      self.is_below_line,
            "movement_direction": self.movement_direction,
            "taken_counted":      self.taken_counted,
            "return_counted":     self.return_counted,
            "last_seen_frame":    self.last_seen_frame,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Product:
        return cls(
            id=data["id"],
            class_id=data["class_id"],
            class_name=data["class_name"],
            current_position=data["current_position"],
            trail_points=data.get("trail_points", []),
            is_below_line=data.get("is_below_line", False),
            movement_direction=data.get("movement_direction", ""),
            taken_counted=data.get("taken_counted", False),
            return_counted=data.get("return_counted", False),
            last_seen_frame=data.get("last_seen_frame", 0),
        )

    def merge_trail(self, other_trail: list[tuple[int, int]]) -> None:
        """Gabungkan trail dari track lain ke depan trail ini."""
        combined = list(other_trail) + list(self.trail_points)
        self.trail_points = deque(combined, maxlen=_TRAIL_BUFFER_SIZE)

    def __repr__(self) -> str:
        return f"Product(id={self.id}, class='{self.class_name}', pos={self.current_position})"

    def __str__(self) -> str:
        return f"{self.class_name} #{self.id} at {self.current_position}"
