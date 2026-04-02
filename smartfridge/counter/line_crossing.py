"""LineCrossingCounter — counts taken/returned products via virtual line crossing.

Encapsulates all state and logic for the current intersection-based approach so
it can be swapped out for a different counting strategy without touching predict.py.
"""

from __future__ import annotations

import io

import numpy as np

from smartfridge.counter.geometry import (
    intersect,
    is_point_above_line,
    is_point_below_line,
    get_direction,
)


class LineCrossingCounter:
    """Counts product taken/returned events by detecting virtual-line crossings.

    Usage:
        counter = LineCrossingCounter(virtual_line)
        # each frame, after tracker assigns IDs and products are upserted:
        counter.update(tracked_products, track_ids, frame)
        # read results:
        counter.taken_counts   # dict[class_name, int]
        counter.returned_counts
    """

    def __init__(
        self,
        virtual_line: tuple,
        camera_from_top: bool = True,
        debug: bool = False,
        debug_file: str = "debug.txt",
    ) -> None:
        self._camera_from_top = camera_from_top
        self.virtual_line = virtual_line
        self.debug = debug
        self._debug_fh: io.TextIOWrapper | None = None
        if debug:
            self._debug_fh = open(debug_file, "w", buffering=1)

        self.taken_counts: dict[str, int] = {}
        self.returned_counts: dict[str, int] = {}

        # Track IDs currently situated below / above the virtual line
        self._ids_below_line: set[int] = set()
        self._ids_above_line: set[int] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, tracked_products: dict, track_ids: list[int] | np.ndarray, frame: int) -> None:
        """Run crossing detection for the current frame.

        Mutates tracked_products (removes completed products) and updates
        taken_counts / returned_counts in-place.

        Args:
            tracked_products: dict mapping track_id → Product (shared with predictor).
            track_ids:        array of active track IDs this frame.
            frame:            current frame number.
        """
        self._remove_completed_products(tracked_products)

        if self.debug:
            self._print_debug(tracked_products, track_ids, frame)

        for track_id in track_ids:
            product = tracked_products.get(track_id)
            if product is None:
                continue

            product.append_anchor(self._camera_from_top)
            self._check_position_crossing(product, track_id)
            self._check_trail_crossing(product, track_id, frame)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _remove_completed_products(self, tracked_products: dict) -> None:
        for track_id in list(tracked_products.keys()):
            if tracked_products[track_id].is_complete:
                tracked_products.pop(track_id)

    def _check_position_crossing(self, product, track_id: int) -> None:
        """Detect crossing based on current position relative to the virtual line."""
        pos = product.current_position
        line = self.virtual_line

        if is_point_below_line(pos, line[0], line[1]):
            self._ids_below_line.add(track_id)

            # Already taken and first trail point also below → returned
            if (product.taken_counted and not product.return_counted
                    and len(product.trail_points) > 0
                    and is_point_below_line(product.trail_points[0], line[0], line[1])):
                self._mark_returned(product, track_id)

        elif is_point_above_line(pos, line[0], line[1]) and track_id in self._ids_below_line:
            if not product.taken_counted:
                self._mark_taken(product, track_id)

        if is_point_above_line(pos, line[0], line[1]):
            self._ids_above_line.add(track_id)

        elif is_point_below_line(pos, line[0], line[1]) and track_id in self._ids_above_line:
            if product.taken_counted and not product.return_counted:
                self._mark_returned(product, track_id)
                product.movement_direction = "South"

    def _check_trail_crossing(self, product, track_id: int, frame: int) -> None:
        """Detect crossing via segment intersect on the last two trail points."""
        if len(product.trail_points) < 2:
            return

        p0 = tuple(map(int, product.trail_points[0]))
        p1 = tuple(map(int, product.trail_points[1]))
        direction = get_direction(p0, p1)
        product.movement_direction = direction

        line = self.virtual_line
        if not intersect(p0, p1, line[0], line[1]):
            return

        safe_frame = frame if frame is not None else 0

        if "North" in direction and not product.taken_counted:
            self._mark_taken(product, track_id)
            product.last_seen_frame = safe_frame

        if "South" in direction and product.taken_counted and not product.return_counted:
            self._mark_returned(product, track_id)
            product.last_seen_frame = safe_frame

    def _print_debug(self, tracked_products: dict, track_ids, frame: int) -> None:
        line = self.virtual_line
        lines_out = [f"[LineCrossing] frame={frame}  tracking={list(track_ids)}"]
        for track_id in track_ids:
            p = tracked_products.get(track_id)
            if p is None:
                continue
            pos = p.current_position
            side = "below" if is_point_below_line(pos, line[0], line[1]) else \
                   "above" if is_point_above_line(pos, line[0], line[1]) else "on"
            trail_len = len(p.trail_points)
            lines_out.append(
                f"  id={track_id}  cls={p.class_name}  pos={pos}  side={side}"
                f"  trail_len={trail_len}  taken={p.taken_counted}  returned={p.return_counted}"
                f"  dir={p.movement_direction}"
            )
        self._debug_fh.write("\n".join(lines_out) + "\n")

    def _mark_taken(self, product, track_id: int) -> None:
        self.taken_counts[product.class_name] = self.taken_counts.get(product.class_name, 0) + 1
        product.mark_taken()
        self._ids_below_line.discard(track_id)

    def _mark_returned(self, product, track_id: int) -> None:
        self.returned_counts[product.class_name] = self.returned_counts.get(product.class_name, 0) + 1
        product.mark_returned()
        self._ids_above_line.discard(track_id)
