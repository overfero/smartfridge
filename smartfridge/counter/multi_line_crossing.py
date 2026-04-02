"""MultiLineCrossingCounter — deferred history-based taken/returned detection.

Logic:
  - 5 virtual lines ordered bottom → top (index 0 = L1, index 4 = L5).
  - Per track, ALL line crossings are recorded in order.
  - When the track disappears, finalize() evaluates first vs last crossed line:
      * |first - last| < 2  → INVALID (only crossed 1-2 lines), no event
      * last > first         → TAKEN  (moved upward)
      * last < first         → RETURNED (moved downward)
  - For each line crossed for the FIRST TIME, on_first_cross callback fires so
    the caller can capture the best frame from its rolling buffer.
  - taken_counts / returned_counts are updated only on valid finalized events.
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np

from smartfridge.counter.geometry import intersect


class MultiLineCrossingCounter:
    def __init__(
        self,
        lines: list[tuple],
        debug: bool = False,
        debug_file: str = "debug_counter.txt",
    ) -> None:
        self.lines = lines
        self.debug = debug
        self._debug_fh: io.TextIOWrapper | None = None
        if debug:
            self._debug_fh = open(debug_file, "w", buffering=1)

        self.taken_counts: dict[str, int] = {}
        self.returned_counts: dict[str, int] = {}

        # Fired when a line is crossed for the FIRST TIME by a track.
        # Signature: on_first_cross(track_id, line_idx, frame)
        self.on_first_cross: Callable | None = None

        self._crossing_history: dict[int, list[int]] = {}
        self._first_cross: dict[int, int] = {}
        self._last_cross: dict[int, int] = {}
        self._first_crossed_lines: dict[int, set[int]] = {}
        self._prev_positions: dict[int, tuple] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, tracked_products: dict, track_ids: list[int] | np.ndarray, frame: int) -> None:
        """Detect crossings for the current frame and fire on_first_cross for new lines."""
        if self.debug:
            self._print_debug(tracked_products, track_ids, frame)

        for track_id in track_ids:
            product = tracked_products.get(track_id)
            if product is None:
                continue

            curr_pos = product.current_position
            prev_pos = self._prev_positions.get(track_id)
            self._prev_positions[track_id] = curr_pos

            if prev_pos is None:
                continue

            crossed_indices = self._find_crossed_lines(prev_pos, curr_pos)
            if not crossed_indices:
                continue

            history    = self._crossing_history.setdefault(track_id, [])
            seen_lines = self._first_crossed_lines.setdefault(track_id, set())

            for crossed_idx in crossed_indices:
                if history and history[-1] == crossed_idx:
                    continue

                history.append(crossed_idx)
                product.movement_direction = self._direction_label(history)

                if track_id not in self._first_cross:
                    self._first_cross[track_id] = crossed_idx

                self._last_cross[track_id] = crossed_idx

                if crossed_idx not in seen_lines:
                    seen_lines.add(crossed_idx)
                    if self.on_first_cross:
                        self.on_first_cross(track_id, crossed_idx, frame)

    def finalize(self, track_id: int) -> dict | None:
        """Evaluate a completed track. Returns event dict or None if invalid."""
        first = self._first_cross.get(track_id)
        last  = self._last_cross.get(track_id)

        history        = self._crossing_history.get(track_id, [])[:]
        hist_str       = "→".join(f"L{i+1}" for i in history) if history else "(none)"
        captured_lines = self._first_crossed_lines.get(track_id, set()).copy()

        if len(history) >= 2 and history[-1] < history[-2]:
            last = last - 1

        if first is None or last is None or abs(first - last) < 2:
            if self.debug and self._debug_fh:
                span = abs(first - last) if (first is not None and last is not None) else 0
                reason = "no crossings" if first is None else f"span={span} < 2"
                self._debug_fh.write(
                    f"[FINALIZE INVALID] id={track_id}  {reason}"
                    f"  history={hist_str}\n"
                )
            self._cleanup(track_id)
            return None

        direction = "taken" if last > first else "returned"

        if self.debug and self._debug_fh:
            self._debug_fh.write(
                f"[FINALIZE VALID] id={track_id}  direction={direction}"
                f"  first=L{first+1}  last=L{last+1} (effective)  span={abs(first-last)}"
                f"  history={hist_str}\n"
            )

        self._cleanup(track_id)
        return {
            "valid":          True,
            "direction":      direction,
            "first":          first,
            "last":           last,
            "captured_lines": captured_lines,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cleanup(self, track_id: int) -> None:
        self._crossing_history.pop(track_id, None)
        self._first_cross.pop(track_id, None)
        self._last_cross.pop(track_id, None)
        self._first_crossed_lines.pop(track_id, None)
        self._prev_positions.pop(track_id, None)

    def _find_crossed_lines(self, p0: tuple, p1: tuple) -> list[int]:
        """Return all line indices crossed by segment p0→p1."""
        crossed = [
            idx for idx, (start, end) in enumerate(self.lines)
            if intersect(p0, p1, start, end)
        ]
        if not crossed:
            return []
        moving_up = p1[1] < p0[1]
        return sorted(crossed) if moving_up else sorted(crossed, reverse=True)

    def _direction_label(self, history: list[int]) -> str:
        if len(history) < 2:
            return ""
        return "North" if history[-1] > history[-2] else "South"

    def _print_debug(self, tracked_products: dict, track_ids, frame: int) -> None:
        lines_out = [f"[MultiLine] frame={frame}  tracking={list(track_ids)}"]
        for track_id in track_ids:
            p = tracked_products.get(track_id)
            if p is None:
                continue
            history  = self._crossing_history.get(track_id, [])
            hist_str = "→".join(f"L{i+1}" for i in history) if history else "-"
            first    = self._first_cross.get(track_id)
            last     = self._last_cross.get(track_id)
            prev_pos = self._prev_positions.get(track_id)
            segment  = f"{prev_pos}→{p.current_position}" if prev_pos else "first_frame"
            lines_out.append(
                f"  id={track_id}  cls={p.class_name}  pos={p.current_position}"
                f"  history={hist_str}  first=L{first+1 if first is not None else '?'}"
                f"  last=L{last+1 if last is not None else '?'}  seg={segment}"
            )
        self._debug_fh.write("\n".join(lines_out) + "\n")
