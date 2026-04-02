"""SpatialReID — class-agnostic track re-identification via spatial heuristics.

Two strategies for a fixed-camera, monotonic-motion environment:

1. Blink direction recovery
   A product moving consistently up or down briefly disappears (occluded by
   hand). If a new detection appears further along the same direction within
   `blink_max_frames` and `blink_max_dist_px`, the old track ID is restored.

2. Top-third re-entry
   A product last seen in the upper 1/3 of the frame reappears in the upper
   1/3 after a pause (user is examining the product before returning it).
   The old track ID is restored.

Does NOT use class labels — works even with class-agnostic detectors.
"""

from __future__ import annotations

import numpy as np

DEBUG_FILE = "debug_track.txt"


class SpatialReID:
    def __init__(self, args) -> None:
        self.args = args
        # {track_id: {'last_frame', 'last_position', 'movement_direction'}}
        self.lost_tracks: dict = {}
        # {track_id: [(cx, cy), ...]}  — bounded to last 10 positions
        self._position_history: dict = {}

    def apply(
        self,
        output_tracks: np.ndarray,
        raw_tracks: np.ndarray,
        frame_id: int,
        frame_height: int,
        trackers: list,
    ) -> np.ndarray:
        """Apply spatial re-ID. Modifies output_tracks in-place."""
        blink_max_frames     = int(getattr(self.args, 'blink_max_frames',     45))
        blink_max_dist_px    = float(getattr(self.args, 'blink_max_dist_px',  400))
        top_third_max_frames = int(getattr(self.args, 'top_third_max_frames', 120))
        debug = bool(getattr(self.args, 'debug_track', False))
        log: list[str] = []

        current_track_ids: set[int] = set()

        for i in range(len(raw_tracks)):
            track_id = int(output_tracks[i, 4])
            current_track_ids.add(track_id)

            new_pos = (
                float((output_tracks[i, 0] + output_tracks[i, 2]) / 2),
                float((output_tracks[i, 1] + output_tracks[i, 3]) / 2),
            )

            matched_lost_id = None
            match_method    = None

            # ── Priority 1: blink direction recovery ─────────────────────────
            best_dist = float('inf')
            for lost_id, info in self.lost_tracks.items():
                if lost_id in current_track_ids:
                    continue
                if self._match_blink_direction(new_pos, info, frame_id, blink_max_frames, blink_max_dist_px):
                    last_pos = info['last_position']
                    dist = _dist(new_pos, last_pos)
                    if dist < best_dist:
                        best_dist = dist
                        matched_lost_id = lost_id
                        match_method = ('blink', info, dist)

            # ── Priority 2: top-third re-entry ───────────────────────────────
            if matched_lost_id is None:
                best_dist = float('inf')
                for lost_id, info in self.lost_tracks.items():
                    if lost_id in current_track_ids:
                        continue
                    if self._match_top_third(new_pos, info, frame_id, frame_height, top_third_max_frames):
                        last_pos = info.get('last_position') or (0.0, 0.0)
                        dist = _dist(new_pos, last_pos)
                        if dist < best_dist:
                            best_dist = dist
                            matched_lost_id = lost_id
                            match_method = ('top3rd', info, dist)

            # ── Apply match ───────────────────────────────────────────────────
            if matched_lost_id is not None:
                output_tracks[i, 4] = matched_lost_id
                for trk in trackers:
                    if trk.id + 1 == track_id:
                        trk.id = matched_lost_id - 1
                        break
                del self.lost_tracks[matched_lost_id]
                current_track_ids.add(matched_lost_id)
                current_track_ids.discard(track_id)
                if debug and match_method is not None:
                    method, info, dist = match_method
                    age = frame_id - info['last_frame']
                    last_pos = info.get('last_position', (0, 0))
                    direction = info.get('movement_direction', 'none')
                    if method == 'blink':
                        log.append(
                            f"  BLINK    new={track_id} → restored={matched_lost_id}"
                            f"  dir={direction}  dist={dist:.0f}px  age={age}f"
                            f"  last_pos=({last_pos[0]:.0f},{last_pos[1]:.0f})"
                            f"  new_pos=({new_pos[0]:.0f},{new_pos[1]:.0f})"
                        )
                    else:
                        log.append(
                            f"  TOP3RD   new={track_id} → restored={matched_lost_id}"
                            f"  age={age}f"
                            f"  last_pos=({last_pos[0]:.0f},{last_pos[1]:.0f})"
                            f"  new_pos=({new_pos[0]:.0f},{new_pos[1]:.0f})"
                        )
                track_id = matched_lost_id

            # Update position history
            hist = self._position_history.setdefault(track_id, [])
            hist.append(new_pos)
            if len(hist) > 10:
                hist.pop(0)

        # Remove re-appeared IDs
        for tid in list(self.lost_tracks):
            if tid in current_track_ids:
                self.lost_tracks.pop(tid, None)

        # Register newly-lost tracks (any track that disappeared this frame)
        expire_after = int(getattr(self.args, 'lost_track_expire_frames', 90))
        all_known = set(self.lost_tracks) | set(self._position_history)
        for tid in all_known - current_track_ids:
            pos_list = self._position_history.get(tid)
            if not pos_list:
                continue
            if tid not in self.lost_tracks:
                direction = self._estimate_direction(tid)
                self.lost_tracks[tid] = {
                    'last_frame':         frame_id,
                    'last_position':      pos_list[-1],
                    'movement_direction': direction,
                }
                if debug:
                    log.append(
                        f"  LOST_REG id={tid}"
                        f"  pos=({pos_list[-1][0]:.0f},{pos_list[-1][1]:.0f})"
                        f"  dir={direction}"
                    )

        # Expire stale entries
        for tid, info in list(self.lost_tracks.items()):
            if frame_id - int(info.get('last_frame', 0)) > expire_after:
                self.lost_tracks.pop(tid, None)
                if debug:
                    log.append(f"  EXPIRED  id={tid}")

        if debug:
            self._write_debug(frame_id, log, current_track_ids)

        return output_tracks

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _estimate_direction(self, track_id: int) -> str | None:
        positions = self._position_history.get(track_id, [])
        if len(positions) < 3:
            return None
        recent = positions[-3:]
        dys = [recent[j + 1][1] - recent[j][1] for j in range(len(recent) - 1)]
        mean_dy = sum(dys) / len(dys)
        if mean_dy < -5:
            return 'up'
        if mean_dy > 5:
            return 'down'
        return None

    def _match_blink_direction(
        self, new_pos, lost_info, frame_id, blink_max_frames, blink_max_dist_px
    ) -> bool:
        direction = lost_info.get('movement_direction')
        last_pos  = lost_info.get('last_position')
        if direction is None or last_pos is None:
            return False
        if frame_id - lost_info['last_frame'] > blink_max_frames:
            return False
        if _dist(new_pos, last_pos) > blink_max_dist_px:
            return False
        if direction == 'up'   and new_pos[1] < last_pos[1]:
            return True
        if direction == 'down' and new_pos[1] > last_pos[1]:
            return True
        return False

    def _match_top_third(
        self, new_pos, lost_info, frame_id, frame_height, top_third_max_frames
    ) -> bool:
        last_pos = lost_info.get('last_position')
        if last_pos is None:
            return False
        if frame_id - lost_info['last_frame'] > top_third_max_frames:
            return False
        threshold = frame_height / 3
        return last_pos[1] < threshold and new_pos[1] < threshold

    def _write_debug(self, frame_id: int, log: list[str], active_ids: set) -> None:
        lost_summary = {
            tid: f"pos=({info['last_position'][0]:.0f},{info['last_position'][1]:.0f}) dir={info['movement_direction']}"
            for tid, info in self.lost_tracks.items()
        }
        try:
            with open(DEBUG_FILE, "a") as f:
                f.write(f"\n--- SPATIAL_REID | frame={frame_id} ---\n")
                f.write(f"  active={sorted(active_ids)}\n")
                if lost_summary:
                    f.write(f"  lost={{\n")
                    for tid, summary in lost_summary.items():
                        f.write(f"    {tid}: {summary}\n")
                    f.write(f"  }}\n")
                else:
                    f.write("  lost={}\n")
                if log:
                    f.write("\n".join(log) + "\n")
                else:
                    f.write("  (no events)\n")
        except Exception:
            pass


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
