"""ClassBasedReID — class-dependent track re-identification and class smoothing.

Restores lost track IDs when a new track appears with the same class as a
recently-lost track. Also stabilises the displayed class via mode voting over
the last 5 detections.

Depends on class labels from the detection model — not usable when running
class-agnostic detection.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from smartfridge.trackers.utils.iou import calculate_iou

DEBUG_FILE = "debug_track.txt"


class ClassBasedReID:
    def __init__(self, args) -> None:
        self.args = args
        # {track_id: {'last_frame': int, 'cls': int}}
        self.lost_tracks: dict = {}
        # {track_id: [frame_ids]}
        self.track_history: dict = {}
        # {track_id: int}
        self.track_last_class: dict = {}
        # {track_id: [class_ids]}  — bounded to last 50 entries
        self.track_class_history: dict = {}

    def apply(
        self,
        output_tracks: np.ndarray,
        raw_tracks: np.ndarray,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        frame_id: int,
        trackers: list,
    ) -> np.ndarray:
        """Apply class-based re-ID and class smoothing. Modifies output_tracks in-place."""
        debug = bool(getattr(self.args, 'debug_track', False))
        log: list[str] = []

        current_track_ids: set[int] = set()

        for i, track in enumerate(raw_tracks):
            track_box = track[:4]
            track_id  = int(output_tracks[i, 4])
            current_track_ids.add(track_id)

            # Get raw class via best-IoU detection match
            ious = calculate_iou(track_box, xyxy)
            track_cls = -1
            if len(ious) > 0:
                best = np.argmax(ious)
                if ious[best] > 0.01:
                    track_cls = int(cls[best])

            # Class-based ID recovery: find lost track with same class, closest ID
            matched_lost_id = None
            min_id_diff = float('inf')
            for lost_id, info in self.lost_tracks.items():
                if lost_id not in current_track_ids and track_cls == info['cls']:
                    d = abs(track_id - lost_id)
                    if d < min_id_diff:
                        min_id_diff = d
                        matched_lost_id = lost_id

            if matched_lost_id is not None:
                output_tracks[i, 4] = matched_lost_id
                for trk in trackers:
                    if trk.id + 1 == track_id:
                        trk.id = matched_lost_id - 1
                        break
                del self.lost_tracks[matched_lost_id]
                current_track_ids.add(matched_lost_id)
                current_track_ids.discard(track_id)
                if debug:
                    log.append(
                        f"  RECOVER  new={track_id} → restored={matched_lost_id}"
                        f"  cls={track_cls}  id_diff={int(min_id_diff)}"
                    )
                track_id = matched_lost_id

            # Appearance history
            self.track_history.setdefault(track_id, []).append(frame_id)

            # Class smoothing: mode of last 5 classes
            raw_class = int(output_tracks[i, 6])
            hist = self.track_class_history.setdefault(track_id, [])
            hist.append(raw_class)
            if len(hist) > 50:
                hist.pop(0)
            smoothed = raw_class
            try:
                smoothed = int(Counter(hist[-5:]).most_common(1)[0][0])
                output_tracks[i, 6] = smoothed
            except Exception:
                pass
            if debug and smoothed != raw_class:
                log.append(
                    f"  SMOOTH   id={track_id}  raw_cls={raw_class} → smoothed={smoothed}"
                    f"  history={hist[-5:]}"
                )

        # Persist last known class
        for track in output_tracks:
            self.track_last_class[int(track[4])] = int(track[6])

        # Remove re-appeared IDs from lost_tracks
        for tid in list(self.lost_tracks):
            if tid in current_track_ids:
                self.lost_tracks.pop(tid, None)

        # Register newly-lost tracks that had ≥5 appearances in the last 15 frames
        previous_ids = set(self.lost_tracks) | set(self.track_history)
        for tid in previous_ids - current_track_ids:
            if tid not in self.track_history:
                continue
            recent = [f for f in self.track_history[tid] if frame_id - f <= 15]
            if len(recent) < 5:
                continue
            hist = self.track_class_history.get(tid, [])[-5:]
            try:
                track_cls = int(Counter(hist).most_common(1)[0][0]) if hist \
                    else self.track_last_class.get(tid, 0)
            except Exception:
                track_cls = 0
            if tid not in self.lost_tracks:
                self.lost_tracks[tid] = {'last_frame': frame_id, 'cls': track_cls}
                if debug:
                    log.append(f"  LOST_REG id={tid}  cls={track_cls}  appearances={len(recent)}/15f")

        # Keep class fresh for any lost track still visible
        for track in output_tracks:
            tid = int(track[4])
            if tid in self.lost_tracks:
                self.lost_tracks[tid]['cls'] = int(track[6])

        # Expire stale lost tracks
        expire_after = int(getattr(self.args, 'lost_track_expire_frames', 90))
        for tid, info in list(self.lost_tracks.items()):
            if frame_id - int(info.get('last_frame', 0)) > expire_after:
                self.lost_tracks.pop(tid, None)
                if debug:
                    log.append(f"  EXPIRED  id={tid}")

        if debug:
            self._write_debug(frame_id, log, current_track_ids)

        return output_tracks

    def _write_debug(self, frame_id: int, log: list[str], active_ids: set) -> None:
        lost_summary = {tid: info['cls'] for tid, info in self.lost_tracks.items()}
        try:
            with open(DEBUG_FILE, "a") as f:
                f.write(f"\n--- CLASS_REID | frame={frame_id} ---\n")
                f.write(f"  active={sorted(active_ids)}  lost={lost_summary}\n")
                if log:
                    f.write("\n".join(log) + "\n")
                else:
                    f.write("  (no events)\n")
        except Exception:
            pass
