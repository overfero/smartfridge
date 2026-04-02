"""Pipeline profiler — ukur waktu per stage dan cetak ringkasan."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext

# Urutan stage sesuai pipeline
STAGES = ["decode", "preprocess", "infer", "postprocess", "track", "logic", "render", "write"]

_SEP_WIDTH = 64
_SEP       = "─" * _SEP_WIDTH


class Profiler:
    """Akumulasi waktu per stage dan cetak ringkasan di akhir."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {s: 0.0 for s in STAGES}

    @contextmanager
    def measure(self, stage: str):
        """Context manager: ukur waktu eksekusi blok dan tambahkan ke stage."""
        t = time.perf_counter()
        try:
            yield
        finally:
            self._totals[stage] = self._totals.get(stage, 0.0) + (time.perf_counter() - t)

    def report(self, n_frames: int) -> str:
        """Hasilkan string ringkasan profiling."""
        total_s   = sum(self._totals.values())
        total_fps = n_frames / total_s if total_s > 0 else 0.0

        lines = [
            _SEP,
            f"{'PIPELINE PROFILING SUMMARY':^{_SEP_WIDTH}}",
            _SEP,
            f"  {'Stage':<14}  {'Total (s)':>9}  {'Per-frame (ms)':>14}  {'%':>6}",
            _SEP,
        ]

        for stage in STAGES:
            t_s      = self._totals[stage]
            ms_frame = (t_s / n_frames * 1000) if n_frames > 0 else 0.0
            pct      = (t_s / total_s * 100) if total_s > 0 else 0.0
            lines.append(
                f"  {stage:<14}  {t_s:>9.4f}  {ms_frame:>14.2f}  {pct:>5.1f}%"
            )

        ms_total = (total_s / n_frames * 1000) if n_frames > 0 else 0.0
        lines += [
            _SEP,
            f"  {'TOTAL':<14}  {total_s:>9.4f}  {ms_total:>14.2f}  {'100.0':>5}%",
            _SEP,
            f"  Frames processed : {n_frames}",
            f"  Effective FPS    : {total_fps:.1f} fps",
            _SEP,
        ]
        return "\n".join(lines)


def measure_or_null(profiler: Profiler | None, stage: str):
    """Return profiler.measure(stage) atau nullcontext() jika profiler=None."""
    return profiler.measure(stage) if profiler is not None else nullcontext()
