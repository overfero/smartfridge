"""
SmartFridge — entry point tracking.

Model dan semua parameter dibaca dari smartfridge/cfg/default.yaml.
Untuk ganti model: ubah field `model:` di default.yaml ke nano_v1/nano_v2/small_v1/small_v2.
"""

from __future__ import annotations

import time

from smartfridge import SmartFridgeYOLO

SOURCE = "/home/overfero/Project/glair/Jumpstart - Smart Fridge/Ambil Biasa - Atas/WIN_20260126_10_28_41_Pro.mp4"

model = SmartFridgeYOLO()

t_start     = time.perf_counter()
frame_count = 0

for _ in model.track(source=SOURCE, profiling=model.cfg.profiling):
    frame_count += 1

t_elapsed = time.perf_counter() - t_start
fps        = frame_count / t_elapsed
ms_frame   = t_elapsed / frame_count * 1000
print(f"Total: {t_elapsed:.2f}s  |  {ms_frame:.1f} ms/frame  |  {fps:.1f} FPS  ({frame_count} frames)")
