"""
SmartFridgeModel — direct ONNX Runtime inference dengan OpenVINO EP.

Menggantikan ultralytics.YOLO sebagai engine inferensi. Menangani:
- Pembuatan ORT InferenceSession dengan OpenVINO EP
- Letterbox preprocessing (pure NumPy/cv2)
- Scale-back bounding boxes ke koordinat gambar asli
- Membaca metadata model (names, imgsz) dari ONNX custom metadata

Model diasumsikan end2end (NMS sudah baked-in): output shape [1, N, 6]
dengan format [x1, y1, x2, y2, conf, cls] dalam koordinat letterbox.
"""

from __future__ import annotations

import ast
import os

import cv2
import numpy as np
import onnxruntime as ort

from smartfridge.core.profiler import Profiler, measure_or_null
from smartfridge.core.types import SimpleDetections


class SmartFridgeModel:
    """Wrapper ONNX Runtime untuk model YOLO11 end2end."""

    def __init__(self, model_path: str) -> None:
        available = ort.get_available_providers()
        if "OpenVINOExecutionProvider" in available:
            # Biarkan OpenVINO manage threading-nya sendiri — jangan override
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(model_path, providers=providers)
        else:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = os.cpu_count() or 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        active_providers = self.session.get_providers()
        if "OpenVINOExecutionProvider" in active_providers:
            ov_opts = self.session.get_provider_options().get("OpenVINOExecutionProvider", {})
            device = ov_opts.get("device_type", "CPU")
            print(f"[inference] running on OpenVINO ({device})")
        else:
            print("[inference] running on CPU (OnnxRuntime)")
        self.input_name: str = self.session.get_inputs()[0].name

        # Baca metadata dari model ONNX
        meta = self.session.get_modelmeta().custom_metadata_map
        raw_names = meta.get("names", "{0: 'object'}")
        self.names: dict[int, str] = {
            int(k): v for k, v in ast.literal_eval(raw_names).items()
        }
        imgsz_raw = meta.get("imgsz", "[640, 640]")
        self.imgsz: int = ast.literal_eval(imgsz_raw)[0]

        # Pre-alokasi buffer reusable — hindari alokasi per-frame
        self._canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        self._tensor = np.empty((3, self.imgsz, self.imgsz), dtype=np.float32)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _letterbox(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        """Resize gambar dengan letterbox padding ke imgsz x imgsz.

        Menulis ke buffer pre-alokasi untuk menghindari alokasi per-frame.

        Returns:
            tensor   : CHW float32 [0-1], view ke buffer internal
            r        : skala resize (dipakai untuk scale-back)
            pad_left : padding horizontal dalam piksel
            pad_top  : padding vertikal dalam piksel
        """
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_w = int(round(w * r))
        new_h = int(round(h * r))
        pad_left = int(round((self.imgsz - new_w) / 2 - 0.1))
        pad_top  = int(round((self.imgsz - new_h) / 2 - 0.1))

        # Reset padding area dan tulis hasil resize ke canvas yang sama
        self._canvas[:] = 114
        self._canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = (
            cv2.resize(img, (new_w, new_h))
        )
        # HWC BGR → CHW RGB, tulis langsung ke _tensor (tanpa alokasi baru)
        np.divide(
            self._canvas[:, :, ::-1].transpose(2, 0, 1),
            255.0,
            out=self._tensor,
        )
        return self._tensor, r, pad_left, pad_top

    # ── Inference ─────────────────────────────────────────────────────────────

    def infer(
        self,
        img: np.ndarray,
        conf_thresh: float = 0.25,
        profiler: Profiler | None = None,
    ) -> SimpleDetections:
        """Jalankan full inference pada satu frame.

        Args:
            img        : BGR numpy array (H, W, 3).
            conf_thresh: Ambang batas confidence minimum.
            profiler   : Profiler opsional — ukur preprocess/infer/postprocess.

        Returns:
            SimpleDetections dengan xyxy dalam koordinat gambar asli.
        """
        orig_h, orig_w = img.shape[:2]

        with measure_or_null(profiler, "preprocess"):
            tensor, r, pad_left, pad_top = self._letterbox(img)

        with measure_or_null(profiler, "infer"):
            # raw shape: (N_max, 6) — [x1, y1, x2, y2, conf, cls] dalam letterbox coords
            raw = self.session.run(None, {self.input_name: tensor[None]})[0][0]

        with measure_or_null(profiler, "postprocess"):
            mask = raw[:, 4] >= conf_thresh
            dets = raw[mask]  # (N, 6)

            if len(dets) == 0:
                return SimpleDetections(
                    xyxy=np.empty((0, 4), dtype=np.float32),
                    conf=np.empty((0,),   dtype=np.float32),
                    cls= np.empty((0,),   dtype=np.float32),
                )

            # Scale-back dari letterbox ke koordinat asli
            boxes = dets[:, :4].copy()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / r
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top)  / r
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)

            result = SimpleDetections(
                xyxy=boxes.astype(np.float32),
                conf=dets[:, 4].astype(np.float32),
                cls= dets[:, 5].astype(np.float32),
            )

        return result
