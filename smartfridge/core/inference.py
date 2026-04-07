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

    def __init__(self, model_path: str, num_threads: int | None = None) -> None:
        available = ort.get_available_providers()
        if "OpenVINOExecutionProvider" in available:
            # Batasi thread OpenVINO sesuai core yang di-pin — tanpa ini OV spawn
            # thread sebanyak total CPU logical core meski process sudah di-pin,
            # menyebabkan kontestasi dan FPS tidak stabil.
            n_threads = num_threads or len(os.sched_getaffinity(0))
            ov_options = {"num_of_threads": str(n_threads)}
            providers = [("OpenVINOExecutionProvider", ov_options), "CPUExecutionProvider"]
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

        self._output_name: str = self.session.get_outputs()[0].name

        # Buffer dan IO binding di-init saat frame pertama tiba (_init_buffers).
        # Perlu tahu resolusi frame dulu untuk hitung inp_h × inp_w rect.
        self._inp_h:    int | None = None
        self._inp_w:    int | None = None
        self._canvas:   np.ndarray | None = None
        self._tensor:   np.ndarray | None = None
        self._io_binding = None

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _init_buffers(self, frame_h: int, frame_w: int) -> None:
        """Hitung inp_h × inp_w rect dan alokasi buffer saat frame pertama tiba.

        Rect-inference: pertahankan aspect ratio, pad satu sisi saja, bulatkan
        ke kelipatan 32 (syarat stride YOLO). Lebih sedikit padding → lebih
        sedikit piksel yang diproses model vs square 640 × 640.
        """
        scale        = self.imgsz / max(frame_h, frame_w)
        self._inp_h  = int(round(frame_h * scale / 32)) * 32
        self._inp_w  = int(round(frame_w * scale / 32)) * 32
        self._canvas = np.full((self._inp_h, self._inp_w, 3), 114, dtype=np.uint8)
        self._tensor = np.empty((3, self._inp_h, self._inp_w), dtype=np.float32)
        self._io_binding = self.session.io_binding()
        print(f"[inference] input {frame_w}×{frame_h}  →  tensor {self._inp_w}×{self._inp_h}")

    def _letterbox(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        """Resize gambar dengan letterbox padding ke inp_h × inp_w.

        Menulis ke buffer pre-alokasi untuk menghindari alokasi per-frame.

        Returns:
            tensor   : CHW float32 [0-1], view ke buffer internal
            r        : skala resize (dipakai untuk scale-back)
            pad_left : padding horizontal dalam piksel
            pad_top  : padding vertikal dalam piksel
        """
        h, w = img.shape[:2]
        r        = min(self._inp_h / h, self._inp_w / w)
        new_w    = int(round(w * r))
        new_h    = int(round(h * r))
        pad_left = int(round((self._inp_w - new_w) / 2 - 0.1))
        pad_top  = int(round((self._inp_h - new_h) / 2 - 0.1))

        self._canvas[:] = 114
        cv2.resize(img, (new_w, new_h),
                   dst=self._canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w])
        rgb = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2RGB)
        np.divide(rgb.transpose(2, 0, 1), 255.0, out=self._tensor)
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

        if self._inp_h is None:
            self._init_buffers(orig_h, orig_w)

        with measure_or_null(profiler, "preprocess"):
            tensor, r, pad_left, pad_top = self._letterbox(img)

        with measure_or_null(profiler, "infer"):
            # IO binding: zero-copy input, output tanpa alokasi baru per-frame
            self._io_binding.bind_cpu_input(self.input_name, tensor[None])
            self._io_binding.bind_output(self._output_name)
            self.session.run_with_iobinding(self._io_binding)
            # .numpy() mengembalikan view ke buffer ORT — tidak ada copy
            raw = self._io_binding.get_outputs()[0].numpy()[0]

        with measure_or_null(profiler, "postprocess"):
            mask = raw[:, 4] >= conf_thresh
            dets = raw[mask]  # (N, 6)

            if len(dets) == 0:
                return SimpleDetections(
                    xyxy=np.empty((0, 4), dtype=np.float32),
                    conf=np.empty((0,),   dtype=np.float32),
                    cls= np.empty((0,),   dtype=np.float32),
                )

            # Scale-back dari letterbox ke koordinat asli — satu operasi broadcast
            inv_r  = np.float32(1.0 / r)
            offset = np.array([pad_left, pad_top, pad_left, pad_top], dtype=np.float32)
            clamp  = np.array([orig_w,   orig_h,  orig_w,   orig_h],  dtype=np.float32)
            boxes  = np.clip((dets[:, :4] - offset) * inv_r, 0.0, clamp)

            result = SimpleDetections(
                xyxy=boxes,
                conf=dets[:, 4].astype(np.float32),
                cls= dets[:, 5].astype(np.float32),
            )

        return result
