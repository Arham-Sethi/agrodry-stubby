"""
QFM-CNN inference wrapper (moisture-only API).
- Supports "stub" (default), "torch" (TorchScript), and "onnx".
- Returns only: moisture + model info + latency_ms. No crop. No particle size.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

_TORCH_OK = False
try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

_ONNX_OK = False
try:
    import onnxruntime as ort
    _ONNX_OK = True
except Exception:
    _ONNX_OK = False


@dataclass
class QFMConfig:
    backend: str = os.getenv("QFM_BACKEND", "stub")     # "stub" | "torch" | "onnx"
    version: str = os.getenv("QFM_VERSION", "0.0.1")
    device: str  = os.getenv("QFM_DEVICE", "cpu")
    model_path: str = os.getenv("QFM_MODEL_PATH", "")
    input_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float]  = (0.229, 0.224, 0.225)


def _preprocess_rgb(img: np.ndarray, cfg: QFMConfig) -> np.ndarray:
    # img: HxWx3 RGB uint8 -> 1x3xHxW float32 normalized
    h, w = cfg.input_size
    if cv2 is not None:
        arr = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    else:
        from PIL import Image
        pil = Image.fromarray(img)
        pil = pil.resize((w, h), Image.BILINEAR)
        arr = np.asarray(pil).astype(np.float32) / 255.0

    mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(cfg.std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))[None, ...]
    return chw.astype(np.float32)


class QFMModel:
    def __init__(self):
        self.cfg = QFMConfig()
        self._torch_model = None
        self._onnx_sess = None

        if self.cfg.backend == "torch" and self.cfg.model_path and _TORCH_OK:
            self._load_torch()
        elif self.cfg.backend == "onnx" and self.cfg.model_path and _ONNX_OK:
            self._load_onnx()
        else:
            # stub
            pass

    def _load_torch(self):
        map_location = "cuda" if (self.cfg.device.startswith("cuda") and _TORCH_OK and torch.cuda.is_available()) else "cpu"
        self.cfg.device = map_location
        self._torch_model = torch.jit.load(self.cfg.model_path, map_location=map_location)
        self._torch_model.eval()
        if map_location == "cpu":
            try:
                torch.set_num_threads(max(1, os.cpu_count() // 2))
            except Exception:
                pass

    def _load_onnx(self):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.cfg.device.startswith("cuda") else ["CPUExecutionProvider"]
        self._onnx_sess = ort.InferenceSession(self.cfg.model_path, providers=providers)

    def predict(self, rgb_image: np.ndarray) -> dict:
        """
        Returns:
          {
            "moisture": {"value_pct": float, "confidence": float},
            "model": {"version": str, "backend": str},
            "latency_ms": int
          }
        """
        t0 = time.time()
        if self.cfg.backend == "torch" and self._torch_model is not None:
            out = self._predict_torch(rgb_image)
        elif self.cfg.backend == "onnx" and self._onnx_sess is not None:
            out = self._predict_onnx(rgb_image)
        else:
            out = self._predict_stub(rgb_image)
        out["model"] = {"version": self.cfg.version, "backend": self.cfg.backend}
        out["latency_ms"] = int((time.time() - t0) * 1000)
        return out

    def _predict_torch(self, rgb_image: np.ndarray) -> dict:
        x = _preprocess_rgb(rgb_image, self.cfg)
        with torch.no_grad():
            xt = torch.from_numpy(x)
            if self.cfg.device.startswith("cuda"):
                xt = xt.cuda(non_blocking=True)
            y = self._torch_model(xt)
            # Accept either tensor or dict {"moisture": tensor}
            if isinstance(y, dict):
                m = y.get("moisture")
            else:
                m = y
            if isinstance(m, torch.Tensor):
                m = m.detach().float().cpu().numpy()
        val = float(np.array(m).reshape(-1)[0])
        return self._post_moisture(val)

    def _predict_onnx(self, rgb_image: np.ndarray) -> dict:
        x = _preprocess_rgb(rgb_image, self.cfg).astype(np.float32)
        inputs = self._onnx_sess.get_inputs()
        in_name = inputs[0].name
        outs = self._onnx_sess.run(None, {in_name: x})
        # Assume first output is moisture scalar
        val = float(np.array(outs[0]).reshape(-1)[0])
        return self._post_moisture(val)

    def _post_moisture(self, raw_value: float) -> dict:
        # If your model outputs 0..1, convert to percent; if already 0..100 keep it.
        value_pct = float(np.clip(raw_value * 100.0, 0.0, 100.0)) if 0.0 <= raw_value <= 1.2 else float(np.clip(raw_value, 0.0, 100.0))
        return {"moisture": {"value_pct": value_pct, "confidence": 0.90}}

    def _predict_stub(self, rgb_image: np.ndarray) -> dict:
        base = 24 + (np.random.rand() * 6 - 3)
        value_pct = float(np.clip(base, 2, 60))
        return {"moisture": {"value_pct": value_pct, "confidence": 0.90}}


# Singleton
qfm_model: Optional[QFMModel] = None

def get_model() -> QFMModel:
    global qfm_model
    if qfm_model is None:
        qfm_model = QFMModel()
    return qfm_model
