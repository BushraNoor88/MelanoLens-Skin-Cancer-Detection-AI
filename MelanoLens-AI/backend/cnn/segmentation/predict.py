"""Inference with a trained U-Net. The model is loaded from disk (never trained here)."""
import threading
from pathlib import Path

import cv2
import numpy as np

from .. import utils
from ..errors import ModelNotAvailable
from . import dataset, model as seg_model

THRESHOLD = 0.5


class SegmentationPredictor:
    def __init__(self, model, info, model_path):
        self.model, self.info, self.model_path = model, info, Path(model_path)
        self.img_size = int(model.input_shape[1])
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotAvailable("No trained segmentation model was found. Train it first: "
                                    "cd backend && python -m cnn.segmentation.train", "model_missing")
        _, keras = utils.require_tf()
        try:
            model = keras.models.load_model(model_path, custom_objects=seg_model.custom_objects(), compile=False)
        except Exception as exc:                                  # corrupt / incompatible file
            raise ModelNotAvailable(f"The segmentation model file could not be loaded ({type(exc).__name__}). "
                                    "Re-train it or restore a valid file.", "model_invalid")
        info = utils.read_json(utils.sidecar(model_path, "meta"), {}) or {}
        return cls(model, info, model_path)

    def predict(self, bgr, postprocess=True):
        """bgr uint8 image -> dict(probability float32 HxW in [0,1], mask bool HxW) at the ORIGINAL size."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = dataset.preprocess_image(rgb, self.img_size)[None]
        with self._lock:
            prob = np.asarray(self.model(x, training=False))[0, ..., 0]
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)       # smooth upsampling of probabilities
        mask = prob >= THRESHOLD
        if postprocess:
            mask = utils.clean_mask(mask)
        return {"probability": prob.astype(np.float32), "mask": mask}

    def describe(self):
        return {"name": "U-Net", "architecture": "U-Net encoder-decoder (skip connections), sigmoid output",
                "input_size": self.img_size, "threshold": THRESHOLD,
                "postprocessing": "largest connected component + hole filling",
                "trained_at": self.info.get("trained_at"), "n_train": (self.info.get("split_sizes") or {}).get("train")}
