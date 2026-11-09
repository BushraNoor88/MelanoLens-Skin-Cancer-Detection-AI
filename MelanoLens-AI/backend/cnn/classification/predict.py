"""Inference with the trained classifier. Class names come from the file written at training time."""
import threading
from pathlib import Path

import cv2
import numpy as np

from .. import utils
from ..errors import ModelNotAvailable
from . import dataset


class ClassificationPredictor:
    def __init__(self, model, info, model_path):
        self.model, self.info, self.model_path = model, info, Path(model_path)
        self.class_names = list(info["class_names"])
        self.img_size = int(model.input_shape[1])
        self.threshold = None      # optional decision threshold for the positive class (binary models only)
        dt = info.get("decision_threshold")
        if dt and len(self.class_names) == 2 and dt.get("positive_class") in self.class_names:
            self.threshold = dt
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotAvailable("No trained classification model was found. Train it first: "
                                    "cd backend && python -m cnn.classification.train", "model_missing")
        info = utils.read_json(utils.sidecar(model_path, "meta"))
        if not info or not info.get("class_names"):
            raise ModelNotAvailable("The classifier's class-name file is missing, so predictions cannot be labelled "
                                    "safely. Re-train the model.", "model_invalid")
        _, keras = utils.require_tf()
        try:
            model = keras.models.load_model(model_path, compile=False)
        except Exception as exc:
            raise ModelNotAvailable(f"The classification model file could not be loaded ({type(exc).__name__}). "
                                    "Re-train it or restore a valid file.", "model_invalid")
        if model.output_shape[-1] != len(info["class_names"]):
            raise ModelNotAvailable("Model output size does not match its class list. Re-train the model.", "model_invalid")
        return cls(model, info, model_path)

    def predict(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = utils.resize_image(rgb, self.img_size).astype(np.float32)[None]      # raw 0-255, as in training
        with self._lock:
            p = np.asarray(self.model(x, training=False))[0].astype(float)
        order = np.argsort(-p)
        top = int(order[0])
        decision = {"rule": "argmax (highest probability)"}
        if self.threshold:
            pos = self.class_names.index(self.threshold["positive_class"])
            top = pos if p[pos] >= self.threshold["value"] else 1 - pos
            decision = {"rule": f"predict {self.threshold['positive_class']!r} when its probability >= {self.threshold['value']}",
                        "positive_class": self.threshold["positive_class"], "threshold": self.threshold["value"],
                        "positive_probability": round(float(p[pos]), 4), "criterion": self.threshold.get("criterion")}
        return {"predicted_class": self.class_names[top], "confidence": round(float(p[top]), 4), "decision": decision,
                "probabilities": {self.class_names[int(i)]: round(float(p[i]), 4) for i in order},
                "disclaimer": utils.DISCLAIMER}

    def describe(self):
        i = self.info
        return {"name": "MobileNetV2 (transfer learning)", "architecture": f"MobileNetV2 (alpha={i.get('alpha')}) + dense softmax head",
                "input_size": self.img_size, "classes": self.class_names, "pretrained_weights": i.get("weights"),
                "trained_at": i.get("trained_at"), "n_train": (i.get("split_sizes") or {}).get("train"),
                "label_source": i.get("label_column"),
                "decision_threshold": (self.threshold or {}).get("value")}
