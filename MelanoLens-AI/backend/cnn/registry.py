"""Loads each trained model once and keeps it in memory (reloaded only if the file changes)."""
import threading

from . import utils
from .classification.predict import ClassificationPredictor
from .errors import ModelNotAvailable
from .segmentation.predict import SegmentationPredictor

_cache, _lock = {}, threading.Lock()


def _get(kind, cls, path):
    """Cache key = model file + its metadata file, so re-tuning the threshold is picked up without a restart."""
    import os
    path = str(path)
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        raise ModelNotAvailable(f"No trained {kind} model was found. See the README section 'CNN modules' for the training command.", "model_missing")
    try:
        meta_mtime = os.stat(utils.sidecar(path, "meta")).st_mtime_ns
    except OSError:
        meta_mtime = 0
    key = (kind, path, mtime, meta_mtime)
    with _lock:
        for k in [k for k in _cache if k[:2] == (kind, path) and k != key]:
            _cache.pop(k)                                  # file or metadata changed -> drop the old model
        if key not in _cache:
            _cache[key] = cls.load(path)
        return _cache[key]


def get_segmenter(path):
    return _get("segmentation", SegmentationPredictor, path)


def get_classifier(path):
    return _get("classification", ClassificationPredictor, path)


def model_status(kind, model_path):
    """Cheap status (does NOT import TensorFlow or load the model). Never exposes file paths."""
    from pathlib import Path
    p = Path(model_path)
    meta = utils.read_json(utils.sidecar(p, "meta"), {}) or {}
    ev = utils.read_json(utils.sidecar(p, "eval"))
    tf_ok = utils.tensorflow_installed()
    if not tf_ok:
        reason, available = "TensorFlow is not installed (pip install -r backend/requirements.txt).", False
    elif not p.is_file():
        reason, available = "Model not trained yet.", False
    elif kind == "classification" and not meta.get("class_names"):
        reason, available = "Class-name file missing; re-train the model.", False
    else:
        reason, available = None, True
    return {"available": available, "reason": reason, "tensorflow_installed": tf_ok, "model_file": p.name,
            "trained_at": meta.get("trained_at"), "input_size": meta.get("img_size"),
            "classes": meta.get("class_names"), "evaluated": ev is not None}
