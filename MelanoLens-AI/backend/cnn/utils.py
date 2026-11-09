"""Helpers shared by both CNN modules: file pairing, splits, JSON sidecars, mask clean-up, overlays."""
import base64
import importlib.util
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

from services.metrics import mask_boundary

from .errors import DatasetError, ModelNotAvailable

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_MASK_SUFFIX = re.compile(r"(_segmentation|_mask|_gt|_seg)$", re.IGNORECASE)
DISCLAIMER = ("Model output only. It is NOT a medical diagnosis. MelanoLens AI is an educational project; "
              "consult a qualified clinician for any skin concern.")


# ---------------------------------------------------------------- TensorFlow (lazy) ----
def tensorflow_installed() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def require_tf():
    """Import and return (tensorflow, keras). Raises ModelNotAvailable if TensorFlow is missing."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if not tensorflow_installed():
        raise ModelNotAvailable("TensorFlow is not installed. Run: pip install -r backend/requirements.txt",
                                "tensorflow_missing")
    import tensorflow as tf
    import keras
    return tf, keras


# ---------------------------------------------------------------- JSON sidecars ----
def sidecar(model_path, kind: str) -> Path:
    """models/unet_skin_lesion.keras -> models/unet_skin_lesion_<kind>.json"""
    p = Path(model_path)
    return p.with_name(f"{p.stem}_{kind}.json")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- files / pairing ----
def list_image_files(directory):
    """Image files in the folder AND its subfolders (ISIC downloads extract into a nested folder)."""
    d = Path(directory)
    if not d.is_dir():
        raise DatasetError(f"Folder not found: {d.name!r}. Check the dataset path (see README).")
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def normalize_id(name: str) -> str:
    """ISIC_0000000.jpg / ISIC_0000000_Segmentation.png / ISIC_0000000 -> 'ISIC_0000000'"""
    stem = Path(str(name).strip()).stem if Path(str(name).strip()).suffix.lower() in IMAGE_EXTS else str(name).strip()
    return _MASK_SUFFIX.sub("", stem)


def pair_images_masks(images_dir, masks_dir):
    """Match images to masks by ID (never by sort order). Returns (pairs, report)."""
    images = {}
    for p in list_image_files(images_dir):
        images.setdefault(normalize_id(p.name), p)
    masks = {}
    duplicate_masks = []
    for p in list_image_files(masks_dir):
        key = normalize_id(p.name)
        if key in masks:
            duplicate_masks.append(key)
        masks[key] = p
    common = sorted(set(images) & set(masks))
    report = {"n_images": len(images), "n_masks": len(masks), "n_pairs": len(common),
              "images_without_mask": sorted(set(images) - set(masks))[:10],
              "n_images_without_mask": len(set(images) - set(masks)),
              "masks_without_image": sorted(set(masks) - set(images))[:10],
              "n_masks_without_image": len(set(masks) - set(images)),
              "duplicate_mask_ids": duplicate_masks[:10],
              "example_image_name": next(iter(images.values())).name if images else None,
              "example_mask_name": next(iter(masks.values())).name if masks else None}
    if not common:
        raise DatasetError("No image/mask pairs were found. Image IDs and mask IDs must match "
                           "(e.g. ISIC_0000000.jpg and ISIC_0000000_Segmentation.png).")
    return [(k, images[k], masks[k]) for k in common], report


# ---------------------------------------------------------------- images ----
def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise DatasetError(f"Could not read image file {Path(path).name!r}.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise DatasetError(f"Could not read mask file {Path(path).name!r}.")
    return m


def resize_image(rgb, size):
    """Images: area interpolation (good for shrinking)."""
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask, size):
    """Masks: nearest neighbour so values stay 0/1 (no blended edge values)."""
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------- splitting ----
def split_indices(n, labels=None, val_frac=0.15, test_frac=0.15, seed=42):
    """Disjoint train/val/test index lists. Stratified by label when labels are given. Deterministic."""
    if n < 3:
        raise DatasetError("Need at least 3 samples to make train/validation/test splits.")
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 0.9:
        raise ValueError("val_frac + test_frac must be below 0.9.")
    rng = np.random.default_rng(seed)
    groups = [np.arange(n)] if labels is None else [np.flatnonzero(np.asarray(labels) == c) for c in sorted(set(labels))]
    train, val, test = [], [], []
    for g in groups:
        g = rng.permutation(g)
        n_test, n_val = int(round(len(g) * test_frac)), int(round(len(g) * val_frac))
        if len(g) >= 3:                                   # keep every sizeable class in every split
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        n_test, n_val = min(n_test, len(g) - 1), min(n_val, max(len(g) - 1 - n_test, 0))
        test += g[:n_test].tolist(); val += g[n_test:n_test + n_val].tolist(); train += g[n_test + n_val:].tolist()
    out = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    assert not (set(out["train"]) & set(out["val"]) or set(out["train"]) & set(out["test"]) or set(out["val"]) & set(out["test"]))
    return out


def split_ids(ids, indices):
    return {k: [ids[i] for i in v] for k, v in indices.items()}


def check_no_leakage(split):
    """Raise if any ID appears in more than one split."""
    seen = {}
    for name, items in split.items():
        for i in items:
            if i in seen and seen[i] != name:
                raise DatasetError(f"Data leakage: {i} is in both {seen[i]} and {name}.")
            seen[i] = name


# ---------------------------------------------------------------- masks and drawings ----
def clean_mask(mask_bool, keep_largest=True, fill_holes=True):
    """Post-process a predicted lesion mask: keep the largest blob and fill holes (single-lesion images)."""
    m = mask_bool.astype(np.uint8)
    if keep_largest and m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 2:
            m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    if fill_holes and m.any():
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(m)
        cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
        m = filled
    return m.astype(bool)


def draw_boundary(bgr, mask_bool, color=(255, 0, 200), thickness=2):
    out = bgr.copy()
    b = mask_boundary(mask_bool).astype(np.uint8)
    if thickness > 1:
        b = cv2.dilate(b, np.ones((thickness, thickness), np.uint8))
    out[b > 0] = color
    return out


def tint_region(bgr, mask_bool, color=(166, 234, 94), alpha=0.45, dim_background=0.55):
    """Lesion-region visualisation: tinted lesion, dimmed background."""
    out = bgr.astype(np.float32)
    bg = ~mask_bool
    out[bg] *= dim_background
    out[mask_bool] = out[mask_bool] * (1 - alpha) + np.array(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def heatmap(prob):
    return cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def mask_stats(mask_bool):
    area = int(mask_bool.sum())
    perim = 0.0
    if area:
        cnts, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = float(sum(cv2.arcLength(c, True) for c in cnts))
    return {"lesion_area_px": area, "lesion_area_fraction": round(area / mask_bool.size, 4),
            "perimeter_px": round(perim, 1)}


def png_data_uri(arr):
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("Could not encode PNG")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")