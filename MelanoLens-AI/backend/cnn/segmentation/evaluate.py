"""Evaluate the trained U-Net on the HELD-OUT test split, at ORIGINAL image resolution."""
import argparse
import sys
from pathlib import Path

import numpy as np

from config import Config
from services.metrics import evaluate_masks

from .. import utils
from ..errors import DatasetError
from .predict import SegmentationPredictor

KEYS = ("dice", "iou", "precision", "recall", "pixel_accuracy")


def _summary(rows, key):
    v = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
    return None if v.size == 0 else {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                                     "median": round(float(np.median(v)), 4), "n": int(v.size)}


def evaluate_model(model_path, images_dir, masks_dir, split_path=None, out_path=None):
    model_path = Path(model_path)
    split = utils.read_json(split_path or utils.sidecar(model_path, "split"))
    if not split:
        raise DatasetError("No split file next to the model. Train with cnn.segmentation.train first, so the "
                           "same held-out test images can be used.")
    utils.check_no_leakage({k: split[k] for k in ("train", "val", "test")})
    pairs, _ = utils.pair_images_masks(images_dir, masks_dir)
    by_id = {p[0]: p for p in pairs}
    test_ids = [i for i in split["test"] if i in by_id]
    if not test_ids:
        raise DatasetError("None of the test-split images were found on disk.")
    predictor = SegmentationPredictor.load(model_path)
    rows = []
    for i in test_ids:
        _id, ip, mp = by_id[i]
        rgb = utils.read_rgb(ip)
        ref = utils.read_mask(mp) >= 128
        pred = predictor.predict(rgb[..., ::-1])["mask"]
        m = evaluate_masks(pred.astype(np.uint8) * 255, ref.astype(np.uint8) * 255, allow_resize=True)
        total = m["tp"] + m["fp"] + m["fn"] + m["tn"]
        rows.append({"id": _id, "dice": round(m["dice"], 4), "iou": round(m["iou"], 4),
                     "precision": None if m["precision"] is None else round(m["precision"], 4),
                     "recall": None if m["recall"] is None else round(m["recall"], 4),
                     "pixel_accuracy": round((m["tp"] + m["tn"]) / total, 4)})
    result = {"task": "segmentation", "model": "U-Net", "evaluated_on": "held-out test split",
              "n_test": len(rows), "split_sizes": {k: len(v) for k, v in split.items() if k in ("train", "val", "test")},
              "evaluated_at": utils.now_iso(), "metrics": {k: _summary(rows, k) for k in KEYS},
              "per_image": rows,
              "note": "Metrics are computed at original image resolution on images never used for training or "
                      "model selection."}
    utils.write_json(out_path or utils.sidecar(model_path, "eval"), result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the segmentation model on its held-out test split.")
    ap.add_argument("--model-path", default=str(Config.SEG_MODEL_PATH))
    ap.add_argument("--images-dir", default=str(Config.ISIC_IMAGES_DIR))
    ap.add_argument("--masks-dir", default=str(Config.ISIC_MASKS_DIR))
    a = ap.parse_args(argv)
    r = evaluate_model(a.model_path, a.images_dir, a.masks_dir)
    print(f"Test images: {r['n_test']}")
    for k, v in r["metrics"].items():
        print(f"  {k:15s} mean={v['mean']:.4f}  std={v['std']:.4f}" if v else f"  {k}: n/a")
    return 0


if __name__ == "__main__":
    sys.exit(main())
