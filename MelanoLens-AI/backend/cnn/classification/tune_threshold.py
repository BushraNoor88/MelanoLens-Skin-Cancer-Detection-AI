"""Choose the decision threshold for the positive (malignant) class on the VALIDATION split, then evaluate once on TEST.

    cd backend
    python -m cnn.classification.tune_threshold --target-recall 0.80

Why: with a 0.5 cut-off the model misses many malignant cases when they are rare in the training data.
A lower threshold catches more of them but also flags more benign lesions (lower precision). That trade-off is
shown in full; the test split is NOT used to pick the threshold.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from config import Config

from .. import utils
from ..errors import DatasetError
from . import dataset, evaluate as cls_eval
from .predict import ClassificationPredictor

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def sweep(y_pos, p_pos, grid=GRID):
    """Metrics of the positive class for each candidate threshold."""
    y, p = np.asarray(y_pos).astype(bool), np.asarray(p_pos, float)
    rows = []
    for t in grid:
        pred = p >= t
        tp, fp, fn, tn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum())
        rec = tp / (tp + fn) if tp + fn else None
        prec = tp / (tp + fp) if tp + fp else None
        spec = tn / (tn + fp) if tn + fp else None
        f2 = 5 * tp / (5 * tp + 4 * fn + fp) if tp else 0.0
        rows.append({"threshold": float(t), "recall": rec, "precision": prec, "specificity": spec, "f2": round(f2, 4),
                     "flagged": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return rows


def choose(rows, criterion="recall", target_recall=0.8):
    """'recall': the HIGHEST threshold whose validation recall >= target (fewest false alarms for that recall).
    'f2': the threshold with the best F2 (recall counts twice as much as precision)."""
    if criterion == "f2":
        best = max(rows, key=lambda r: (r["f2"], r["threshold"]))
        return best, None
    ok = [r for r in rows if r["recall"] is not None and r["recall"] >= target_recall]
    if ok:
        return max(ok, key=lambda r: r["threshold"]), None
    best = max(rows, key=lambda r: ((r["recall"] or 0), r["threshold"]))
    return best, f"No threshold reached validation recall {target_recall}; using the best available ({best['recall']})."


def tune(model_path, images_dir, csv_path, label_column=None, criterion="recall", target_recall=0.8, min_positives=5):
    model_path = Path(model_path)
    split = utils.read_json(utils.sidecar(model_path, "split"))
    if not split:
        raise DatasetError("No split file next to the model; train first.")
    data = dataset.load_labeled_samples(images_dir, csv_path, label_column)
    pred = ClassificationPredictor.load(model_path)
    if len(pred.class_names) != 2:
        raise DatasetError("Threshold tuning is only defined for binary classification.")
    if data["class_names"] != pred.class_names:
        raise DatasetError("The CSV classes differ from the classes the model was trained on.")
    index = {i: k for k, i in enumerate(data["ids"])}
    val = [index[i] for i in split["val"] if i in index]
    pos, why = dataset.pick_positive_class(pred.class_names)
    x = dataset.load_arrays([data["paths"][i] for i in val], pred.img_size).astype(np.float32)
    p_pos = np.asarray(pred.model.predict(x, verbose=0, batch_size=32))[:, pos]
    y_pos = data["labels"][val] == pos
    if y_pos.sum() < min_positives:
        raise DatasetError(f"Only {int(y_pos.sum())} positive cases in the validation split; too few to tune a threshold.")
    rows = sweep(y_pos, p_pos)
    best, warning = choose(rows, criterion, target_recall)
    at_default = min(rows, key=lambda r: abs(r["threshold"] - 0.5))
    meta = utils.read_json(utils.sidecar(model_path, "meta"), {}) or {}
    meta["decision_threshold"] = {
        "positive_class": pred.class_names[pos], "positive_class_index": pos, "positive_class_note": why,
        "value": best["threshold"], "criterion": criterion, "target_recall": target_recall if criterion == "recall" else None,
        "tuned_on": f"validation split ({len(val)} images, {int(y_pos.sum())} {pred.class_names[pos]})",
        "validation_at_threshold": {k: best[k] for k in ("recall", "precision", "specificity", "flagged")},
        "validation_at_0.5": {k: at_default[k] for k in ("recall", "precision", "specificity", "flagged")},
        "warning": warning, "tuned_at": utils.now_iso()}
    utils.write_json(utils.sidecar(model_path, "meta"), meta)
    utils.write_json(utils.sidecar(model_path, "threshold"), {"positive_class": pred.class_names[pos], "n_validation": len(val),
                     "n_positive": int(y_pos.sum()), "sweep": rows[::5], "chosen": best})
    return meta["decision_threshold"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tune the malignant decision threshold on the validation split.")
    ap.add_argument("--model-path", default=str(Config.CLS_MODEL_PATH))
    ap.add_argument("--images-dir", default=str(Config.ISIC_IMAGES_DIR))
    ap.add_argument("--labels-csv", default=str(Config.ISIC_LABELS_CSV) if Config.ISIC_LABELS_CSV else None)
    ap.add_argument("--label-column", default=None)
    ap.add_argument("--criterion", choices=["recall", "f2"], default="recall")
    ap.add_argument("--target-recall", type=float, default=0.80, help="used with --criterion recall")
    ap.add_argument("--reset", action="store_true", help="remove the tuned threshold (back to plain 0.5 / argmax)")
    a = ap.parse_args(argv)
    csv_path = dataset.find_labels_csv([Path(a.images_dir).parent, Config.ISIC_DIR], a.labels_csv)
    if a.reset:
        meta = utils.read_json(utils.sidecar(a.model_path, "meta"), {}) or {}
        meta.pop("decision_threshold", None); utils.write_json(utils.sidecar(a.model_path, "meta"), meta)
        print("Threshold removed.")
    else:
        t = tune(a.model_path, a.images_dir, csv_path, a.label_column, a.criterion, a.target_recall)
        print(f"Chosen threshold for {t['positive_class']!r}: {t['value']}  ({t['tuned_on']})")
        print(f"  validation at 0.5:       {t['validation_at_0.5']}")
        print(f"  validation at threshold: {t['validation_at_threshold']}")
        if t["warning"]:
            print("  WARNING:", t["warning"])
    print("\nEvaluating ONCE on the held-out test split...")
    cls_eval.print_summary(cls_eval.evaluate_model(a.model_path, a.images_dir, csv_path, a.label_column))
    return 0


if __name__ == "__main__":
    sys.exit(main())
