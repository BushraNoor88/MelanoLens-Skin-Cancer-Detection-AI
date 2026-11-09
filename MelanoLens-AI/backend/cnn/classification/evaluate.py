"""Evaluate the classifier on the HELD-OUT test split (pure NumPy metrics, so they are easy to verify)."""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from config import Config

from .. import utils
from ..errors import DatasetError
from . import dataset
from .predict import ClassificationPredictor


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm                                                   # rows = true class, columns = predicted class


def _div(a, b):
    return None if b == 0 else a / b


def roc_auc_binary(y_true_pos, score):
    """AUC via the rank-sum (Mann-Whitney) formula; ties get average ranks. None if one class is absent."""
    y, s = np.asarray(y_true_pos).astype(bool), np.asarray(score, float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve_points(y_true_pos, score, max_points=60):
    y, s = np.asarray(y_true_pos).astype(bool), np.asarray(score, float)
    if y.all() or (~y).all():
        return None
    thr = np.unique(s)[::-1]
    tpr = [0.0] + [float(((s >= t) & y).sum() / y.sum()) for t in thr]
    fpr = [0.0] + [float(((s >= t) & ~y).sum() / (~y).sum()) for t in thr]
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = [fpr[i] for i in keep], [tpr[i] for i in keep]
    return {"fpr": [round(x, 4) for x in fpr], "tpr": [round(x, 4) for x in tpr]}


pick_positive_class = dataset.pick_positive_class      # re-exported (used by tests and the tuner)


def predict_labels(proba, threshold=None, positive=None):
    """argmax, or (binary only) 'positive' when P(positive) >= threshold."""
    proba = np.asarray(proba, float)
    if threshold is None or proba.shape[1] != 2:
        return proba.argmax(1)
    return np.where(proba[:, positive] >= threshold, positive, 1 - positive)


def classification_metrics(y_true, proba, class_names, threshold=None):
    """threshold: optional decision threshold for the positive class (binary only); default = plain argmax."""
    y_true, proba = np.asarray(y_true), np.asarray(proba, float)
    n = len(class_names)
    pos_idx, _ = dataset.pick_positive_class(class_names)
    y_pred = predict_labels(proba, threshold, pos_idx)
    cm = confusion_matrix(y_true, y_pred, n)
    per_class = []
    for c in range(n):
        tp, fp, fn = int(cm[c, c]), int(cm[:, c].sum() - cm[c, c]), int(cm[c].sum() - cm[c, c])
        support = int(cm[c].sum())
        # precision is undefined when the class is never predicted; F1 = 2TP / (2TP + FP + FN)
        per_class.append({"class": class_names[c], "precision": _div(tp, tp + fp), "recall": _div(tp, tp + fn),
                          "f1": _div(2 * tp, 2 * tp + fp + fn), "support": support})
    r4 = lambda v: None if v is None else round(float(v), 4)

    def macro(k):    # average over classes that exist in the test set; an undefined value counts as 0 (never predicted)
        vals = [(d[k] or 0.0) for d in per_class if d["support"] > 0]
        return r4(np.mean(vals)) if vals else None
    macros = {k: macro(k) for k in ("precision", "recall", "f1")}
    for d in per_class:
        for k in ("precision", "recall", "f1"):
            d[k] = r4(d[k])
    out = {"n_test": int(len(y_true)), "accuracy": r4((y_pred == y_true).mean()),
           "macro_precision": macros["precision"], "macro_recall": macros["recall"], "macro_f1": macros["f1"],
           "macro_note": "average over classes present in the test set; undefined precision (class never predicted) counts as 0",
           "decision_rule": (f"predict {class_names[pos_idx]!r} when P >= {threshold:.3f}" if threshold is not None and n == 2 else "argmax (highest probability)"),
           "per_class": per_class, "confusion_matrix": {"labels": list(class_names), "matrix": cm.tolist(),
                                                         "note": "rows = true class, columns = predicted class"}}
    if n == 2:
        pos, why = pick_positive_class(class_names)
        out["roc_auc"] = r4(roc_auc_binary(y_true == pos, proba[:, pos]))
        out["roc_auc_note"] = f"binary, positive class = {class_names[pos]!r} ({why})"
        out["roc_curve"] = roc_curve_points(y_true == pos, proba[:, pos])
    else:
        aucs = [roc_auc_binary(y_true == c, proba[:, c]) for c in range(n)]
        valid = [a for a in aucs if a is not None]
        out["roc_auc"] = r4(np.mean(valid)) if valid else None
        out["roc_auc_note"] = "macro-average one-vs-rest over classes present in the test set"
        out["roc_curve"] = None
    return out


def evaluate_model(model_path, images_dir, csv_path, label_column=None, out_path=None):
    model_path = Path(model_path)
    split = utils.read_json(utils.sidecar(model_path, "split"))
    if not split:
        raise DatasetError("No split file next to the model. Train with cnn.classification.train first.")
    utils.check_no_leakage({k: split[k] for k in ("train", "val", "test")})
    data = dataset.load_labeled_samples(images_dir, csv_path, label_column)
    predictor = ClassificationPredictor.load(model_path)
    if data["class_names"] != predictor.class_names:
        raise DatasetError("The CSV classes differ from the classes the model was trained on.")
    index = {i: k for k, i in enumerate(data["ids"])}
    test = [index[i] for i in split["test"] if i in index]
    if not test:
        raise DatasetError("None of the test-split images were found on disk.")
    x = dataset.load_arrays([data["paths"][i] for i in test], predictor.img_size).astype(np.float32)
    proba = np.asarray(predictor.model.predict(x, verbose=0, batch_size=32))
    meta = utils.read_json(utils.sidecar(model_path, "meta"), {}) or {}
    dt = meta.get("decision_threshold")
    y_test, names = data["labels"][test], data["class_names"]
    m = classification_metrics(y_test, proba, names, dt["value"] if dt else None)
    extra = {}
    if dt:        # always also report the untuned (argmax) result so the effect of the threshold is visible
        base = classification_metrics(y_test, proba, names)
        extra = {"decision_threshold": {k: dt[k] for k in ("positive_class", "value", "criterion", "target_recall", "tuned_on") if k in dt},
                 "at_default_threshold": {k: base[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class", "confusion_matrix", "decision_rule")}}
    result = {"task": "classification", "model": "MobileNetV2 (transfer learning)", "evaluated_on": "held-out test split",
              "evaluated_at": utils.now_iso(), "class_names": data["class_names"],
              "test_class_counts": {data["class_names"][c]: int((data["labels"][test] == c).sum()) for c in range(len(data["class_names"]))},
              "split_sizes": {k: len(v) for k, v in split.items() if k in ("train", "val", "test")},
              **m, **extra, "note": "Computed on images never used for training or model selection. "
              + ("The decision threshold was chosen on the validation split only." if dt else "")}
    utils.write_json(out_path or utils.sidecar(model_path, "eval"), result)
    return result


def print_summary(r):
    if r.get("decision_threshold"):
        d = r["decision_threshold"]; b = r["at_default_threshold"]
        mal = lambda x: next(c for c in x["per_class"] if c["class"] == d["positive_class"])
        print(f"Decision threshold for {d['positive_class']!r}: {d['value']} ({d['criterion']}; tuned on validation only)")
        print(f"  at default argmax:  accuracy={b['accuracy']}  {d['positive_class']} recall={mal(b)['recall']} precision={mal(b)['precision']}")
        print(f"  at tuned threshold: accuracy={r['accuracy']}  {d['positive_class']} recall={mal(r)['recall']} precision={mal(r)['precision']}")
    print(f"Test images: {r['n_test']}   accuracy={r['accuracy']}   macro-F1={r['macro_f1']}   ROC-AUC={r['roc_auc']}")
    for d in r["per_class"]:
        print(f"  {d['class']:>12}: precision={d['precision']} recall={d['recall']} f1={d['f1']} (n={d['support']})")
    print("  confusion matrix (rows=true, cols=pred):", r["confusion_matrix"]["matrix"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the classifier on its held-out test split.")
    ap.add_argument("--model-path", default=str(Config.CLS_MODEL_PATH))
    ap.add_argument("--images-dir", default=str(Config.ISIC_IMAGES_DIR))
    ap.add_argument("--labels-csv", default=str(Config.ISIC_LABELS_CSV) if Config.ISIC_LABELS_CSV else None)
    ap.add_argument("--label-column", default=None)
    a = ap.parse_args(argv)
    csv_path = dataset.find_labels_csv([Path(a.images_dir).parent, Config.ISIC_DIR], a.labels_csv)
    print_summary(evaluate_model(a.model_path, a.images_dir, csv_path, a.label_column))
    return 0


if __name__ == "__main__":
    sys.exit(main())
