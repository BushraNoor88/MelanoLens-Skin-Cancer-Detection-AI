"""Classification metrics checked against values worked out by hand."""
import numpy as np
import pytest

from cnn.classification.evaluate import classification_metrics, confusion_matrix, roc_auc_binary


def test_confusion_matrix_orientation():
    cm = confusion_matrix(np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 0]), 2)
    assert cm.tolist() == [[1, 1], [1, 2]]                       # rows = true, cols = predicted


def test_binary_metrics_by_hand():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p1 = np.array([.1, .2, .6, .3, .9, .8, .4, .7])               # predictions: 0,0,1,0,1,1,0,1
    proba = np.stack([1 - p1, p1], 1)
    m = classification_metrics(y, proba, ["benign", "malignant"])
    assert m["confusion_matrix"]["matrix"] == [[3, 1], [1, 3]]
    assert m["accuracy"] == 0.75
    mal = m["per_class"][1]
    assert (mal["precision"], mal["recall"], mal["f1"]) == (0.75, 0.75, 0.75)
    # positives 0.9,0.8,0.4,0.7 vs negatives 0.1,0.2,0.6,0.3 -> 15 of 16 pairs ranked correctly
    assert m["roc_auc"] == pytest.approx(15 / 16, abs=1e-4)
    assert "malignant" in m["roc_auc_note"] and m["roc_curve"]["fpr"][0] == 0 and m["roc_curve"]["tpr"][-1] == 1


def test_auc_perfect_inverse_ties_and_undefined():
    assert roc_auc_binary([0, 0, 1, 1], [.1, .2, .8, .9]) == 1.0
    assert roc_auc_binary([0, 0, 1, 1], [.9, .8, .2, .1]) == 0.0
    assert roc_auc_binary([0, 1, 0, 1], [.5, .5, .5, .5]) == 0.5
    assert roc_auc_binary([1, 1, 1], [.1, .2, .3]) is None


def test_never_predicted_class_counts_as_zero_in_macro_f1():
    y = np.array([0] * 6 + [1] * 2)
    proba = np.tile([.9, .1], (8, 1))                             # always predicts class 0
    m = classification_metrics(y, proba, ["a", "b"])
    assert m["per_class"][1]["precision"] is None and m["per_class"][1]["f1"] == 0.0
    assert m["macro_f1"] == pytest.approx((2 * 6 / (2 * 6 + 2) + 0) / 2, abs=1e-4)   # not inflated by dropping class b


def test_multiclass_macro_auc_and_no_roc_curve():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y] * .8 + .1
    m = classification_metrics(y, proba, ["a", "b", "c"])
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0 and m["roc_curve"] is None


# ---------------- decision threshold ----------------
from cnn.classification.tune_threshold import choose, sweep
from cnn.classification.evaluate import predict_labels


def test_threshold_changes_predictions_by_hand():
    p_mal = np.array([0.9, 0.45, 0.35, 0.2, 0.1])
    proba = np.stack([1 - p_mal, p_mal], 1)
    assert predict_labels(proba, None, 1).tolist() == [1, 0, 0, 0, 0]          # plain argmax
    assert predict_labels(proba, 0.4, 1).tolist() == [1, 1, 0, 0, 0]
    assert predict_labels(proba, 0.3, 1).tolist() == [1, 1, 1, 0, 0]
    assert predict_labels(proba, 0.0, 1).tolist() == [1] * 5


def test_lower_threshold_raises_recall_and_lowers_precision():
    y = np.array([0] * 8 + [1] * 4)
    p = np.array([.05, .1, .15, .2, .25, .3, .45, .55, .35, .4, .6, .9])        # malignant probabilities
    proba = np.stack([1 - p, p], 1)
    hi = classification_metrics(y, proba, ["benign", "malignant"])               # argmax = threshold 0.5
    lo = classification_metrics(y, proba, ["benign", "malignant"], threshold=0.3)
    assert hi["per_class"][1]["recall"] == 0.5 and lo["per_class"][1]["recall"] == 1.0
    assert lo["per_class"][1]["precision"] < 1.0 and "0.300" in lo["decision_rule"]
    assert lo["roc_auc"] == hi["roc_auc"]                                        # AUC does not depend on the cut-off


def test_choose_highest_threshold_meeting_target_recall():
    y = np.array([1] * 10 + [0] * 10)
    p = np.concatenate([np.linspace(0.2, 0.9, 10), np.linspace(0.02, 0.6, 10)])
    rows = sweep(y, p)
    best, warn = choose(rows, "recall", 0.8)
    assert warn is None and best["recall"] >= 0.8
    assert all(r["recall"] < 0.8 for r in rows if r["threshold"] > best["threshold"] and r["recall"] is not None)
    best2, warn2 = choose(rows, "recall", 1.01)                                   # impossible target -> honest warning
    assert warn2 and "No threshold reached" in warn2
    assert choose(rows, "f2")[0]["f2"] == max(r["f2"] for r in rows)
