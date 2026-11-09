"""Segmentation metrics. All functions compare MASKS (filled regions), never raw edge maps.

The one exception is edge_map_boundary_f1, which has an explicit definition:
edge pixels are compared with the BOUNDARY of the reference mask, and a pixel
counts as a match if it lies within `tolerance_px` of the other boundary.

Definitions (TP/FP/FN = pixel counts):
  Dice = 2TP / (2TP + FP + FN)      IoU = TP / (TP + FP + FN)
  Precision = TP / (TP + FP)        Recall = TP / (TP + FN)
  F1 = 2PR / (P + R)   (for pixel masks this equals Dice; reported for completeness)
Undefined values (e.g. precision of an empty prediction) are returned as None,
never invented. If BOTH masks are empty, Dice and IoU are 1.0 (perfect agreement).
"""
import cv2
import numpy as np


class MetricsError(ValueError):
    def __init__(self, message, code="metrics_error"):
        super().__init__(message)
        self.code = code


def to_binary(mask):
    if mask.ndim != 2:
        raise MetricsError("Masks must be single-channel images.", "invalid_mask")
    return mask > 0                                   # works for 0/1 and 0/255 masks


def align_masks(pred, ref, allow_resize=False):
    """Return two boolean arrays of equal shape. Resizing is never silent."""
    p, r = to_binary(pred), to_binary(ref)
    if p.shape == r.shape:
        return p, r
    if not allow_resize:
        raise MetricsError(f"Mask dimensions differ: predicted {p.shape[1]}x{p.shape[0]} vs "
                           f"reference {r.shape[1]}x{r.shape[0]} (width x height).", "dimension_mismatch")
    resized = cv2.resize(r.astype(np.uint8) * 255, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_NEAREST)
    return p, resized > 0


def _div(a, b):
    return None if b == 0 else a / b


def confusion_counts(pred_bool, ref_bool):
    tp = int(np.count_nonzero(pred_bool & ref_bool))
    fp = int(np.count_nonzero(pred_bool & ~ref_bool))
    fn = int(np.count_nonzero(~pred_bool & ref_bool))
    tn = int(pred_bool.size - tp - fp - fn)
    return tp, fp, fn, tn


def mask_boundary(mask_bool):
    """1-pixel-wide boundary of a mask (morphological gradient with a 3x3 kernel)."""
    m = mask_bool.astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def _boundary_f1(a, b, tolerance_px):
    if not 0 <= tolerance_px <= 20:
        raise MetricsError("tolerance_px must be between 0 and 20.", "invalid_tolerance")
    if not a.any() or not b.any():
        return None
    dist_to_b = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)   # distance to nearest b pixel
    dist_to_a = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float(np.mean(dist_to_b[a] <= tolerance_px))
    recall = float(np.mean(dist_to_a[b] <= tolerance_px))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def boundary_f1(pred, ref, tolerance_px=2, allow_resize=False):
    p, r = align_masks(pred, ref, allow_resize)
    return _boundary_f1(mask_boundary(p), mask_boundary(r), tolerance_px)


def edge_map_boundary_f1(edge_map, ref_mask, tolerance_px=2):
    """Edge pixels vs the reference mask's boundary (tolerance in pixels)."""
    if edge_map.shape != ref_mask.shape:
        raise MetricsError("Edge map and reference mask must have the same dimensions.", "dimension_mismatch")
    return _boundary_f1(edge_map > 0, mask_boundary(ref_mask > 0), tolerance_px)


def evaluate_masks(pred, ref, tolerance_px=2, allow_resize=False):
    p, r = align_masks(pred, ref, allow_resize)
    tp, fp, fn, tn = confusion_counts(p, r)
    dice = 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    iou = 1.0 if (tp + fp + fn) == 0 else tp / (tp + fp + fn)
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    if precision is not None and recall is not None:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    else:
        f1 = dice
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "f1": f1,
            "boundary_f1": _boundary_f1(mask_boundary(p), mask_boundary(r), tolerance_px),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
