import numpy as np
import pytest

from services.metrics import (MetricsError, align_masks, boundary_f1, edge_map_boundary_f1,
                              evaluate_masks, mask_boundary)


def _square(x, y, s=10, size=40):
    m = np.zeros((size, size), np.uint8)
    m[y:y + s, x:x + s] = 255
    return m


def test_perfect_match():
    m = _square(10, 10)
    r = evaluate_masks(m, m)
    assert r["dice"] == 1.0 and r["iou"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0
    assert r["boundary_f1"] == 1.0


def test_known_partial_overlap_values():
    r = evaluate_masks(_square(10, 10), _square(15, 10))   # 50 px overlap, 50 FP, 50 FN
    assert (r["tp"], r["fp"], r["fn"]) == (50, 50, 50)
    assert abs(r["dice"] - 0.5) < 1e-9
    assert abs(r["iou"] - 1 / 3) < 1e-9
    assert abs(r["precision"] - 0.5) < 1e-9 and abs(r["recall"] - 0.5) < 1e-9
    assert abs(r["f1"] - 0.5) < 1e-9


def test_disjoint_masks_score_zero():
    r = evaluate_masks(_square(0, 0), _square(30, 30))
    assert r["dice"] == 0.0 and r["iou"] == 0.0 and r["precision"] == 0.0 and r["recall"] == 0.0


def test_zero_one_and_zero_255_masks_are_equivalent():
    a = _square(5, 5)
    assert evaluate_masks(a, (a > 0).astype(np.uint8))["dice"] == 1.0


def test_empty_prediction_gives_none_not_fake_numbers():
    r = evaluate_masks(np.zeros((40, 40), np.uint8), _square(10, 10))
    assert r["precision"] is None and r["recall"] == 0.0 and r["dice"] == 0.0
    assert r["boundary_f1"] is None
    both = evaluate_masks(np.zeros((40, 40), np.uint8), np.zeros((40, 40), np.uint8))
    assert both["dice"] == 1.0 and both["precision"] is None


def test_dimension_mismatch_is_an_error_unless_resize_allowed():
    small, big = np.zeros((20, 20), np.uint8), np.zeros((40, 40), np.uint8)
    with pytest.raises(MetricsError) as e:
        evaluate_masks(small, big)
    assert e.value.code == "dimension_mismatch"
    ref = _square(10, 10)                                  # 40x40
    pred = np.kron(ref, np.ones((2, 2), np.uint8))         # 80x80 version
    _, r = align_masks(pred, ref, allow_resize=True)
    assert r.shape == pred.shape
    assert evaluate_masks(pred, ref, allow_resize=True)["dice"] == 1.0   # nearest keeps it binary


def test_three_channel_mask_rejected():
    with pytest.raises(MetricsError):
        evaluate_masks(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.uint8))


def test_boundary_f1_drops_when_shifted_and_respects_tolerance():
    a, b = _square(10, 10, 20, 60), _square(10, 16, 20, 60)
    assert boundary_f1(a, b, tolerance_px=0) < 0.6
    assert boundary_f1(a, b, tolerance_px=8) == 1.0
    with pytest.raises(MetricsError):
        boundary_f1(a, b, tolerance_px=50)


def test_edge_map_boundary_f1():
    ref = _square(10, 10, 20, 60)
    edge = mask_boundary(ref > 0).astype(np.uint8) * 255
    assert edge_map_boundary_f1(edge, ref, 1) == 1.0
    assert edge_map_boundary_f1(np.zeros_like(edge), ref, 1) is None
    with pytest.raises(MetricsError):
        edge_map_boundary_f1(edge[:30], ref, 1)
