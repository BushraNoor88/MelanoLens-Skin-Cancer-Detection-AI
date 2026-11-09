import cv2
import numpy as np
import pytest

from services import synthetic
from services.boundary_extraction import (clean_edge_map, draw_overlay, extract_boundary, find_candidates,
                                          select_candidate, NO_CANDIDATE_MESSAGE)
from services.edge_detection import canny
from services.metrics import evaluate_masks
from services.preprocessing import ParameterError, to_grayscale


def _edges(maker=synthetic.circle, **kw):
    img, ref = maker(**kw)
    return canny(to_grayscale(img))[0], ref


def test_circle_mask_matches_reference():
    edges, ref = _edges()
    ex = extract_boundary(edges)
    assert ex["mask"] is not None and ex["message"] is None
    assert evaluate_masks(ex["mask"], ref)["dice"] > 0.95
    c = ex["candidates"][ex["selected_index"]]
    assert abs(c["area"] - np.pi * 70 ** 2) / (np.pi * 70 ** 2) < 0.06
    assert c["circularity"] > 0.9
    x, y, w, h = c["bbox"]
    assert abs(w - 141) <= 4 and abs(h - 141) <= 4


def test_irregular_polygon_is_less_circular_than_circle():
    e_poly, ref = _edges(synthetic.irregular_polygon)
    e_circ, _ = _edges()
    poly = extract_boundary(e_poly)
    circ = extract_boundary(e_circ)
    assert poly["mask"] is not None
    assert evaluate_masks(poly["mask"], ref)["dice"] > 0.9
    assert poly["candidates"][0]["circularity"] < circ["candidates"][0]["circularity"]


def test_missing_contour_gives_message_not_crash():
    ex = extract_boundary(np.zeros((128, 128), np.uint8))
    assert ex["mask"] is None and ex["selected_index"] is None
    assert ex["message"] == NO_CANDIDATE_MESSAGE
    assert draw_overlay(np.zeros((128, 128, 3), np.uint8), ex["candidates"], None).shape == (128, 128, 3)


def test_closing_connects_a_gap():
    ring = np.zeros((256, 256), np.uint8)
    cv2.circle(ring, (128, 128), 70, 255, 1)
    ring[128 - 1:128 + 2, 128 + 68:128 + 73] = 0          # cut a small gap in the outline
    assert extract_boundary(ring, close_kernel=1)["mask"] is None
    closed = extract_boundary(ring, close_kernel=7)
    assert closed["mask"] is not None
    assert closed["candidates"][0]["area"] > 14000


def test_area_filter_removes_small_blobs():
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, (100, 100), 4, 255, -1)
    assert find_candidates(clean_edge_map(img, 1), min_area_pct=1.0) == []
    assert len(find_candidates(clean_edge_map(img, 1), min_area_pct=0.01)) == 1


def test_selection_strategies_and_validation():
    img = np.zeros((300, 300), np.uint8)
    cv2.circle(img, (60, 60), 40, 255, 1)                 # larger, near corner
    cv2.circle(img, (200, 160), 25, 255, 1)               # smaller, near centre
    cands = find_candidates(clean_edge_map(img, 1))
    assert len(cands) == 2
    assert select_candidate(cands, img.shape, "largest") == 0
    assert cands[select_candidate(cands, img.shape, "closest_to_center")]["area"] < cands[0]["area"]
    assert select_candidate(cands, img.shape, "largest", candidate_index=1) == 1
    with pytest.raises(ParameterError):
        select_candidate(cands, img.shape, "largest", candidate_index=5)
    with pytest.raises(ParameterError):
        select_candidate(cands, img.shape, "bogus")
    with pytest.raises(ParameterError):
        clean_edge_map(img, close_kernel=4)
    with pytest.raises(ParameterError):
        find_candidates(img, min_area_pct=50, max_area_pct=10)
