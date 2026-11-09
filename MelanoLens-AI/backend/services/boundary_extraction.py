"""Boundary extraction: edge map -> contours -> candidate mask.

Three different things (do not mix them up):
  * EDGE MAP   : pixels where an edge detector fired (thin lines, may have gaps/noise).
  * CONTOUR    : an ordered outline (list of points) traced around a connected edge blob.
  * MASK       : the region INSIDE a chosen contour, filled white. This is only a
                 CANDIDATE lesion region derived from edges. It is not a clinically
                 accurate segmentation and not a ground truth.
"""
import math

import cv2
import numpy as np

from .preprocessing import ParameterError, validate_odd_kernel

SELECTIONS = ("largest", "most_circular", "closest_to_center")
NO_CANDIDATE_MESSAGE = ("No reliable candidate contour was found. Try a larger closing kernel, "
                        "lower thresholds, a smaller minimum area, or enable the Edge Booster.")


def clean_edge_map(edges, close_kernel=5, close_iterations=1, open_kernel=0):
    """Binarise, then CLOSE small gaps (dilate->erode) and optionally OPEN (remove specks)."""
    if edges.ndim != 2:
        raise ParameterError("Edge map must be single-channel.")
    out = (edges > 0).astype(np.uint8) * 255
    ck = validate_odd_kernel(close_kernel, "close_kernel", 1, 31)
    if isinstance(close_iterations, bool) or not isinstance(close_iterations, int) or not 1 <= close_iterations <= 5:
        raise ParameterError("close_iterations must be an integer between 1 and 5.")
    if ck > 1:
        # A 1-pixel-wide edge cannot be "closed" directly: the structuring element just slides
        # around the thin line. So we thicken the edges slightly (3x3 dilation) and then close.
        # Side effect: the outer contour sits about 1 pixel outside the original edge.
        out = cv2.dilate(out, np.ones((3, 3), np.uint8))
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, se, iterations=close_iterations)
    if open_kernel:
        ok = validate_odd_kernel(open_kernel, "open_kernel", 3, 15)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, se)
    return out


def measure_contour(contour, shape):
    """Area, perimeter, bounding box, circularity (4*pi*A / P^2, 1.0 = perfect circle)."""
    h, w = shape
    area = float(cv2.contourArea(contour))
    # A raw pixel chain is stair-stepped and over-measures length by ~5%, which would make even a
    # perfect circle read as ~0.9 circular. Simplifying with a 1-pixel tolerance removes the stairs.
    perimeter = float(cv2.arcLength(cv2.approxPolyDP(contour, 1.0, True), True))
    x, y, bw, bh = cv2.boundingRect(contour)
    circularity = min(1.0, 4.0 * math.pi * area / perimeter ** 2) if area > 0 and perimeter > 0 else None
    m = cv2.moments(contour)
    cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] > 0 else (x + bw / 2, y + bh / 2)
    return {"contour": contour, "area": area, "perimeter": perimeter,
            "bbox": (int(x), int(y), int(bw), int(bh)), "circularity": circularity,
            "centroid": (float(cx), float(cy)),
            "touches_border": bool(x <= 0 or y <= 0 or x + bw >= w or y + bh >= h)}


def find_candidates(clean, min_area_pct=0.5, max_area_pct=95.0):
    """External contours whose area is between min and max percent of the image, largest first."""
    if not (0 <= min_area_pct < max_area_pct <= 100):
        raise ParameterError("Area limits must satisfy 0 <= min_area_pct < max_area_pct <= 100.")
    h, w = clean.shape
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for c in contours:
        m = measure_contour(c, (h, w))
        pct = 100.0 * m["area"] / (h * w)
        if m["area"] > 0 and min_area_pct <= pct <= max_area_pct:
            candidates.append(m)
    candidates.sort(key=lambda c: c["area"], reverse=True)
    for i, c in enumerate(candidates):
        c["index"] = i
    return candidates


def select_candidate(candidates, shape, selection="largest", candidate_index=None):
    """Return the index of the chosen candidate (or None if there are none)."""
    if selection not in SELECTIONS:
        raise ParameterError(f"selection must be one of: {', '.join(SELECTIONS)}.")
    if candidate_index is not None:
        if not candidates or not 0 <= candidate_index < len(candidates):
            raise ParameterError(f"candidate_index {candidate_index} is out of range "
                                 f"({len(candidates)} candidate(s) available).")
        return candidate_index
    if not candidates:
        return None
    if selection == "largest":
        return 0
    if selection == "most_circular":
        return max(range(len(candidates)), key=lambda i: candidates[i]["circularity"] or -1.0)
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    return min(range(len(candidates)),
               key=lambda i: math.hypot(candidates[i]["centroid"][0] - cx, candidates[i]["centroid"][1] - cy))


def extract_boundary(edges, close_kernel=5, close_iterations=1, open_kernel=0, min_area_pct=0.5,
                     max_area_pct=95.0, selection="largest", candidate_index=None):
    """Full step. Returns dict with candidates, selected_index, mask (or None) and a message."""
    clean = clean_edge_map(edges, close_kernel, close_iterations, open_kernel)
    candidates = find_candidates(clean, min_area_pct, max_area_pct)
    idx = select_candidate(candidates, edges.shape, selection, candidate_index)
    mask, message = None, None
    if idx is None:
        message = NO_CANDIDATE_MESSAGE
    else:
        mask = np.zeros(edges.shape, np.uint8)
        cv2.drawContours(mask, [candidates[idx]["contour"]], -1, 255, thickness=cv2.FILLED)
    return {"clean_edges": clean, "candidates": candidates, "selected_index": idx,
            "mask": mask, "message": message}


def draw_overlay(bgr, candidates, selected_index):
    """Selected contour thick cyan, other candidates thin yellow, drawn on a COPY of the image."""
    out = bgr.copy()
    for c in candidates:
        if c["index"] != selected_index:
            cv2.drawContours(out, [c["contour"]], -1, (0, 215, 255), 1)
    if selected_index is not None:
        cv2.drawContours(out, [candidates[selected_index]["contour"]], -1, (220, 200, 0), 2)
    return out


def candidate_summary(c):
    """JSON-friendly version of a candidate (no raw contour points)."""
    x, y, w, h = c["bbox"]
    return {"index": int(c["index"]), "area_px": round(c["area"], 1), "perimeter_px": round(c["perimeter"], 1),
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "circularity": None if c["circularity"] is None else round(c["circularity"], 4),
            "centroid": {"x": round(c["centroid"][0], 1), "y": round(c["centroid"][1], 1)},
            "touches_border": c["touches_border"]}
