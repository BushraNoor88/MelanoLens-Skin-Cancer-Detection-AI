"""The whole MelanoLens pipeline in one function (used by POST /api/process).

For EVERY detector it runs two modes on the same input and parameters:
  conventional : preprocessing -> detector
  enhanced     : preprocessing -> Edge Booster -> detector
One (detector, mode) pair is the "selected" one and drives the overlay, mask
and measurements. All pairs get a candidate mask so the Evaluation page can
compare them when a reference mask exists.
"""
import cv2
import numpy as np

from .boundary_extraction import candidate_summary, draw_overlay, extract_boundary
from .edge_booster import edge_boost
from .edge_detection import DETECTORS, canny_auto_thresholds, run_detector
from .evaluation_service import SHORT, comparison_summary, evaluate_combos, NO_REFERENCE_MESSAGE
from .params import MODES
from .preprocessing import preprocess, resize_keep_aspect

NOTES = [
    "Edge map = detected edge pixels. Contour = traced outline. Mask = filled region inside a contour.",
    "The mask is a CANDIDATE region derived from edges. It is not a clinical segmentation or a ground truth.",
    "Edge enhancement can make edges more visible but does not guarantee a more accurate segmentation.",
]


def _difference_image(pred, ref):
    """TP green, FP red, FN blue on a dark background (BGR)."""
    p, r = pred > 0, ref > 0
    out = np.full((*p.shape, 3), 30, np.uint8)
    out[p & r] = (0, 200, 0)
    out[p & ~r] = (0, 0, 220)
    out[~p & r] = (220, 120, 0)
    return out


def _clear_border(edge_map, margin):
    """Zero the outer `margin` pixels. Frames, vignettes and scanner borders produce strong
    edges along the image boundary that are not part of the lesion."""
    if margin > 0:
        edge_map = edge_map.copy()
        edge_map[:margin, :] = 0
        edge_map[-margin:, :] = 0
        edge_map[:, :margin] = 0
        edge_map[:, -margin:] = 0
    return edge_map


def _replicate_border(img, margin):
    """Overwrite the outer `margin` pixels with the nearest pixels just inside.

    A 1-2 px bright/dark frame would otherwise be smeared inwards by blur filters and
    become a strong 'edge'. Replacing it BEFORE filtering removes it without changing
    the image size, so reference masks stay aligned."""
    if margin <= 0:
        return img
    out = img.copy()
    m = min(margin, img.shape[0] // 4, img.shape[1] // 4)
    out[:m] = out[m]
    out[-m:] = out[-m - 1]
    out[:, :m] = out[:, m:m + 1]
    out[:, -m:] = out[:, -m - 1:-m]
    return out


def run_pipeline(source_bgr, p, reference_mask=None):
    pre_params = {k: p[k] for k in ("filter", "kernel_size", "bilateral_sigma_color", "bilateral_sigma_space",
                                    "clahe", "clahe_clip", "clahe_tile", "remove_hair", "max_side")}
    source_bgr = _replicate_border(source_bgr, p["ignore_border_px"])
    pre = preprocess(source_bgr, pre_params)
    base = resize_keep_aspect(source_bgr, p["max_side"])
    gray = pre["preprocessed"]
    boost = edge_boost(gray, p["booster_method"], p["booster_alpha"], p["booster_kernel"],
                       p["multiscale_sigmas"], p["multiscale_weights"])
    inputs = {"conventional": gray, "enhanced": boost["enhanced"]}
    det_params = {k: p[k] for k in ("canny_auto", "canny_percentile", "canny_low", "canny_high", "canny_blur", "sobel_threshold", "prewitt_threshold",
                                    "laplacian_threshold", "log_sigma", "log_threshold")}
    bparams = {k: p[k] for k in ("close_kernel", "close_iterations", "open_kernel", "min_area_pct",
                                 "max_area_pct", "selection")}

    images = {"original": base, "preprocessed": gray, "enhanced": boost["enhanced"], "gradient": boost["gradient"]}
    edges, masks, extractions = {}, {}, {}
    selected = (p["mode"], p["detector"])
    for det in DETECTORS:
        for mode in MODES:
            e, _ = run_detector(det, inputs[mode], det_params)
            e = _clear_border(e, p["ignore_border_px"])
            idx = p["candidate_index"] if (mode, det) == selected else None
            ex = extract_boundary(e, candidate_index=idx, **bparams)
            edges[(mode, det)], masks[(mode, det)], extractions[(mode, det)] = e, ex["mask"], ex
            images[f"edges_{SHORT[mode]}_{det}"] = e
            if ex["mask"] is not None:
                images[f"mask_{SHORT[mode]}_{det}"] = ex["mask"]

    sel = extractions[selected]
    images["edges"] = edges[selected]
    images["contour_overlay"] = draw_overlay(base, sel["candidates"], sel["selected_index"])
    if sel["mask"] is not None:
        images["mask"] = sel["mask"]

    chosen = None if sel["selected_index"] is None else sel["candidates"][sel["selected_index"]]
    boundary = {"detector": p["detector"], "mode": p["mode"], "selection": p["selection"],
                "candidate_count": len(sel["candidates"]), "selected_index": sel["selected_index"],
                "candidates": [candidate_summary(c) for c in sel["candidates"]],
                "measurements": None if chosen is None else candidate_summary(chosen),
                "message": sel["message"]}

    evaluation = {"available": False, "message": NO_REFERENCE_MESSAGE}
    reference_info = {"available": reference_mask is not None, "resized": False}
    if reference_mask is not None:
        ref = (reference_mask > 0).astype(np.uint8) * 255
        if ref.shape != gray.shape:                       # explicit, reported, nearest-neighbour only
            ref = cv2.resize(ref, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            reference_info["resized"] = True
        images["reference_mask"] = ref
        if sel["mask"] is not None:
            images["difference"] = _difference_image(sel["mask"], ref)
        rows = evaluate_combos(ref, masks, edges, p["boundary_tolerance_px"])
        evaluation = {"available": True, "tolerance_px": p["boundary_tolerance_px"], "rows": rows,
                      "comparison": comparison_summary(rows)}

    warnings = []
    if p["booster_alpha"] > 3:
        warnings.append("High enhancement strength can amplify noise.")
    if boundary["selected_index"] is None:
        warnings.append(boundary["message"])
    h, w = gray.shape
    if chosen is not None and chosen["touches_border"] and chosen["bbox"][2] >= 0.9 * w and chosen["bbox"][3] >= 0.9 * h:
        warnings.append("The selected contour spans almost the whole image. It is probably a frame, vignette or edge clutter, "
                        "not the lesion. Increase 'Ignore image border', raise the thresholds, or pick another candidate.")
    canny_used = {}
    for mode in MODES:
        lo, hi = (canny_auto_thresholds(inputs[mode], p["canny_percentile"], p["canny_blur"]) if p["canny_auto"]
                  else (p["canny_low"], p["canny_high"]))
        canny_used[mode] = {"low": round(float(lo), 1), "high": round(float(hi), 1), "auto": p["canny_auto"]}
    meta = {"canny_thresholds_used": canny_used, "processed_size": {"width": int(w), "height": int(h)}, "boundary": boundary,
            "reference": reference_info, "evaluation": evaluation, "warnings": warnings, "notes": NOTES}
    return {"images": images, "meta": meta}
