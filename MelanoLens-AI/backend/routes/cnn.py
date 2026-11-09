"""CNN endpoints: status, segmentation, classification, offline evaluation.

Inputs: either a multipart 'image' (validated like /api/upload, nothing stored, images returned inline)
or JSON {"result_id": ...} for an image already uploaded (outputs are stored with the result so the
analysis page can show them next to the classical results).
No endpoint ever returns fake output: a missing/invalid model gives a JSON error (HTTP 503).
"""
import time

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from cnn import registry, utils as cu
from services.metrics import evaluate_masks
from utils.image_validation import ImageValidationError, decode_image

bp = Blueprint("cnn", __name__)
COMPARE_NOTE = ("Agreement between the two methods is NOT accuracy. Scores against a reference mask are only "
                "meaningful for images the CNN never saw in training; use the Evaluation page's held-out test "
                "metrics for a fair comparison.")


def _store():
    return current_app.config["RESULT_STORE"]


def _input():
    """-> (bgr image, result_id or None)."""
    f = request.files.get("image")
    if f is not None and f.filename:
        cfg = current_app.config
        img = decode_image(f.read(), f.filename, cfg["ALLOWED_EXTENSIONS"], cfg["MAX_CONTENT_LENGTH"], cfg["MIN_IMAGE_SIDE"])
        return img, None
    body = request.get_json(silent=True)
    if isinstance(body, dict) and isinstance(body.get("result_id"), str):
        rid = body["result_id"]
        _store().require(rid)
        return _store().load_source(rid), rid
    raise ImageValidationError("Send an 'image' file (multipart) or JSON {\"result_id\": ...}.", "missing_file")


def _mask_metrics(pred_bool, ref_gray):
    m = evaluate_masks(pred_bool.astype(np.uint8) * 255, ref_gray, allow_resize=True)
    r = lambda v: None if v is None else round(float(v), 4)
    return {k: r(m[k]) for k in ("dice", "iou", "precision", "recall")} | {
        "pixel_accuracy": r((m["tp"] + m["tn"]) / (m["tp"] + m["fp"] + m["fn"] + m["tn"]))}


def _classical_mask(rid, shape):
    """The classical pipeline's selected mask, resized (nearest) to the source image size, or None."""
    if rid is None:
        return None
    g = _store().read_gray(rid, "mask")
    if g is None:
        return None
    return cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def _persist(rid, key, payload, images=None):
    store = _store()
    if images:
        store.save_images(rid, images)
    meta = store.read_meta(rid)
    meta.setdefault("cnn", {})[key] = payload
    meta["images"] = store.list_images(rid)
    store.write_meta(rid, meta)


def _with_urls(rid, payload, images):
    if rid:
        payload["image_urls"] = {n: f"/api/results/{rid}/image/{n}" for n in images}
    else:
        payload["images_inline"] = {n: cu.png_data_uri(a) for n, a in images.items()}
    return payload


@bp.get("/api/cnn/status")
def status():
    cfg = current_app.config
    return jsonify({"segmentation": registry.model_status("segmentation", cfg["SEG_MODEL_PATH"]),
                    "classification": registry.model_status("classification", cfg["CLS_MODEL_PATH"]),
                    "train_commands": {"segmentation": "cd backend && python -m cnn.segmentation.train",
                                       "classification": "cd backend && python -m cnn.classification.train"}})


@bp.post("/api/cnn/segment")
def segment():
    bgr, rid = _input()
    seg = registry.get_segmenter(current_app.config["SEG_MODEL_PATH"])
    t0 = time.perf_counter()
    out = seg.predict(bgr)
    mask, prob = out["mask"], out["probability"]
    images = {"cnn_mask": mask.astype(np.uint8) * 255, "cnn_region": cu.tint_region(bgr, mask),
              "cnn_overlay": cu.draw_boundary(bgr, mask, (200, 0, 255)), "cnn_probability": cu.heatmap(prob)}
    warnings = []
    stats = cu.mask_stats(mask)
    if stats["lesion_area_px"] == 0:
        warnings.append("The model predicted no lesion region in this image.")
    elif stats["lesion_area_fraction"] > 0.9:
        warnings.append("The predicted region covers almost the whole image; the result is probably unreliable.")

    comparison = {"classical_available": False, "note": COMPARE_NOTE}
    classical = _classical_mask(rid, mask.shape)
    if classical is not None and classical.any() and mask.any():
        comparison.update({"classical_available": True,
                           "agreement_cnn_vs_classical": _mask_metrics(mask, classical.astype(np.uint8) * 255)})
        both = cu.draw_boundary(bgr, classical, (255, 220, 0))                       # classical = cyan (BGR)
        images["cnn_compare"] = cu.draw_boundary(both, mask, (200, 0, 255))        # CNN = magenta
    elif rid is not None:
        comparison["classical_unavailable_reason"] = ("Run the classical pipeline on the Workspace page first, "
                                                      "or it found no candidate contour." if classical is None
                                                      else "One of the two masks is empty.")
    if rid is not None:
        ref = _store().load_reference(rid)
        if ref is not None:
            comparison["vs_reference"] = {"cnn": _mask_metrics(mask, ref)}
            if classical is not None and classical.any():
                comparison["vs_reference"]["classical"] = _mask_metrics(classical, ref)

    payload = {"model": seg.describe(), "stats": stats, "warnings": warnings, "comparison": comparison,
               "inference_ms": round((time.perf_counter() - t0) * 1000), "created": cu.now_iso()}
    if rid:
        _persist(rid, "segmentation", payload, images)
    return jsonify(_with_urls(rid, {"result_id": rid, **payload}, images))


@bp.post("/api/cnn/classify")
def classify():
    bgr, rid = _input()
    clf = registry.get_classifier(current_app.config["CLS_MODEL_PATH"])
    t0 = time.perf_counter()
    pred = clf.predict(bgr)
    payload = {"model": clf.describe(), "prediction": pred, "kind": "model_prediction",
               "kind_note": "A model prediction, not a confirmed medical diagnosis.",
               "inference_ms": round((time.perf_counter() - t0) * 1000), "created": cu.now_iso()}
    if rid:
        _persist(rid, "classification", payload)
    return jsonify({"result_id": rid, **payload})


@bp.get("/api/cnn/evaluation")
def evaluation():
    """Held-out TEST-set results written by the evaluate scripts (never recomputed or invented here)."""
    cfg = current_app.config
    out = {}
    for kind, key in (("segmentation", "SEG_MODEL_PATH"), ("classification", "CLS_MODEL_PATH")):
        p = cfg[key]
        ev = cu.read_json(cu.sidecar(p, "eval"))
        hist = cu.read_json(cu.sidecar(p, "history"))
        meta = cu.read_json(cu.sidecar(p, "meta"), {}) or {}
        out[kind] = {"available": ev is not None, "results": ev, "history": hist, "threshold": cu.read_json(cu.sidecar(p, "threshold")),
                     "comparison": cu.read_json(cu.sidecar(p, "comparison")) if kind == "segmentation" else None,
                     "training": {k: meta.get(k) for k in ("trained_at", "img_size", "epochs_run", "split_sizes", "weights", "alpha", "loss", "augmentation") if k in meta},
                     "message": None if ev else "No evaluation yet. Train the model (which evaluates the test split at the end) or run the evaluate command."}
    return jsonify(out)
