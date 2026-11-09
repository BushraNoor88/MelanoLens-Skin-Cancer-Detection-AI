"""POST /api/upload : validate an image (and an optional reference mask) and create a result folder."""
import time

import cv2
from flask import Blueprint, current_app, jsonify, request

from utils.file_utils import new_result_id, safe_filename
from utils.image_validation import ImageValidationError, decode_image

bp = Blueprint("upload", __name__)


def read_mask_file(file_storage, cfg):
    """Decode a mask upload into a clean 0/255 single-channel array."""
    data = file_storage.read()
    bgr = decode_image(data, file_storage.filename or "", cfg["ALLOWED_EXTENSIONS"],
                       cfg["MAX_CONTENT_LENGTH"], cfg["MIN_IMAGE_SIDE"])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    threshold = 0 if gray.max() <= 1 else 127           # 0/1 masks and 0/255 masks both work
    return (gray > threshold).astype("uint8") * 255


def check_same_aspect(img_shape, mask_shape):
    ir, mr = img_shape[1] / img_shape[0], mask_shape[1] / mask_shape[0]
    if abs(ir - mr) / ir > 0.02:
        raise ImageValidationError(
            f"Reference mask ({mask_shape[1]}x{mask_shape[0]}) does not match the image "
            f"({img_shape[1]}x{img_shape[0]}) aspect ratio.", "dimension_mismatch")


@bp.post("/api/upload")
def upload():
    cfg = current_app.config
    f = request.files.get("image")
    if f is None or not f.filename:
        raise ImageValidationError("No image was sent. Use the form field 'image'.", "missing_file")
    data = f.read()
    img = decode_image(data, f.filename, cfg["ALLOWED_EXTENSIONS"], cfg["MAX_CONTENT_LENGTH"], cfg["MIN_IMAGE_SIDE"])

    reference = None
    mf = request.files.get("mask")
    if mf is not None and mf.filename:
        reference = read_mask_file(mf, cfg)
        check_same_aspect(img.shape, reference.shape)

    rid = new_result_id()
    h, w = img.shape[:2]
    info = {"name": safe_filename(f.filename), "width": int(w), "height": int(h),
            "size_bytes": len(data), "type": f.filename.rsplit(".", 1)[-1].lower()}
    meta = {"result_id": rid, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": info,
            "reference_mask_available": reference is not None, "processed": False}
    current_app.config["RESULT_STORE"].create(rid, img, meta, reference)
    return jsonify(meta), 201
