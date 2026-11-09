"""Processing and result endpoints."""
from flask import Blueprint, current_app, jsonify, request, send_file

from services.boundary_extraction import SELECTIONS
from services.edge_booster import METHODS
from services.edge_detection import DETECTORS
from services.params import DEFAULTS, MODES, parse_process_params
from services.pipeline import run_pipeline
from services.preprocessing import FILTERS, ParameterError
from services.results_service import ResultNotFound

bp = Blueprint("processing", __name__)


def store():
    return current_app.config["RESULT_STORE"]


def public_meta(rid, meta):
    out = dict(meta)
    names = meta.get("images", [])
    out["image_urls"] = {n: f"/api/results/{rid}/image/{n}" for n in names}
    out["download_urls"] = {n: f"/api/results/{rid}/download/{n}" for n in names}
    return out


@bp.get("/api/defaults")
def defaults():
    return jsonify({"defaults": DEFAULTS, "options": {"filter": FILTERS, "booster_method": METHODS,
                    "detector": DETECTORS, "mode": MODES, "selection": SELECTIONS}})


@bp.post("/api/process")
def process():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ParameterError("Request body must be a JSON object.")
    rid = body.get("result_id")
    store().require(rid)                                  # 404 if unknown / malformed
    params = parse_process_params(body.get("params") or {})

    out = run_pipeline(store().load_source(rid), params, store().load_reference(rid))
    meta = store().read_meta(rid)
    store().clear_images(rid)
    store().save_images(rid, out["images"])
    meta.update(out["meta"])
    meta.update({"params": params, "processed": True, "images": store().list_images(rid)})
    store().write_meta(rid, meta)
    return jsonify(public_meta(rid, meta))


@bp.get("/api/results/<rid>")
def get_result(rid):
    return jsonify(public_meta(rid, store().read_meta(rid)))


def _image_or_404(rid, image_type):
    path = store().image_path(rid, image_type)
    if path is None:
        raise ResultNotFound(f"Image '{image_type}' is not available for this result.")
    return path


@bp.get("/api/results/<rid>/image/<image_type>")
def get_image(rid, image_type):
    return send_file(_image_or_404(rid, image_type), mimetype="image/png")


@bp.get("/api/results/<rid>/download/<image_type>")
def download_image(rid, image_type):
    path = _image_or_404(rid, image_type)
    return send_file(path, mimetype="image/png", as_attachment=True,
                     download_name=f"melanolens_{image_type}_{rid[:8]}.png")


@bp.delete("/api/results/<rid>")
def delete_result(rid):
    store().delete(rid)
    return jsonify({"deleted": True, "result_id": rid})
