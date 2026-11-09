"""Evaluation endpoints. Scores always come from masks saved on disk, never from cached guesses."""
import json

from flask import Blueprint, Response, current_app, jsonify, request

from services.evaluation_service import (NO_REFERENCE_MESSAGE, evaluate_stored, make_chart_png,
                                         rows_to_csv)
from services.preprocessing import ParameterError
from services.results_service import ResultNotFound

bp = Blueprint("evaluation", __name__)


def _evaluate(rid):
    store = current_app.config["RESULT_STORE"]
    return evaluate_stored(store, rid, store.read_meta(rid))


@bp.post("/api/evaluate")
def evaluate():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("result_id"), str):
        raise ParameterError("Send a JSON body with a 'result_id'.")
    return jsonify(_evaluate(body["result_id"]))


def _available(rid):
    result = _evaluate(rid)
    if not result["available"]:
        raise ResultNotFound(NO_REFERENCE_MESSAGE)
    return result


@bp.get("/api/evaluate/<rid>/export")
def export(rid):
    fmt = request.args.get("format", "csv")
    if fmt not in ("csv", "json"):
        raise ParameterError("format must be 'csv' or 'json'.")
    result = _available(rid)
    if fmt == "csv":
        body, mime = rows_to_csv(result["rows"]), "text/csv"
    else:
        body, mime = json.dumps(result, indent=2), "application/json"
    return Response(body, mimetype=mime,
                    headers={"Content-Disposition": f"attachment; filename=melanolens_metrics_{rid[:8]}.{fmt}"})


@bp.get("/api/evaluate/<rid>/chart.png")
def chart(rid):
    return Response(make_chart_png(_available(rid)["comparison"]), mimetype="image/png")
