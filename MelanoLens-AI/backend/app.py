"""MelanoLens AI - Flask application factory.

Flask serves BOTH the REST API (/api/...) and the frontend files, so the
browser and the API share one origin (http://127.0.0.1:5000) and no CORS
setup is needed. If you ever serve the frontend from another port, install
flask-cors and allow only that origin.

Intended for LOCAL educational use. There is no authentication.
"""
from flask import Flask, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException

from cnn.errors import DatasetError, ModelNotAvailable
from config import Config
from services.metrics import MetricsError
from services.preprocessing import ParameterError
from services.results_service import ResultNotFound, ResultStateError, ResultStore
from utils.image_validation import ImageValidationError


def create_app(config_object=Config) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config.from_object(config_object)
    app.config["RESULT_STORE"] = ResultStore(app.config["RESULTS_DIR"])

    from routes import cnn, evaluation, processing, upload
    for module in (upload, processing, evaluation, cnn):
        app.register_blueprint(module.bp)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "app": app.config["APP_NAME"],
                        "max_upload_mb": app.config["MAX_UPLOAD_MB"]})

    # ---- consistent JSON errors: {"error": {"code": ..., "message": ...}} ----
    def error(code, message, status):
        return jsonify({"error": {"code": code, "message": message}}), status

    @app.errorhandler(ImageValidationError)
    def _img_err(e):
        return error(e.code, str(e), 413 if e.code == "file_too_large" else 400)

    @app.errorhandler(ParameterError)
    def _param_err(e):
        return error("invalid_parameter", str(e), 400)

    @app.errorhandler(MetricsError)
    def _metrics_err(e):
        return error(e.code, str(e), 400)

    @app.errorhandler(ResultNotFound)
    def _not_found(e):
        return error("result_not_found", str(e), 404)

    @app.errorhandler(ResultStateError)
    def _state_err(e):
        return error("not_processed", str(e), 409)

    @app.errorhandler(ModelNotAvailable)
    def _model_err(e):
        return error(e.code, str(e), 503)

    @app.errorhandler(DatasetError)
    def _dataset_err(e):
        return error("dataset_error", str(e), 400)

    @app.errorhandler(HTTPException)
    def _http_err(e):
        names = {404: "not_found", 405: "method_not_allowed", 413: "file_too_large"}
        msg = "File exceeds the upload limit." if e.code == 413 else e.description
        return error(names.get(e.code, "http_error"), msg, e.code)

    @app.errorhandler(Exception)
    def _unexpected(e):
        app.logger.exception("Unexpected error")
        return error("server_error", "Unexpected server error.", 500)

    # ---- frontend (static files) ----
    @app.get("/")
    def index():
        return send_from_directory(app.config["FRONTEND_DIR"], "index.html")

    @app.get("/<path:filename>")
    def frontend(filename):
        return send_from_directory(app.config["FRONTEND_DIR"], filename)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
