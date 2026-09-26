from time import perf_counter
from typing import Any
from uuid import uuid4

from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge, UnsupportedMediaType

from src.config import settings
from src.exception import ApplicationError, ArtifactError, DataValidationError
from src.logger import configure_logging, get_logger
from src.pipeline.prediction_pipeline import PredictionPipeline

configure_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)


def create_app(prediction_service: Any | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=settings.max_content_length,
        JSON_SORT_KEYS=False,
    )

    if prediction_service is None:
        try:
            prediction_service = PredictionPipeline.from_artifacts(settings.artifact_dir)
            app.logger.info("Prediction service initialized", extra={"event": "model_loaded"})
        except ArtifactError as exc:
            app.logger.error(exc.message, extra={"event": "model_load_failed", "error_code": exc.code})
    app.extensions["prediction_service"] = prediction_service

    @app.before_request
    def start_request() -> None:
        g.request_id = request.headers.get("X-Request-ID") or str(uuid4())
        g.request_started_at = perf_counter()

    @app.after_request
    def complete_request(response: Any) -> Any:
        duration_ms = round((perf_counter() - g.request_started_at) * 1000, 2)
        app.logger.info(
            "Request completed",
            extra={
                "event": "request_completed",
                "request_id": g.request_id,
                "method": request.method,
                "path": request.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Request-ID"] = g.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "font-src 'self' data:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def home() -> str:
        return render_template("index.html", model_ready=prediction_service is not None)

    @app.get("/health")
    def health() -> tuple[Any, int]:
        if prediction_service is None:
            return jsonify({"status": "degraded", "model_ready": False}), 200
        return jsonify(prediction_service.health()), 200

    @app.get("/ready")
    def ready() -> tuple[Any, int]:
        if prediction_service is None:
            return jsonify({"status": "not_ready", "model_ready": False}), 503
        return jsonify(prediction_service.health()), 200

    @app.get("/api/metadata")
    def metadata() -> tuple[Any, int]:
        if prediction_service is None:
            raise ArtifactError("The prediction model is not ready.")
        return jsonify(prediction_service.public_metadata()), 200

    @app.post("/api/predict")
    def predict() -> tuple[Any, int]:
        if prediction_service is None:
            raise ArtifactError("The prediction model is not ready.")
        if not request.is_json:
            raise UnsupportedMediaType("Content-Type must be application/json.")
        try:
            payload = request.get_json()
        except BadRequest as exc:
            raise DataValidationError("The request body contains invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise DataValidationError("The request body must be a JSON object.")
        return jsonify(prediction_service.predict(payload)), 200

    @app.errorhandler(ApplicationError)
    def handle_application_error(error: ApplicationError) -> tuple[Any, int]:
        app.logger.warning(
            error.message,
            extra={
                "event": "request_rejected",
                "request_id": getattr(g, "request_id", None),
                "error_code": error.code,
            },
        )
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(error: RequestEntityTooLarge) -> tuple[Any, int]:
        return jsonify({"error": {"code": "payload_too_large", "message": "Request body is too large."}}), 413

    @app.errorhandler(404)
    def handle_not_found(error: Any) -> tuple[Any, int]:
        return jsonify({"error": {"code": "not_found", "message": "Resource not found."}}), 404

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception) -> tuple[Any, int]:
        app.logger.exception(
            "Unhandled request error",
            extra={
                "event": "request_failed",
                "request_id": getattr(g, "request_id", None),
            },
        )
        return jsonify({"error": {"code": "internal_error", "message": "An unexpected error occurred."}}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
