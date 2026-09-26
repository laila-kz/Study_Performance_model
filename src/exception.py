from typing import Any


class ApplicationError(Exception):
    status_code = 500
    code = "application_error"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.details:
            payload["error"]["details"] = self.details
        return payload


class DataIngestionError(ApplicationError):
    status_code = 500
    code = "data_ingestion_error"


class DataValidationError(ApplicationError):
    status_code = 422
    code = "validation_error"


class ArtifactError(ApplicationError):
    status_code = 503
    code = "model_unavailable"


class PredictionError(ApplicationError):
    status_code = 500
    code = "prediction_error"
