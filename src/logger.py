from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any

LOGGER_NAME = "student_performance"
STANDARD_FIELDS = {
    "event",
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "model_name",
    "error_code",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in STANDARD_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )


def configure_logging(level: str = "INFO", output_format: str = "json") -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(TextFormatter() if output_format == "text" else JsonFormatter())
    logger.addHandler(handler)


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or LOGGER_NAME)
