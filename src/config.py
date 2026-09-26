from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    artifact_dir: Path
    data_url: str
    host: str
    port: int
    debug: bool
    log_level: str
    log_format: str
    max_content_length: int

    @classmethod
    def from_env(cls) -> "Settings":
        root = PROJECT_ROOT
        return cls(
            project_root=root,
            artifact_dir=Path(os.getenv("ARTIFACT_DIR", root / "artifacts")),
            data_url=os.getenv(
                "DATA_URL",
                "https://archive.ics.uci.edu/static/public/320/data.csv",
            ),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "3000")),
            debug=_as_bool(os.getenv("DEBUG"), default=False),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_format=os.getenv("LOG_FORMAT", "json").lower(),
            max_content_length=int(os.getenv("MAX_CONTENT_LENGTH", "65536")),
        )


settings = Settings.from_env()
