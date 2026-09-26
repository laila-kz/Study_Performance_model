from pathlib import Path

import pandas as pd

from src.components.data_validation import TARGET_COLUMN, validate_training_frame
from src.exception import DataIngestionError
from src.logger import get_logger


class DataIngestion:
    def __init__(self, data_url: str) -> None:
        self.data_url = data_url
        self.logger = get_logger(__name__)

    def load(self, data_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
        source = str(data_path) if data_path is not None else self.data_url
        self.logger.info("Loading student performance data", extra={"event": "data_load_started", "path": source})
        try:
            frame = pd.read_csv(source, sep=";")
            if len(frame.columns) == 1:
                frame = pd.read_csv(source, sep=",")
        except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
            raise DataIngestionError(
                "Unable to load the UCI Student Performance dataset.",
                details={"source": source},
            ) from exc

        if TARGET_COLUMN not in frame.columns:
            raise DataIngestionError(
                "The dataset does not contain the expected G3 target.",
                details={"source": source, "columns": frame.columns.tolist()},
            )

        features, target = validate_training_frame(frame)
        self.logger.info(
            "Student performance data loaded",
            extra={"event": "data_load_completed", "rows": len(features), "columns": len(features.columns)},
        )
        return features, target
