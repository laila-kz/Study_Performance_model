from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from src.components.data_validation import (
    FEATURE_COLUMNS,
    PASS_THRESHOLD,
    coerce_prediction_input,
)
from src.components.model_explainability import ExplainabilityService
from src.config import settings
from src.exception import ArtifactError, PredictionError
from src.logger import get_logger


class PredictionPipeline:
    def __init__(
        self,
        preprocessor: Any,
        regressor: Any,
        pass_classifier: Any,
        metadata: dict[str, Any],
        metrics: dict[str, Any],
        distributions: dict[str, Any],
        shap_importance: dict[str, Any],
        explainability: ExplainabilityService | None,
    ) -> None:
        self.preprocessor = preprocessor
        self.regressor = regressor
        self.pass_classifier = pass_classifier
        self.metadata = metadata
        self.metrics = metrics
        self.distributions = distributions
        self.shap_importance = shap_importance
        self.explainability = explainability
        self.lock = Lock()
        self.logger = get_logger(__name__)

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path | None = None) -> "PredictionPipeline":
        directory = Path(artifact_dir or settings.artifact_dir)
        required_files = (
            "preprocessor.joblib",
            "serving_model.joblib",
            "pass_classifier.joblib",
            "metadata.json",
            "metrics.json",
            "feature_distributions.json",
            "shap_feature_importance.json",
            "shap_background.joblib",
        )
        missing = [name for name in required_files if not (directory / name).is_file()]
        if missing:
            raise ArtifactError(
                "Model artifacts are missing. Run the training pipeline before serving predictions.",
                details={"missing_artifacts": missing},
            )
        try:
            preprocessor = joblib.load(directory / "preprocessor.joblib")
            regressor = joblib.load(directory / "serving_model.joblib")
            pass_classifier = joblib.load(directory / "pass_classifier.joblib")
            metadata = cls._read_json(directory / "metadata.json")
            metrics = cls._read_json(directory / "metrics.json")
            distributions = cls._read_json(directory / "feature_distributions.json")
            shap_importance = cls._read_json(directory / "shap_feature_importance.json")
            background = joblib.load(directory / "shap_background.joblib")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ArtifactError("Model artifacts could not be loaded.") from exc
        explainability = ExplainabilityService(regressor, background)
        return cls(
            preprocessor=preprocessor,
            regressor=regressor,
            pass_classifier=pass_classifier,
            metadata=metadata,
            metrics=metrics,
            distributions=distributions,
            shap_importance=shap_importance,
            explainability=explainability,
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Expected an object in {path.name}.")
        return payload

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = coerce_prediction_input(payload)
        frame = pd.DataFrame([normalized], columns=list(FEATURE_COLUMNS))
        try:
            with self.lock:
                transformed = self.preprocessor.transform(frame)
                transformed_frame = pd.DataFrame(
                    transformed,
                    columns=self.preprocessor.get_feature_names_out(),
                    index=frame.index,
                )
                raw_prediction = np.asarray(self.regressor.predict(transformed)).reshape(-1)
                if raw_prediction.size != 1:
                    raise PredictionError("The model returned an invalid prediction shape.")
                predicted_grade = float(np.clip(raw_prediction[0], 0.0, 20.0))
                pass_probabilities = np.asarray(
                    self.pass_classifier.predict_proba(frame),
                    dtype=float,
                )
                pass_probability = float(np.clip(pass_probabilities[0, 1], 0.0, 1.0))
                explanation = self._explain(transformed_frame, normalized, predicted_grade)
        except PredictionError:
            raise
        except (ValueError, TypeError, AttributeError, IndexError, KeyError) as exc:
            raise PredictionError("Prediction failed because the model received incompatible data.") from exc

        outcome = "pass" if predicted_grade >= PASS_THRESHOLD else "fail"
        return {
            "prediction_id": str(uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prediction": {
                "predicted_grade": round(predicted_grade, 2),
                "pass_probability": round(pass_probability, 4),
                "outcome": outcome,
                "confidence": round(max(pass_probability, 1.0 - pass_probability), 4),
                "threshold": PASS_THRESHOLD,
            },
            "explanation": explanation,
            "model": {
                "name": self.model_name,
                "trained_at": self.metadata.get("trained_at"),
                "data_source": self.metadata.get("data_source"),
            },
            "disclaimer": "For educational research support only; not a formal academic decision.",
        }

    def _explain(
        self,
        transformed: pd.DataFrame,
        normalized: Mapping[str, Any],
        prediction: float,
    ) -> dict[str, Any]:
        if self.explainability is None:
            return {"available": False, "contributions": [], "global_importance": self.shap_importance.get("features", [])}
        try:
            local = self.explainability.local_explanation(transformed, dict(normalized), prediction)
            return {
                "available": True,
                **local,
                "global_importance": self.shap_importance.get("features", []),
            }
        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError, KeyError) as exc:
            self.logger.warning(
                "Local SHAP explanation failed",
                extra={"event": "shap_local_failed", "model_name": self.model_name},
            )
            return {
                "available": False,
                "message": "Local explanation is temporarily unavailable.",
                "contributions": [],
                "global_importance": self.shap_importance.get("features", []),
                "reason": type(exc).__name__,
            }

    def public_metadata(self) -> dict[str, Any]:
        return {
            "model": {
                "name": self.model_name,
                "kind": self.metadata.get("model_kind"),
                "trained_at": self.metadata.get("trained_at"),
                "data_source": self.metadata.get("data_source"),
                "pass_threshold": self.metadata.get("pass_threshold", PASS_THRESHOLD),
            },
            "schema": self.metadata.get("schema", []),
            "benchmarks": self.metrics.get("models", {}),
            "serving_model": self.metrics.get("serving_model"),
            "distributions": self.distributions,
            "shap_importance": self.shap_importance.get("features", []),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_ready": True,
            "model_name": self.model_name,
            "trained_at": self.metadata.get("trained_at"),
        }
