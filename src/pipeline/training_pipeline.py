from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PASS_THRESHOLD,
    PREDICTION_FIELDS,
    TARGET_COLUMN,
)
from src.components.data_transformation import DataTransformation
from src.components.model_explainability import ExplainabilityService
from src.components.model_training import ModelTrainer, build_pass_classifier, regression_metrics
from src.config import settings
from src.exception import ApplicationError
from src.logger import get_logger


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


class TrainingPipeline:
    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        data_url: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.artifact_dir = Path(artifact_dir or settings.artifact_dir)
        self.data_url = data_url or settings.data_url
        self.random_state = random_state
        self.logger = get_logger(__name__)

    def run(self, data_path: str | Path | None = None) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        models_dir = self.artifact_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        ingestion = DataIngestion(self.data_url)
        features, target = ingestion.load(data_path)
        train_features, test_features, train_target, test_target = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=self.random_state,
        )

        transformation = DataTransformation()
        preprocessor, transformed_train = transformation.fit_transform(train_features)
        transformed_test = transformation.transform(preprocessor, test_features)
        joblib.dump(preprocessor, self.artifact_dir / "preprocessor.joblib", compress=3)

        trainer = ModelTrainer(random_state=self.random_state)
        fitted_models: dict[str, Any] = {}
        benchmark: dict[str, dict[str, Any]] = {}
        estimators: dict[str, Any] = {}

        classical_fitters = {
            "ridge": trainer.fit_ridge,
            "random_forest": trainer.fit_random_forest,
            "xgboost": trainer.fit_xgboost,
        }
        for name, fitter in classical_fitters.items():
            started_at = perf_counter()
            estimator, cv_mae, best_parameters = fitter(transformed_train, train_target)
            predictions = np.asarray(estimator.predict(transformed_test), dtype=float).reshape(-1)
            metrics = regression_metrics(test_target.to_numpy(), predictions)
            metrics["cv_mae"] = round(cv_mae, 6)
            metrics["training_seconds"] = round(perf_counter() - started_at, 3)
            benchmark[name] = {
                **metrics,
                "best_parameters": best_parameters,
            }
            fitted_models[name] = estimator
            estimators[name] = estimator
            joblib.dump(estimator, models_dir / f"{name}.joblib", compress=3)
            self.logger.info("Model benchmark completed", extra={"event": "model_benchmark", "model_name": name})

        started_at = perf_counter()
        keras_model, keras_validation_mae, epochs = trainer.fit_keras(transformed_train, train_target)
        keras_predictions = np.asarray(
            keras_model.predict(transformed_test, verbose=0),
            dtype=float,
        ).reshape(-1)
        keras_metrics = regression_metrics(test_target.to_numpy(), keras_predictions)
        benchmark["keras"] = {
            **keras_metrics,
            "cv_mae": round(keras_validation_mae, 6),
            "training_seconds": round(perf_counter() - started_at, 3),
            "epochs": epochs,
        }
        keras_model.save(self.artifact_dir / "models" / "keras.keras")

        serving_model_name = min(
            ("ridge", "random_forest", "xgboost"),
            key=lambda name: benchmark[name]["cv_mae"],
        )
        serving_model = estimators[serving_model_name]
        joblib.dump(serving_model, self.artifact_dir / "serving_model.joblib", compress=3)

        pass_classifier = build_pass_classifier(preprocessor)
        pass_classifier.fit(train_features, (train_target >= PASS_THRESHOLD).astype(int))
        joblib.dump(pass_classifier, self.artifact_dir / "pass_classifier.joblib", compress=3)

        background = transformed_train.sample(
            n=min(100, len(transformed_train)),
            random_state=self.random_state,
        )
        joblib.dump(background, self.artifact_dir / "shap_background.joblib", compress=3)
        explainability = ExplainabilityService(serving_model, background)
        importance_sample = transformed_test.sample(
            n=min(100, len(transformed_test)),
            random_state=self.random_state,
        )
        global_importance = explainability.global_importance(importance_sample)
        _write_json(self.artifact_dir / "shap_feature_importance.json", {"features": global_importance})

        trained_at = datetime.now(timezone.utc).isoformat()
        metrics_payload = {
            "trained_at": trained_at,
            "test_size": 0.2,
            "random_state": self.random_state,
            "serving_model": serving_model_name,
            "models": benchmark,
        }
        _write_json(self.artifact_dir / "metrics.json", metrics_payload)

        schema = [
            {
                "name": field.name,
                "label": field.label,
                "kind": field.kind,
                "section": field.section,
                "minimum": field.minimum,
                "maximum": field.maximum,
                "choices": list(field.choices),
            }
            for field in PREDICTION_FIELDS
        ]
        metadata_payload = {
            "schema_version": "1.0.0",
            "trained_at": trained_at,
            "model_name": serving_model_name,
            "model_kind": serving_model_name,
            "pass_threshold": PASS_THRESHOLD,
            "data_source": self.data_url,
            "feature_columns": list(FEATURE_COLUMNS),
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "schema": schema,
            "versions": {
                "python": platform.python_version(),
                "numpy": _package_version("numpy"),
                "pandas": _package_version("pandas"),
                "scikit_learn": _package_version("scikit-learn"),
                "xgboost": _package_version("xgboost"),
                "tensorflow": _package_version("tensorflow"),
                "shap": _package_version("shap"),
            },
        }
        _write_json(self.artifact_dir / "metadata.json", metadata_payload)
        _write_json(
            self.artifact_dir / "feature_distributions.json",
            self._build_feature_distributions(features, target),
        )
        self.logger.info(
            "Training pipeline completed",
            extra={"event": "training_completed", "model_name": serving_model_name},
        )
        return metrics_payload

    @staticmethod
    def _build_feature_distributions(
        features: pd.DataFrame,
        target: pd.Series,
    ) -> dict[str, Any]:
        distributions: dict[str, Any] = {
            "labels": list(range(21)),
            "series": {},
        }
        values_by_name = {
            "G1": features["G1"].to_numpy(dtype=float),
            "G2": features["G2"].to_numpy(dtype=float),
            "G3": target.to_numpy(dtype=float),
        }
        bins = np.arange(-0.5, 21.5, 1)
        for name, values in values_by_name.items():
            counts, _ = np.histogram(values, bins=bins)
            distributions["series"][name] = {
                "counts": counts.astype(int).tolist(),
                "median": round(float(np.median(values)), 3),
                "mean": round(float(np.mean(values)), 3),
            }
        return distributions


def main() -> None:
    pipeline = TrainingPipeline()
    try:
        summary = pipeline.run()
    except ApplicationError as exc:
        pipeline.logger.error("Training pipeline failed", extra={"event": "training_failed", "error_code": exc.code})
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
