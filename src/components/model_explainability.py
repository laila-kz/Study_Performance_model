from typing import Any

import numpy as np
import pandas as pd
import shap

from src.components.data_validation import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.logger import get_logger


class ExplainabilityService:
    def __init__(self, model: Any, background: pd.DataFrame) -> None:
        self.model = model
        self.background = background
        self.logger = get_logger(__name__)
        self.explainer = self._build_explainer()

    def _build_explainer(self) -> Any:
        feature_names = self.background.columns.tolist()
        try:
            return shap.TreeExplainer(
                self.model,
                data=self.background,
                feature_names=feature_names,
            )
        except (ValueError, TypeError, AttributeError, NotImplementedError) as exc:
            self.logger.warning(
                "Tree explainer unavailable; using the SHAP model explainer",
                extra={"event": "shap_explainer_fallback"},
            )
            try:
                return shap.Explainer(self.model.predict, self.background)
            except (ValueError, TypeError, AttributeError, NotImplementedError) as fallback_exc:
                raise RuntimeError("Unable to initialize a SHAP explainer.") from fallback_exc

    def _source_feature(self, transformed_name: str) -> str:
        if transformed_name.startswith("numeric__"):
            return transformed_name.removeprefix("numeric__")
        if transformed_name.startswith("categorical__"):
            encoded = transformed_name.removeprefix("categorical__")
            candidates = [
                feature
                for feature in CATEGORICAL_FEATURES
                if encoded.startswith(f"{feature}_")
            ]
            if candidates:
                return max(candidates, key=len)
        return transformed_name

    def _aggregate(
        self,
        transformed_names: list[str],
        values: np.ndarray,
    ) -> dict[str, float]:
        aggregate = {feature: 0.0 for feature in FEATURE_COLUMNS}
        for transformed_name, value in zip(transformed_names, values, strict=True):
            aggregate[self._source_feature(transformed_name)] += float(value)
        return aggregate

    def _raw_explanation(self, transformed: pd.DataFrame) -> Any:
        return self.explainer(transformed)

    def local_explanation(
        self,
        transformed: pd.DataFrame,
        input_values: dict[str, Any],
        prediction: float,
        limit: int = 12,
    ) -> dict[str, Any]:
        explanation = self._raw_explanation(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[..., 0]
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[0] != 1:
            raise RuntimeError("SHAP returned an unexpected explanation shape.")
        transformed_names = transformed.columns.tolist()
        aggregate = self._aggregate(transformed_names, values[0])
        ranked = sorted(aggregate.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]
        contributions = [
            {
                "feature": feature,
                "value": input_values.get(feature),
                "contribution": round(contribution, 6),
            }
            for feature, contribution in ranked
        ]
        base_value = self._base_value(explanation)
        reconstructed = base_value + sum(aggregate.values())
        return {
            "base_value": round(base_value, 6),
            "prediction": round(prediction, 6),
            "reconstructed_prediction": round(reconstructed, 6),
            "additivity_error": round(abs(prediction - reconstructed), 6),
            "contributions": contributions,
        }

    def global_importance(
        self,
        transformed: pd.DataFrame,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        explanation = self._raw_explanation(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[..., 0]
        mean_absolute = np.abs(values).mean(axis=0)
        aggregate = self._aggregate(transformed.columns.tolist(), mean_absolute)
        ranked = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            {"feature": feature, "mean_abs_shap": round(importance, 6)}
            for feature, importance in ranked
        ]

    @staticmethod
    def _base_value(explanation: Any) -> float:
        base_values = getattr(
            explanation,
            "base_values",
            getattr(explanation, "expected_value", 0.0),
        )
        array = np.asarray(base_values, dtype=float).reshape(-1)
        return float(array[0])
