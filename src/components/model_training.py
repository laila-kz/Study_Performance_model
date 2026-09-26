from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


class ModelTrainer:
    def __init__(self, random_state: int = 42, cv_folds: int = 5) -> None:
        self.random_state = random_state
        self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def _search(
        self,
        estimator: Any,
        parameter_grid: dict[str, list[Any]],
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[Any, float, dict[str, Any]]:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring="neg_mean_absolute_error",
            cv=self.cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(features, target)
        return search.best_estimator_, -float(search.best_score_), search.best_params_

    def fit_ridge(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[Ridge, float, dict[str, Any]]:
        estimator = Ridge()
        grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
        return self._search(estimator, grid, features, target)

    def fit_random_forest(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[RandomForestRegressor, float, dict[str, Any]]:
        estimator = RandomForestRegressor(random_state=self.random_state, n_jobs=1)
        grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 8, 12],
            "min_samples_leaf": [1, 2, 4],
        }
        return self._search(estimator, grid, features, target)

    def fit_xgboost(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[XGBRegressor, float, dict[str, Any]]:
        estimator = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
        )
        grid = {
            "n_estimators": [200, 400],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
        return self._search(estimator, grid, features, target)

    def fit_keras(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        epochs: int = 300,
    ) -> tuple[Any, float, int]:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.random_state)
        model = tf.keras.Sequential(
            (
                tf.keras.layers.Input(shape=(features.shape[1],)),
                tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
                tf.keras.layers.Dense(1, activation="linear"),
            ),
            name="student_performance_regressor",
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=25,
            min_delta=0.01,
            restore_best_weights=True,
        )
        history = model.fit(
            features.to_numpy(dtype=np.float32),
            target.to_numpy(dtype=np.float32),
            validation_split=0.2,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            shuffle=True,
            verbose=0,
        )
        validation_mae = float(history.history["val_mae"][-1]) if "val_mae" in history.history else float(
            min(history.history["val_loss"])
        ) ** 0.5
        return model, validation_mae, len(history.history["loss"])


def build_pass_classifier(preprocessor: ColumnTransformer) -> Pipeline:
    logistic_regression = LogisticRegression(max_iter=2000, solver="lbfgs")
    calibrated_classifier = CalibratedClassifierCV(
        estimator=logistic_regression,
        method="sigmoid",
        cv=5,
    )
    return Pipeline(
        steps=(
            ("preprocessor", clone(preprocessor)),
            ("classifier", calibrated_classifier),
        )
    )
