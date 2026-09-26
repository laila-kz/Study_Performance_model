import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components.data_validation import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class DataTransformation:
    def create_preprocessor(self) -> ColumnTransformer:
        numeric_pipeline = StandardScaler()
        categorical_pipeline = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype="float64",
        )
        return ColumnTransformer(
            transformers=(
                ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
                ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
            ),
            remainder="drop",
            verbose_feature_names_out=True,
        )

    def fit_transform(
        self,
        features: pd.DataFrame,
    ) -> tuple[ColumnTransformer, pd.DataFrame]:
        preprocessor = self.create_preprocessor()
        transformed = preprocessor.fit_transform(features)
        transformed_frame = pd.DataFrame(
            transformed,
            columns=preprocessor.get_feature_names_out(),
            index=features.index,
        )
        return preprocessor, transformed_frame

    def transform(
        self,
        preprocessor: ColumnTransformer,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        transformed = preprocessor.transform(features)
        return pd.DataFrame(
            transformed,
            columns=preprocessor.get_feature_names_out(),
            index=features.index,
        )
