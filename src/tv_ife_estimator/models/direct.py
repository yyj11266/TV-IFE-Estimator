"""Direct predictive model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..constants import CORE_DIRECT_CATEGORICAL_FEATURES, CORE_DIRECT_NUMERIC_FEATURES


class DirectRidgeForecaster:
    """Regularized direct forecaster for next-month sales."""

    model_name = "direct_ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.numeric_features = list(CORE_DIRECT_NUMERIC_FEATURES)
        self.categorical_features = list(CORE_DIRECT_CATEGORICAL_FEATURES)
        self.pipeline_: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.numeric_features),
                ("categorical", categorical_pipeline, self.categorical_features),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> "DirectRidgeForecaster":
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(
            frame[self.numeric_features + self.categorical_features],
            frame["target_delta_log_sales"].astype(float),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.pipeline_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        predicted_delta = self.pipeline_.predict(frame[self.numeric_features + self.categorical_features])
        predicted_log_sales = frame["log_prev_sales"].astype(float).to_numpy() + predicted_delta
        return pd.Series(np.expm1(predicted_log_sales).clip(min=0), index=frame.index, name="y_pred")
