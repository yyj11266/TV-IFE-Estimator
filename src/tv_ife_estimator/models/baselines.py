"""Baseline forecasting models."""

from __future__ import annotations

import pandas as pd


class NaiveLastMonthModel:
    """Last-month sales benchmark."""

    model_name = "naive_last_month"

    def fit(self, frame: pd.DataFrame) -> "NaiveLastMonthModel":
        self._fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Model must be fitted before prediction.")
        return frame["prev_sales"].clip(lower=0).astype(float)
