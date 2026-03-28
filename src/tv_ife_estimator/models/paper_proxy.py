"""Paper-inspired factor model with time-varying coefficients."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from ..constants import CORE_PAPER_NUMERIC_FEATURES


def _epanechnikov_weights(distance: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(distance, dtype=float)
    inside = np.abs(distance) <= 1.0
    weights[inside] = 0.75 * (1.0 - distance[inside] ** 2)
    return weights


def _linear_extrapolation(values: np.ndarray, indices: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[-1])
    use = min(3, len(values))
    x = indices[-use:].astype(float)
    y = values[-use:].astype(float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float(y[-1])
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(intercept + slope * (x[-1] + 1.0))


def _ar1_forecast(series: np.ndarray) -> float:
    if len(series) == 0:
        return 0.0
    if len(series) == 1:
        return float(series[-1])
    x = series[:-1].astype(float)
    y = series[1:].astype(float)
    if np.allclose(x, x[0]):
        return float(series[-1])
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(intercept + slope * series[-1])


@dataclass
class PaperTrackArtifacts:
    """Artifacts exported from the paper-inspired model."""

    coefficients: pd.DataFrame
    factors: pd.DataFrame
    loadings: pd.DataFrame


class PaperInspiredFactorForecaster:
    """A pragmatic proxy for the paper's time-varying coefficient + factor idea."""

    model_name = "paper_inspired_factor"

    def __init__(
        self,
        n_factors: int = 1,
        ridge_alpha: float = 1.0,
        min_window: float = 2.0,
    ) -> None:
        self.n_factors = n_factors
        self.ridge_alpha = ridge_alpha
        self.min_window = min_window
        self.feature_columns = list(CORE_PAPER_NUMERIC_FEATURES)

        self.coefficients_: pd.DataFrame | None = None
        self.factors_: pd.DataFrame | None = None
        self.loadings_: pd.DataFrame | None = None
        self.family_index_: list[str] | None = None
        self.feature_means_: pd.Series | None = None

    def _prepare_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = frame[self.feature_columns].astype(float).copy()
        means = matrix.mean()
        if self.feature_means_ is None:
            self.feature_means_ = means
        matrix = matrix.fillna(self.feature_means_)
        return matrix

    def fit(self, frame: pd.DataFrame) -> "PaperInspiredFactorForecaster":
        if frame.empty:
            raise ValueError("Paper-inspired model cannot fit an empty frame.")

        train = frame.copy().sort_values(["source_month_index", "family_id"]).reset_index(drop=True)
        x_matrix = self._prepare_matrix(train)
        design_matrix = np.column_stack([np.ones(len(train)), x_matrix.to_numpy()])

        month_indices = train["source_month_index"].astype(int).to_numpy()
        unique_month_indices = np.array(sorted(train["source_month_index"].dropna().unique().astype(int)))
        family_ids = sorted(train["family_id"].unique())
        n_families = len(family_ids)
        n_months = len(unique_month_indices)

        bandwidth = max(
            self.min_window,
            (2.35 / math.sqrt(12.0)) * ((max(n_families * n_months, 1)) ** (-0.25)) * max(n_months, 1),
        )

        beta_rows: list[dict[str, float]] = []
        beta_lookup: dict[int, np.ndarray] = {}
        target = train["target_log_sales"].astype(float).to_numpy()

        for month_index in unique_month_indices:
            distances = (month_indices - month_index) / bandwidth
            weights = _epanechnikov_weights(distances)
            if float(weights.sum()) <= 0:
                weights = np.ones_like(weights)
            local_model = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            local_model.fit(design_matrix, target, sample_weight=weights)
            beta = local_model.coef_.astype(float)
            beta_lookup[int(month_index)] = beta
            beta_rows.append(
                {
                    "source_month_index": int(month_index),
                    "intercept": float(beta[0]),
                    "beta_current_log_asp": float(beta[1]),
                    "beta_current_discount": float(beta[2]),
                }
            )

        fitted = np.array([design_matrix[i] @ beta_lookup[int(month_indices[i])] for i in range(len(train))])
        residual = target - fitted

        residual_panel = train[["source_month", "source_month_index", "family_id"]].copy()
        residual_panel["residual"] = residual
        residual_matrix = (
            residual_panel.pivot(index="source_month", columns="family_id", values="residual")
            .reindex(columns=family_ids)
            .sort_index()
        )
        residual_matrix = residual_matrix.fillna(0.0)

        max_rank = min(residual_matrix.shape)
        usable_factors = min(max(self.n_factors, 0), max(max_rank - 1, 0))
        factor_rows: list[dict[str, float]] = []
        loading_rows: list[dict[str, float]] = []
        family_loading_map = {family_id: np.zeros(usable_factors, dtype=float) for family_id in family_ids}

        if usable_factors > 0 and residual_matrix.shape[0] > 1:
            pca = PCA(n_components=usable_factors)
            factor_values = pca.fit_transform(residual_matrix.to_numpy())
            loadings = pca.components_.T

            for row_index, source_month in enumerate(residual_matrix.index):
                row = {
                    "source_month": source_month,
                    "source_month_index": int(
                        train.loc[train["source_month"] == source_month, "source_month_index"].iloc[0]
                    ),
                }
                for factor_index in range(usable_factors):
                    row[f"factor_{factor_index + 1}"] = float(factor_values[row_index, factor_index])
                factor_rows.append(row)

            for family_position, family_id in enumerate(family_ids):
                row = {"family_id": family_id}
                family_loading_map[family_id] = loadings[family_position]
                for factor_index in range(usable_factors):
                    row[f"loading_{factor_index + 1}"] = float(loadings[family_position, factor_index])
                loading_rows.append(row)
        else:
            for source_month in residual_matrix.index:
                factor_rows.append(
                    {
                        "source_month": source_month,
                        "source_month_index": int(
                            train.loc[train["source_month"] == source_month, "source_month_index"].iloc[0]
                        ),
                    }
                )
            for family_id in family_ids:
                loading_rows.append({"family_id": family_id})

        self.coefficients_ = pd.DataFrame(beta_rows).sort_values("source_month_index").reset_index(drop=True)
        self.factors_ = pd.DataFrame(factor_rows).sort_values("source_month_index").reset_index(drop=True)
        self.loadings_ = pd.DataFrame(loading_rows).sort_values("family_id").reset_index(drop=True)
        self.family_index_ = family_ids
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.coefficients_ is None or self.factors_ is None or self.loadings_ is None:
            raise RuntimeError("Model must be fitted before prediction.")

        forecast = frame.copy().sort_values("family_id").reset_index(drop=True)
        x_matrix = forecast[self.feature_columns].astype(float).fillna(self.feature_means_)
        design_matrix = np.column_stack([np.ones(len(forecast)), x_matrix.to_numpy()])

        coefficient_indices = self.coefficients_["source_month_index"].to_numpy(dtype=float)
        coefficient_matrix = self.coefficients_[["intercept", "beta_current_log_asp", "beta_current_discount"]].to_numpy(
            dtype=float
        )
        beta_next = np.array(
            [
                _linear_extrapolation(coefficient_matrix[:, column], coefficient_indices)
                for column in range(coefficient_matrix.shape[1])
            ]
        )

        factor_columns = [column for column in self.factors_.columns if column.startswith("factor_")]
        if factor_columns:
            factor_indices = self.factors_["source_month_index"].to_numpy(dtype=float)
            factor_matrix = self.factors_[factor_columns].to_numpy(dtype=float)
            factor_next = np.array([_ar1_forecast(factor_matrix[:, column]) for column in range(factor_matrix.shape[1])])
            loading_columns = [column for column in self.loadings_.columns if column.startswith("loading_")]
            loading_frame = self.loadings_.set_index("family_id")[loading_columns]
            factor_term = np.array(
                [
                    float(np.dot(loading_frame.loc[row["family_id"]].to_numpy(dtype=float), factor_next))
                    if row["family_id"] in loading_frame.index
                    else 0.0
                    for _, row in forecast.iterrows()
                ]
            )
        else:
            factor_next = np.array([], dtype=float)
            factor_term = np.zeros(len(forecast), dtype=float)

        predicted_log_sales = design_matrix @ beta_next + factor_term
        return pd.Series(np.expm1(predicted_log_sales).clip(min=0), index=forecast.index, name="y_pred")

    def export_artifacts(self) -> PaperTrackArtifacts:
        if self.coefficients_ is None or self.factors_ is None or self.loadings_ is None:
            raise RuntimeError("Model must be fitted before exporting artifacts.")
        return PaperTrackArtifacts(
            coefficients=self.coefficients_.copy(),
            factors=self.factors_.copy(),
            loadings=self.loadings_.copy(),
        )
