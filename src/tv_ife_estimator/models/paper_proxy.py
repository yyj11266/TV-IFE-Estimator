"""Local LLS + local PCA forecaster with IC-based factor selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..constants import CORE_PAPER_NUMERIC_FEATURES

_EPSILON = 1e-8


def _epanechnikov_weights(distance: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(distance, dtype=float)
    inside = np.abs(distance) <= 1.0
    weights[inside] = 0.75 * (1.0 - distance[inside] ** 2)
    total = float(weights.sum())
    if total <= 0:
        return np.full_like(distance, 1.0 / max(len(distance), 1), dtype=float)
    return weights / total


def _safe_inverse(matrix: np.ndarray, ridge: float = _EPSILON) -> np.ndarray:
    if matrix.size == 0:
        return matrix.copy()
    eye = np.eye(matrix.shape[0], dtype=float)
    return np.linalg.pinv(matrix + ridge * eye)


def _bounded_linear_extrapolation(values: np.ndarray, indices: np.ndarray) -> float:
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
    prediction = float(intercept + slope * (x[-1] + 1.0))
    y_min = float(y.min())
    y_max = float(y.max())
    span = max(y_max - y_min, 1e-6)
    return float(np.clip(prediction, y_min - 0.5 * span, y_max + 0.5 * span))


def _bounded_ar1_forecast(series: np.ndarray) -> float:
    if len(series) == 0:
        return 0.0
    if len(series) == 1:
        return float(series[-1])

    x = series[:-1].astype(float)
    y = series[1:].astype(float)
    if np.allclose(x, x[0]):
        return float(series[-1])

    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = float(intercept + slope * series[-1])
    series_min = float(series.min())
    series_max = float(series.max())
    span = max(series_max - series_min, 1e-6)
    return float(np.clip(prediction, series_min - 0.5 * span, series_max + 0.5 * span))


@dataclass
class _TVIFEPanel:
    y: np.ndarray
    design: np.ndarray
    source_months: list[str]
    source_month_index: np.ndarray
    family_ids: list[str]
    feature_means: np.ndarray
    bandwidth: float
    kernel_weights: np.ndarray
    max_target_log_sales: float

    @property
    def n_months(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_families(self) -> int:
        return int(self.y.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.design.shape[2])


@dataclass
class _TVIFEFitState:
    factor_count: int
    beta_hat: np.ndarray
    beta_bc: np.ndarray
    factors: np.ndarray
    loadings: np.ndarray
    fitted: np.ndarray
    residual_variance: float
    ic_rho1: float
    ic_rho2: float


@dataclass
class PaperTrackArtifacts:
    """Artifacts exported from the paper track."""

    coefficients: pd.DataFrame
    factors: pd.DataFrame
    loadings: pd.DataFrame
    selection: pd.DataFrame
    fitted: pd.DataFrame
    fit_summary: pd.DataFrame


class PaperInspiredFactorForecaster:
    """A fuller TV-IFE estimator based on local LLS + local PCA + IC."""

    model_name = "paper_inspired_factor"

    def __init__(
        self,
        n_factors: int | None = None,
        factor_candidates: tuple[int, ...] = (),
        max_factor_count: int = 6,
        ic_penalty_variant: str = "rho2",
        ridge_alpha: float = 1e-6,
        min_window: float = 2.0,
        max_iter: int = 50,
        tolerance: float = 1e-6,
    ) -> None:
        self.n_factors = n_factors
        self.factor_candidates = tuple(sorted({int(value) for value in factor_candidates if int(value) >= 0}))
        self.max_factor_count = max_factor_count
        self.ic_penalty_variant = ic_penalty_variant
        self.ridge_alpha = ridge_alpha
        self.min_window = min_window
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.feature_columns = list(CORE_PAPER_NUMERIC_FEATURES)

        self.panel_: _TVIFEPanel | None = None
        self.state_: _TVIFEFitState | None = None
        self.selection_table_: pd.DataFrame | None = None
        self.selected_factor_count_: int | None = None

    def _prepare_panel(self, frame: pd.DataFrame) -> _TVIFEPanel:
        train = frame.copy().sort_values(["source_month_index", "family_id"]).reset_index(drop=True)
        family_ids = sorted(train["family_id"].astype(str).unique())
        source_month_index = np.array(sorted(train["source_month_index"].dropna().astype(int).unique()), dtype=float)

        month_lookup = (
            train[["source_month_index", "source_month"]]
            .drop_duplicates(subset=["source_month_index"])
            .sort_values("source_month_index")
        )
        source_months = month_lookup["source_month"].astype(str).tolist()

        feature_matrices: list[np.ndarray] = []
        feature_means: list[float] = []
        for feature_name in self.feature_columns:
            matrix = (
                train.pivot(index="source_month_index", columns="family_id", values=feature_name)
                .reindex(index=source_month_index, columns=family_ids)
                .to_numpy(dtype=float)
            )
            feature_mean = float(np.nanmean(matrix)) if np.isfinite(np.nanmean(matrix)) else 0.0
            feature_matrices.append(np.where(np.isnan(matrix), feature_mean, matrix))
            feature_means.append(feature_mean)

        y = (
            train.pivot(index="source_month_index", columns="family_id", values="target_log_sales")
            .reindex(index=source_month_index, columns=family_ids)
            .to_numpy(dtype=float)
        )
        if np.isnan(y).any():
            raise ValueError("Paper track requires a balanced target panel without missing values.")

        x = np.stack(feature_matrices, axis=2)
        intercept = np.ones((x.shape[0], x.shape[1], 1), dtype=float)
        design = np.concatenate([intercept, x], axis=2)

        sigma_hat = float(np.std(y)) if y.size else 0.0
        bandwidth = max(
            self.min_window / max(len(source_month_index), 1),
            (2.35 / math.sqrt(12.0)) * max(sigma_hat, 1.0) * (max(len(source_month_index) * len(family_ids), 1) ** (-2.0 / 9.0)),
        )

        positions = np.arange(len(source_month_index), dtype=float)
        kernel_weights = np.vstack(
            [
                _epanechnikov_weights((positions - target_index) / max(len(source_month_index) * bandwidth, _EPSILON))
                for target_index in positions
            ]
        )

        return _TVIFEPanel(
            y=y,
            design=design,
            source_months=source_months,
            source_month_index=source_month_index,
            family_ids=family_ids,
            feature_means=np.array(feature_means, dtype=float),
            bandwidth=bandwidth,
            kernel_weights=kernel_weights,
            max_target_log_sales=float(np.max(y)),
        )

    def _candidate_factor_counts(self, panel: _TVIFEPanel) -> list[int]:
        if self.n_factors is not None:
            return [max(int(self.n_factors), 0)]

        max_rank = max(min(panel.n_months, panel.n_families) - 1, 0)
        effective_cap = max(int(math.floor(panel.n_months * panel.bandwidth)) - 1, 0)
        derived_cap = min(max_rank, effective_cap, self.max_factor_count)
        if self.factor_candidates:
            valid = sorted({value for value in self.factor_candidates if value <= derived_cap})
            if 0 not in valid:
                valid = [0, *valid]
            return valid or [0]
        return list(range(0, derived_cap + 1))

    def _update_beta(self, weighted_design: np.ndarray, weighted_y: np.ndarray, projection_residual: np.ndarray) -> np.ndarray:
        lhs = np.zeros((weighted_design.shape[2], weighted_design.shape[2]), dtype=float)
        rhs = np.zeros(weighted_design.shape[2], dtype=float)
        for family_index in range(weighted_design.shape[1]):
            x_i = weighted_design[:, family_index, :]
            y_i = weighted_y[:, family_index]
            lhs += x_i.T @ projection_residual @ x_i
            rhs += x_i.T @ projection_residual @ y_i
        return _safe_inverse(lhs, ridge=self.ridge_alpha) @ rhs

    def _second_stage_factors(self, panel: _TVIFEPanel, beta: np.ndarray, loadings: np.ndarray) -> np.ndarray:
        if loadings.shape[2] == 0:
            return np.zeros((panel.n_months, 0), dtype=float)

        factors = np.zeros((panel.n_months, loadings.shape[2]), dtype=float)
        for month_index in range(panel.n_months):
            lambda_t = loadings[month_index]
            lambda_gram = _safe_inverse(lambda_t.T @ lambda_t, ridge=self.ridge_alpha)
            fitted_regression = np.einsum("nk,k->n", panel.design[month_index], beta[month_index])
            residual_t = panel.y[month_index] - fitted_regression
            factors[month_index] = lambda_gram @ lambda_t.T @ residual_t
        return factors

    def _compute_fitted(self, panel: _TVIFEPanel, beta: np.ndarray, loadings: np.ndarray, factors: np.ndarray) -> np.ndarray:
        regression_fit = np.einsum("tnk,tk->tn", panel.design, beta)
        if loadings.shape[2] == 0:
            return regression_fit
        common_component = np.einsum("tnr,tr->tn", loadings, factors)
        return regression_fit + common_component

    def _bias_correct_coefficients(
        self,
        panel: _TVIFEPanel,
        beta_hat: np.ndarray,
        loadings: np.ndarray,
        factors: np.ndarray,
    ) -> np.ndarray:
        factor_count = loadings.shape[2]
        if factor_count == 0:
            return beta_hat.copy()

        common_component = np.einsum("tnr,tr->tn", loadings, factors)
        residual = panel.y - np.einsum("tnk,tk->tn", panel.design, beta_hat) - common_component
        beta_bc = beta_hat.copy()
        truncation = max(1, min(panel.n_months - 1, int(round(math.sqrt(max(panel.n_months * panel.bandwidth, 1.0))))))

        for target_index in range(panel.n_months):
            weights = panel.kernel_weights[target_index]
            sqrt_weights = np.sqrt(weights)

            design_weighted = panel.design * sqrt_weights[:, None, None]
            factor_weighted = factors * sqrt_weights[:, None]

            gram_f = (factor_weighted.T @ factor_weighted) / panel.n_months
            projection_f = factor_weighted @ _safe_inverse(gram_f, ridge=self.ridge_alpha) @ factor_weighted.T / panel.n_months
            residual_projection = np.eye(panel.n_months, dtype=float) - projection_f

            lambda_t = loadings[target_index]
            lambda_gram = _safe_inverse((lambda_t.T @ lambda_t) / panel.n_families, ridge=self.ridge_alpha)
            a_matrix = lambda_t @ lambda_gram @ lambda_t.T
            v_weighted = np.einsum("ij,tjk->tik", a_matrix / panel.n_families, design_weighted)
            z_weighted = np.zeros_like(design_weighted)
            for family_index in range(panel.n_families):
                z_weighted[:, family_index, :] = residual_projection @ (design_weighted[:, family_index, :] - v_weighted[:, family_index, :])

            d_hat = np.zeros((panel.n_features, panel.n_features), dtype=float)
            for family_index in range(panel.n_families):
                z_i = z_weighted[:, family_index, :]
                d_hat += z_i.T @ z_i
            d_hat /= panel.n_families * panel.n_months
            d_hat_inv = _safe_inverse(d_hat, ridge=self.ridge_alpha)

            b2_hat = np.zeros(panel.n_features, dtype=float)
            weighted_cov = np.zeros((panel.n_families, panel.n_families), dtype=float)
            weighted_residual = residual * sqrt_weights[:, None]
            for family_i in range(panel.n_families):
                e_i = weighted_residual[:, family_i]
                for row_index in range(panel.n_months - 1):
                    upper = min(panel.n_months, row_index + truncation + 1)
                    for column_index in range(row_index + 1, upper):
                        b2_hat -= (
                            panel.bandwidth
                            / panel.n_families
                            * projection_f[row_index, column_index]
                            * e_i[row_index]
                            * design_weighted[column_index, family_i, :]
                        )
            for family_i in range(panel.n_families):
                for family_j in range(panel.n_families):
                    weighted_cov[family_i, family_j] = (
                        np.sum(weights * residual[:, family_i] * residual[:, family_j]) / panel.n_months
                    )

            b3_hat = np.zeros(panel.n_features, dtype=float)
            for family_i in range(panel.n_families):
                x_minus_v = design_weighted[:, family_i, :] - v_weighted[:, family_i, :]
                temp = (x_minus_v.T @ factor_weighted) / panel.n_months
                for family_j in range(panel.n_families):
                    b3_hat -= (
                        (temp @ lambda_gram @ lambda_t[family_j])
                        * weighted_cov[family_i, family_j]
                        / panel.n_families
                    )

            b4_hat = np.zeros(panel.n_features, dtype=float)
            for family_i in range(panel.n_families):
                x_i = design_weighted[:, family_i, :]
                for family_j in range(panel.n_families):
                    e_j = weighted_residual[:, family_j]
                    temp = x_i.T @ residual_projection @ np.outer(e_j, e_j) @ factor_weighted
                    b4_hat -= (
                        panel.bandwidth
                        / (panel.n_families * panel.n_families * panel.n_months)
                        * temp
                        @ lambda_gram
                        @ lambda_t[family_i]
                    )

            correction = d_hat_inv @ (
                b2_hat / max(panel.n_months * panel.bandwidth, _EPSILON)
                + b3_hat / panel.n_families
                + b4_hat / max(panel.n_months * panel.bandwidth, _EPSILON)
            )
            beta_candidate = beta_hat[target_index] - correction
            if np.all(np.isfinite(beta_candidate)):
                beta_bc[target_index] = beta_candidate

        return beta_bc

    def _information_criteria(self, residual_variance: float, factor_count: int, panel: _TVIFEPanel) -> tuple[float, float]:
        effective_time = max(panel.n_months * panel.bandwidth, 1.0)
        rho_nt_1 = (
            (panel.n_families + effective_time)
            / (panel.n_families * effective_time)
            * math.log(max(panel.n_families * effective_time / (panel.n_families + effective_time), 1.0 + _EPSILON))
        )
        c_nt = min(math.sqrt(effective_time), math.sqrt(panel.n_families), panel.bandwidth ** -2)
        rho_nt_2 = (
            (panel.n_families + effective_time)
            / (panel.n_families * effective_time)
            * math.log(max(c_nt ** 2, 1.0 + _EPSILON))
        )
        log_variance = math.log(max(residual_variance, 1e-12))
        return (
            log_variance + factor_count * rho_nt_1,
            log_variance + factor_count * rho_nt_2,
        )

    def _fit_fixed_factor_count(self, panel: _TVIFEPanel, factor_count: int) -> _TVIFEFitState:
        beta_hat = np.zeros((panel.n_months, panel.n_features), dtype=float)
        loadings = np.zeros((panel.n_months, panel.n_families, factor_count), dtype=float)

        for target_index in range(panel.n_months):
            sqrt_weights = np.sqrt(panel.kernel_weights[target_index])
            weighted_design = panel.design * sqrt_weights[:, None, None]
            weighted_y = panel.y * sqrt_weights[:, None]

            beta_current = self._update_beta(
                weighted_design=weighted_design,
                weighted_y=weighted_y,
                projection_residual=np.eye(panel.n_months, dtype=float),
            )

            local_factors = np.zeros((panel.n_months, factor_count), dtype=float)
            for _ in range(self.max_iter):
                fitted = np.einsum("tnk,k->tn", weighted_design, beta_current)
                residual = weighted_y - fitted

                if factor_count > 0:
                    u_matrix, _, _ = np.linalg.svd(residual, full_matrices=False)
                    local_factors = math.sqrt(panel.n_months) * u_matrix[:, :factor_count]
                    projection_residual = np.eye(panel.n_months, dtype=float) - local_factors @ local_factors.T / panel.n_months
                else:
                    projection_residual = np.eye(panel.n_months, dtype=float)

                beta_next = self._update_beta(
                    weighted_design=weighted_design,
                    weighted_y=weighted_y,
                    projection_residual=projection_residual,
                )
                if float(np.max(np.abs(beta_next - beta_current))) <= self.tolerance:
                    beta_current = beta_next
                    break
                beta_current = beta_next

            fitted = np.einsum("tnk,k->tn", weighted_design, beta_current)
            residual = weighted_y - fitted
            if factor_count > 0:
                u_matrix, _, _ = np.linalg.svd(residual, full_matrices=False)
                local_factors = math.sqrt(panel.n_months) * u_matrix[:, :factor_count]
                loadings[target_index] = residual.T @ local_factors / panel.n_months
            beta_hat[target_index] = beta_current

        factors = self._second_stage_factors(panel, beta_hat, loadings)
        beta_bc = self._bias_correct_coefficients(panel, beta_hat, loadings, factors)
        factors_bc = self._second_stage_factors(panel, beta_bc, loadings)
        fitted = self._compute_fitted(panel, beta_bc, loadings, factors_bc)
        residual_variance = float(np.mean((panel.y - fitted) ** 2))
        ic_rho1, ic_rho2 = self._information_criteria(
            residual_variance=residual_variance,
            factor_count=factor_count,
            panel=panel,
        )

        return _TVIFEFitState(
            factor_count=factor_count,
            beta_hat=beta_hat,
            beta_bc=beta_bc,
            factors=factors_bc,
            loadings=loadings,
            fitted=fitted,
            residual_variance=residual_variance,
            ic_rho1=ic_rho1,
            ic_rho2=ic_rho2,
        )

    def fit(self, frame: pd.DataFrame) -> "PaperInspiredFactorForecaster":
        if frame.empty:
            raise ValueError("Paper-inspired model cannot fit an empty frame.")

        panel = self._prepare_panel(frame)
        candidate_factor_counts = self._candidate_factor_counts(panel)
        states = [self._fit_fixed_factor_count(panel, factor_count) for factor_count in candidate_factor_counts]

        selection_rows = []
        for state in states:
            selection_rows.append(
                {
                    "factor_count": state.factor_count,
                    "residual_variance": state.residual_variance,
                    "ic_rho1": state.ic_rho1,
                    "ic_rho2": state.ic_rho2,
                    "effective_window": panel.n_months * panel.bandwidth,
                    "bandwidth": panel.bandwidth,
                }
            )
        selection = pd.DataFrame(selection_rows).sort_values("factor_count").reset_index(drop=True)
        criterion_column = "ic_rho2" if self.ic_penalty_variant == "rho2" else "ic_rho1"
        best_row = selection.sort_values([criterion_column, "factor_count"]).iloc[0]
        selected_factor_count = int(best_row["factor_count"])
        selected_state = next(state for state in states if state.factor_count == selected_factor_count)

        self.panel_ = panel
        self.state_ = selected_state
        self.selection_table_ = selection
        self.selected_factor_count_ = selected_factor_count
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.panel_ is None or self.state_ is None or self.selection_table_ is None:
            raise RuntimeError("Model must be fitted before prediction.")

        forecast = frame.copy().sort_values("family_id")
        x_matrix = forecast[self.feature_columns].astype(float).to_numpy()
        nan_rows, nan_columns = np.where(np.isnan(x_matrix))
        if len(nan_rows):
            x_matrix[nan_rows, nan_columns] = self.panel_.feature_means[nan_columns]

        design_matrix = np.column_stack([np.ones(len(forecast), dtype=float), x_matrix])
        beta_next = np.array(
            [
                _bounded_linear_extrapolation(self.state_.beta_bc[:, column], self.panel_.source_month_index)
                for column in range(self.panel_.n_features)
            ]
        )

        factor_count = self.state_.factor_count
        factor_term = np.zeros(len(forecast), dtype=float)
        if factor_count > 0:
            factor_next = np.array(
                [_bounded_ar1_forecast(self.state_.factors[:, column]) for column in range(factor_count)],
                dtype=float,
            )
            family_lookup = {family_id: index for index, family_id in enumerate(self.panel_.family_ids)}
            for row_position, family_id in enumerate(forecast["family_id"].astype(str)):
                family_index = family_lookup.get(family_id)
                if family_index is None:
                    continue
                loading_next = np.array(
                    [
                        _bounded_linear_extrapolation(
                            self.state_.loadings[:, family_index, factor_index],
                            self.panel_.source_month_index,
                        )
                        for factor_index in range(factor_count)
                    ],
                    dtype=float,
                )
                factor_term[row_position] = float(loading_next @ factor_next)

        predicted_log_sales = design_matrix @ beta_next + factor_term
        upper_log = self.panel_.max_target_log_sales + 1.0
        bounded_log_sales = np.clip(predicted_log_sales, 0.0, upper_log)
        ordered_prediction = pd.Series(np.expm1(bounded_log_sales).clip(min=0), index=forecast.index, name="y_pred")
        return ordered_prediction.reindex(frame.index)

    def export_artifacts(self) -> PaperTrackArtifacts:
        if self.panel_ is None or self.state_ is None or self.selection_table_ is None or self.selected_factor_count_ is None:
            raise RuntimeError("Model must be fitted before exporting artifacts.")

        coefficient_rows = []
        beta_columns = ["intercept", *[f"beta_{feature}" for feature in self.feature_columns]]
        for month_position, source_month in enumerate(self.panel_.source_months):
            row = {
                "source_month": source_month,
                "source_month_index": int(self.panel_.source_month_index[month_position]),
                "selected_factor_count": self.selected_factor_count_,
                "bandwidth": self.panel_.bandwidth,
            }
            for column_position, column_name in enumerate(beta_columns):
                row[column_name] = float(self.state_.beta_bc[month_position, column_position])
            coefficient_rows.append(row)

        factor_rows = []
        for month_position, source_month in enumerate(self.panel_.source_months):
            row = {
                "source_month": source_month,
                "source_month_index": int(self.panel_.source_month_index[month_position]),
                "selected_factor_count": self.selected_factor_count_,
            }
            for factor_index in range(self.state_.factor_count):
                row[f"factor_{factor_index + 1}"] = float(self.state_.factors[month_position, factor_index])
            factor_rows.append(row)

        loading_rows = []
        for month_position, source_month in enumerate(self.panel_.source_months):
            for family_position, family_id in enumerate(self.panel_.family_ids):
                row = {
                    "family_id": family_id,
                    "source_month": source_month,
                    "source_month_index": int(self.panel_.source_month_index[month_position]),
                    "selected_factor_count": self.selected_factor_count_,
                }
                for factor_index in range(self.state_.factor_count):
                    row[f"loading_{factor_index + 1}"] = float(self.state_.loadings[month_position, family_position, factor_index])
                loading_rows.append(row)

        selection = self.selection_table_.copy()
        selection["selected_factor_count"] = self.selected_factor_count_
        selection["criterion_used"] = self.ic_penalty_variant

        fitted_rows = []
        fitted_sales = np.expm1(np.clip(self.state_.fitted, 0.0, None))
        observed_sales = np.expm1(np.clip(self.panel_.y, 0.0, None))
        for month_position, source_month in enumerate(self.panel_.source_months):
            for family_position, family_id in enumerate(self.panel_.family_ids):
                fitted_rows.append(
                    {
                        "family_id": family_id,
                        "source_month": source_month,
                        "source_month_index": int(self.panel_.source_month_index[month_position]),
                        "observed_log_sales": float(self.panel_.y[month_position, family_position]),
                        "fitted_log_sales": float(self.state_.fitted[month_position, family_position]),
                        "observed_sales": float(observed_sales[month_position, family_position]),
                        "fitted_sales": float(fitted_sales[month_position, family_position]),
                        "selected_factor_count": self.selected_factor_count_,
                    }
                )

        fitted_frame = pd.DataFrame(fitted_rows)
        observed = fitted_frame["observed_sales"].to_numpy(dtype=float)
        fitted_values = fitted_frame["fitted_sales"].to_numpy(dtype=float)
        denominator = np.abs(observed) + np.abs(fitted_values)
        valid = denominator > 0
        fit_summary = pd.DataFrame(
            [
                {
                    "selected_factor_count": self.selected_factor_count_,
                    "bandwidth": self.panel_.bandwidth,
                    "residual_variance": self.state_.residual_variance,
                    "in_sample_mae": float(np.mean(np.abs(observed - fitted_values))),
                    "in_sample_rmse": float(np.sqrt(np.mean((observed - fitted_values) ** 2))),
                    "in_sample_smape": float(
                        np.mean(2.0 * np.abs(observed[valid] - fitted_values[valid]) / denominator[valid]) if valid.any() else 0.0
                    ),
                    "n_panel_rows": int(len(fitted_frame)),
                }
            ]
        )

        return PaperTrackArtifacts(
            coefficients=pd.DataFrame(coefficient_rows),
            factors=pd.DataFrame(factor_rows),
            loadings=pd.DataFrame(loading_rows),
            selection=selection,
            fitted=fitted_frame,
            fit_summary=fit_summary,
        )
