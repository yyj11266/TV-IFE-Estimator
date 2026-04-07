"""Lifecycle-aware predictive TV forecaster."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .paper_proxy import PaperInspiredFactorForecaster, PaperTrackArtifacts


def _smape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    valid = denominator > 0
    if not valid.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(actual[valid] - predicted[valid]) / denominator[valid]))


def _mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def _rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(residual**2)))


class LaunchAwarePredictiveTVForecaster:
    """Mixture of launch and mature TV experts with a sales-history gate."""

    model_name = "tv_ife_predictive_augmented"

    def __init__(
        self,
        launch_feature_columns: tuple[str, ...] | list[str],
        mature_feature_columns: tuple[str, ...] | list[str],
        factor_candidates: tuple[int, ...] | list[int] = (),
        max_factor_count: int = 6,
        ic_penalty_variant: str = "rho2",
        ridge_alpha: float = 3e-2,
        bandwidth_scale: float = 0.65,
        bandwidth_method: str = "response_std",
        bandwidth_pilot_scale: float = 2.0,
        min_window: float = 0.0,
        history_column: str = "source_has_sales_history",
        history_count_column: str = "source_positive_history_count",
        first_sale_column: str = "first_sale_month_index",
        mature_history_threshold: int = 4,
        mature_blend_weight: float = 0.6,
        n_factors: int | None = 0,
        mature_feature_candidates: tuple[tuple[str, ...], ...] | list[tuple[str, ...] | list[str]] | None = None,
        mature_history_threshold_candidates: tuple[int, ...] | list[int] | None = None,
        mature_blend_weight_candidates: tuple[float, ...] | list[float] | None = None,
        enable_inner_validation: bool = True,
        inner_validation_min_months: int = 4,
    ) -> None:
        self.launch_feature_columns = list(launch_feature_columns)
        self.mature_feature_columns = list(mature_feature_columns)
        self.factor_candidates = tuple(sorted({int(value) for value in factor_candidates if int(value) >= 0}))
        self.max_factor_count = int(max_factor_count)
        self.ic_penalty_variant = str(ic_penalty_variant)
        self.ridge_alpha = ridge_alpha
        self.bandwidth_scale = bandwidth_scale
        self.bandwidth_method = str(bandwidth_method)
        self.bandwidth_pilot_scale = float(bandwidth_pilot_scale)
        self.min_window = min_window
        self.history_column = history_column
        self.history_count_column = history_count_column
        self.first_sale_column = first_sale_column
        self.mature_history_threshold = int(mature_history_threshold)
        self.mature_blend_weight = float(mature_blend_weight)
        self.n_factors = None if n_factors is None else int(n_factors)
        self.enable_inner_validation = bool(enable_inner_validation)
        self.inner_validation_min_months = int(inner_validation_min_months)

        self.mature_feature_candidates = self._normalize_feature_candidates(
            self.mature_feature_columns,
            mature_feature_candidates,
        )
        self.mature_history_threshold_candidates = self._normalize_int_candidates(
            self.mature_history_threshold,
            mature_history_threshold_candidates,
        )
        self.mature_blend_weight_candidates = self._normalize_float_candidates(
            self.mature_blend_weight,
            mature_blend_weight_candidates,
        )

        self.launch_model_: PaperInspiredFactorForecaster | None = None
        self.mature_model_: PaperInspiredFactorForecaster | None = None
        self.training_frame_: pd.DataFrame | None = None
        self.mature_family_ids_: set[str] = set()
        self.selection_table_: pd.DataFrame | None = None
        self.selected_factor_count_: int = 0
        self.selected_launch_factor_count_: int = 0
        self.selected_mature_factor_count_: int = 0
        self.selected_mature_feature_columns_: list[str] = list(self.mature_feature_columns)
        self.selected_mature_history_threshold_: int = self.mature_history_threshold
        self.selected_mature_blend_weight_: float = self.mature_blend_weight
        self.inner_validation_results_: pd.DataFrame | None = None
        self.parameter_source_: str = "configured_default"

    @staticmethod
    def _normalize_feature_candidates(
        default_columns: list[str],
        candidates: tuple[tuple[str, ...], ...] | list[tuple[str, ...] | list[str]] | None,
    ) -> list[list[str]]:
        ordered_candidates = [tuple(default_columns), *[tuple(candidate) for candidate in (candidates or [])]]
        normalized: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for candidate in ordered_candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(list(candidate))
        if not normalized:
            raise ValueError("At least one mature feature candidate is required.")
        return normalized

    @staticmethod
    def _normalize_int_candidates(default_value: int, candidates: tuple[int, ...] | list[int] | None) -> list[int]:
        values = [int(default_value), *[int(value) for value in (candidates or [])]]
        normalized: list[int] = []
        seen: set[int] = set()
        for value in values:
            if value < 0 or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        if not normalized:
            raise ValueError("At least one history-threshold candidate is required.")
        return normalized

    @staticmethod
    def _normalize_float_candidates(
        default_value: float,
        candidates: tuple[float, ...] | list[float] | None,
    ) -> list[float]:
        values = [float(default_value), *[float(value) for value in (candidates or [])]]
        normalized: list[float] = []
        seen: set[float] = set()
        for value in values:
            if value < 0.0 or value > 1.0 or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        if not normalized:
            raise ValueError("At least one blend-weight candidate is required.")
        return normalized

    @staticmethod
    def _feature_signature(feature_columns: list[str]) -> str:
        return ",".join(feature_columns)

    def _fit_expert(self, frame: pd.DataFrame, feature_columns: list[str]) -> PaperInspiredFactorForecaster:
        model = PaperInspiredFactorForecaster(
            n_factors=self.n_factors,
            feature_columns=tuple(feature_columns),
            factor_candidates=self.factor_candidates,
            max_factor_count=self.max_factor_count,
            ic_penalty_variant=self.ic_penalty_variant,
            ridge_alpha=self.ridge_alpha,
            bandwidth_scale=self.bandwidth_scale,
            bandwidth_method=self.bandwidth_method,
            bandwidth_pilot_scale=self.bandwidth_pilot_scale,
            min_window=self.min_window,
        ).fit(frame)
        model.model_name = self.model_name
        return model

    def _validate_frame(self, frame: pd.DataFrame, purpose: str, extra_columns: set[str] | None = None) -> None:
        required = {self.history_column, self.history_count_column, self.first_sale_column, "family_id", "source_month_index"}
        required.update(extra_columns or set())
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Predictive TV model requires columns missing from {purpose}: {missing}")

    def _select_mature_family_ids(self, frame: pd.DataFrame) -> set[str]:
        earliest_source_month_index = int(frame["source_month_index"].astype(float).min())
        mature_family_mask = frame[self.first_sale_column].fillna(np.inf).astype(float) <= earliest_source_month_index
        mature_family_ids = frame.loc[mature_family_mask, "family_id"].astype(str).unique().tolist()
        return set(mature_family_ids)

    def _fit_expert_pair(
        self,
        frame: pd.DataFrame,
        mature_feature_columns: list[str],
        launch_model: PaperInspiredFactorForecaster | None = None,
    ) -> tuple[PaperInspiredFactorForecaster, PaperInspiredFactorForecaster, set[str]]:
        launch_model = launch_model or self._fit_expert(frame, self.launch_feature_columns)
        mature_family_ids = self._select_mature_family_ids(frame)
        mature_training_frame = frame[frame["family_id"].astype(str).isin(mature_family_ids)].copy()
        if mature_training_frame.empty:
            mature_model = launch_model
        else:
            mature_model = self._fit_expert(mature_training_frame, mature_feature_columns)
        return launch_model, mature_model, mature_family_ids

    def _combine_predictions(
        self,
        frame: pd.DataFrame,
        launch_prediction: pd.Series,
        mature_prediction: pd.Series,
        mature_history_threshold: int,
        mature_blend_weight: float,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        history_counts = frame[self.history_count_column].fillna(0).astype(float).to_numpy()
        use_mature = history_counts >= float(mature_history_threshold)
        launch_values = launch_prediction.to_numpy(dtype=float)
        mature_values = mature_prediction.to_numpy(dtype=float)
        blended_values = np.where(
            use_mature,
            mature_blend_weight * mature_values + (1.0 - mature_blend_weight) * launch_values,
            launch_values,
        )
        history_gate = frame[self.history_column].fillna(0).astype(int).gt(0).to_numpy()
        ensemble_values = np.where(history_gate, blended_values, 0.0)

        launch_series = pd.Series(launch_values, index=frame.index, name="launch_prediction")
        mature_series = pd.Series(mature_values, index=frame.index, name="mature_prediction")
        use_mature_series = pd.Series(use_mature.astype(int), index=frame.index, name="use_mature_expert")
        ensemble_series = pd.Series(ensemble_values, index=frame.index, name="y_pred")
        return launch_series, mature_series, ensemble_series, use_mature_series

    def _build_inner_validation_folds(self, frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
        if not self.enable_inner_validation:
            return []

        self._validate_frame(frame, "training frame", {"target_month", "target_month_index"})
        month_table = (
            frame[["target_month", "target_month_index"]]
            .drop_duplicates(subset=["target_month"])
            .sort_values("target_month_index")
            .reset_index(drop=True)
        )

        folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
        for row in month_table.itertuples(index=False):
            validation_frame = frame[frame["target_month"] == str(row.target_month)].copy()
            training_frame = frame[frame["target_month_index"].astype(float) < float(row.target_month_index)].copy()
            if validation_frame.empty or training_frame.empty:
                continue
            if training_frame["source_month_index"].dropna().nunique() < self.inner_validation_min_months:
                continue
            folds.append((str(row.target_month), training_frame.reset_index(drop=True), validation_frame.reset_index(drop=True)))
        return folds

    def _fallback_inner_validation_results(self, parameter_source: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "selection_stage": "inner_validation",
                    "expert_name": "ensemble",
                    "selection_rank": 1,
                    "selected": 1,
                    "parameter_source": parameter_source,
                    "mature_feature_columns": self._feature_signature(self.mature_feature_columns),
                    "mature_feature_count": len(self.mature_feature_columns),
                    "mature_history_threshold": self.mature_history_threshold,
                    "mature_blend_weight": self.mature_blend_weight,
                    "n_validation_folds": 0,
                    "n_validation_rows": 0,
                    "validation_months": "",
                    "mean_smape": np.nan,
                    "mean_mae": np.nan,
                    "mean_rmse": np.nan,
                }
            ]
        )

    def _run_inner_validation(
        self,
        frame: pd.DataFrame,
    ) -> tuple[list[str], int, float, pd.DataFrame, str]:
        if not self.enable_inner_validation:
            fallback = self._fallback_inner_validation_results("configured_default")
            return list(self.mature_feature_columns), self.mature_history_threshold, self.mature_blend_weight, fallback, "configured_default"

        folds = self._build_inner_validation_folds(frame)
        if not folds:
            fallback = self._fallback_inner_validation_results("configured_default_insufficient_history")
            return (
                list(self.mature_feature_columns),
                self.mature_history_threshold,
                self.mature_blend_weight,
                fallback,
                "configured_default_insufficient_history",
            )

        fold_records: list[dict[str, object]] = []
        feature_candidates = [list(columns) for columns in self.mature_feature_candidates]
        for validation_month, inner_train, inner_validation in folds:
            launch_model = self._fit_expert(inner_train, self.launch_feature_columns)
            launch_prediction = launch_model.predict(inner_validation).astype(float)

            mature_predictions_by_feature: dict[tuple[str, ...], tuple[pd.Series, int]] = {}
            for mature_feature_columns in feature_candidates:
                feature_key = tuple(mature_feature_columns)
                _, mature_model, mature_family_ids = self._fit_expert_pair(
                    inner_train,
                    mature_feature_columns,
                    launch_model=launch_model,
                )
                mature_prediction = mature_model.predict(inner_validation).astype(float)
                mature_predictions_by_feature[feature_key] = (mature_prediction, len(mature_family_ids))

            for mature_feature_columns in feature_candidates:
                feature_key = tuple(mature_feature_columns)
                mature_prediction, mature_training_family_count = mature_predictions_by_feature[feature_key]
                for mature_history_threshold, mature_blend_weight in product(
                    self.mature_history_threshold_candidates,
                    self.mature_blend_weight_candidates,
                ):
                    _, _, ensemble_prediction, use_mature = self._combine_predictions(
                        inner_validation,
                        launch_prediction,
                        mature_prediction,
                        mature_history_threshold=mature_history_threshold,
                        mature_blend_weight=mature_blend_weight,
                    )
                    fold_records.append(
                        {
                            "validation_month": validation_month,
                            "mature_feature_columns": self._feature_signature(mature_feature_columns),
                            "mature_feature_count": len(mature_feature_columns),
                            "mature_history_threshold": int(mature_history_threshold),
                            "mature_blend_weight": float(mature_blend_weight),
                            "fold_smape": _smape(inner_validation["target_sales"], ensemble_prediction),
                            "fold_mae": _mae(inner_validation["target_sales"], ensemble_prediction),
                            "fold_rmse": _rmse(inner_validation["target_sales"], ensemble_prediction),
                            "n_validation_rows": int(len(inner_validation)),
                            "mature_training_family_count": int(mature_training_family_count),
                            "mature_rows_in_validation": int(use_mature.sum()),
                        }
                    )

        if not fold_records:
            fallback = self._fallback_inner_validation_results("configured_default_inner_validation_error")
            return (
                list(self.mature_feature_columns),
                self.mature_history_threshold,
                self.mature_blend_weight,
                fallback,
                "configured_default_inner_validation_error",
            )

        fold_table = pd.DataFrame(fold_records)
        results = (
            fold_table.groupby(
                ["mature_feature_columns", "mature_feature_count", "mature_history_threshold", "mature_blend_weight"],
                as_index=False,
            )
            .agg(
                mean_smape=("fold_smape", "mean"),
                mean_mae=("fold_mae", "mean"),
                mean_rmse=("fold_rmse", "mean"),
                n_validation_folds=("validation_month", "nunique"),
                n_validation_rows=("n_validation_rows", "sum"),
                avg_mature_training_family_count=("mature_training_family_count", "mean"),
                avg_mature_rows_in_validation=("mature_rows_in_validation", "mean"),
                validation_months=("validation_month", lambda values: ",".join(sorted({str(value) for value in values}))),
            )
            .sort_values(
                [
                    "mean_smape",
                    "mean_mae",
                    "mean_rmse",
                    "mature_feature_count",
                    "mature_history_threshold",
                    "mature_blend_weight",
                ]
            )
            .reset_index(drop=True)
        )
        results["selection_stage"] = "inner_validation"
        results["expert_name"] = "ensemble"
        results["selection_rank"] = np.arange(1, len(results) + 1)
        results["selected"] = 0
        results["parameter_source"] = "inner_validation"
        results.loc[0, "selected"] = 1

        best_row = results.iloc[0]
        selected_feature_columns = str(best_row["mature_feature_columns"]).split(",")
        return (
            selected_feature_columns,
            int(best_row["mature_history_threshold"]),
            float(best_row["mature_blend_weight"]),
            results,
            "inner_validation",
        )

    def fit(self, frame: pd.DataFrame) -> "LaunchAwarePredictiveTVForecaster":
        if frame.empty:
            raise ValueError("Predictive TV model cannot fit an empty frame.")

        self._validate_frame(frame, "training frame", {"target_month", "target_month_index"})
        self.training_frame_ = frame.copy()

        (
            self.selected_mature_feature_columns_,
            self.selected_mature_history_threshold_,
            self.selected_mature_blend_weight_,
            self.inner_validation_results_,
            self.parameter_source_,
        ) = self._run_inner_validation(frame)

        self.launch_model_, self.mature_model_, self.mature_family_ids_ = self._fit_expert_pair(
            frame,
            self.selected_mature_feature_columns_,
        )

        launch_selection = self.launch_model_.selection_table_.copy() if self.launch_model_.selection_table_ is not None else pd.DataFrame()
        launch_selection["selection_stage"] = "expert_ic"
        launch_selection["expert_name"] = "launch"
        launch_selection["training_family_count"] = int(frame["family_id"].astype(str).nunique())
        launch_selection["feature_columns"] = self._feature_signature(self.launch_feature_columns)

        mature_selection = self.mature_model_.selection_table_.copy() if self.mature_model_.selection_table_ is not None else pd.DataFrame()
        mature_selection["selection_stage"] = "expert_ic"
        mature_selection["expert_name"] = "mature"
        mature_selection["training_family_count"] = int(len(self.mature_family_ids_))
        mature_selection["feature_columns"] = self._feature_signature(self.selected_mature_feature_columns_)

        selection_frames = [launch_selection, mature_selection]
        if self.inner_validation_results_ is not None and not self.inner_validation_results_.empty:
            selection_frames.append(self.inner_validation_results_.copy())
        self.selection_table_ = pd.concat(selection_frames, ignore_index=True, sort=False)
        selected_feature_signature = self._feature_signature(self.selected_mature_feature_columns_)
        launch_mask = (
            self.selection_table_["expert_name"].astype(str).eq("launch")
            if "expert_name" in self.selection_table_.columns
            else pd.Series(False, index=self.selection_table_.index)
        )
        if "mature_feature_columns" in self.selection_table_.columns:
            fill_mask = self.selection_table_["mature_feature_columns"].isna() & ~launch_mask
            self.selection_table_.loc[fill_mask, "mature_feature_columns"] = selected_feature_signature
        else:
            self.selection_table_["mature_feature_columns"] = np.where(launch_mask, np.nan, selected_feature_signature)
        if "mature_history_threshold" in self.selection_table_.columns:
            self.selection_table_["mature_history_threshold"] = self.selection_table_["mature_history_threshold"].fillna(
                self.selected_mature_history_threshold_
            )
        else:
            self.selection_table_["mature_history_threshold"] = self.selected_mature_history_threshold_
        if "mature_blend_weight" in self.selection_table_.columns:
            self.selection_table_["mature_blend_weight"] = self.selection_table_["mature_blend_weight"].fillna(
                self.selected_mature_blend_weight_
            )
        else:
            self.selection_table_["mature_blend_weight"] = self.selected_mature_blend_weight_
        self.selection_table_["selected_mature_feature_columns"] = selected_feature_signature
        self.selection_table_["selected_mature_history_threshold"] = self.selected_mature_history_threshold_
        self.selection_table_["selected_mature_blend_weight"] = self.selected_mature_blend_weight_
        self.selection_table_["launch_gate_enabled"] = 1
        self.selection_table_["parameter_source"] = self.selection_table_["parameter_source"].fillna(self.parameter_source_)
        self.selected_launch_factor_count_ = int(self.launch_model_.selected_factor_count_ or 0)
        self.selected_mature_factor_count_ = int(self.mature_model_.selected_factor_count_ or 0)
        self.selected_factor_count_ = int(self.launch_model_.selected_factor_count_ or 0)
        return self

    def _build_predictions(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        if self.launch_model_ is None or self.mature_model_ is None:
            raise RuntimeError("Model must be fitted before prediction.")

        self._validate_frame(frame, "forecast frame")
        launch_prediction = self.launch_model_.predict(frame).astype(float)
        mature_prediction = self.mature_model_.predict(frame).astype(float)
        return self._combine_predictions(
            frame,
            launch_prediction,
            mature_prediction,
            mature_history_threshold=self.selected_mature_history_threshold_,
            mature_blend_weight=self.selected_mature_blend_weight_,
        )

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        _, _, ensemble_prediction, _ = self._build_predictions(frame)
        return ensemble_prediction

    @staticmethod
    def _annotate_artifacts(artifacts: PaperTrackArtifacts, expert_name: str) -> PaperTrackArtifacts:
        return PaperTrackArtifacts(
            coefficients=artifacts.coefficients.assign(expert_name=expert_name),
            factors=artifacts.factors.assign(expert_name=expert_name),
            loadings=artifacts.loadings.assign(expert_name=expert_name),
            selection=artifacts.selection.assign(expert_name=expert_name),
            fitted=artifacts.fitted.assign(expert_name=expert_name),
            fit_summary=artifacts.fit_summary.assign(expert_name=expert_name),
        )

    def export_artifacts(self) -> PaperTrackArtifacts:
        if self.launch_model_ is None or self.mature_model_ is None or self.training_frame_ is None or self.selection_table_ is None:
            raise RuntimeError("Model must be fitted before exporting artifacts.")

        launch_artifacts = self._annotate_artifacts(self.launch_model_.export_artifacts(), "launch")
        mature_artifacts = self._annotate_artifacts(self.mature_model_.export_artifacts(), "mature")
        launch_prediction, mature_prediction, ensemble_prediction, use_mature = self._build_predictions(self.training_frame_)

        training_frame = self.training_frame_.copy()
        training_frame["launch_prediction"] = launch_prediction
        training_frame["mature_prediction"] = mature_prediction
        training_frame["fitted_sales"] = ensemble_prediction
        training_frame["fitted_log_sales"] = np.log1p(training_frame["fitted_sales"].clip(lower=0))
        training_frame["observed_sales"] = training_frame["target_sales"].astype(float)
        training_frame["observed_log_sales"] = training_frame["target_log_sales"].astype(float)
        training_frame["use_mature_expert"] = use_mature
        training_frame["selected_factor_count"] = self.selected_factor_count_
        training_frame["selected_launch_factor_count"] = self.selected_launch_factor_count_
        training_frame["selected_mature_factor_count"] = self.selected_mature_factor_count_

        denominator = np.abs(training_frame["observed_sales"].to_numpy(dtype=float)) + np.abs(
            training_frame["fitted_sales"].to_numpy(dtype=float)
        )
        valid = denominator > 0
        selected_validation_row = (
            self.inner_validation_results_[self.inner_validation_results_["selected"] == 1].iloc[0]
            if self.inner_validation_results_ is not None and not self.inner_validation_results_.empty
            else None
        )
        fit_summary = pd.DataFrame(
            [
                {
                    "selected_factor_count": self.selected_factor_count_,
                    "selected_launch_factor_count": self.selected_launch_factor_count_,
                    "selected_mature_factor_count": self.selected_mature_factor_count_,
                    "launch_bandwidth": float(launch_artifacts.fit_summary["bandwidth"].iloc[0]),
                    "mature_bandwidth": float(mature_artifacts.fit_summary["bandwidth"].iloc[0]),
                    "mature_training_family_count": int(len(self.mature_family_ids_)),
                    "mature_history_threshold": self.selected_mature_history_threshold_,
                    "mature_blend_weight": self.selected_mature_blend_weight_,
                    "selected_mature_feature_columns": self._feature_signature(self.selected_mature_feature_columns_),
                    "launch_feature_columns": self._feature_signature(self.launch_feature_columns),
                    "launch_gate_enabled": 1,
                    "parameter_source": self.parameter_source_,
                    "inner_validation_enabled": int(self.enable_inner_validation),
                    "inner_validation_folds": int(selected_validation_row["n_validation_folds"]) if selected_validation_row is not None else 0,
                    "inner_validation_rows": int(selected_validation_row["n_validation_rows"]) if selected_validation_row is not None else 0,
                    "inner_validation_smape": float(selected_validation_row["mean_smape"]) if selected_validation_row is not None and pd.notna(selected_validation_row["mean_smape"]) else np.nan,
                    "inner_validation_mae": float(selected_validation_row["mean_mae"]) if selected_validation_row is not None and pd.notna(selected_validation_row["mean_mae"]) else np.nan,
                    "inner_validation_rmse": float(selected_validation_row["mean_rmse"]) if selected_validation_row is not None and pd.notna(selected_validation_row["mean_rmse"]) else np.nan,
                    "in_sample_mae": float(np.mean(np.abs(training_frame["observed_sales"] - training_frame["fitted_sales"]))),
                    "in_sample_rmse": float(
                        np.sqrt(np.mean((training_frame["observed_sales"] - training_frame["fitted_sales"]) ** 2))
                    ),
                    "in_sample_smape": float(
                        np.mean(
                            2.0
                            * np.abs(
                                training_frame["observed_sales"].to_numpy(dtype=float)[valid]
                                - training_frame["fitted_sales"].to_numpy(dtype=float)[valid]
                            )
                            / denominator[valid]
                        )
                        if valid.any()
                        else 0.0
                    ),
                    "n_panel_rows": int(len(training_frame)),
                    "mature_expert_rows": int(training_frame["use_mature_expert"].sum()),
                    "launch_only_rows": int((training_frame["use_mature_expert"] == 0).sum()),
                }
            ]
        )

        selection = self.selection_table_.copy()

        fitted = training_frame[
            [
                "family_id",
                "source_month",
                "source_month_index",
                self.history_column,
                self.history_count_column,
                "is_prelaunch_target",
                "use_mature_expert",
                "observed_log_sales",
                "fitted_log_sales",
                "observed_sales",
                "fitted_sales",
                "launch_prediction",
                "mature_prediction",
                "selected_factor_count",
                "selected_launch_factor_count",
                "selected_mature_factor_count",
            ]
        ].copy()
        fitted["selected_mature_feature_columns"] = self._feature_signature(self.selected_mature_feature_columns_)
        fitted["mature_history_threshold"] = self.selected_mature_history_threshold_
        fitted["mature_blend_weight"] = self.selected_mature_blend_weight_

        coefficients = pd.concat([launch_artifacts.coefficients, mature_artifacts.coefficients], ignore_index=True)
        factors = pd.concat([launch_artifacts.factors, mature_artifacts.factors], ignore_index=True)
        loadings = pd.concat([launch_artifacts.loadings, mature_artifacts.loadings], ignore_index=True)

        return PaperTrackArtifacts(
            coefficients=coefficients,
            factors=factors,
            loadings=loadings,
            selection=selection,
            fitted=fitted,
            fit_summary=fit_summary,
        )
