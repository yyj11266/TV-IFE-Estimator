"""Backtesting and evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .models.baselines import NaiveLastMonthModel
from .models.direct import DirectRidgeForecaster
from .models.paper_proxy import PaperInspiredFactorForecaster
from .models.predictive_tv import LaunchAwarePredictiveTVForecaster


@dataclass
class BacktestArtifacts:
    """Full backtest outputs."""

    evaluation: pd.DataFrame
    summary: pd.DataFrame
    paper_selection: pd.DataFrame


def smape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Symmetric mean absolute percentage error."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    valid = denominator > 0
    if not valid.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(actual[valid] - predicted[valid]) / denominator[valid]))


def _append_fold_result(
    model_name: str,
    frame: pd.DataFrame,
    prediction: pd.Series | np.ndarray,
) -> pd.DataFrame:
    if isinstance(prediction, pd.Series):
        predicted_values = prediction.reindex(frame.index).to_numpy(dtype=float)
    else:
        predicted_values = np.asarray(prediction, dtype=float)
    predicted = pd.Series(predicted_values, index=frame.index, dtype=float).clip(lower=0)
    result = frame[["family_id", "brand", "target_month", "target_sales"]].copy()
    result["model_name"] = model_name
    result["y_true"] = result["target_sales"].astype(float)
    result["y_pred"] = predicted
    result["abs_err"] = (result["y_true"] - result["y_pred"]).abs()
    result["ape"] = np.where(result["y_true"] > 0, result["abs_err"] / result["y_true"], np.nan)
    return result.drop(columns="target_sales")


def _brand_weighted_mae(model_frame: pd.DataFrame) -> float:
    brand_totals = (
        model_frame.groupby(["target_month", "brand"], as_index=False)[["y_true", "y_pred"]]
        .sum()
        .sort_values(["target_month", "brand"])
    )
    weighted_errors: list[float] = []
    for target_month, target_frame in brand_totals.groupby("target_month"):
        total_sales = float(target_frame["y_true"].sum())
        if total_sales <= 0:
            continue
        weights = target_frame["y_true"] / total_sales
        weighted_errors.append(float((weights * (target_frame["y_true"] - target_frame["y_pred"]).abs()).sum()))
    return float(np.mean(weighted_errors)) if weighted_errors else 0.0


def summarize_evaluation(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level metrics by model."""

    summary_rows: list[dict[str, float | str]] = []
    for model_name, model_frame in evaluation.groupby("model_name"):
        y_true = model_frame["y_true"].astype(float)
        y_pred = model_frame["y_pred"].astype(float)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        summary_rows.append(
            {
                "model_name": model_name,
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "rmse": rmse,
                "smape": smape(y_true, y_pred),
                "brand_weighted_mae": _brand_weighted_mae(model_frame),
                "n_predictions": int(len(model_frame)),
                "n_target_months": int(model_frame["target_month"].nunique()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["smape", "mae", "model_name"]).reset_index(drop=True)


def _fit_paper_model(
    train_frame: pd.DataFrame,
    config: PipelineConfig,
    n_factors: int | None = None,
    feature_columns: tuple[str, ...] | None = None,
) -> tuple[PaperInspiredFactorForecaster, list[dict[str, float | int | str]]]:
    model = PaperInspiredFactorForecaster(
        n_factors=n_factors,
        feature_columns=feature_columns or config.paper_feature_columns,
        factor_candidates=config.factor_candidates,
        max_factor_count=config.paper_max_factor_count,
        ic_penalty_variant=config.paper_ic_penalty_variant,
        ridge_alpha=config.paper_model_alpha,
        bandwidth_scale=config.paper_bandwidth_scale,
        bandwidth_method=config.paper_bandwidth_method,
        bandwidth_pilot_scale=config.paper_bandwidth_pilot_scale,
        min_window=config.paper_min_window,
    ).fit(train_frame)
    records = model.selection_table_.to_dict(orient="records") if model.selection_table_ is not None else []
    for record in records:
        record["criterion_used"] = f"bic_{config.paper_ic_penalty_variant}"
        record["selected_factor_count"] = model.selected_factor_count_
        record["selected"] = int(record["factor_count"] == model.selected_factor_count_)
    return model, records


def _fit_predictive_tv_ife_model(
    train_frame: pd.DataFrame,
    config: PipelineConfig,
) -> LaunchAwarePredictiveTVForecaster:
    model = LaunchAwarePredictiveTVForecaster(
        launch_feature_columns=config.predictive_tv_ife_feature_columns,
        mature_feature_columns=config.predictive_tv_ife_mature_feature_columns,
        mature_feature_candidates=config.predictive_tv_ife_mature_feature_candidates,
        factor_candidates=config.factor_candidates,
        max_factor_count=config.paper_max_factor_count,
        ic_penalty_variant=config.paper_ic_penalty_variant,
        ridge_alpha=config.predictive_tv_ife_ridge_alpha,
        bandwidth_scale=config.predictive_tv_ife_bandwidth_scale,
        bandwidth_method=config.predictive_tv_ife_bandwidth_method,
        bandwidth_pilot_scale=config.predictive_tv_ife_bandwidth_pilot_scale,
        min_window=config.paper_min_window,
        n_factors=config.predictive_tv_ife_n_factors,
        mature_history_threshold=config.predictive_tv_ife_mature_history_threshold,
        mature_history_threshold_candidates=config.predictive_tv_ife_mature_history_threshold_candidates,
        mature_blend_weight=config.predictive_tv_ife_mature_blend_weight,
        mature_blend_weight_candidates=config.predictive_tv_ife_mature_blend_weight_candidates,
        enable_inner_validation=config.predictive_tv_ife_enable_inner_validation,
        inner_validation_min_months=config.predictive_tv_ife_inner_validation_min_months,
    ).fit(train_frame)
    return model


def run_backtest(
    direct_training_frame: pd.DataFrame,
    paper_training_frame: pd.DataFrame,
    config: PipelineConfig,
) -> BacktestArtifacts:
    """Run expanding-window one-step-ahead backtests."""

    direct_training_frame = direct_training_frame.copy()
    paper_training_frame = paper_training_frame.copy()
    direct_training_frame["target_month"] = direct_training_frame["target_month"].astype(str)
    paper_training_frame["target_month"] = paper_training_frame["target_month"].astype(str)

    evaluation_rows: list[pd.DataFrame] = []
    paper_selection_rows: list[dict[str, float | int | str]] = []

    for target_month in config.target_months:
        direct_train = direct_training_frame[direct_training_frame["target_month"] < target_month].copy()
        direct_test = direct_training_frame[direct_training_frame["target_month"] == target_month].copy()
        paper_train = paper_training_frame[paper_training_frame["target_month"] < target_month].copy()
        paper_test = paper_training_frame[paper_training_frame["target_month"] == target_month].copy()

        if direct_test.empty or paper_test.empty:
            raise ValueError(f"Missing test rows for target month {target_month}.")

        naive_model = NaiveLastMonthModel().fit(direct_train)
        evaluation_rows.append(_append_fold_result(naive_model.model_name, direct_test, naive_model.predict(direct_test)))

        direct_model = DirectRidgeForecaster(alpha=config.direct_model_alpha).fit(direct_train)
        evaluation_rows.append(_append_fold_result(direct_model.model_name, direct_test, direct_model.predict(direct_test)))

        paper_model, selection_records = _fit_paper_model(paper_train, config)
        for record in selection_records:
            record["forecast_target_month"] = target_month
        paper_selection_rows.extend(selection_records)

        paper_fold_result = _append_fold_result(paper_model.model_name, paper_test, paper_model.predict(paper_test))
        paper_fold_result["selected_factor_count"] = paper_model.selected_factor_count_
        evaluation_rows.append(paper_fold_result)

        predictive_tv_model = _fit_predictive_tv_ife_model(direct_train, config)
        predictive_fold_result = _append_fold_result(
            predictive_tv_model.model_name,
            direct_test,
            predictive_tv_model.predict(direct_test),
        )
        predictive_fold_result["selected_factor_count"] = predictive_tv_model.selected_factor_count_
        predictive_fold_result["selected_launch_factor_count"] = predictive_tv_model.selected_launch_factor_count_
        predictive_fold_result["selected_mature_factor_count"] = predictive_tv_model.selected_mature_factor_count_
        predictive_fold_result["selected_mature_feature_columns"] = ",".join(
            predictive_tv_model.selected_mature_feature_columns_
        )
        predictive_fold_result["selected_mature_history_threshold"] = (
            predictive_tv_model.selected_mature_history_threshold_
        )
        predictive_fold_result["selected_mature_blend_weight"] = predictive_tv_model.selected_mature_blend_weight_
        predictive_fold_result["parameter_source"] = predictive_tv_model.parameter_source_
        evaluation_rows.append(predictive_fold_result)

    evaluation = pd.concat(evaluation_rows, ignore_index=True)
    summary = summarize_evaluation(evaluation)
    paper_selection = pd.DataFrame(paper_selection_rows)
    return BacktestArtifacts(
        evaluation=evaluation.sort_values(["model_name", "target_month", "family_id"]).reset_index(drop=True),
        summary=summary,
        paper_selection=paper_selection,
    )


def fit_final_models_and_forecast(
    direct_training_frame: pd.DataFrame,
    paper_training_frame: pd.DataFrame,
    direct_forecast_frame: pd.DataFrame,
    paper_forecast_frame: pd.DataFrame,
    next_target_month: str,
    config: PipelineConfig,
    paper_n_factors: int | None = None,
    paper_feature_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, PaperInspiredFactorForecaster, LaunchAwarePredictiveTVForecaster]:
    """Fit final models on all available data and forecast the next target month."""

    naive_model = NaiveLastMonthModel().fit(direct_training_frame)
    direct_model = DirectRidgeForecaster(alpha=config.direct_model_alpha).fit(direct_training_frame)

    paper_model, _ = _fit_paper_model(
        paper_training_frame,
        config,
        n_factors=paper_n_factors,
        feature_columns=paper_feature_columns,
    )
    predictive_tv_model = _fit_predictive_tv_ife_model(direct_training_frame, config)

    forecast_rows = []
    for model_name, forecast_frame, prediction in [
        (
            naive_model.model_name,
            direct_forecast_frame,
            naive_model.predict(direct_forecast_frame),
        ),
        (
            direct_model.model_name,
            direct_forecast_frame,
            direct_model.predict(direct_forecast_frame),
        ),
        (
            paper_model.model_name,
            paper_forecast_frame,
            paper_model.predict(paper_forecast_frame),
        ),
        (
            predictive_tv_model.model_name,
            direct_forecast_frame,
            predictive_tv_model.predict(direct_forecast_frame),
        ),
    ]:
        output = forecast_frame[["family_id", "brand"]].copy()
        output["target_month"] = next_target_month
        output["model_name"] = model_name
        if isinstance(prediction, pd.Series):
            predicted_values = prediction.reindex(forecast_frame.index).to_numpy(dtype=float)
        else:
            predicted_values = np.asarray(prediction, dtype=float)
        output["y_pred"] = pd.Series(predicted_values, index=forecast_frame.index).astype(float).clip(lower=0)
        if model_name == paper_model.model_name:
            output["selected_factor_count"] = paper_model.selected_factor_count_
        if model_name == predictive_tv_model.model_name:
            output["selected_factor_count"] = predictive_tv_model.selected_factor_count_
            output["selected_launch_factor_count"] = predictive_tv_model.selected_launch_factor_count_
            output["selected_mature_factor_count"] = predictive_tv_model.selected_mature_factor_count_
            output["selected_mature_feature_columns"] = ",".join(
                predictive_tv_model.selected_mature_feature_columns_
            )
            output["selected_mature_history_threshold"] = predictive_tv_model.selected_mature_history_threshold_
            output["selected_mature_blend_weight"] = predictive_tv_model.selected_mature_blend_weight_
            output["parameter_source"] = predictive_tv_model.parameter_source_
        forecast_rows.append(output)

    final_forecast = pd.concat(forecast_rows, ignore_index=True)
    return (
        final_forecast.sort_values(["model_name", "family_id"]).reset_index(drop=True),
        paper_model,
        predictive_tv_model,
    )
