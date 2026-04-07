"""Empirical diagnostics and sensitivity study for the paper track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import smape
from .config import PipelineConfig
from .constants import CORE_PAPER_NUMERIC_FEATURES
from .models.paper_proxy import PaperInspiredFactorForecaster
from .models.predictive_tv import LaunchAwarePredictiveTVForecaster

_EMPIRICAL_FEATURE_SPECS: dict[str, tuple[str, ...]] = {
    "baseline_price_level": CORE_PAPER_NUMERIC_FEATURES,
    "deleveled_price_gap": ("aligned_price_gap",),
    "discount_mesh_interaction": ("current_discount", "current_discount_x_mesh"),
}
_PREDICTIVE_AUGMENTED_SPECS: dict[str, tuple[str, ...]] = {
    "paper_baseline": CORE_PAPER_NUMERIC_FEATURES,
    "lag_sales_only": ("log_sales",),
    "lag_sales_discount": ("log_sales", "current_discount"),
    "lag_sales_price_gap": ("log_sales", "aligned_price_gap"),
    "lag_sales_full_price": ("log_sales", "current_log_asp", "current_discount"),
}


@dataclass
class PaperEmpiricalStudyArtifacts:
    """Empirical study outputs for the paper track."""

    feature_diagnostics: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    sensitivity_fold_results: pd.DataFrame
    recommendation: pd.DataFrame
    predictive_tv_sensitivity_summary: pd.DataFrame
    predictive_tv_sensitivity_fold_results: pd.DataFrame
    predictive_tv_recommendation: pd.DataFrame
    predictive_summary: pd.DataFrame
    predictive_recommendation: pd.DataFrame


def _factor_mode_to_value(factor_mode: str) -> int | None:
    if factor_mode == "auto_bic":
        return None
    return int(factor_mode.removeprefix("fixed_"))


def _empirical_factor_modes(config: PipelineConfig) -> tuple[str, ...]:
    return ("auto_bic", *(f"fixed_{count}" for count in range(config.paper_max_factor_count + 1)))


def _paper_empirical_feature_specs(config: PipelineConfig) -> list[tuple[str, tuple[str, ...]]]:
    ordered_specs = [("configured_paper_model", tuple(config.paper_feature_columns)), *_EMPIRICAL_FEATURE_SPECS.items()]
    normalized: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()
    for spec_name, feature_columns in ordered_specs:
        normalized_columns = tuple(feature_columns)
        if not normalized_columns or normalized_columns in seen:
            continue
        seen.add(normalized_columns)
        normalized.append((spec_name, normalized_columns))
    return normalized


def _series_effective_rank(values: np.ndarray) -> tuple[float, float]:
    singular_values = np.linalg.svd(values, compute_uv=False)
    power = singular_values**2
    total_power = float(power.sum())
    if total_power <= 0:
        return 0.0, 0.0
    shares = power / total_power
    entropy = float(-(shares * np.log(shares + 1e-12)).sum())
    return float(shares[0]), float(np.exp(entropy))


def _compute_vif(design: pd.DataFrame, column: str) -> float:
    if design.shape[1] <= 1:
        return 1.0

    y = design[column].to_numpy(dtype=float)
    others = design[[name for name in design.columns if name != column]].copy()
    others.insert(0, "intercept", 1.0)
    beta_hat = np.linalg.lstsq(others.to_numpy(dtype=float), y, rcond=None)[0]
    fitted = others.to_numpy(dtype=float) @ beta_hat
    total = float(((y - y.mean()) ** 2).sum())
    if total <= 0:
        return 1.0
    residual = float(((y - fitted) ** 2).sum())
    r_squared = max(0.0, min(1.0 - residual / total, 0.999999))
    return float(1.0 / max(1.0 - r_squared, 1e-6))


def _feature_diagnostics(frame: pd.DataFrame, spec_name: str, feature_columns: tuple[str, ...]) -> list[dict[str, float | int | str]]:
    usable = frame[list(feature_columns)].astype(float)
    filled = usable.fillna(usable.mean())
    design = np.column_stack([np.ones(len(filled), dtype=float), filled.to_numpy(dtype=float)])
    singular_values = np.linalg.svd(design, compute_uv=False)
    condition_number = float(singular_values[0] / max(singular_values[-1], 1e-8))

    correlation = filled.corr()
    max_abs_pairwise_corr = 0.0
    if len(feature_columns) > 1:
        upper = np.triu(np.ones(correlation.shape, dtype=bool), k=1)
        max_abs_pairwise_corr = float(np.nanmax(np.abs(correlation.where(upper).to_numpy())))

    rows: list[dict[str, float | int | str]] = []
    for feature_name in feature_columns:
        matrix = (
            frame.pivot(index="source_month_index", columns="family_id", values=feature_name)
            .sort_index()
            .astype(float)
        )
        filled_matrix = matrix.fillna(matrix.stack().mean()).to_numpy(dtype=float)
        pc1_share, effective_rank = _series_effective_rank(filled_matrix)
        rows.append(
            {
                "spec_name": spec_name,
                "feature_name": feature_name,
                "n_features_in_spec": int(len(feature_columns)),
                "n_families": int(frame["family_id"].nunique()),
                "n_months": int(frame["source_month"].nunique()),
                "missing_share": float(frame[feature_name].isna().mean()),
                "overall_std": float(frame[feature_name].astype(float).std()),
                "mean_within_family_std": float(frame.groupby("family_id")[feature_name].std().mean()),
                "mean_within_month_std": float(frame.groupby("source_month")[feature_name].std().mean()),
                "matrix_pc1_share": pc1_share,
                "matrix_effective_rank": effective_rank,
                "vif": _compute_vif(filled, feature_name),
                "max_abs_pairwise_corr": max_abs_pairwise_corr,
                "design_condition_number": condition_number,
            }
        )
    return rows


def _run_paper_sensitivity(
    paper_training_frame: pd.DataFrame,
    config: PipelineConfig,
    feature_specs: list[tuple[str, tuple[str, ...]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, float | int | str]] = []

    for spec_name, feature_columns in feature_specs:
        for factor_mode in _empirical_factor_modes(config):
            forced_factor_count = _factor_mode_to_value(factor_mode)
            for target_month in config.target_months:
                train = paper_training_frame[paper_training_frame["target_month"].astype(str) < target_month].copy()
                test = paper_training_frame[paper_training_frame["target_month"].astype(str) == target_month].copy()

                model = PaperInspiredFactorForecaster(
                    n_factors=forced_factor_count,
                    feature_columns=feature_columns,
                    factor_candidates=config.factor_candidates,
                    max_factor_count=config.paper_max_factor_count,
                    ic_penalty_variant=config.paper_ic_penalty_variant,
                    ridge_alpha=config.paper_model_alpha,
                    bandwidth_scale=config.paper_bandwidth_scale,
                    bandwidth_method=config.paper_bandwidth_method,
                    bandwidth_pilot_scale=config.paper_bandwidth_pilot_scale,
                    min_window=config.paper_min_window,
                ).fit(train)

                prediction = model.predict(test).to_numpy(dtype=float)
                actual = test["target_sales"].to_numpy(dtype=float)
                denominator = np.abs(actual) + np.abs(prediction)
                valid = denominator > 0
                fold_rows.append(
                    {
                        "spec_name": spec_name,
                        "factor_mode": factor_mode,
                        "feature_columns": ",".join(feature_columns),
                        "target_month": target_month,
                        "selected_factor_count": int(model.selected_factor_count_ or 0),
                        "mae": float(np.mean(np.abs(actual - prediction))),
                        "rmse": float(np.sqrt(np.mean((actual - prediction) ** 2))),
                        "smape": float(
                            np.mean(2.0 * np.abs(actual[valid] - prediction[valid]) / denominator[valid]) if valid.any() else 0.0
                        ),
                        "n_predictions": int(len(test)),
                    }
                )

    fold_results = pd.DataFrame(fold_rows).sort_values(["spec_name", "factor_mode", "target_month"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | int | str]] = []
    for (spec_name, factor_mode, feature_columns), group in fold_results.groupby(
        ["spec_name", "factor_mode", "feature_columns"],
        sort=True,
    ):
        summary_rows.append(
            {
                "spec_name": spec_name,
                "factor_mode": factor_mode,
                "feature_columns": feature_columns,
                "mae": float(group["mae"].mean()),
                "rmse": float(np.sqrt(np.mean(group["rmse"].astype(float) ** 2))),
                "smape": float(group["smape"].mean()),
                "selected_factor_path": ",".join(group["selected_factor_count"].astype(int).astype(str)),
                "selection_consistent": int(group["selected_factor_count"].nunique() == 1),
                "mean_selected_factor_count": float(group["selected_factor_count"].mean()),
                "n_target_months": int(group["target_month"].nunique()),
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["smape", "rmse", "mae", "spec_name", "factor_mode"])
        .reset_index(drop=True)
    )
    return fold_results, summary


def _build_recommendation(sensitivity_summary: pd.DataFrame) -> pd.DataFrame:
    recommendation = sensitivity_summary.sort_values(["smape", "rmse", "mae"]).iloc[0].copy()
    reason = "Lowest average backtest SMAPE across the empirical study."
    metric_tolerance = 1e-10
    stable_auto_candidates = sensitivity_summary[
        (sensitivity_summary["factor_mode"] == "auto_bic")
        & (sensitivity_summary["selection_consistent"] == 1)
    ].sort_values(["smape", "rmse", "mae", "spec_name"])
    for _, stable_auto in stable_auto_candidates.iterrows():
        auto_is_effectively_best = (
            abs(float(stable_auto["smape"]) - float(recommendation["smape"])) <= metric_tolerance
            and abs(float(stable_auto["rmse"]) - float(recommendation["rmse"])) <= metric_tolerance
            and abs(float(stable_auto["mae"]) - float(recommendation["mae"])) <= metric_tolerance
        )
        if auto_is_effectively_best:
            recommendation = stable_auto.copy()
            selected_factor_count = int(round(float(stable_auto["mean_selected_factor_count"])))
            reason = (
                f"Auto BIC selected {selected_factor_count} factors in every backtest fold for this feature set, "
                "so the recommendation keeps the original automatic factor-selection rule."
            )
            break

    return pd.DataFrame(
        [
            {
                "recommended_spec_name": str(recommendation["spec_name"]),
                "recommended_factor_mode": str(recommendation["factor_mode"]),
                "recommended_feature_columns": str(recommendation["feature_columns"]),
                "backtest_mae": float(recommendation["mae"]),
                "backtest_rmse": float(recommendation["rmse"]),
                "backtest_smape": float(recommendation["smape"]),
                "selected_factor_path": str(recommendation["selected_factor_path"]),
                "recommendation_reason": reason,
            }
        ]
    )


def _fit_predictive_tv_model(
    direct_training_frame: pd.DataFrame,
    config: PipelineConfig,
    n_factors: int | None,
) -> LaunchAwarePredictiveTVForecaster:
    return LaunchAwarePredictiveTVForecaster(
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
        n_factors=n_factors,
        mature_history_threshold=config.predictive_tv_ife_mature_history_threshold,
        mature_history_threshold_candidates=config.predictive_tv_ife_mature_history_threshold_candidates,
        mature_blend_weight=config.predictive_tv_ife_mature_blend_weight,
        mature_blend_weight_candidates=config.predictive_tv_ife_mature_blend_weight_candidates,
        enable_inner_validation=config.predictive_tv_ife_enable_inner_validation,
        inner_validation_min_months=config.predictive_tv_ife_inner_validation_min_months,
    ).fit(direct_training_frame)


def _run_predictive_tv_sensitivity(
    direct_training_frame: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, float | int | str]] = []

    for factor_mode in _empirical_factor_modes(config):
        forced_factor_count = _factor_mode_to_value(factor_mode)
        for target_month in config.target_months:
            train = direct_training_frame[direct_training_frame["target_month"].astype(str) < target_month].copy()
            test = direct_training_frame[direct_training_frame["target_month"].astype(str) == target_month].copy()

            model = _fit_predictive_tv_model(train, config, forced_factor_count)
            prediction = model.predict(test).to_numpy(dtype=float)
            actual = test["target_sales"].to_numpy(dtype=float)
            denominator = np.abs(actual) + np.abs(prediction)
            valid = denominator > 0
            fold_rows.append(
                {
                    "factor_mode": factor_mode,
                    "target_month": target_month,
                    "selected_factor_count": int(model.selected_factor_count_ or 0),
                    "selected_launch_factor_count": int(model.selected_launch_factor_count_ or 0),
                    "selected_mature_factor_count": int(model.selected_mature_factor_count_ or 0),
                    "selected_mature_feature_columns": ",".join(model.selected_mature_feature_columns_),
                    "selected_mature_history_threshold": int(model.selected_mature_history_threshold_),
                    "selected_mature_blend_weight": float(model.selected_mature_blend_weight_),
                    "parameter_source": str(model.parameter_source_),
                    "mae": float(np.mean(np.abs(actual - prediction))),
                    "rmse": float(np.sqrt(np.mean((actual - prediction) ** 2))),
                    "smape": float(
                        np.mean(2.0 * np.abs(actual[valid] - prediction[valid]) / denominator[valid]) if valid.any() else 0.0
                    ),
                    "n_predictions": int(len(test)),
                }
            )

    fold_results = pd.DataFrame(fold_rows).sort_values(["factor_mode", "target_month"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | int | str]] = []
    for factor_mode, group in fold_results.groupby("factor_mode", sort=True):
        summary_rows.append(
            {
                "factor_mode": factor_mode,
                "mae": float(group["mae"].mean()),
                "rmse": float(np.sqrt(np.mean(group["rmse"].astype(float) ** 2))),
                "smape": float(group["smape"].mean()),
                "selected_launch_factor_path": ",".join(group["selected_launch_factor_count"].astype(int).astype(str)),
                "selected_mature_factor_path": ",".join(group["selected_mature_factor_count"].astype(int).astype(str)),
                "factor_selection_consistent": int(
                    group["selected_launch_factor_count"].nunique() == 1
                    and group["selected_mature_factor_count"].nunique() == 1
                ),
                "mean_selected_launch_factor_count": float(group["selected_launch_factor_count"].mean()),
                "mean_selected_mature_factor_count": float(group["selected_mature_factor_count"].mean()),
                "selected_mature_feature_path": "|".join(group["selected_mature_feature_columns"].astype(str)),
                "mature_feature_consistent": int(group["selected_mature_feature_columns"].nunique() == 1),
                "selected_mature_history_threshold_path": ",".join(
                    group["selected_mature_history_threshold"].astype(int).astype(str)
                ),
                "mature_history_threshold_consistent": int(group["selected_mature_history_threshold"].nunique() == 1),
                "selected_mature_blend_weight_path": ",".join(
                    group["selected_mature_blend_weight"].map(lambda value: f"{float(value):g}")
                ),
                "mature_blend_weight_consistent": int(group["selected_mature_blend_weight"].nunique() == 1),
                "parameter_source_path": ",".join(group["parameter_source"].astype(str)),
                "parameter_source_consistent": int(group["parameter_source"].nunique() == 1),
                "n_target_months": int(group["target_month"].nunique()),
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["smape", "rmse", "mae", "factor_mode"])
        .reset_index(drop=True)
    )
    return fold_results, summary


def _build_predictive_tv_recommendation(predictive_tv_sensitivity_summary: pd.DataFrame) -> pd.DataFrame:
    recommendation = predictive_tv_sensitivity_summary.sort_values(["smape", "rmse", "mae"]).iloc[0]
    return pd.DataFrame(
        [
            {
                "recommended_factor_mode": str(recommendation["factor_mode"]),
                "backtest_mae": float(recommendation["mae"]),
                "backtest_rmse": float(recommendation["rmse"]),
                "backtest_smape": float(recommendation["smape"]),
                "selected_launch_factor_path": str(recommendation["selected_launch_factor_path"]),
                "selected_mature_factor_path": str(recommendation["selected_mature_factor_path"]),
                "selected_mature_feature_path": str(recommendation["selected_mature_feature_path"]),
                "selected_mature_history_threshold_path": str(recommendation["selected_mature_history_threshold_path"]),
                "selected_mature_blend_weight_path": str(recommendation["selected_mature_blend_weight_path"]),
                "parameter_source_path": str(recommendation["parameter_source_path"]),
                "recommendation_reason": "Lowest average backtest SMAPE across the predictive TV fixed-factor study.",
            }
        ]
    )


def _run_predictive_augmentation_study(
    paper_training_frame: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    for spec_name, feature_columns in _PREDICTIVE_AUGMENTED_SPECS.items():
        actual_values: list[float] = []
        predicted_values: list[float] = []
        for target_month in config.target_months:
            train = paper_training_frame[paper_training_frame["target_month"].astype(str) < target_month].copy()
            test = paper_training_frame[paper_training_frame["target_month"].astype(str) == target_month].copy()
            model = PaperInspiredFactorForecaster(
                n_factors=0,
                feature_columns=feature_columns,
                factor_candidates=config.factor_candidates,
                max_factor_count=config.paper_max_factor_count,
                ic_penalty_variant=config.paper_ic_penalty_variant,
                ridge_alpha=config.paper_model_alpha,
                bandwidth_scale=config.paper_bandwidth_scale,
                bandwidth_method=config.paper_bandwidth_method,
                bandwidth_pilot_scale=config.paper_bandwidth_pilot_scale,
                min_window=config.paper_min_window,
            ).fit(train)
            actual_values.extend(test["target_sales"].astype(float).tolist())
            predicted_values.extend(model.predict(test).astype(float).tolist())

        actual = np.array(actual_values, dtype=float)
        predicted = np.array(predicted_values, dtype=float)
        rows.append(
            {
                "spec_name": spec_name,
                "feature_columns": ",".join(feature_columns),
                "paper_alignment": "strict_paper" if spec_name == "paper_baseline" else "prediction_oriented",
                "factor_mode": "fixed_0",
                "mae": float(np.mean(np.abs(actual - predicted))),
                "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
                "smape": float(smape(actual, predicted)),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["smape", "rmse", "mae", "spec_name"]).reset_index(drop=True)
    recommendation = pd.DataFrame(
        [
            {
                "recommended_spec_name": str(summary.iloc[0]["spec_name"]),
                "recommended_feature_columns": str(summary.iloc[0]["feature_columns"]),
                "paper_alignment": str(summary.iloc[0]["paper_alignment"]),
                "factor_mode": "fixed_0",
                "backtest_mae": float(summary.iloc[0]["mae"]),
                "backtest_rmse": float(summary.iloc[0]["rmse"]),
                "backtest_smape": float(summary.iloc[0]["smape"]),
            }
        ]
    )
    return summary, recommendation


def run_paper_empirical_study(
    direct_training_frame: pd.DataFrame,
    paper_training_frame: pd.DataFrame,
    config: PipelineConfig,
) -> PaperEmpiricalStudyArtifacts:
    """Run paper-track diagnostics plus fixed-factor sensitivity studies."""

    feature_specs = _paper_empirical_feature_specs(config)
    diagnostics_rows: list[dict[str, float | int | str]] = []
    for spec_name, feature_columns in feature_specs:
        diagnostics_rows.extend(_feature_diagnostics(paper_training_frame, spec_name, feature_columns))

    fold_results, sensitivity_summary = _run_paper_sensitivity(paper_training_frame, config, feature_specs)
    recommendation = _build_recommendation(sensitivity_summary)
    predictive_tv_fold_results, predictive_tv_sensitivity_summary = _run_predictive_tv_sensitivity(
        direct_training_frame,
        config,
    )
    predictive_tv_recommendation = _build_predictive_tv_recommendation(predictive_tv_sensitivity_summary)
    predictive_summary, predictive_recommendation = _run_predictive_augmentation_study(paper_training_frame, config)
    return PaperEmpiricalStudyArtifacts(
        feature_diagnostics=pd.DataFrame(diagnostics_rows).sort_values(["spec_name", "feature_name"]).reset_index(drop=True),
        sensitivity_summary=sensitivity_summary,
        sensitivity_fold_results=fold_results,
        recommendation=recommendation,
        predictive_tv_sensitivity_summary=predictive_tv_sensitivity_summary,
        predictive_tv_sensitivity_fold_results=predictive_tv_fold_results,
        predictive_tv_recommendation=predictive_tv_recommendation,
        predictive_summary=predictive_summary,
        predictive_recommendation=predictive_recommendation,
    )
