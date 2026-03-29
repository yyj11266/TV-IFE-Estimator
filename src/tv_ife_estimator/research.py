"""Empirical diagnostics and sensitivity study for the paper track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import smape
from .config import PipelineConfig
from .constants import CORE_PAPER_NUMERIC_FEATURES
from .models.paper_proxy import PaperInspiredFactorForecaster

_EMPIRICAL_FEATURE_SPECS: dict[str, tuple[str, ...]] = {
    "baseline_price_level": CORE_PAPER_NUMERIC_FEATURES,
    "deleveled_price_gap": ("current_price_gap",),
    "discount_mesh_interaction": ("current_discount", "current_discount_x_mesh"),
}
_EMPIRICAL_FACTOR_MODES: tuple[str, ...] = ("auto_ic", "fixed_0", "fixed_1", "fixed_2")
_PREDICTIVE_AUGMENTED_SPECS: dict[str, tuple[str, ...]] = {
    "paper_baseline": CORE_PAPER_NUMERIC_FEATURES,
    "lag_sales_only": ("log_sales",),
    "lag_sales_discount": ("log_sales", "current_discount"),
    "lag_sales_price_gap": ("log_sales", "current_price_gap"),
    "lag_sales_full_price": ("log_sales", "current_log_asp", "current_discount"),
}


@dataclass
class PaperEmpiricalStudyArtifacts:
    """Empirical study outputs for the paper track."""

    feature_diagnostics: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    sensitivity_fold_results: pd.DataFrame
    recommendation: pd.DataFrame
    predictive_summary: pd.DataFrame
    predictive_recommendation: pd.DataFrame


def _factor_mode_to_value(factor_mode: str) -> int | None:
    if factor_mode == "auto_ic":
        return None
    return int(factor_mode.removeprefix("fixed_"))


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, float | int | str]] = []

    for spec_name, feature_columns in _EMPIRICAL_FEATURE_SPECS.items():
        for factor_mode in _EMPIRICAL_FACTOR_MODES:
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

    stable_auto_zero = sensitivity_summary[
        (sensitivity_summary["factor_mode"] == "auto_ic")
        & (sensitivity_summary["selected_factor_path"].str.fullmatch(r"0(,0)*"))
    ]
    if not stable_auto_zero.empty:
        stable_auto_zero = stable_auto_zero.sort_values(["smape", "rmse", "mae"]).iloc[0]
        fixed_zero = sensitivity_summary[
            (sensitivity_summary["spec_name"] == stable_auto_zero["spec_name"])
            & (sensitivity_summary["factor_mode"] == "fixed_0")
        ]
        if not fixed_zero.empty:
            fixed_zero = fixed_zero.iloc[0]
            if abs(float(fixed_zero["smape"]) - float(stable_auto_zero["smape"])) <= 1e-12:
                recommendation = fixed_zero.copy()
                reason = (
                    "Auto IC selected 0 factors in every backtest fold for this feature set, "
                    "so the deployment recommendation is fixed_0 to avoid small-sample factor instability."
                )

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
    paper_training_frame: pd.DataFrame,
    config: PipelineConfig,
) -> PaperEmpiricalStudyArtifacts:
    """Run paper-track diagnostics and a small-sample sensitivity study."""

    diagnostics_rows: list[dict[str, float | int | str]] = []
    for spec_name, feature_columns in _EMPIRICAL_FEATURE_SPECS.items():
        diagnostics_rows.extend(_feature_diagnostics(paper_training_frame, spec_name, feature_columns))

    fold_results, sensitivity_summary = _run_paper_sensitivity(paper_training_frame, config)
    recommendation = _build_recommendation(sensitivity_summary)
    predictive_summary, predictive_recommendation = _run_predictive_augmentation_study(paper_training_frame, config)
    return PaperEmpiricalStudyArtifacts(
        feature_diagnostics=pd.DataFrame(diagnostics_rows).sort_values(["spec_name", "feature_name"]).reset_index(drop=True),
        sensitivity_summary=sensitivity_summary,
        sensitivity_fold_results=fold_results,
        recommendation=recommendation,
        predictive_summary=predictive_summary,
        predictive_recommendation=predictive_recommendation,
    )
