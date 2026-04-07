"""Top-level project pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestArtifacts, fit_final_models_and_forecast, run_backtest, summarize_evaluation
from .config import PipelineConfig, ProjectPaths, default_paths
from .features import (
    build_direct_forecast_frame,
    build_direct_training_frame,
    build_paper_forecast_frame,
    build_paper_training_frame,
)
from .io import SubsetValidationReport, load_enhanced_market, load_raw_market, validate_subset
from .preprocessing import (
    build_diagnostics,
    build_family_lookup,
    build_family_month_panel,
    build_family_static_profile,
)
from .research import PaperEmpiricalStudyArtifacts, run_paper_empirical_study


@dataclass
class PipelineArtifacts:
    """All exported pipeline artifacts."""

    subset_validation: SubsetValidationReport
    diagnostics: dict[str, object]
    family_lookup: pd.DataFrame
    family_static_profile: pd.DataFrame
    family_month_panel: pd.DataFrame
    direct_training_frame: pd.DataFrame
    paper_training_frame: pd.DataFrame
    paper_empirical_study: PaperEmpiricalStudyArtifacts
    backtest: BacktestArtifacts
    next_month_forecast: pd.DataFrame


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_outputs(
    paths: ProjectPaths,
    artifacts: PipelineArtifacts,
    final_paper_model,
    final_predictive_tv_model,
) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    _save_csv(artifacts.family_lookup, paths.output_dir / "family_lookup.csv")
    _save_csv(artifacts.family_static_profile, paths.output_dir / "family_static_profile.csv")
    _save_csv(artifacts.family_month_panel, paths.output_dir / "family_month_panel.csv")
    _save_csv(artifacts.direct_training_frame, paths.output_dir / "direct_training_frame.csv")
    _save_csv(artifacts.paper_training_frame, paths.output_dir / "paper_training_frame.csv")
    _save_csv(artifacts.backtest.evaluation, paths.output_dir / "forecast_eval.csv")
    _save_csv(artifacts.backtest.summary, paths.output_dir / "model_summary.csv")
    _save_csv(artifacts.backtest.paper_selection, paths.output_dir / "paper_factor_selection.csv")
    _save_csv(
        artifacts.paper_empirical_study.feature_diagnostics,
        paths.output_dir / "paper_empirical_feature_diagnostics.csv",
    )
    _save_csv(
        artifacts.paper_empirical_study.sensitivity_summary,
        paths.output_dir / "paper_empirical_sensitivity_summary.csv",
    )
    _save_csv(
        artifacts.paper_empirical_study.sensitivity_fold_results,
        paths.output_dir / "paper_empirical_sensitivity_folds.csv",
    )
    _save_csv(
        artifacts.paper_empirical_study.recommendation,
        paths.output_dir / "paper_empirical_recommendation.csv",
    )
    _save_csv(
        artifacts.paper_empirical_study.predictive_tv_sensitivity_summary,
        paths.output_dir / "predictive_tv_ife_factor_sensitivity_summary.csv",
    )
    _save_csv(
        artifacts.paper_empirical_study.predictive_tv_sensitivity_fold_results,
        paths.output_dir / "predictive_tv_ife_factor_sensitivity_folds.csv",
    )
    _save_csv(
        artifacts.paper_empirical_study.predictive_tv_recommendation,
        paths.output_dir / "predictive_tv_ife_factor_recommendation.csv",
    )
    _save_csv(
        artifacts.paper_empirical_study.predictive_summary,
        paths.output_dir / "paper_predictive_augmented_summary.csv",
    )
    _save_csv(
        artifacts.paper_empirical_study.predictive_recommendation,
        paths.output_dir / "paper_predictive_augmented_recommendation.csv",
    )
    _save_csv(artifacts.next_month_forecast, paths.output_dir / "next_month_forecast.csv")

    paper_artifacts = final_paper_model.export_artifacts()
    _save_csv(paper_artifacts.coefficients, paths.output_dir / "paper_track_coefficients.csv")
    _save_csv(paper_artifacts.factors, paths.output_dir / "paper_track_factors.csv")
    _save_csv(paper_artifacts.loadings, paths.output_dir / "paper_track_loadings.csv")
    _save_csv(paper_artifacts.selection, paths.output_dir / "paper_track_ic.csv")
    _save_csv(paper_artifacts.fitted, paths.output_dir / "paper_track_fitted.csv")
    _save_csv(paper_artifacts.fit_summary, paths.output_dir / "paper_track_fit_summary.csv")

    predictive_artifacts = final_predictive_tv_model.export_artifacts()
    _save_csv(predictive_artifacts.coefficients, paths.output_dir / "predictive_tv_ife_coefficients.csv")
    _save_csv(predictive_artifacts.factors, paths.output_dir / "predictive_tv_ife_factors.csv")
    _save_csv(predictive_artifacts.loadings, paths.output_dir / "predictive_tv_ife_loadings.csv")
    _save_csv(predictive_artifacts.selection, paths.output_dir / "predictive_tv_ife_ic.csv")
    _save_csv(predictive_artifacts.fitted, paths.output_dir / "predictive_tv_ife_fitted.csv")
    _save_csv(predictive_artifacts.fit_summary, paths.output_dir / "predictive_tv_ife_fit_summary.csv")

    diagnostics_payload = {
        "subset_validation": {
            "raw_shape": artifacts.subset_validation.raw_shape,
            "enhanced_shape": artifacts.subset_validation.enhanced_shape,
            "shared_columns": artifacts.subset_validation.shared_columns,
            "all_links_present": artifacts.subset_validation.all_links_present,
            "differing_columns": list(artifacts.subset_validation.differing_columns),
        },
        "panel_diagnostics": artifacts.diagnostics,
    }
    with (paths.output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(diagnostics_payload), handle, ensure_ascii=False, indent=2)


def run_pipeline(
    project_root: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> PipelineArtifacts:
    """Run the full pipeline and persist outputs."""

    config = config or PipelineConfig()
    paths = default_paths(project_root)

    raw_market = load_raw_market(paths.raw_market_path)
    enhanced_market = load_enhanced_market(paths.enhanced_market_path)
    subset_validation = validate_subset(raw_market, enhanced_market)

    family_lookup = build_family_lookup(enhanced_market)
    family_static_profile = build_family_static_profile(enhanced_market, family_lookup)
    family_month_panel = build_family_month_panel(enhanced_market, family_lookup, config.full_months)
    diagnostics = build_diagnostics(family_lookup, family_static_profile, family_month_panel)

    direct_training_frame = build_direct_training_frame(family_month_panel, family_static_profile)
    paper_training_frame = build_paper_training_frame(family_month_panel, family_static_profile)
    paper_empirical_study = run_paper_empirical_study(direct_training_frame, paper_training_frame, config)
    backtest = run_backtest(direct_training_frame, paper_training_frame, config)
    backtest.summary = summarize_evaluation(backtest.evaluation)
    configured_paper_signature = ",".join(config.paper_feature_columns)
    configured_paper_recommendations = paper_empirical_study.sensitivity_summary[
        paper_empirical_study.sensitivity_summary["feature_columns"].astype(str) == configured_paper_signature
    ]
    if configured_paper_recommendations.empty:
        recommended_factor_mode = str(paper_empirical_study.recommendation.iloc[0]["recommended_factor_mode"])
    else:
        configured_recommendation_row = configured_paper_recommendations.sort_values(
            ["smape", "rmse", "mae", "spec_name", "factor_mode"]
        ).iloc[0]
        recommended_factor_mode = str(configured_recommendation_row["factor_mode"])
    recommended_n_factors = (
        int(recommended_factor_mode.removeprefix("fixed_"))
        if recommended_factor_mode.startswith("fixed_")
        else None
    )

    next_target_month = "202407"
    direct_forecast_frame = build_direct_forecast_frame(family_month_panel, family_static_profile, source_month="202406")
    paper_forecast_frame = build_paper_forecast_frame(family_month_panel, family_static_profile, source_month="202406")
    next_month_forecast, final_paper_model, final_predictive_tv_model = fit_final_models_and_forecast(
        direct_training_frame=direct_training_frame,
        paper_training_frame=paper_training_frame,
        direct_forecast_frame=direct_forecast_frame,
        paper_forecast_frame=paper_forecast_frame,
        next_target_month=next_target_month,
        config=config,
        paper_n_factors=recommended_n_factors,
        paper_feature_columns=config.paper_feature_columns,
    )

    artifacts = PipelineArtifacts(
        subset_validation=subset_validation,
        diagnostics=diagnostics,
        family_lookup=family_lookup,
        family_static_profile=family_static_profile,
        family_month_panel=family_month_panel,
        direct_training_frame=direct_training_frame,
        paper_training_frame=paper_training_frame,
        paper_empirical_study=paper_empirical_study,
        backtest=backtest,
        next_month_forecast=next_month_forecast,
    )
    _write_outputs(paths, artifacts, final_paper_model, final_predictive_tv_model)
    return artifacts
