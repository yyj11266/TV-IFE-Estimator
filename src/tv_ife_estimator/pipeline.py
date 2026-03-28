"""Top-level project pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestArtifacts, fit_final_models_and_forecast, run_backtest
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
    _save_csv(artifacts.next_month_forecast, paths.output_dir / "next_month_forecast.csv")

    paper_artifacts = final_paper_model.export_artifacts()
    _save_csv(paper_artifacts.coefficients, paths.output_dir / "paper_track_coefficients.csv")
    _save_csv(paper_artifacts.factors, paths.output_dir / "paper_track_factors.csv")
    _save_csv(paper_artifacts.loadings, paths.output_dir / "paper_track_loadings.csv")

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
    backtest = run_backtest(direct_training_frame, paper_training_frame, config)

    next_target_month = "202407"
    direct_forecast_frame = build_direct_forecast_frame(family_month_panel, family_static_profile, source_month="202406")
    paper_forecast_frame = build_paper_forecast_frame(family_month_panel, family_static_profile, source_month="202406")
    next_month_forecast, final_paper_model = fit_final_models_and_forecast(
        direct_training_frame=direct_training_frame,
        paper_training_frame=paper_training_frame,
        direct_forecast_frame=direct_forecast_frame,
        paper_forecast_frame=paper_forecast_frame,
        next_target_month=next_target_month,
        config=config,
    )

    artifacts = PipelineArtifacts(
        subset_validation=subset_validation,
        diagnostics=diagnostics,
        family_lookup=family_lookup,
        family_static_profile=family_static_profile,
        family_month_panel=family_month_panel,
        direct_training_frame=direct_training_frame,
        paper_training_frame=paper_training_frame,
        backtest=backtest,
        next_month_forecast=next_month_forecast,
    )
    _write_outputs(paths, artifacts, final_paper_model)
    return artifacts
