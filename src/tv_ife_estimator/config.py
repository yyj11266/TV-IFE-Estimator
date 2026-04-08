"""Path and runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    DEFAULT_TARGET_MONTHS,
    ENHANCED_FILENAME,
    FULL_MONTHS,
    RAW_FILENAME,
)


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths."""

    root: Path
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    raw_market_path: Path = field(init=False)
    enhanced_market_path: Path = field(init=False)

    def __post_init__(self) -> None:
        root = self.root.resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "data_dir", root / "data")
        object.__setattr__(self, "output_dir", root / "outputs")
        object.__setattr__(self, "raw_market_path", root / "data" / RAW_FILENAME)
        object.__setattr__(self, "enhanced_market_path", root / "data" / ENHANCED_FILENAME)


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline behavior configuration."""

    full_months: tuple[str, ...] = FULL_MONTHS
    target_months: tuple[str, ...] = DEFAULT_TARGET_MONTHS
    family_filter_mode: str = "all"
    family_filter_reference_month: str | None = None
    paper_feature_columns: tuple[str, ...] = (
        "log_prev_sales",
        "log_prev_prev_sales",
        "prev_sales_growth",
        "prev_discount",
        "log_ref_price",
    )
    factor_candidates: tuple[int, ...] = ()
    paper_max_factor_count: int = 6
    paper_ic_penalty_variant: str = "rho2"
    direct_model_alpha: float = 1.0
    paper_model_alpha: float = 1e-6
    paper_bandwidth_scale: float = 0.5
    paper_bandwidth_method: str = "paper_residual"
    paper_bandwidth_pilot_scale: float = 2.0
    paper_min_window: float = 0.0
    predictive_tv_ife_feature_columns: tuple[str, ...] = (
        "log_prev_sales",
        "log_prev_prev_sales",
        "prev_sales_growth",
    )
    predictive_tv_ife_mature_feature_columns: tuple[str, ...] = (
        "log_prev_sales",
        "log_prev_prev_sales",
        "prev_sales_growth",
        "prev_discount",
        "log_ref_price",
    )
    predictive_tv_ife_mature_feature_candidates: tuple[tuple[str, ...], ...] = (
        (
            "log_prev_sales",
            "log_prev_prev_sales",
            "prev_sales_growth",
            "prev_discount",
            "log_ref_price",
        ),
        (
            "log_prev_sales",
            "log_prev_prev_sales",
            "prev_sales_growth",
        ),
    )
    predictive_tv_ife_mature_history_threshold: int = 4
    predictive_tv_ife_mature_history_threshold_candidates: tuple[int, ...] = (3, 4, 5)
    predictive_tv_ife_mature_blend_weight: float = 0.6
    predictive_tv_ife_mature_blend_weight_candidates: tuple[float, ...] = (0.5, 0.6, 0.7)
    predictive_tv_ife_enable_inner_validation: bool = False
    predictive_tv_ife_inner_validation_min_months: int = 5
    predictive_tv_ife_n_factors: int | None = 0
    predictive_tv_ife_ridge_alpha: float = 3e-2
    predictive_tv_ife_bandwidth_scale: float = 0.65
    predictive_tv_ife_bandwidth_method: str = "response_std"
    predictive_tv_ife_bandwidth_pilot_scale: float = 2.0


def default_paths(root: str | Path | None = None) -> ProjectPaths:
    """Build project paths from a root directory."""

    return ProjectPaths(root=Path(root or Path.cwd()))
