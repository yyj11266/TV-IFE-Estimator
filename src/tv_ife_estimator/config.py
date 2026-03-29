"""Path and runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    DEFAULT_TARGET_MONTHS,
    ENHANCED_FILENAME,
    FULL_MONTHS,
    CORE_PAPER_NUMERIC_FEATURES,
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
    paper_feature_columns: tuple[str, ...] = CORE_PAPER_NUMERIC_FEATURES
    factor_candidates: tuple[int, ...] = ()
    paper_max_factor_count: int = 6
    paper_ic_penalty_variant: str = "rho2"
    direct_model_alpha: float = 1.0
    paper_model_alpha: float = 1e-6
    paper_bandwidth_scale: float = 0.5
    paper_min_window: float = 0.0
    predictive_tv_ife_feature_columns: tuple[str, ...] = (
        "log_prev_sales",
        "log_prev_prev_sales",
        "prev_sales_growth",
    )
    predictive_tv_ife_n_factors: int = 0
    predictive_tv_ife_ridge_alpha: float = 3e-2
    predictive_tv_ife_bandwidth_scale: float = 0.65


def default_paths(root: str | Path | None = None) -> ProjectPaths:
    """Build project paths from a root directory."""

    return ProjectPaths(root=Path(root or Path.cwd()))
