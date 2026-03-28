"""Input/output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import ENHANCED_SHEET_NAME, RAW_SHEET_NAME


@dataclass(frozen=True)
class SubsetValidationReport:
    """Validation result for raw/enhanced overlap."""

    raw_shape: tuple[int, int]
    enhanced_shape: tuple[int, int]
    shared_columns: int
    all_links_present: bool
    differing_columns: tuple[str, ...]


def _load_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_excel(path, sheet_name=sheet_name)


def load_raw_market(path: Path) -> pd.DataFrame:
    """Load the raw market sheet."""

    return _load_excel(path, RAW_SHEET_NAME)


def load_enhanced_market(path: Path) -> pd.DataFrame:
    """Load the enhanced flagship sheet."""

    return _load_excel(path, ENHANCED_SHEET_NAME)


def _normalize_time_like(value: object) -> str:
    if pd.isna(value):
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value).strip()
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def validate_subset(raw_market: pd.DataFrame, enhanced_market: pd.DataFrame) -> SubsetValidationReport:
    """Validate that enhanced rows are a strict subset of the raw data."""

    shared_columns = [column for column in enhanced_market.columns if column in raw_market.columns]
    differing_columns: list[str] = []

    raw_indexed = raw_market.set_index("商品链接")
    enhanced_indexed = enhanced_market.set_index("商品链接")
    all_links_present = enhanced_indexed.index.isin(raw_indexed.index).all()

    if not all_links_present:
        missing = sorted(set(enhanced_indexed.index) - set(raw_indexed.index))
        raise ValueError(f"Enhanced data contains links missing from raw data: {missing[:5]}")

    for column in shared_columns:
        if column == "商品链接":
            continue
        raw_series = raw_indexed.loc[enhanced_indexed.index, column]
        enhanced_series = enhanced_indexed[column]
        if column == "更新时间":
            raw_values = raw_series.map(_normalize_time_like)
            enhanced_values = enhanced_series.map(_normalize_time_like)
        else:
            raw_values = raw_series.astype(str).where(~raw_series.isna(), "")
            enhanced_values = enhanced_series.astype(str).where(~enhanced_series.isna(), "")
        if not raw_values.equals(enhanced_values):
            differing_columns.append(column)

    unexpected_diffs = tuple(column for column in differing_columns if column != "更新时间")
    if unexpected_diffs:
        raise ValueError(f"Unexpected shared-column differences found: {unexpected_diffs}")

    return SubsetValidationReport(
        raw_shape=tuple(raw_market.shape),
        enhanced_shape=tuple(enhanced_market.shape),
        shared_columns=len(shared_columns),
        all_links_present=all_links_present,
        differing_columns=tuple(differing_columns),
    )
