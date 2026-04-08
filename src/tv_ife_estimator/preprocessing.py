"""Data cleaning and family-level aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import FULL_MONTHS, STATIC_CATEGORICAL_FIELDS, STATIC_NUMERIC_FIELDS, WIFI_GENERATION_MAP


@dataclass(frozen=True)
class FamilyFilterReport:
    """Metadata describing the selected family scope."""

    mode: str
    reference_month: str | None
    total_family_count: int
    kept_family_count: int
    dropped_family_count: int


def clean_text(value: object) -> str | None:
    """Normalize free-text values."""

    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def extract_param_value(param_text: object, key: str) -> str | None:
    """Extract a single value from the parameter blob."""

    if pd.isna(param_text):
        return None
    match = re.search(rf"{re.escape(key)}:\s*([^,，]+)", str(param_text), flags=re.IGNORECASE)
    if not match:
        return None
    return clean_text(match.group(1))


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values."""

    return pd.to_numeric(series, errors="coerce")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    valid = values.notna()
    if not valid.any():
        return None
    values = values[valid].astype(float)
    aligned_weights = weights.loc[values.index].fillna(0).clip(lower=0)
    if aligned_weights.sum() <= 0:
        return float(values.mean())
    return float(np.average(values, weights=aligned_weights))


def _dominant_value(group: pd.DataFrame, column: str) -> object:
    sorted_group = group.sort_values(
        by=["listing_weight", column],
        ascending=[False, True],
        kind="mergesort",
    )
    for value in sorted_group[column]:
        if not pd.isna(value):
            return value
    return np.nan


def _support_flag(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = clean_text(value)
    if text in {"支持", "是", "yes", "true"}:
        return 1.0
    if text in {"不支持", "否", "no", "false"}:
        return 0.0
    return None


def _wifi_generation(protocol: object) -> float | None:
    if pd.isna(protocol):
        return None
    return float(WIFI_GENERATION_MAP.get(clean_text(protocol), np.nan))


def month_sort_key(month: str) -> int:
    """Sortable integer for YYYYMM strings."""

    return int(month)


def apply_family_filter(
    family_lookup: pd.DataFrame,
    static_profile: pd.DataFrame,
    month_panel: pd.DataFrame,
    *,
    mode: str = "all",
    reference_month: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FamilyFilterReport]:
    """Restrict the dataset to a configurable family scope."""

    normalized_mode = str(mode).strip().lower()
    total_family_count = int(month_panel["family_id"].nunique())
    resolved_reference_month = reference_month

    if normalized_mode == "all":
        kept_family_ids = month_panel["family_id"].astype(str).unique().tolist()
    elif normalized_mode == "start_nonzero_only":
        if resolved_reference_month is None:
            resolved_reference_month = min(month_panel["month"].astype(str).unique(), key=month_sort_key)
        reference_rows = month_panel[month_panel["month"].astype(str) == str(resolved_reference_month)].copy()
        if reference_rows.empty:
            raise ValueError(f"No rows found for family filter reference month {resolved_reference_month}.")
        kept_family_ids = (
            reference_rows.loc[reference_rows["sales"].astype(float) > 0, "family_id"].astype(str).drop_duplicates().tolist()
        )
    else:
        raise ValueError(f"Unsupported family_filter_mode: {mode}")

    family_id_set = set(kept_family_ids)
    filtered_lookup = family_lookup[family_lookup["family_id"].astype(str).isin(family_id_set)].copy().reset_index(drop=True)
    filtered_static = static_profile[static_profile["family_id"].astype(str).isin(family_id_set)].copy().reset_index(drop=True)
    filtered_panel = month_panel[month_panel["family_id"].astype(str).isin(family_id_set)].copy().reset_index(drop=True)

    report = FamilyFilterReport(
        mode=normalized_mode,
        reference_month=str(resolved_reference_month) if resolved_reference_month is not None else None,
        total_family_count=total_family_count,
        kept_family_count=int(filtered_panel["family_id"].nunique()),
        dropped_family_count=int(total_family_count - filtered_panel["family_id"].nunique()),
    )
    if report.kept_family_count <= 0:
        raise ValueError(f"Family filter mode {normalized_mode} removed every family.")
    return filtered_lookup, filtered_static, filtered_panel, report


def build_family_lookup(enhanced_market: pd.DataFrame) -> pd.DataFrame:
    """Create the listing-to-family lookup table."""

    frame = enhanced_market.copy()
    frame["brand"] = frame["品牌"].map(clean_text)
    frame["model"] = frame["参数信息"].map(lambda value: extract_param_value(value, "型号"))
    if frame["model"].isna().any():
        missing_count = int(frame["model"].isna().sum())
        raise ValueError(f"Unable to extract model for {missing_count} enhanced listings.")

    frame["family_id"] = frame["brand"].fillna("UNKNOWN") + "||" + frame["model"].fillna("UNKNOWN")
    frame["listing_weight"] = to_numeric(frame["销量总计"]).fillna(0.0)

    return frame[
        [
            "商品链接",
            "宝贝名称",
            "brand",
            "model",
            "family_id",
            "listing_weight",
        ]
    ].sort_values(["family_id", "商品链接"]).reset_index(drop=True)


def build_family_static_profile(
    enhanced_market: pd.DataFrame,
    family_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate listing-level static fields to the family level."""

    frame = enhanced_market.merge(
        family_lookup[["商品链接", "family_id", "brand", "model", "listing_weight"]],
        on="商品链接",
        how="left",
        validate="one_to_one",
    )

    for column in STATIC_NUMERIC_FIELDS:
        frame[column] = to_numeric(frame[column])

    records: list[dict[str, object]] = []
    for family_id, group in frame.groupby("family_id", sort=True):
        dominant = group.sort_values("listing_weight", ascending=False).iloc[0]
        record: dict[str, object] = {
            "family_id": family_id,
            "brand": dominant["brand"],
            "model": dominant["model"],
            "listing_count": int(group["商品链接"].nunique()),
            "ref_price": _weighted_mean(group["参考价格"], group["listing_weight"]),
            "network_protocol": _dominant_value(group, "网络协议"),
            "mesh_grouping": _dominant_value(group, "mesh组网"),
            "wifi_gen": _wifi_generation(_dominant_value(group, "网络协议")),
            "speed_mbps": _weighted_mean(group["传输速率\nMbps"], group["listing_weight"]),
            "wan_lan_ports": _weighted_mean(group["WAN/LAN口数量"], group["listing_weight"]),
            "port25_count": _weighted_mean(group["2.5G网口数"], group["listing_weight"]),
            "port25_missing": int(group["2.5G网口数"].isna().all()),
            "mesh_flag": _support_flag(_dominant_value(group, "mesh组网")),
        }
        records.append(record)

    static_profile = pd.DataFrame.from_records(records).sort_values("family_id").reset_index(drop=True)
    static_profile["log_ref_price"] = np.log1p(static_profile["ref_price"])
    static_profile["log_speed_mbps"] = np.log1p(static_profile["speed_mbps"])
    return static_profile


def build_family_month_panel(
    enhanced_market: pd.DataFrame,
    family_lookup: pd.DataFrame,
    months: Iterable[str] = FULL_MONTHS,
) -> pd.DataFrame:
    """Create a balanced family-month panel."""

    months = tuple(months)
    frame = enhanced_market.merge(
        family_lookup[["商品链接", "family_id", "brand", "listing_weight"]],
        on="商品链接",
        how="left",
        validate="one_to_one",
    )

    family_order = family_lookup[["family_id", "brand"]].drop_duplicates().sort_values("family_id")
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        for month in months:
            sales = pd.to_numeric(pd.Series([row[f"{month}销量"]]), errors="coerce").iloc[0]
            revenue = pd.to_numeric(pd.Series([row[f"{month}销售额"]]), errors="coerce").iloc[0]
            records.append(
                {
                    "family_id": row["family_id"],
                    "brand": row["brand"],
                    "month": month,
                    "sales": float(sales) if not pd.isna(sales) else 0.0,
                    "revenue": float(revenue) if not pd.isna(revenue) else 0.0,
                }
            )

    long_panel = pd.DataFrame.from_records(records)
    aggregated = (
        long_panel.groupby(["family_id", "brand", "month"], as_index=False)[["sales", "revenue"]]
        .sum()
        .sort_values(["family_id", "month"])
    )

    full_index = (
        family_order.assign(_key=1)
        .merge(pd.DataFrame({"month": months, "_key": 1}), on="_key", how="outer")
        .drop(columns="_key")
    )
    aggregated = full_index.merge(
        aggregated,
        on=["family_id", "brand", "month"],
        how="left",
        validate="one_to_one",
    )
    aggregated["sales"] = aggregated["sales"].fillna(0.0)
    aggregated["revenue"] = aggregated["revenue"].fillna(0.0)
    aggregated["asp"] = np.where(
        aggregated["sales"] > 0,
        aggregated["revenue"] / aggregated["sales"],
        np.nan,
    )
    aggregated["log_sales"] = np.log1p(aggregated["sales"])
    month_index = {month: idx + 1 for idx, month in enumerate(sorted(months, key=month_sort_key))}
    aggregated["month_index"] = aggregated["month"].map(month_index).astype(int)
    return aggregated.sort_values(["family_id", "month"]).reset_index(drop=True)


def build_diagnostics(
    family_lookup: pd.DataFrame,
    static_profile: pd.DataFrame,
    month_panel: pd.DataFrame,
) -> dict[str, object]:
    """Assemble high-level diagnostics for export."""

    zero_sales_share = float((month_panel["sales"] <= 0).mean())
    return {
        "family_count": int(static_profile["family_id"].nunique()),
        "listing_count": int(family_lookup["商品链接"].nunique()),
        "duplicated_family_count": int((family_lookup["family_id"].value_counts() > 1).sum()),
        "panel_rows": int(len(month_panel)),
        "zero_sales_share": zero_sales_share,
        "port25_missing_family_count": int(static_profile["port25_missing"].sum()),
        "mesh_missing_family_count": int(static_profile["mesh_flag"].isna().sum()),
    }
