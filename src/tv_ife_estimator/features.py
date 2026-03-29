"""Feature construction for model-ready tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _add_paper_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add paper-track regressors that remain available at forecast time."""

    asp_proxy = frame["asp"].fillna(frame["ref_price"])
    discount_proxy = frame["asp"] / frame["ref_price"]
    discount_proxy = discount_proxy.where(frame["asp"].notna(), 1.0)

    frame["current_log_asp"] = np.log1p(asp_proxy)
    frame["current_discount"] = discount_proxy
    frame["current_price_gap"] = frame["current_log_asp"] - frame["log_ref_price"]
    frame["current_discount_x_mesh"] = frame["current_discount"] * frame["mesh_flag"]
    frame["current_discount_x_wifi"] = frame["current_discount"] * frame["wifi_gen"]
    return frame


def build_direct_training_frame(
    month_panel: pd.DataFrame,
    static_profile: pd.DataFrame,
) -> pd.DataFrame:
    """Create the direct-model training frame."""

    frame = month_panel.merge(
        static_profile,
        on=["family_id", "brand"],
        how="left",
        validate="many_to_one",
    ).sort_values(["family_id", "month_index"])

    grouped = frame.groupby("family_id", group_keys=False)
    frame["prev_sales"] = grouped["sales"].shift(1)
    frame["prev_prev_sales"] = grouped["sales"].shift(2)
    frame["prev_asp"] = grouped["asp"].shift(1)
    frame["prev_two_month_mean"] = (frame["prev_sales"] + frame["prev_prev_sales"]) / 2.0
    frame["prev_sales_growth"] = np.where(
        frame["prev_prev_sales"] > 0,
        frame["prev_sales"] / frame["prev_prev_sales"],
        np.nan,
    )
    frame["prev_discount"] = frame["prev_asp"] / frame["ref_price"]
    frame["log_prev_sales"] = np.log1p(frame["prev_sales"])
    frame["log_prev_prev_sales"] = np.log1p(frame["prev_prev_sales"])
    frame["log_prev_two_month_mean"] = np.log1p(frame["prev_two_month_mean"])
    frame["target_month"] = frame["month"]
    frame["target_sales"] = frame["sales"]
    frame["target_log_sales"] = frame["log_sales"]
    frame["target_delta_log_sales"] = frame["target_log_sales"] - frame["log_prev_sales"]
    frame["source_month"] = grouped["month"].shift(1)
    frame["source_month_index"] = grouped["month_index"].shift(1)

    usable = frame[frame["prev_prev_sales"].notna()].copy()
    usable["prev_sales_growth"] = usable["prev_sales_growth"].replace([np.inf, -np.inf], np.nan).clip(lower=0, upper=8)
    usable["prev_discount"] = usable["prev_discount"].replace([np.inf, -np.inf], np.nan).clip(lower=0, upper=3)
    return usable.reset_index(drop=True)


def build_direct_forecast_frame(
    month_panel: pd.DataFrame,
    static_profile: pd.DataFrame,
    source_month: str,
) -> pd.DataFrame:
    """Create the direct-model frame for the next-month forecast."""

    frame = month_panel.merge(
        static_profile,
        on=["family_id", "brand"],
        how="left",
        validate="many_to_one",
    ).sort_values(["family_id", "month_index"])

    grouped = frame.groupby("family_id", group_keys=False)
    frame["prev_sales"] = grouped["sales"].shift(1)
    frame["prev_prev_sales"] = grouped["sales"].shift(2)
    frame["prev_asp"] = frame["asp"]
    frame["prev_two_month_mean"] = (frame["sales"] + frame["prev_sales"]) / 2.0
    frame["prev_sales_growth"] = np.where(
        frame["prev_sales"] > 0,
        frame["sales"] / frame["prev_sales"],
        np.nan,
    )
    frame["prev_discount"] = frame["prev_asp"] / frame["ref_price"]
    frame["log_prev_sales"] = np.log1p(frame["sales"])
    frame["log_prev_prev_sales"] = np.log1p(frame["prev_prev_sales"])
    frame["log_prev_two_month_mean"] = np.log1p(frame["prev_two_month_mean"])
    frame["source_month"] = frame["month"]
    frame["source_month_index"] = frame["month_index"]

    source_rows = frame[frame["source_month"] == source_month].copy()
    if source_rows.empty:
        raise ValueError(f"No direct-model forecast rows found for source month {source_month}.")
    source_rows["prev_sales_growth"] = source_rows["prev_sales_growth"].replace([np.inf, -np.inf], np.nan).clip(
        lower=0,
        upper=8,
    )
    source_rows["prev_discount"] = source_rows["prev_discount"].replace([np.inf, -np.inf], np.nan).clip(
        lower=0,
        upper=3,
    )
    return source_rows.reset_index(drop=True)


def build_paper_training_frame(
    month_panel: pd.DataFrame,
    static_profile: pd.DataFrame,
) -> pd.DataFrame:
    """Create the paper-inspired training frame."""

    frame = month_panel.merge(
        static_profile,
        on=["family_id", "brand"],
        how="left",
        validate="many_to_one",
    ).sort_values(["family_id", "month_index"])

    grouped = frame.groupby("family_id", group_keys=False)
    frame["target_sales"] = grouped["sales"].shift(-1)
    frame["target_log_sales"] = np.log1p(frame["target_sales"])
    frame["target_month"] = grouped["month"].shift(-1)
    frame["target_month_index"] = grouped["month_index"].shift(-1)
    frame["source_month"] = frame["month"]
    frame["source_month_index"] = frame["month_index"]
    frame = _add_paper_feature_columns(frame)

    usable = frame[frame["target_month"].notna()].copy()
    usable["current_discount"] = usable["current_discount"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    usable["current_price_gap"] = usable["current_price_gap"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    usable["current_discount_x_mesh"] = usable["current_discount_x_mesh"].replace([np.inf, -np.inf], np.nan)
    usable["current_discount_x_wifi"] = usable["current_discount_x_wifi"].replace([np.inf, -np.inf], np.nan)
    return usable.reset_index(drop=True)


def build_paper_forecast_frame(
    month_panel: pd.DataFrame,
    static_profile: pd.DataFrame,
    source_month: str,
) -> pd.DataFrame:
    """Create the paper-inspired forecast frame for the next month."""

    frame = month_panel.merge(
        static_profile,
        on=["family_id", "brand"],
        how="left",
        validate="many_to_one",
    ).sort_values(["family_id", "month_index"])
    source_rows = frame[frame["month"] == source_month].copy()
    if source_rows.empty:
        raise ValueError(f"No paper-model forecast rows found for source month {source_month}.")
    source_rows["source_month"] = source_rows["month"]
    source_rows["source_month_index"] = source_rows["month_index"]
    source_rows = _add_paper_feature_columns(source_rows)
    source_rows["current_discount"] = source_rows["current_discount"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    source_rows["current_price_gap"] = source_rows["current_price_gap"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    source_rows["current_discount_x_mesh"] = source_rows["current_discount_x_mesh"].replace([np.inf, -np.inf], np.nan)
    source_rows["current_discount_x_wifi"] = source_rows["current_discount_x_wifi"].replace([np.inf, -np.inf], np.nan)
    return source_rows.reset_index(drop=True)
