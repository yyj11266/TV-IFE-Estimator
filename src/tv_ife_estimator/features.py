"""Feature construction for model-ready tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned_price_gap(price_proxy: pd.Series, log_ref_price: pd.Series) -> pd.Series:
    """Shared price-gap feature used by both paper and predictive TV models."""

    gap = np.log1p(price_proxy.astype(float)) - log_ref_price.astype(float)
    return gap.replace([np.inf, -np.inf], np.nan)


def _attach_launch_history_columns(
    frame: pd.DataFrame,
    *,
    source_includes_current_month: bool,
) -> pd.DataFrame:
    positive_sales = frame["sales"].fillna(0).gt(0).astype(int)
    cumulative_positive = positive_sales.groupby(frame["family_id"]).cumsum()
    if source_includes_current_month:
        source_positive_history = cumulative_positive
    else:
        source_positive_history = cumulative_positive.groupby(frame["family_id"]).shift(1).fillna(0)

    first_sale_month_index = frame["month_index"].where(positive_sales > 0).groupby(frame["family_id"]).transform("min")
    frame["first_sale_month_index"] = first_sale_month_index
    frame["source_positive_history_count"] = source_positive_history.astype(float)
    frame["source_has_sales_history"] = frame["source_positive_history_count"].gt(0).astype(int)
    frame["is_prelaunch_target"] = np.where(
        frame["first_sale_month_index"].notna(),
        frame["month_index"] < frame["first_sale_month_index"],
        True,
    ).astype(int)
    return frame


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


def _add_paper_predictive_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add predictive-style lag features using the paper track's source month."""

    grouped = frame.groupby("family_id", group_keys=False)
    previous_sales = grouped["sales"].shift(1)
    frame["log_prev_sales"] = frame["log_sales"]
    frame["log_prev_prev_sales"] = grouped["log_sales"].shift(1)
    frame["prev_sales_growth"] = np.where(
        previous_sales > 0,
        frame["sales"] / previous_sales,
        np.nan,
    )
    frame["prev_discount"] = frame["asp"] / frame["ref_price"]
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
    frame["aligned_price_gap"] = _aligned_price_gap(frame["prev_asp"].fillna(frame["ref_price"]), frame["log_ref_price"])
    frame["target_month"] = frame["month"]
    frame["target_sales"] = frame["sales"]
    frame["target_log_sales"] = frame["log_sales"]
    frame["target_delta_log_sales"] = frame["target_log_sales"] - frame["log_prev_sales"]
    frame["source_month"] = grouped["month"].shift(1)
    frame["source_month_index"] = grouped["month_index"].shift(1)
    frame = _attach_launch_history_columns(frame, source_includes_current_month=False)
    frame["target_month_index"] = frame["month_index"]

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
    frame["aligned_price_gap"] = _aligned_price_gap(frame["prev_asp"].fillna(frame["ref_price"]), frame["log_ref_price"])
    frame["source_month"] = frame["month"]
    frame["source_month_index"] = frame["month_index"]
    frame = _attach_launch_history_columns(frame, source_includes_current_month=True)

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
    frame = _add_paper_predictive_feature_columns(frame)
    frame["aligned_price_gap"] = frame["current_price_gap"]

    usable = frame[frame["target_month"].notna()].copy()
    usable["current_discount"] = usable["current_discount"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    usable["current_price_gap"] = usable["current_price_gap"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    usable["aligned_price_gap"] = usable["aligned_price_gap"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    usable["prev_sales_growth"] = usable["prev_sales_growth"].replace([np.inf, -np.inf], np.nan).clip(lower=0, upper=8)
    usable["prev_discount"] = usable["prev_discount"].replace([np.inf, -np.inf], np.nan).clip(lower=0, upper=3)
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
    frame = _add_paper_predictive_feature_columns(frame)
    source_rows = frame[frame["month"] == source_month].copy()
    if source_rows.empty:
        raise ValueError(f"No paper-model forecast rows found for source month {source_month}.")
    source_rows["source_month"] = source_rows["month"]
    source_rows["source_month_index"] = source_rows["month_index"]
    source_rows = _add_paper_feature_columns(source_rows)
    source_rows["aligned_price_gap"] = source_rows["current_price_gap"]
    source_rows["current_discount"] = source_rows["current_discount"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    source_rows["current_price_gap"] = source_rows["current_price_gap"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    source_rows["aligned_price_gap"] = source_rows["aligned_price_gap"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    source_rows["prev_sales_growth"] = source_rows["prev_sales_growth"].replace([np.inf, -np.inf], np.nan).clip(
        lower=0,
        upper=8,
    )
    source_rows["prev_discount"] = source_rows["prev_discount"].replace([np.inf, -np.inf], np.nan).clip(
        lower=0,
        upper=3,
    )
    source_rows["current_discount_x_mesh"] = source_rows["current_discount_x_mesh"].replace([np.inf, -np.inf], np.nan)
    source_rows["current_discount_x_wifi"] = source_rows["current_discount_x_wifi"].replace([np.inf, -np.inf], np.nan)
    return source_rows.reset_index(drop=True)
