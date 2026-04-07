"""Ad hoc reporting visuals built from pipeline outputs."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "tv_ife_report_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import default_paths

BASE_MODEL_ORDER = [
    "naive_last_month",
    "direct_ridge",
    "paper_inspired_factor",
    "tv_ife_predictive_augmented",
]
MODEL_LABELS = {
    "naive_last_month": "Naive",
    "direct_ridge": "Direct Ridge",
    "paper_inspired_factor": "Paper TV-IFE",
    "tv_ife_predictive_augmented": "Launch-Aware TV-IFE",
}
MODEL_COLORS = {
    "naive_last_month": "#3b3b3b",
    "direct_ridge": "#197278",
    "paper_inspired_factor": "#c44536",
    "tv_ife_predictive_augmented": "#4b68b8",
}
METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "smape": "sMAPE",
}
FALLBACK_MODEL_COLORS = ["#6f7d8c", "#2a6f97", "#ff7f51", "#6a994e", "#a44a3f", "#7b6d8d"]


@dataclass
class ReportingFrames:
    summary: pd.DataFrame
    evaluation: pd.DataFrame
    paper_selection: pd.DataFrame
    paper_fit_summary: pd.DataFrame
    paper_fitted: pd.DataFrame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate report-only visuals from outputs/*.csv.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing the outputs/ directory.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Optional plot output directory. Defaults to outputs/report_plots under project root.",
    )
    return parser


def _load_reporting_frames(output_dir: Path) -> ReportingFrames:
    required = {
        "summary": output_dir / "model_summary.csv",
        "evaluation": output_dir / "forecast_eval.csv",
        "paper_selection": output_dir / "paper_factor_selection.csv",
        "paper_fit_summary": output_dir / "paper_track_fit_summary.csv",
        "paper_fitted": output_dir / "paper_track_fitted.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing pipeline outputs. Run `python -m tv_ife_estimator` first.\n"
            + "\n".join(missing)
        )

    return ReportingFrames(
        summary=pd.read_csv(required["summary"]),
        evaluation=pd.read_csv(required["evaluation"]),
        paper_selection=pd.read_csv(required["paper_selection"]),
        paper_fit_summary=pd.read_csv(required["paper_fit_summary"]),
        paper_fitted=pd.read_csv(required["paper_fitted"]),
    )


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f3eb",
            "axes.facecolor": "#fffdf8",
            "axes.edgecolor": "#c9c0b5",
            "axes.labelcolor": "#2f2a24",
            "axes.titleweight": "bold",
            "xtick.color": "#2f2a24",
            "ytick.color": "#2f2a24",
            "text.color": "#2f2a24",
            "font.size": 10,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _resolve_model_order(model_names: pd.Series | list[str] | tuple[str, ...]) -> list[str]:
    unique_names = [str(name) for name in dict.fromkeys(model_names) if pd.notna(name)]
    preferred = [name for name in BASE_MODEL_ORDER if name in unique_names]
    remaining = sorted(name for name in unique_names if name not in preferred)
    return preferred + remaining


def _resolve_model_colors(model_names: pd.Series | list[str] | tuple[str, ...]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    fallback_index = 0
    for model_name in _resolve_model_order(model_names):
        if model_name in MODEL_COLORS:
            resolved[model_name] = MODEL_COLORS[model_name]
            continue
        resolved[model_name] = FALLBACK_MODEL_COLORS[fallback_index % len(FALLBACK_MODEL_COLORS)]
        fallback_index += 1
    return resolved


def _label_for_model(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


def _format_month_label(target_month: str) -> str:
    month_text = str(target_month)
    if len(month_text) == 6 and month_text.isdigit():
        return f"{month_text[:4]}-{month_text[4:6]}"
    return month_text


def _monthly_metrics(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (model_name, target_month), frame in evaluation.groupby(["model_name", "target_month"], sort=True):
        y_true = frame["y_true"].astype(float).to_numpy()
        y_pred = frame["y_pred"].astype(float).to_numpy()
        denominator = np.abs(y_true) + np.abs(y_pred)
        valid = denominator > 0
        actual_total = float(np.sum(y_true))
        pred_total = float(np.sum(y_pred))
        signed_total_error = pred_total - actual_total
        rows.append(
            {
                "model_name": model_name,
                "target_month": str(target_month),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "smape": float(
                    np.mean(2.0 * np.abs(y_true[valid] - y_pred[valid]) / denominator[valid]) if valid.any() else 0.0
                ),
                "actual_total": actual_total,
                "pred_total": pred_total,
                "signed_total_error": signed_total_error,
                "abs_total_error": abs(signed_total_error),
                "total_pct_error": float(signed_total_error / actual_total) if actual_total > 0 else np.nan,
                "total_abs_pct_error": float(abs(signed_total_error) / actual_total) if actual_total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["target_month", "model_name"]).reset_index(drop=True)


def plot_model_summary(summary: pd.DataFrame, plots_dir: Path) -> None:
    ordered = summary.copy()
    model_order = _resolve_model_order(ordered["model_name"].astype(str).tolist())
    model_colors = _resolve_model_colors(model_order)
    ordered["model_name"] = pd.Categorical(ordered["model_name"], categories=model_order, ordered=True)
    ordered = ordered.sort_values("model_name")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, metric in zip(axes, ["mae", "rmse", "smape"], strict=True):
        values = ordered[metric].to_numpy(dtype=float)
        bars = axis.bar(
            [_label_for_model(str(name)) for name in ordered["model_name"]],
            values,
            color=[model_colors[str(name)] for name in ordered["model_name"]],
            width=0.62,
        )
        axis.set_title(f"Backtest {METRIC_LABELS[metric]}")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}" if metric == "smape" else f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Model Comparison on Rolling Backtest", fontsize=15, y=1.02)
    _save_figure(fig, plots_dir / "01_model_summary.png")


def plot_monthly_metrics(monthly: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    months = sorted(monthly["target_month"].astype(str).unique())
    month_labels = [_format_month_label(month) for month in months]
    model_order = _resolve_model_order(monthly["model_name"].astype(str).tolist())
    model_colors = _resolve_model_colors(model_order)

    for axis, metric in zip(axes, ["mae", "rmse", "smape"], strict=True):
        for model_name in model_order:
            subset = monthly[monthly["model_name"] == model_name].sort_values("target_month")
            axis.plot(
                [_format_month_label(month) for month in subset["target_month"]],
                subset[metric],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=model_colors[model_name],
                label=_label_for_model(model_name),
            )
        axis.set_title(f"Monthly {METRIC_LABELS[metric]}")
        axis.set_xticks(month_labels)
        axis.tick_params(axis="x", rotation=25)

    axes[0].legend(frameon=True)
    fig.suptitle("Backtest Metrics by Target Month", fontsize=15, y=1.02)
    _save_figure(fig, plots_dir / "02_monthly_metrics.png")


def plot_total_sales(monthly: pd.DataFrame, plots_dir: Path) -> None:
    actual = (
        monthly[["target_month", "actual_total"]]
        .drop_duplicates(subset=["target_month"])
        .sort_values("target_month")
        .reset_index(drop=True)
    )
    model_order = _resolve_model_order(monthly["model_name"].astype(str).tolist())
    model_colors = _resolve_model_colors(model_order)

    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    axis.plot(
        [_format_month_label(month) for month in actual["target_month"]],
        actual["actual_total"],
        marker="o",
        linewidth=2.6,
        markersize=7,
        color="#111111",
        label="Actual Total Sales",
    )
    for model_name in model_order:
        subset = monthly[monthly["model_name"] == model_name].sort_values("target_month")
        axis.plot(
            [_format_month_label(month) for month in subset["target_month"]],
            subset["pred_total"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=model_colors[model_name],
            label=f"{_label_for_model(model_name)} Predicted",
        )

    axis.set_title("Total Sales by Target Month")
    axis.set_ylabel("Sales")
    axis.legend(frameon=True, ncol=2)
    _save_figure(fig, plots_dir / "03_total_sales.png")


def plot_monthly_error_trend(monthly: pd.DataFrame, plots_dir: Path, metric: str = "smape") -> None:
    model_order = _resolve_model_order(monthly["model_name"].astype(str).tolist())
    model_colors = _resolve_model_colors(model_order)
    month_order = sorted(monthly["target_month"].astype(str).unique())
    month_labels = [_format_month_label(month) for month in month_order]

    fig, axis = plt.subplots(figsize=(10.8, 5.2))
    for model_name in model_order:
        subset = monthly[monthly["model_name"] == model_name].sort_values("target_month")
        values = subset[metric].astype(float).to_numpy() * 100.0 if metric == "smape" else subset[metric].astype(float)
        axis.plot(
            [_format_month_label(month) for month in subset["target_month"]],
            values,
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=model_colors[model_name],
            label=_label_for_model(model_name),
        )

    axis.set_title("Monthly Forecast Error Trend by Model")
    axis.set_xlabel("Target Month")
    axis.set_ylabel("sMAPE (%)" if metric == "smape" else METRIC_LABELS.get(metric, metric.upper()))
    axis.set_xticks(month_labels)
    axis.tick_params(axis="x", rotation=20)
    if metric == "smape":
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
    axis.legend(frameon=True, ncol=2)
    _save_figure(fig, plots_dir / f"07_monthly_{metric}_trend.png")


def export_monthly_error_tables(monthly: pd.DataFrame, output_dir: Path) -> None:
    export = monthly.copy()
    export["month_label"] = export["target_month"].astype(str).map(_format_month_label)
    export["model_label"] = export["model_name"].astype(str).map(_label_for_model)
    export = export[
        [
            "target_month",
            "month_label",
            "model_name",
            "model_label",
            "actual_total",
            "pred_total",
            "signed_total_error",
            "abs_total_error",
            "total_pct_error",
            "total_abs_pct_error",
            "mae",
            "rmse",
            "smape",
        ]
    ].sort_values(["target_month", "smape", "mae", "model_name"])
    export.to_csv(output_dir / "monthly_error_summary.csv", index=False, encoding="utf-8-sig")

    best_by_month = (
        export.sort_values(["target_month", "smape", "mae", "abs_total_error", "model_name"])
        .groupby("target_month", as_index=False)
        .first()
    )
    best_by_month.to_csv(output_dir / "monthly_best_model.csv", index=False, encoding="utf-8-sig")


def plot_monthly_actual_vs_pred(monthly: pd.DataFrame, plots_dir: Path) -> None:
    model_order = _resolve_model_order(monthly["model_name"].astype(str).tolist())
    model_colors = _resolve_model_colors(model_order)

    for target_month in sorted(monthly["target_month"].astype(str).unique()):
        subset = monthly[monthly["target_month"].astype(str) == target_month].copy()
        subset["model_name"] = pd.Categorical(subset["model_name"], categories=model_order, ordered=True)
        subset = subset.sort_values("model_name")

        actual_total = float(subset["actual_total"].iloc[0])
        labels = ["Actual"] + [_label_for_model(str(name)) for name in subset["model_name"]]
        values = [actual_total] + subset["pred_total"].astype(float).tolist()
        colors = ["#111111"] + [model_colors[str(name)] for name in subset["model_name"]]

        fig, axis = plt.subplots(figsize=(10.5, 5.4))
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.axhline(actual_total, color="#111111", linestyle="--", linewidth=1.2, alpha=0.6)
        axis.set_title(f"Actual vs Predicted Total Sales ({_format_month_label(target_month)})")
        axis.set_ylabel("Sales")
        axis.tick_params(axis="x", rotation=15)

        max_value = max(values) if values else 0.0
        axis.set_ylim(0, max_value * 1.18 if max_value > 0 else 1.0)

        for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
            label = f"{value:,.0f}"
            if index > 0 and actual_total > 0:
                label = f"{label}\n{((value - actual_total) / actual_total):+.1%}"
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

        _save_figure(fig, plots_dir / f"06_monthly_actual_vs_pred_{target_month}.png")


def plot_paper_ic(paper_selection: pd.DataFrame, plots_dir: Path) -> None:
    selection = paper_selection.copy().sort_values(["forecast_target_month", "factor_count"])
    criterion_column = "bic_rho2" if "bic_rho2" in selection.columns else "ic_rho2"
    months = selection["forecast_target_month"].astype(str).unique().tolist()
    if not months:
        return

    ncols = 2
    nrows = int(np.ceil(len(months) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.4 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for axis, month in zip(flat_axes, months, strict=False):
        subset = selection[selection["forecast_target_month"].astype(str) == month]
        axis.plot(
            subset["factor_count"],
            subset[criterion_column],
            color=MODEL_COLORS["paper_inspired_factor"],
            linewidth=2.2,
            marker="o",
        )
        selected = subset[subset["selected"] == 1]
        if not selected.empty:
            axis.scatter(
                selected["factor_count"],
                selected[criterion_column],
                s=90,
                color="#111111",
                zorder=3,
                label=f"Selected: {int(selected.iloc[0]['selected_factor_count'])}",
            )
            axis.legend(frameon=True, loc="best")
        axis.set_title(f"Paper BIC by Factor Count ({month})")
        axis.set_xlabel("Factor Count")
        axis.set_ylabel("BIC-type criterion (rho2)")
        axis.set_xticks(subset["factor_count"].tolist())

    for axis in flat_axes[len(months) :]:
        axis.axis("off")

    fig.suptitle("Paper Track BIC-Type Factor Selection in Backtest", fontsize=15, y=1.01)
    _save_figure(fig, plots_dir / "04_paper_ic.png")


def plot_paper_fit(paper_fitted: pd.DataFrame, paper_fit_summary: pd.DataFrame, plots_dir: Path) -> None:
    monthly_fit = (
        paper_fitted.groupby("source_month", as_index=False)[["observed_sales", "fitted_sales"]]
        .sum()
        .sort_values("source_month")
    )
    sample = paper_fitted.copy()
    sample["observed_log1p"] = np.log1p(sample["observed_sales"].clip(lower=0))
    sample["fitted_log1p"] = np.log1p(sample["fitted_sales"].clip(lower=0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    axes[0].plot(
        monthly_fit["source_month"],
        monthly_fit["observed_sales"],
        marker="o",
        linewidth=2.5,
        color="#111111",
        label="Observed",
    )
    axes[0].plot(
        monthly_fit["source_month"],
        monthly_fit["fitted_sales"],
        marker="o",
        linewidth=2.2,
        color=MODEL_COLORS["paper_inspired_factor"],
        label="Fitted",
    )
    axes[0].set_title("Paper Track In-Sample Totals")
    axes[0].set_ylabel("Sales")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=True)

    axes[1].scatter(
        sample["observed_log1p"],
        sample["fitted_log1p"],
        s=22,
        alpha=0.55,
        color=MODEL_COLORS["paper_inspired_factor"],
        edgecolors="none",
    )
    lower = float(min(sample["observed_log1p"].min(), sample["fitted_log1p"].min()))
    upper = float(max(sample["observed_log1p"].max(), sample["fitted_log1p"].max()))
    axes[1].plot([lower, upper], [lower, upper], color="#111111", linestyle="--", linewidth=1.4)
    axes[1].set_title("Observed vs Fitted (log1p sales)")
    axes[1].set_xlabel("Observed")
    axes[1].set_ylabel("Fitted")

    summary = paper_fit_summary.iloc[0]
    axes[1].text(
        0.04,
        0.96,
        "\n".join(
            [
                f"Selected factors: {int(summary['selected_factor_count'])}",
                f"Residual variance: {summary['residual_variance']:.3f}",
                f"In-sample sMAPE: {summary['in_sample_smape']:.3f}",
            ]
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fff7e8", "edgecolor": "#d4c3a3"},
    )

    fig.suptitle("Paper Track Fit Diagnostics", fontsize=15, y=1.02)
    _save_figure(fig, plots_dir / "05_paper_fit.png")


def generate_report_visuals(project_root: str | Path | None = None, plots_dir: str | Path | None = None) -> list[Path]:
    paths = default_paths(project_root)
    output_dir = paths.output_dir
    target_plots_dir = Path(plots_dir).resolve() if plots_dir is not None else output_dir / "report_plots"

    _apply_style()
    frames = _load_reporting_frames(output_dir)
    monthly = _monthly_metrics(frames.evaluation)
    export_monthly_error_tables(monthly, output_dir)

    plot_model_summary(frames.summary, target_plots_dir)
    plot_monthly_metrics(monthly, target_plots_dir)
    plot_total_sales(monthly, target_plots_dir)
    plot_monthly_error_trend(monthly, target_plots_dir, metric="smape")
    plot_monthly_actual_vs_pred(monthly, target_plots_dir)
    plot_paper_ic(frames.paper_selection, target_plots_dir)
    plot_paper_fit(frames.paper_fitted, frames.paper_fit_summary, target_plots_dir)

    return sorted(target_plots_dir.glob("*.png"))


def main() -> None:
    args = build_parser().parse_args()
    generated = generate_report_visuals(project_root=args.project_root, plots_dir=args.plots_dir)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
