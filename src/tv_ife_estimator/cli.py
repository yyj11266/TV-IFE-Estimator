"""Command-line entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Run the TV-IFE router forecasting pipeline.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing data/ and outputs/ directories.",
    )
    parser.add_argument(
        "--family-filter-mode",
        choices=("all", "start_nonzero_only"),
        default="all",
        help="Restrict the pipeline to families matching the selected scope.",
    )
    parser.add_argument(
        "--family-filter-reference-month",
        default=None,
        help="Reference month (YYYYMM) used by start_nonzero_only. Defaults to the first full month in config.",
    )
    return parser


def main() -> None:
    """CLI main function."""

    args = build_parser().parse_args()
    config = replace(
        PipelineConfig(),
        family_filter_mode=args.family_filter_mode,
        family_filter_reference_month=args.family_filter_reference_month,
    )
    run_pipeline(project_root=args.project_root, config=config)


if __name__ == "__main__":
    main()
