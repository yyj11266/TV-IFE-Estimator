"""Command-line entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    return parser


def main() -> None:
    """CLI main function."""

    args = build_parser().parse_args()
    run_pipeline(project_root=args.project_root)


if __name__ == "__main__":
    main()
