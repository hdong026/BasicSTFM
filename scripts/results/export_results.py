#!/usr/bin/env python3
"""Collect stage result files and export benchmark tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from basicstfm.utils.results import (
    build_markdown_table,
    discover_stage_result_files,
    filter_stage_rows,
    flatten_stage_results,
    load_stage_result_payload,
    summarize_stage_rows,
    write_csv,
    write_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export BasicSTFM stage results")
    parser.add_argument(
        "--input-roots",
        nargs="+",
        default=["runs"],
        help="Directories (or stage_results.json files) to search. Default: runs",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        default=[],
        help="Optional stage-name filter, e.g. zero_shot_test five_percent_prompt_tuning",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset-name filter inferred from data_path parent directory",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Optional experiment_name filter",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Metric split used by the summary table. Default: test",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["metric/mae", "metric/rmse"],
        help="Metric keys to include in the summary table. Prefix with train/val/test if needed.",
    )
    parser.add_argument(
        "--flat-output",
        default=None,
        help="Optional CSV path for the flattened one-row-per-stage export",
    )
    parser.add_argument(
        "--table-output",
        default=None,
        help="Optional .md or .csv path for the benchmark summary table",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print the summary table to stdout in Markdown format",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    files = discover_stage_result_files(args.input_roots)
    if not files:
        raise SystemExit("No stage_results.json files found.")

    rows = []
    for path in files:
        payload = load_stage_result_payload(path)
        rows.extend(flatten_stage_results(payload, path))

    rows = filter_stage_rows(
        rows,
        stages=args.stages,
        dataset=args.dataset,
        experiment=args.experiment,
    )
    if not rows:
        raise SystemExit("No stage rows matched the requested filters.")

    summary = summarize_stage_rows(rows, split=args.split, metrics=args.metrics)

    if args.flat_output:
        write_csv(args.flat_output, rows)
        print(f"Flat stage export written to {Path(args.flat_output)}")

    if args.table_output:
        output_path = Path(args.table_output)
        if output_path.suffix.lower() == ".md":
            write_markdown(output_path, summary)
        else:
            write_csv(output_path, summary)
        print(f"Summary table written to {output_path}")

    if args.print_table or not args.table_output:
        print(build_markdown_table(summary))


if __name__ == "__main__":
    main()
