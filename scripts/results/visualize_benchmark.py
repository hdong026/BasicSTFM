#!/usr/bin/env python3
"""Create paper-style tables and figures from BasicSTFM benchmark runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

from basicstfm.utils.results import (
    build_markdown_table,
    build_paper_summary,
    discover_stage_result_files,
    flatten_stage_results,
    load_stage_result_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize BasicSTFM benchmark results")
    parser.add_argument(
        "--input-roots",
        nargs="+",
        default=["runs"],
        help="Directories or stage_results.json files to search.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional ordered dataset list for the transfer table/figure.",
    )
    parser.add_argument(
        "--metric",
        default="metric/mae",
        help="Metric to visualize. Default: metric/mae",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Split used for reporting. Default: test",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark/paper",
        help="Directory for the generated paper artifacts.",
    )
    parser.add_argument(
        "--prefix",
        default="traffic_12x12",
        help="Prefix for generated files.",
    )
    parser.add_argument(
        "--title",
        default="Traffic Transfer Benchmark (12->12)",
        help="Figure title.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = _load_rows(args.input_roots)
    dataset_order = args.datasets or None
    datasets, summary = build_paper_summary(
        rows,
        split=args.split,
        metric=args.metric,
        datasets=dataset_order,
        model_order=("OpenCity", "FactoST", "UniST"),
    )
    if not summary:
        raise SystemExit("No transfer stages found for visualization.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / f"{args.prefix}_table.md"
    latex_path = output_dir / f"{args.prefix}_table.tex"
    csv_path = output_dir / f"{args.prefix}_table.csv"
    png_path = output_dir / f"{args.prefix}_main.png"
    pdf_path = output_dir / f"{args.prefix}_main.pdf"

    markdown_path.write_text(build_markdown_table(summary), encoding="utf-8")
    latex_path.write_text(build_latex_table(summary), encoding="utf-8")
    write_csv(csv_path, summary)

    make_figure(
        summary=summary,
        datasets=datasets,
        title=args.title,
        metric=args.metric,
        split=args.split,
        png_path=png_path,
        pdf_path=pdf_path,
    )

    print(f"Markdown table written to {markdown_path}")
    print(f"LaTeX table written to {latex_path}")
    print(f"CSV summary written to {csv_path}")
    print(f"Figure written to {png_path}")
    print(f"Figure written to {pdf_path}")


def _load_rows(input_roots: Sequence[str]) -> List[Dict[str, object]]:
    files = discover_stage_result_files(input_roots)
    if not files:
        raise SystemExit("No stage_results.json files found.")
    rows: List[Dict[str, object]] = []
    for path in files:
        payload = load_stage_result_payload(path)
        rows.extend(flatten_stage_results(payload, path))
    return rows


def build_latex_table(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return ""

    columns = list(rows[0].keys())
    numeric_best_low = {
        column: _best_numeric(rows, column, maximize=False)
        for column in columns
        if column.endswith(" ZS") or column.endswith(" FS") or column.startswith("Avg ")
    }
    numeric_best_high = {
        column: _best_numeric(rows, column, maximize=True)
        for column in columns
        if column.endswith(" Gain")
    }

    align = "l" + "r" * (len(columns) - 1)
    lines = [
        "\\begin{tabular}{" + align + "}",
        "\\toprule",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                cell = f"{value:.3f}"
                if column in numeric_best_low and numeric_best_low[column] == value:
                    cell = f"\\textbf{{{cell}}}"
                if column in numeric_best_high and numeric_best_high[column] == value:
                    cell = f"\\textbf{{{cell}}}"
            elif value is None:
                cell = "--"
            else:
                cell = _latex_escape(str(value))
            values.append(cell)
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_figure(
    *,
    summary: Sequence[Dict[str, object]],
    datasets: Sequence[str],
    title: str,
    metric: str,
    split: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for benchmark visualization. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    model_names = [str(row["Model"]) for row in summary]
    palette = {
        "OpenCity": "#4C78A8",
        "FactoST": "#E45756",
        "UniST": "#54A24B",
    }
    colors = [palette.get(name, "#4C78A8") for name in model_names]

    zero = {name: [row.get(f"{dataset} ZS") for dataset in datasets] for name, row in zip(model_names, summary)}
    few = {name: [row.get(f"{dataset} FS") for dataset in datasets] for name, row in zip(model_names, summary)}
    gain = {
        name: [row.get(f"{dataset} Gain") for dataset in datasets]
        for name, row in zip(model_names, summary)
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), constrained_layout=True)
    fig.suptitle(title, fontsize=13, y=1.02)

    metric_label = f"{split.upper()} {metric.split('/')[-1].upper()}"
    _plot_grouped_bars(
        axes[0],
        datasets,
        model_names,
        zero,
        colors,
        title="Zero-shot Transfer",
        ylabel=metric_label,
        higher_is_better=False,
    )
    _plot_grouped_bars(
        axes[1],
        datasets,
        model_names,
        few,
        colors,
        title="5% Few-shot Transfer",
        ylabel=metric_label,
        higher_is_better=False,
    )
    _plot_grouped_bars(
        axes[2],
        datasets,
        model_names,
        gain,
        colors,
        title="Few-shot Gain",
        ylabel=f"{split.upper()} {metric.split('/')[-1].upper()} Reduction",
        higher_is_better=True,
    )

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    axes[1].legend(
        handles,
        model_names,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=len(model_names),
        frameon=False,
    )

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bars(
    ax,
    datasets: Sequence[str],
    model_names: Sequence[str],
    values_by_model: Dict[str, Sequence[object]],
    colors: Sequence[str],
    *,
    title: str,
    ylabel: str,
    higher_is_better: bool,
) -> None:
    import numpy as np

    x = np.arange(len(datasets), dtype=float)
    width = 0.22
    offsets = np.linspace(-(len(model_names) - 1) / 2, (len(model_names) - 1) / 2, len(model_names)) * width

    for index, name in enumerate(model_names):
        raw_values = values_by_model[name]
        values = [float(item) if item is not None else np.nan for item in raw_values]
        bars = ax.bar(x + offsets[index], values, width=width, color=colors[index], label=name)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if higher_is_better:
        ax.axhline(0.0, color="#666666", linewidth=0.8)


def _best_numeric(rows: Sequence[Dict[str, object]], column: str, maximize: bool) -> object:
    values = [row.get(column) for row in rows]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return max(numeric) if maximize else min(numeric)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "_": "\\_",
        "#": "\\#",
        "{": "\\{",
        "}": "\\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    main()
