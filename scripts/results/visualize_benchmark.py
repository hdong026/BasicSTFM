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
    parser.add_argument(
        "--model-order",
        nargs="*",
        default=[],
        help="Optional ordered model names for the table/figure.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = _load_rows(args.input_roots)
    dataset_order = args.datasets or None
    # Order matters for both table rows and bar grouping. DPM (our method) is 4th among the
    # four main LargeST-1Ch baselines. Include OpenCity display variants so they sort first
    # instead of falling back to file iteration order (which also caused palette/fallback
    # collisions: OpenCity-LargeST-1Ch used fallback[3] == DPM's purple).
    default_model_order = (
        "OpenCity-LargeST-1Ch",
        "OpenCity",
        "FactoST",
        "UniST",
        "DPM-STFM",
        "DPM-Scratch",
        "DPM-StableOnly",
        "DPM-NoDiffusion",
        "DPM-NoDisentangle",
    )
    datasets, summary = build_paper_summary(
        rows,
        split=args.split,
        metric=args.metric,
        datasets=dataset_order,
        model_order=tuple(args.model_order) if args.model_order else default_model_order,
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


# Publication-style, colorblind-friendly palette (inspired by Okabe–Ito / Nature guidelines).
# Every known `pretty_model_name` gets a key so we never fall back to index 3 == DPM purple.
_MODEL_COLORS: Dict[str, str] = {
    "OpenCity": "#0072B2",
    "OpenCity-LargeST": "#0072B2",
    "OpenCity-LargeST-1Ch": "#0072B2",
    "FactoST": "#D55E00",
    "UniST": "#009E73",
    "DPM-STFM": "#CC79A7",
    "DPM-Scratch": "#F0E442",
    "DPM-StableOnly": "#56B4E9",
    "DPM-NoDiffusion": "#E69F00",
    "DPM-NoDisentangle": "#A6761D",
}
# Extras: distinct hues for any other model name (never overlap with values above)
_EXTRA_MODEL_COLORS = [
    "#332288",
    "#882255",
    "#44AA99",
    "#AA4499",
    "#999933",
    "#117733",
    "#661100",
    "#6699CC",
]


def _bar_color_for_model(name: str, index: int) -> str:
    if name in _MODEL_COLORS:
        return _MODEL_COLORS[name]
    h = abs(hash(name))
    return _EXTRA_MODEL_COLORS[(h + index) % len(_EXTRA_MODEL_COLORS)]


def _is_highlight_model(name: str) -> bool:
    return str(name) == "DPM-STFM"


# Short, familiar labels for x-axis when many traffic benchmarks sit side by side
_DATASET_XLABEL_ABBREV: Dict[str, str] = {
    "METR-LA": "M-LA",
    "PEMS-BAY": "P-Bay",
    "PEMS04": "PEMS-04",
    "PEMS07": "PEMS-07",
    "PEMS08": "PEMS-08",
}


def _x_labels_for_datasets(datasets: Sequence[str]) -> list[str]:
    return [_DATASET_XLABEL_ABBREV.get(str(d), str(d)) for d in datasets]


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
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for benchmark visualization. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    model_names = [str(row["Model"]) for row in summary]
    colors = [_bar_color_for_model(name, i) for i, name in enumerate(model_names)]

    zero = {name: [row.get(f"{dataset} ZS") for dataset in datasets] for name, row in zip(model_names, summary)}
    few = {name: [row.get(f"{dataset} FS") for dataset in datasets] for name, row in zip(model_names, summary)}

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans", "sans-serif"],
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#2F2F2F",
            "text.color": "#1A1A1A",
            "axes.labelcolor": "#1A1A1A",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFC",
            "grid.color": "#B0B0B0",
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Wider with more groups so x labels (rotated) do not crowd; also raise figure height
    # slightly for the bottom label band.
    _n = max(len(datasets), 1)
    # Two panels (zero-shot + few-shot); a bit wider per column than the old 3-panel layout.
    _fig_w = min(1.6 * _n + 4.5, 18.0)
    # Extra bottom space for slanted + abbreviated dataset names.
    _bottom = 0.26 + min(0.12, 0.02 * _n)
    fig, axes = plt.subplots(1, 2, figsize=(_fig_w, 4.1))
    fig.subplots_adjust(wspace=0.28, top=0.9, bottom=_bottom)
    fig.suptitle(title, fontsize=12.5, fontweight="600", y=0.99, color="#111111")

    metric_label = f"{split.upper()} {metric.split('/')[-1].replace('_', ' ').upper()}"
    _xlab = _x_labels_for_datasets(datasets)
    _plot_grouped_bars(
        axes[0],
        _xlab,
        model_names,
        zero,
        colors,
        title="Zero-shot",
        ylabel=metric_label,
        higher_is_better=False,
    )
    _plot_grouped_bars(
        axes[1],
        _xlab,
        model_names,
        few,
        colors,
        title="5% few-shot",
        ylabel=metric_label,
        higher_is_better=False,
    )

    n = len(model_names)
    ncol = min(n, 4)
    handles = []
    for name, color in zip(model_names, colors):
        edge = "#1A1A1A" if _is_highlight_model(name) else "#4D4D4D"
        lw = 1.25 if _is_highlight_model(name) else 0.6
        rect = mpl.patches.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
        )
        handles.append((name, rect))

    leg_handles = [h[1] for h in handles]
    leg_labels = [f"{h[0]} (ours)" if _is_highlight_model(h[0]) else h[0] for h in handles]
    axes[1].legend(
        leg_handles,
        leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        facecolor="#FAFAFA",
        borderpad=0.6,
        labelspacing=0.4,
        handlelength=1.2,
        handleheight=0.9,
    )

    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight", facecolor="white", edgecolor="none")
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
    width = min(0.8 / max(len(model_names), 1), 0.22)
    offsets = np.linspace(-(len(model_names) - 1) / 2, (len(model_names) - 1) / 2, len(model_names)) * width

    for index, name in enumerate(model_names):
        raw_values = values_by_model[name]
        values = [float(item) if item is not None else np.nan for item in raw_values]
        edge_w = 1.2 if _is_highlight_model(name) else 0.55
        edge_c = "#1A1A1A" if _is_highlight_model(name) else "#FAFAFA"
        z = 5 + index
        if _is_highlight_model(name):
            z = 25
        bars = ax.bar(
            x + offsets[index],
            values,
            width=width,
            color=colors[index],
            label=name,
            edgecolor=edge_c,
            linewidth=edge_w,
            zorder=z,
        )
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
                color="#2A2A2A",
                clip_on=False,
            )

    ax.set_title(title, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        list(datasets),
        rotation=32,
        ha="right",
        rotation_mode="anchor",
        fontsize=8.5,
    )
    ax.tick_params(axis="x", which="major", length=3, pad=3)
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
