#!/usr/bin/env python3
"""
Plot average decisions over problem size per model from an OOD results JSON file.

Usage:
    python plot_ood_results.py <ood_results_file.json> [--output <output_file>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

MARCH_UNMODIFIED_COLOR = "#e05c2e"  # fixed orange-red for the unmodified march baseline
MARCH_UNMODIFIED_KEY = "MarchUnmodified"


def parse_size(size_key: str) -> int:
    """Extract numeric size from a key like 'n350' or '350'."""
    match = re.search(r"\d+", size_key)
    if match is None:
        raise ValueError(f"Cannot parse numeric size from key: {size_key!r}")
    return int(match.group())


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_series(data: dict) -> dict[str, tuple[list[int], list[float]]]:
    """Return {model_name: (sorted sizes, avg_decisions)} for all models."""
    sizes_data = data.get("sizes", {})
    series: dict[str, dict[int, float]] = {}

    for size_key, model_results in sizes_data.items():
        size = parse_size(size_key)
        for model, stats in model_results.items():
            if model not in series:
                series[model] = {}
            series[model][size] = stats.get("avg_decisions", float("nan"))

    result = {}
    for model, size_map in series.items():
        xs = sorted(size_map.keys())
        ys = [size_map[x] for x in xs]
        result[model] = (xs, ys)
    return result


def pick_colors(models: list[str]) -> dict[str, str]:
    """Assign colors; MarchUnmodified always gets its fixed color."""
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors: dict[str, str] = {}
    palette_iter = iter(c for c in palette if c != MARCH_UNMODIFIED_COLOR)
    for model in models:
        if model == MARCH_UNMODIFIED_KEY:
            colors[model] = MARCH_UNMODIFIED_COLOR
        else:
            colors[model] = next(palette_iter)
    return colors


def plot(data: dict, output: str | None = None, y_max: float | None = None) -> None:
    series = build_series(data)
    if not series:
        print("No size data found in results file.", file=sys.stderr)
        sys.exit(1)

    # Put MarchUnmodified last so it renders on top as a clear baseline
    models = sorted(series.keys(), key=lambda m: (m == MARCH_UNMODIFIED_KEY, m))
    colors = pick_colors(models)

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in models:
        xs, ys = series[model]
        linestyle = "--" if model == MARCH_UNMODIFIED_KEY else "-"
        linewidth = 2.0 if model == MARCH_UNMODIFIED_KEY else 1.8
        ax.plot(
            xs,
            ys,
            label=model,
            color=colors[model],
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Problem size (number of variables)", fontsize=12)
    ax.set_ylabel("Average decisions", fontsize=12)
    ax.set_title("Average decisions over problem size per model", fontsize=14)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, linestyle=":", alpha=0.5)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot avg decisions vs problem size from an OOD results JSON file."
    )
    parser.add_argument("results_file", help="Path to the OOD results JSON file.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Save the plot to this file (e.g. plot.png) instead of showing it.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Maximum value for the y-axis.",
    )
    args = parser.parse_args()

    data = load_results(args.results_file)
    plot(data, output=args.output, y_max=args.y_max)


if __name__ == "__main__":
    main()
