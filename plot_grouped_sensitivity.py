from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np


SUMMARY_LOG = Path("outputs/data/csv/summary/extended_sensitivity_v2_summary.csv")
OUTPUT_DIR = Path("outputs/figures/sensitivity")


Spec = Tuple[str, str, str, Callable[[Dict[str, str]], object]]


SPECS: List[Spec] = [
    ("energy_sweep_100m_v2", "Initial energy", "Eini (J)", lambda row: float(row["initial_energy_j"])),
    ("packet_sweep_100m_v2", "Packet size", "Packet (bits)", lambda row: int(float(row["packet_size_bits"]))),
    (
        "candidate_ratio_sweep_100m_v2",
        "Candidate ratio",
        "EULC candidate ratio",
        lambda row: float(row["eulc_candidate_ratio"]),
    ),
    (
        "pso_coeff_sweep_100m_v2",
        "PSO coefficients",
        "(c1,c2)",
        lambda row: f"({float(row['pso_c1']):g},{float(row['pso_c2']):g})",
    ),
    ("objective_weight_sweep_100m_v2", "Objective weight", "omega", lambda row: float(row["pso_omega"])),
    ("range_sweep_100m_v2", "Transmission range", "R (m)", lambda row: int(float(row["transmission_range_m"]))),
]


SINGLE_FIGURE_FILENAMES = {
    "energy_sweep_100m_v2": "sensitivity_initial_energy_100m_v2.png",
    "packet_sweep_100m_v2": "sensitivity_packet_size_100m_v2.png",
    "candidate_ratio_sweep_100m_v2": "sensitivity_candidate_ratio_100m_v2.png",
    "pso_coeff_sweep_100m_v2": "sensitivity_pso_coefficients_100m_v2.png",
    "objective_weight_sweep_100m_v2": "sensitivity_objective_weight_100m_v2.png",
    "range_sweep_100m_v2": "sensitivity_transmission_range_100m_v2.png",
}


def _load_rows() -> List[Dict[str, str]]:
    with SUMMARY_LOG.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sorted_values(values: set[object]) -> List[object]:
    try:
        return sorted(values, key=lambda value: float(value))
    except (TypeError, ValueError):
        return sorted(values, key=str)


def _nice_upper(max_value: float) -> float:
    if max_value <= 0:
        return 1.0
    tick_step = 500.0
    return max(tick_step, math.ceil(max_value * 1.08 / tick_step) * tick_step)


def _value_label(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _fnd_label(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _rows_for_suite(rows: List[Dict[str, str]], suite: str) -> List[Dict[str, str]]:
    return [row for row in rows if row["suite"] == suite]


def save_grouped_line_overview(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    selected_rows = [row for spec in SPECS for row in _rows_for_suite(rows, spec[0]) if row.get("fnd_mean")]
    if not selected_rows:
        return

    node_counts = sorted({int(row["node_count"]) for row in selected_rows})
    y_upper = _nice_upper(max(float(row["fnd_mean"]) for row in selected_rows))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
    color_map = plt.get_cmap("tab10", len(node_counts))

    for ax, (suite, title, x_label, x_getter) in zip(axes.flat, SPECS):
        suite_rows = _rows_for_suite(rows, suite)
        x_values = _sorted_values({x_getter(row) for row in suite_rows})
        x_positions = list(range(len(x_values)))

        for idx, node_count in enumerate(node_counts):
            y_values: List[float | None] = []
            for x_value in x_values:
                matches = [
                    row
                    for row in suite_rows
                    if int(row["node_count"]) == node_count and x_getter(row) == x_value
                ]
                y_values.append(float(matches[0]["fnd_mean"]) if matches and matches[0].get("fnd_mean") else None)
            ax.plot(
                x_positions,
                y_values,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=color_map(idx),
                label=f"n={node_count}",
            )

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_xticks(x_positions, [_value_label(value) for value in x_values])
        ax.tick_params(axis="x", rotation=25 if len(x_values) > 4 else 0)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_ylim(0, y_upper)
        ax.grid(True, linestyle="--", alpha=0.35)

    for ax in axes.flat:
        ax.set_ylabel("FND round")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(node_counts), framealpha=0.9)
    fig.suptitle("Grouped FND sensitivity comparison (space=100m, objective v2)")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))

    output_path = OUTPUT_DIR / "extended_grouped_sensitivity_lines_100m_v2.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_individual_line_figures(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    selected_rows = [row for spec in SPECS for row in _rows_for_suite(rows, spec[0]) if row.get("fnd_mean")]
    if not selected_rows:
        return

    node_counts = sorted({int(row["node_count"]) for row in selected_rows})
    color_map = plt.get_cmap("tab10", len(node_counts))

    for suite, title, x_label, x_getter in SPECS:
        suite_rows = [row for row in _rows_for_suite(rows, suite) if row.get("fnd_mean")]
        if not suite_rows:
            continue

        x_values = _sorted_values({x_getter(row) for row in suite_rows})
        x_positions = list(range(len(x_values)))
        y_upper = _nice_upper(max(float(row["fnd_mean"]) for row in suite_rows))

        fig, ax = plt.subplots(figsize=(9.2, 5.6))
        series_values: Dict[int, List[float | None]] = {}
        for idx, node_count in enumerate(node_counts):
            y_values: List[float | None] = []
            for x_value in x_values:
                matches = [
                    row
                    for row in suite_rows
                    if int(row["node_count"]) == node_count and x_getter(row) == x_value
                ]
                y_values.append(float(matches[0]["fnd_mean"]) if matches else None)

            ax.plot(
                x_positions,
                y_values,
                marker="o",
                linewidth=2.2,
                markersize=6,
                    color=color_map(idx),
                    label=f"n={node_count}",
            )
            series_values[node_count] = y_values

        min_label_gap = y_upper * 0.035
        top_label_limit = y_upper * 0.965
        bottom_label_limit = y_upper * 0.025
        for x_idx, x_pos in enumerate(x_positions):
            labels = []
            for idx, node_count in enumerate(node_counts):
                y_value = series_values[node_count][x_idx]
                if y_value is not None:
                    labels.append((idx, y_value))

            adjusted_labels = []
            previous_y = -float("inf")
            for idx, y_value in sorted(labels, key=lambda item: item[1]):
                label_y = max(y_value, previous_y + min_label_gap, bottom_label_limit)
                adjusted_labels.append([idx, y_value, label_y])
                previous_y = label_y

            if adjusted_labels:
                overflow = adjusted_labels[-1][2] - top_label_limit
                if overflow > 0:
                    for item in adjusted_labels:
                        item[2] -= overflow
                underflow = bottom_label_limit - adjusted_labels[0][2]
                if underflow > 0:
                    for item in adjusted_labels:
                        item[2] += underflow

            for idx, y_value, label_y in adjusted_labels:
                x_jitter = (idx - (len(node_counts) - 1) / 2) * 0.035
                ax.text(
                    x_pos + x_jitter,
                    label_y,
                    _fnd_label(y_value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color_map(idx),
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6},
                    clip_on=False,
                    zorder=5,
                )

        ax.set_title(f"{title} sensitivity (space=100m, objective v2)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("FND round")
        ax.set_xticks(x_positions, [_value_label(value) for value in x_values])
        ax.tick_params(axis="x", rotation=25 if len(x_values) > 4 else 0)
        ax.set_xlim(-0.25, len(x_values) - 0.75)
        ax.set_ylim(0, y_upper)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", framealpha=0.92)
        fig.tight_layout()

        output_path = OUTPUT_DIR / SINGLE_FIGURE_FILENAMES[suite]
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def save_grouped_heatmaps(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    selected_rows = [row for spec in SPECS for row in _rows_for_suite(rows, spec[0]) if row.get("fnd_mean")]
    if not selected_rows:
        return

    node_counts = sorted({int(row["node_count"]) for row in selected_rows})
    vmax = _nice_upper(max(float(row["fnd_mean"]) for row in selected_rows))

    fig, axes = plt.subplots(2, 3, figsize=(19, 9.8), constrained_layout=True)
    image = None

    for ax, (suite, title, x_label, x_getter) in zip(axes.flat, SPECS):
        suite_rows = _rows_for_suite(rows, suite)
        y_values = _sorted_values({x_getter(row) for row in suite_rows})
        matrix = np.full((len(y_values), len(node_counts)), np.nan)

        for row in suite_rows:
            y_idx = y_values.index(x_getter(row))
            x_idx = node_counts.index(int(row["node_count"]))
            matrix[y_idx, x_idx] = float(row["fnd_mean"]) if row.get("fnd_mean") else np.nan

        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel(x_label)
        ax.set_xticks(range(len(node_counts)), [str(value) for value in node_counts])
        ax.set_yticks(range(len(y_values)), [_value_label(value) for value in y_values])

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    continue
                text_color = "white" if value > vmax * 0.45 else "black"
                ax.text(col_idx, row_idx, _fnd_label(value), ha="center", va="center", color=text_color, fontsize=8)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), location="right", shrink=0.82, pad=0.012)
        cbar.set_label("FND round")

    fig.suptitle("Grouped FND sensitivity heatmaps (space=100m, objective v2)")

    output_path = OUTPUT_DIR / "extended_grouped_sensitivity_heatmaps_100m_v2.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    rows = _load_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_grouped_line_overview(rows)
    save_individual_line_figures(rows)
    save_grouped_heatmaps(rows)


if __name__ == "__main__":
    main()
