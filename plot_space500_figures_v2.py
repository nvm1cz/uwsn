from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from plot_stability_stress_v2 import build_combined_summary


OUTPUT_DIR = Path("outputs/figures/drafts/space500_figures")
NODE_COUNTS = (1000, 1500, 2000, 2500)
PANEL_SPECS = (
    (4000, 0.5),
    (4000, 1.0),
    (6400, 0.5),
    (6400, 1.0),
)


def _f(row: Dict[str, str], field: str) -> float:
    return float(row[field])


def _nice_upper(max_value: float, tick_step: int) -> int:
    if max_value <= 0:
        return tick_step
    return max(tick_step, int(math.ceil(max_value * 1.08 / tick_step) * tick_step))


def _space500_rows() -> List[Dict[str, str]]:
    return [
        row
        for row in build_combined_summary()
        if row["suite"] == "space500_scale_stress_v2"
    ]


def _common_axis(rows: List[Dict[str, str]]) -> Tuple[int, List[int]]:
    y_upper = _nice_upper(max(_f(row, "fnd_mean") + _f(row, "fnd_std") for row in rows), 500)
    return y_upper, list(range(0, y_upper + 1, 500))


def _save_single_case(
    rows: List[Dict[str, str]],
    *,
    packet_size: int,
    initial_energy: float,
    y_upper: int,
    y_ticks: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    series = sorted(
        [
            row
            for row in rows
            if int(row["packet_size_bits"]) == packet_size
            and float(row["initial_energy_j"]) == initial_energy
        ],
        key=lambda row: int(row["node_count"]),
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    node_counts = [int(row["node_count"]) for row in series]
    means = [_f(row, "fnd_mean") for row in series]
    stds = [_f(row, "fnd_std") for row in series]
    cvs = [_f(row, "fnd_cv_pct") for row in series]
    runs = [int(row["runs"]) for row in series]

    ax.errorbar(node_counts, means, yerr=stds, marker="o", capsize=4, linewidth=2.1)
    for node_count, mean_value, std_value, cv_value, run_count in zip(node_counts, means, stds, cvs, runs):
        ax.annotate(
            f"{mean_value:.0f}\nCV={cv_value:.1f}%\n{run_count} seeds",
            xy=(node_count, mean_value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6},
        )

    ax.set_title(f"space=500m, pkt={packet_size}, Eini={initial_energy:g}J")
    fig.suptitle("FND mean +/- std across seeds, objective v2", fontsize=10)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("FND round")
    ax.set_xlim(min(NODE_COUNTS) - 180, max(NODE_COUNTS) + 180)
    ax.set_xticks(NODE_COUNTS)
    ax.set_ylim(0, y_upper)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    energy_label = str(initial_energy).replace(".", "p")
    output_path = OUTPUT_DIR / f"space500_pkt{packet_size}_e{energy_label}_fnd_vs_nodes.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _save_faceted(rows: List[Dict[str, str]], y_upper: int, y_ticks: Sequence[int]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, (packet_size, initial_energy) in zip(axes.flat, PANEL_SPECS):
        series = sorted(
            [
                row
                for row in rows
                if int(row["packet_size_bits"]) == packet_size
                and float(row["initial_energy_j"]) == initial_energy
            ],
            key=lambda row: int(row["node_count"]),
        )
        node_counts = [int(row["node_count"]) for row in series]
        means = [_f(row, "fnd_mean") for row in series]
        stds = [_f(row, "fnd_std") for row in series]
        cvs = [_f(row, "fnd_cv_pct") for row in series]

        ax.errorbar(node_counts, means, yerr=stds, marker="o", linewidth=2.1, capsize=4)
        for node_count, mean_value, cv_value in zip(node_counts, means, cvs):
            ax.annotate(
                f"{mean_value:.0f}\nCV={cv_value:.1f}%",
                xy=(node_count, mean_value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6},
            )
        ax.set_title(f"pkt={packet_size}, Eini={initial_energy:g}J")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("FND round")
        ax.set_xlim(min(NODE_COUNTS) - 180, max(NODE_COUNTS) + 180)
        ax.set_xticks(NODE_COUNTS)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Space=500m FND stability by node count (common axes, objective v2)")
    output_path = OUTPUT_DIR / "space500_fnd_vs_nodes_faceted.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _space500_rows()
    y_upper, y_ticks = _common_axis(rows)
    _save_faceted(rows, y_upper, y_ticks)
    for packet_size, initial_energy in PANEL_SPECS:
        _save_single_case(
            rows,
            packet_size=packet_size,
            initial_energy=initial_energy,
            y_upper=y_upper,
            y_ticks=y_ticks,
        )


if __name__ == "__main__":
    main()
