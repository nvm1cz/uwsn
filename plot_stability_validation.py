from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict
import math
from statistics import mean, median, pstdev
from typing import Dict, List, Tuple

import numpy as np

from uwsn.cli import _parse_fnd_log_files


SUMMARY_LOG = Path("outputs/data/csv/summary/stability_topk_v2_summary.csv")
RAW_LOG = Path("outputs/data/csv/raw/stability_topk_v2_raw.csv")
SUMMARY_30_LOG = Path("outputs/data/csv/summary/stability_30seed_v2_summary.csv")
OLD_LOG = Path("outputs/data/logs/space100_to_fnd_runs10_v2.log")
OUTPUT_DIR = Path("outputs/figures/stability")


C_PAIRS = [(1.5, 1.5), (2.5, 0.5), (0.5, 2.5)]
NODE_COUNTS = (100, 200, 300, 500)
PACKET_SIZE = 6400
INITIAL_ENERGY = 0.5
SPACE_M = 100
TRANSMISSION_RANGE_M = 150.0

SUMMARY_30_FIELDS = [
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "pso_c1",
    "pso_c2",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
    "transmission_range_m",
    "pso_particles",
    "pso_iterations",
    "unique_seeds",
    "seed_min",
    "seed_max",
    "max_rounds",
    "fnd_mean",
    "fnd_std",
    "fnd_cv_pct",
    "fnd_min",
    "fnd_q1",
    "fnd_median",
    "fnd_q3",
    "fnd_max",
    "fnd_iqr",
    "residual_energy_pct_at_fnd_mean",
    "dead_nodes_pct_at_fnd_mean",
    "packets_received_mean",
]


def _load_stable_rows() -> List[Dict[str, str]]:
    with SUMMARY_LOG.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_30seed_rows() -> List[Dict[str, str]]:
    with SUMMARY_30_LOG.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_30seed_summary() -> List[Dict[str, object]]:
    with RAW_LOG.open("r", encoding="utf-8", newline="") as f:
        raw_rows = list(csv.DictReader(f))

    deduped: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in raw_rows:
        key = (
            row["space_m"],
            row["node_count"],
            row["packet_size_bits"],
            row["initial_energy_j"],
            row["pso_c1"],
            row["pso_c2"],
            row["cluster_head_ratio"],
            row["eulc_candidate_ratio"],
            row["pso_particles"],
            row["pso_iterations"],
            row["seed"],
        )
        deduped[key] = row

    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in deduped.values():
        key = (
            row["space_m"],
            row["node_count"],
            row["packet_size_bits"],
            row["initial_energy_j"],
            row["pso_c1"],
            row["pso_c2"],
            row["cluster_head_ratio"],
            row["eulc_candidate_ratio"],
            row["pso_particles"],
            row["pso_iterations"],
        )
        grouped[key].append(row)

    summary_rows: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(float(x) for x in item[0])):
        fnd_values = sorted(float(row["fnd_round"]) for row in rows if row["fnd_round"])
        if not fnd_values:
            continue

        (
            space_m,
            node_count,
            packet_size_bits,
            initial_energy_j,
            pso_c1,
            pso_c2,
            cluster_head_ratio,
            eulc_candidate_ratio,
            pso_particles,
            pso_iterations,
        ) = key
        seeds = sorted(int(row["seed"]) for row in rows)
        fnd_mean = mean(fnd_values)
        fnd_std = pstdev(fnd_values) if len(fnd_values) > 1 else 0.0
        fnd_q1, fnd_q3 = np.percentile(fnd_values, [25, 75])
        node_total_energy = float(node_count) * float(initial_energy_j)
        residual_pct = [
            float(row["residual_energy"]) / node_total_energy * 100.0
            for row in rows
            if row["residual_energy"]
        ]
        dead_pct = [
            (int(node_count) - int(row["alive_nodes"])) / int(node_count) * 100.0
            for row in rows
            if row["alive_nodes"]
        ]
        packets_received = [float(row["packets_received"]) for row in rows if row["packets_received"]]

        summary_rows.append(
            {
                "space_m": int(float(space_m)),
                "node_count": int(float(node_count)),
                "packet_size_bits": int(float(packet_size_bits)),
                "initial_energy_j": float(initial_energy_j),
                "pso_c1": float(pso_c1),
                "pso_c2": float(pso_c2),
                "cluster_head_ratio": float(cluster_head_ratio),
                "eulc_candidate_ratio": float(eulc_candidate_ratio),
                "transmission_range_m": TRANSMISSION_RANGE_M,
                "pso_particles": int(float(pso_particles)),
                "pso_iterations": int(float(pso_iterations)),
                "unique_seeds": len(seeds),
                "seed_min": min(seeds),
                "seed_max": max(seeds),
                "max_rounds": int(float(rows[0]["max_rounds"])),
                "fnd_mean": round(fnd_mean, 4),
                "fnd_std": round(fnd_std, 4),
                "fnd_cv_pct": round(fnd_std / fnd_mean * 100.0, 4),
                "fnd_min": round(min(fnd_values), 4),
                "fnd_q1": round(float(fnd_q1), 4),
                "fnd_median": round(median(fnd_values), 4),
                "fnd_q3": round(float(fnd_q3), 4),
                "fnd_max": round(max(fnd_values), 4),
                "fnd_iqr": round(float(fnd_q3 - fnd_q1), 4),
                "residual_energy_pct_at_fnd_mean": round(mean(residual_pct), 4),
                "dead_nodes_pct_at_fnd_mean": round(mean(dead_pct), 4),
                "packets_received_mean": round(mean(packets_received), 4),
            }
        )

    with SUMMARY_30_LOG.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_30_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {SUMMARY_30_LOG}")
    return summary_rows


def _stable_lookup(rows: List[Dict[str, str]]) -> Dict[Tuple[int, float, float], Tuple[float, float]]:
    lookup: Dict[Tuple[int, float, float], Tuple[float, float]] = {}
    for row in rows:
        if int(row["space_m"]) != SPACE_M:
            continue
        if int(row["packet_size_bits"]) != PACKET_SIZE:
            continue
        if float(row["initial_energy_j"]) != INITIAL_ENERGY:
            continue
        key = (int(row["node_count"]), float(row["pso_c1"]), float(row["pso_c2"]))
        lookup[key] = (float(row["fnd_mean"]), float(row["fnd_std"]))
    return lookup


def _format_c_pair(c1: float, c2: float) -> str:
    return f"c=({c1:g},{c2:g})"


def _nice_upper(max_value: float, tick_step: int = 500) -> int:
    if max_value <= 0:
        return tick_step
    return max(tick_step, int(math.ceil(max_value * 1.08 / tick_step) * tick_step))


def _row_lookup(rows: List[Dict[str, str]]) -> Dict[Tuple[int, int, float, float, float], Dict[str, str]]:
    lookup = {}
    for row in rows:
        key = (
            int(row["node_count"]),
            int(row["packet_size_bits"]),
            float(row["initial_energy_j"]),
            float(row["pso_c1"]),
            float(row["pso_c2"]),
        )
        lookup[key] = row
    return lookup


def _load_raw_fnd_groups() -> Dict[Tuple[int, int, float, float, float], List[float]]:
    with RAW_LOG.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    deduped: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        if int(row["space_m"]) != SPACE_M or not row["fnd_round"]:
            continue
        key = (
            row["node_count"],
            row["packet_size_bits"],
            row["initial_energy_j"],
            row["pso_c1"],
            row["pso_c2"],
            row["seed"],
        )
        deduped[key] = row

    groups: Dict[Tuple[int, int, float, float, float], List[float]] = defaultdict(list)
    for row in deduped.values():
        key = (
            int(row["node_count"]),
            int(row["packet_size_bits"]),
            float(row["initial_energy_j"]),
            float(row["pso_c1"]),
            float(row["pso_c2"]),
        )
        groups[key].append(float(row["fnd_round"]))
    return groups


def save_30seed_errorbar_grid(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    lookup = _row_lookup(rows)
    all_upper_values = [
        float(row["fnd_mean"]) + float(row["fnd_std"])
        for row in rows
        if row.get("fnd_mean") and row.get("fnd_std")
    ]
    y_upper = _nice_upper(max(all_upper_values), tick_step=500)
    y_ticks = list(range(0, y_upper + 1, 500))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), sharex=True, sharey=True, constrained_layout=True)
    colors = plt.get_cmap("tab10", len(C_PAIRS))
    x_positions = list(range(len(NODE_COUNTS)))
    bar_width = 0.22
    panel_specs = [
        (4000, 0.5),
        (6400, 0.5),
        (4000, 1.0),
        (6400, 1.0),
    ]

    for ax, (packet_size, initial_energy) in zip(axes.flat, panel_specs):
        for idx, (c1, c2) in enumerate(C_PAIRS):
            means = []
            stds = []
            for node_count in NODE_COUNTS:
                row = lookup[(node_count, packet_size, initial_energy, c1, c2)]
                means.append(float(row["fnd_mean"]))
                stds.append(float(row["fnd_std"]))
            offsets = [x + (idx - (len(C_PAIRS) - 1) / 2) * bar_width for x in x_positions]
            ax.bar(
                offsets,
                means,
                yerr=stds,
                capsize=4,
                color=colors(idx),
                alpha=0.84,
                width=bar_width,
                label=_format_c_pair(c1, c2),
            )
        ax.set_title(f"pkt={packet_size}, Eini={initial_energy:g}J")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("FND round")
        ax.set_xlim(-0.6, len(NODE_COUNTS) - 0.4)
        ax.set_xticks(x_positions, [str(value) for value in NODE_COUNTS])
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0][0].legend(loc="lower left", framealpha=0.92)
    fig.suptitle("FND mean +/- std across 30 seeds (space=100m, objective v2)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = OUTPUT_DIR / "stability_30seed_v2_fnd_errorbars.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_30seed_cv_heatmap(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    row_labels = []
    matrix = []
    for packet_size in (4000, 6400):
        for initial_energy in (0.5, 1.0):
            for c1, c2 in C_PAIRS:
                label = f"pkt={packet_size}, E={initial_energy:g}, {_format_c_pair(c1, c2)}"
                row_labels.append(label)
                values = []
                for node_count in NODE_COUNTS:
                    match = [
                        row
                        for row in rows
                        if int(row["node_count"]) == node_count
                        and int(row["packet_size_bits"]) == packet_size
                        and float(row["initial_energy_j"]) == initial_energy
                        and float(row["pso_c1"]) == c1
                        and float(row["pso_c2"]) == c2
                    ][0]
                    values.append(float(match["fnd_cv_pct"]))
                matrix.append(values)

    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10.8, 7.8))
    image = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=max(10.0, float(np.nanmax(data))))
    ax.set_title("FND coefficient of variation across 30 seeds (lower is more stable)")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Scenario")
    ax.set_xticks(range(len(NODE_COUNTS)), [str(value) for value in NODE_COUNTS])
    ax.set_yticks(range(len(row_labels)), row_labels)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value:.1f}%", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("CV (%)")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "stability_30seed_v2_cv_heatmap.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_30seed_boxplot(packet_size: int = 6400, initial_energy: float = 0.5) -> None:
    import matplotlib.pyplot as plt

    groups = _load_raw_fnd_groups()
    data = []
    labels = []
    colors = plt.get_cmap("tab10", len(C_PAIRS))
    box_colors = []

    for node_count in NODE_COUNTS:
        for idx, (c1, c2) in enumerate(C_PAIRS):
            key = (node_count, packet_size, initial_energy, c1, c2)
            data.append(sorted(groups[key]))
            labels.append(f"n={node_count}\n{_format_c_pair(c1, c2)}")
            box_colors.append(colors(idx))

    fig, ax = plt.subplots(figsize=(14.5, 5.8))
    box = ax.boxplot(data, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for median_line in box["medians"]:
        median_line.set_color("black")
        median_line.set_linewidth(1.3)

    ax.set_title(f"FND distribution across 30 seeds - pkt={packet_size}, Eini={initial_energy:g}J")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("FND round")
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"stability_30seed_v2_fnd_boxplot_pkt{packet_size}_e{str(initial_energy).replace('.', 'p')}.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_stable_errorbar_chart(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    stable = _stable_lookup(rows)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    colors = plt.get_cmap("tab10", len(C_PAIRS))

    for idx, (c1, c2) in enumerate(C_PAIRS):
        means = []
        stds = []
        for node_count in NODE_COUNTS:
            mean, std = stable[(node_count, c1, c2)]
            means.append(mean)
            stds.append(std)
        ax.errorbar(
            NODE_COUNTS,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linewidth=2.2,
            markersize=6,
            color=colors(idx),
            label=_format_c_pair(c1, c2),
        )
        for node_count, mean in zip(NODE_COUNTS, means):
            ax.annotate(f"{mean:.0f}", (node_count, mean), textcoords="offset points", xytext=(0, 9), ha="center")

    ax.set_title("Stabilized FND validation - pkt=6400, e=0.5J, space=100m")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("FND round")
    ax.set_xticks(NODE_COUNTS)
    ax.set_ylim(800, 1100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "stability_pkt6400_e05_topk_v2.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_before_after_chart(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    old_values = _parse_fnd_log_files([OLD_LOG])
    stable = _stable_lookup(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (c1, c2) in zip(axes, C_PAIRS):
        before = [
            old_values.get((SPACE_M, node_count, c1, c2, PACKET_SIZE, INITIAL_ENERGY), np.nan)
            for node_count in NODE_COUNTS
        ]
        after = [stable[(node_count, c1, c2)][0] for node_count in NODE_COUNTS]
        ax.plot(NODE_COUNTS, before, marker="s", linewidth=2.0, label="before")
        ax.plot(NODE_COUNTS, after, marker="o", linewidth=2.2, label="stabilized")
        for node_count, value in zip(NODE_COUNTS, after):
            ax.annotate(f"{value:.0f}", (node_count, value), textcoords="offset points", xytext=(0, 9), ha="center")
        ax.set_title(_format_c_pair(c1, c2))
        ax.set_xlabel("Number of nodes")
        ax.set_xticks(NODE_COUNTS)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("FND round")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle("Before vs stabilized FND - pkt=6400, e=0.5J, space=100m")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_path = OUTPUT_DIR / "stability_before_after_pkt6400_e05_100m.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_30seed_summary()

    rows = _load_stable_rows()
    save_stable_errorbar_chart(rows)
    save_before_after_chart(rows)

    rows_30 = _load_30seed_rows()
    save_30seed_errorbar_grid(rows_30)
    save_30seed_cv_heatmap(rows_30)
    save_30seed_boxplot(packet_size=6400, initial_energy=0.5)


if __name__ == "__main__":
    main()
