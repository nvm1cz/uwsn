from __future__ import annotations

import csv
import math
import argparse
from pathlib import Path
from typing import Dict, List, Sequence


SUMMARY_LOG = Path("outputs/data/csv/summary/algorithm_node_sweep_v2_summary.csv")
OUTPUT_DIR = Path("outputs/figures/drafts/algorithm_params_by_node")
NODE_COUNTS = (100, 200, 300, 500)
SPACE_M = 100
PACKET_SIZE_BITS = 6400
INITIAL_ENERGY_J = 0.5
WEIGHT_PARAM_SETS = (
    (0.60, 0.25, 0.15),
    (0.75, 0.15, 0.10),
    (0.45, 0.45, 0.10),
    (0.45, 0.15, 0.40),
    (0.34, 0.33, 0.33),
)


def _load_rows() -> List[Dict[str, str]]:
    with SUMMARY_LOG.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], field: str) -> float:
    return float(row[field])


def _nice_upper(max_value: float, tick_step: int) -> int:
    if max_value <= 0:
        return tick_step
    return max(tick_step, int(math.ceil(max_value * 1.08 / tick_step) * tick_step))


def _format_float(value: float) -> str:
    return f"{value:g}"


def _format_weight_tuple(alpha: float, beta: float, gamma: float) -> str:
    return f"a={alpha:.2f}\nb={beta:.2f}\ng={gamma:.2f}"


def _matches_weight(row: Dict[str, str], alpha: float, beta: float, gamma: float) -> bool:
    return (
        abs(_f(row, "alpha_weight") - alpha) < 1e-9
        and abs(_f(row, "beta_weight") - beta) < 1e-9
        and abs(_f(row, "gamma_weight") - gamma) < 1e-9
    )


def _common_axis(rows: List[Dict[str, str]]) -> tuple[int, List[int]]:
    upper = _nice_upper(max(_f(row, "fnd_mean") + _f(row, "fnd_std") for row in rows), 200)
    return upper, list(range(0, upper + 1, 200))


def _save_numeric_sweep(
    rows: List[Dict[str, str]],
    *,
    suite: str,
    x_field: str,
    title: str,
    xlabel: str,
    output_name: str,
    y_upper: int,
    y_ticks: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    suite_rows = [row for row in rows if row["suite"] == suite]
    x_values = sorted({_f(row, x_field) for row in suite_rows})
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    color_map = plt.get_cmap("tab10", len(NODE_COUNTS))

    for idx, node_count in enumerate(NODE_COUNTS):
        series = sorted(
            [row for row in suite_rows if int(row["node_count"]) == node_count],
            key=lambda row: _f(row, x_field),
        )
        ax.errorbar(
            [_f(row, x_field) for row in series],
            [_f(row, "fnd_mean") for row in series],
            yerr=[_f(row, "fnd_std") for row in series],
            marker="o",
            capsize=4,
            linewidth=2.0,
            color=color_map(idx),
            label=f"n={node_count}",
        )

    x_min = min(x_values)
    x_max = max(x_values)
    x_pad = max((x_max - x_min) * 0.08, 0.04)
    ax.set_title(title)
    fig.suptitle(
        f"Algorithm parameter sweep by node count, space={SPACE_M}m, "
        f"pkt={PACKET_SIZE_BITS}, Eini={INITIAL_ENERGY_J:g}J",
        fontsize=10,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FND round")
    ax.set_xticks(x_values, [_format_float(value) for value in x_values])
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0, y_upper)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _save_weight_sweep(rows: List[Dict[str, str]], y_upper: int, y_ticks: Sequence[int]) -> None:
    import matplotlib.pyplot as plt

    suite_rows = [row for row in rows if row["suite"] == "weight_by_node_v2"]
    labels = [_format_weight_tuple(alpha, beta, gamma) for alpha, beta, gamma in WEIGHT_PARAM_SETS]
    x_positions = list(range(len(labels)))
    width = 0.17
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    color_map = plt.get_cmap("tab10", len(NODE_COUNTS))

    for idx, node_count in enumerate(NODE_COUNTS):
        series = []
        for alpha, beta, gamma in WEIGHT_PARAM_SETS:
            matches = [
                row
                for row in suite_rows
                if int(row["node_count"]) == node_count and _matches_weight(row, alpha, beta, gamma)
            ]
            if matches:
                series.append(matches[0])
        offsets = [x + (idx - 1.5) * width for x in x_positions]
        ax.bar(
            offsets,
            [_f(row, "fnd_mean") for row in series],
            yerr=[_f(row, "fnd_std") for row in series],
            width=width,
            capsize=3,
            color=color_map(idx),
            alpha=0.82,
            label=f"n={node_count}",
        )

    ax.set_title("EULC alpha/beta/gamma")
    fig.suptitle(
        f"Algorithm parameter sweep by node count, space={SPACE_M}m, "
        f"pkt={PACKET_SIZE_BITS}, Eini={INITIAL_ENERGY_J:g}J",
        fontsize=10,
    )
    ax.set_xlabel("alpha / beta / gamma")
    ax.set_ylabel("FND round")
    ax.set_xticks(x_positions, labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, y_upper)
    ax.set_yticks(y_ticks)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "04_alpha_beta_gamma_by_node.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _save_cv_overview(rows: List[Dict[str, str]], y_upper: int, y_ticks: Sequence[int]) -> None:
    import matplotlib.pyplot as plt

    suites = [
        ("omega_by_node_v2", "omega"),
        ("candidate_by_node_v2", "candidate"),
        ("ch_ratio_by_node_v2", "Pc"),
        ("weight_by_node_v2", "alpha/beta/gamma"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), sharey=True, constrained_layout=True)
    for ax, (suite, title) in zip(axes, suites):
        suite_rows = [row for row in rows if row["suite"] == suite]
        for node_count in NODE_COUNTS:
            values = [_f(row, "fnd_cv_pct") for row in suite_rows if int(row["node_count"]) == node_count]
            ax.scatter([node_count] * len(values), values, s=34, alpha=0.78)
        ax.axhline(10, color="#D62728", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Number of nodes")
        ax.set_xticks(NODE_COUNTS)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.grid(True, linestyle="--", alpha=0.28)
        ax.tick_params(axis="y", labelleft=True)
    axes[0].set_ylabel("CV (%)")
    fig.suptitle("CV distribution by algorithm parameter and node count")
    output_path = OUTPUT_DIR / "05_cv_overview_by_node.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    global INITIAL_ENERGY_J, NODE_COUNTS, OUTPUT_DIR, PACKET_SIZE_BITS, SPACE_M, SUMMARY_LOG

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY_LOG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--space-m", type=int, default=SPACE_M)
    parser.add_argument("--node-counts", type=str, default="100,200,300,500")
    parser.add_argument("--packet-size-bits", type=int, default=PACKET_SIZE_BITS)
    parser.add_argument("--initial-energy-j", type=float, default=INITIAL_ENERGY_J)
    args = parser.parse_args()

    SUMMARY_LOG = args.summary
    OUTPUT_DIR = args.output_dir
    SPACE_M = args.space_m
    NODE_COUNTS = tuple(int(value.strip()) for value in args.node_counts.split(",") if value.strip())
    PACKET_SIZE_BITS = args.packet_size_bits
    INITIAL_ENERGY_J = args.initial_energy_j

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    y_upper, y_ticks = _common_axis(rows)

    _save_numeric_sweep(
        rows,
        suite="omega_by_node_v2",
        x_field="pso_omega",
        title="PSO objective weight omega",
        xlabel="pso_omega",
        output_name="01_pso_omega_by_node.png",
        y_upper=y_upper,
        y_ticks=y_ticks,
    )
    _save_numeric_sweep(
        rows,
        suite="candidate_by_node_v2",
        x_field="eulc_candidate_ratio",
        title="EULC candidate ratio",
        xlabel="eulc_candidate_ratio",
        output_name="02_eulc_candidate_ratio_by_node.png",
        y_upper=y_upper,
        y_ticks=y_ticks,
    )
    _save_numeric_sweep(
        rows,
        suite="ch_ratio_by_node_v2",
        x_field="cluster_head_ratio",
        title="Cluster-head ratio",
        xlabel="cluster_head_ratio",
        output_name="03_cluster_head_ratio_by_node.png",
        y_upper=y_upper,
        y_ticks=y_ticks,
    )
    _save_weight_sweep(rows, y_upper, y_ticks)

    max_cv = max(_f(row, "fnd_cv_pct") for row in rows)
    cv_upper = _nice_upper(max(12.0, max_cv), 2)
    _save_cv_overview(rows, cv_upper, list(range(0, cv_upper + 1, 2)))


if __name__ == "__main__":
    main()
