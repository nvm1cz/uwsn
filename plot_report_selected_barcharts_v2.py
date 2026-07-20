from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

from plot_stability_stress_v2 import build_combined_summary


OUTPUT_DIR = Path("outputs/report/selected_barcharts")
SPACE100_ALGO = Path("outputs/data/csv/summary/algorithm_node_sweep_v2_summary.csv")
SPACE500_ALGO = Path("outputs/data/csv/summary/algorithm_space500_node_sweep_v2_r5_summary.csv")
SPACE100_ALGO_RAW = Path("outputs/data/csv/raw/algorithm_node_sweep_v2_raw.csv")
SPACE500_ALGO_RAW = Path("outputs/data/csv/raw/algorithm_space500_node_sweep_v2_r5_raw.csv")
SPACE100_SYSTEM = Path("outputs/data/csv/summary/stability_30seed_v2_summary.csv")

NODE_COUNTS_BY_SPACE = {
    100: (100, 200, 300, 500),
    500: (1000, 1500, 2000, 2500),
}
PKT_ENERGY_CASES = (
    (4000, 0.5),
    (4000, 1.0),
    (6400, 0.5),
    (6400, 1.0),
)

WEIGHT_PARAM_SETS = (
    (0.60, 0.25, 0.15),
    (0.75, 0.15, 0.10),
    (0.45, 0.45, 0.10),
    (0.45, 0.15, 0.40),
    (0.34, 0.33, 0.33),
)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _combine_algorithm_raw(raw_path: Path) -> List[Dict[str, str]]:
    rows = _load_csv(raw_path)
    deduped: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        if not row.get("fnd_round"):
            continue
        key = (
            row["suite"],
            row["label"],
            row["space_m"],
            row["node_count"],
            row["packet_size_bits"],
            row["initial_energy_j"],
            row["pso_c1"],
            row["pso_c2"],
            row["pso_omega"],
            row["transmission_range_m"],
            row["cluster_head_ratio"],
            row["eulc_candidate_ratio"],
            row["alpha_weight"],
            row["beta_weight"],
            row["gamma_weight"],
            row["pso_particles"],
            row["pso_iterations"],
            row["seed"],
        )
        deduped[key] = row

    groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for row in deduped.values():
        key = (
            row["suite"],
            row["label"],
            row["space_m"],
            row["node_count"],
            row["packet_size_bits"],
            row["initial_energy_j"],
            row["pso_c1"],
            row["pso_c2"],
            row["pso_omega"],
            row["transmission_range_m"],
            row["cluster_head_ratio"],
            row["eulc_candidate_ratio"],
            row["alpha_weight"],
            row["beta_weight"],
            row["gamma_weight"],
            row["pso_particles"],
            row["pso_iterations"],
        )
        groups.setdefault(key, []).append(row)

    combined: List[Dict[str, str]] = []
    fields = [
        "suite",
        "label",
        "space_m",
        "node_count",
        "packet_size_bits",
        "initial_energy_j",
        "pso_c1",
        "pso_c2",
        "pso_omega",
        "transmission_range_m",
        "cluster_head_ratio",
        "eulc_candidate_ratio",
        "alpha_weight",
        "beta_weight",
        "gamma_weight",
        "pso_particles",
        "pso_iterations",
    ]
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        fnd_values = [float(row["fnd_round"]) for row in group_rows]
        fnd_mean = mean(fnd_values)
        fnd_std = pstdev(fnd_values) if len(fnd_values) > 1 else 0.0
        row = dict(zip(fields, key))
        row.update(
            {
                "runs": str(len({r["seed"] for r in group_rows})),
                "fnd_mean": f"{fnd_mean:.4f}",
                "fnd_std": f"{fnd_std:.4f}",
                "fnd_cv_pct": f"{(fnd_std / fnd_mean * 100.0):.4f}",
            }
        )
        combined.append(row)
    return combined


def _f(row: Dict[str, str], field: str) -> float:
    return float(row[field])


def _nice_upper(max_value: float, tick_step: int) -> int:
    if max_value <= 0:
        return tick_step
    return max(tick_step, int(math.ceil(max_value * 1.08 / tick_step) * tick_step))


def _format_float(value: float) -> str:
    return f"{value:g}"


def _format_weight(alpha: float, beta: float, gamma: float) -> str:
    return f"a={alpha:.2f}\nb={beta:.2f}\ng={gamma:.2f}"


def _matches_weight(row: Dict[str, str], alpha: float, beta: float, gamma: float) -> bool:
    return (
        abs(_f(row, "alpha_weight") - alpha) < 1e-9
        and abs(_f(row, "beta_weight") - beta) < 1e-9
        and abs(_f(row, "gamma_weight") - gamma) < 1e-9
    )


def _axis(rows: Sequence[Dict[str, str]], tick_step: int = 200) -> Tuple[int, List[int]]:
    y_upper = _nice_upper(max(_f(row, "fnd_mean") + _f(row, "fnd_std") for row in rows), tick_step)
    return y_upper, list(range(0, y_upper + 1, tick_step))


def _grouped_series(
    rows: List[Dict[str, str]],
    *,
    space_m: int,
    suite: str,
    x_field: str | None,
    x_labels: Sequence[str] | None = None,
) -> Tuple[List[str], List[Tuple[str, List[float], List[float]]]]:
    suite_rows = [row for row in rows if row["suite"] == suite]
    if x_labels is None and x_field:
        x_values = sorted({_f(row, x_field) for row in suite_rows})
        labels = [_format_float(value) for value in x_values]
    else:
        labels = list(x_labels or [])
        x_values = None

    series = []
    for node_count in NODE_COUNTS_BY_SPACE[space_m]:
        means: List[float] = []
        stds: List[float] = []
        for idx, label in enumerate(labels):
            if x_values is not None and x_field:
                value = x_values[idx]
                matches = [
                    row
                    for row in suite_rows
                    if int(row["node_count"]) == node_count and abs(_f(row, x_field) - value) < 1e-9
                ]
            else:
                alpha, beta, gamma = WEIGHT_PARAM_SETS[idx]
                matches = [
                    row
                    for row in suite_rows
                    if int(row["node_count"]) == node_count and _matches_weight(row, alpha, beta, gamma)
                ]
            means.append(_f(matches[0], "fnd_mean"))
            stds.append(_f(matches[0], "fnd_std"))
        series.append((f"n={node_count}", means, stds))
    return labels, series


def _draw_grouped_panel(
    ax,
    *,
    x_labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], Sequence[float]]],
    title: str,
    xlabel: str,
    y_upper: int,
    y_ticks: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    x_positions = list(range(len(x_labels)))
    width = min(0.18, 0.78 / max(1, len(series)))
    colors = plt.get_cmap("tab10", len(series))
    for idx, (label, means, stds) in enumerate(series):
        offsets = [x + (idx - (len(series) - 1) / 2) * width for x in x_positions]
        ax.bar(
            offsets,
            means,
            yerr=stds,
            capsize=3,
            width=width,
            color=colors(idx),
            alpha=0.84,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FND round")
    ax.set_xticks(x_positions, x_labels)
    ax.set_xlim(-0.6, len(x_labels) - 0.4)
    ax.set_ylim(0, y_upper)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="y", labelleft=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", framealpha=0.92)


def save_algorithm_pair(
    space100_rows: List[Dict[str, str]],
    space500_rows: List[Dict[str, str]],
    *,
    suite: str,
    x_field: str | None,
    title: str,
    xlabel: str,
    filename: str,
    weight_labels: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    labels = (
        [_format_weight(alpha, beta, gamma) for alpha, beta, gamma in WEIGHT_PARAM_SETS]
        if weight_labels
        else None
    )
    rows = [row for row in space100_rows + space500_rows if row["suite"] == suite]
    y_upper, y_ticks = _axis(rows, tick_step=200)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), sharey=True, constrained_layout=True)
    for ax, space_m, rows_for_space, runs_label in (
        (axes[0], 100, space100_rows, "30 seeds/case"),
        (axes[1], 500, space500_rows, "30 seeds/case"),
    ):
        x_labels, series = _grouped_series(
            rows_for_space,
            space_m=space_m,
            suite=suite,
            x_field=x_field,
            x_labels=labels,
        )
        _draw_grouped_panel(
            ax,
            x_labels=x_labels,
            series=series,
            title=f"space={space_m}m ({runs_label})",
            xlabel=xlabel,
            y_upper=y_upper,
            y_ticks=y_ticks,
        )
    fig.suptitle(f"{title} - pkt=6400, Eini=0.5J")
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_pkt_energy_overview(space100_system: List[Dict[str, str]], stress_rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    space100_rows = [
        row
        for row in space100_system
        if float(row["pso_c1"]) == 1.5 and float(row["pso_c2"]) == 1.5
    ]
    space500_rows = [
        row
        for row in stress_rows
        if row["suite"] == "space500_scale_stress_v2"
    ]
    y_upper, y_ticks = _axis(space100_rows + space500_rows, tick_step=500)

    for idx, (packet_size, initial_energy) in enumerate(PKT_ENERGY_CASES, start=1):
        panel_rows_100 = sorted(
            [
                row
                for row in space100_rows
                if int(row["packet_size_bits"]) == packet_size
                and float(row["initial_energy_j"]) == initial_energy
            ],
            key=lambda row: int(row["node_count"]),
        )
        panel_rows_500 = sorted(
            [
                row
                for row in space500_rows
                if int(row["packet_size_bits"]) == packet_size
                and float(row["initial_energy_j"]) == initial_energy
            ],
            key=lambda row: int(row["node_count"]),
        )

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharey=True, constrained_layout=True)
        for ax, rows, space_m, runs_label in (
            (axes[0], panel_rows_100, 100, "30 seeds/case"),
            (axes[1], panel_rows_500, 500, "30 seeds/case"),
        ):
            labels = [row["node_count"] for row in rows]
            means = [_f(row, "fnd_mean") for row in rows]
            stds = [_f(row, "fnd_std") for row in rows]
            cvs = [_f(row, "fnd_cv_pct") for row in rows]
            positions = list(range(len(labels)))
            bars = ax.bar(positions, means, yerr=stds, capsize=4, color="#4C78A8", alpha=0.84)
            for bar, mean_value, std_value, cv_value in zip(bars, means, stds, cvs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_value + std_value + y_upper * 0.015,
                    f"{mean_value:.0f}\nCV={cv_value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.set_title(f"space={space_m}m ({runs_label})")
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel("FND round")
            ax.set_xticks(positions, labels)
            ax.set_ylim(0, y_upper)
            ax.set_yticks(y_ticks)
            ax.tick_params(axis="y", labelleft=True)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        fig.suptitle(f"FND stability: pkt={packet_size}, Eini={initial_energy:g}J (objective v2)")
        energy_label = str(initial_energy).replace(".", "p")
        output_path = OUTPUT_DIR / f"01{chr(96 + idx)}_pkt{packet_size}_e{energy_label}_space100_vs_space500_bar.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def save_hard_case_pair(space100_system: List[Dict[str, str]], stress_rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    space100_rows = sorted(
        [
            row
            for row in space100_system
            if int(row["packet_size_bits"]) == 6400
            and float(row["initial_energy_j"]) == 0.5
            and float(row["pso_c1"]) == 1.5
            and float(row["pso_c2"]) == 1.5
        ],
        key=lambda row: int(row["node_count"]),
    )
    space500_rows = sorted(
        [
            row
            for row in stress_rows
            if row["suite"] == "space500_scale_stress_v2"
            and int(row["packet_size_bits"]) == 6400
            and float(row["initial_energy_j"]) == 0.5
        ],
        key=lambda row: int(row["node_count"]),
    )
    y_upper, y_ticks = _axis(space100_rows + space500_rows, tick_step=200)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), sharey=True, constrained_layout=True)
    for ax, rows, space_m, runs_label in (
        (axes[0], space100_rows, 100, "30 seeds/case"),
        (axes[1], space500_rows, 500, "10-30 seeds/case"),
    ):
        labels = [row["node_count"] for row in rows]
        means = [_f(row, "fnd_mean") for row in rows]
        stds = [_f(row, "fnd_std") for row in rows]
        cvs = [_f(row, "fnd_cv_pct") for row in rows]
        positions = list(range(len(labels)))
        bars = ax.bar(positions, means, yerr=stds, capsize=4, color="#4C78A8", alpha=0.84)
        for bar, mean_value, std_value, cv_value in zip(bars, means, stds, cvs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean_value + std_value + y_upper * 0.02,
                f"{mean_value:.0f}\nCV={cv_value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"space={space_m}m ({runs_label})")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("FND round")
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Hard-case stability: pkt=6400, Eini=0.5J, objective v2")
    output_path = OUTPUT_DIR / "01_hard_case_space100_vs_space500_bar.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    space100_algo = _combine_algorithm_raw(SPACE100_ALGO_RAW) if SPACE100_ALGO_RAW.exists() else _load_csv(SPACE100_ALGO)
    space500_algo = _combine_algorithm_raw(SPACE500_ALGO_RAW) if SPACE500_ALGO_RAW.exists() else _load_csv(SPACE500_ALGO)
    space100_system = _load_csv(SPACE100_SYSTEM)
    stress_rows = build_combined_summary()

    save_pkt_energy_overview(space100_system, stress_rows)
    save_algorithm_pair(
        space100_algo,
        space500_algo,
        suite="omega_by_node_v2",
        x_field="pso_omega",
        title="PSO objective weight omega",
        xlabel="pso_omega",
        filename="02_pso_omega_space100_vs_space500_bar.png",
    )
    save_algorithm_pair(
        space100_algo,
        space500_algo,
        suite="candidate_by_node_v2",
        x_field="eulc_candidate_ratio",
        title="EULC candidate ratio",
        xlabel="eulc_candidate_ratio",
        filename="03_candidate_ratio_space100_vs_space500_bar.png",
    )
    save_algorithm_pair(
        space100_algo,
        space500_algo,
        suite="ch_ratio_by_node_v2",
        x_field="cluster_head_ratio",
        title="Cluster-head ratio",
        xlabel="cluster_head_ratio",
        filename="04_cluster_head_ratio_space100_vs_space500_bar.png",
    )
    save_algorithm_pair(
        space100_algo,
        space500_algo,
        suite="weight_by_node_v2",
        x_field=None,
        title="EULC alpha/beta/gamma",
        xlabel="alpha / beta / gamma",
        filename="05_alpha_beta_gamma_space100_vs_space500_bar.png",
        weight_labels=True,
    )


if __name__ == "__main__":
    main()
