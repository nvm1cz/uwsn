from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from plot_stability_stress_v2 import build_combined_summary


OUTPUT_DIR = Path("outputs/figures/final_barcharts")
STRESS_SUMMARY = Path("outputs/data/csv/summary/stability_stress_v2_combined_summary.csv")
WEIGHT_SUMMARY = Path("outputs/data/csv/summary/stability_weight_v2_summary.csv")
STABILITY_30_SUMMARY = Path("outputs/data/csv/summary/stability_30seed_v2_summary.csv")
ALGORITHM_NODE_SUMMARY = Path("outputs/data/csv/summary/algorithm_node_sweep_v2_summary.csv")
ALGORITHM_SPACE500_NODE_SUMMARY = Path("outputs/data/csv/summary/algorithm_space500_node_sweep_v2_r5_summary.csv")

NODE_COUNTS_100 = (100, 200, 300, 500)
NODE_COUNTS_500 = (1000, 1500, 2000, 2500)
INITIAL_ENERGIES = (0.5, 1.0)
PACKET_SIZES = (4000, 6400)
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


def _axis_from_rows(rows: Iterable[Dict[str, str]], tick_step: int) -> Tuple[int, List[int]]:
    values = [_f(row, "fnd_mean") + _f(row, "fnd_std") for row in rows if row.get("fnd_mean")]
    upper = _nice_upper(max(values), tick_step)
    return upper, list(range(0, upper + 1, tick_step))


def _save_bar(
    *,
    labels: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    cvs: Sequence[float] | None,
    title: str,
    subtitle: str,
    xlabel: str,
    output_path: Path,
    y_upper: int,
    y_ticks: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    bars = ax.bar(x_positions, means, yerr=stds, capsize=5, color="#4C78A8", alpha=0.84, width=0.68)
    ax.set_title(title)
    fig.suptitle(subtitle, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FND round")
    ax.set_xticks(x_positions, labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, y_upper)
    ax.set_yticks(y_ticks)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for idx, (bar, mean_value, std_value) in enumerate(zip(bars, means, stds)):
        label = f"{mean_value:.0f}"
        if cvs is not None:
            label = f"{label}\nCV={cvs[idx]:.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_value + std_value + y_upper * 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _save_grouped_bar(
    *,
    x_labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], Sequence[float]]],
    title: str,
    subtitle: str,
    xlabel: str,
    output_path: Path,
    y_upper: int,
    y_ticks: Sequence[int],
    annotate: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_positions = list(range(len(x_labels)))
    width = min(0.18, 0.78 / max(1, len(series)))
    color_map = plt.get_cmap("tab10", len(series))
    fig, ax = plt.subplots(figsize=(8.2, 5.3))

    for idx, (label, means, stds) in enumerate(series):
        offsets = [x + (idx - (len(series) - 1) / 2) * width for x in x_positions]
        bars = ax.bar(
            offsets,
            means,
            yerr=stds,
            capsize=3,
            width=width,
            color=color_map(idx),
            alpha=0.84,
            label=label,
        )
        if annotate:
            for bar, mean_value, std_value in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_value + std_value + y_upper * 0.015,
                    f"{mean_value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_title(title)
    fig.suptitle(subtitle, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FND round")
    ax.set_xticks(x_positions, x_labels)
    ax.set_xlim(-0.6, len(x_labels) - 0.4)
    ax.set_ylim(0, y_upper)
    ax.set_yticks(y_ticks)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _rows_by_numeric(
    rows: List[Dict[str, str]],
    suite: str,
    field: str,
) -> List[Dict[str, str]]:
    return sorted([row for row in rows if row["suite"] == suite], key=lambda row: _f(row, field))


def save_algorithm_representative(stress_rows: List[Dict[str, str]], weight_rows: List[Dict[str, str]]) -> None:
    out_dir = OUTPUT_DIR / "algorithm_space100_representative"
    suites = {
        "omega_stress_100m_v2",
        "candidate_stress_100m_v2",
        "ch_ratio_stress_100m_v2",
    }
    axis_rows = [row for row in stress_rows if row["suite"] in suites] + weight_rows
    y_upper, y_ticks = _axis_from_rows(axis_rows, 200)
    subtitle = "Barchart + errorbar, 20 seeds/case, space=100m, n=300, pkt=6400, Eini=0.5J"

    specs = [
        ("omega_stress_100m_v2", "pso_omega", "PSO objective weight omega", "pso_omega", "01_pso_omega_bar.png"),
        (
            "candidate_stress_100m_v2",
            "eulc_candidate_ratio",
            "EULC candidate ratio",
            "eulc_candidate_ratio",
            "02_eulc_candidate_ratio_bar.png",
        ),
        (
            "ch_ratio_stress_100m_v2",
            "cluster_head_ratio",
            "Cluster-head ratio",
            "cluster_head_ratio",
            "03_cluster_head_ratio_bar.png",
        ),
    ]
    for suite, field, title, xlabel, filename in specs:
        rows = _rows_by_numeric(stress_rows, suite, field)
        _save_bar(
            labels=[_format_float(_f(row, field)) for row in rows],
            means=[_f(row, "fnd_mean") for row in rows],
            stds=[_f(row, "fnd_std") for row in rows],
            cvs=[_f(row, "fnd_cv_pct") for row in rows],
            title=title,
            subtitle=subtitle,
            xlabel=xlabel,
            output_path=out_dir / filename,
            y_upper=y_upper,
            y_ticks=y_ticks,
        )

    _save_bar(
        labels=[
            _format_weight_tuple(_f(row, "alpha_weight"), _f(row, "beta_weight"), _f(row, "gamma_weight"))
            for row in weight_rows
        ],
        means=[_f(row, "fnd_mean") for row in weight_rows],
        stds=[_f(row, "fnd_std") for row in weight_rows],
        cvs=[_f(row, "fnd_cv_pct") for row in weight_rows],
        title="EULC alpha/beta/gamma",
        subtitle=subtitle,
        xlabel="alpha / beta / gamma",
        output_path=out_dir / "04_alpha_beta_gamma_bar.png",
        y_upper=y_upper,
        y_ticks=y_ticks,
    )


def _grouped_by_node_series(
    rows: List[Dict[str, str]],
    *,
    suite: str,
    x_field: str | None,
    node_counts: Sequence[int],
    x_labels: Sequence[str] | None = None,
) -> Tuple[List[str], List[Tuple[str, List[float], List[float]]]]:
    suite_rows = [row for row in rows if row["suite"] == suite]
    if x_labels is None and x_field is not None:
        x_values = sorted({_f(row, x_field) for row in suite_rows})
        x_labels = [_format_float(value) for value in x_values]
    else:
        x_values = None

    series = []
    for node_count in node_counts:
        means: List[float] = []
        stds: List[float] = []
        for label in x_labels or []:
            if x_values is not None and x_field is not None:
                target_value = float(label)
                matches = [
                    row
                    for row in suite_rows
                    if int(row["node_count"]) == node_count and abs(_f(row, x_field) - target_value) < 1e-9
                ]
            else:
                matches = [
                    row
                    for row in suite_rows
                    if int(row["node_count"]) == node_count and row["label"] == label.replace("\n", "-")
                ]
            means.append(_f(matches[0], "fnd_mean") if matches else 0.0)
            stds.append(_f(matches[0], "fnd_std") if matches else 0.0)
        series.append((f"n={node_count}", means, stds))
    return list(x_labels or []), series


def save_algorithm_by_node(
    rows: List[Dict[str, str]],
    *,
    out_dir: Path,
    node_counts: Sequence[int],
    space_m: int,
    runs_label: str,
    y_upper: int,
    y_ticks: Sequence[int],
) -> None:
    subtitle = f"Barchart + errorbar, {runs_label}, space={space_m}m, pkt=6400, Eini=0.5J"
    specs = [
        ("omega_by_node_v2", "pso_omega", "PSO objective weight omega", "pso_omega", "01_pso_omega_by_node_bar.png"),
        (
            "candidate_by_node_v2",
            "eulc_candidate_ratio",
            "EULC candidate ratio",
            "eulc_candidate_ratio",
            "02_eulc_candidate_ratio_by_node_bar.png",
        ),
        (
            "ch_ratio_by_node_v2",
            "cluster_head_ratio",
            "Cluster-head ratio",
            "cluster_head_ratio",
            "03_cluster_head_ratio_by_node_bar.png",
        ),
    ]
    for suite, field, title, xlabel, filename in specs:
        labels, series = _grouped_by_node_series(rows, suite=suite, x_field=field, node_counts=node_counts)
        _save_grouped_bar(
            x_labels=labels,
            series=series,
            title=title,
            subtitle=subtitle,
            xlabel=xlabel,
            output_path=out_dir / filename,
            y_upper=y_upper,
            y_ticks=y_ticks,
        )

    labels = [_format_weight_tuple(alpha, beta, gamma) for alpha, beta, gamma in WEIGHT_PARAM_SETS]
    series = []
    weight_rows = [row for row in rows if row["suite"] == "weight_by_node_v2"]
    for node_count in node_counts:
        means = []
        stds = []
        for alpha, beta, gamma in WEIGHT_PARAM_SETS:
            match = [
                row
                for row in weight_rows
                if int(row["node_count"]) == node_count and _matches_weight(row, alpha, beta, gamma)
            ][0]
            means.append(_f(match, "fnd_mean"))
            stds.append(_f(match, "fnd_std"))
        series.append((f"n={node_count}", means, stds))
    _save_grouped_bar(
        x_labels=labels,
        series=series,
        title="EULC alpha/beta/gamma",
        subtitle=subtitle,
        xlabel="alpha / beta / gamma",
        output_path=out_dir / "04_alpha_beta_gamma_by_node_bar.png",
        y_upper=y_upper,
        y_ticks=y_ticks,
    )


def _baseline_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if float(row["pso_c1"]) == 1.5 and float(row["pso_c2"]) == 1.5
    ]


def save_four_case_grid(
    rows: List[Dict[str, str]],
    *,
    space_m: int,
    output_path: Path,
    y_upper: int,
    y_ticks: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2), sharey=True, constrained_layout=True)
    specs = [
        (4000, 0.5),
        (6400, 0.5),
        (4000, 1.0),
        (6400, 1.0),
    ]

    for ax, (packet_size, initial_energy) in zip(axes.flat, specs):
        panel_rows = sorted(
            [
                row
                for row in rows
                if int(row["packet_size_bits"]) == packet_size
                and float(row["initial_energy_j"]) == initial_energy
            ],
            key=lambda row: int(row["node_count"]),
        )
        labels = [row["node_count"] for row in panel_rows]
        means = [_f(row, "fnd_mean") for row in panel_rows]
        stds = [_f(row, "fnd_std") for row in panel_rows]
        cvs = [_f(row, "fnd_cv_pct") for row in panel_rows]
        positions = list(range(len(labels)))
        bars = ax.bar(positions, means, yerr=stds, capsize=4, color="#4C78A8", alpha=0.84, width=0.66)
        for bar, mean_value, std_value, cv_value in zip(bars, means, stds, cvs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean_value + std_value + y_upper * 0.015,
                f"{mean_value:.0f}\nCV={cv_value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"pkt={packet_size}, Eini={initial_energy:g}J")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("FND round")
        ax.set_xticks(positions, labels)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(f"FND stability across 4 cases - space={space_m}m, 30 seeds/case, objective v2")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_system_space100(stability_rows: List[Dict[str, str]], stress_rows: List[Dict[str, str]]) -> None:
    out_dir = OUTPUT_DIR / "system_space100"
    baseline = _baseline_rows(stability_rows)
    range_rows = [row for row in stress_rows if row["suite"] == "range_stress_100m_v2"]
    y_upper, y_ticks = _axis_from_rows(baseline + range_rows, 500)
    subtitle = "Barchart + errorbar, baseline c=(1.5,1.5), objective v2"

    for packet_size in PACKET_SIZES:
        x_labels = [_format_float(value) for value in INITIAL_ENERGIES]
        series = []
        for node_count in NODE_COUNTS_100:
            means = []
            stds = []
            for initial_energy in INITIAL_ENERGIES:
                match = [
                    row
                    for row in baseline
                    if int(row["packet_size_bits"]) == packet_size
                    and int(row["node_count"]) == node_count
                    and float(row["initial_energy_j"]) == initial_energy
                ][0]
                means.append(_f(match, "fnd_mean"))
                stds.append(_f(match, "fnd_std"))
            series.append((f"n={node_count}", means, stds))
        _save_grouped_bar(
            x_labels=x_labels,
            series=series,
            title=f"Initial energy - pkt={packet_size}",
            subtitle=subtitle,
            xlabel="Eini (J)",
            output_path=out_dir / f"01_initial_energy_pkt{packet_size}_bar.png",
            y_upper=y_upper,
            y_ticks=y_ticks,
        )

    for initial_energy in INITIAL_ENERGIES:
        x_labels = [str(value) for value in PACKET_SIZES]
        series = []
        for node_count in NODE_COUNTS_100:
            means = []
            stds = []
            for packet_size in PACKET_SIZES:
                match = [
                    row
                    for row in baseline
                    if int(row["packet_size_bits"]) == packet_size
                    and int(row["node_count"]) == node_count
                    and float(row["initial_energy_j"]) == initial_energy
                ][0]
                means.append(_f(match, "fnd_mean"))
                stds.append(_f(match, "fnd_std"))
            series.append((f"n={node_count}", means, stds))
        energy_label = str(initial_energy).replace(".", "p")
        _save_grouped_bar(
            x_labels=x_labels,
            series=series,
            title=f"Packet size - Eini={initial_energy:g}J",
            subtitle=subtitle,
            xlabel="Packet size (bits)",
            output_path=out_dir / f"02_packet_size_e{energy_label}_bar.png",
            y_upper=y_upper,
            y_ticks=y_ticks,
        )

    for packet_size in PACKET_SIZES:
        for initial_energy in INITIAL_ENERGIES:
            rows = sorted(
                [
                    row
                    for row in baseline
                    if int(row["packet_size_bits"]) == packet_size
                    and float(row["initial_energy_j"]) == initial_energy
                ],
                key=lambda row: int(row["node_count"]),
            )
            energy_label = str(initial_energy).replace(".", "p")
            _save_bar(
                labels=[row["node_count"] for row in rows],
                means=[_f(row, "fnd_mean") for row in rows],
                stds=[_f(row, "fnd_std") for row in rows],
                cvs=[_f(row, "fnd_cv_pct") for row in rows],
                title=f"Number of nodes - pkt={packet_size}, Eini={initial_energy:g}J",
                subtitle=subtitle,
                xlabel="Number of nodes",
                output_path=out_dir / f"03_node_count_pkt{packet_size}_e{energy_label}_bar.png",
                y_upper=y_upper,
                y_ticks=y_ticks,
            )

    range_rows = sorted(range_rows, key=lambda row: _f(row, "transmission_range_m"))
    _save_bar(
        labels=[_format_float(_f(row, "transmission_range_m")) for row in range_rows],
        means=[_f(row, "fnd_mean") for row in range_rows],
        stds=[_f(row, "fnd_std") for row in range_rows],
        cvs=[_f(row, "fnd_cv_pct") for row in range_rows],
        title="Transmission range",
        subtitle="Barchart + errorbar, 20 seeds/case, space=100m, n=300, pkt=6400, Eini=0.5J",
        xlabel="R (m)",
        output_path=out_dir / "04_transmission_range_m_bar.png",
        y_upper=y_upper,
        y_ticks=y_ticks,
    )


def save_space500(stress_rows: List[Dict[str, str]]) -> None:
    out_dir = OUTPUT_DIR / "space500"
    rows = [row for row in stress_rows if row["suite"] == "space500_scale_stress_v2"]
    y_upper, y_ticks = _axis_from_rows(rows, 500)
    subtitle = "Barchart + errorbar, common axes, objective v2"
    specs = [
        (4000, 0.5),
        (4000, 1.0),
        (6400, 0.5),
        (6400, 1.0),
    ]
    for packet_size, initial_energy in specs:
        series = sorted(
            [
                row
                for row in rows
                if int(row["packet_size_bits"]) == packet_size
                and float(row["initial_energy_j"]) == initial_energy
            ],
            key=lambda row: int(row["node_count"]),
        )
        energy_label = str(initial_energy).replace(".", "p")
        _save_bar(
            labels=[row["node_count"] for row in series],
            means=[_f(row, "fnd_mean") for row in series],
            stds=[_f(row, "fnd_std") for row in series],
            cvs=[_f(row, "fnd_cv_pct") for row in series],
            title=f"space=500m, pkt={packet_size}, Eini={initial_energy:g}J",
            subtitle=subtitle,
            xlabel="Number of nodes",
            output_path=out_dir / f"space500_pkt{packet_size}_e{energy_label}_bar.png",
            y_upper=y_upper,
            y_ticks=y_ticks,
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stress_rows = build_combined_summary()
    weight_rows = _load_csv(WEIGHT_SUMMARY)
    stability_rows = _load_csv(STABILITY_30_SUMMARY)
    algorithm_node_rows = _load_csv(ALGORITHM_NODE_SUMMARY)
    algorithm_space500_node_rows = (
        _load_csv(ALGORITHM_SPACE500_NODE_SUMMARY)
        if ALGORITHM_SPACE500_NODE_SUMMARY.exists()
        else []
    )
    algorithm_by_node_axis_rows = algorithm_node_rows + algorithm_space500_node_rows
    algorithm_by_node_y_upper, algorithm_by_node_y_ticks = _axis_from_rows(algorithm_by_node_axis_rows, 200)
    baseline_space100_rows = _baseline_rows(stability_rows)
    space500_rows = [row for row in stress_rows if row["suite"] == "space500_scale_stress_v2"]
    four_case_y_upper, four_case_y_ticks = _axis_from_rows(baseline_space100_rows + space500_rows, 500)

    save_algorithm_representative(stress_rows, weight_rows)
    save_algorithm_by_node(
        algorithm_node_rows,
        out_dir=OUTPUT_DIR / "algorithm_space100_by_node",
        node_counts=NODE_COUNTS_100,
        space_m=100,
        runs_label="10 seeds/case",
        y_upper=algorithm_by_node_y_upper,
        y_ticks=algorithm_by_node_y_ticks,
    )
    if algorithm_space500_node_rows:
        save_algorithm_by_node(
            algorithm_space500_node_rows,
            out_dir=OUTPUT_DIR / "algorithm_space500_by_node",
            node_counts=NODE_COUNTS_500,
            space_m=500,
            runs_label="5 seeds/case",
            y_upper=algorithm_by_node_y_upper,
            y_ticks=algorithm_by_node_y_ticks,
        )
    save_four_case_grid(
        baseline_space100_rows,
        space_m=100,
        output_path=OUTPUT_DIR / "system_space100" / "00_space100_4case_30seed_bar.png",
        y_upper=four_case_y_upper,
        y_ticks=four_case_y_ticks,
    )
    save_four_case_grid(
        space500_rows,
        space_m=500,
        output_path=OUTPUT_DIR / "space500" / "00_space500_4case_30seed_bar.png",
        y_upper=four_case_y_upper,
        y_ticks=four_case_y_ticks,
    )
    save_system_space100(stability_rows, stress_rows)
    save_space500(stress_rows)


if __name__ == "__main__":
    main()
