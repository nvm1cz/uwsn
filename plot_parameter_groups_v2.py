from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from plot_stability_stress_v2 import build_combined_summary


OUTPUT_DIR = Path("outputs")
ALGORITHM_DIR = OUTPUT_DIR / "figures/drafts/algorithm_params"
SYSTEM_DIR = OUTPUT_DIR / "figures/drafts/system_params"
STRESS_SUMMARY = OUTPUT_DIR / "data/csv/summary/stability_stress_v2_combined_summary.csv"
WEIGHT_SUMMARY = OUTPUT_DIR / "data/csv/summary/stability_weight_v2_summary.csv"
STABILITY_30_SUMMARY = OUTPUT_DIR / "data/csv/summary/stability_30seed_v2_summary.csv"


C_PAIRS = [(1.5, 1.5), (2.5, 0.5), (0.5, 2.5)]
NODE_COUNTS_100M = [100, 200, 300, 500]
PACKET_SIZES = [4000, 6400]
INITIAL_ENERGIES = [0.5, 1.0]


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _nice_upper(max_value: float, tick_step: int) -> int:
    if max_value <= 0:
        return tick_step
    return max(tick_step, int(math.ceil(max_value * 1.08 / tick_step) * tick_step))


def _f(row: Dict[str, str], field: str) -> float:
    return float(row[field])


def _format_float(value: float) -> str:
    return f"{value:g}"


def _format_c_pair(c1: float, c2: float) -> str:
    return f"c=({c1:g},{c2:g})"


def _format_weight_tuple(row: Dict[str, str]) -> str:
    return (
        f"a={_f(row, 'alpha_weight'):.2f}\n"
        f"b={_f(row, 'beta_weight'):.2f}\n"
        f"g={_f(row, 'gamma_weight'):.2f}"
    )


def _algo_common_axis(stress_rows: List[Dict[str, str]], weight_rows: List[Dict[str, str]]) -> Tuple[int, List[int]]:
    algo_suites = {
        "omega_stress_100m_v2",
        "candidate_stress_100m_v2",
        "ch_ratio_stress_100m_v2",
    }
    values = [
        _f(row, "fnd_mean") + _f(row, "fnd_std")
        for row in stress_rows
        if row["suite"] in algo_suites
    ]
    values.extend(_f(row, "fnd_mean") + _f(row, "fnd_std") for row in weight_rows)
    y_upper = _nice_upper(max(values), tick_step=200)
    return y_upper, list(range(0, y_upper + 1, 200))


def _save_bar_chart(
    *,
    rows: List[Dict[str, str]],
    labels: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    cvs: Sequence[float],
    title: str,
    subtitle: str,
    xlabel: str,
    output_path: Path,
    y_upper: int,
    y_ticks: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    x_positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=5,
        color="#4C78A8",
        alpha=0.82,
        width=0.72,
    )
    ax.set_title(title)
    fig.suptitle(subtitle, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FND round")
    ax.set_xticks(x_positions, labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, y_upper)
    ax.set_yticks(y_ticks)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar, mean_value, std_value, cv_value in zip(bars, means, stds, cvs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_value + std_value + y_upper * 0.025,
            f"{mean_value:.0f}\nCV={cv_value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _save_line_chart(
    *,
    rows: List[Dict[str, str]],
    x_values: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
    cvs: Sequence[float],
    title: str,
    subtitle: str,
    xlabel: str,
    output_path: Path,
    y_upper: int,
    y_ticks: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    if not x_values:
        return

    x_min = min(x_values)
    x_max = max(x_values)
    x_pad = max((x_max - x_min) * 0.08, 0.08)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.errorbar(
        x_values,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
        linewidth=2.0,
        color="#4C78A8",
    )
    ax.set_title(title)
    fig.suptitle(subtitle, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FND round")
    ax.set_xticks(x_values, [_format_float(value) for value in x_values])
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0, y_upper)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle="--", alpha=0.35)

    for x_value, mean_value, std_value, cv_value in zip(x_values, means, stds, cvs):
        ax.text(
            x_value,
            mean_value + std_value + y_upper * 0.025,
            f"{mean_value:.0f}\nCV={cv_value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_algorithm_parameter_figures(stress_rows: List[Dict[str, str]], stability_rows: List[Dict[str, str]]) -> None:
    ALGORITHM_DIR.mkdir(parents=True, exist_ok=True)
    weight_rows = _load_csv(WEIGHT_SUMMARY)
    y_upper, y_ticks = _algo_common_axis(stress_rows, weight_rows)
    subtitle = "20 seeds/case, space=100m, n=300, pkt=6400, Eini=0.5J"

    line_specs = [
        (
            "omega_stress_100m_v2",
            "PSO objective weight omega",
            "pso_omega",
            "pso_omega",
            "01_pso_omega.png",
        ),
        (
            "candidate_stress_100m_v2",
            "EULC candidate ratio",
            "eulc_candidate_ratio",
            "eulc_candidate_ratio",
            "02_eulc_candidate_ratio.png",
        ),
        (
            "ch_ratio_stress_100m_v2",
            "Cluster-head ratio",
            "cluster_head_ratio",
            "cluster_head_ratio",
            "03_cluster_head_ratio.png",
        ),
    ]

    for suite, title, x_field, xlabel, filename in line_specs:
        rows = sorted([row for row in stress_rows if row["suite"] == suite], key=lambda row: _f(row, x_field))
        _save_line_chart(
            rows=rows,
            x_values=[_f(row, x_field) for row in rows],
            means=[_f(row, "fnd_mean") for row in rows],
            stds=[_f(row, "fnd_std") for row in rows],
            cvs=[_f(row, "fnd_cv_pct") for row in rows],
            title=title,
            subtitle=subtitle,
            xlabel=xlabel,
            output_path=ALGORITHM_DIR / filename,
            y_upper=y_upper,
            y_ticks=y_ticks,
        )

    _save_bar_chart(
        rows=weight_rows,
        labels=[_format_weight_tuple(row) for row in weight_rows],
        means=[_f(row, "fnd_mean") for row in weight_rows],
        stds=[_f(row, "fnd_std") for row in weight_rows],
        cvs=[_f(row, "fnd_cv_pct") for row in weight_rows],
        title="EULC alpha/beta/gamma",
        subtitle=subtitle,
        xlabel="alpha / beta / gamma",
        output_path=ALGORITHM_DIR / "04_alpha_beta_gamma.png",
        y_upper=y_upper,
        y_ticks=y_ticks,
    )

    save_algorithm_c_pair_figures(stability_rows)


def save_algorithm_c_pair_figures(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    y_upper = _nice_upper(
        max(_f(row, "fnd_mean") + _f(row, "fnd_std") for row in rows),
        tick_step=500,
    )
    y_ticks = list(range(0, y_upper + 1, 500))
    x_limits = (min(NODE_COUNTS_100M) - 25, max(NODE_COUNTS_100M) + 25)
    color_map = plt.get_cmap("tab10", len(C_PAIRS))

    for packet_size in PACKET_SIZES:
        for initial_energy in INITIAL_ENERGIES:
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            for idx, (c1, c2) in enumerate(C_PAIRS):
                series = sorted(
                    [
                        row
                        for row in rows
                        if int(row["packet_size_bits"]) == packet_size
                        and float(row["initial_energy_j"]) == initial_energy
                        and float(row["pso_c1"]) == c1
                        and float(row["pso_c2"]) == c2
                    ],
                    key=lambda row: int(row["node_count"]),
                )
                node_counts = [int(row["node_count"]) for row in series]
                means = [_f(row, "fnd_mean") for row in series]
                stds = [_f(row, "fnd_std") for row in series]
                ax.errorbar(
                    node_counts,
                    means,
                    yerr=stds,
                    marker="o",
                    capsize=4,
                    linewidth=2.0,
                    color=color_map(idx),
                    label=_format_c_pair(c1, c2),
                )

            ax.set_title(f"PSO coefficients - pkt={packet_size}, Eini={initial_energy:g}J")
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel("FND round")
            ax.set_xlim(*x_limits)
            ax.set_xticks(NODE_COUNTS_100M)
            ax.set_ylim(0, y_upper)
            ax.set_yticks(y_ticks)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="lower right", framealpha=0.92)
            fig.suptitle("30 seeds/case, space=100m, objective v2", fontsize=10)
            fig.tight_layout()

            energy_label = str(initial_energy).replace(".", "p")
            output_path = ALGORITHM_DIR / f"05_pso_c1_c2_pkt{packet_size}_e{energy_label}.png"
            fig.savefig(output_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {output_path}")


def _baseline_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if float(row["pso_c1"]) == 1.5 and float(row["pso_c2"]) == 1.5
    ]


def save_system_parameter_figures(stress_rows: List[Dict[str, str]], stability_rows: List[Dict[str, str]]) -> None:
    SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _baseline_rows(stability_rows)
    y_upper = _nice_upper(
        max(_f(row, "fnd_mean") + _f(row, "fnd_std") for row in baseline),
        tick_step=500,
    )
    y_ticks = list(range(0, y_upper + 1, 500))

    save_system_initial_energy_figures(baseline, y_upper, y_ticks)
    save_system_packet_size_figures(baseline, y_upper, y_ticks)
    save_system_node_count_figures(baseline, y_upper, y_ticks)
    save_system_transmission_range_figure(stress_rows, y_upper, y_ticks)


def save_system_initial_energy_figures(rows: List[Dict[str, str]], y_upper: int, y_ticks: Sequence[int]) -> None:
    import matplotlib.pyplot as plt

    for packet_size in PACKET_SIZES:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for node_count in NODE_COUNTS_100M:
            series = sorted(
                [
                    row
                    for row in rows
                    if int(row["packet_size_bits"]) == packet_size
                    and int(row["node_count"]) == node_count
                ],
                key=lambda row: float(row["initial_energy_j"]),
            )
            ax.errorbar(
                [float(row["initial_energy_j"]) for row in series],
                [_f(row, "fnd_mean") for row in series],
                yerr=[_f(row, "fnd_std") for row in series],
                marker="o",
                capsize=4,
                linewidth=2.0,
                label=f"n={node_count}",
            )
        ax.set_title(f"Initial energy - pkt={packet_size}")
        ax.set_xlabel("Eini (J)")
        ax.set_ylabel("FND round")
        ax.set_xticks(INITIAL_ENERGIES)
        ax.set_xlim(min(INITIAL_ENERGIES) - 0.08, max(INITIAL_ENERGIES) + 0.08)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", framealpha=0.92)
        fig.suptitle("System parameter, 30 seeds/case, baseline c=(1.5,1.5)", fontsize=10)
        fig.tight_layout()
        output_path = SYSTEM_DIR / f"01_initial_energy_pkt{packet_size}.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def save_system_packet_size_figures(rows: List[Dict[str, str]], y_upper: int, y_ticks: Sequence[int]) -> None:
    import matplotlib.pyplot as plt

    for initial_energy in INITIAL_ENERGIES:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for node_count in NODE_COUNTS_100M:
            series = sorted(
                [
                    row
                    for row in rows
                    if float(row["initial_energy_j"]) == initial_energy
                    and int(row["node_count"]) == node_count
                ],
                key=lambda row: int(row["packet_size_bits"]),
            )
            ax.errorbar(
                [int(row["packet_size_bits"]) for row in series],
                [_f(row, "fnd_mean") for row in series],
                yerr=[_f(row, "fnd_std") for row in series],
                marker="o",
                capsize=4,
                linewidth=2.0,
                label=f"n={node_count}",
            )
        ax.set_title(f"Packet size - Eini={initial_energy:g}J")
        ax.set_xlabel("Packet size (bits)")
        ax.set_ylabel("FND round")
        ax.set_xticks(PACKET_SIZES)
        ax.set_xlim(min(PACKET_SIZES) - 300, max(PACKET_SIZES) + 300)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", framealpha=0.92)
        fig.suptitle("System parameter, 30 seeds/case, baseline c=(1.5,1.5)", fontsize=10)
        fig.tight_layout()
        energy_label = str(initial_energy).replace(".", "p")
        output_path = SYSTEM_DIR / f"02_packet_size_e{energy_label}.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def save_system_node_count_figures(rows: List[Dict[str, str]], y_upper: int, y_ticks: Sequence[int]) -> None:
    import matplotlib.pyplot as plt

    x_limits = (min(NODE_COUNTS_100M) - 25, max(NODE_COUNTS_100M) + 25)
    for packet_size in PACKET_SIZES:
        for initial_energy in INITIAL_ENERGIES:
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
            ax.errorbar(
                [int(row["node_count"]) for row in series],
                [_f(row, "fnd_mean") for row in series],
                yerr=[_f(row, "fnd_std") for row in series],
                marker="o",
                capsize=4,
                linewidth=2.0,
            )
            ax.set_title(f"Number of nodes - pkt={packet_size}, Eini={initial_energy:g}J")
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel("FND round")
            ax.set_xlim(*x_limits)
            ax.set_xticks(NODE_COUNTS_100M)
            ax.set_ylim(0, y_upper)
            ax.set_yticks(y_ticks)
            ax.grid(True, linestyle="--", alpha=0.35)
            fig.suptitle("System parameter, 30 seeds/case, baseline c=(1.5,1.5)", fontsize=10)
            fig.tight_layout()
            energy_label = str(initial_energy).replace(".", "p")
            output_path = SYSTEM_DIR / f"03_node_count_pkt{packet_size}_e{energy_label}.png"
            fig.savefig(output_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {output_path}")


def save_system_transmission_range_figure(
    stress_rows: List[Dict[str, str]],
    y_upper: int,
    y_ticks: Sequence[int],
) -> None:
    rows = sorted(
        [row for row in stress_rows if row["suite"] == "range_stress_100m_v2"],
        key=lambda row: _f(row, "transmission_range_m"),
    )
    _save_line_chart(
        rows=rows,
        x_values=[_f(row, "transmission_range_m") for row in rows],
        means=[_f(row, "fnd_mean") for row in rows],
        stds=[_f(row, "fnd_std") for row in rows],
        cvs=[_f(row, "fnd_cv_pct") for row in rows],
        title="Transmission range",
        subtitle="System parameter, 20 seeds/case, space=100m, n=300, pkt=6400, Eini=0.5J",
        xlabel="R (m)",
        output_path=SYSTEM_DIR / "04_transmission_range_m.png",
        y_upper=y_upper,
        y_ticks=y_ticks,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stress_rows = build_combined_summary()
    stability_rows = _load_csv(STABILITY_30_SUMMARY)
    save_algorithm_parameter_figures(stress_rows, stability_rows)
    save_system_parameter_figures(stress_rows, stability_rows)


if __name__ == "__main__":
    main()
