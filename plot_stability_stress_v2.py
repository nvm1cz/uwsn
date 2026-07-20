from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple


SUMMARY_LOG = Path("outputs/data/csv/summary/stability_stress_v2_summary.csv")
RAW_LOG = Path("outputs/data/csv/raw/stability_stress_v2_raw.csv")
COMBINED_SUMMARY_LOG = Path("outputs/data/csv/summary/stability_stress_v2_combined_summary.csv")
OUTPUT_DIR = Path("outputs/figures/stability")


def _load_rows() -> List[Dict[str, str]]:
    with SUMMARY_LOG.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_combined_summary() -> List[Dict[str, str]]:
    with RAW_LOG.open("r", encoding="utf-8", newline="") as f:
        raw_rows = list(csv.DictReader(f))

    deduped: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in raw_rows:
        key = (
            row["suite"],
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
            row["pso_particles"],
            row["pso_iterations"],
            row["seed"],
        )
        deduped[key] = row

    groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in deduped.values():
        key = (
            row["suite"],
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
            row["pso_particles"],
            row["pso_iterations"],
        )
        groups[key].append(row)

    fieldnames = [
        "suite",
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
        "pso_particles",
        "pso_iterations",
        "runs",
        "seed_min",
        "seed_max",
        "fnd_mean",
        "fnd_std",
        "fnd_min",
        "fnd_max",
        "fnd_cv_pct",
    ]
    combined_rows: List[Dict[str, str]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(item[0])):
        fnd_values = [float(row["fnd_round"]) for row in rows if row["fnd_round"]]
        seeds = [int(row["seed"]) for row in rows]
        if not fnd_values:
            continue
        fnd_mean = mean(fnd_values)
        fnd_std = pstdev(fnd_values) if len(fnd_values) > 1 else 0.0
        row = dict(zip(fieldnames[:13], key))
        row.update(
            {
                "runs": str(len(set(seeds))),
                "seed_min": str(min(seeds)),
                "seed_max": str(max(seeds)),
                "fnd_mean": f"{fnd_mean:.4f}",
                "fnd_std": f"{fnd_std:.4f}",
                "fnd_min": f"{min(fnd_values):.0f}",
                "fnd_max": f"{max(fnd_values):.0f}",
                "fnd_cv_pct": f"{(fnd_std / fnd_mean * 100.0):.4f}",
            }
        )
        combined_rows.append(row)

    with COMBINED_SUMMARY_LOG.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    print(f"Saved {COMBINED_SUMMARY_LOG}")
    return combined_rows


def _f(value: str) -> float:
    return float(value)


def _nice_upper(max_value: float, tick_step: int = 500) -> int:
    if max_value <= 0:
        return tick_step
    return max(tick_step, int(math.ceil(max_value * 1.08 / tick_step) * tick_step))


def save_param_stress_chart(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    specs = [
        ("omega_stress_100m_v2", "Objective weight omega", "pso_omega"),
        ("candidate_stress_100m_v2", "Candidate ratio", "eulc_candidate_ratio"),
        ("range_stress_100m_v2", "Transmission range", "transmission_range_m"),
    ]

    all_param_rows = [row for suite, _, _ in specs for row in rows if row["suite"] == suite]
    y_upper = _nice_upper(
        max(_f(row["fnd_mean"]) + _f(row["fnd_std"]) for row in all_param_rows),
        tick_step=200,
    )
    y_ticks = list(range(0, y_upper + 1, 200))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    for ax, (suite, title, x_field) in zip(axes, specs):
        suite_rows = sorted(
            [row for row in rows if row["suite"] == suite],
            key=lambda row: _f(row[x_field]),
        )
        x_labels = [f"{_f(row[x_field]):g}" for row in suite_rows]
        x_positions = list(range(len(suite_rows)))
        means = [_f(row["fnd_mean"]) for row in suite_rows]
        stds = [_f(row["fnd_std"]) for row in suite_rows]
        cvs = [_f(row["fnd_cv_pct"]) for row in suite_rows]

        bars = ax.bar(x_positions, means, yerr=stds, capsize=4, color="#4C78A8", alpha=0.82)
        ax.set_title(title)
        ax.set_xlabel(x_field)
        ax.set_ylabel("FND round")
        ax.set_xticks(x_positions, x_labels)
        ax.set_xlim(-0.6, len(x_positions) - 0.4)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        for bar, mean, cv in zip(bars, means, cvs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 16,
                f"{mean:.0f}\nCV={cv:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Parameter stress stability (20 seeds/case, space=100m, n=300, pkt=6400, Eini=0.5J)")
    output_path = OUTPUT_DIR / "stability_stress_v2_param_summary.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_individual_param_stress_charts(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    specs = [
        (
            "omega_stress_100m_v2",
            "Objective weight omega",
            "pso_omega",
            "pso_omega",
            "stability_stress_v2_pso_omega.png",
        ),
        (
            "candidate_stress_100m_v2",
            "EULC candidate ratio",
            "eulc_candidate_ratio",
            "eulc_candidate_ratio",
            "stability_stress_v2_eulc_candidate_ratio.png",
        ),
        (
            "range_stress_100m_v2",
            "Transmission range",
            "transmission_range_m",
            "transmission_range_m (m)",
            "stability_stress_v2_transmission_range_m.png",
        ),
    ]

    all_param_rows = [row for suite, _, _, _, _ in specs for row in rows if row["suite"] == suite]
    y_upper = _nice_upper(
        max(_f(row["fnd_mean"]) + _f(row["fnd_std"]) for row in all_param_rows),
        tick_step=200,
    )
    y_ticks = list(range(0, y_upper + 1, 200))

    for suite, title, x_field, x_label, filename in specs:
        suite_rows = sorted(
            [row for row in rows if row["suite"] == suite],
            key=lambda row: _f(row[x_field]),
        )
        x_labels = [f"{_f(row[x_field]):g}" for row in suite_rows]
        x_positions = list(range(len(suite_rows)))
        means = [_f(row["fnd_mean"]) for row in suite_rows]
        stds = [_f(row["fnd_std"]) for row in suite_rows]
        cvs = [_f(row["fnd_cv_pct"]) for row in suite_rows]

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
        ax.set_title(f"{title} stability")
        ax.set_xlabel(x_label)
        ax.set_ylabel("FND round")
        ax.set_xticks(x_positions, x_labels)
        ax.set_xlim(-0.6, len(x_positions) - 0.4)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        for bar, mean, std, cv in zip(bars, means, stds, cvs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + y_upper * 0.025,
                f"{mean:.0f}\nCV={cv:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.suptitle("20 seeds/case, space=100m, n=300, pkt=6400, Eini=0.5J", fontsize=10)
        fig.tight_layout()

        output_path = OUTPUT_DIR / filename
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def save_scale_stress_chart(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    scale_rows = [row for row in rows if row["suite"] == "space500_scale_stress_v2"]
    common_node_counts = sorted({int(row["node_count"]) for row in scale_rows})
    y_upper = _nice_upper(
        max(_f(row["fnd_mean"]) + _f(row["fnd_std"]) for row in scale_rows),
        tick_step=500,
    )
    y_ticks = list(range(0, y_upper + 1, 500))
    x_margin = 180
    x_limits = (min(common_node_counts) - x_margin, max(common_node_counts) + x_margin)
    panel_specs = [
        (4000, 0.5),
        (6400, 0.5),
        (6400, 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), sharex=True, sharey=True, constrained_layout=True)
    for ax, (packet_size, initial_energy) in zip(axes, panel_specs):
        panel_rows = sorted(
            [
                row
                for row in scale_rows
                if int(row["packet_size_bits"]) == packet_size
                and float(row["initial_energy_j"]) == initial_energy
            ],
            key=lambda row: int(row["node_count"]),
        )
        node_counts = [int(row["node_count"]) for row in panel_rows]
        means = [_f(row["fnd_mean"]) for row in panel_rows]
        stds = [_f(row["fnd_std"]) for row in panel_rows]
        cvs = [_f(row["fnd_cv_pct"]) for row in panel_rows]
        runs = [int(row["runs"]) for row in panel_rows]

        ax.errorbar(node_counts, means, yerr=stds, marker="o", linewidth=2.1, capsize=4)
        for node_count, mean, cv, run_count in zip(node_counts, means, cvs, runs):
            ax.annotate(
                f"{mean:.0f}\nCV={cv:.1f}%\n{run_count} seeds",
                xy=(node_count, mean),
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
        ax.set_xlim(*x_limits)
        ax.set_xticks(common_node_counts)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Space=500m scale stress stability (10-30 seeds/case, objective v2)")
    output_path = OUTPUT_DIR / "stability_stress_v2_space500_summary.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_combined_summary()
    save_param_stress_chart(rows)
    save_individual_param_stress_charts(rows)
    save_scale_stress_chart(rows)


if __name__ == "__main__":
    main()
