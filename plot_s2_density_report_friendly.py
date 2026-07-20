from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

from uwsn.run_config import SIMULATION_CASE, SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


SUMMARY_LOG = Path("outputs/data/csv/summary/s2_density_convergence_summary.csv")
NEIGHBOR_LOG = Path("outputs/data/csv/summary/s2_density_neighbor_summary.csv")
FIGURE_DIR = Path("outputs/figures/s2_density_realism")

S2_CASES = {
    100: (50, 70, 90),
    500: (100, 200, 300),
    1000: (500, 700, 1000),
}

NEIGHBOR_FIELDS = [
    "space_m",
    "node_count",
    "runs",
    "seed_start",
    "seed_end",
    "avg_neighbor_count_mean",
    "avg_neighbor_count_std",
    "node_density_per_m3",
]


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _nice_upper(max_value: float, tick_step: float) -> float:
    if max_value <= 0:
        return tick_step
    return max(tick_step, math.ceil(max_value * 1.10 / tick_step) * tick_step)


def _summary_rows() -> List[Dict[str, str]]:
    rows = _load_csv(SUMMARY_LOG)
    if not rows:
        raise FileNotFoundError(f"Missing summary log: {SUMMARY_LOG}")
    return sorted(rows, key=lambda row: (int(row["space_m"]), int(row["node_count"])))


def build_neighbor_summary(summary_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in summary_rows:
        space_m = int(summary["space_m"])
        node_count = int(summary["node_count"])
        seed_start = int(summary["seed_start"])
        seed_end = int(summary["seed_end"])
        values: List[float] = []
        for seed in range(seed_start, seed_end + 1):
            case = replace(
                SIMULATION_CASE,
                node_count=node_count,
                packet_size_bits=int(summary["packet_size_bits"]),
                initial_energy=float(summary["initial_energy_j"]),
            )
            params = replace(
                SIMULATION_PARAMS,
                width_m=float(space_m),
                height_m=float(space_m),
                depth_m=float(space_m),
                contention_penalty_factor=float(summary["contention_penalty_factor"]),
                pso_particles=int(summary["pso_particles"]),
                pso_iterations=int(summary["pso_iterations"]),
            )
            sim = PsoEulcSimulator(case, params, seed, verbose=False)
            values.append(float(sim.neighbor_count.mean()))

        rows.append(
            {
                "space_m": space_m,
                "node_count": node_count,
                "runs": len(values),
                "seed_start": seed_start,
                "seed_end": seed_end,
                "avg_neighbor_count_mean": round(mean(values), 4),
                "avg_neighbor_count_std": round(_std(values), 4),
                "node_density_per_m3": f"{node_count / (float(space_m) ** 3):.10f}",
            }
        )
    _write_rows(NEIGHBOR_LOG, NEIGHBOR_FIELDS, rows)
    return rows


def plot_relative_ft5(summary_rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    colors = ["#4C78A8", "#4C78A8", "#4C78A8"]
    for ax, space_m in zip(axes, S2_CASES):
        panel = [row for row in summary_rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        baseline = float(panel[0]["ft5_mean"])
        labels = [str(row["node_count"]) for row in panel]
        rel_values = [float(row["ft5_mean"]) / baseline * 100.0 for row in panel]
        positions = list(range(len(labels)))
        bars = ax.bar(positions, rel_values, color=colors[0], alpha=0.84, width=0.66)
        for bar, value in zip(bars, rel_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.2,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.axhline(100, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("FT5 relative to lowest-node case (%)")
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, 120)
        ax.set_yticks([0, 25, 50, 75, 100, 120])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Relative FT5 trend within each fixed space, density/contention model")
    output_path = FIGURE_DIR / "s2_density_ft5_relative_by_space.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_ft5_vs_neighbor(summary_rows: List[Dict[str, str]], neighbor_rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    neighbor_lookup = {
        (int(row["space_m"]), int(row["node_count"])): float(row["avg_neighbor_count_mean"])
        for row in neighbor_rows
    }
    points: List[Tuple[float, float, int, int]] = []
    for row in summary_rows:
        space_m = int(row["space_m"])
        node_count = int(row["node_count"])
        points.append(
            (
                neighbor_lookup[(space_m, node_count)],
                float(row["ft5_mean"]),
                space_m,
                node_count,
            )
        )
    points.sort(key=lambda item: item[0])

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    colors = {100: "tab:blue", 500: "tab:orange", 1000: "tab:green"}
    for space_m in S2_CASES:
        space_points = [point for point in points if point[2] == space_m]
        xs = [point[0] for point in space_points]
        ys = [point[1] for point in space_points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors[space_m], label=f"space={space_m}m")
        for x_value, y_value, _, node_count in space_points:
            ax.text(x_value, y_value + 18, f"n={node_count}", ha="center", va="bottom", fontsize=8)

    all_x = [point[0] for point in points]
    all_y = [point[1] for point in points]
    ax.plot(all_x, all_y, color="#333333", linewidth=1.0, linestyle="--", alpha=0.35)
    ax.set_title("FT5 vs average neighbor count, density/contention model")
    ax.set_xlabel("Average neighbors within transmission range")
    ax.set_ylabel("FT5 round")
    ax.set_ylim(0, _nice_upper(max(all_y), 250))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.92)

    output_path = FIGURE_DIR / "s2_density_ft5_vs_avg_neighbor_count.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    summary_rows = _summary_rows()
    neighbor_rows = build_neighbor_summary(summary_rows)
    plot_relative_ft5(summary_rows)
    plot_ft5_vs_neighbor(summary_rows, neighbor_rows)


if __name__ == "__main__":
    main()
