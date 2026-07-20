from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from uwsn.clustering import select_eulc_candidates
from uwsn.optimizers import (
    run_de_cluster_head_selection,
    run_ebrec_cluster_head_selection,
    run_eeumc_cluster_head_selection,
    run_eulc_cluster_head_selection,
    run_ga_cluster_head_selection,
    run_leach_cluster_head_selection,
    run_pso_cluster_head_selection,
)
from uwsn.run_config import SIMULATION_CASE, SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


RAW_CSV = Path("outputs/data/csv/raw/ch_optimizer_runtime_runs.csv")
SUMMARY_CSV = Path("outputs/data/csv/summary/ch_optimizer_runtime_summary.csv")
FIGURE_DIR = Path("outputs/figures/report_approved_optimizer_comparison")

NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
    1000: (500, 1000, 1500, 2000),
}
OPTIMIZERS = ("pso", "ga", "de", "leach", "eulc", "eeumc", "ebrec")
OPTIMIZER_FUNCTIONS = {
    "pso": run_pso_cluster_head_selection,
    "ga": run_ga_cluster_head_selection,
    "de": run_de_cluster_head_selection,
    "leach": run_leach_cluster_head_selection,
    "eulc": run_eulc_cluster_head_selection,
    "eeumc": run_eeumc_cluster_head_selection,
    "ebrec": run_ebrec_cluster_head_selection,
}
COLORS = {
    "pso": "#4C78A8",
    "ga": "#E45756",
    "de": "#54A24B",
    "leach": "#9D755D",
    "eulc": "#F58518",
    "eeumc": "#B279A2",
    "ebrec": "#72B7B2",
}
MARKERS = {
    "pso": "o",
    "ga": "s",
    "de": "^",
    "leach": "D",
    "eulc": "v",
    "eeumc": "P",
    "ebrec": "X",
}

RAW_FIELDS = [
    "optimizer",
    "space_m",
    "node_count",
    "seed",
    "population_size",
    "iterations",
    "candidate_count",
    "cluster_head_count",
    "optimizer_runtime_sec",
]
SUMMARY_FIELDS = [
    "optimizer",
    "space_m",
    "node_count",
    "runs",
    "population_size",
    "iterations",
    "candidate_count_mean",
    "cluster_head_count_mean",
    "optimizer_runtime_mean_sec",
    "optimizer_runtime_std_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure one cluster-head election runtime for all CH-selection algorithms."
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=12000)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: Dict[str, object]) -> Tuple[str, int, int, int, int, int]:
    return (
        str(row["optimizer"]),
        int(row["space_m"]),
        int(row["node_count"]),
        int(row["seed"]),
        int(row["population_size"]),
        int(row["iterations"]),
    )


def benchmark_case(
    space_m: int,
    node_count: int,
    seed: int,
    population_size: int,
    iterations: int,
) -> List[Dict[str, object]]:
    case = replace(
        SIMULATION_CASE,
        name=f"ch_runtime_s{space_m}_n{node_count}_seed{seed}",
        node_count=node_count,
        packet_size_bits=6400,
        initial_energy=0.5,
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        pso_particles=population_size,
        pso_iterations=iterations,
        contention_penalty_factor=0.01,
        contention_reference_degree=10.0,
        contention_max_multiplier=4.0,
    )
    sim = PsoEulcSimulator(case, params, seed, verbose=False)
    candidates = select_eulc_candidates(
        case,
        params,
        sim.energies,
        sim.layers,
        sim.dist_to_sink,
        sim.neighbor_degree,
        sim.distance_matrix,
    )

    # Rotate execution order to reduce systematic warm-cache/order bias.
    offset = seed % len(OPTIMIZERS)
    optimizer_order = OPTIMIZERS[offset:] + OPTIMIZERS[:offset]
    rows: List[Dict[str, object]] = []
    for optimizer in optimizer_order:
        optimizer_fn = OPTIMIZER_FUNCTIONS[optimizer]
        optimizer_rng = np.random.default_rng(seed + 1_000_003)
        started = perf_counter()
        cluster_heads, _ = optimizer_fn(
            case,
            params,
            sim.distance_matrix,
            sim.energies.copy(),
            sim.layers,
            sim.dist_to_sink,
            candidates,
            optimizer_rng,
        )
        elapsed = perf_counter() - started
        rows.append(
            {
                "optimizer": optimizer,
                "space_m": space_m,
                "node_count": node_count,
                "seed": seed,
                "population_size": population_size,
                "iterations": iterations,
                "candidate_count": len(candidates),
                "cluster_head_count": len(cluster_heads),
                "optimizer_runtime_sec": round(elapsed, 8),
            }
        )
    return rows


def build_summary(
    rows: List[Dict[str, object]],
    population_size: int,
    iterations: int,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int], List[Dict[str, object]]] = {}
    for row in rows:
        if (
            int(row["population_size"]) == population_size
            and int(row["iterations"]) == iterations
        ):
            grouped.setdefault(
                (str(row["optimizer"]), int(row["space_m"]), int(row["node_count"])),
                [],
            ).append(row)

    summary: List[Dict[str, object]] = []
    for (optimizer, space_m, node_count), group in sorted(grouped.items()):
        runtimes = [float(row["optimizer_runtime_sec"]) for row in group]
        candidate_counts = [float(row["candidate_count"]) for row in group]
        ch_counts = [float(row["cluster_head_count"]) for row in group]
        summary.append(
            {
                "optimizer": optimizer,
                "space_m": space_m,
                "node_count": node_count,
                "runs": len(group),
                "population_size": population_size,
                "iterations": iterations,
                "candidate_count_mean": round(mean(candidate_counts), 4),
                "cluster_head_count_mean": round(mean(ch_counts), 4),
                "optimizer_runtime_mean_sec": round(mean(runtimes), 8),
                "optimizer_runtime_std_sec": round(
                    pstdev(runtimes) if len(runtimes) > 1 else 0.0,
                    8,
                ),
            }
        )
    return summary


def plot_summary(summary: List[Dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    for ax, space_m in zip(axes, (100, 500, 1000)):
        for optimizer in OPTIMIZERS:
            panel = [
                row
                for row in summary
                if str(row["optimizer"]) == optimizer
                and int(row["space_m"]) == space_m
            ]
            panel.sort(key=lambda row: int(row["node_count"]))
            xs = [int(row["node_count"]) for row in panel]
            means_ms = [
                1000.0 * float(row["optimizer_runtime_mean_sec"]) for row in panel
            ]
            stds_ms = [
                1000.0 * float(row["optimizer_runtime_std_sec"]) for row in panel
            ]
            ax.errorbar(
                xs,
                means_ms,
                yerr=stds_ms,
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                linewidth=1.9,
                capsize=3,
                label=optimizer.upper(),
            )
        ax.set_title(f"KhÃ´ng gian {space_m} m")
        ax.set_xlabel("Sá»‘ nÃºt cáº£m biáº¿n")
        ax.set_ylabel("Thá»i gian báº§u CH (ms)")
        ax.set_xticks(NODE_SETS[space_m])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", framealpha=0.92)

    first = summary[0]
    fig.suptitle("So sÃ¡nh thuat toan theo thoi gian bau CH")
    output = FIGURE_DIR / "07_ch_optimizer_runtime_mean_std_all.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    existing = load_rows(RAW_CSV)
    rows_by_key: Dict[Tuple[str, int, int, int, int, int], Dict[str, object]] = {
        row_key(row): dict(row) for row in existing
    }

    total_cases = sum(len(nodes) for nodes in NODE_SETS.values()) * args.runs
    completed_cases = 0
    for space_m, node_counts in NODE_SETS.items():
        for node_count in node_counts:
            for seed in range(args.seed_start, args.seed_start + args.runs):
                expected_keys = [
                    (optimizer, space_m, node_count, seed, args.population_size, args.iterations)
                    for optimizer in OPTIMIZERS
                ]
                if not all(key in rows_by_key for key in expected_keys):
                    for row in benchmark_case(
                        space_m,
                        node_count,
                        seed,
                        args.population_size,
                        args.iterations,
                    ):
                        rows_by_key[row_key(row)] = row
                    write_rows(
                        RAW_CSV,
                        RAW_FIELDS,
                        sorted(rows_by_key.values(), key=row_key),
                    )
                completed_cases += 1
                print(
                    f"[{completed_cases}/{total_cases}] space={space_m}, "
                    f"n={node_count}, seed={seed}",
                    flush=True,
                )

    all_rows = sorted(rows_by_key.values(), key=row_key)
    summary = build_summary(all_rows, args.population_size, args.iterations)
    write_rows(SUMMARY_CSV, SUMMARY_FIELDS, summary)
    output = plot_summary(summary)
    print(f"Saved {RAW_CSV}")
    print(f"Saved {SUMMARY_CSV}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

