from __future__ import annotations

import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

from uwsn.run_config import SIMULATION_CASE, SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


RAW_LOG = Path("outputs/data/csv/raw/report_node_set_convergence_50iter.csv")
SUMMARY_LOG = Path("outputs/data/csv/summary/report_node_set_convergence_50iter_summary.csv")
FIGURE_PATH = Path("outputs/figures/report_node_set_smooth/report_node_set_pso_convergence_by_node_count.png")
FIGURE_PATH_50 = Path("outputs/figures/report_node_set_smooth/report_node_set_pso_convergence_by_node_count_50iter.png")
ALGORITHM_LABEL = "PSO-EULC (PSO)"

REPORT_NODE_SETS = {
    100: (50, 100, 150, 200),
    500: (100, 200, 300, 400, 500),
    1000: (200, 300, 400, 500),
}

PSO_ITERATIONS = 50
RUNS = 30
SEED_START = 9500

RAW_FIELDS = [
    "space_m",
    "node_count",
    "seed",
    "iteration",
    "best_score",
    "relative_best_score",
    "candidate_count",
    "avg_neighbor_count",
    "runtime_sec",
]

SUMMARY_FIELDS = [
    "space_m",
    "node_count",
    "runs",
    "avg_neighbor_count_mean",
    "candidate_count_mean",
    "runtime_mean_sec",
    "relative_iter_10_mean",
    "relative_iter_50_mean",
    "drop_iter_10_pct",
    "drop_iter_50_pct",
]


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_convergence(events: List[Dict[str, float]]) -> List[Dict[str, object]]:
    initial = None
    normalized: List[Dict[str, object]] = []
    for event in sorted(events, key=lambda row: int(row["iteration"])):
        score = float(event["best_score"])
        if initial is None and math.isfinite(score) and score > 0:
            initial = score
        relative = "" if initial is None or initial <= 0 or not math.isfinite(score) else round(score / initial, 8)
        normalized.append({**event, "relative_best_score": relative})
    return normalized


def _run_one(job: Dict[str, int]) -> List[Dict[str, object]]:
    space_m = int(job["space_m"])
    node_count = int(job["node_count"])
    seed = int(job["seed"])

    case = replace(
        SIMULATION_CASE,
        name=f"report_convergence_50_s{space_m}_n{node_count}_seed{seed}",
        node_count=node_count,
        packet_size_bits=6400,
        initial_energy=0.5,
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        pso_particles=4,
        pso_iterations=PSO_ITERATIONS,
        contention_penalty_factor=0.01,
        contention_reference_degree=10.0,
        contention_max_multiplier=4.0,
    )

    started = perf_counter()
    sim = PsoEulcSimulator(case, params, seed, verbose=False, track_pso_convergence=True)
    avg_neighbor_count = float(sim.neighbor_count.mean())
    result = sim.run(stop_on_first_dead=False, min_alive_ratio=None, max_rounds=1)
    elapsed = perf_counter() - started

    rows: List[Dict[str, object]] = []
    for event in _normalize_convergence(result.pso_convergence):
        rows.append(
            {
                "space_m": space_m,
                "node_count": node_count,
                "seed": seed,
                "iteration": int(event["iteration"]),
                "best_score": round(float(event["best_score"]), 8),
                "relative_best_score": event["relative_best_score"],
                "candidate_count": int(event["candidate_count"]),
                "avg_neighbor_count": round(avg_neighbor_count, 4),
                "runtime_sec": round(elapsed, 4),
            }
        )
    return rows


def _group_rows(rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> Dict[Tuple[object, ...], List[Dict[str, object]]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def build_summary(raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_case = _group_rows(raw_rows, ("space_m", "node_count"))
    summary_rows: List[Dict[str, object]] = []
    for key, rows in sorted(by_case.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        seeds = sorted({int(row["seed"]) for row in rows})
        neighbor_values = [float(row["avg_neighbor_count"]) for row in rows if int(row["iteration"]) == 0]
        candidate_values = [float(row["candidate_count"]) for row in rows if int(row["iteration"]) == 0]
        runtime_values = [float(row["runtime_sec"]) for row in rows if int(row["iteration"]) == 0]

        def iter_mean(iteration: int) -> float:
            values = [
                float(row["relative_best_score"])
                for row in rows
                if int(row["iteration"]) == iteration and row["relative_best_score"] != ""
            ]
            return mean(values) if values else float("nan")

        rel_10 = iter_mean(10)
        rel_50 = iter_mean(PSO_ITERATIONS)
        summary_rows.append(
            {
                "space_m": int(key[0]),
                "node_count": int(key[1]),
                "runs": len(seeds),
                "avg_neighbor_count_mean": round(mean(neighbor_values), 4),
                "candidate_count_mean": round(mean(candidate_values), 2),
                "runtime_mean_sec": round(mean(runtime_values), 4),
                "relative_iter_10_mean": round(rel_10, 4),
                "relative_iter_50_mean": round(rel_50, 4),
                "drop_iter_10_pct": round((1.0 - rel_10) * 100.0, 2),
                "drop_iter_50_pct": round((1.0 - rel_50) * 100.0, 2),
            }
        )
    _write_rows(SUMMARY_LOG, SUMMARY_FIELDS, summary_rows)
    return summary_rows


def plot(raw_rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    grouped = _group_rows(raw_rows, ("space_m", "node_count", "iteration"))
    stats: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
    for key, rows in grouped.items():
        values = [float(row["relative_best_score"]) for row in rows if row["relative_best_score"] != ""]
        if values:
            stats[(int(key[0]), int(key[1]), int(key[2]))] = (mean(values), _std(values))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, [100, 500, 1000]):
        for node_count in REPORT_NODE_SETS[space_m]:
            iterations = sorted(
                iteration
                for s, n, iteration in stats
                if s == space_m and n == node_count
            )
            means = [stats[(space_m, node_count, iteration)][0] for iteration in iterations]
            stds = [stats[(space_m, node_count, iteration)][1] for iteration in iterations]
            lower = [max(0.0, m - s) for m, s in zip(means, stds)]
            upper = [min(1.05, m + s) for m, s in zip(means, stds)]
            ax.plot(iterations, means, marker="o", markersize=2.8, linewidth=1.6, label=f"n={node_count}")
            ax.fill_between(iterations, lower, upper, alpha=0.10)

        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("PSO iteration")
        ax.set_ylabel("Best cost / initial best cost")
        ax.set_xlim(0, PSO_ITERATIONS)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower left", framealpha=0.92, fontsize=8)

    fig.suptitle(f"{ALGORITHM_LABEL} convergence for report node sets, first round, 50 iterations, 30 seeds/case")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    fig.savefig(FIGURE_PATH_50, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURE_PATH}")
    print(f"Saved {FIGURE_PATH_50}")


def main() -> None:
    jobs = [
        {"space_m": space_m, "node_count": node_count, "seed": seed}
        for space_m, node_counts in REPORT_NODE_SETS.items()
        for node_count in node_counts
        for seed in range(SEED_START, SEED_START + RUNS)
    ]
    raw_rows: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_job = {executor.submit(_run_one, job): job for job in jobs}
        for future in as_completed(future_to_job):
            rows = future.result()
            raw_rows.extend(rows)
            if rows:
                first = rows[0]
                print(
                    "Done "
                    f"space={first['space_m']}, n={first['node_count']}, seed={first['seed']}, "
                    f"candidates={first['candidate_count']}, neighbors={first['avg_neighbor_count']}"
                )

    raw_rows.sort(
        key=lambda row: (
            int(row["space_m"]),
            int(row["node_count"]),
            int(row["seed"]),
            int(row["iteration"]),
        )
    )
    _write_rows(RAW_LOG, RAW_FIELDS, raw_rows)
    build_summary(raw_rows)
    plot(raw_rows)


if __name__ == "__main__":
    main()

