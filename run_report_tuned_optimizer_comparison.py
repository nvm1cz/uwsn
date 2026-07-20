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


PSO_FULL_RAW = Path("outputs/data/csv/raw/report_tuned_node_set_full_runs.csv")
PSO_CONVERGENCE_RAW = Path("outputs/data/csv/raw/report_convergence_tuned_nodes_50iter.csv")
RAW_LOG = Path("outputs/data/csv/raw/report_tuned_optimizer_full_runs.csv")
CONVERGENCE_LOG = Path("outputs/data/csv/raw/report_tuned_optimizer_convergence_50iter.csv")
SUMMARY_LOG = Path("outputs/data/csv/summary/report_tuned_optimizer_summary.csv")
FIGURE_DIR = Path("outputs/figures/report_tuned_optimizer_comparison")

NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
    1000: (500, 1000, 1500, 2000),
}
OPTIMIZERS = ("pso", "ga", "de")
RUNS = 30
FULL_SEED_START = 4000
CONVERGENCE_SEED_START = 9500
MAX_ROUNDS = 5000
FULL_ITERATIONS = 10
CONVERGENCE_ITERATIONS = 50
POPULATION_SIZE = 4

RAW_FIELDS = [
    "suite",
    "optimizer",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "contention_penalty_factor",
    "population_size",
    "iterations",
    "seed",
    "max_rounds",
    "ft5_round",
    "fnd_round",
    "dead_nodes_at_stop",
    "alive_nodes_at_stop",
    "avg_neighbor_count",
    "runtime_sec",
]

CONVERGENCE_FIELDS = [
    "suite",
    "optimizer",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "contention_penalty_factor",
    "population_size",
    "iterations",
    "seed",
    "iteration",
    "best_score",
    "relative_best_score",
    "improvement_pct",
    "candidate_count",
]

SUMMARY_FIELDS = [
    "suite",
    "optimizer",
    "space_m",
    "node_count",
    "runs",
    "ft5_mean",
    "ft5_std",
    "ft5_cv_pct",
    "fnd_mean",
    "fnd_std",
    "runtime_mean_sec",
    "runtime_std_sec",
    "avg_neighbor_count_mean",
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


def _cv_pct(values: Sequence[float]) -> float:
    avg = mean(values) if values else 0.0
    return (_std(values) / avg * 100.0) if avg else 0.0


def _nice_upper(max_value: float, tick_step: float) -> float:
    if max_value <= 0:
        return tick_step
    return max(tick_step, math.ceil(max_value * 1.10 / tick_step) * tick_step)


def _runtime_tick_step(max_value: float) -> float:
    if max_value <= 10:
        return 2
    if max_value <= 50:
        return 10
    if max_value <= 200:
        return 50
    return 100


def _is_case(space_m: int, node_count: int) -> bool:
    return space_m in NODE_SETS and node_count in NODE_SETS[space_m]


def _key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (str(row["optimizer"]), str(row["space_m"]), str(row["node_count"]), str(row["seed"]))


def _normalize_pso_full_row(row: Dict[str, str]) -> Dict[str, object] | None:
    space_m = int(row["space_m"])
    node_count = int(row["node_count"])
    seed = int(row["seed"])
    if not _is_case(space_m, node_count) or seed not in range(FULL_SEED_START, FULL_SEED_START + RUNS):
        return None
    return {
        "suite": "REPORT_TUNED_OPT",
        "optimizer": "pso",
        "space_m": space_m,
        "node_count": node_count,
        "packet_size_bits": row["packet_size_bits"],
        "initial_energy_j": row["initial_energy_j"],
        "transmission_range_m": row["transmission_range_m"],
        "contention_penalty_factor": row["contention_penalty_factor"],
        "population_size": POPULATION_SIZE,
        "iterations": FULL_ITERATIONS,
        "seed": seed,
        "max_rounds": row["max_rounds"],
        "ft5_round": row["ft5_round"],
        "fnd_round": row["fnd_round"],
        "dead_nodes_at_stop": row["dead_nodes_at_stop"],
        "alive_nodes_at_stop": row["alive_nodes_at_stop"],
        "avg_neighbor_count": row["avg_neighbor_count"],
        "runtime_sec": row["runtime_sec"],
    }


def _normalize_pso_convergence_row(row: Dict[str, str]) -> Dict[str, object] | None:
    space_m = int(row["space_m"])
    node_count = int(row["node_count"])
    seed = int(row["seed"])
    if not _is_case(space_m, node_count) or seed not in range(CONVERGENCE_SEED_START, CONVERGENCE_SEED_START + RUNS):
        return None
    return {
        "suite": "REPORT_TUNED_OPT",
        "optimizer": "pso",
        "space_m": space_m,
        "node_count": node_count,
        "packet_size_bits": row.get("packet_size_bits", 6400),
        "initial_energy_j": row.get("initial_energy_j", 0.5),
        "transmission_range_m": row.get("transmission_range_m", 150.0),
        "contention_penalty_factor": row.get("contention_penalty_factor", 0.01),
        "population_size": POPULATION_SIZE,
        "iterations": CONVERGENCE_ITERATIONS,
        "seed": seed,
        "iteration": row["iteration"],
        "best_score": row["best_score"],
        "relative_best_score": row["relative_best_score"],
        "improvement_pct": row.get("improvement_pct", ""),
        "candidate_count": row["candidate_count"],
    }


def _normalize_events(events: List[Dict[str, float]]) -> List[Dict[str, object]]:
    initial = None
    normalized: List[Dict[str, object]] = []
    for event in sorted(events, key=lambda item: int(item["iteration"])):
        score = float(event["best_score"])
        if initial is None and math.isfinite(score) and score > 0:
            initial = score
        if initial is None or initial <= 0 or not math.isfinite(score):
            relative = ""
            improvement = ""
        else:
            relative_value = score / initial
            relative = round(relative_value, 8)
            improvement = round(max(0.0, (1.0 - relative_value) * 100.0), 4)
        normalized.append({**event, "relative_best_score": relative, "improvement_pct": improvement})
    return normalized


def _run_full(job: Dict[str, object]) -> Dict[str, object]:
    optimizer = str(job["optimizer"])
    space_m = int(job["space_m"])
    node_count = int(job["node_count"])
    seed = int(job["seed"])
    case = replace(
        SIMULATION_CASE,
        name=f"report_tuned_{optimizer}_s{space_m}_n{node_count}_seed{seed}",
        node_count=node_count,
        packet_size_bits=6400,
        initial_energy=0.5,
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        pso_particles=POPULATION_SIZE,
        pso_iterations=FULL_ITERATIONS,
        optimizer=optimizer,
        contention_penalty_factor=0.01,
        contention_reference_degree=10.0,
        contention_max_multiplier=4.0,
    )
    started = perf_counter()
    sim = PsoEulcSimulator(case, params, seed, verbose=False, track_pso_convergence=False)
    avg_neighbor_count = float(sim.neighbor_count.mean())
    result = sim.run(stop_on_first_dead=False, min_alive_ratio=0.95, max_rounds=MAX_ROUNDS)
    elapsed = perf_counter() - started
    return {
        "suite": "REPORT_TUNED_OPT",
        "optimizer": optimizer,
        "space_m": space_m,
        "node_count": node_count,
        "packet_size_bits": case.packet_size_bits,
        "initial_energy_j": case.initial_energy,
        "transmission_range_m": params.transmission_range_m,
        "contention_penalty_factor": params.contention_penalty_factor,
        "population_size": POPULATION_SIZE,
        "iterations": FULL_ITERATIONS,
        "seed": seed,
        "max_rounds": MAX_ROUNDS,
        "ft5_round": result.first_5pct_round or "",
        "fnd_round": result.fnd_round or "",
        "dead_nodes_at_stop": node_count - result.alive_nodes,
        "alive_nodes_at_stop": result.alive_nodes,
        "avg_neighbor_count": round(avg_neighbor_count, 4),
        "runtime_sec": round(elapsed, 4),
    }


def _run_convergence(job: Dict[str, object]) -> List[Dict[str, object]]:
    optimizer = str(job["optimizer"])
    space_m = int(job["space_m"])
    node_count = int(job["node_count"])
    seed = int(job["seed"])
    case = replace(
        SIMULATION_CASE,
        name=f"report_tuned_conv_{optimizer}_s{space_m}_n{node_count}_seed{seed}",
        node_count=node_count,
        packet_size_bits=6400,
        initial_energy=0.5,
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        pso_particles=POPULATION_SIZE,
        pso_iterations=CONVERGENCE_ITERATIONS,
        optimizer=optimizer,
        contention_penalty_factor=0.01,
        contention_reference_degree=10.0,
        contention_max_multiplier=4.0,
    )
    sim = PsoEulcSimulator(case, params, seed, verbose=False, track_pso_convergence=True)
    result = sim.run(stop_on_first_dead=False, min_alive_ratio=None, max_rounds=1)
    rows: List[Dict[str, object]] = []
    for event in _normalize_events(result.pso_convergence):
        rows.append(
            {
                "suite": "REPORT_TUNED_OPT",
                "optimizer": optimizer,
                "space_m": space_m,
                "node_count": node_count,
                "packet_size_bits": case.packet_size_bits,
                "initial_energy_j": case.initial_energy,
                "transmission_range_m": params.transmission_range_m,
                "contention_penalty_factor": params.contention_penalty_factor,
                "population_size": POPULATION_SIZE,
                "iterations": CONVERGENCE_ITERATIONS,
                "seed": seed,
                "iteration": int(event["iteration"]),
                "best_score": round(float(event["best_score"]), 8),
                "relative_best_score": event["relative_best_score"],
                "improvement_pct": event["improvement_pct"],
                "candidate_count": int(event["candidate_count"]),
            }
        )
    return rows


def load_full_rows() -> Dict[Tuple[str, str, str, str], Dict[str, object]]:
    rows_by_key: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for row in _load_csv(PSO_FULL_RAW):
        normalized = _normalize_pso_full_row(row)
        if normalized is not None:
            rows_by_key[_key(normalized)] = normalized
    for row in _load_csv(RAW_LOG):
        if row.get("suite") == "REPORT_TUNED_OPT":
            rows_by_key[_key(row)] = row
    return rows_by_key


def load_convergence_rows() -> Dict[Tuple[str, str, str, str, str], Dict[str, object]]:
    rows_by_key: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}
    for row in _load_csv(PSO_CONVERGENCE_RAW):
        normalized = _normalize_pso_convergence_row(row)
        if normalized is not None:
            key = (
                str(normalized["optimizer"]),
                str(normalized["space_m"]),
                str(normalized["node_count"]),
                str(normalized["seed"]),
                str(normalized["iteration"]),
            )
            rows_by_key[key] = normalized
    for row in _load_csv(CONVERGENCE_LOG):
        if row.get("suite") == "REPORT_TUNED_OPT":
            key = (row["optimizer"], row["space_m"], row["node_count"], row["seed"], row["iteration"])
            rows_by_key[key] = row
    return rows_by_key


def run_missing_full(rows_by_key: Dict[Tuple[str, str, str, str], Dict[str, object]]) -> None:
    jobs: List[Dict[str, object]] = []
    for optimizer in ("ga", "de"):
        for space_m, node_counts in NODE_SETS.items():
            for node_count in node_counts:
                for seed in range(FULL_SEED_START, FULL_SEED_START + RUNS):
                    key = (optimizer, str(space_m), str(node_count), str(seed))
                    if key not in rows_by_key:
                        jobs.append({"optimizer": optimizer, "space_m": space_m, "node_count": node_count, "seed": seed})
    if not jobs:
        return

    print(f"Running {len(jobs)} missing GA/DE FT5-runtime jobs", flush=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_job = {executor.submit(_run_full, job): job for job in jobs}
        for future in as_completed(future_to_job):
            row = future.result()
            rows_by_key[_key(row)] = row
            print(
                "Done opt full "
                f"{row['optimizer']} space={row['space_m']}, n={row['node_count']}, seed={row['seed']}, "
                f"FT5={row['ft5_round']}, runtime={row['runtime_sec']}s",
                flush=True,
            )
            _write_rows(RAW_LOG, RAW_FIELDS, sorted(rows_by_key.values(), key=lambda item: _key(item)))


def run_missing_convergence(rows_by_key: Dict[Tuple[str, str, str, str, str], Dict[str, object]]) -> None:
    jobs: List[Dict[str, object]] = []
    for optimizer in ("ga", "de"):
        for space_m, node_counts in NODE_SETS.items():
            for node_count in node_counts:
                for seed in range(CONVERGENCE_SEED_START, CONVERGENCE_SEED_START + RUNS):
                    prefix = (optimizer, str(space_m), str(node_count), str(seed))
                    if not any(key[:4] == prefix for key in rows_by_key):
                        jobs.append({"optimizer": optimizer, "space_m": space_m, "node_count": node_count, "seed": seed})
    if not jobs:
        return

    print(f"Running {len(jobs)} missing GA/DE convergence jobs", flush=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_job = {executor.submit(_run_convergence, job): job for job in jobs}
        for future in as_completed(future_to_job):
            rows = future.result()
            for row in rows:
                key = (row["optimizer"], str(row["space_m"]), str(row["node_count"]), str(row["seed"]), str(row["iteration"]))
                rows_by_key[key] = row
            if rows:
                first = rows[0]
                print(
                    "Done opt convergence "
                    f"{first['optimizer']} space={first['space_m']}, n={first['node_count']}, seed={first['seed']}",
                    flush=True,
                )
            _write_rows(CONVERGENCE_LOG, CONVERGENCE_FIELDS, sorted(rows_by_key.values(), key=lambda item: (
                str(item["optimizer"]), int(item["space_m"]), int(item["node_count"]), int(item["seed"]), int(item["iteration"])
            )))


def build_summary(full_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int], List[Dict[str, object]]] = {}
    for row in full_rows:
        grouped.setdefault((str(row["optimizer"]), int(row["space_m"]), int(row["node_count"])), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for optimizer in OPTIMIZERS:
        for space_m, node_counts in NODE_SETS.items():
            for node_count in node_counts:
                rows = grouped.get((optimizer, space_m, node_count), [])
                if not rows:
                    continue
                ft5_values = [float(row["ft5_round"]) for row in rows if row.get("ft5_round")]
                fnd_values = [float(row["fnd_round"]) for row in rows if row.get("fnd_round")]
                runtime_values = [float(row["runtime_sec"]) for row in rows if row.get("runtime_sec")]
                neighbor_values = [float(row["avg_neighbor_count"]) for row in rows if row.get("avg_neighbor_count")]
                summary_rows.append(
                    {
                        "suite": "REPORT_TUNED_OPT",
                        "optimizer": optimizer,
                        "space_m": space_m,
                        "node_count": node_count,
                        "runs": len(rows),
                        "ft5_mean": round(mean(ft5_values), 4) if ft5_values else "",
                        "ft5_std": round(_std(ft5_values), 4) if ft5_values else "",
                        "ft5_cv_pct": round(_cv_pct(ft5_values), 4) if ft5_values else "",
                        "fnd_mean": round(mean(fnd_values), 4) if fnd_values else "",
                        "fnd_std": round(_std(fnd_values), 4) if fnd_values else "",
                        "runtime_mean_sec": round(mean(runtime_values), 4) if runtime_values else "",
                        "runtime_std_sec": round(_std(runtime_values), 4) if runtime_values else "",
                        "avg_neighbor_count_mean": round(mean(neighbor_values), 4) if neighbor_values else "",
                    }
                )
    _write_rows(SUMMARY_LOG, SUMMARY_FIELDS, summary_rows)
    return summary_rows


def _setup_node_axis(ax, space_m: int, y_upper: float, y_step: float, ylabel: str) -> None:
    ticks = list(NODE_SETS[space_m])
    pad = max(5, int((max(ticks) - min(ticks)) * 0.10))
    ax.set_title(f"space={space_m}m")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(ticks) - pad, max(ticks) + pad)
    ax.set_xticks(ticks)
    ax.set_ylim(0, y_upper)
    ax.set_yticks([y_step * idx for idx in range(int(round(y_upper / y_step)) + 1)])
    ax.tick_params(axis="y", labelleft=True)
    ax.grid(True, linestyle="--", alpha=0.35)


def plot_metric(summary_rows: List[Dict[str, object]], metric: str, std_metric: str, ylabel: str, title: str, filename: str) -> None:
    import matplotlib.pyplot as plt

    max_value = max(
        float(row[metric]) + float(row[std_metric] or 0)
        for row in summary_rows
        if row.get(metric) not in ("", None)
    )
    step = 250 if metric == "ft5_mean" else _runtime_tick_step(max_value)
    y_upper = _nice_upper(max_value, step)
    colors = {"pso": "#4C78A8", "ga": "#E45756", "de": "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, [100, 500, 1000]):
        for optimizer in OPTIMIZERS:
            panel = [
                row for row in summary_rows
                if row["optimizer"] == optimizer and int(row["space_m"]) == space_m
            ]
            panel.sort(key=lambda item: int(item["node_count"]))
            xs = [int(row["node_count"]) for row in panel]
            means = [float(row[metric]) for row in panel]
            stds = [float(row[std_metric] or 0) for row in panel]
            ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2.0, capsize=3, label=optimizer.upper(), color=colors[optimizer])
        _setup_node_axis(ax, space_m, y_upper, step, ylabel)
        ax.legend(loc="best", framealpha=0.92)
    fig.suptitle(title)
    output_path = FIGURE_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_convergence(convergence_rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in convergence_rows if row.get("relative_best_score") not in ("", None)]
    grouped: Dict[Tuple[str, int, int], List[float]] = {}
    for row in rows:
        grouped.setdefault((str(row["optimizer"]), int(row["space_m"]), int(row["iteration"])), []).append(float(row["relative_best_score"]))

    colors = {"pso": "#4C78A8", "ga": "#E45756", "de": "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, [100, 500, 1000]):
        for optimizer in OPTIMIZERS:
            iterations = sorted(iteration for opt, space, iteration in grouped if opt == optimizer and space == space_m)
            means = [mean(grouped[(optimizer, space_m, iteration)]) for iteration in iterations]
            stds = [_std(grouped[(optimizer, space_m, iteration)]) for iteration in iterations]
            lower = [max(0.0, m - s) for m, s in zip(means, stds)]
            upper = [min(1.1, m + s) for m, s in zip(means, stds)]
            ax.plot(iterations, means, marker="o", markersize=2.8, linewidth=1.8, label=optimizer.upper(), color=colors[optimizer])
            ax.fill_between(iterations, lower, upper, alpha=0.12, color=colors[optimizer])
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best cost / initial best cost")
        ax.set_xlim(0, CONVERGENCE_ITERATIONS)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", framealpha=0.92)
    fig.suptitle("Optimizer convergence on tuned node sets, 50 iterations, 30 seeds/case")
    output_path = FIGURE_DIR / "report_tuned_optimizer_convergence_line.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_all(summary_rows: List[Dict[str, object]], convergence_rows: List[Dict[str, object]]) -> None:
    plot_metric(
        summary_rows,
        "ft5_mean",
        "ft5_std",
        "FT5 round",
        "Optimizer comparison: FT5 across tuned PSO cases, 30 seeds/case",
        "report_tuned_optimizer_ft5_line.png",
    )
    plot_metric(
        summary_rows,
        "runtime_mean_sec",
        "runtime_std_sec",
        "Runtime per seed (s)",
        "Optimizer comparison: runtime across tuned PSO cases, 30 seeds/case",
        "report_tuned_optimizer_runtime_line.png",
    )
    plot_convergence(convergence_rows)


def main() -> None:
    full_by_key = load_full_rows()
    run_missing_full(full_by_key)
    full_rows = sorted(full_by_key.values(), key=lambda item: (
        str(item["optimizer"]), int(item["space_m"]), int(item["node_count"]), int(item["seed"])
    ))
    _write_rows(RAW_LOG, RAW_FIELDS, full_rows)

    convergence_by_key = load_convergence_rows()
    run_missing_convergence(convergence_by_key)
    convergence_rows = sorted(convergence_by_key.values(), key=lambda item: (
        str(item["optimizer"]), int(item["space_m"]), int(item["node_count"]), int(item["seed"]), int(item["iteration"])
    ))
    _write_rows(CONVERGENCE_LOG, CONVERGENCE_FIELDS, convergence_rows)

    summary_rows = build_summary(full_rows)
    plot_all(summary_rows, convergence_rows)


if __name__ == "__main__":
    main()

