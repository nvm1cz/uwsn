from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

from uwsn.run_config import BASE_SEED, SIMULATION_CASE, SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


RAW_LOG = Path("outputs/data/csv/raw/s2_density_convergence_runs.csv")
CONVERGENCE_LOG = Path("outputs/data/csv/raw/s2_density_convergence_events.csv")
SUMMARY_LOG = Path("outputs/data/csv/summary/s2_density_convergence_summary.csv")
FIGURE_DIR = Path("outputs/figures/s2_density_realism")

S2_CASES = {
    100: (50, 70, 90),
    500: (100, 200, 300),
    1000: (500, 700, 1000),
}

RAW_FIELDS = [
    "suite",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "contention_penalty_factor",
    "contention_reference_degree",
    "contention_max_multiplier",
    "pso_particles",
    "pso_iterations",
    "seed",
    "max_rounds",
    "ft5_round",
    "fnd_round",
    "dead_nodes_at_stop",
    "alive_nodes_at_stop",
    "recluster_count",
    "runtime_sec",
]

CONVERGENCE_FIELDS = [
    "suite",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "contention_penalty_factor",
    "pso_particles",
    "pso_iterations",
    "seed",
    "simulation_round",
    "refresh_index",
    "iteration",
    "best_score",
    "relative_best_score",
    "improvement_pct",
    "candidate_count",
]

SUMMARY_FIELDS = [
    "suite",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "contention_penalty_factor",
    "contention_reference_degree",
    "contention_max_multiplier",
    "pso_particles",
    "pso_iterations",
    "runs",
    "seed_start",
    "seed_end",
    "max_rounds",
    "fnd_mean",
    "fnd_std",
    "fnd_cv_pct",
    "ft5_mean",
    "ft5_std",
    "ft5_cv_pct",
    "final_relative_cost_mean",
    "final_relative_cost_std",
    "final_improvement_mean_pct",
    "final_improvement_std_pct",
    "recluster_mean",
    "runtime_mean_sec",
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


def _append_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _cv_pct(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return (_std(values) / avg * 100.0) if avg else 0.0


def _nice_upper(max_value: float, tick_step: float) -> float:
    if max_value <= 0:
        return tick_step
    return max(tick_step, math.ceil(max_value * 1.10 / tick_step) * tick_step)


def _run_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str]:
    return (
        row["space_m"],
        row["node_count"],
        row["seed"],
        row.get("contention_penalty_factor", ""),
        row.get("pso_particles", ""),
        row.get("pso_iterations", ""),
    )


def _existing_keys() -> set[Tuple[str, str, str, str, str, str]]:
    return {
        _run_key(row)
        for row in _load_csv(RAW_LOG)
        if row.get("suite") == "S2_DENSITY_CONVERGENCE"
    }


def _normalize_convergence(events: List[Dict[str, float]]) -> List[Dict[str, float]]:
    initial_by_refresh: Dict[Tuple[int, int], float] = {}
    for event in sorted(events, key=lambda row: (row["refresh_index"], row["iteration"])):
        key = (int(event["refresh_index"]), int(event["round"]))
        score = float(event["best_score"])
        if key not in initial_by_refresh and math.isfinite(score) and score > 0:
            initial_by_refresh[key] = score

    normalized: List[Dict[str, float]] = []
    for event in events:
        key = (int(event["refresh_index"]), int(event["round"]))
        initial = initial_by_refresh.get(key)
        score = float(event["best_score"])
        if initial is None or initial <= 0 or not math.isfinite(score):
            relative = ""
            improvement = ""
        else:
            relative_value = score / initial
            improvement_value = max(0.0, (initial - score) / initial * 100.0)
            relative = round(relative_value, 8)
            improvement = round(improvement_value, 4)
        normalized.append({**event, "relative_best_score": relative, "improvement_pct": improvement})
    return normalized


def _run_one_case(args: Dict[str, object]) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    space_m = int(args["space_m"])
    node_count = int(args["node_count"])
    seed = int(args["seed"])
    packet_size_bits = int(args["packet_size_bits"])
    initial_energy_j = float(args["initial_energy_j"])
    max_rounds = int(args["max_rounds"])
    pso_particles = int(args["pso_particles"])
    pso_iterations = int(args["pso_iterations"])
    contention_penalty_factor = float(args["contention_penalty_factor"])
    contention_reference_degree = float(args["contention_reference_degree"])
    contention_max_multiplier = float(args["contention_max_multiplier"])

    case = replace(
        SIMULATION_CASE,
        name=f"S2_density_conv_s{space_m}_n{node_count}_seed{seed}",
        node_count=node_count,
        packet_size_bits=packet_size_bits,
        initial_energy=initial_energy_j,
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        pso_particles=pso_particles,
        pso_iterations=pso_iterations,
        contention_penalty_factor=contention_penalty_factor,
        contention_reference_degree=contention_reference_degree,
        contention_max_multiplier=contention_max_multiplier,
    )

    started = perf_counter()
    sim = PsoEulcSimulator(case, params, seed, verbose=False, track_pso_convergence=True)
    result = sim.run(stop_on_first_dead=False, min_alive_ratio=0.95, max_rounds=max_rounds)
    elapsed = perf_counter() - started

    convergence_rows: List[Dict[str, object]] = []
    for event in _normalize_convergence(result.pso_convergence):
        convergence_rows.append(
            {
                "suite": "S2_DENSITY_CONVERGENCE",
                "space_m": space_m,
                "node_count": node_count,
                "packet_size_bits": packet_size_bits,
                "initial_energy_j": initial_energy_j,
                "transmission_range_m": params.transmission_range_m,
                "contention_penalty_factor": contention_penalty_factor,
                "pso_particles": pso_particles,
                "pso_iterations": pso_iterations,
                "seed": seed,
                "simulation_round": int(event["round"]),
                "refresh_index": int(event["refresh_index"]),
                "iteration": int(event["iteration"]),
                "best_score": round(float(event["best_score"]), 8),
                "relative_best_score": event["relative_best_score"],
                "improvement_pct": event["improvement_pct"],
                "candidate_count": int(event["candidate_count"]),
            }
        )

    run_row = {
        "suite": "S2_DENSITY_CONVERGENCE",
        "space_m": space_m,
        "node_count": node_count,
        "packet_size_bits": packet_size_bits,
        "initial_energy_j": initial_energy_j,
        "transmission_range_m": params.transmission_range_m,
        "contention_penalty_factor": contention_penalty_factor,
        "contention_reference_degree": contention_reference_degree,
        "contention_max_multiplier": contention_max_multiplier,
        "pso_particles": pso_particles,
        "pso_iterations": pso_iterations,
        "seed": seed,
        "max_rounds": max_rounds,
        "ft5_round": result.first_5pct_round or "",
        "fnd_round": result.fnd_round or "",
        "dead_nodes_at_stop": node_count - result.alive_nodes,
        "alive_nodes_at_stop": result.alive_nodes,
        "recluster_count": len({row["refresh_index"] for row in convergence_rows}),
        "runtime_sec": round(elapsed, 4),
    }
    return run_row, convergence_rows


def _group_rows(rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def build_summary() -> List[Dict[str, object]]:
    raw_rows = [
        row for row in _load_csv(RAW_LOG)
        if row.get("suite") == "S2_DENSITY_CONVERGENCE"
    ]
    convergence_rows = [
        row for row in _load_csv(CONVERGENCE_LOG)
        if row.get("suite") == "S2_DENSITY_CONVERGENCE"
    ]

    final_by_case: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    by_refresh = _group_rows(
        convergence_rows,
        ("space_m", "node_count", "seed", "simulation_round", "refresh_index"),
    )
    for key, rows in by_refresh.items():
        valid = [row for row in rows if row.get("relative_best_score") not in ("", None)]
        if not valid:
            continue
        final_row = max(valid, key=lambda row: int(row["iteration"]))
        final_by_case.setdefault((key[0], key[1]), []).append(
            (float(final_row["relative_best_score"]), float(final_row["improvement_pct"]))
        )

    summary_rows: List[Dict[str, object]] = []
    grouped = _group_rows(raw_rows, ("space_m", "node_count", "contention_penalty_factor", "pso_particles", "pso_iterations"))
    for _key, rows in sorted(
        grouped.items(),
        key=lambda item: (
            int(item[0][0]),
            int(item[0][1]),
            float(item[0][2]),
            int(item[0][3]),
            int(item[0][4]),
        ),
    ):
        first = rows[0]
        fnd_values = [float(row["fnd_round"]) for row in rows if row.get("fnd_round")]
        ft5_values = [float(row["ft5_round"]) for row in rows if row.get("ft5_round")]
        runtime_values = [float(row["runtime_sec"]) for row in rows if row.get("runtime_sec")]
        recluster_values = [float(row["recluster_count"]) for row in rows if row.get("recluster_count")]
        seeds = sorted(int(row["seed"]) for row in rows)
        final_pairs = final_by_case.get((first["space_m"], first["node_count"]), [])
        final_relative = [item[0] for item in final_pairs]
        final_improvement = [item[1] for item in final_pairs]
        summary_rows.append(
            {
                "suite": "S2_DENSITY_CONVERGENCE",
                "space_m": int(first["space_m"]),
                "node_count": int(first["node_count"]),
                "packet_size_bits": int(first["packet_size_bits"]),
                "initial_energy_j": float(first["initial_energy_j"]),
                "transmission_range_m": float(first["transmission_range_m"]),
                "contention_penalty_factor": float(first["contention_penalty_factor"]),
                "contention_reference_degree": float(first["contention_reference_degree"]),
                "contention_max_multiplier": float(first["contention_max_multiplier"]),
                "pso_particles": int(first["pso_particles"]),
                "pso_iterations": int(first["pso_iterations"]),
                "runs": len(rows),
                "seed_start": min(seeds),
                "seed_end": max(seeds),
                "max_rounds": int(first["max_rounds"]),
                "fnd_mean": round(mean(fnd_values), 4) if fnd_values else "",
                "fnd_std": round(_std(fnd_values), 4) if fnd_values else "",
                "fnd_cv_pct": round(_cv_pct(fnd_values), 4) if fnd_values else "",
                "ft5_mean": round(mean(ft5_values), 4) if ft5_values else "",
                "ft5_std": round(_std(ft5_values), 4) if ft5_values else "",
                "ft5_cv_pct": round(_cv_pct(ft5_values), 4) if ft5_values else "",
                "final_relative_cost_mean": round(mean(final_relative), 6) if final_relative else "",
                "final_relative_cost_std": round(_std(final_relative), 6) if final_relative else "",
                "final_improvement_mean_pct": round(mean(final_improvement), 4) if final_improvement else "",
                "final_improvement_std_pct": round(_std(final_improvement), 4) if final_improvement else "",
                "recluster_mean": round(mean(recluster_values), 4) if recluster_values else "",
                "runtime_mean_sec": round(mean(runtime_values), 4) if runtime_values else "",
            }
        )
    _write_rows(SUMMARY_LOG, SUMMARY_FIELDS, summary_rows)
    return summary_rows


def _iteration_stats() -> Dict[Tuple[str, str, int], Tuple[float, float]]:
    rows = [
        row for row in _load_csv(CONVERGENCE_LOG)
        if row.get("suite") == "S2_DENSITY_CONVERGENCE"
        and row.get("relative_best_score") not in ("", None)
    ]
    grouped = _group_rows(rows, ("space_m", "node_count", "iteration"))
    stats: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
    for key, group_rows in grouped.items():
        values = [float(row["relative_best_score"]) for row in group_rows]
        stats[(key[0], key[1], int(key[2]))] = (mean(values), _std(values))
    return stats


def plot_convergence(summary_rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    stats = _iteration_stats()
    if not stats:
        return

    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    for ax, (space_m, node_counts) in zip(axes, S2_CASES.items()):
        for color, node_count in zip(colors, node_counts):
            iterations = sorted(
                iteration
                for (space_value, node_value, iteration) in stats
                if int(space_value) == space_m and int(node_value) == node_count
            )
            means = [stats[(str(space_m), str(node_count), iteration)][0] for iteration in iterations]
            stds = [stats[(str(space_m), str(node_count), iteration)][1] for iteration in iterations]
            lower = [max(0.0, m - s) for m, s in zip(means, stds)]
            upper = [min(1.05, m + s) for m, s in zip(means, stds)]
            ax.plot(iterations, means, marker="o", markersize=3, linewidth=1.8, color=color, label=f"n={node_count}")
            ax.fill_between(iterations, lower, upper, color=color, alpha=0.16)

        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("PSO iteration")
        ax.set_ylabel("Best cost / initial best cost")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(0, int(max(int(row["pso_iterations"]) for row in summary_rows)) + 1, 2))
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower left", framealpha=0.92)

    first = summary_rows[0]
    fig.suptitle(
        "PSO convergence, mean +/- std across reclustering runs, "
        f"factor={first['contention_penalty_factor']}, "
        f"particles={first['pso_particles']}, iterations={first['pso_iterations']}, "
        f"runs={first['runs']}/case"
    )
    output_path = FIGURE_DIR / "s2_density_pso_convergence_by_node_count.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_final_improvement(summary_rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    max_value = max(
        float(row["final_improvement_mean_pct"]) + float(row["final_improvement_std_pct"] or 0)
        for row in summary_rows
        if row.get("final_improvement_mean_pct") not in ("", None)
    )
    y_upper = _nice_upper(max_value, 10)
    y_ticks = [10 * idx for idx in range(int(round(y_upper / 10)) + 1)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, S2_CASES):
        panel = [row for row in summary_rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        labels = [str(row["node_count"]) for row in panel]
        means = [float(row["final_improvement_mean_pct"]) for row in panel]
        stds = [float(row["final_improvement_std_pct"] or 0) for row in panel]
        positions = list(range(len(labels)))
        bars = ax.bar(positions, means, yerr=stds, capsize=4, color="#4C78A8", alpha=0.84, width=0.66)
        for bar, mean_value, std_value in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean_value + std_value + y_upper * 0.025,
                f"{mean_value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("Final PSO improvement per reclustering (%)")
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    first = summary_rows[0]
    fig.suptitle(
        "Final PSO improvement mean +/- std across reclustering runs, "
        f"factor={first['contention_penalty_factor']}, runs={first['runs']}/case"
    )
    output_path = FIGURE_DIR / "s2_density_pso_final_improvement_std_bar.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_ft5_cv(summary_rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    max_value = max(
        float(row["ft5_cv_pct"])
        for row in summary_rows
        if row.get("ft5_cv_pct") not in ("", None)
    )
    y_upper = _nice_upper(max_value, 2.5)
    y_ticks = [2.5 * idx for idx in range(int(round(y_upper / 2.5)) + 1)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, S2_CASES):
        panel = [row for row in summary_rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        labels = [str(row["node_count"]) for row in panel]
        cv_values = [float(row["ft5_cv_pct"]) for row in panel]
        std_values = [float(row["ft5_std"]) for row in panel]
        positions = list(range(len(labels)))
        bars = ax.bar(positions, cv_values, color="#4C78A8", alpha=0.84, width=0.66)
        for bar, cv_value, std_value in zip(bars, cv_values, std_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                cv_value + y_upper * 0.025,
                f"CV={cv_value:.1f}%\nstd={std_value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("FT5 coefficient of variation (%)")
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    first = summary_rows[0]
    fig.suptitle(
        "FT5 stability across seeds under density/contention model, "
        f"factor={first['contention_penalty_factor']}, runs={first['runs']}/case"
    )
    output_path = FIGURE_DIR / "s2_density_ft5_cv_std_bar.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def check_monotonic_mean() -> List[str]:
    stats = _iteration_stats()
    issues: List[str] = []
    for space_m, node_counts in S2_CASES.items():
        for node_count in node_counts:
            iterations = sorted(
                iteration
                for (space_value, node_value, iteration) in stats
                if int(space_value) == space_m and int(node_value) == node_count
            )
            means = [stats[(str(space_m), str(node_count), iteration)][0] for iteration in iterations]
            for prev_iteration, next_iteration, prev_mean, next_mean in zip(
                iterations,
                iterations[1:],
                means,
                means[1:],
            ):
                if next_mean > prev_mean + 1e-9:
                    issues.append(
                        f"space={space_m}, n={node_count}, iter {prev_iteration}->{next_iteration}: "
                        f"{prev_mean:.6f}->{next_mean:.6f}"
                    )
    return issues


def plot_figures(summary_rows: List[Dict[str, object]]) -> None:
    if not summary_rows:
        return
    plot_convergence(summary_rows)
    plot_final_improvement(summary_rows)
    plot_ft5_cv(summary_rows)
    issues = check_monotonic_mean()
    if issues:
        print("Mean convergence has non-monotonic points:")
        for issue in issues:
            print(issue)
    else:
        print("Mean convergence check passed: relative best cost is non-increasing for every space/node case.")


def run_all(args: argparse.Namespace) -> None:
    seeds = list(range(args.seed_start, args.seed_start + args.runs))
    existing = set() if args.force else _existing_keys()
    jobs: List[Dict[str, object]] = []
    factor_key = str(float(args.contention_penalty_factor))
    pso_particles_key = str(int(args.pso_particles))
    pso_iterations_key = str(int(args.pso_iterations))

    for space_m, node_counts in S2_CASES.items():
        for node_count in node_counts:
            for seed in seeds:
                key = (
                    str(space_m),
                    str(node_count),
                    str(seed),
                    factor_key,
                    pso_particles_key,
                    pso_iterations_key,
                )
                if key in existing:
                    print(f"Skip existing convergence S2 space={space_m}, n={node_count}, seed={seed}")
                    continue
                jobs.append(
                    {
                        "space_m": space_m,
                        "node_count": node_count,
                        "seed": seed,
                        "packet_size_bits": args.packet_size_bits,
                        "initial_energy_j": args.initial_energy_j,
                        "max_rounds": args.max_rounds,
                        "pso_particles": args.pso_particles,
                        "pso_iterations": args.pso_iterations,
                        "contention_penalty_factor": args.contention_penalty_factor,
                        "contention_reference_degree": args.contention_reference_degree,
                        "contention_max_multiplier": args.contention_max_multiplier,
                    }
                )

    if jobs:
        print(f"Running {len(jobs)} density-convergence jobs with {args.workers} workers")
        run_rows: List[Dict[str, object]] = []
        convergence_rows: List[Dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_to_job = {executor.submit(_run_one_case, job): job for job in jobs}
            for future in as_completed(future_to_job):
                run_row, event_rows = future.result()
                run_rows.append(run_row)
                convergence_rows.extend(event_rows)
                print(
                    "Done convergence S2 "
                    f"space={run_row['space_m']}, n={run_row['node_count']}, seed={run_row['seed']}, "
                    f"FND={run_row['fnd_round']}, FT5={run_row['ft5_round']}, "
                    f"events={len(event_rows)}"
                )
        run_rows.sort(key=lambda row: (int(row["space_m"]), int(row["node_count"]), int(row["seed"])))
        convergence_rows.sort(
            key=lambda row: (
                int(row["space_m"]),
                int(row["node_count"]),
                int(row["seed"]),
                int(row["refresh_index"]),
                int(row["iteration"]),
            )
        )
        _append_rows(RAW_LOG, RAW_FIELDS, run_rows)
        _append_rows(CONVERGENCE_LOG, CONVERGENCE_FIELDS, convergence_rows)

    summary_rows = build_summary()
    plot_figures(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track PSO convergence for S2 under density/contention model.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=BASE_SEED + 2600)
    parser.add_argument("--max-rounds", type=int, default=5000)
    parser.add_argument("--packet-size-bits", type=int, default=6400)
    parser.add_argument("--initial-energy-j", type=float, default=0.5)
    parser.add_argument("--pso-particles", type=int, default=4)
    parser.add_argument("--pso-iterations", type=int, default=10)
    parser.add_argument("--contention-penalty-factor", type=float, default=0.35)
    parser.add_argument("--contention-reference-degree", type=float, default=10.0)
    parser.add_argument("--contention-max-multiplier", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())

