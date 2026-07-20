from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

from uwsn.run_config import BASE_SEED, SIMULATION_CASE, SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


RAW_LOG = Path("outputs/data/csv/raw/s2_fault_runtime_raw.csv")
CONVERGENCE_LOG = Path("outputs/data/csv/raw/s2_pso_convergence_raw.csv")
SUMMARY_LOG = Path("outputs/data/csv/summary/s2_fault_runtime_summary.csv")
FIGURE_DIR = Path("outputs/figures/s2")

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
    "pso_particles",
    "pso_iterations",
    "seed",
    "max_rounds",
    "ft5_dead_threshold_nodes",
    "ft5_round",
    "fnd_round",
    "dead_nodes_at_stop",
    "alive_nodes_at_stop",
    "residual_energy",
    "packets_received",
    "pso_recluster_count",
    "runtime_sec",
]

CONVERGENCE_FIELDS = [
    "suite",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
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
    "pso_particles",
    "pso_iterations",
    "runs",
    "seed_start",
    "seed_end",
    "max_rounds",
    "ft5_dead_threshold_nodes",
    "ft5_mean",
    "ft5_std",
    "ft5_min",
    "ft5_max",
    "ft5_cv_pct",
    "fnd_mean",
    "fnd_std",
    "runtime_mean_sec",
    "runtime_std_sec",
    "runtime_total_sec",
    "pso_recluster_mean",
    "pso_final_improvement_mean_pct",
    "pso_final_improvement_std_pct",
]


def _append_row(path: Path, fieldnames: Sequence[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _existing_run_keys() -> set[Tuple[str, str, str]]:
    return {
        (row["space_m"], row["node_count"], row["seed"])
        for row in _load_csv(RAW_LOG)
        if row.get("suite") == "S2"
    }


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


def _runtime_tick_step(max_value: float) -> float:
    if max_value <= 5:
        return 1
    if max_value <= 20:
        return 5
    if max_value <= 100:
        return 20
    return 50


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


def _run_one_case(
    *,
    space_m: int,
    node_count: int,
    seed: int,
    packet_size_bits: int,
    initial_energy_j: float,
    max_rounds: int,
    pso_particles: int,
    pso_iterations: int,
) -> None:
    case = replace(
        SIMULATION_CASE,
        name=f"S2_s{space_m}_n{node_count}_seed{seed}",
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
    )

    started = perf_counter()
    sim = PsoEulcSimulator(case, params, seed, verbose=False, track_pso_convergence=True)
    result = sim.run(stop_on_first_dead=False, min_alive_ratio=0.95, max_rounds=max_rounds)
    elapsed = perf_counter() - started

    ft5_dead_threshold = int(math.ceil(0.05 * node_count))
    dead_nodes_at_stop = node_count - result.alive_nodes
    _append_row(
        RAW_LOG,
        RAW_FIELDS,
        {
            "suite": "S2",
            "space_m": space_m,
            "node_count": node_count,
            "packet_size_bits": packet_size_bits,
            "initial_energy_j": initial_energy_j,
            "transmission_range_m": params.transmission_range_m,
            "pso_particles": pso_particles,
            "pso_iterations": pso_iterations,
            "seed": seed,
            "max_rounds": max_rounds,
            "ft5_dead_threshold_nodes": ft5_dead_threshold,
            "ft5_round": result.first_5pct_round or "",
            "fnd_round": result.fnd_round or "",
            "dead_nodes_at_stop": dead_nodes_at_stop,
            "alive_nodes_at_stop": result.alive_nodes,
            "residual_energy": round(result.residual_energy, 6),
            "packets_received": result.packets_received,
            "pso_recluster_count": len({int(row["refresh_index"]) for row in result.pso_convergence}),
            "runtime_sec": round(elapsed, 4),
        },
    )

    convergence_rows = _normalize_convergence(result.pso_convergence)
    for event in convergence_rows:
        _append_row(
            CONVERGENCE_LOG,
            CONVERGENCE_FIELDS,
            {
                "suite": "S2",
                "space_m": space_m,
                "node_count": node_count,
                "packet_size_bits": packet_size_bits,
                "initial_energy_j": initial_energy_j,
                "seed": seed,
                "simulation_round": int(event["round"]),
                "refresh_index": int(event["refresh_index"]),
                "iteration": int(event["iteration"]),
                "best_score": round(float(event["best_score"]), 8),
                "relative_best_score": event["relative_best_score"],
                "improvement_pct": event["improvement_pct"],
                "candidate_count": int(event["candidate_count"]),
            },
        )


def _group_rows(rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def build_summary() -> List[Dict[str, object]]:
    raw_rows = [row for row in _load_csv(RAW_LOG) if row.get("suite") == "S2"]
    convergence_rows = [row for row in _load_csv(CONVERGENCE_LOG) if row.get("suite") == "S2"]
    final_improvement_by_case: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_refresh = _group_rows(
        convergence_rows,
        ("space_m", "node_count", "seed", "simulation_round", "refresh_index"),
    )
    for key, rows in by_refresh.items():
        rows_with_improvement = [row for row in rows if row.get("improvement_pct") not in ("", None)]
        if not rows_with_improvement:
            continue
        final_row = max(rows_with_improvement, key=lambda row: int(row["iteration"]))
        final_improvement_by_case[(key[0], key[1])].append(float(final_row["improvement_pct"]))

    summary_rows: List[Dict[str, object]] = []
    grouped = _group_rows(raw_rows, ("space_m", "node_count"))
    for (space_m, node_count), rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        ft5_values = [float(row["ft5_round"]) for row in rows if row.get("ft5_round")]
        fnd_values = [float(row["fnd_round"]) for row in rows if row.get("fnd_round")]
        runtime_values = [float(row["runtime_sec"]) for row in rows if row.get("runtime_sec")]
        recluster_values = [float(row["pso_recluster_count"]) for row in rows if row.get("pso_recluster_count")]
        improvements = final_improvement_by_case.get((space_m, node_count), [])
        first = rows[0]
        seeds = sorted(int(row["seed"]) for row in rows)
        summary_rows.append(
            {
                "suite": "S2",
                "space_m": int(space_m),
                "node_count": int(node_count),
                "packet_size_bits": int(first["packet_size_bits"]),
                "initial_energy_j": float(first["initial_energy_j"]),
                "transmission_range_m": float(first["transmission_range_m"]),
                "pso_particles": int(first["pso_particles"]),
                "pso_iterations": int(first["pso_iterations"]),
                "runs": len(rows),
                "seed_start": min(seeds),
                "seed_end": max(seeds),
                "max_rounds": int(first["max_rounds"]),
                "ft5_dead_threshold_nodes": int(first["ft5_dead_threshold_nodes"]),
                "ft5_mean": round(mean(ft5_values), 4) if ft5_values else "",
                "ft5_std": round(_std(ft5_values), 4) if ft5_values else "",
                "ft5_min": min(ft5_values) if ft5_values else "",
                "ft5_max": max(ft5_values) if ft5_values else "",
                "ft5_cv_pct": round(_cv_pct(ft5_values), 4) if ft5_values else "",
                "fnd_mean": round(mean(fnd_values), 4) if fnd_values else "",
                "fnd_std": round(_std(fnd_values), 4) if fnd_values else "",
                "runtime_mean_sec": round(mean(runtime_values), 4) if runtime_values else "",
                "runtime_std_sec": round(_std(runtime_values), 4) if runtime_values else "",
                "runtime_total_sec": round(sum(runtime_values), 4) if runtime_values else "",
                "pso_recluster_mean": round(mean(recluster_values), 4) if recluster_values else "",
                "pso_final_improvement_mean_pct": round(mean(improvements), 4) if improvements else "",
                "pso_final_improvement_std_pct": round(_std(improvements), 4) if improvements else "",
            }
        )
    _write_rows(SUMMARY_LOG, SUMMARY_FIELDS, summary_rows)
    return summary_rows


def _plot_bar_panels(
    rows: List[Dict[str, object]],
    *,
    value_field: str,
    std_field: str,
    ylabel: str,
    title: str,
    output_path: Path,
    tick_step: float,
    label_suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_value = max(
        float(row[value_field]) + float(row[std_field] or 0)
        for row in rows
        if row.get(value_field) not in ("", None)
    )
    y_upper = _nice_upper(max_value, tick_step)
    y_ticks = [tick_step * idx for idx in range(int(round(y_upper / tick_step)) + 1)]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, S2_CASES):
        panel = [row for row in rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        labels = [str(row["node_count"]) for row in panel]
        means = [float(row[value_field]) for row in panel]
        stds = [float(row[std_field] or 0) for row in panel]
        positions = list(range(len(labels)))
        bars = ax.bar(positions, means, yerr=stds, capsize=4, color="#4C78A8", alpha=0.84, width=0.66)
        for bar, mean_value, std_value in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean_value + std_value + y_upper * 0.025,
                f"{mean_value:.1f}{label_suffix}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_ft5(summary_rows: List[Dict[str, object]]) -> None:
    _plot_bar_panels(
        summary_rows,
        value_field="ft5_mean",
        std_field="ft5_std",
        ylabel="Round until 5% dead",
        title="S2 fault tolerance: first round with 5% dead nodes",
        output_path=FIGURE_DIR / "s2_ft5_fault_tolerance_bar.png",
        tick_step=500,
    )


def plot_runtime(summary_rows: List[Dict[str, object]]) -> None:
    max_runtime = max(float(row["runtime_mean_sec"]) + float(row["runtime_std_sec"]) for row in summary_rows)
    tick_step = _runtime_tick_step(max_runtime)
    _plot_bar_panels(
        summary_rows,
        value_field="runtime_mean_sec",
        std_field="runtime_std_sec",
        ylabel="Runtime per seed (s)",
        title="S2 runtime comparison",
        output_path=FIGURE_DIR / "s2_runtime_comparison_bar.png",
        tick_step=tick_step,
        label_suffix="s",
    )


def plot_convergence() -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in _load_csv(CONVERGENCE_LOG)
        if row.get("suite") == "S2" and row.get("relative_best_score") not in ("", None)
    ]
    if not rows:
        return

    grouped = _group_rows(rows, ("space_m", "iteration"))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = {100: "tab:blue", 500: "tab:orange", 1000: "tab:green"}
    for space_m in S2_CASES:
        iterations = sorted(
            {
                int(row["iteration"])
                for row in rows
                if int(row["space_m"]) == space_m
            }
        )
        means: List[float] = []
        stds: List[float] = []
        for iteration in iterations:
            values = [
                float(row["relative_best_score"])
                for row in grouped.get((str(space_m), str(iteration)), [])
                if row.get("relative_best_score") not in ("", None)
            ]
            means.append(mean(values))
            stds.append(_std(values))
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.plot(iterations, means, marker="o", markersize=3, linewidth=1.8, label=f"space={space_m}m", color=colors[space_m])
        ax.fill_between(iterations, lower, upper, color=colors[space_m], alpha=0.16)

    ax.set_title("S2 PSO convergence: mean relative best cost +/- std")
    ax.set_xlabel("PSO iteration")
    ax.set_ylabel("Best cost / initial best cost")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.92)
    output_path = FIGURE_DIR / "s2_pso_convergence_relative_cost.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_round_progress() -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in _load_csv(CONVERGENCE_LOG)
        if row.get("suite") == "S2" and row.get("improvement_pct") not in ("", None)
    ]
    if not rows:
        return

    by_refresh = _group_rows(rows, ("space_m", "node_count", "seed", "simulation_round", "refresh_index"))
    final_rows: List[Dict[str, str]] = []
    for refresh_rows in by_refresh.values():
        final_rows.append(max(refresh_rows, key=lambda row: int(row["iteration"])))

    bin_size = 100
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in final_rows:
        space_m = row["space_m"]
        sim_round = int(row["simulation_round"])
        round_bin = ((sim_round - 1) // bin_size) * bin_size
        grouped[(space_m, round_bin)].append(float(row["improvement_pct"]))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True, constrained_layout=True)
    colors = {100: "tab:blue", 500: "tab:orange", 1000: "tab:green"}
    for ax, space_m in zip(axes, S2_CASES):
        bins = [
            round_bin
            for (space_value, round_bin) in sorted(grouped)
            if int(space_value) == space_m and len(grouped[(space_value, round_bin)]) >= 5
        ]
        means = [mean(grouped[(str(space_m), round_bin)]) for round_bin in bins]
        stds = [_std(grouped[(str(space_m), round_bin)]) for round_bin in bins]
        ax.errorbar(
            bins,
            means,
            yerr=stds,
            marker="o",
            markersize=3,
            capsize=2,
            linewidth=1.4,
            color=colors[space_m],
        )
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel(f"Reclustering round ({bin_size}-round bins)")
        ax.set_ylabel("Final PSO improvement (%)")
        ax.set_xlim(-25, 1025)
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
        ax.set_ylim(0, 120)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("S2 PSO progress by reclustering round")
    output_path = FIGURE_DIR / "s2_pso_round_progress.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def run_all(args: argparse.Namespace) -> None:
    existing = _existing_run_keys()
    seeds = list(range(args.seed_start, args.seed_start + args.runs))
    for space_m, node_counts in S2_CASES.items():
        for node_count in node_counts:
            for seed in seeds:
                key = (str(space_m), str(node_count), str(seed))
                if key in existing and not args.force:
                    print(f"Skip existing S2 space={space_m}, n={node_count}, seed={seed}")
                    continue
                print(f"Run S2 space={space_m}, n={node_count}, seed={seed}")
                _run_one_case(
                    space_m=space_m,
                    node_count=node_count,
                    seed=seed,
                    packet_size_bits=args.packet_size_bits,
                    initial_energy_j=args.initial_energy_j,
                    max_rounds=args.max_rounds,
                    pso_particles=args.pso_particles,
                    pso_iterations=args.pso_iterations,
                )
                existing.add(key)

    summary_rows = build_summary()
    plot_ft5(summary_rows)
    plot_runtime(summary_rows)
    plot_convergence()
    plot_round_progress()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S2 fault-tolerance, PSO convergence, and runtime experiments.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=BASE_SEED + 2600)
    parser.add_argument("--max-rounds", type=int, default=5000)
    parser.add_argument("--packet-size-bits", type=int, default=6400)
    parser.add_argument("--initial-energy-j", type=float, default=0.5)
    parser.add_argument("--pso-particles", type=int, default=SIMULATION_PARAMS.pso_particles)
    parser.add_argument("--pso-iterations", type=int, default=SIMULATION_PARAMS.pso_iterations)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())

