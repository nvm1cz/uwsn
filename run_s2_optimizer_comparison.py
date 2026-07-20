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


PSO_RAW = Path("outputs/data/csv/raw/s2_fault_runtime_raw.csv")
PSO_CONVERGENCE = Path("outputs/data/csv/raw/s2_pso_convergence_raw.csv")
RAW_LOG = Path("outputs/data/csv/raw/s2_optimizer_comparison_raw.csv")
CONVERGENCE_LOG = Path("outputs/data/csv/raw/s2_optimizer_convergence_raw.csv")
CLEAN_RAW_LOG = Path("outputs/data/csv/raw/s2_optimizer_comparison_raw_deduped.csv")
CLEAN_CONVERGENCE_LOG = Path("outputs/data/csv/raw/s2_optimizer_convergence_raw_deduped.csv")
SUMMARY_LOG = Path("outputs/data/csv/summary/s2_optimizer_comparison_summary.csv")
FIGURE_DIR = Path("outputs/figures/s2_optimizer_comparison")

S2_CASES = {
    100: (50, 70, 90),
    500: (100, 200, 300),
    1000: (500, 700, 1000),
}
OPTIMIZERS = ("pso", "ga", "de")

RAW_FIELDS = [
    "suite",
    "optimizer",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "population_size",
    "iterations",
    "seed",
    "max_rounds",
    "ft5_dead_threshold_nodes",
    "ft5_round",
    "fnd_round",
    "dead_nodes_at_stop",
    "alive_nodes_at_stop",
    "residual_energy",
    "packets_received",
    "recluster_count",
    "runtime_sec",
]

CONVERGENCE_FIELDS = [
    "suite",
    "optimizer",
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
    "optimizer",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "population_size",
    "iterations",
    "runs",
    "seed_start",
    "seed_end",
    "max_rounds",
    "ft5_dead_threshold_nodes",
    "ft5_mean",
    "ft5_std",
    "ft5_cv_pct",
    "fnd_mean",
    "fnd_std",
    "runtime_mean_sec",
    "runtime_std_sec",
    "runtime_total_sec",
    "recluster_mean",
    "final_improvement_mean_pct",
    "final_improvement_std_pct",
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
    return 100


def _group_rows(rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def _dedupe_rows(rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> List[Dict[str, str]]:
    deduped: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        deduped[tuple(row[field] for field in fields)] = row
    return list(deduped.values())


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


def _comparison_existing_keys() -> set[Tuple[str, str, str, str]]:
    keys = {
        (row["optimizer"], row["space_m"], row["node_count"], row["seed"])
        for row in _load_csv(RAW_LOG)
        if row.get("suite") == "S2_OPT"
    }
    for row in _load_csv(PSO_RAW):
        if row.get("suite") == "S2":
            keys.add(("pso", row["space_m"], row["node_count"], row["seed"]))
    return keys


def _run_one_case(
    *,
    optimizer: str,
    space_m: int,
    node_count: int,
    seed: int,
    packet_size_bits: int,
    initial_energy_j: float,
    max_rounds: int,
    population_size: int,
    iterations: int,
) -> None:
    case = replace(
        SIMULATION_CASE,
        name=f"S2_OPT_{optimizer}_s{space_m}_n{node_count}_seed{seed}",
        node_count=node_count,
        packet_size_bits=packet_size_bits,
        initial_energy=initial_energy_j,
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        pso_particles=population_size,
        pso_iterations=iterations,
        optimizer=optimizer,
    )

    started = perf_counter()
    sim = PsoEulcSimulator(case, params, seed, verbose=False, track_pso_convergence=True)
    result = sim.run(stop_on_first_dead=False, min_alive_ratio=0.95, max_rounds=max_rounds)
    elapsed = perf_counter() - started

    ft5_dead_threshold = int(math.ceil(0.05 * node_count))
    _append_row(
        RAW_LOG,
        RAW_FIELDS,
        {
            "suite": "S2_OPT",
            "optimizer": optimizer,
            "space_m": space_m,
            "node_count": node_count,
            "packet_size_bits": packet_size_bits,
            "initial_energy_j": initial_energy_j,
            "transmission_range_m": params.transmission_range_m,
            "population_size": population_size,
            "iterations": iterations,
            "seed": seed,
            "max_rounds": max_rounds,
            "ft5_dead_threshold_nodes": ft5_dead_threshold,
            "ft5_round": result.first_5pct_round or "",
            "fnd_round": result.fnd_round or "",
            "dead_nodes_at_stop": node_count - result.alive_nodes,
            "alive_nodes_at_stop": result.alive_nodes,
            "residual_energy": round(result.residual_energy, 6),
            "packets_received": result.packets_received,
            "recluster_count": len({int(row["refresh_index"]) for row in result.pso_convergence}),
            "runtime_sec": round(elapsed, 4),
        },
    )

    for event in _normalize_convergence(result.pso_convergence):
        _append_row(
            CONVERGENCE_LOG,
            CONVERGENCE_FIELDS,
            {
                "suite": "S2_OPT",
                "optimizer": optimizer,
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


def _baseline_pso_raw_rows() -> List[Dict[str, str]]:
    rows = []
    for row in _load_csv(PSO_RAW):
        if row.get("suite") != "S2":
            continue
        rows.append(
            {
                "suite": "S2_OPT",
                "optimizer": "pso",
                "space_m": row["space_m"],
                "node_count": row["node_count"],
                "packet_size_bits": row["packet_size_bits"],
                "initial_energy_j": row["initial_energy_j"],
                "transmission_range_m": row["transmission_range_m"],
                "population_size": row["pso_particles"],
                "iterations": row["pso_iterations"],
                "seed": row["seed"],
                "max_rounds": row["max_rounds"],
                "ft5_dead_threshold_nodes": row["ft5_dead_threshold_nodes"],
                "ft5_round": row["ft5_round"],
                "fnd_round": row["fnd_round"],
                "dead_nodes_at_stop": row["dead_nodes_at_stop"],
                "alive_nodes_at_stop": row["alive_nodes_at_stop"],
                "residual_energy": row["residual_energy"],
                "packets_received": row["packets_received"],
                "recluster_count": row["pso_recluster_count"],
                "runtime_sec": row["runtime_sec"],
            }
        )
    return rows


def _baseline_pso_convergence_rows() -> List[Dict[str, str]]:
    rows = []
    for row in _load_csv(PSO_CONVERGENCE):
        if row.get("suite") != "S2":
            continue
        rows.append(
            {
                "suite": "S2_OPT",
                "optimizer": "pso",
                "space_m": row["space_m"],
                "node_count": row["node_count"],
                "packet_size_bits": row["packet_size_bits"],
                "initial_energy_j": row["initial_energy_j"],
                "seed": row["seed"],
                "simulation_round": row["simulation_round"],
                "refresh_index": row["refresh_index"],
                "iteration": row["iteration"],
                "best_score": row["best_score"],
                "relative_best_score": row["relative_best_score"],
                "improvement_pct": row["improvement_pct"],
                "candidate_count": row["candidate_count"],
            }
        )
    return rows


def _all_raw_rows() -> List[Dict[str, str]]:
    return _dedupe_rows(
        _baseline_pso_raw_rows() + [row for row in _load_csv(RAW_LOG) if row.get("suite") == "S2_OPT"],
        ("optimizer", "space_m", "node_count", "seed"),
    )


def _all_convergence_rows() -> List[Dict[str, str]]:
    return _dedupe_rows(
        _baseline_pso_convergence_rows()
        + [row for row in _load_csv(CONVERGENCE_LOG) if row.get("suite") == "S2_OPT"],
        ("optimizer", "space_m", "node_count", "seed", "simulation_round", "refresh_index", "iteration"),
    )


def build_summary() -> List[Dict[str, object]]:
    raw_rows = _all_raw_rows()
    convergence_rows = _all_convergence_rows()
    _write_rows(CLEAN_RAW_LOG, RAW_FIELDS, raw_rows)
    _write_rows(CLEAN_CONVERGENCE_LOG, CONVERGENCE_FIELDS, convergence_rows)
    final_improvement_by_case: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    by_refresh = _group_rows(
        convergence_rows,
        ("optimizer", "space_m", "node_count", "seed", "simulation_round", "refresh_index"),
    )
    for key, rows in by_refresh.items():
        valid = [row for row in rows if row.get("improvement_pct") not in ("", None)]
        if not valid:
            continue
        final_row = max(valid, key=lambda row: int(row["iteration"]))
        final_improvement_by_case[(key[0], key[1], key[2])].append(float(final_row["improvement_pct"]))

    summary_rows: List[Dict[str, object]] = []
    grouped = _group_rows(raw_rows, ("optimizer", "space_m", "node_count"))
    for (optimizer, space_m, node_count), rows in sorted(
        grouped.items(),
        key=lambda item: (OPTIMIZERS.index(item[0][0]), int(item[0][1]), int(item[0][2])),
    ):
        ft5_values = [float(row["ft5_round"]) for row in rows if row.get("ft5_round")]
        fnd_values = [float(row["fnd_round"]) for row in rows if row.get("fnd_round")]
        runtime_values = [float(row["runtime_sec"]) for row in rows if row.get("runtime_sec")]
        recluster_values = [float(row["recluster_count"]) for row in rows if row.get("recluster_count")]
        improvements = final_improvement_by_case.get((optimizer, space_m, node_count), [])
        seeds = sorted(int(row["seed"]) for row in rows)
        first = rows[0]
        summary_rows.append(
            {
                "suite": "S2_OPT",
                "optimizer": optimizer,
                "space_m": int(space_m),
                "node_count": int(node_count),
                "packet_size_bits": int(first["packet_size_bits"]),
                "initial_energy_j": float(first["initial_energy_j"]),
                "transmission_range_m": float(first["transmission_range_m"]),
                "population_size": int(first["population_size"]),
                "iterations": int(first["iterations"]),
                "runs": len(rows),
                "seed_start": min(seeds),
                "seed_end": max(seeds),
                "max_rounds": int(first["max_rounds"]),
                "ft5_dead_threshold_nodes": int(first["ft5_dead_threshold_nodes"]),
                "ft5_mean": round(mean(ft5_values), 4) if ft5_values else "",
                "ft5_std": round(_std(ft5_values), 4) if ft5_values else "",
                "ft5_cv_pct": round(_cv_pct(ft5_values), 4) if ft5_values else "",
                "fnd_mean": round(mean(fnd_values), 4) if fnd_values else "",
                "fnd_std": round(_std(fnd_values), 4) if fnd_values else "",
                "runtime_mean_sec": round(mean(runtime_values), 4) if runtime_values else "",
                "runtime_std_sec": round(_std(runtime_values), 4) if runtime_values else "",
                "runtime_total_sec": round(sum(runtime_values), 4) if runtime_values else "",
                "recluster_mean": round(mean(recluster_values), 4) if recluster_values else "",
                "final_improvement_mean_pct": round(mean(improvements), 4) if improvements else "",
                "final_improvement_std_pct": round(_std(improvements), 4) if improvements else "",
            }
        )
    _write_rows(SUMMARY_LOG, SUMMARY_FIELDS, summary_rows)
    return summary_rows


def _plot_grouped_optimizer_bars(
    rows: List[Dict[str, object]],
    *,
    value_field: str,
    std_field: str,
    ylabel: str,
    title: str,
    output_path: Path,
    tick_step: float,
    label_format: str,
) -> None:
    import matplotlib.pyplot as plt

    max_value = max(
        float(row[value_field]) + float(row[std_field] or 0)
        for row in rows
        if row.get(value_field) not in ("", None)
    )
    y_upper = _nice_upper(max_value, tick_step)
    y_ticks = [tick_step * idx for idx in range(int(round(y_upper / tick_step)) + 1)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), sharey=True, constrained_layout=True)
    colors = {"pso": "#4C78A8", "ga": "#E45756", "de": "#54A24B"}
    for ax, space_m in zip(axes, S2_CASES):
        labels = [str(node_count) for node_count in S2_CASES[space_m]]
        x_positions = list(range(len(labels)))
        width = 0.22
        for opt_idx, optimizer in enumerate(OPTIMIZERS):
            means = []
            stds = []
            for node_count in S2_CASES[space_m]:
                matches = [
                    row
                    for row in rows
                    if row["optimizer"] == optimizer
                    and int(row["space_m"]) == space_m
                    and int(row["node_count"]) == node_count
                ]
                if matches:
                    means.append(float(matches[0][value_field]))
                    stds.append(float(matches[0][std_field] or 0))
                else:
                    means.append(0.0)
                    stds.append(0.0)
            offsets = [x + (opt_idx - 1) * width for x in x_positions]
            bars = ax.bar(
                offsets,
                means,
                yerr=stds,
                capsize=3,
                color=colors[optimizer],
                alpha=0.84,
                width=width,
                label=optimizer.upper(),
            )
            for bar, value, std_value in zip(bars, means, stds):
                if value <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + std_value + y_upper * 0.015,
                    label_format.format(value),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions, labels)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="lower left", framealpha=0.92)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_ft5(summary_rows: List[Dict[str, object]]) -> None:
    _plot_grouped_optimizer_bars(
        summary_rows,
        value_field="ft5_mean",
        std_field="ft5_std",
        ylabel="Round until 5% dead",
        title="S2 optimizer comparison: FT5 fault tolerance",
        output_path=FIGURE_DIR / "s2_optimizer_ft5_bar.png",
        tick_step=500,
        label_format="{:.0f}",
    )


def plot_runtime(summary_rows: List[Dict[str, object]]) -> None:
    max_runtime = max(float(row["runtime_mean_sec"]) + float(row["runtime_std_sec"]) for row in summary_rows)
    _plot_grouped_optimizer_bars(
        summary_rows,
        value_field="runtime_mean_sec",
        std_field="runtime_std_sec",
        ylabel="Runtime per seed (s)",
        title="S2 optimizer comparison: runtime",
        output_path=FIGURE_DIR / "s2_optimizer_runtime_bar.png",
        tick_step=_runtime_tick_step(max_runtime),
        label_format="{:.1f}",
    )


def plot_convergence() -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in _all_convergence_rows()
        if row.get("relative_best_score") not in ("", None)
    ]
    if not rows:
        return

    grouped = _group_rows(rows, ("optimizer", "iteration"))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = {"pso": "#4C78A8", "ga": "#E45756", "de": "#54A24B"}
    for optimizer in OPTIMIZERS:
        iterations = sorted(
            {
                int(row["iteration"])
                for row in rows
                if row["optimizer"] == optimizer
            }
        )
        means: List[float] = []
        stds: List[float] = []
        for iteration in iterations:
            values = [
                float(row["relative_best_score"])
                for row in grouped.get((optimizer, str(iteration)), [])
                if row.get("relative_best_score") not in ("", None)
            ]
            means.append(mean(values))
            stds.append(_std(values))
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.plot(iterations, means, marker="o", markersize=3, linewidth=1.8, label=optimizer.upper(), color=colors[optimizer])
        ax.fill_between(iterations, lower, upper, color=colors[optimizer], alpha=0.16)

    ax.set_title("S2 optimizer convergence: mean relative best cost +/- std")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost / initial best cost")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.92)
    output_path = FIGURE_DIR / "s2_optimizer_convergence_relative_cost.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def run_all(args: argparse.Namespace) -> None:
    existing = _comparison_existing_keys()
    seeds = list(range(args.seed_start, args.seed_start + args.runs))
    requested_optimizers = [optimizer.lower() for optimizer in args.optimizers]

    for optimizer in requested_optimizers:
        if optimizer == "pso" and not args.force_pso:
            print("Use existing S2 PSO baseline; pass --force-pso to rerun PSO.")
            continue
        for space_m, node_counts in S2_CASES.items():
            for node_count in node_counts:
                for seed in seeds:
                    key = (optimizer, str(space_m), str(node_count), str(seed))
                    if key in existing and not args.force:
                        print(f"Skip existing {optimizer} space={space_m}, n={node_count}, seed={seed}")
                        continue
                    print(f"Run {optimizer} space={space_m}, n={node_count}, seed={seed}")
                    _run_one_case(
                        optimizer=optimizer,
                        space_m=space_m,
                        node_count=node_count,
                        seed=seed,
                        packet_size_bits=args.packet_size_bits,
                        initial_energy_j=args.initial_energy_j,
                        max_rounds=args.max_rounds,
                        population_size=args.population_size,
                        iterations=args.iterations,
                    )
                    existing.add(key)

    summary_rows = build_summary()
    plot_ft5(summary_rows)
    plot_runtime(summary_rows)
    plot_convergence()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PSO, GA, and DE for S2 cases.")
    parser.add_argument("--optimizers", nargs="+", default=["ga", "de"], choices=list(OPTIMIZERS))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=BASE_SEED + 2600)
    parser.add_argument("--max-rounds", type=int, default=5000)
    parser.add_argument("--packet-size-bits", type=int, default=6400)
    parser.add_argument("--initial-energy-j", type=float, default=0.5)
    parser.add_argument("--population-size", type=int, default=SIMULATION_PARAMS.pso_particles)
    parser.add_argument("--iterations", type=int, default=SIMULATION_PARAMS.pso_iterations)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-pso", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())

