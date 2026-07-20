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


RAW_LOG = Path("outputs/data/csv/raw/s2_density_realism_raw.csv")
SUMMARY_LOG = Path("outputs/data/csv/summary/s2_density_realism_summary.csv")
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
    "ft5_dead_threshold_nodes",
    "ft5_round",
    "fnd_round",
    "dead_nodes_at_stop",
    "alive_nodes_at_stop",
    "residual_energy",
    "packets_received",
    "runtime_sec",
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
    "ft5_dead_threshold_nodes",
    "ft5_mean",
    "ft5_std",
    "ft5_cv_pct",
    "fnd_mean",
    "fnd_std",
    "fnd_cv_pct",
    "runtime_mean_sec",
    "runtime_total_sec",
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


def _existing_keys() -> set[Tuple[str, str, str, str, str, str]]:
    return {
        (
            row["space_m"],
            row["node_count"],
            row["seed"],
            row.get("contention_penalty_factor", ""),
            row.get("pso_particles", ""),
            row.get("pso_iterations", ""),
        )
        for row in _load_csv(RAW_LOG)
        if row.get("suite") == "S2_DENSITY_REALISM"
    }


def _run_one_case(args: Dict[str, object]) -> Dict[str, object]:
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
        name=f"S2_density_s{space_m}_n{node_count}_seed{seed}",
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
    sim = PsoEulcSimulator(case, params, seed, verbose=False)
    result = sim.run(stop_on_first_dead=False, min_alive_ratio=0.95, max_rounds=max_rounds)
    elapsed = perf_counter() - started

    ft5_dead_threshold = int(math.ceil(0.05 * node_count))
    dead_nodes_at_stop = node_count - result.alive_nodes
    return {
        "suite": "S2_DENSITY_REALISM",
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
        "ft5_dead_threshold_nodes": ft5_dead_threshold,
        "ft5_round": result.first_5pct_round or "",
        "fnd_round": result.fnd_round or "",
        "dead_nodes_at_stop": dead_nodes_at_stop,
        "alive_nodes_at_stop": result.alive_nodes,
        "residual_energy": round(result.residual_energy, 6),
        "packets_received": result.packets_received,
        "runtime_sec": round(elapsed, 4),
    }


def build_summary() -> List[Dict[str, object]]:
    raw_rows = [row for row in _load_csv(RAW_LOG) if row.get("suite") == "S2_DENSITY_REALISM"]
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = {}
    for row in raw_rows:
        key = (
            row["space_m"],
            row["node_count"],
            row["contention_penalty_factor"],
            row["pso_particles"],
            row["pso_iterations"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
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
        ft5_values = [float(row["ft5_round"]) for row in rows if row.get("ft5_round")]
        fnd_values = [float(row["fnd_round"]) for row in rows if row.get("fnd_round")]
        runtime_values = [float(row["runtime_sec"]) for row in rows if row.get("runtime_sec")]
        first = rows[0]
        seeds = sorted(int(row["seed"]) for row in rows)
        summary_rows.append(
            {
                "suite": "S2_DENSITY_REALISM",
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
                "ft5_dead_threshold_nodes": int(first["ft5_dead_threshold_nodes"]),
                "ft5_mean": round(mean(ft5_values), 4) if ft5_values else "",
                "ft5_std": round(_std(ft5_values), 4) if ft5_values else "",
                "ft5_cv_pct": round(_cv_pct(ft5_values), 4) if ft5_values else "",
                "fnd_mean": round(mean(fnd_values), 4) if fnd_values else "",
                "fnd_std": round(_std(fnd_values), 4) if fnd_values else "",
                "fnd_cv_pct": round(_cv_pct(fnd_values), 4) if fnd_values else "",
                "runtime_mean_sec": round(mean(runtime_values), 4) if runtime_values else "",
                "runtime_total_sec": round(sum(runtime_values), 4) if runtime_values else "",
            }
        )
    _write_rows(SUMMARY_LOG, SUMMARY_FIELDS, summary_rows)
    return summary_rows


def _plot_metric(summary_rows: List[Dict[str, object]], metric: str, output_name: str) -> None:
    import matplotlib.pyplot as plt

    mean_field = f"{metric}_mean"
    std_field = f"{metric}_std"
    max_value = max(
        float(row[mean_field]) + float(row[std_field] or 0)
        for row in summary_rows
        if row.get(mean_field) not in ("", None)
    )
    y_upper = _nice_upper(max_value, 250)
    y_ticks = [250 * idx for idx in range(int(round(y_upper / 250)) + 1)]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, S2_CASES):
        panel = [row for row in summary_rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        labels = [str(row["node_count"]) for row in panel]
        means = [float(row[mean_field]) for row in panel]
        stds = [float(row[std_field] or 0) for row in panel]
        positions = list(range(len(labels)))
        bars = ax.bar(positions, means, yerr=stds, capsize=4, color="#4C78A8", alpha=0.84, width=0.66)
        for bar, mean_value, std_value in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean_value + std_value + y_upper * 0.025,
                f"{mean_value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel(f"{metric.upper()} round")
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, y_upper)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    first = summary_rows[0]
    fig.suptitle(
        "S2 density/contention model, "
        f"factor={first['contention_penalty_factor']}, "
        f"ref_degree={first['contention_reference_degree']}, "
        f"runs={first['runs']}/case"
    )
    output_path = FIGURE_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_figures(summary_rows: List[Dict[str, object]]) -> None:
    if not summary_rows:
        return
    try:
        _plot_metric(summary_rows, "fnd", "s2_density_realism_fnd_bar.png")
        _plot_metric(summary_rows, "ft5", "s2_density_realism_ft5_bar.png")
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("Skip figures because matplotlib is not installed in this Python environment.")


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
                    print(f"Skip existing density S2 space={space_m}, n={node_count}, seed={seed}")
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
        print(f"Running {len(jobs)} density-realism jobs with {args.workers} workers")
        completed: List[Dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_to_job = {executor.submit(_run_one_case, job): job for job in jobs}
            for future in as_completed(future_to_job):
                row = future.result()
                completed.append(row)
                print(
                    "Done density S2 "
                    f"space={row['space_m']}, n={row['node_count']}, seed={row['seed']}, "
                    f"FND={row['fnd_round']}, FT5={row['ft5_round']}"
                )
        completed.sort(key=lambda row: (int(row["space_m"]), int(row["node_count"]), int(row["seed"])))
        _append_rows(RAW_LOG, RAW_FIELDS, completed)

    summary_rows = build_summary()
    plot_figures(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S2 with an explicit density/contention overhead model.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=BASE_SEED + 2600)
    parser.add_argument("--max-rounds", type=int, default=5000)
    parser.add_argument("--packet-size-bits", type=int, default=6400)
    parser.add_argument("--initial-energy-j", type=float, default=0.5)
    parser.add_argument("--pso-particles", type=int, default=SIMULATION_PARAMS.pso_particles)
    parser.add_argument("--pso-iterations", type=int, default=SIMULATION_PARAMS.pso_iterations)
    parser.add_argument("--contention-penalty-factor", type=float, default=0.35)
    parser.add_argument("--contention-reference-degree", type=float, default=10.0)
    parser.add_argument("--contention-max-multiplier", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())

