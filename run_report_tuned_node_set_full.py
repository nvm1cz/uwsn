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


SOURCE_RAW_LOG = Path("outputs/data/csv/raw/report_node_set_smooth_runs.csv")
RAW_LOG = Path("outputs/data/csv/raw/report_tuned_node_set_full_runs.csv")
SUMMARY_LOG = Path("outputs/data/csv/summary/report_tuned_node_set_full_summary.csv")
FIGURE_DIR = Path("outputs/figures/report_node_set_smooth")

TUNED_NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
    1000: (500, 1000, 1500, 2000),
}

RUNS = 30
SEED_START = 4000
MAX_ROUNDS = 5000

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
    "avg_neighbor_count",
    "recluster_count",
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
    "fnd_mean",
    "fnd_std",
    "fnd_cv_pct",
    "ft5_mean",
    "ft5_std",
    "ft5_cv_pct",
    "avg_neighbor_count_mean",
    "avg_neighbor_count_std",
    "node_density_per_m3",
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


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _cv_pct(values: Sequence[float]) -> float:
    avg = mean(values) if values else 0.0
    return (_std(values) / avg * 100.0) if avg else 0.0


def _key(row: Dict[str, object]) -> Tuple[str, str, str]:
    return (str(row["space_m"]), str(row["node_count"]), str(row["seed"]))


def _is_tuned_case(row: Dict[str, str]) -> bool:
    try:
        space_m = int(row["space_m"])
        node_count = int(row["node_count"])
    except (KeyError, ValueError):
        return False
    return space_m in TUNED_NODE_SETS and node_count in TUNED_NODE_SETS[space_m]


def _normalized_existing_row(row: Dict[str, str]) -> Dict[str, object]:
    return {
        field: row.get(field, "")
        for field in RAW_FIELDS
    }


def _run_one(job: Dict[str, int]) -> Dict[str, object]:
    space_m = int(job["space_m"])
    node_count = int(job["node_count"])
    seed = int(job["seed"])

    case = replace(
        SIMULATION_CASE,
        name=f"report_tuned_full_s{space_m}_n{node_count}_seed{seed}",
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
        pso_iterations=10,
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
        "suite": "REPORT_TUNED_NODE_SET_FULL",
        "space_m": space_m,
        "node_count": node_count,
        "packet_size_bits": case.packet_size_bits,
        "initial_energy_j": case.initial_energy,
        "transmission_range_m": params.transmission_range_m,
        "contention_penalty_factor": params.contention_penalty_factor,
        "contention_reference_degree": params.contention_reference_degree,
        "contention_max_multiplier": params.contention_max_multiplier,
        "pso_particles": params.pso_particles,
        "pso_iterations": params.pso_iterations,
        "seed": seed,
        "max_rounds": MAX_ROUNDS,
        "ft5_round": result.first_5pct_round or "",
        "fnd_round": result.fnd_round or "",
        "dead_nodes_at_stop": node_count - result.alive_nodes,
        "alive_nodes_at_stop": result.alive_nodes,
        "avg_neighbor_count": round(avg_neighbor_count, 4),
        "recluster_count": "",
        "runtime_sec": round(elapsed, 4),
    }


def build_rows() -> List[Dict[str, object]]:
    rows_by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    for row in _load_csv(SOURCE_RAW_LOG):
        if _is_tuned_case(row) and int(row.get("seed", -1)) in range(SEED_START, SEED_START + RUNS):
            normalized = _normalized_existing_row(row)
            normalized["suite"] = "REPORT_TUNED_NODE_SET_FULL"
            rows_by_key[_key(normalized)] = normalized

    for row in _load_csv(RAW_LOG):
        if _is_tuned_case(row) and int(row.get("seed", -1)) in range(SEED_START, SEED_START + RUNS):
            rows_by_key[_key(row)] = _normalized_existing_row(row)

    jobs: List[Dict[str, int]] = []
    for space_m, node_counts in TUNED_NODE_SETS.items():
        for node_count in node_counts:
            for seed in range(SEED_START, SEED_START + RUNS):
                key = (str(space_m), str(node_count), str(seed))
                if key not in rows_by_key:
                    jobs.append({"space_m": space_m, "node_count": node_count, "seed": seed})

    if jobs:
        print(f"Running {len(jobs)} missing tuned full jobs")
        with ProcessPoolExecutor(max_workers=4) as executor:
            future_to_job = {executor.submit(_run_one, job): job for job in jobs}
            for future in as_completed(future_to_job):
                row = future.result()
                rows_by_key[_key(row)] = row
                print(
                    "Done tuned full "
                    f"space={row['space_m']}, n={row['node_count']}, seed={row['seed']}, "
                    f"FT5={row['ft5_round']}, FND={row['fnd_round']}, runtime={row['runtime_sec']}s",
                    flush=True,
                )
                checkpoint_rows = list(rows_by_key.values())
                checkpoint_rows.sort(key=lambda item: (int(item["space_m"]), int(item["node_count"]), int(item["seed"])))
                _write_rows(RAW_LOG, RAW_FIELDS, checkpoint_rows)

    rows = list(rows_by_key.values())
    rows.sort(key=lambda row: (int(row["space_m"]), int(row["node_count"]), int(row["seed"])))
    _write_rows(RAW_LOG, RAW_FIELDS, rows)
    return rows


def build_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["space_m"]), int(row["node_count"])), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (space_m, node_count), group in sorted(grouped.items()):
        first = group[0]
        fnd_values = [float(row["fnd_round"]) for row in group if row.get("fnd_round")]
        ft5_values = [float(row["ft5_round"]) for row in group if row.get("ft5_round")]
        neighbor_values = [float(row["avg_neighbor_count"]) for row in group if row.get("avg_neighbor_count")]
        runtime_values = [float(row["runtime_sec"]) for row in group if row.get("runtime_sec")]
        recluster_values = [float(row["recluster_count"]) for row in group if row.get("recluster_count")]
        seeds = sorted(int(row["seed"]) for row in group)
        summary_rows.append(
            {
                "suite": "REPORT_TUNED_NODE_SET_FULL",
                "space_m": space_m,
                "node_count": node_count,
                "packet_size_bits": int(float(first["packet_size_bits"])),
                "initial_energy_j": float(first["initial_energy_j"]),
                "transmission_range_m": float(first["transmission_range_m"]),
                "contention_penalty_factor": float(first["contention_penalty_factor"]),
                "contention_reference_degree": float(first["contention_reference_degree"]),
                "contention_max_multiplier": float(first["contention_max_multiplier"]),
                "pso_particles": int(float(first["pso_particles"])),
                "pso_iterations": int(float(first["pso_iterations"])),
                "runs": len(group),
                "seed_start": min(seeds),
                "seed_end": max(seeds),
                "max_rounds": int(float(first["max_rounds"])),
                "fnd_mean": round(mean(fnd_values), 4) if fnd_values else "",
                "fnd_std": round(_std(fnd_values), 4) if fnd_values else "",
                "fnd_cv_pct": round(_cv_pct(fnd_values), 4) if fnd_values else "",
                "ft5_mean": round(mean(ft5_values), 4) if ft5_values else "",
                "ft5_std": round(_std(ft5_values), 4) if ft5_values else "",
                "ft5_cv_pct": round(_cv_pct(ft5_values), 4) if ft5_values else "",
                "avg_neighbor_count_mean": round(mean(neighbor_values), 4) if neighbor_values else "",
                "avg_neighbor_count_std": round(_std(neighbor_values), 4) if neighbor_values else "",
                "node_density_per_m3": f"{node_count / (float(space_m) ** 3):.10f}",
                "recluster_mean": round(mean(recluster_values), 4) if recluster_values else "",
                "runtime_mean_sec": round(mean(runtime_values), 4) if runtime_values else "",
            }
        )

    _write_rows(SUMMARY_LOG, SUMMARY_FIELDS, summary_rows)
    return summary_rows


def _nice_upper(max_value: float, tick_step: float) -> float:
    if max_value <= 0:
        return tick_step
    return max(tick_step, math.ceil(max_value * 1.10 / tick_step) * tick_step)


def _panel_setup(ax, space_m: int, y_upper: float, y_tick_step: float, ylabel: str) -> None:
    ticks = list(TUNED_NODE_SETS[space_m])
    pad = max(5, int((max(ticks) - min(ticks)) * 0.10))
    ax.set_title(f"space={space_m}m")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(ticks) - pad, max(ticks) + pad)
    ax.set_xticks(ticks)
    ax.set_ylim(0, y_upper)
    ax.set_yticks([y_tick_step * idx for idx in range(int(round(y_upper / y_tick_step)) + 1)])
    ax.tick_params(axis="y", labelleft=True)
    ax.grid(True, linestyle="--", alpha=0.35)


def plot_ft5(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    max_y = max(float(row["ft5_mean"]) + float(row["ft5_std"]) for row in rows)
    y_upper = _nice_upper(max_y, 250)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, [100, 500, 1000]):
        panel = [row for row in rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        xs = [int(row["node_count"]) for row in panel]
        means = [float(row["ft5_mean"]) for row in panel]
        stds = [float(row["ft5_std"]) for row in panel]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2.0, capsize=4, color="#4C78A8")
        for x_value, mean_value, std_value in zip(xs, means, stds):
            ax.text(x_value, mean_value + std_value + y_upper * 0.025, f"{mean_value:.0f}", ha="center", fontsize=8)
        _panel_setup(ax, space_m, y_upper, 250, "FT5 round")
    fig.suptitle("FT5 mean +/- std across 30 seeds, tuned node sets")
    output_path = FIGURE_DIR / "report_node_set_ft5_line_errorbar.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_cv(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    max_y = max(float(row["ft5_cv_pct"]) for row in rows)
    y_upper = _nice_upper(max_y, 2.5)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, [100, 500, 1000]):
        panel = [row for row in rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        xs = [int(row["node_count"]) for row in panel]
        cvs = [float(row["ft5_cv_pct"]) for row in panel]
        ax.plot(xs, cvs, marker="o", linewidth=2.0, color="#4C78A8")
        for x_value, cv_value in zip(xs, cvs):
            ax.text(x_value, cv_value + y_upper * 0.025, f"{cv_value:.1f}%", ha="center", fontsize=8)
        _panel_setup(ax, space_m, y_upper, 2.5, "FT5 CV (%)")
    fig.suptitle("FT5 coefficient of variation across 30 seeds, tuned node sets")
    output_path = FIGURE_DIR / "report_node_set_ft5_cv_line.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_runtime(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    max_y = max(float(row["runtime_mean_sec"]) for row in rows)
    y_upper = _nice_upper(max_y, 20)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, [100, 500, 1000]):
        panel = [row for row in rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        xs = [int(row["node_count"]) for row in panel]
        runtimes = [float(row["runtime_mean_sec"]) for row in panel]
        ax.plot(xs, runtimes, marker="o", linewidth=2.0, color="#4C78A8")
        for x_value, runtime_value in zip(xs, runtimes):
            ax.text(x_value, runtime_value + y_upper * 0.025, f"{runtime_value:.1f}s", ha="center", fontsize=8)
        _panel_setup(ax, space_m, y_upper, 20, "Runtime per seed (s)")
    fig.suptitle("Runtime mean across 30 seeds, tuned node sets")
    output_path = FIGURE_DIR / "report_node_set_runtime_line.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    raw_rows = build_rows()
    summary_rows = build_summary(raw_rows)
    plot_ft5(summary_rows)
    plot_cv(summary_rows)
    plot_runtime(summary_rows)


if __name__ == "__main__":
    main()

