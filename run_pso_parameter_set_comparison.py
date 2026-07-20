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

from uwsn.run_config import SIMULATION_CASE, SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


RAW_LOG = Path("outputs/data/csv/raw/pso_parameter_set_comparison_runs.csv")
CONVERGENCE_LOG = Path("outputs/data/csv/raw/pso_parameter_set_comparison_convergence.csv")
SUMMARY_LOG = Path("outputs/data/csv/summary/pso_parameter_set_comparison_summary.csv")
FIGURE_DIR = Path("outputs/figures/pso_parameter_set_comparison")

NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
    1000: (500, 1000, 1500, 2000),
}

PARAMETER_SETS = (
    {
        "label": "baseline",
        "alpha_weight": 0.60,
        "beta_weight": 0.25,
        "gamma_weight": 0.15,
        "pso_omega": 0.65,
        "cluster_head_ratio": 0.20,
        "eulc_candidate_ratio": 0.40,
    },
    {
        "label": "energy-focused",
        "alpha_weight": 0.75,
        "beta_weight": 0.15,
        "gamma_weight": 0.10,
        "pso_omega": 0.65,
        "cluster_head_ratio": 0.20,
        "eulc_candidate_ratio": 0.40,
    },
    {
        "label": "distance-focused",
        "alpha_weight": 0.45,
        "beta_weight": 0.45,
        "gamma_weight": 0.10,
        "pso_omega": 0.65,
        "cluster_head_ratio": 0.20,
        "eulc_candidate_ratio": 0.40,
    },
    {
        "label": "balanced-low-omega",
        "alpha_weight": 0.34,
        "beta_weight": 0.33,
        "gamma_weight": 0.33,
        "pso_omega": 0.50,
        "cluster_head_ratio": 0.20,
        "eulc_candidate_ratio": 0.40,
    },
)

RUN_FIELDS = [
    "suite",
    "param_label",
    "space_m",
    "node_count",
    "seed",
    "runs",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "pso_omega",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
    "pso_particles",
    "pso_iterations",
    "max_rounds",
    "ft5_round",
    "fnd_round",
    "dead_nodes_at_stop",
    "alive_nodes_at_stop",
    "runtime_sec",
]

CONVERGENCE_FIELDS = [
    "suite",
    "param_label",
    "space_m",
    "node_count",
    "seed",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "iteration",
    "best_score",
    "relative_best_score",
    "improvement_pct",
    "candidate_count",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "pso_omega",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
    "pso_particles",
    "pso_iterations",
]

SUMMARY_FIELDS = [
    "suite",
    "param_label",
    "space_m",
    "node_count",
    "runs",
    "ft5_mean",
    "ft5_std",
    "ft5_cv_pct",
    "fnd_mean",
    "fnd_std",
    "runtime_mean_sec",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "pso_omega",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
    "pso_particles",
    "pso_iterations",
]

COLORS = {
    "baseline": "#4C78A8",
    "energy-focused": "#E45756",
    "distance-focused": "#54A24B",
    "balanced-low-omega": "#B279A2",
}
MARKERS = {
    "baseline": "o",
    "energy-focused": "s",
    "distance-focused": "^",
    "balanced-low-omega": "D",
}


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


def _key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (str(row["param_label"]), str(row["space_m"]), str(row["node_count"]), str(row["seed"]))


def _conv_key(row: Dict[str, object]) -> Tuple[str, str, str, str, str, str]:
    return (
        str(row["param_label"]),
        str(row["space_m"]),
        str(row["node_count"]),
        str(row["seed"]),
        str(row["pso_iterations"]),
        str(row["iteration"]),
    )


def _normalize_events(events: List[Dict[str, float]]) -> List[Dict[str, object]]:
    initial = None
    rows: List[Dict[str, object]] = []
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
        rows.append({**event, "relative_best_score": relative, "improvement_pct": improvement})
    return rows


def _build_case(param_set: Dict[str, object], space_m: int, node_count: int, seed: int, args: argparse.Namespace):
    case = replace(
        SIMULATION_CASE,
        name=f"pso_param_{param_set['label']}_s{space_m}_n{node_count}_seed{seed}",
        node_count=node_count,
        packet_size_bits=args.packet_size_bits,
        initial_energy=args.initial_energy_j,
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        optimizer="pso",
        pso_particles=args.pso_particles,
        pso_iterations=args.pso_iterations,
        transmission_range_m=args.transmission_range_m,
        contention_penalty_factor=args.contention_penalty_factor,
        contention_reference_degree=10.0,
        contention_max_multiplier=4.0,
        alpha_weight=float(param_set["alpha_weight"]),
        beta_weight=float(param_set["beta_weight"]),
        gamma_weight=float(param_set["gamma_weight"]),
        pso_omega=float(param_set["pso_omega"]),
        cluster_head_ratio=float(param_set["cluster_head_ratio"]),
        eulc_candidate_ratio=float(param_set["eulc_candidate_ratio"]),
    )
    return case, params


def _common_fields(param_set: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    return {
        "suite": "PSO_PARAMETER_SET_COMPARISON",
        "param_label": param_set["label"],
        "packet_size_bits": args.packet_size_bits,
        "initial_energy_j": args.initial_energy_j,
        "transmission_range_m": args.transmission_range_m,
        "alpha_weight": param_set["alpha_weight"],
        "beta_weight": param_set["beta_weight"],
        "gamma_weight": param_set["gamma_weight"],
        "pso_omega": param_set["pso_omega"],
        "cluster_head_ratio": param_set["cluster_head_ratio"],
        "eulc_candidate_ratio": param_set["eulc_candidate_ratio"],
        "pso_particles": args.pso_particles,
        "pso_iterations": args.pso_iterations,
    }


def _run_full_job(job: Dict[str, object]) -> Dict[str, object]:
    param_set = job["param_set"]
    args = job["args"]
    space_m = int(job["space_m"])
    node_count = int(job["node_count"])
    seed = int(job["seed"])
    case, params = _build_case(param_set, space_m, node_count, seed, args)
    started = perf_counter()
    sim = PsoEulcSimulator(case, params, seed, verbose=False, track_pso_convergence=False)
    result = sim.run(stop_on_first_dead=False, min_alive_ratio=0.95, max_rounds=args.max_rounds)
    elapsed = perf_counter() - started
    return {
        **_common_fields(param_set, args),
        "space_m": space_m,
        "node_count": node_count,
        "seed": seed,
        "runs": args.runs,
        "max_rounds": args.max_rounds,
        "ft5_round": result.first_5pct_round or "",
        "fnd_round": result.fnd_round or "",
        "dead_nodes_at_stop": node_count - result.alive_nodes,
        "alive_nodes_at_stop": result.alive_nodes,
        "runtime_sec": round(elapsed, 4),
    }


def _run_convergence_job(job: Dict[str, object]) -> List[Dict[str, object]]:
    param_set = job["param_set"]
    args = job["args"]
    space_m = int(job["space_m"])
    node_count = int(job["node_count"])
    seed = int(job["seed"])
    case, params = _build_case(param_set, space_m, node_count, seed, args)
    params = replace(params, pso_iterations=args.convergence_iterations)
    sim = PsoEulcSimulator(case, params, seed, verbose=False, track_pso_convergence=True)
    result = sim.run(stop_on_first_dead=False, min_alive_ratio=None, max_rounds=1)
    rows: List[Dict[str, object]] = []
    for event in _normalize_events(result.pso_convergence):
        rows.append(
            {
                **_common_fields(param_set, args),
                "space_m": space_m,
                "node_count": node_count,
                "seed": seed,
                "pso_iterations": args.convergence_iterations,
                "iteration": int(event["iteration"]),
                "best_score": round(float(event["best_score"]), 8),
                "relative_best_score": event["relative_best_score"],
                "improvement_pct": event["improvement_pct"],
                "candidate_count": int(event["candidate_count"]),
            }
        )
    return rows


def _jobs(
    existing_keys: set[Tuple[str, ...]],
    args: argparse.Namespace,
    *,
    convergence: bool = False,
) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    seeds = range(args.seed_start, args.seed_start + args.runs)
    for param_set in PARAMETER_SETS:
        for space_m, node_counts in NODE_SETS.items():
            for node_count in node_counts:
                for seed in seeds:
                    key = (
                        str(param_set["label"]),
                        str(space_m),
                        str(node_count),
                        str(seed),
                        str(args.convergence_iterations),
                    ) if convergence else (
                        str(param_set["label"]),
                        str(space_m),
                        str(node_count),
                        str(seed),
                    )
                    if key not in existing_keys:
                        jobs.append(
                            {
                                "param_set": param_set,
                                "space_m": space_m,
                                "node_count": node_count,
                                "seed": seed,
                                "args": args,
                            }
                        )
    if args.limit > 0:
        jobs = jobs[: args.limit]
    return jobs


def run_missing(args: argparse.Namespace) -> None:
    full_rows = _load_csv(RAW_LOG)
    full_by_key = {_key(row): row for row in full_rows}
    full_jobs = _jobs(set(full_by_key), args)
    if full_jobs:
        print(f"Running {len(full_jobs)} missing FT5 jobs, RUNS={args.runs}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(_run_full_job, job): job for job in full_jobs}
            for future in as_completed(future_to_job):
                row = future.result()
                full_by_key[_key(row)] = row
                print(
                    f"Done FT5 {row['param_label']} s={row['space_m']} n={row['node_count']} "
                    f"seed={row['seed']} ft5={row['ft5_round']}",
                    flush=True,
                )
                _write_rows(RAW_LOG, RUN_FIELDS, sorted(full_by_key.values(), key=lambda item: _key(item)))
    else:
        print("No missing FT5 jobs.", flush=True)

    conv_rows = _load_csv(CONVERGENCE_LOG)
    conv_by_key = {_conv_key(row): row for row in conv_rows}
    existing_prefixes = {key[:5] for key in conv_by_key}
    conv_jobs = _jobs(existing_prefixes, args, convergence=True)
    if conv_jobs:
        print(f"Running {len(conv_jobs)} missing convergence jobs, RUNS={args.runs}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(_run_convergence_job, job): job for job in conv_jobs}
            for future in as_completed(future_to_job):
                rows = future.result()
                for row in rows:
                    conv_by_key[_conv_key(row)] = row
                if rows:
                    first = rows[0]
                    print(
                        f"Done convergence {first['param_label']} s={first['space_m']} "
                        f"n={first['node_count']} seed={first['seed']}",
                        flush=True,
                    )
                _write_rows(
                    CONVERGENCE_LOG,
                    CONVERGENCE_FIELDS,
                    sorted(conv_by_key.values(), key=lambda item: _conv_key(item)),
                )
    else:
        print("No missing convergence jobs.", flush=True)


def build_summary() -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int], List[Dict[str, str]]] = {}
    for row in _load_csv(RAW_LOG):
        grouped.setdefault((row["param_label"], int(row["space_m"]), int(row["node_count"])), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    param_lookup = {str(item["label"]): item for item in PARAMETER_SETS}
    for param_set in PARAMETER_SETS:
        label = str(param_set["label"])
        for space_m, node_counts in NODE_SETS.items():
            for node_count in node_counts:
                rows = grouped.get((label, space_m, node_count), [])
                if not rows:
                    continue
                ft5_values = [float(row["ft5_round"]) for row in rows if row.get("ft5_round")]
                fnd_values = [float(row["fnd_round"]) for row in rows if row.get("fnd_round")]
                runtime_values = [float(row["runtime_sec"]) for row in rows if row.get("runtime_sec")]
                source = param_lookup[label]
                summary_rows.append(
                    {
                        "suite": "PSO_PARAMETER_SET_COMPARISON",
                        "param_label": label,
                        "space_m": space_m,
                        "node_count": node_count,
                        "runs": len(rows),
                        "ft5_mean": round(mean(ft5_values), 4) if ft5_values else "",
                        "ft5_std": round(_std(ft5_values), 4) if ft5_values else "",
                        "ft5_cv_pct": round(_cv_pct(ft5_values), 4) if ft5_values else "",
                        "fnd_mean": round(mean(fnd_values), 4) if fnd_values else "",
                        "fnd_std": round(_std(fnd_values), 4) if fnd_values else "",
                        "runtime_mean_sec": round(mean(runtime_values), 4) if runtime_values else "",
                        "alpha_weight": source["alpha_weight"],
                        "beta_weight": source["beta_weight"],
                        "gamma_weight": source["gamma_weight"],
                        "pso_omega": source["pso_omega"],
                        "cluster_head_ratio": source["cluster_head_ratio"],
                        "eulc_candidate_ratio": source["eulc_candidate_ratio"],
                        "pso_particles": rows[0]["pso_particles"],
                        "pso_iterations": rows[0]["pso_iterations"],
                    }
                )
    _write_rows(SUMMARY_LOG, SUMMARY_FIELDS, summary_rows)
    return summary_rows


def _panel(rows: List[Dict[str, object]], label: str, space_m: int) -> List[Dict[str, object]]:
    panel = [row for row in rows if row["param_label"] == label and int(row["space_m"]) == space_m]
    panel.sort(key=lambda row: int(row["node_count"]))
    return panel


def _spaces() -> List[int]:
    return list(NODE_SETS.keys())


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


def plot_ft5(summary_rows: List[Dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    max_y = max(float(row["ft5_mean"]) + float(row["ft5_std"] or 0) for row in summary_rows if row.get("ft5_mean"))
    y_upper = _nice_upper(max_y, 250)
    spaces = _spaces()
    fig, axes = plt.subplots(1, len(spaces), figsize=(5.2 * len(spaces), 4.8), sharey=True, constrained_layout=True)
    axes = [axes] if len(spaces) == 1 else axes
    for ax, space_m in zip(axes, spaces):
        for param_set in PARAMETER_SETS:
            label = str(param_set["label"])
            panel = _panel(summary_rows, label, space_m)
            xs = [int(row["node_count"]) for row in panel]
            means = [float(row["ft5_mean"]) for row in panel]
            stds = [float(row["ft5_std"] or 0) for row in panel]
            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=stds,
                    marker=MARKERS[label],
                    linewidth=1.8,
                    capsize=3,
                    color=COLORS[label],
                    label=label,
                )
        _setup_node_axis(ax, space_m, y_upper, 250, "FT5 round")
        ax.legend(loc="best", framealpha=0.92, fontsize=8)
    fig.suptitle("FT5 mean +/- std across PSO parameter sets")
    output = FIGURE_DIR / "01_ft5_parameter_sets.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_cv(summary_rows: List[Dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    max_y = max(float(row["ft5_cv_pct"]) for row in summary_rows if row.get("ft5_cv_pct"))
    y_upper = _nice_upper(max_y, 2.5)
    spaces = _spaces()
    fig, axes = plt.subplots(1, len(spaces), figsize=(5.2 * len(spaces), 4.8), sharey=True, constrained_layout=True)
    axes = [axes] if len(spaces) == 1 else axes
    for ax, space_m in zip(axes, spaces):
        for param_set in PARAMETER_SETS:
            label = str(param_set["label"])
            panel = _panel(summary_rows, label, space_m)
            xs = [int(row["node_count"]) for row in panel]
            cvs = [float(row["ft5_cv_pct"]) for row in panel]
            if xs:
                ax.plot(
                    xs,
                    cvs,
                    marker=MARKERS[label],
                    linewidth=1.8,
                    color=COLORS[label],
                    label=label,
                )
        _setup_node_axis(ax, space_m, y_upper, 2.5, "FT5 CV (%)")
        ax.legend(loc="best", framealpha=0.92, fontsize=8)
    fig.suptitle("FT5 coefficient of variation across PSO parameter sets")
    output = FIGURE_DIR / "03_ft5_cv_parameter_sets.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_convergence() -> Path:
    import matplotlib.pyplot as plt

    grouped: Dict[Tuple[str, int, int], List[float]] = {}
    max_iteration = 0
    rows = _load_csv(CONVERGENCE_LOG)
    max_budget = max((int(row["pso_iterations"]) for row in rows if row.get("pso_iterations")), default=0)
    for row in rows:
        if row.get("relative_best_score") in ("", None):
            continue
        if max_budget and int(row["pso_iterations"]) != max_budget:
            continue
        iteration = int(row["iteration"])
        max_iteration = max(max_iteration, iteration)
        key = (row["param_label"], int(row["space_m"]), int(row["iteration"]))
        grouped.setdefault(key, []).append(float(row["relative_best_score"]))

    spaces = _spaces()
    fig, axes = plt.subplots(1, len(spaces), figsize=(5.2 * len(spaces), 4.8), sharey=True, constrained_layout=True)
    axes = [axes] if len(spaces) == 1 else axes
    for ax, space_m in zip(axes, spaces):
        for param_set in PARAMETER_SETS:
            label = str(param_set["label"])
            iterations = sorted(iteration for param, space, iteration in grouped if param == label and space == space_m)
            if not iterations:
                continue
            means = [mean(grouped[(label, space_m, iteration)]) for iteration in iterations]
            stds = [_std(grouped[(label, space_m, iteration)]) for iteration in iterations]
            lower = [max(0.0, avg - sd) for avg, sd in zip(means, stds)]
            upper = [min(1.1, avg + sd) for avg, sd in zip(means, stds)]
            ax.plot(
                iterations,
                means,
                marker=MARKERS[label],
                markersize=2.6,
                linewidth=1.7,
                color=COLORS[label],
                label=label,
            )
            ax.fill_between(iterations, lower, upper, color=COLORS[label], alpha=0.09)
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("PSO iteration")
        ax.set_ylabel("Best cost / initial best cost")
        x_upper = max(50, max_iteration)
        ax.set_xlim(0, x_upper)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([tick for tick in [0, 10, 20, 30, 40, 50, 75, 100] if tick <= x_upper])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", framealpha=0.92, fontsize=8)
    fig.suptitle("PSO convergence across parameter sets")
    output = FIGURE_DIR / "02_convergence_parameter_sets.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_all() -> List[Path]:
    summary_rows = build_summary()
    return [plot_ft5(summary_rows), plot_convergence(), plot_cv(summary_rows)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N missing jobs, then plot existing data.")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--space-m", type=int, choices=[100, 500, 1000], default=0, help="Optional single space to run.")
    parser.add_argument("--max-rounds", type=int, default=5000)
    parser.add_argument("--packet-size-bits", type=int, default=6400)
    parser.add_argument("--initial-energy-j", type=float, default=0.5)
    parser.add_argument("--transmission-range-m", type=float, default=150.0)
    parser.add_argument("--contention-penalty-factor", type=float, default=0.01)
    parser.add_argument("--pso-particles", type=int, default=4)
    parser.add_argument("--pso-iterations", type=int, default=10)
    parser.add_argument("--convergence-iterations", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.space_m:
        global NODE_SETS
        NODE_SETS = {args.space_m: NODE_SETS[args.space_m]}
    if not args.plot_only:
        run_missing(args)
    outputs = plot_all()
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()

