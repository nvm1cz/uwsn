from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

from uwsn.metrics import summarize_runs
from uwsn.run_batch import run_simulation_batch
from uwsn.run_config import BASE_SEED, SIMULATION_CASE, SIMULATION_PARAMS


SUMMARY_LOG = Path("outputs/data/csv/summary/algorithm_node_sweep_v2_summary.csv")
RAW_LOG = Path("outputs/data/csv/raw/algorithm_node_sweep_v2_raw.csv")

NODE_COUNTS = (100, 200, 300, 500)
WEIGHT_SETS = (
    ("default", 0.60, 0.25, 0.15),
    ("energy-focused", 0.75, 0.15, 0.10),
    ("distance-focused", 0.45, 0.45, 0.10),
    ("degree-focused", 0.45, 0.15, 0.40),
    ("balanced", 0.34, 0.33, 0.33),
)

SUMMARY_FIELDS = [
    "suite",
    "label",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "pso_c1",
    "pso_c2",
    "pso_omega",
    "transmission_range_m",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "pso_particles",
    "pso_iterations",
    "runs",
    "seed_start",
    "seed_end",
    "max_rounds",
    "fnd_mean",
    "fnd_std",
    "fnd_min",
    "fnd_max",
    "fnd_cv_pct",
    "residual_energy_pct_at_fnd",
    "dead_nodes_pct_at_fnd",
    "packets_received_mean",
    "elapsed_sec",
]

RAW_FIELDS = [
    "suite",
    "label",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "pso_c1",
    "pso_c2",
    "pso_omega",
    "transmission_range_m",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "pso_particles",
    "pso_iterations",
    "seed",
    "max_rounds",
    "fnd_round",
    "residual_energy",
    "alive_nodes",
    "packets_received",
]


def _append_row(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _case_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return (
        row["suite"],
        row["label"],
        row["space_m"],
        row["node_count"],
        row["packet_size_bits"],
        row["initial_energy_j"],
        row["pso_omega"],
        row["cluster_head_ratio"],
        row["eulc_candidate_ratio"],
        row["alpha_weight"],
        row["beta_weight"],
        row["gamma_weight"],
        row["pso_particles"],
        row["pso_iterations"],
        row["runs"],
        row["seed_start"],
    )


def _existing_keys(path: Path) -> set[Tuple[str, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {_case_key(row) for row in csv.DictReader(f)}


def algorithm_cases(
    *,
    space_m: int,
    node_counts: Tuple[int, ...],
    packet_size_bits: int,
    initial_energy_j: float,
) -> Iterable[Dict[str, object]]:
    base = {
        "space_m": space_m,
        "packet_size_bits": packet_size_bits,
        "initial_energy_j": initial_energy_j,
        "pso_c1": 1.5,
        "pso_c2": 1.5,
        "pso_omega": 0.65,
        "transmission_range_m": 150.0,
        "cluster_head_ratio": 0.2,
        "eulc_candidate_ratio": 0.4,
        "alpha_weight": 0.60,
        "beta_weight": 0.25,
        "gamma_weight": 0.15,
    }
    for node_count in node_counts:
        for omega in (0.35, 0.5, 0.65, 0.8):
            yield {**base, "suite": "omega_by_node_v2", "label": f"omega={omega:g}", "node_count": node_count, "pso_omega": omega}

        for candidate_ratio in (0.2, 0.3, 0.4, 0.5):
            yield {
                **base,
                "suite": "candidate_by_node_v2",
                "label": f"candidate={candidate_ratio:g}",
                "node_count": node_count,
                "eulc_candidate_ratio": candidate_ratio,
            }

        for cluster_head_ratio in (0.1, 0.15, 0.2, 0.25, 0.3):
            yield {
                **base,
                "suite": "ch_ratio_by_node_v2",
                "label": f"Pc={cluster_head_ratio:g}",
                "node_count": node_count,
                "cluster_head_ratio": cluster_head_ratio,
            }

        for label, alpha, beta, gamma in WEIGHT_SETS:
            yield {
                **base,
                "suite": "weight_by_node_v2",
                "label": label,
                "node_count": node_count,
                "alpha_weight": alpha,
                "beta_weight": beta,
                "gamma_weight": gamma,
            }


def run_case(
    params: Dict[str, object],
    *,
    pso_particles: int,
    pso_iterations: int,
    runs: int,
    seed_start: int,
    max_rounds: int,
    workers: int,
) -> None:
    seeds = list(range(seed_start, seed_start + runs))
    case = replace(
        SIMULATION_CASE,
        name=f"{params['suite']}_{params['label']}_n{params['node_count']}",
        node_count=int(params["node_count"]),
        packet_size_bits=int(params["packet_size_bits"]),
        initial_energy=float(params["initial_energy_j"]),
    )
    sim_params = replace(
        SIMULATION_PARAMS,
        width_m=float(params["space_m"]),
        height_m=float(params["space_m"]),
        depth_m=float(params["space_m"]),
        pso_c1=float(params["pso_c1"]),
        pso_c2=float(params["pso_c2"]),
        pso_omega=float(params["pso_omega"]),
        transmission_range_m=float(params["transmission_range_m"]),
        cluster_head_ratio=float(params["cluster_head_ratio"]),
        eulc_candidate_ratio=float(params["eulc_candidate_ratio"]),
        alpha_weight=float(params["alpha_weight"]),
        beta_weight=float(params["beta_weight"]),
        gamma_weight=float(params["gamma_weight"]),
        pso_particles=pso_particles,
        pso_iterations=pso_iterations,
    )

    started = perf_counter()
    sim_runs = run_simulation_batch(
        case,
        sim_params,
        seeds,
        stop_on_first_dead=True,
        max_rounds=max_rounds,
        workers=workers,
    )
    elapsed = perf_counter() - started
    summary = summarize_runs(case, sim_runs)
    fnd_values = [run.fnd_round for run in sim_runs if run.fnd_round is not None]
    fnd_mean = mean(fnd_values) if fnd_values else None
    fnd_std = pstdev(fnd_values) if len(fnd_values) > 1 else 0.0
    fnd_cv_pct = (fnd_std / fnd_mean * 100.0) if fnd_mean else 0.0

    common = {
        **params,
        "pso_particles": pso_particles,
        "pso_iterations": pso_iterations,
    }
    for run in sim_runs:
        _append_row(
            RAW_LOG,
            RAW_FIELDS,
            {
                **common,
                "seed": run.seed,
                "max_rounds": max_rounds,
                "fnd_round": run.fnd_round or "",
                "residual_energy": round(run.residual_energy, 6),
                "alive_nodes": run.alive_nodes,
                "packets_received": run.packets_received,
            },
        )

    _append_row(
        SUMMARY_LOG,
        SUMMARY_FIELDS,
        {
            **common,
            "runs": runs,
            "seed_start": seed_start,
            "seed_end": seed_start + runs - 1,
            "max_rounds": max_rounds,
            "fnd_mean": round(fnd_mean, 4) if fnd_mean else "",
            "fnd_std": round(fnd_std, 4),
            "fnd_min": min(fnd_values) if fnd_values else "",
            "fnd_max": max(fnd_values) if fnd_values else "",
            "fnd_cv_pct": round(fnd_cv_pct, 4),
            "residual_energy_pct_at_fnd": summary["residual_energy_pct"],
            "dead_nodes_pct_at_fnd": summary["dead_nodes_pct"],
            "packets_received_mean": round(mean([run.packets_received for run in sim_runs]), 4),
            "elapsed_sec": round(elapsed, 2),
        },
    )
    print(
        f"[{params['suite']}] n={params['node_count']}, {params['label']} -> "
        f"FND={fnd_mean:.2f} +/- {fnd_std:.2f}, CV={fnd_cv_pct:.2f}%, t={elapsed:.2f}s",
        flush=True,
    )


def main() -> None:
    global RAW_LOG, SUMMARY_LOG

    parser = argparse.ArgumentParser()
    parser.add_argument("--space-m", type=int, default=100)
    parser.add_argument("--node-counts", type=str, default="100,200,300,500")
    parser.add_argument("--packet-size-bits", type=int, default=6400)
    parser.add_argument("--initial-energy-j", type=float, default=0.5)
    parser.add_argument("--output-tag", type=str, default="algorithm_node_sweep_v2")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=BASE_SEED + 90)
    parser.add_argument("--max-rounds", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pso-particles", type=int, default=4)
    parser.add_argument("--pso-iterations", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    node_counts = tuple(int(value.strip()) for value in args.node_counts.split(",") if value.strip())
    SUMMARY_LOG = Path(f"outputs/data/csv/summary/{args.output_tag}_summary.csv")
    RAW_LOG = Path(f"outputs/data/csv/raw/{args.output_tag}_raw.csv")

    existing = _existing_keys(SUMMARY_LOG)
    pending = []
    for params in algorithm_cases(
        space_m=args.space_m,
        node_counts=node_counts,
        packet_size_bits=args.packet_size_bits,
        initial_energy_j=args.initial_energy_j,
    ):
        key = (
            str(params["suite"]),
            str(params["label"]),
            str(params["space_m"]),
            str(params["node_count"]),
            str(params["packet_size_bits"]),
            str(params["initial_energy_j"]),
            str(params["pso_omega"]),
            str(params["cluster_head_ratio"]),
            str(params["eulc_candidate_ratio"]),
            str(params["alpha_weight"]),
            str(params["beta_weight"]),
            str(params["gamma_weight"]),
            str(args.pso_particles),
            str(args.pso_iterations),
            str(args.runs),
            str(args.seed_start),
        )
        if key not in existing:
            pending.append(params)

    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        f"Algorithm node sweep v2: space={args.space_m}, nodes={node_counts}, "
        f"pending={len(pending)}, runs/case={args.runs}, "
        f"seeds={args.seed_start}-{args.seed_start + args.runs - 1}",
        flush=True,
    )
    for params in pending:
        run_case(
            params,
            pso_particles=args.pso_particles,
            pso_iterations=args.pso_iterations,
            runs=args.runs,
            seed_start=args.seed_start,
            max_rounds=args.max_rounds,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()

