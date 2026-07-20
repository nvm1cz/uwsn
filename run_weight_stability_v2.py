from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Dict, List, Tuple

from uwsn.metrics import summarize_runs
from uwsn.run_batch import run_simulation_batch
from uwsn.run_config import BASE_SEED, SIMULATION_CASE, SIMULATION_PARAMS


SUMMARY_LOG = Path("outputs/data/csv/summary/stability_weight_v2_summary.csv")
RAW_LOG = Path("outputs/data/csv/raw/stability_weight_v2_raw.csv")

SUMMARY_FIELDS = [
    "label",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "pso_omega",
    "transmission_range_m",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
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
    "label",
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "seed",
    "max_rounds",
    "fnd_round",
    "residual_energy",
    "alive_nodes",
    "packets_received",
]

WEIGHT_SETS = [
    ("default", 0.60, 0.25, 0.15),
    ("energy-focused", 0.75, 0.15, 0.10),
    ("distance-focused", 0.45, 0.45, 0.10),
    ("degree-focused", 0.45, 0.15, 0.40),
    ("balanced", 0.34, 0.33, 0.33),
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
        row["label"],
        row["space_m"],
        row["node_count"],
        row["packet_size_bits"],
        row["initial_energy_j"],
        row["alpha_weight"],
        row["beta_weight"],
        row["gamma_weight"],
        row["runs"],
        row["seed_start"],
    )


def _existing_keys(path: Path) -> set[Tuple[str, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {_case_key(row) for row in csv.DictReader(f)}


def _params_key(
    label: str,
    space_m: int,
    node_count: int,
    packet_size_bits: int,
    initial_energy_j: float,
    alpha_weight: float,
    beta_weight: float,
    gamma_weight: float,
    runs: int,
    seed_start: int,
) -> Tuple[str, ...]:
    return (
        label,
        str(space_m),
        str(node_count),
        str(packet_size_bits),
        str(initial_energy_j),
        str(alpha_weight),
        str(beta_weight),
        str(gamma_weight),
        str(runs),
        str(seed_start),
    )


def run_case(
    *,
    label: str,
    space_m: int,
    node_count: int,
    packet_size_bits: int,
    initial_energy_j: float,
    alpha_weight: float,
    beta_weight: float,
    gamma_weight: float,
    runs: int,
    seed_start: int,
    max_rounds: int,
    workers: int,
) -> None:
    pso_omega = 0.65
    transmission_range_m = 150.0
    cluster_head_ratio = 0.2
    eulc_candidate_ratio = 0.4

    case = replace(
        SIMULATION_CASE,
        name=f"weight_{label}_s{space_m}_n{node_count}_pkt{packet_size_bits}_e{initial_energy_j}",
        node_count=node_count,
        packet_size_bits=packet_size_bits,
        initial_energy=initial_energy_j,
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        alpha_weight=alpha_weight,
        beta_weight=beta_weight,
        gamma_weight=gamma_weight,
        pso_omega=pso_omega,
        transmission_range_m=transmission_range_m,
        cluster_head_ratio=cluster_head_ratio,
        eulc_candidate_ratio=eulc_candidate_ratio,
        pso_particles=4,
        pso_iterations=3,
    )

    seeds = list(range(seed_start, seed_start + runs))
    started = perf_counter()
    sim_runs = run_simulation_batch(
        case,
        params,
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

    for run in sim_runs:
        _append_row(
            RAW_LOG,
            RAW_FIELDS,
            {
                "label": label,
                "space_m": space_m,
                "node_count": node_count,
                "packet_size_bits": packet_size_bits,
                "initial_energy_j": initial_energy_j,
                "alpha_weight": alpha_weight,
                "beta_weight": beta_weight,
                "gamma_weight": gamma_weight,
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
            "label": label,
            "space_m": space_m,
            "node_count": node_count,
            "packet_size_bits": packet_size_bits,
            "initial_energy_j": initial_energy_j,
            "alpha_weight": alpha_weight,
            "beta_weight": beta_weight,
            "gamma_weight": gamma_weight,
            "pso_omega": pso_omega,
            "transmission_range_m": transmission_range_m,
            "cluster_head_ratio": cluster_head_ratio,
            "eulc_candidate_ratio": eulc_candidate_ratio,
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
        f"{label}: alpha={alpha_weight}, beta={beta_weight}, gamma={gamma_weight} -> "
        f"FND={fnd_mean:.2f} +/- {fnd_std:.2f}, CV={fnd_cv_pct:.2f}%, t={elapsed:.2f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=BASE_SEED)
    parser.add_argument("--max-rounds", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    space_m = 100
    node_count = 300
    packet_size_bits = 6400
    initial_energy_j = 0.5

    existing = _existing_keys(SUMMARY_LOG)
    pending = [
        (label, alpha, beta, gamma)
        for label, alpha, beta, gamma in WEIGHT_SETS
        if _params_key(
            label,
            space_m,
            node_count,
            packet_size_bits,
            initial_energy_j,
            alpha,
            beta,
            gamma,
            args.runs,
            args.seed_start,
        )
        not in existing
    ]

    print(
        f"EULC weight stability v2: pending={len(pending)}, runs/case={args.runs}, "
        f"seeds={args.seed_start}-{args.seed_start + args.runs - 1}",
        flush=True,
    )

    for label, alpha_weight, beta_weight, gamma_weight in pending:
        run_case(
            label=label,
            space_m=space_m,
            node_count=node_count,
            packet_size_bits=packet_size_bits,
            initial_energy_j=initial_energy_j,
            alpha_weight=alpha_weight,
            beta_weight=beta_weight,
            gamma_weight=gamma_weight,
            runs=args.runs,
            seed_start=args.seed_start,
            max_rounds=args.max_rounds,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()

