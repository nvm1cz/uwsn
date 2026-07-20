from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Dict, List, Tuple

from uwsn.metrics import summarize_runs
from uwsn.run_batch import run_simulation_batch
from uwsn.run_config import BASE_SEED, SIMULATION_CASE, SIMULATION_PARAMS


SUMMARY_LOG = Path("outputs/data/csv/summary/stability_topk_v2_summary.csv")
RAW_LOG = Path("outputs/data/csv/raw/stability_topk_v2_raw.csv")

SUMMARY_FIELDS = [
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "pso_c1",
    "pso_c2",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
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
    "elapsed_sec",
]

RAW_FIELDS = [
    "space_m",
    "node_count",
    "packet_size_bits",
    "initial_energy_j",
    "pso_c1",
    "pso_c2",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
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
        row["space_m"],
        row["node_count"],
        row["packet_size_bits"],
        row["initial_energy_j"],
        row["pso_c1"],
        row["pso_c2"],
        row["cluster_head_ratio"],
        row["eulc_candidate_ratio"],
        row["pso_particles"],
        row["pso_iterations"],
        row["runs"],
        row["seed_start"],
    )


def _existing_keys(path: Path) -> set[Tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {_case_key(row) for row in csv.DictReader(f)}


def _params_key(
    space_m: int,
    node_count: int,
    packet_size_bits: int,
    initial_energy_j: float,
    pso_c1: float,
    pso_c2: float,
    cluster_head_ratio: float,
    eulc_candidate_ratio: float,
    pso_particles: int,
    pso_iterations: int,
    runs: int,
    seed_start: int,
) -> Tuple[str, ...]:
    return (
        str(space_m),
        str(node_count),
        str(packet_size_bits),
        str(initial_energy_j),
        str(pso_c1),
        str(pso_c2),
        str(cluster_head_ratio),
        str(eulc_candidate_ratio),
        str(pso_particles),
        str(pso_iterations),
        str(runs),
        str(seed_start),
    )


def run_case(
    *,
    space_m: int,
    node_count: int,
    packet_size_bits: int,
    initial_energy_j: float,
    pso_c1: float,
    pso_c2: float,
    cluster_head_ratio: float,
    eulc_candidate_ratio: float,
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
        name=(
            f"stable_s{space_m}_n{node_count}_pkt{packet_size_bits}_e{initial_energy_j}"
            f"_c{pso_c1}_{pso_c2}"
        ),
        node_count=node_count,
        packet_size_bits=packet_size_bits,
        initial_energy=initial_energy_j,
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        pso_c1=pso_c1,
        pso_c2=pso_c2,
        cluster_head_ratio=cluster_head_ratio,
        eulc_candidate_ratio=eulc_candidate_ratio,
        pso_particles=pso_particles,
        pso_iterations=pso_iterations,
    )

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
                "space_m": space_m,
                "node_count": node_count,
                "packet_size_bits": packet_size_bits,
                "initial_energy_j": initial_energy_j,
                "pso_c1": pso_c1,
                "pso_c2": pso_c2,
                "cluster_head_ratio": cluster_head_ratio,
                "eulc_candidate_ratio": eulc_candidate_ratio,
                "pso_particles": pso_particles,
                "pso_iterations": pso_iterations,
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
            "space_m": space_m,
            "node_count": node_count,
            "packet_size_bits": packet_size_bits,
            "initial_energy_j": initial_energy_j,
            "pso_c1": pso_c1,
            "pso_c2": pso_c2,
            "cluster_head_ratio": cluster_head_ratio,
            "eulc_candidate_ratio": eulc_candidate_ratio,
            "pso_particles": pso_particles,
            "pso_iterations": pso_iterations,
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
            "elapsed_sec": round(elapsed, 2),
        },
    )
    print(
        f"space={space_m}, n={node_count}, pkt={packet_size_bits}, e={initial_energy_j}, "
        f"c=({pso_c1},{pso_c2}) -> FND={fnd_mean:.2f} +/- {fnd_std:.2f}, "
        f"cv={fnd_cv_pct:.2f}%, t={elapsed:.2f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=BASE_SEED)
    parser.add_argument("--max-rounds", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pso-particles", type=int, default=4)
    parser.add_argument("--pso-iterations", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    space_m = 100
    node_counts = (100, 200, 300, 500)
    packet_sizes = (4000, 6400)
    initial_energies = (0.5, 1.0)
    c_pairs = ((1.5, 1.5), (2.5, 0.5), (0.5, 2.5))
    cluster_head_ratio = 0.2
    eulc_candidate_ratio = 0.4

    existing = _existing_keys(SUMMARY_LOG)
    pending = []
    for node_count, packet_size_bits, initial_energy_j, (pso_c1, pso_c2) in product(
        node_counts,
        packet_sizes,
        initial_energies,
        c_pairs,
    ):
        key = _params_key(
            space_m,
            node_count,
            packet_size_bits,
            initial_energy_j,
            pso_c1,
            pso_c2,
            cluster_head_ratio,
            eulc_candidate_ratio,
            args.pso_particles,
            args.pso_iterations,
            args.runs,
            args.seed_start,
        )
        if key not in existing:
            pending.append((node_count, packet_size_bits, initial_energy_j, pso_c1, pso_c2))

    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        f"Stability validation: pending={len(pending)}, runs/case={args.runs}, "
        f"seeds={args.seed_start}-{args.seed_start + args.runs - 1}",
        flush=True,
    )
    for node_count, packet_size_bits, initial_energy_j, pso_c1, pso_c2 in pending:
        run_case(
            space_m=space_m,
            node_count=node_count,
            packet_size_bits=packet_size_bits,
            initial_energy_j=initial_energy_j,
            pso_c1=pso_c1,
            pso_c2=pso_c2,
            cluster_head_ratio=cluster_head_ratio,
            eulc_candidate_ratio=eulc_candidate_ratio,
            pso_particles=args.pso_particles,
            pso_iterations=args.pso_iterations,
            runs=args.runs,
            seed_start=args.seed_start,
            max_rounds=args.max_rounds,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()

