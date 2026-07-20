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


SUMMARY_LOG = Path("outputs/data/csv/summary/stability_stress_v2_summary.csv")
RAW_LOG = Path("outputs/data/csv/raw/stability_stress_v2_raw.csv")


FIELDS = [
    "suite",
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
        row["space_m"],
        row["node_count"],
        row["packet_size_bits"],
        row["initial_energy_j"],
        row["pso_c1"],
        row["pso_c2"],
        row["pso_omega"],
        row["transmission_range_m"],
        row["cluster_head_ratio"],
        row["eulc_candidate_ratio"],
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


def _params_key(params: Dict[str, object], runs: int, seed_start: int, pso_particles: int, pso_iterations: int) -> Tuple[str, ...]:
    return (
        str(params["suite"]),
        str(params["space_m"]),
        str(params["node_count"]),
        str(params["packet_size_bits"]),
        str(params["initial_energy_j"]),
        str(params["pso_c1"]),
        str(params["pso_c2"]),
        str(params["pso_omega"]),
        str(params["transmission_range_m"]),
        str(params["cluster_head_ratio"]),
        str(params["eulc_candidate_ratio"]),
        str(pso_particles),
        str(pso_iterations),
        str(runs),
        str(seed_start),
    )


def stress_cases(profile: str) -> Iterable[Dict[str, object]]:
    base_100m = {
        "space_m": 100,
        "node_count": 300,
        "packet_size_bits": 6400,
        "initial_energy_j": 0.5,
        "pso_c1": 1.5,
        "pso_c2": 1.5,
        "pso_omega": 0.65,
        "transmission_range_m": 150.0,
        "cluster_head_ratio": 0.2,
        "eulc_candidate_ratio": 0.4,
    }
    base_500m = {
        "space_m": 500,
        "pso_c1": 1.5,
        "pso_c2": 1.5,
        "pso_omega": 0.65,
        "transmission_range_m": 150.0,
        "cluster_head_ratio": 0.2,
        "eulc_candidate_ratio": 0.4,
    }

    if profile in {"all", "param"}:
        for omega in (0.35, 0.5, 0.65, 0.8):
            yield {**base_100m, "suite": "omega_stress_100m_v2", "pso_omega": omega}

        for candidate_ratio in (0.2, 0.3, 0.4, 0.5):
            yield {
                **base_100m,
                "suite": "candidate_stress_100m_v2",
                "eulc_candidate_ratio": candidate_ratio,
            }

        for cluster_head_ratio in (0.1, 0.15, 0.2, 0.25, 0.3):
            yield {
                **base_100m,
                "suite": "ch_ratio_stress_100m_v2",
                "cluster_head_ratio": cluster_head_ratio,
            }

        for transmission_range in (100.0, 150.0, 200.0, 250.0):
            yield {
                **base_100m,
                "suite": "range_stress_100m_v2",
                "transmission_range_m": transmission_range,
            }

    if profile in {"all", "scale"}:
        for node_count in (1000, 1500, 2000, 2500):
            for packet_size, initial_energy in ((4000, 0.5), (4000, 1.0), (6400, 0.5), (6400, 1.0)):
                yield {
                    **base_500m,
                    "suite": "space500_scale_stress_v2",
                    "node_count": node_count,
                    "packet_size_bits": packet_size,
                    "initial_energy_j": initial_energy,
                }


def run_case(
    *,
    suite: str,
    space_m: int,
    node_count: int,
    packet_size_bits: int,
    initial_energy_j: float,
    pso_c1: float,
    pso_c2: float,
    pso_omega: float,
    transmission_range_m: float,
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
            f"{suite}_s{space_m}_n{node_count}_pkt{packet_size_bits}_e{initial_energy_j}"
            f"_c{pso_c1}_{pso_c2}_om{pso_omega}_r{transmission_range_m}_cand{eulc_candidate_ratio}"
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
        pso_omega=pso_omega,
        transmission_range_m=transmission_range_m,
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
                "suite": suite,
                "space_m": space_m,
                "node_count": node_count,
                "packet_size_bits": packet_size_bits,
                "initial_energy_j": initial_energy_j,
                "pso_c1": pso_c1,
                "pso_c2": pso_c2,
                "pso_omega": pso_omega,
                "transmission_range_m": transmission_range_m,
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
        FIELDS,
        {
            "suite": suite,
            "space_m": space_m,
            "node_count": node_count,
            "packet_size_bits": packet_size_bits,
            "initial_energy_j": initial_energy_j,
            "pso_c1": pso_c1,
            "pso_c2": pso_c2,
            "pso_omega": pso_omega,
            "transmission_range_m": transmission_range_m,
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
            "packets_received_mean": round(mean([run.packets_received for run in sim_runs]), 4),
            "elapsed_sec": round(elapsed, 2),
        },
    )
    print(
        f"[{suite}] space={space_m}, n={node_count}, pkt={packet_size_bits}, e={initial_energy_j}, "
        f"omega={pso_omega}, R={transmission_range_m}, cand={eulc_candidate_ratio} -> "
        f"FND={fnd_mean:.2f} +/- {fnd_std:.2f}, CV={fnd_cv_pct:.2f}%, t={elapsed:.2f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["all", "param", "scale"], default="all")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=BASE_SEED)
    parser.add_argument("--max-rounds", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pso-particles", type=int, default=4)
    parser.add_argument("--pso-iterations", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    existing = _existing_keys(SUMMARY_LOG)
    pending = []
    for params in stress_cases(args.profile):
        key = _params_key(params, args.runs, args.seed_start, args.pso_particles, args.pso_iterations)
        if key not in existing:
            pending.append(params)

    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        f"Stability stress v2: profile={args.profile}, pending={len(pending)}, "
        f"runs/case={args.runs}, seeds={args.seed_start}-{args.seed_start + args.runs - 1}",
        flush=True,
    )

    for params in pending:
        run_case(
            suite=str(params["suite"]),
            space_m=int(params["space_m"]),
            node_count=int(params["node_count"]),
            packet_size_bits=int(params["packet_size_bits"]),
            initial_energy_j=float(params["initial_energy_j"]),
            pso_c1=float(params["pso_c1"]),
            pso_c2=float(params["pso_c2"]),
            pso_omega=float(params["pso_omega"]),
            transmission_range_m=float(params["transmission_range_m"]),
            cluster_head_ratio=float(params["cluster_head_ratio"]),
            eulc_candidate_ratio=float(params["eulc_candidate_ratio"]),
            pso_particles=args.pso_particles,
            pso_iterations=args.pso_iterations,
            runs=args.runs,
            seed_start=args.seed_start,
            max_rounds=args.max_rounds,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()

