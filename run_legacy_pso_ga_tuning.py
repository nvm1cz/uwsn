from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from uwsn.run_config import SIMULATION_CASE, SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


SPACE_NODES = {
    100: (50, 100, 150),
    500: (100, 200, 300),
    1000: (500, 1000, 1500),
}
TOPOLOGIES = ("uniform", "gaussian", "exponential")
PACKET_ENERGY = ((6400, 1.0), (6400, 0.5), (4000, 1.0), (4000, 0.5))
CHECKPOINT_ROUNDS = (300, 700, 1000)


PARAMETER_SETS: dict[str, dict[str, Any]] = {
    "P1": {
        "optimizer": "pso", "population": 30, "iterations": 50,
        "pso_inertia": 0.7298, "pso_inertia_schedule": "fixed",
        "pso_inertia_start": 0.7298, "pso_inertia_end": 0.7298,
        "pso_c1": 1.49618, "pso_c1_schedule": "fixed",
        "pso_c1_start": 1.49618, "pso_c1_end": 1.49618,
        "pso_c2": 1.49618, "pso_c2_schedule": "fixed",
        "pso_c2_start": 1.49618, "pso_c2_end": 1.49618,
    },
    "P2": {
        "optimizer": "pso", "population": 30, "iterations": 50,
        "pso_inertia": 0.9, "pso_inertia_schedule": "linear",
        "pso_inertia_start": 0.9, "pso_inertia_end": 0.4,
        "pso_c1": 2.0, "pso_c1_schedule": "fixed",
        "pso_c1_start": 2.0, "pso_c1_end": 2.0,
        "pso_c2": 2.0, "pso_c2_schedule": "fixed",
        "pso_c2_start": 2.0, "pso_c2_end": 2.0,
    },
    "P3": {
        "optimizer": "pso", "population": 30, "iterations": 50,
        "pso_inertia": 0.9, "pso_inertia_schedule": "linear",
        "pso_inertia_start": 0.9, "pso_inertia_end": 0.4,
        "pso_c1": 2.5, "pso_c1_schedule": "linear",
        "pso_c1_start": 2.5, "pso_c1_end": 0.5,
        "pso_c2": 0.5, "pso_c2_schedule": "linear",
        "pso_c2_start": 0.5, "pso_c2_end": 2.5,
    },
    "P4": {
        "optimizer": "pso", "population": 50, "iterations": 30,
        "pso_inertia": 0.9, "pso_inertia_schedule": "linear",
        "pso_inertia_start": 0.9, "pso_inertia_end": 0.4,
        "pso_c1": 2.0, "pso_c1_schedule": "fixed",
        "pso_c1_start": 2.0, "pso_c1_end": 2.0,
        "pso_c2": 2.0, "pso_c2_schedule": "fixed",
        "pso_c2_start": 2.0, "pso_c2_end": 2.0,
    },
    "P5": {
        "optimizer": "pso", "population": 20, "iterations": 75,
        "pso_inertia": 0.9, "pso_inertia_schedule": "linear",
        "pso_inertia_start": 0.9, "pso_inertia_end": 0.4,
        "pso_c1": 2.0, "pso_c1_schedule": "fixed",
        "pso_c1_start": 2.0, "pso_c1_end": 2.0,
        "pso_c2": 2.0, "pso_c2_schedule": "fixed",
        "pso_c2_start": 2.0, "pso_c2_end": 2.0,
    },
    "G1": {"optimizer": "ga", "population": 30, "iterations": 50,
           "ga_crossover_rate": 0.8, "ga_mutation_rate": 0.01},
    "G2": {"optimizer": "ga", "population": 30, "iterations": 50,
           "ga_crossover_rate": 0.8, "ga_mutation_rate": 0.05},
    "G3": {"optimizer": "ga", "population": 30, "iterations": 50,
           "ga_crossover_rate": 0.9, "ga_mutation_rate": 0.10},
    "G4": {"optimizer": "ga", "population": 50, "iterations": 30,
           "ga_crossover_rate": 0.8, "ga_mutation_rate": 0.05},
    "G5": {"optimizer": "ga", "population": 20, "iterations": 75,
           "ga_crossover_rate": 0.8, "ga_mutation_rate": 0.05},
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _scenario_id(space: int, nodes: int, topology: str, packet: int, energy: float) -> str:
    energy_text = str(float(energy)).replace(".", "p")
    return f"s{space}_n{nodes}_{topology}_p{packet}_e{energy_text}"


def _jobs(seeds: int) -> Iterable[dict[str, Any]]:
    for space, node_counts in SPACE_NODES.items():
        for nodes in node_counts:
            for topology in TOPOLOGIES:
                for packet, energy in PACKET_ENERGY:
                    scenario = _scenario_id(space, nodes, topology, packet, energy)
                    for parameter_id, config in PARAMETER_SETS.items():
                        for seed in range(seeds):
                            yield {
                                "key": f"{scenario}|{parameter_id}|{seed}",
                                "scenario_id": scenario,
                                "space_m": space,
                                "node_count": nodes,
                                "topology": topology,
                                "packet_size_bits": packet,
                                "initial_energy_j": energy,
                                "parameter_id": parameter_id,
                                "parameter_config": config,
                                "seed": seed,
                            }


def _snapshot(sim: PsoEulcSimulator, round_index: int) -> dict[str, Any]:
    energies = sim.energies.copy()
    alive = energies > sim.params.dead_energy_threshold_j
    alive_count = int(alive.sum())
    minimum = float(energies[alive].min()) if alive_count else 0.0
    average = float(energies.mean())
    std = float(energies.std())
    balance = max(0.0, 1.0 - std / max(average, 1e-12))
    return {
        "round": round_index,
        "minimum_energy_j": minimum,
        "mean_energy_j": average,
        "energy_balance": balance,
        "alive_nodes": alive_count,
        "residual_energy_j": float(energies.sum()),
    }


def _run(job: dict[str, Any]) -> dict[str, Any]:
    config = job["parameter_config"]
    case = replace(
        SIMULATION_CASE,
        name=job["scenario_id"],
        node_count=int(job["node_count"]),
        packet_size_bits=int(job["packet_size_bits"]),
        initial_energy=float(job["initial_energy_j"]),
        rounds=1000,
    )
    updates = {
        "width_m": float(job["space_m"]),
        "height_m": float(job["space_m"]),
        "depth_m": float(job["space_m"]),
        "deployment_distribution": str(job["topology"]),
        "optimizer": str(config["optimizer"]),
        "pso_particles": int(config["population"]),
        "pso_iterations": int(config["iterations"]),
    }
    for key, value in config.items():
        if key not in {"optimizer", "population", "iterations"}:
            updates[key] = value
    params = replace(SIMULATION_PARAMS, **updates)
    snapshots: dict[int, dict[str, Any]] = {}

    def capture(simulator: PsoEulcSimulator, round_index: int, _: int) -> None:
        if round_index in CHECKPOINT_ROUNDS:
            snapshots[round_index] = _snapshot(simulator, round_index)

    started = time.perf_counter()
    simulator = PsoEulcSimulator(
        case, params, int(job["seed"]), verbose=False,
        track_pso_convergence=False,
    )
    metrics = simulator.run(
        stop_on_first_dead=False,
        max_rounds=1000,
        round_callback=capture,
    )
    runtime = time.perf_counter() - started
    terminal = _snapshot(simulator, len(metrics.residual_energy_by_round))
    for checkpoint in CHECKPOINT_ROUNDS:
        snapshots.setdefault(checkpoint, {**terminal, "round": checkpoint})
    alive_curve = [case.node_count - dead for dead in metrics.dead_nodes_by_round]
    if len(alive_curve) < 1000:
        alive_curve.extend([alive_curve[-1] if alive_curve else 0] * (1000 - len(alive_curve)))
    alive_auc = float(sum(alive_curve[:1000])) / float(case.node_count * 1000)
    return {
        **{key: value for key, value in job.items() if key != "parameter_config"},
        "optimizer": config["optimizer"],
        "population": config["population"],
        "iterations": config["iterations"],
        "runtime_seconds": runtime,
        "rounds_executed": len(metrics.residual_energy_by_round),
        "fnd_round": metrics.fnd_round,
        "hnd_round": metrics.hnd_round,
        "lnd_round": metrics.lnd_round,
        "alive_auc_1000": alive_auc,
        "snapshots": {str(key): value for key, value in snapshots.items()},
        "parameter_config": config,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((row["scenario_id"], row["parameter_id"]), []).append(row)
    aggregate: list[dict[str, Any]] = []
    for (scenario, parameter_id), rows in grouped.items():
        config = rows[0]["parameter_config"]
        item: dict[str, Any] = {
            "scenario_id": scenario,
            "parameter_id": parameter_id,
            "optimizer": config["optimizer"],
            "runs": len(rows),
            "runtime_mean_seconds": mean(row["runtime_seconds"] for row in rows),
            "parameter_config": json.dumps(config, sort_keys=True),
        }
        for checkpoint in CHECKPOINT_ROUNDS:
            snapshots = [row["snapshots"][str(checkpoint)] for row in rows]
            for metric in ("minimum_energy_j", "mean_energy_j", "energy_balance", "alive_nodes", "residual_energy_j"):
                values = [float(snapshot[metric]) for snapshot in snapshots]
                item[f"r{checkpoint}_{metric}_mean"] = mean(values)
                item[f"r{checkpoint}_{metric}_std"] = pstdev(values)
        for event in ("fnd_round", "hnd_round", "lnd_round"):
            values = [float(row[event] if row[event] is not None else 1001) for row in rows]
            item[f"{event}_mean_censored1001"] = mean(values)
            item[f"{event}_std_censored1001"] = pstdev(values)
        item["alive_auc_1000_mean"] = mean(row["alive_auc_1000"] for row in rows)
        # User-defined 300-round predictor; every term is dimensionless.
        initial = float(rows[0]["initial_energy_j"])
        nodes = int(rows[0]["node_count"])
        item["score_300"] = (
            item["r300_minimum_energy_j_mean"] / max(initial, 1e-12)
            + item["r300_mean_energy_j_mean"] / max(initial, 1e-12)
            + item["r300_energy_balance_mean"]
            + item["r300_alive_nodes_mean"] / max(nodes, 1)
        )
        aggregate.append(item)

    stage300: list[dict[str, Any]] = []
    stage700: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    scenarios = sorted({row["scenario_id"] for row in aggregate})
    for scenario in scenarios:
        selected300: dict[str, list[dict[str, Any]]] = {}
        for optimizer in ("pso", "ga"):
            candidates = [
                row for row in aggregate
                if row["scenario_id"] == scenario and row["optimizer"] == optimizer
            ]
            candidates.sort(key=lambda row: (
                -row["score_300"], row["runtime_mean_seconds"], row["parameter_id"]
            ))
            for rank, row in enumerate(candidates, 1):
                stage300.append({**row, "rank_300": rank, "kept_top3": rank <= 3})
            selected300[optimizer] = candidates[:3]

        winners700: dict[str, dict[str, Any]] = {}
        for optimizer, candidates in selected300.items():
            candidates.sort(key=lambda row: (
                -row["fnd_round_mean_censored1001"],
                -row["r700_alive_nodes_mean"],
                -row["r700_energy_balance_mean"],
                row["runtime_mean_seconds"],
            ))
            for rank, row in enumerate(candidates, 1):
                stage700.append({**row, "rank_700": rank, "kept_top1": rank == 1})
            winners700[optimizer] = candidates[0]

        finalists = []
        for optimizer, row in winners700.items():
            lifetime_score = (
                row["fnd_round_mean_censored1001"] / 1000.0
                + row["hnd_round_mean_censored1001"] / 1000.0
                + row["lnd_round_mean_censored1001"] / 1000.0
                + row["alive_auc_1000_mean"]
            )
            finalists.append({**row, "lifetime_score_1000": lifetime_score})
        finalists.sort(key=lambda row: (
            -row["lifetime_score_1000"], row["runtime_mean_seconds"]
        ))
        for rank, row in enumerate(finalists, 1):
            final.append({**row, "final_rank": rank, "scenario_winner": rank == 1})
    return stage300, stage700, final


def _finalize(output: Path, results: list[dict[str, Any]]) -> None:
    stage300, stage700, final = _aggregate(results)
    _write_csv(output / "stage300_rankings.csv", stage300)
    _write_csv(output / "stage700_rankings.csv", stage700)
    _write_csv(output / "final_best_configs.csv", final)
    best = [row for row in final if row["final_rank"] == 1]
    _write_csv(output / "best_algorithm_by_scenario.csv", best)
    _atomic_json(output / "final_summary.json", {
        "schema_version": 1,
        "scenario_count": len({row['scenario_id'] for row in results}),
        "raw_run_count": len(results),
        "best": best,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "trials.jsonl"
    results: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                results[row["key"]] = row
    all_jobs = list(_jobs(args.seeds))
    missing = [job for job in all_jobs if job["key"] not in results]
    if args.max_runs is not None:
        missing = missing[: args.max_runs]
    manifest = {
        "schema_version": 1,
        "total_runs": len(all_jobs),
        "completed_runs": len(results),
        "remaining_runs": len(all_jobs) - len(results),
        "workers": args.workers,
        "seeds": args.seeds,
        "status": "running",
    }
    _atomic_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        iterator = iter(missing)
        pending = {}
        while len(pending) < args.workers:
            try:
                job = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(_run, job)] = job
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                row = future.result()
                results[row["key"]] = row
                _append_jsonl(raw_path, row)
                completed = len(results)
                elapsed = time.perf_counter() - started
                line = (
                    f"[{completed}/{len(all_jobs)}] {row['scenario_id']} "
                    f"{row['parameter_id']} seed={row['seed']} "
                    f"runtime={row['runtime_seconds']:.2f}s elapsed={elapsed:.1f}s"
                )
                print(line, flush=True)
                with (output / "progress.log").open("a", encoding="utf-8") as log:
                    log.write(line + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                _atomic_json(output / "manifest.json", {
                    **manifest,
                    "completed_runs": completed,
                    "remaining_runs": len(all_jobs) - completed,
                    "last_completed_key": row["key"],
                    "status": "running" if completed < len(all_jobs) else "complete",
                })
                try:
                    next_job = next(iterator)
                except StopIteration:
                    continue
                pending[executor.submit(_run, next_job)] = next_job
    if len(results) == len(all_jobs):
        _finalize(output, list(results.values()))
    print(json.dumps({
        "completed_runs": len(results),
        "total_runs": len(all_jobs),
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
