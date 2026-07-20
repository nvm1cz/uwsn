from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

from uwsn.metrics import RunMetrics, summarize_runs
from uwsn.run_batch import run_simulation_batch
from uwsn.run_config import BASE_SEED, SIMULATION_CASE, SIMULATION_PARAMS


SUMMARY_LOG = Path("outputs/data/csv/summary/extended_sensitivity_v2_summary.csv")
RAW_LOG = Path("outputs/data/csv/raw/extended_sensitivity_v2_raw.csv")


SUMMARY_FIELDS = [
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
    "hnd_mean",
    "lnd_mean",
    "first_5pct_mean",
    "residual_energy_pct_at_stop",
    "dead_nodes_pct_at_stop",
    "packets_received_mean",
    "alive_nodes_mean",
    "censored_runs",
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
    "hnd_round",
    "lnd_round",
    "first_5pct_round",
    "residual_energy",
    "alive_nodes",
    "packets_received",
]


def _migrate_csv_columns(path: Path, fieldnames: List[str], defaults: Dict[str, object]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        if old_fields == fieldnames:
            return
        rows = list(reader)
    path.with_suffix(path.suffix + ".bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            migrated = {field: row.get(field, defaults.get(field, "")) for field in fieldnames}
            writer.writerow(migrated)


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
        row.get("eulc_candidate_ratio", "0.2"),
        row["pso_particles"],
        row["pso_iterations"],
        row["runs"],
        row["seed_start"],
    )


def _existing_summary_keys(path: Path) -> set[Tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {_case_key(row) for row in csv.DictReader(f)}


def _key_from_params(
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
) -> Tuple[str, ...]:
    return (
        suite,
        str(space_m),
        str(node_count),
        str(packet_size_bits),
        str(initial_energy_j),
        str(pso_c1),
        str(pso_c2),
        str(pso_omega),
        str(transmission_range_m),
        str(cluster_head_ratio),
        str(eulc_candidate_ratio),
        str(pso_particles),
        str(pso_iterations),
        str(runs),
        str(seed_start),
    )


def _fmt_metric(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


def _run_case(
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
            f"_c{pso_c1}_{pso_c2}_om{pso_omega}_r{transmission_range_m}_ch{cluster_head_ratio}"
            f"_cand{eulc_candidate_ratio}"
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

    start = perf_counter()
    runs_out = run_simulation_batch(
        case,
        params,
        seeds,
        stop_on_first_dead=True,
        max_rounds=max_rounds,
        workers=workers,
    )
    elapsed = perf_counter() - start
    summary = summarize_runs(case, runs_out)

    for run in runs_out:
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
                "fnd_round": _fmt_metric(run.fnd_round),
                "hnd_round": _fmt_metric(run.hnd_round),
                "lnd_round": _fmt_metric(run.lnd_round),
                "first_5pct_round": _fmt_metric(run.first_5pct_round),
                "residual_energy": _fmt_metric(run.residual_energy),
                "alive_nodes": run.alive_nodes,
                "packets_received": run.packets_received,
            },
        )

    fnd_values = [run.fnd_round for run in runs_out if run.fnd_round is not None]
    censored_runs = len(runs_out) - len(fnd_values)
    summary_row = {
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
        "fnd_mean": _fmt_metric(summary.get("fnd_mean")),
        "fnd_std": _fmt_metric(pstdev(fnd_values) if len(fnd_values) > 1 else 0.0),
        "fnd_min": min(fnd_values) if fnd_values else "",
        "fnd_max": max(fnd_values) if fnd_values else "",
        "hnd_mean": _fmt_metric(summary.get("hnd_mean")),
        "lnd_mean": _fmt_metric(summary.get("lnd_mean")),
        "first_5pct_mean": _fmt_metric(summary.get("first_5pct_mean")),
        "residual_energy_pct_at_stop": _fmt_metric(summary.get("residual_energy_pct")),
        "dead_nodes_pct_at_stop": _fmt_metric(summary.get("dead_nodes_pct")),
        "packets_received_mean": _fmt_metric(mean([run.packets_received for run in runs_out]) if runs_out else 0),
        "alive_nodes_mean": _fmt_metric(mean([run.alive_nodes for run in runs_out]) if runs_out else 0),
        "censored_runs": censored_runs,
        "elapsed_sec": round(elapsed, 2),
    }
    _append_row(SUMMARY_LOG, SUMMARY_FIELDS, summary_row)
    print(
        f"[{suite}] space={space_m}, n={node_count}, pkt={packet_size_bits}, "
        f"e={initial_energy_j}, c=({pso_c1},{pso_c2}), omega={pso_omega}, "
        f"R={transmission_range_m}, Pc={cluster_head_ratio}, cand={eulc_candidate_ratio} -> "
        f"FND={summary_row['fnd_mean']} +/- {summary_row['fnd_std']} "
        f"(min={summary_row['fnd_min']}, max={summary_row['fnd_max']}), "
        f"censored={censored_runs}, t={summary_row['elapsed_sec']}s",
        flush=True,
    )


def _cases_100m_sensitivity() -> Iterable[Dict[str, object]]:
    space_m = 100
    node_counts = (100, 200, 300, 500)
    baseline = {
        "space_m": space_m,
        "packet_size_bits": 6400,
        "initial_energy_j": 0.5,
        "pso_c1": 1.5,
        "pso_c2": 1.5,
        "pso_omega": 0.65,
        "transmission_range_m": 150.0,
        "cluster_head_ratio": 0.2,
        "eulc_candidate_ratio": 0.2,
    }

    seen: set[Tuple[object, ...]] = set()

    def emit(suite: str, **kwargs: object) -> Iterable[Dict[str, object]]:
        for node_count in node_counts:
            params = {**baseline, **kwargs, "node_count": node_count, "suite": suite}
            key = tuple(params.items())
            if key in seen:
                continue
            seen.add(key)
            yield params

    for energy in (0.25, 0.5, 0.75, 1.0, 1.25):
        yield from emit("energy_sweep_100m_v2", initial_energy_j=energy)

    for packet_size in (3200, 4000, 5200, 6400, 8000):
        yield from emit("packet_sweep_100m_v2", packet_size_bits=packet_size)

    for c1, c2 in ((0.5, 2.5), (1.0, 2.0), (1.5, 1.5), (2.0, 1.0), (2.5, 0.5)):
        yield from emit("pso_coeff_sweep_100m_v2", pso_c1=c1, pso_c2=c2)

    for pso_omega in (0.35, 0.5, 0.65, 0.8):
        yield from emit("objective_weight_sweep_100m_v2", pso_omega=pso_omega)

    for transmission_range_m in (100.0, 150.0, 200.0, 250.0):
        yield from emit("range_sweep_100m_v2", transmission_range_m=transmission_range_m)

    for cluster_head_ratio in (0.1, 0.15, 0.2, 0.25, 0.3):
        yield from emit("ch_ratio_sweep_100m_v2", cluster_head_ratio=cluster_head_ratio)

    for candidate_ratio in (0.05, 0.1, 0.2, 0.3, 0.4):
        yield from emit(
            "candidate_ratio_sweep_100m_v2",
            cluster_head_ratio=candidate_ratio,
            eulc_candidate_ratio=candidate_ratio,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=BASE_SEED)
    parser.add_argument("--max-rounds", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pso-particles", type=int, default=4)
    parser.add_argument("--pso-iterations", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Run only first N pending cases (0 = all).")
    args = parser.parse_args()

    _migrate_csv_columns(SUMMARY_LOG, SUMMARY_FIELDS, {"eulc_candidate_ratio": "0.2"})
    _migrate_csv_columns(RAW_LOG, RAW_FIELDS, {"eulc_candidate_ratio": "0.2"})
    existing = _existing_summary_keys(SUMMARY_LOG)
    pending = []
    for params in _cases_100m_sensitivity():
        key = _key_from_params(
            str(params["suite"]),
            int(params["space_m"]),
            int(params["node_count"]),
            int(params["packet_size_bits"]),
            float(params["initial_energy_j"]),
            float(params["pso_c1"]),
            float(params["pso_c2"]),
            float(params["pso_omega"]),
            float(params["transmission_range_m"]),
            float(params["cluster_head_ratio"]),
            float(params["eulc_candidate_ratio"]),
            args.pso_particles,
            args.pso_iterations,
            args.runs,
            args.seed_start,
        )
        if key not in existing:
            pending.append(params)

    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        f"Extended sensitivity sweep: pending={len(pending)}, "
        f"runs/case={args.runs}, seeds={args.seed_start}-{args.seed_start + args.runs - 1}, "
        f"logs=({SUMMARY_LOG}, {RAW_LOG})",
        flush=True,
    )

    for params in pending:
        _run_case(
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

