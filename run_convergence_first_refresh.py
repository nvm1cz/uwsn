from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

from run_experiment_inputs import (
    CONVERGENCE_FIELDS,
    build_simulator,
    load_positions,
    load_rows,
    normalize_convergence,
    write_rows,
)


ITERATIVE_OPTIMIZERS = {"pso", "ga", "de"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record first cluster-head selection convergence without running full simulations."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/csv/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data/csv/output"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--distribution", action="append", default=[])
    parser.add_argument("--space", type=int, action="append", default=[])
    parser.add_argument("--node-count", type=int, action="append", default=[])
    parser.add_argument("--optimizer", action="append", default=[])
    parser.add_argument("--parameter-set", action="append", default=[])
    parser.add_argument("--transmission-range-m", type=float, action="append", default=[])
    parser.add_argument("--packet-size-bits", type=int, action="append", default=[])
    parser.add_argument("--initial-energy-j", type=float, action="append", default=[])
    parser.add_argument("--particles", type=int, action="append", default=[])
    parser.add_argument("--c1", type=float, action="append", default=[])
    parser.add_argument("--c2", type=float, action="append", default=[])
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def filter_rows(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    distributions = {value.lower() for value in args.distribution}
    spaces = set(args.space)
    node_counts = set(args.node_count)
    optimizers = {value.lower() for value in args.optimizer} or ITERATIVE_OPTIMIZERS
    parameter_sets = {value.lower() for value in args.parameter_set}
    ranges = list(args.transmission_range_m)
    packet_sizes = set(args.packet_size_bits)
    initial_energies = list(args.initial_energy_j)
    particles = set(args.particles)
    c1_values = list(args.c1)
    c2_values = list(args.c2)
    seed_stop = None if args.seed_start is None or args.runs is None else args.seed_start + args.runs

    selected = []
    for row in rows:
        if row["optimizer"].lower() not in optimizers:
            continue
        if distributions and row["distribution"].lower() not in distributions:
            continue
        if spaces and int(row["space_m"]) not in spaces:
            continue
        if node_counts and int(row["node_count"]) not in node_counts:
            continue
        if parameter_sets and row["parameter_set"].lower() not in parameter_sets:
            continue
        if ranges and not any(abs(float(row["transmission_range_m"]) - value) < 1e-9 for value in ranges):
            continue
        if packet_sizes and int(float(row["packet_size_bits"])) not in packet_sizes:
            continue
        if initial_energies and not any(abs(float(row["initial_energy_j"]) - value) < 1e-9 for value in initial_energies):
            continue
        if particles and int(float(row["pso_particles"])) not in particles:
            continue
        if c1_values and not any(abs(float(row["pso_c1"]) - value) < 1e-9 for value in c1_values):
            continue
        if c2_values and not any(abs(float(row["pso_c2"]) - value) < 1e-9 for value in c2_values):
            continue
        if args.seed_start is not None:
            seed = int(float(row["seed"]))
            if seed < args.seed_start:
                continue
            if seed_stop is not None and seed >= seed_stop:
                continue
        selected.append(row)
    return selected


def convergence_key(row: Dict[str, object]) -> tuple[str, int, int]:
    return (
        str(row["input_id"]),
        int(float(row.get("refresh_index", 1))),
        int(float(row.get("iteration", 0))),
    )


def save_convergence(path: Path, rows: List[Dict[str, object]]) -> None:
    rows = sorted(rows, key=convergence_key)
    write_rows(path, CONVERGENCE_FIELDS, rows)


def main() -> None:
    args = parse_args()
    input_rows = load_rows(args.input_dir / "experiment_inputs.csv")
    node_rows = load_rows(args.input_dir / "experiment_nodes.csv")
    if not input_rows or not node_rows:
        raise SystemExit("Missing experiment input files. Run build_experiment_inputs.py first.")

    selected_rows = filter_rows(input_rows, args)
    convergence_path = args.output_dir / "experiment_convergence.csv"
    convergence_rows: List[Dict[str, object]] = [dict(row) for row in load_rows(convergence_path)]
    completed_ids = {
        row["input_id"]
        for row in convergence_rows
        if row.get("optimizer", "").lower() in ITERATIVE_OPTIMIZERS
        and str(row.get("refresh_index", "1")) == "1"
    }
    pending_rows = [
        row
        for row in selected_rows
        if args.rerun or row["input_id"] not in completed_ids
    ]
    if args.limit is not None:
        pending_rows = pending_rows[: args.limit]

    positions_by_topology = load_positions(node_rows)
    print(f"Completed before run: {len(completed_ids & {row['input_id'] for row in selected_rows})}/{len(selected_rows)}")
    print(f"Pending convergence inputs: {len(pending_rows)}")

    done = 0
    for row in pending_rows:
        if args.rerun:
            convergence_rows = [existing for existing in convergence_rows if existing["input_id"] != row["input_id"]]
        sim = build_simulator(row, positions_by_topology[row["topology_id"]], track_convergence=True)
        sim._refresh_clusters(1)
        new_rows = normalize_convergence(row, sim.pso_convergence)
        convergence_rows.extend(new_rows)
        done += 1
        print(
            f"[{done}/{len(pending_rows)}] {row['input_id']} convergence_rows={len(new_rows)}",
            flush=True,
        )
        if done % max(1, args.flush_every) == 0:
            save_convergence(convergence_path, convergence_rows)

    save_convergence(convergence_path, convergence_rows)
    print(f"Saved {convergence_path} ({len(convergence_rows)} rows)")


if __name__ == "__main__":
    main()
