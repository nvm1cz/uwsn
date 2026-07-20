from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from run_experiment_inputs import (
    CONVERGENCE_FIELDS,
    build_simulator,
    load_positions,
    load_rows,
    normalize_convergence,
)


NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
}
DEFAULT_DISTRIBUTIONS = ("uniform", "gaussian", "exponential")
DEFAULT_OPTIMIZERS = ("pso", "ga", "de")
DEFAULT_SEEDS = (4000, 4001, 4002)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight representative optimizer convergence only."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/csv/input"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data/csv/raw/s100_s500_representative_optimizer_convergence.csv"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--space", type=int, action="append", default=[])
    parser.add_argument("--node-count", type=int, action="append", default=[])
    parser.add_argument("--distribution", action="append", default=[])
    parser.add_argument("--optimizer", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    return parser.parse_args()


def _matches(row: Dict[str, str], args: argparse.Namespace) -> bool:
    space_m = int(float(row["space_m"]))
    node_count = int(float(row["node_count"]))
    spaces = set(args.space) if args.space else set(NODE_SETS)
    node_counts = set(args.node_count) if args.node_count else set(NODE_SETS.get(space_m, ()))
    distributions = set(args.distribution) if args.distribution else set(DEFAULT_DISTRIBUTIONS)
    optimizers = {value.lower() for value in args.optimizer} if args.optimizer else set(DEFAULT_OPTIMIZERS)
    seeds = set(args.seed) if args.seed else set(DEFAULT_SEEDS)

    return (
        space_m in spaces
        and node_count in node_counts
        and node_count in NODE_SETS.get(space_m, ())
        and row["distribution"] in distributions
        and row["optimizer"].lower() in optimizers
        and row["parameter_set"] == "baseline"
        and int(float(row["seed"])) in seeds
        and abs(float(row["transmission_range_m"]) - 200.0) < 1e-9
        and int(float(row["packet_size_bits"])) == 6400
        and abs(float(row["initial_energy_j"]) - 0.5) < 1e-9
        and abs(float(row["cluster_head_ratio"]) - 0.2) < 1e-9
        and int(float(row["pso_particles"])) == 20
        and int(float(row["pso_iterations"])) == 50
    )


def select_rows(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    selected = [row for row in rows if _matches(row, args)]
    selected.sort(
        key=lambda row: (
            int(float(row["space_m"])),
            int(float(row["node_count"])),
            row["distribution"],
            int(float(row["seed"])),
            row["optimizer"],
        )
    )
    return selected


def run_one(row: Dict[str, str], positions: np.ndarray) -> List[Dict[str, object]]:
    sim = build_simulator(row, positions, track_convergence=True)
    # Only run the first cluster-head election. This records optimizer
    # convergence without simulating all communication rounds to FT5.
    sim._refresh_clusters(round_idx=1)
    return normalize_convergence(row, sim.pso_convergence)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CONVERGENCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_rows = load_rows(args.input_dir / "experiment_inputs.csv")
    node_rows = load_rows(args.input_dir / "experiment_nodes.csv")
    positions_by_topology = load_positions(node_rows)

    selected = select_rows(input_rows, args)
    if not selected:
        raise SystemExit("No matching representative input rows.")

    start = time.perf_counter()
    workers = max(1, min(int(args.workers), len(selected)))
    print(f"Selected inputs: {len(selected)}")
    print(f"Workers: {workers}")

    all_rows: List[Dict[str, object]] = []
    if workers == 1:
        for idx, row in enumerate(selected, start=1):
            rows = run_one(row, positions_by_topology[row["topology_id"]])
            all_rows.extend(rows)
            print(f"[{idx}/{len(selected)}] {row['input_id']} convergence_rows={len(rows)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_row = {
                pool.submit(run_one, row, positions_by_topology[row["topology_id"]]): row
                for row in selected
            }
            done = 0
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                rows = future.result()
                done += 1
                all_rows.extend(rows)
                print(f"[{done}/{len(selected)}] {row['input_id']} convergence_rows={len(rows)}")

    all_rows.sort(
        key=lambda row: (
            int(float(row["space_m"])),
            int(float(row["node_count"])),
            str(row["distribution"]),
            int(float(row["seed"])),
            str(row["optimizer"]),
            int(row["refresh_index"]),
            int(row["iteration"]),
        )
    )
    write_csv(args.output, all_rows)
    elapsed = time.perf_counter() - start
    print(f"Saved: {args.output}")
    print(f"Rows: {len(all_rows)}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
