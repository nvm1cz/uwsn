from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

from run_experiment_inputs import (
    CH_FIELDS,
    CONVERGENCE_FIELDS,
    MEMBER_FIELDS,
    OUTPUT_FIELDS,
    load_positions,
    load_rows,
    run_input,
    write_rows,
)


POSITIONS_BY_TOPOLOGY = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed experiment inputs in parallel.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/csv/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data/csv/output"))
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--input-id", action="append", default=[])
    parser.add_argument("--input-id-file", type=Path, action="append", default=[])
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
    parser.add_argument("--preserve-input-order", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--convergence", action="store_true")
    return parser.parse_args()


def load_input_id_filters(args: argparse.Namespace) -> set[str]:
    selected_ids = {str(value).strip() for value in args.input_id if str(value).strip()}
    for path in args.input_id_file:
        if not path.exists():
            raise SystemExit(f"Missing input-id file: {path}")
        with path.open("r", encoding="utf-8", newline="") as file:
            sample = file.read(2048)
            file.seek(0)
            if "input_id" in sample.splitlines()[0]:
                reader = csv.DictReader(file)
                selected_ids.update(str(row["input_id"]).strip() for row in reader if row.get("input_id"))
            else:
                selected_ids.update(line.strip() for line in file if line.strip())
    return selected_ids


def filter_rows(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    selected_ids = load_input_id_filters(args)
    distributions = {value.lower() for value in args.distribution}
    spaces = set(args.space)
    node_counts = set(args.node_count)
    optimizers = {value.lower() for value in args.optimizer}
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
        if selected_ids and row["input_id"] not in selected_ids:
            continue
        if distributions and row["distribution"].lower() not in distributions:
            continue
        if spaces and int(row["space_m"]) not in spaces:
            continue
        if node_counts and int(row["node_count"]) not in node_counts:
            continue
        if optimizers and row["optimizer"].lower() not in optimizers:
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


def init_worker(input_dir: str) -> None:
    global POSITIONS_BY_TOPOLOGY
    node_rows = load_rows(Path(input_dir) / "experiment_nodes.csv")
    POSITIONS_BY_TOPOLOGY = load_positions(node_rows)


def run_one(
    row: Dict[str, str],
    write_details: bool,
    track_convergence: bool,
) -> Dict[str, object]:
    positions = POSITIONS_BY_TOPOLOGY[row["topology_id"]]
    output_row, ch_rows, member_rows, convergence_rows = run_input(
        row,
        positions,
        write_details=write_details,
        track_convergence=track_convergence,
    )
    return {
        "output": output_row,
        "cluster_heads": ch_rows,
        "cluster_members": member_rows,
        "convergence": convergence_rows,
    }


def save_outputs(path: Path, rows_by_id: Dict[str, Dict[str, object]]) -> None:
    rows = sorted(rows_by_id.values(), key=lambda item: str(item["input_id"]))
    write_rows(path, OUTPUT_FIELDS, rows)


def append_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def interleave_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped = defaultdict(deque)
    for row in rows:
        key = (
            row["distribution"],
            int(row["space_m"]),
            int(row["node_count"]),
            int(float(row["seed"])),
        )
        grouped[key].append(row)

    queues = deque(grouped[key] for key in sorted(grouped))
    ordered: List[Dict[str, str]] = []
    while queues:
        queue = queues.popleft()
        ordered.append(queue.popleft())
        if queue:
            queues.append(queue)
    return ordered


def main() -> None:
    args = parse_args()
    input_path = args.input_dir / "experiment_inputs.csv"
    output_path = args.output_dir / "experiment_outputs.csv"
    input_rows = load_rows(input_path)
    if not input_rows:
        raise SystemExit(f"Missing {input_path}")

    input_rows = filter_rows(input_rows, args)
    existing_rows = load_rows(output_path)
    rows_by_id: Dict[str, Dict[str, object]] = {row["input_id"]: dict(row) for row in existing_rows}
    pending_rows = [row for row in input_rows if row["input_id"] not in rows_by_id]
    if not args.preserve_input_order:
        pending_rows = interleave_rows(pending_rows)
    if args.limit is not None:
        pending_rows = pending_rows[: args.limit]

    selected_ids = {row["input_id"] for row in input_rows}
    completed_in_scope = sum(1 for input_id in selected_ids if input_id in rows_by_id)
    print(f"Completed before run: {completed_in_scope}/{len(input_rows)}")
    print(f"Pending in selected scope: {len(pending_rows)}")
    if not pending_rows:
        save_outputs(output_path, rows_by_id)
        print(f"Saved {output_path} ({len(rows_by_id)} rows)")
        return

    done = 0
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=init_worker,
        initargs=(str(args.input_dir),),
    ) as executor:
        futures = {
            executor.submit(
                run_one,
                row,
                args.details,
                args.convergence,
            ): row["input_id"]
            for row in pending_rows
        }
        for future in as_completed(futures):
            input_id = futures[future]
            result = future.result()
            output_row = result["output"]
            if args.details:
                append_rows(args.output_dir / "experiment_cluster_heads.csv", CH_FIELDS, result["cluster_heads"])
                append_rows(args.output_dir / "experiment_cluster_members.csv", MEMBER_FIELDS, result["cluster_members"])
            if args.convergence:
                append_rows(args.output_dir / "experiment_convergence.csv", CONVERGENCE_FIELDS, result["convergence"])
            rows_by_id[input_id] = output_row
            done += 1
            print(
                f"[{done}/{len(pending_rows)}] {input_id} FT5={output_row['ft5_round']} "
                f"runtime={output_row['runtime_sec']}s",
                flush=True,
            )
            if done % max(1, args.flush_every) == 0:
                save_outputs(output_path, rows_by_id)

    save_outputs(output_path, rows_by_id)
    print(f"Saved {output_path} ({len(rows_by_id)} rows)")


if __name__ == "__main__":
    main()
