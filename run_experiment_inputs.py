from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Sequence

import numpy as np

from uwsn.run_config import SIMULATION_CASE, SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


INPUT_FIELDS = [
    "input_id",
    "topology_id",
    "distribution",
    "optimizer",
    "parameter_set",
    "space_m",
    "node_count",
    "seed",
    "transmission_range_m",
    "packet_size_bits",
    "initial_energy_j",
    "max_rounds",
    "min_alive_ratio",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
    "pso_particles",
    "pso_iterations",
    "pso_inertia",
    "pso_c1",
    "pso_c2",
    "pso_omega",
    "contention_penalty_factor",
    "contention_reference_degree",
    "contention_max_multiplier",
]
OUTPUT_FIELDS = [
    *INPUT_FIELDS,
    "ft5_round",
    "fnd_round",
    "hnd_round",
    "lnd_round",
    "dead_nodes_at_stop",
    "alive_nodes_at_stop",
    "residual_energy_j",
    "residual_energy_pct",
    "packets_received",
    "recluster_count",
    "routing_edge_count",
    "runtime_sec",
]
CH_FIELDS = [
    "input_id",
    "round",
    "refresh_index",
    "ch_order",
    "ch_id",
    "x_m",
    "y_m",
    "depth_m",
    "final_energy_j",
]
MEMBER_FIELDS = [
    "input_id",
    "round",
    "refresh_index",
    "ch_id",
    "member_id",
    "is_ch",
]
CONVERGENCE_FIELDS = [
    *INPUT_FIELDS,
    "round",
    "refresh_index",
    "iteration",
    "best_score",
    "relative_best_score",
    "improvement_pct",
    "candidate_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulations from fixed experiment input CSVs.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/csv/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data/csv/output"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input-id", action="append", default=[])
    parser.add_argument("--distribution", action="append", default=[])
    parser.add_argument("--parameter-set", action="append", default=[])
    parser.add_argument("--transmission-range-m", type=float, action="append", default=[])
    parser.add_argument("--packet-size-bits", type=int, action="append", default=[])
    parser.add_argument("--initial-energy-j", type=float, action="append", default=[])
    parser.add_argument("--particles", type=int, action="append", default=[])
    parser.add_argument("--c1", type=float, action="append", default=[])
    parser.add_argument("--c2", type=float, action="append", default=[])
    parser.add_argument("--optimizer", action="append", default=[])
    parser.add_argument("--space", type=int, action="append", default=[])
    parser.add_argument("--node-count", type=int, action="append", default=[])
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--no-details", action="store_true")
    parser.add_argument("--no-convergence", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def filter_input_rows(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    distributions = {value.lower() for value in args.distribution}
    parameter_sets = {value.lower() for value in args.parameter_set}
    optimizers = {value.lower() for value in args.optimizer}
    spaces = set(args.space)
    node_counts = set(args.node_count)
    ranges = list(args.transmission_range_m)
    packet_sizes = set(args.packet_size_bits)
    initial_energies = list(args.initial_energy_j)
    particles = set(args.particles)
    c1_values = list(args.c1)
    c2_values = list(args.c2)
    seed_stop = None if args.seed_start is None or args.runs is None else args.seed_start + args.runs

    filtered: List[Dict[str, str]] = []
    for row in rows:
        if distributions and row["distribution"].lower() not in distributions:
            continue
        if parameter_sets and row["parameter_set"].lower() not in parameter_sets:
            continue
        if optimizers and row["optimizer"].lower() not in optimizers:
            continue
        if spaces and int(row["space_m"]) not in spaces:
            continue
        if node_counts and int(row["node_count"]) not in node_counts:
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
        filtered.append(row)
    return filtered


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_positions(node_rows: Sequence[Dict[str, str]]) -> Dict[str, np.ndarray]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in node_rows:
        grouped[row["topology_id"]].append(row)

    positions_by_topology: Dict[str, np.ndarray] = {}
    for topology_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["node_id"]))
        positions_by_topology[topology_id] = np.asarray(
            [
                [
                    float(row["x_m"]),
                    float(row["y_m"]),
                    float(row["depth_m"]),
                ]
                for row in rows
            ],
            dtype=float,
        )
    return positions_by_topology


def build_simulator(row: Dict[str, str], positions: np.ndarray, track_convergence: bool) -> PsoEulcSimulator:
    space_m = float(row["space_m"])
    case = replace(
        SIMULATION_CASE,
        name=str(row["input_id"]),
        node_count=int(row["node_count"]),
        packet_size_bits=int(float(row["packet_size_bits"])),
        initial_energy=float(row["initial_energy_j"]),
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=space_m,
        height_m=space_m,
        depth_m=space_m,
        transmission_range_m=float(row["transmission_range_m"]),
        alpha_weight=float(row["alpha_weight"]),
        beta_weight=float(row["beta_weight"]),
        gamma_weight=float(row["gamma_weight"]),
        cluster_head_ratio=float(row["cluster_head_ratio"]),
        eulc_candidate_ratio=float(row["eulc_candidate_ratio"]),
        pso_particles=int(float(row["pso_particles"])),
        pso_iterations=int(float(row["pso_iterations"])),
        pso_inertia=float(row["pso_inertia"]),
        pso_c1=float(row["pso_c1"]),
        pso_c2=float(row["pso_c2"]),
        pso_omega=float(row["pso_omega"]),
        optimizer=str(row["optimizer"]),
        contention_penalty_factor=float(row["contention_penalty_factor"]),
        contention_reference_degree=float(row["contention_reference_degree"]),
        contention_max_multiplier=float(row["contention_max_multiplier"]),
        deployment_distribution=str(row["distribution"]),
    )
    return PsoEulcSimulator(
        case,
        params,
        int(float(row["seed"])),
        verbose=False,
        track_pso_convergence=track_convergence,
        initial_positions=positions,
    )


def normalize_convergence(row: Dict[str, str], events: Sequence[Dict[str, float]]) -> List[Dict[str, object]]:
    initial_by_refresh: Dict[int, float] = {}
    normalized: List[Dict[str, object]] = []
    for event in sorted(
        events,
        key=lambda item: (int(item["refresh_index"]), int(item["iteration"])),
    ):
        refresh_index = int(event["refresh_index"])
        score = float(event["best_score"])
        if refresh_index not in initial_by_refresh and math.isfinite(score) and score > 0:
            initial_by_refresh[refresh_index] = score
        initial = initial_by_refresh.get(refresh_index)
        if initial is None or initial <= 0 or not math.isfinite(score):
            relative = ""
            improvement = ""
        else:
            relative_value = score / initial
            relative = round(relative_value, 8)
            improvement = round(max(0.0, (1.0 - relative_value) * 100.0), 4)
        normalized.append(
            {
                **row,
                "round": int(event["round"]),
                "refresh_index": refresh_index,
                "iteration": int(event["iteration"]),
                "best_score": round(score, 8),
                "relative_best_score": relative,
                "improvement_pct": improvement,
                "candidate_count": int(event["candidate_count"]),
            }
        )
    return normalized


def run_input(
    row: Dict[str, str],
    positions: np.ndarray,
    write_details: bool,
    track_convergence: bool,
) -> tuple[
    Dict[str, object],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    started = perf_counter()
    sim = build_simulator(row, positions, track_convergence)
    routing_edge_count = 0

    def on_round(current_sim: PsoEulcSimulator, round_idx: int, packets_received: int) -> None:
        nonlocal routing_edge_count
        routing_edge_count += len(current_sim.last_routing_edges)

    result = sim.run(
        stop_on_first_dead=False,
        stop_at_ft5=True,
        min_alive_ratio=None,
        max_rounds=int(float(row["max_rounds"])),
        round_callback=on_round,
    )
    elapsed = perf_counter() - started

    total_energy = float(row["initial_energy_j"]) * int(row["node_count"])
    output_row = {
        **row,
        "ft5_round": result.first_5pct_round or "",
        "fnd_round": result.fnd_round or "",
        "hnd_round": result.hnd_round or "",
        "lnd_round": result.lnd_round or "",
        "dead_nodes_at_stop": int(row["node_count"]) - result.alive_nodes,
        "alive_nodes_at_stop": result.alive_nodes,
        "residual_energy_j": round(result.residual_energy, 6),
        "residual_energy_pct": round((result.residual_energy / max(total_energy, 1e-9)) * 100.0, 4),
        "packets_received": result.packets_received,
        "recluster_count": len(sim.cluster_snapshots),
        "routing_edge_count": routing_edge_count,
        "runtime_sec": round(elapsed, 4),
    }

    ch_rows: List[Dict[str, object]] = []
    member_rows: List[Dict[str, object]] = []
    convergence_rows = normalize_convergence(row, result.pso_convergence) if track_convergence else []
    if write_details:
        for snapshot in sim.cluster_snapshots:
            round_idx = int(snapshot["round"])
            refresh_index = int(snapshot["refresh_index"])
            cluster_heads = list(snapshot["cluster_heads"])
            assignments = dict(snapshot["assignments"])
            for ch_order, ch_id in enumerate(cluster_heads):
                ch_id = int(ch_id)
                ch_rows.append(
                    {
                        "input_id": row["input_id"],
                        "round": round_idx,
                        "refresh_index": refresh_index,
                        "ch_order": ch_order,
                        "ch_id": ch_id,
                        "x_m": round(float(sim.positions[ch_id, 0]), 6),
                        "y_m": round(float(sim.positions[ch_id, 1]), 6),
                        "depth_m": round(float(sim.positions[ch_id, 2]), 6),
                        "final_energy_j": round(float(sim.energies[ch_id]), 6),
                    }
                )
                for member_id in assignments.get(ch_id, []):
                    member_rows.append(
                        {
                            "input_id": row["input_id"],
                            "round": round_idx,
                            "refresh_index": refresh_index,
                            "ch_id": ch_id,
                            "member_id": int(member_id),
                            "is_ch": int(int(member_id) == ch_id),
                        }
                    )

    return output_row, ch_rows, member_rows, convergence_rows


def main() -> None:
    args = parse_args()
    input_rows = load_rows(args.input_dir / "experiment_inputs.csv")
    node_rows = load_rows(args.input_dir / "experiment_nodes.csv")
    if not input_rows or not node_rows:
        raise SystemExit("Missing experiment input files. Run build_experiment_inputs.py first.")

    selected_ids = set(args.input_id)
    if selected_ids:
        input_rows = [row for row in input_rows if row["input_id"] in selected_ids]
    else:
        input_rows = filter_input_rows(input_rows, args)

    output_path = args.output_dir / "experiment_outputs.csv"
    existing_outputs = load_rows(output_path)
    completed_ids = {row["input_id"] for row in existing_outputs}
    if args.only_missing:
        input_rows = [row for row in input_rows if row["input_id"] not in completed_ids]
    if args.limit is not None:
        input_rows = input_rows[: args.limit]

    positions_by_topology = load_positions(node_rows)
    output_rows: List[Dict[str, object]] = [dict(row) for row in existing_outputs]
    ch_rows: List[Dict[str, object]] = load_rows(args.output_dir / "experiment_cluster_heads.csv")
    member_rows: List[Dict[str, object]] = load_rows(args.output_dir / "experiment_cluster_members.csv")
    write_details = not args.no_details
    track_convergence = not args.no_convergence
    convergence_rows: List[Dict[str, object]] = load_rows(args.output_dir / "experiment_convergence.csv")

    for index, row in enumerate(input_rows, start=1):
        positions = positions_by_topology[row["topology_id"]]
        output_row, input_ch_rows, input_member_rows, input_convergence_rows = run_input(
            row,
            positions,
            write_details,
            track_convergence,
        )
        output_rows = [existing for existing in output_rows if existing["input_id"] != row["input_id"]]
        output_rows.append(output_row)
        if track_convergence:
            convergence_rows = [
                existing
                for existing in convergence_rows
                if existing["input_id"] != row["input_id"]
            ]
            convergence_rows.extend(input_convergence_rows)
        ch_rows.extend(input_ch_rows)
        member_rows.extend(input_member_rows)

        write_rows(output_path, OUTPUT_FIELDS, sorted(output_rows, key=lambda item: str(item["input_id"])))
        if track_convergence:
            write_rows(
                args.output_dir / "experiment_convergence.csv",
                CONVERGENCE_FIELDS,
                sorted(
                    convergence_rows,
                    key=lambda item: (
                        str(item["input_id"]),
                        int(item["refresh_index"]),
                        int(item["iteration"]),
                    ),
                ),
            )
        if write_details:
            write_rows(args.output_dir / "experiment_cluster_heads.csv", CH_FIELDS, ch_rows)
            write_rows(args.output_dir / "experiment_cluster_members.csv", MEMBER_FIELDS, member_rows)
        print(
            f"[{index}/{len(input_rows)}] {row['input_id']} FT5={output_row['ft5_round']} "
            f"residual={output_row['residual_energy_pct']}%",
            flush=True,
        )

    print(f"Saved {output_path} ({len(output_rows)} rows)")
    if write_details:
        print(f"Saved {args.output_dir / 'experiment_cluster_heads.csv'}")
        print(f"Saved {args.output_dir / 'experiment_cluster_members.csv'}")
    if track_convergence:
        print(f"Saved {args.output_dir / 'experiment_convergence.csv'}")


if __name__ == "__main__":
    main()

