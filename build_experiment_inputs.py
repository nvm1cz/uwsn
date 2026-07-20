from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from uwsn.cases import SimulationCase
from uwsn.deployment import compute_layers, deploy_nodes
from uwsn.run_config import SIMULATION_CASE, SIMULATION_PARAMS


NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
    1000: (500, 1000, 1500, 2000),
}
DISTRIBUTIONS = ("uniform", "gaussian", "exponential")
TRANSMISSION_RANGES_M = (200.0,)
PACKET_SIZE_BITS = (6400,)
INITIAL_ENERGY_J = (0.5,)
PARTICLE_COUNTS = (20, 30)
ACCELERATION_COEFFICIENTS = ((1.5, 1.5),)
OPTIMIZERS = ("pso", "ga", "de", "leach", "eulc", "eeumc", "ebrec")
DIRECT_BASELINE_OPTIMIZERS = {"leach", "eulc", "eeumc", "ebrec"}

PARAMETER_SETS: Dict[str, Dict[str, float | int]] = {
    "baseline": {
        "alpha_weight": 0.60,
        "beta_weight": 0.25,
        "gamma_weight": 0.15,
        "pso_particles": 20,
        "pso_iterations": 50,
        "pso_inertia": 0.70,
        "pso_c1": 1.50,
        "pso_c2": 1.50,
        "pso_omega": 0.65,
        "cluster_head_ratio": 0.20,
        "eulc_candidate_ratio": 0.40,
    },
    "energy_focused": {
        "alpha_weight": 0.75,
        "beta_weight": 0.15,
        "gamma_weight": 0.10,
        "pso_particles": 20,
        "pso_iterations": 50,
        "pso_inertia": 0.70,
        "pso_c1": 1.50,
        "pso_c2": 1.50,
        "pso_omega": 0.50,
        "cluster_head_ratio": 0.20,
        "eulc_candidate_ratio": 0.40,
    },
    "distance_focused": {
        "alpha_weight": 0.35,
        "beta_weight": 0.45,
        "gamma_weight": 0.20,
        "pso_particles": 20,
        "pso_iterations": 50,
        "pso_inertia": 0.70,
        "pso_c1": 1.50,
        "pso_c2": 1.50,
        "pso_omega": 0.80,
        "cluster_head_ratio": 0.20,
        "eulc_candidate_ratio": 0.40,
    },
    "balanced_low_omega": {
        "alpha_weight": 0.34,
        "beta_weight": 0.33,
        "gamma_weight": 0.33,
        "pso_particles": 20,
        "pso_iterations": 50,
        "pso_inertia": 0.50,
        "pso_c1": 1.50,
        "pso_c2": 1.50,
        "pso_omega": 0.40,
        "cluster_head_ratio": 0.20,
        "eulc_candidate_ratio": 0.40,
    },
}

TOPOLOGY_FIELDS = [
    "topology_id",
    "distribution",
    "space_m",
    "node_count",
    "seed",
    "width_m",
    "height_m",
    "depth_m",
    "sink_x_m",
    "sink_y_m",
    "sink_depth_m",
    "gaussian_std_fraction",
    "exponential_scale_fraction",
]
NODE_FIELDS = [
    "topology_id",
    "node_id",
    "x_m",
    "y_m",
    "depth_m",
    "initial_energy_j",
    "layer",
]
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
MANIFEST_FIELDS = ["metric", "value"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed experimental input CSVs from one-time random topologies."
    )
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--seed-start", type=int, default=4000)
    parser.add_argument("--max-rounds", type=int, default=5000)
    parser.add_argument("--min-alive-ratio", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=Path("data/csv/input"))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def topology_id(distribution: str, space_m: int, node_count: int, seed: int) -> str:
    return f"{distribution}_s{space_m}_n{node_count}_seed{seed}"


def input_id(
    topology: str,
    optimizer: str,
    parameter_set: str,
    transmission_range_m: float,
    packet_size_bits: int,
    initial_energy_j: float,
    particles: int,
    c1: float,
    c2: float,
) -> str:
    range_tag = str(transmission_range_m).replace(".", "p")
    energy_tag = str(initial_energy_j).replace(".", "p")
    c1_tag = str(c1).replace(".", "p")
    c2_tag = str(c2).replace(".", "p")
    return (
        f"{topology}_{optimizer}_{parameter_set}_r{range_tag}_pkt{packet_size_bits}"
        f"_e{energy_tag}_p{particles}_c{c1_tag}_{c2_tag}"
    )


def build_topology_rows(args: argparse.Namespace) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    topology_rows: List[Dict[str, object]] = []
    node_rows: List[Dict[str, object]] = []

    for distribution in DISTRIBUTIONS:
        for space_m, node_counts in NODE_SETS.items():
            for node_count in node_counts:
                case = replace(
                    SIMULATION_CASE,
                    name=f"input_{distribution}_s{space_m}_n{node_count}",
                    node_count=node_count,
                    packet_size_bits=6400,
                    initial_energy=0.5,
                )
                params = replace(
                    SIMULATION_PARAMS,
                    width_m=float(space_m),
                    height_m=float(space_m),
                    depth_m=float(space_m),
                    deployment_distribution=distribution,
                )
                sink = params.get_sink_position()
                for seed in range(args.seed_start, args.seed_start + args.runs):
                    tid = topology_id(distribution, space_m, node_count, seed)
                    rng = np.random.default_rng(seed)
                    positions = deploy_nodes(case, params, rng)
                    layers = compute_layers(positions, params)
                    topology_rows.append(
                        {
                            "topology_id": tid,
                            "distribution": distribution,
                            "space_m": space_m,
                            "node_count": node_count,
                            "seed": seed,
                            "width_m": space_m,
                            "height_m": space_m,
                            "depth_m": space_m,
                            "sink_x_m": round(float(sink[0]), 6),
                            "sink_y_m": round(float(sink[1]), 6),
                            "sink_depth_m": round(float(sink[2]), 6),
                            "gaussian_std_fraction": params.gaussian_std_fraction,
                            "exponential_scale_fraction": params.exponential_scale_fraction,
                        }
                    )
                    for node_id, position in enumerate(positions):
                        node_rows.append(
                            {
                                "topology_id": tid,
                                "node_id": node_id,
                                "x_m": round(float(position[0]), 6),
                                "y_m": round(float(position[1]), 6),
                                "depth_m": round(float(position[2]), 6),
                                "initial_energy_j": case.initial_energy,
                                "layer": int(layers[node_id]),
                            }
                        )

    return topology_rows, node_rows


def build_input_rows(args: argparse.Namespace, topology_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    input_rows: List[Dict[str, object]] = []
    for topology in topology_rows:
        tid = str(topology["topology_id"])
        for transmission_range_m in TRANSMISSION_RANGES_M:
            for packet_size_bits in PACKET_SIZE_BITS:
                for initial_energy_j in INITIAL_ENERGY_J:
                    for optimizer in OPTIMIZERS:
                        is_direct_baseline = optimizer in DIRECT_BASELINE_OPTIMIZERS
                        parameter_names = ("baseline",) if is_direct_baseline else tuple(PARAMETER_SETS)
                        for parameter_set in parameter_names:
                            values = PARAMETER_SETS[parameter_set]
                            particle_options = (int(values["pso_particles"]),) if is_direct_baseline else PARTICLE_COUNTS
                            c_options = (
                                (float(values["pso_c1"]), float(values["pso_c2"])),
                            ) if is_direct_baseline else ACCELERATION_COEFFICIENTS
                            for particles in particle_options:
                                for c1, c2 in c_options:
                                    input_rows.append(
                                        {
                                            "input_id": input_id(
                                                tid,
                                                optimizer,
                                                parameter_set,
                                                transmission_range_m,
                                                packet_size_bits,
                                                initial_energy_j,
                                                particles,
                                                c1,
                                                c2,
                                            ),
                                            "topology_id": tid,
                                            "distribution": topology["distribution"],
                                            "optimizer": optimizer,
                                            "parameter_set": parameter_set,
                                            "space_m": topology["space_m"],
                                            "node_count": topology["node_count"],
                                            "seed": topology["seed"],
                                            "transmission_range_m": transmission_range_m,
                                            "packet_size_bits": packet_size_bits,
                                            "initial_energy_j": initial_energy_j,
                                            "max_rounds": args.max_rounds,
                                            "min_alive_ratio": args.min_alive_ratio,
                                            "alpha_weight": values["alpha_weight"],
                                            "beta_weight": values["beta_weight"],
                                            "gamma_weight": values["gamma_weight"],
                                            "cluster_head_ratio": values["cluster_head_ratio"],
                                            "eulc_candidate_ratio": values["eulc_candidate_ratio"],
                                            "pso_particles": particles,
                                            "pso_iterations": values["pso_iterations"],
                                            "pso_inertia": values["pso_inertia"],
                                            "pso_c1": c1,
                                            "pso_c2": c2,
                                            "pso_omega": values["pso_omega"],
                                            "contention_penalty_factor": 0.01,
                                            "contention_reference_degree": 10.0,
                                            "contention_max_multiplier": 4.0,
                                        }
                                    )
    return input_rows


def main() -> None:
    args = parse_args()
    topology_rows, node_rows = build_topology_rows(args)
    input_rows = build_input_rows(args, topology_rows)
    manifest_rows = [
        {"metric": "distributions", "value": ",".join(DISTRIBUTIONS)},
        {"metric": "spaces_m", "value": ",".join(str(space) for space in NODE_SETS)},
        {"metric": "node_set_count", "value": sum(len(nodes) for nodes in NODE_SETS.values())},
        {"metric": "runs_per_distribution_space_node", "value": args.runs},
        {"metric": "fixed_topology_count", "value": len(topology_rows)},
        {"metric": "fixed_node_coordinate_rows", "value": len(node_rows)},
        {"metric": "transmission_range_count", "value": len(TRANSMISSION_RANGES_M)},
        {"metric": "packet_size_bits", "value": ",".join(str(value) for value in PACKET_SIZE_BITS)},
        {"metric": "packet_size_count", "value": len(PACKET_SIZE_BITS)},
        {"metric": "initial_energy_j", "value": ",".join(str(value) for value in INITIAL_ENERGY_J)},
        {"metric": "initial_energy_count", "value": len(INITIAL_ENERGY_J)},
        {"metric": "particle_counts", "value": ",".join(str(value) for value in PARTICLE_COUNTS)},
        {"metric": "particle_count_options", "value": len(PARTICLE_COUNTS)},
        {
            "metric": "acceleration_coefficients_c1_c2",
            "value": ",".join(f"{c1}:{c2}" for c1, c2 in ACCELERATION_COEFFICIENTS),
        },
        {"metric": "acceleration_coefficient_pair_count", "value": len(ACCELERATION_COEFFICIENTS)},
        {"metric": "optimizer_count", "value": len(OPTIMIZERS)},
        {"metric": "pso_ga_de_parameter_set_count", "value": len(PARAMETER_SETS)},
        {"metric": "direct_baseline_parameter_set_count", "value": 1},
        {"metric": "direct_baseline_optimizers", "value": ",".join(sorted(DIRECT_BASELINE_OPTIMIZERS))},
        {"metric": "experiment_input_count", "value": len(input_rows)},
    ]

    write_rows(args.output_dir / "experiment_topologies.csv", TOPOLOGY_FIELDS, topology_rows)
    write_rows(args.output_dir / "experiment_nodes.csv", NODE_FIELDS, node_rows)
    write_rows(args.output_dir / "experiment_inputs.csv", INPUT_FIELDS, input_rows)
    write_rows(args.output_dir / "experiment_manifest.csv", MANIFEST_FIELDS, manifest_rows)

    print(f"Saved {args.output_dir / 'experiment_topologies.csv'} ({len(topology_rows)} topologies)")
    print(f"Saved {args.output_dir / 'experiment_nodes.csv'} ({len(node_rows)} node rows)")
    print(f"Saved {args.output_dir / 'experiment_inputs.csv'} ({len(input_rows)} experiment inputs)")
    print(f"Saved {args.output_dir / 'experiment_manifest.csv'}")


if __name__ == "__main__":
    main()

