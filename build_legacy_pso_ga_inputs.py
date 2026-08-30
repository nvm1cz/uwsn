from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from run_legacy_pso_ga_tuning import (
    PACKET_ENERGY,
    PARAMETER_SETS,
    SPACE_NODES,
    TOPOLOGIES,
    _scenario_id,
)


OUTPUT = Path("inputs/legacy_pso_ga_tuning")
SEEDS = tuple(range(20))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for space, node_counts in SPACE_NODES.items():
        for nodes in node_counts:
            for topology in TOPOLOGIES:
                for packet, energy in PACKET_ENERGY:
                    scenarios.append({
                        "scenario_id": _scenario_id(
                            space, nodes, topology, packet, energy
                        ),
                        "space_x_m": space,
                        "space_y_m": space,
                        "space_z_m": space,
                        "node_count": nodes,
                        "topology": topology,
                        "packet_size_bits": packet,
                        "initial_energy_j": energy,
                        "transmission_range_m": 150.0,
                        "max_rounds": 1000,
                        "mobility_enabled": False,
                        "noise_enabled": False,
                        "delay_enabled": False,
                    })

    parameter_rows = []
    for parameter_id, config in PARAMETER_SETS.items():
        parameter_rows.append({
            "parameter_id": parameter_id,
            "optimizer": config["optimizer"],
            "population": config["population"],
            "iterations_or_generations": config["iterations"],
            "computational_budget": (
                int(config["population"]) * int(config["iterations"])
            ),
            "parameters_json": json.dumps(config, sort_keys=True),
        })

    run_plan = []
    for scenario in scenarios:
        for parameter in parameter_rows:
            for seed in SEEDS:
                run_plan.append({
                    "run_id": (
                        f"{scenario['scenario_id']}__"
                        f"{parameter['parameter_id']}__seed{seed:02d}"
                    ),
                    "scenario_id": scenario["scenario_id"],
                    "parameter_id": parameter["parameter_id"],
                    "optimizer": parameter["optimizer"],
                    "seed": seed,
                    "max_rounds": 1000,
                })

    _write_csv(OUTPUT / "scenarios.csv", scenarios)
    _write_csv(OUTPUT / "optimizer_parameter_sets.csv", parameter_rows)
    _write_csv(OUTPUT / "run_plan.csv", run_plan)
    (OUTPUT / "scenarios.json").write_text(
        json.dumps(scenarios, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "optimizer_parameter_sets.json").write_text(
        json.dumps(PARAMETER_SETS, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "model": "legacy_static_without_mobility_noise_delay",
        "space_node_combinations": sum(len(value) for value in SPACE_NODES.values()),
        "topology_count": len(TOPOLOGIES),
        "packet_energy_count": len(PACKET_ENERGY),
        "scenario_count": len(scenarios),
        "parameter_set_count": len(parameter_rows),
        "pso_parameter_set_count": sum(
            row["optimizer"] == "pso" for row in parameter_rows
        ),
        "ga_parameter_set_count": sum(
            row["optimizer"] == "ga" for row in parameter_rows
        ),
        "paired_seeds": list(SEEDS),
        "runs_per_scenario": len(parameter_rows) * len(SEEDS),
        "total_runs": len(run_plan),
        "excluded_cases": [
            {"space_m": 100, "node_count": 20},
            {"space_m": 1000, "node_count": 2000},
        ],
        "files": {
            "scenarios_csv": "scenarios.csv",
            "scenarios_json": "scenarios.json",
            "optimizer_parameter_sets_csv": "optimizer_parameter_sets.csv",
            "optimizer_parameter_sets_json": "optimizer_parameter_sets.json",
            "run_plan_csv": "run_plan.csv",
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
