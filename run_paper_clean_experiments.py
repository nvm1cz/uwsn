from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uwsn.cases import SimulationCase
from uwsn.deployment import compute_layers, deploy_nodes
from uwsn.run_config import SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


PROTOCOL_VERSION = "paper_clean_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/paper_clean")
DISTRIBUTIONS = ("uniform", "gaussian", "exponential")
NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
}
SEED_START = 4000

METHODS = {
    "pso_eulc": {
        "label": "PSO-EULC",
        "optimizer": "pso",
        "connectivity_penalty_enabled": True,
    },
    "pso_eulc_no_conn_penalty": {
        "label": "PSO-EULC w/o conn. penalty",
        "optimizer": "pso",
        "connectivity_penalty_enabled": False,
    },
    "eeumc": {
        "label": "EEUMC",
        "optimizer": "eeumc",
        "connectivity_penalty_enabled": True,
    },
    "ebrec": {
        "label": "EBREC",
        "optimizer": "ebrec",
        "connectivity_penalty_enabled": True,
    },
    "eulc": {
        "label": "EULC",
        "optimizer": "eulc",
        "connectivity_penalty_enabled": True,
    },
    "leach": {
        "label": "LEACH",
        "optimizer": "leach",
        "connectivity_penalty_enabled": True,
    },
}
MAIN_METHODS = ("pso_eulc", "eeumc", "ebrec", "eulc", "leach")
ABLATION_METHODS = ("pso_eulc", "pso_eulc_no_conn_penalty", "eulc")

METHOD_COLORS = {
    "pso_eulc": "#0072B2",
    "pso_eulc_no_conn_penalty": "#D55E00",
    "eeumc": "#009E73",
    "ebrec": "#E69F00",
    "eulc": "#CC79A7",
    "leach": "#666666",
}
METHOD_MARKERS = {
    "pso_eulc": "o",
    "pso_eulc_no_conn_penalty": "X",
    "eeumc": "s",
    "ebrec": "^",
    "eulc": "D",
    "leach": "v",
}

RUN_FIELDS = [
    "run_id",
    "protocol_version",
    "method",
    "method_label",
    "optimizer",
    "distribution",
    "space_m",
    "node_count",
    "seed",
    "topology_id",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "max_rounds",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
    "pso_particles",
    "pso_iterations",
    "pso_inertia",
    "pso_c1",
    "pso_c2",
    "pso_omega",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "recluster_interval",
    "connectivity_penalty_enabled",
    "status",
    "error",
    "runtime_s",
    "rounds_completed",
    "ft5_round",
    "ft5_censored",
    "fnd_round",
    "hnd_round",
    "lnd_round",
    "final_alive_nodes",
    "final_dead_nodes",
    "final_dead_nodes_pct",
    "final_residual_energy_j",
    "final_residual_energy_pct",
    "packets_received",
    "packets_per_round",
]
ROUND_FIELDS = [
    "run_id",
    "method",
    "distribution",
    "space_m",
    "node_count",
    "seed",
    "round",
    "residual_energy_j",
    "residual_energy_pct",
    "dead_nodes",
    "packets_received",
]
CONVERGENCE_FIELDS = [
    "run_id",
    "method",
    "distribution",
    "space_m",
    "node_count",
    "seed",
    "round",
    "refresh_index",
    "iteration",
    "best_score",
    "candidate_count",
]
INPUT_FIELDS = [
    "run_id",
    "method",
    "method_label",
    "topology_id",
    "distribution",
    "space_m",
    "node_count",
    "seed",
    "packet_size_bits",
    "initial_energy_j",
    "transmission_range_m",
    "max_rounds",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
    "pso_particles",
    "pso_iterations",
    "pso_inertia",
    "pso_c1",
    "pso_c2",
    "pso_omega",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "recluster_interval",
    "connectivity_penalty_enabled",
]
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
    "deployment_distribution",
    "gaussian_std_fraction",
    "exponential_scale_fraction",
]
NODE_FIELDS = [
    "topology_id",
    "node_id",
    "x_m",
    "y_m",
    "depth_m",
    "layer",
]
SNAPSHOT_FIELDS = [
    "run_id",
    "method",
    "distribution",
    "space_m",
    "node_count",
    "seed",
    "round",
    "refresh_index",
    "cluster_head_count",
    "cluster_heads_json",
    "assignments_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a clean, reproducible UWSN paper experiment suite."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runs", type=int, default=10, help="Seeds per topology setting.")
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--max-rounds", type=int, default=1000)
    parser.add_argument("--spaces", default="100,500")
    parser.add_argument("--distributions", default="uniform,gaussian,exponential")
    parser.add_argument(
        "--methods",
        default="pso_eulc,pso_eulc_no_conn_penalty,eeumc,ebrec,eulc,leach",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--snapshot-seed",
        type=int,
        default=SEED_START,
        help="Only this seed is saved to the representative cluster snapshot CSV.",
    )
    parser.add_argument("--packet-size-bits", type=int, default=6400)
    parser.add_argument("--initial-energy-j", type=float, default=0.5)
    parser.add_argument("--transmission-range-m", type=float, default=200.0)
    return parser.parse_args()


def _split_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _split_names(text: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in text.split(",") if item.strip())


def _append_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _existing_ok_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["run_id", "status"], dtype=str)
    return set(df.loc[df["status"].eq("ok"), "run_id"])


def _topology_id(distribution: str, space_m: int, node_count: int, seed: int) -> str:
    return f"{distribution}_s{space_m}_n{node_count}_seed{seed}"


def _run_id(
    method: str,
    distribution: str,
    space_m: int,
    node_count: int,
    seed: int,
    max_rounds: int,
) -> str:
    return (
        f"{PROTOCOL_VERSION}_{method}_{distribution}_s{space_m}_n{node_count}"
        f"_seed{seed}_r{max_rounds}"
    )


def _base_case(
    distribution: str,
    space_m: int,
    node_count: int,
    packet_size_bits: int,
    initial_energy_j: float,
    max_rounds: int,
) -> SimulationCase:
    return SimulationCase(
        name=f"{PROTOCOL_VERSION}_{distribution}_s{space_m}_n{node_count}",
        initial_energy=float(initial_energy_j),
        packet_size_bits=int(packet_size_bits),
        node_count=int(node_count),
        rounds=int(max_rounds),
    )


def _base_params(args: argparse.Namespace, method: str, space_m: int):
    method_spec = METHODS[method]
    return replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        transmission_range_m=float(args.transmission_range_m),
        optimizer=str(method_spec["optimizer"]),
        deployment_distribution="uniform",
        cluster_head_ratio=0.20,
        eulc_candidate_ratio=0.40,
        pso_particles=20,
        pso_iterations=50,
        pso_inertia=0.70,
        pso_c1=1.50,
        pso_c2=1.50,
        pso_omega=0.65,
        alpha_weight=0.60,
        beta_weight=0.25,
        gamma_weight=0.15,
        recluster_interval=20,
        contention_penalty_factor=0.0,
        connectivity_penalty_enabled=bool(method_spec["connectivity_penalty_enabled"]),
    )


def _topology_params(args: argparse.Namespace, distribution: str, space_m: int):
    return replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        transmission_range_m=float(args.transmission_range_m),
        deployment_distribution=distribution,
    )


def _build_run_plan(args: argparse.Namespace) -> list[dict[str, object]]:
    spaces = _split_ints(args.spaces)
    distributions = _split_names(args.distributions)
    methods = _split_names(args.methods)
    invalid_methods = [method for method in methods if method not in METHODS]
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {invalid_methods}")

    plan: list[dict[str, object]] = []
    for distribution in distributions:
        if distribution not in DISTRIBUTIONS:
            raise ValueError(f"Unsupported distribution: {distribution}")
        for space_m in spaces:
            if space_m not in NODE_SETS:
                raise ValueError(f"Unsupported space_m: {space_m}; expected one of {sorted(NODE_SETS)}")
            for node_count in NODE_SETS[space_m]:
                for seed in range(args.seed_start, args.seed_start + args.runs):
                    tid = _topology_id(distribution, space_m, node_count, seed)
                    for method in methods:
                        params = _base_params(args, method, space_m)
                        rid = _run_id(method, distribution, space_m, node_count, seed, args.max_rounds)
                        plan.append(
                            {
                                "run_id": rid,
                                "method": method,
                                "method_label": METHODS[method]["label"],
                                "optimizer": METHODS[method]["optimizer"],
                                "distribution": distribution,
                                "space_m": space_m,
                                "node_count": node_count,
                                "seed": seed,
                                "topology_id": tid,
                                "packet_size_bits": args.packet_size_bits,
                                "initial_energy_j": args.initial_energy_j,
                                "transmission_range_m": args.transmission_range_m,
                                "max_rounds": args.max_rounds,
                                "cluster_head_ratio": params.cluster_head_ratio,
                                "eulc_candidate_ratio": params.eulc_candidate_ratio,
                                "pso_particles": params.pso_particles,
                                "pso_iterations": params.pso_iterations,
                                "pso_inertia": params.pso_inertia,
                                "pso_c1": params.pso_c1,
                                "pso_c2": params.pso_c2,
                                "pso_omega": params.pso_omega,
                                "alpha_weight": params.alpha_weight,
                                "beta_weight": params.beta_weight,
                                "gamma_weight": params.gamma_weight,
                                "recluster_interval": params.recluster_interval,
                                "connectivity_penalty_enabled": params.connectivity_penalty_enabled,
                            }
                        )
    if args.max_cases is not None:
        plan = plan[: max(0, args.max_cases)]
    return plan


def _build_topologies(
    args: argparse.Namespace,
    plan: list[dict[str, object]],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]]]:
    topology_keys = sorted(
        {
            (
                str(row["topology_id"]),
                str(row["distribution"]),
                int(row["space_m"]),
                int(row["node_count"]),
                int(row["seed"]),
            )
            for row in plan
        }
    )
    positions_by_topology: dict[str, np.ndarray] = {}
    topology_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []

    for topology_id, distribution, space_m, node_count, seed in topology_keys:
        case = _base_case(
            distribution,
            space_m,
            node_count,
            args.packet_size_bits,
            args.initial_energy_j,
            args.max_rounds,
        )
        params = _topology_params(args, distribution, space_m)
        rng = np.random.default_rng(seed)
        positions = deploy_nodes(case, params, rng)
        layers = compute_layers(positions, params)
        sink = params.get_sink_position()
        positions_by_topology[topology_id] = positions
        topology_rows.append(
            {
                "topology_id": topology_id,
                "distribution": distribution,
                "space_m": space_m,
                "node_count": node_count,
                "seed": seed,
                "width_m": space_m,
                "height_m": space_m,
                "depth_m": space_m,
                "sink_x_m": sink[0],
                "sink_y_m": sink[1],
                "sink_depth_m": sink[2],
                "deployment_distribution": distribution,
                "gaussian_std_fraction": params.gaussian_std_fraction,
                "exponential_scale_fraction": params.exponential_scale_fraction,
            }
        )
        for node_id, pos in enumerate(positions):
            node_rows.append(
                {
                    "topology_id": topology_id,
                    "node_id": node_id,
                    "x_m": round(float(pos[0]), 6),
                    "y_m": round(float(pos[1]), 6),
                    "depth_m": round(float(pos[2]), 6),
                    "layer": int(layers[node_id]),
                }
            )
    return positions_by_topology, topology_rows, node_rows


def _write_protocol_inputs(
    out_dir: Path,
    plan: list[dict[str, object]],
    topology_rows: list[dict[str, object]],
    node_rows: list[dict[str, object]],
) -> None:
    input_dir = out_dir / "inputs"
    _write_rows(input_dir / "paper_clean_run_plan.csv", INPUT_FIELDS, plan)
    _write_rows(input_dir / "paper_clean_topologies.csv", TOPOLOGY_FIELDS, topology_rows)
    _write_rows(input_dir / "paper_clean_nodes.csv", NODE_FIELDS, node_rows)


def _round_rows(row: dict[str, object], metrics, total_energy: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for round_idx, (residual, dead_nodes, packets) in enumerate(
        zip(
            metrics.residual_energy_by_round,
            metrics.dead_nodes_by_round,
            metrics.packets_received_by_round,
        ),
        start=1,
    ):
        rows.append(
            {
                "run_id": row["run_id"],
                "method": row["method"],
                "distribution": row["distribution"],
                "space_m": row["space_m"],
                "node_count": row["node_count"],
                "seed": row["seed"],
                "round": round_idx,
                "residual_energy_j": residual,
                "residual_energy_pct": 100.0 * float(residual) / max(total_energy, 1e-12),
                "dead_nodes": dead_nodes,
                "packets_received": packets,
            }
        )
    return rows


def _convergence_rows(row: dict[str, object], metrics) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in metrics.pso_convergence:
        rows.append(
            {
                "run_id": row["run_id"],
                "method": row["method"],
                "distribution": row["distribution"],
                "space_m": row["space_m"],
                "node_count": row["node_count"],
                "seed": row["seed"],
                "round": int(event["round"]),
                "refresh_index": int(event["refresh_index"]),
                "iteration": int(event["iteration"]),
                "best_score": float(event["best_score"]),
                "candidate_count": float(event["candidate_count"]),
            }
        )
    return rows


def _snapshot_rows(row: dict[str, object], simulator: PsoEulcSimulator, snapshot_seed: int) -> list[dict[str, object]]:
    if int(row["seed"]) != int(snapshot_seed):
        return []
    if str(row["distribution"]) != "uniform":
        return []

    rows: list[dict[str, object]] = []
    for snapshot in simulator.cluster_snapshots[:3]:
        assignments = {str(k): v for k, v in snapshot["assignments"].items()}
        rows.append(
            {
                "run_id": row["run_id"],
                "method": row["method"],
                "distribution": row["distribution"],
                "space_m": row["space_m"],
                "node_count": row["node_count"],
                "seed": row["seed"],
                "round": snapshot["round"],
                "refresh_index": snapshot["refresh_index"],
                "cluster_head_count": len(snapshot["cluster_heads"]),
                "cluster_heads_json": json.dumps(snapshot["cluster_heads"]),
                "assignments_json": json.dumps(assignments),
            }
        )
    return rows


def run_experiments(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    raw_dir = out_dir / "raw"
    run_csv = raw_dir / "paper_clean_runs.csv"
    round_csv = raw_dir / "paper_clean_rounds.csv"
    convergence_csv = raw_dir / "paper_clean_convergence.csv"
    snapshot_csv = raw_dir / "paper_clean_cluster_snapshots.csv"

    plan = _build_run_plan(args)
    positions_by_topology, topology_rows, node_rows = _build_topologies(args, plan)
    _write_protocol_inputs(out_dir, plan, topology_rows, node_rows)

    completed = set() if args.no_resume else _existing_ok_run_ids(run_csv)
    total = len(plan)
    for idx, row in enumerate(plan, start=1):
        run_id = str(row["run_id"])
        if run_id in completed:
            print(f"[{idx}/{total}] skip {run_id}")
            continue

        print(f"[{idx}/{total}] run {run_id}", flush=True)
        case = _base_case(
            str(row["distribution"]),
            int(row["space_m"]),
            int(row["node_count"]),
            int(row["packet_size_bits"]),
            float(row["initial_energy_j"]),
            int(row["max_rounds"]),
        )
        params = _base_params(args, str(row["method"]), int(row["space_m"]))
        params = replace(
            params,
            deployment_distribution=str(row["distribution"]),
            transmission_range_m=float(row["transmission_range_m"]),
        )
        total_energy = case.initial_energy * case.node_count
        started = perf_counter()
        try:
            simulator = PsoEulcSimulator(
                case,
                params,
                seed=int(row["seed"]),
                verbose=False,
                track_pso_convergence=True,
                initial_positions=positions_by_topology[str(row["topology_id"])],
            )
            metrics = simulator.run(
                stop_on_first_dead=False,
                stop_at_ft5=False,
                min_alive_ratio=None,
                max_rounds=int(row["max_rounds"]),
            )
            runtime_s = perf_counter() - started
            rounds_completed = len(metrics.residual_energy_by_round)
            final_dead_nodes = case.node_count - metrics.alive_nodes
            ft5_censored = metrics.first_5pct_round is None
            packets_per_round = (
                float(metrics.packets_received) / rounds_completed if rounds_completed else 0.0
            )
            run_row = {
                **row,
                "protocol_version": PROTOCOL_VERSION,
                "status": "ok",
                "error": "",
                "runtime_s": runtime_s,
                "rounds_completed": rounds_completed,
                "ft5_round": metrics.first_5pct_round,
                "ft5_censored": ft5_censored,
                "fnd_round": metrics.fnd_round,
                "hnd_round": metrics.hnd_round,
                "lnd_round": metrics.lnd_round,
                "final_alive_nodes": metrics.alive_nodes,
                "final_dead_nodes": final_dead_nodes,
                "final_dead_nodes_pct": 100.0 * final_dead_nodes / max(case.node_count, 1),
                "final_residual_energy_j": metrics.residual_energy,
                "final_residual_energy_pct": 100.0 * metrics.residual_energy / max(total_energy, 1e-12),
                "packets_received": metrics.packets_received,
                "packets_per_round": packets_per_round,
            }
            _append_rows(run_csv, RUN_FIELDS, [run_row])
            _append_rows(round_csv, ROUND_FIELDS, _round_rows(row, metrics, total_energy))
            _append_rows(convergence_csv, CONVERGENCE_FIELDS, _convergence_rows(row, metrics))
            _append_rows(snapshot_csv, SNAPSHOT_FIELDS, _snapshot_rows(row, simulator, args.snapshot_seed))
        except Exception as exc:
            runtime_s = perf_counter() - started
            run_row = {
                **row,
                "protocol_version": PROTOCOL_VERSION,
                "status": "error",
                "error": repr(exc),
                "runtime_s": runtime_s,
                "rounds_completed": 0,
                "ft5_round": "",
                "ft5_censored": "",
                "fnd_round": "",
                "hnd_round": "",
                "lnd_round": "",
                "final_alive_nodes": "",
                "final_dead_nodes": "",
                "final_dead_nodes_pct": "",
                "final_residual_energy_j": "",
                "final_residual_energy_pct": "",
                "packets_received": "",
                "packets_per_round": "",
            }
            _append_rows(run_csv, RUN_FIELDS, [run_row])
            print(f"  ERROR {run_id}: {exc}", flush=True)


def _setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 4.8,
        }
    )


def _save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.pdf")
    fig.savefig(figures_dir / f"{stem}.png")
    plt.close(fig)


def _metric_panel(
    df: pd.DataFrame,
    methods: tuple[str, ...],
    metric: str,
    ylabel: str,
    title: str,
    figures_dir: Path,
    stem: str,
) -> None:
    data = df[df["method"].isin(methods)].copy()
    if data.empty:
        return

    grouped = (
        data.groupby(["space_m", "node_count", "method"], as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    spaces = sorted(grouped["space_m"].unique())
    fig, axes = plt.subplots(1, len(spaces), figsize=(6.75, 2.65), sharey=True)
    if len(spaces) == 1:
        axes = [axes]
    y_min = float((grouped["mean"] - grouped["std"].fillna(0)).min())
    y_max = float((grouped["mean"] + grouped["std"].fillna(0)).max())
    pad = max((y_max - y_min) * 0.08, 1e-6)

    for ax, space_m in zip(axes, spaces):
        panel = grouped[grouped["space_m"].eq(space_m)]
        for method in methods:
            series = panel[panel["method"].eq(method)].sort_values("node_count")
            if series.empty:
                continue
            ax.errorbar(
                series["node_count"],
                series["mean"],
                yerr=series["std"].fillna(0),
                label=METHODS[method]["label"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                capsize=2.5,
            )
        ax.set_title(f"Space {int(space_m)} m")
        ax.set_xlabel("Number of nodes")
        ax.set_xticks(sorted(panel["node_count"].unique()))
        ax.set_ylim(y_min - pad, y_max + pad)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(methods), 5), frameon=False)
    fig.suptitle(title, y=1.08, fontsize=10.5)
    _save_figure(fig, figures_dir, stem)


def _plot_convergence(df: pd.DataFrame, figures_dir: Path) -> None:
    convergence_csv = figures_dir.parent / "raw" / "paper_clean_convergence.csv"
    if not convergence_csv.exists():
        return
    conv = pd.read_csv(convergence_csv)
    conv = conv[
        conv["method"].isin(("pso_eulc", "pso_eulc_no_conn_penalty"))
        & conv["refresh_index"].eq(1)
    ].copy()
    if conv.empty:
        return

    first_scores = conv[conv["iteration"].eq(0)][["run_id", "best_score"]].rename(
        columns={"best_score": "initial_best_score"}
    )
    conv = conv.merge(first_scores, on="run_id", how="left")
    conv = conv[conv["initial_best_score"].gt(0)]
    conv["relative_best_score"] = conv["best_score"] / conv["initial_best_score"]
    grouped = (
        conv.groupby(["space_m", "method", "iteration"], as_index=False)["relative_best_score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    spaces = sorted(grouped["space_m"].unique())
    fig, axes = plt.subplots(1, len(spaces), figsize=(6.75, 2.55), sharey=True)
    if len(spaces) == 1:
        axes = [axes]

    for ax, space_m in zip(axes, spaces):
        panel = grouped[grouped["space_m"].eq(space_m)]
        for method in ("pso_eulc", "pso_eulc_no_conn_penalty"):
            series = panel[panel["method"].eq(method)].sort_values("iteration")
            if series.empty:
                continue
            ax.plot(
                series["iteration"],
                series["mean"],
                label=METHODS[method]["label"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markevery=max(1, int(len(series) / 8)),
            )
            ax.fill_between(
                series["iteration"],
                series["mean"] - series["std"].fillna(0),
                series["mean"] + series["std"].fillna(0),
                color=METHOD_COLORS[method],
                alpha=0.14,
            )
        ax.set_title(f"Space {int(space_m)} m")
        ax.set_xlabel("PSO iteration")
        ax.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Best cost / initial best cost")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("PSO convergence during the first clustering refresh", y=1.08, fontsize=10.5)
    _save_figure(fig, figures_dir, "fig05_pso_convergence_first_refresh")


def _write_readme(out_dir: Path, runs_df: pd.DataFrame) -> None:
    ok = runs_df[runs_df["status"].eq("ok")]
    errors = runs_df[~runs_df["status"].eq("ok")]
    completed = len(ok)
    total = len(runs_df)
    method_counts = ok.groupby("method").size().to_dict() if completed else {}
    readme = f"""# Paper Clean Experiments

Protocol version: `{PROTOCOL_VERSION}`

## Goal

This folder contains the clean, reproducible experiment suite for the UWSN paper draft. It is separated from exploratory outputs to avoid mixing earlier trial runs with final paper evidence.

## Main Comparison

Methods: PSO-EULC, EEUMC, EBREC, EULC, and LEACH.

All methods run on the same fixed 3D topologies for each distribution, space, node count, and seed. The simulator uses common routing and energy accounting for fairness.

## Ablation

The suite also includes `pso_eulc_no_conn_penalty`, which disables only the inter-cluster connectivity penalty. This isolates the effect of the connectivity-aware extension added to the PSO-EULC objective.

## Fixed Parameters

- Spaces: `{sorted(ok['space_m'].unique().tolist()) if completed else 'not completed yet'}`
- Distributions: `{sorted(ok['distribution'].unique().tolist()) if completed else 'not completed yet'}`
- Transmission range: `{ok['transmission_range_m'].iloc[0] if completed else 'not completed yet'} m`
- Packet size: `{ok['packet_size_bits'].iloc[0] if completed else 'not completed yet'} bits`
- Initial energy: `{ok['initial_energy_j'].iloc[0] if completed else 'not completed yet'} J`
- Max rounds: `{ok['max_rounds'].iloc[0] if completed else 'not completed yet'}`
- CH ratio Pc: `{ok['cluster_head_ratio'].iloc[0] if completed else 'not completed yet'}`
- EULC candidate ratio: `{ok['eulc_candidate_ratio'].iloc[0] if completed else 'not completed yet'}`
- PSO particles/iterations: `{ok['pso_particles'].iloc[0] if completed else 'not completed yet'}` / `{ok['pso_iterations'].iloc[0] if completed else 'not completed yet'}`
- PSO inertia/c1/c2/omega: `{ok['pso_inertia'].iloc[0] if completed else 'not completed yet'}` / `{ok['pso_c1'].iloc[0] if completed else 'not completed yet'}` / `{ok['pso_c2'].iloc[0] if completed else 'not completed yet'}` / `{ok['pso_omega'].iloc[0] if completed else 'not completed yet'}`

## Outputs

- `inputs/paper_clean_run_plan.csv`: exact run plan.
- `inputs/paper_clean_topologies.csv`: fixed topology metadata.
- `inputs/paper_clean_nodes.csv`: fixed node coordinates.
- `raw/paper_clean_runs.csv`: per-run summary metrics.
- `raw/paper_clean_rounds.csv`: residual energy, dead nodes, and delivered packets by round.
- `raw/paper_clean_convergence.csv`: PSO best-cost convergence events.
- `raw/paper_clean_cluster_snapshots.csv`: representative CH-member assignments.
- `figures/*.pdf` and `figures/*.png`: paper-ready figures.

## Metrics

- FT5: first round where at least 5% of nodes are dead. If FT5 is not reached by `max_rounds`, the value is treated as censored at `max_rounds` in plots.
- Residual energy: final residual energy as percentage of initial total energy.
- Dead nodes: number of dead nodes at the final simulated round.
- Runtime: wall-clock simulator runtime for one method/topology/seed run.
- Convergence: PSO best cost normalized by the initial best cost during the first clustering refresh.

## Current Status

- Completed OK runs: `{completed}/{total}`
- Error runs: `{len(errors)}`
- OK runs by method: `{method_counts}`

## Scope and Limitations

These experiments support claims for static 3D UWSN simulation only. Node drift, acoustic channel fading, mobility, and environmental noise are not modeled. EEUMC and EBREC are implemented from their published algorithm descriptions with simulator-specific mapping for the 3D layered topology.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def plot_results(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    run_csv = out_dir / "raw" / "paper_clean_runs.csv"
    if not run_csv.exists():
        print(f"No run CSV found: {run_csv}")
        return
    _setup_plot_style()
    runs_df = pd.read_csv(run_csv)
    if runs_df.empty:
        return
    ok = runs_df[runs_df["status"].eq("ok")].copy()
    if ok.empty:
        _write_readme(out_dir, runs_df)
        return

    ok["ft5_plot_round"] = ok["ft5_round"].fillna(ok["max_rounds"])
    figures_dir = out_dir / "figures"
    _metric_panel(
        ok,
        MAIN_METHODS,
        "ft5_plot_round",
        "FT5 round",
        "Network lifetime comparison by FT5",
        figures_dir,
        "fig01_ft5_by_node_count",
    )
    _metric_panel(
        ok,
        MAIN_METHODS,
        "final_residual_energy_pct",
        "Residual energy (%)",
        "Final residual energy comparison",
        figures_dir,
        "fig02_residual_energy_by_node_count",
    )
    _metric_panel(
        ok,
        MAIN_METHODS,
        "final_dead_nodes",
        "Dead nodes",
        "Final dead-node comparison",
        figures_dir,
        "fig03_dead_nodes_by_node_count",
    )
    _metric_panel(
        ok,
        MAIN_METHODS,
        "runtime_s",
        "Runtime (s)",
        "Runtime comparison",
        figures_dir,
        "fig04_runtime_by_node_count",
    )
    _metric_panel(
        ok,
        ABLATION_METHODS,
        "ft5_plot_round",
        "FT5 round",
        "Ablation of the connectivity penalty",
        figures_dir,
        "fig06_connectivity_penalty_ablation_ft5",
    )
    _plot_convergence(ok, figures_dir)
    summary = (
        ok.groupby(["method", "space_m", "node_count"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            ft5_mean=("ft5_plot_round", "mean"),
            ft5_std=("ft5_plot_round", "std"),
            residual_mean=("final_residual_energy_pct", "mean"),
            dead_nodes_mean=("final_dead_nodes", "mean"),
            runtime_mean=("runtime_s", "mean"),
        )
        .sort_values(["space_m", "node_count", "method"])
    )
    summary.to_csv(out_dir / "raw" / "paper_clean_summary.csv", index=False)
    _write_readme(out_dir, runs_df)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        run_experiments(args)
    if not args.no_plot:
        plot_results(args)


if __name__ == "__main__":
    main()
