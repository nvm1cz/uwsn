from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from uwsn.cases import SimulationCase
from uwsn.run_config import SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


ALGORITHMS = ("pso", "eeumc", "ebrec", "eulc", "leach")
LABELS = {
    "pso": "PSO-EULC",
    "eeumc": "EEUMC",
    "ebrec": "EBREC",
    "eulc": "EULC",
    "leach": "LEACH",
}
COLORS = {
    "pso": "#4C78A8",
    "eeumc": "#F58518",
    "ebrec": "#54A24B",
    "eulc": "#B279A2",
    "leach": "#E45756",
}
MARKERS = {
    "pso": "o",
    "eeumc": "s",
    "ebrec": "^",
    "eulc": "D",
    "leach": "x",
}

OUT_DIR = Path("outputs/figures/paper_result_reproduction_model")
RAW_DIR = Path("outputs/data/csv/raw")
RUNS_CSV = RAW_DIR / "paper_result_model_runs.csv"
ROUNDS_CSV = RAW_DIR / "paper_result_model_rounds.csv"
ROUND_FIELDS = [
    "scenario_id",
    "figures",
    "optimizer",
    "seed",
    "rounds_requested",
    "round",
    "residual_energy_j",
    "dead_nodes",
    "packets_received",
]
LEGACY_ROUNDS_REQUESTED = 300


def _base_params(space_m: float, transmission_range_m: float, optimizer: str):
    return replace(
        SIMULATION_PARAMS,
        width_m=float(space_m),
        height_m=float(space_m),
        depth_m=float(space_m),
        transmission_range_m=float(transmission_range_m),
        optimizer=optimizer,
    )


def _scenario_id(prefix: str, node_count: int, space_m: float, r_m: float, e0: float, packet: int) -> str:
    return f"{prefix}_s{int(space_m)}_n{node_count}_r{int(r_m)}_e{e0:g}_p{packet}"


def _round_curve_scenarios() -> list[dict[str, object]]:
    return [
        {
            "scenario_id": _scenario_id("fig05_10", 100, 100.0, 200.0, 1.0, 6400),
            "figures": "5,10",
            "node_count": 100,
            "space_m": 100.0,
            "transmission_range_m": 200.0,
            "initial_energy_j": 1.0,
            "packet_size_bits": 6400,
        },
        {
            "scenario_id": _scenario_id("fig06_11", 100, 100.0, 150.0, 1.0, 4000),
            "figures": "6,11",
            "node_count": 100,
            "space_m": 100.0,
            "transmission_range_m": 150.0,
            "initial_energy_j": 1.0,
            "packet_size_bits": 4000,
        },
    ]


def _node_sweep_scenarios(max_node: int) -> list[dict[str, object]]:
    node_values = [50, 100, 150, 200, 300, 400, 500]
    node_values = [n for n in node_values if n <= max_node]
    fig9_values = [100, 200, 300, 400, 500, 750, 1000]
    fig9_values = [n for n in fig9_values if n <= max_node]
    scenarios: list[dict[str, object]] = []

    for n in node_values:
        scenarios.append(
            {
                "scenario_id": _scenario_id("fig07_12", n, 100.0, 150.0, 0.5, 4000),
                "figures": "7,12",
                "node_count": n,
                "space_m": 100.0,
                "transmission_range_m": 150.0,
                "initial_energy_j": 0.5,
                "packet_size_bits": 4000,
            }
        )
        scenarios.append(
            {
                "scenario_id": _scenario_id("fig08_13", n, 100.0, 200.0, 0.5, 6400),
                "figures": "8,13",
                "node_count": n,
                "space_m": 100.0,
                "transmission_range_m": 200.0,
                "initial_energy_j": 0.5,
                "packet_size_bits": 6400,
            }
        )

    for n in fig9_values:
        space_m = 100.0 if n <= 500 else 500.0
        scenarios.append(
            {
                "scenario_id": _scenario_id("fig09", n, space_m, 200.0, 0.5, 6400),
                "figures": "9",
                "node_count": n,
                "space_m": space_m,
                "transmission_range_m": 200.0,
                "initial_energy_j": 0.5,
                "packet_size_bits": 6400,
            }
        )
        scenarios.append(
            {
                "scenario_id": _scenario_id("fig14", n, space_m, 200.0, 1.0, 6400),
                "figures": "14",
                "node_count": n,
                "space_m": space_m,
                "transmission_range_m": 200.0,
                "initial_energy_j": 1.0,
                "packet_size_bits": 6400,
            }
        )
    return scenarios


def _all_scenarios(max_node: int) -> list[dict[str, object]]:
    seen = set()
    scenarios = []
    for scenario in [*_round_curve_scenarios(), *_node_sweep_scenarios(max_node)]:
        sid = str(scenario["scenario_id"])
        if sid in seen:
            continue
        seen.add(sid)
        scenarios.append(scenario)
    return scenarios


def _existing_run_keys() -> set[tuple[str, str, int, int]]:
    if not RUNS_CSV.exists():
        return set()
    df = pd.read_csv(
        RUNS_CSV,
        usecols=["scenario_id", "optimizer", "seed", "rounds_requested", "status"],
    )
    df = df[df["status"].eq("ok")]
    return set(
        zip(
            df["scenario_id"],
            df["optimizer"],
            df["seed"].astype(int),
            df["rounds_requested"].astype(int),
        )
    )


def _migrate_rounds_csv_schema() -> None:
    if not ROUNDS_CSV.exists():
        return
    with ROUNDS_CSV.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return
    header = rows[0]
    if header == ROUND_FIELDS and all(len(row) == len(ROUND_FIELDS) for row in rows[1:]):
        return

    migrated: list[list[object]] = [ROUND_FIELDS]
    for row in rows[1:]:
        if not row:
            continue
        if len(row) == 8:
            scenario_id, figures, optimizer, seed, round_idx, residual, dead, packets = row
            migrated.append(
                [
                    scenario_id,
                    figures,
                    optimizer,
                    seed,
                    LEGACY_ROUNDS_REQUESTED,
                    round_idx,
                    residual,
                    dead,
                    packets,
                ]
            )
        elif len(row) == 9:
            migrated.append(row)
        else:
            print(f"Skipping malformed rounds CSV row with {len(row)} fields: {row[:3]}")

    tmp_path = ROUNDS_CSV.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(migrated)
    tmp_path.replace(ROUNDS_CSV)


def _append_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _run_one(scenario: dict[str, object], optimizer: str, seed: int, rounds: int) -> None:
    params = _base_params(
        float(scenario["space_m"]),
        float(scenario["transmission_range_m"]),
        optimizer,
    )
    case = SimulationCase(
        name=str(scenario["scenario_id"]),
        initial_energy=float(scenario["initial_energy_j"]),
        packet_size_bits=int(scenario["packet_size_bits"]),
        node_count=int(scenario["node_count"]),
        rounds=rounds,
    )
    started = perf_counter()
    run_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    try:
        sim = PsoEulcSimulator(case, params, seed=seed, verbose=False)
        metrics = sim.run(stop_on_first_dead=False, max_rounds=rounds)
        runtime_sec = perf_counter() - started
        completed_rounds = len(metrics.residual_energy_by_round)
        possible_packets = max(1, completed_rounds * case.node_count)
        pdr_pct = 100.0 * float(metrics.packets_received) / possible_packets
        run_rows.append(
            {
                **scenario,
                "optimizer": optimizer,
                "seed": seed,
                "rounds_requested": rounds,
                "rounds_completed": completed_rounds,
                "final_residual_energy_j": float(metrics.residual_energy),
                "final_residual_energy_pct": 100.0
                * float(metrics.residual_energy)
                / max(1e-12, case.initial_energy * case.node_count),
                "final_dead_nodes": int(case.node_count - metrics.alive_nodes),
                "packets_received": int(metrics.packets_received),
                "pdr_pct": pdr_pct,
                "fnd_round": metrics.fnd_round if metrics.fnd_round is not None else "",
                "hnd_round": metrics.hnd_round if metrics.hnd_round is not None else "",
                "lnd_round": metrics.lnd_round if metrics.lnd_round is not None else "",
                "ft5_round": metrics.first_5pct_round if metrics.first_5pct_round is not None else "",
                "runtime_sec": runtime_sec,
                "status": "ok",
                "error": "",
            }
        )
        for idx, (residual, dead, packets) in enumerate(
            zip(
                metrics.residual_energy_by_round,
                metrics.dead_nodes_by_round,
                metrics.packets_received_by_round,
            ),
            start=1,
        ):
            round_rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "figures": scenario["figures"],
                    "optimizer": optimizer,
                    "seed": seed,
                    "rounds_requested": rounds,
                    "round": idx,
                    "residual_energy_j": float(residual),
                    "dead_nodes": int(dead),
                    "packets_received": int(packets),
                }
            )
    except Exception as exc:
        runtime_sec = perf_counter() - started
        run_rows.append(
            {
                **scenario,
                "optimizer": optimizer,
                "seed": seed,
                "rounds_requested": rounds,
                "rounds_completed": 0,
                "final_residual_energy_j": "",
                "final_residual_energy_pct": "",
                "final_dead_nodes": "",
                "packets_received": "",
                "pdr_pct": "",
                "fnd_round": "",
                "hnd_round": "",
                "lnd_round": "",
                "ft5_round": "",
                "runtime_sec": runtime_sec,
                "status": "error",
                "error": str(exc)[:240],
            }
        )

    run_fields = [
        "scenario_id",
        "figures",
        "node_count",
        "space_m",
        "transmission_range_m",
        "initial_energy_j",
        "packet_size_bits",
        "optimizer",
        "seed",
        "rounds_requested",
        "rounds_completed",
        "final_residual_energy_j",
        "final_residual_energy_pct",
        "final_dead_nodes",
        "packets_received",
        "pdr_pct",
        "fnd_round",
        "hnd_round",
        "lnd_round",
        "ft5_round",
        "runtime_sec",
        "status",
        "error",
    ]
    round_fields = [
        *ROUND_FIELDS,
    ]
    _migrate_rounds_csv_schema()
    _append_rows(RUNS_CSV, run_rows, run_fields)
    _append_rows(ROUNDS_CSV, round_rows, round_fields)


def run_missing_experiments(runs: int, rounds: int, max_node: int, base_seed: int, limit: int | None) -> None:
    existing = _existing_run_keys()
    jobs = []
    for scenario in _all_scenarios(max_node):
        for optimizer in ALGORITHMS:
            for offset in range(runs):
                seed = base_seed + offset
                key = (str(scenario["scenario_id"]), optimizer, seed, rounds)
                if key not in existing:
                    jobs.append((scenario, optimizer, seed))
    if limit is not None:
        jobs = jobs[: max(0, limit)]
    total = len(jobs)
    for idx, (scenario, optimizer, seed) in enumerate(jobs, start=1):
        print(
            f"[{idx}/{total}] {scenario['scenario_id']} {LABELS[optimizer]} seed={seed}",
            flush=True,
        )
        _run_one(scenario, optimizer, seed, rounds)


def _mean_rounds(rounds_df: pd.DataFrame, scenario_prefix: str, metric: str) -> pd.DataFrame:
    df = rounds_df[rounds_df["scenario_id"].str.startswith(scenario_prefix)].copy()
    return (
        df.groupby(["optimizer", "round"], as_index=False)[metric]
        .mean()
        .sort_values(["optimizer", "round"])
    )


def _plot_round_figure(rounds_df: pd.DataFrame, scenario_prefix: str, metric: str, ylabel: str, title: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    mean_df = _mean_rounds(rounds_df, scenario_prefix, metric)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for optimizer in ALGORITHMS:
        sub = mean_df[mean_df["optimizer"].eq(optimizer)]
        if sub.empty:
            continue
        ax.plot(
            sub["round"],
            sub[metric],
            label=LABELS[optimizer],
            color=COLORS[optimizer],
            linewidth=2.0,
        )
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(ncol=3, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _summary_by_scenario(runs_df: pd.DataFrame, scenario_prefix: str, metric: str) -> pd.DataFrame:
    df = runs_df[
        runs_df["status"].eq("ok") & runs_df["scenario_id"].str.startswith(scenario_prefix)
    ].copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return (
        df.groupby(["node_count", "optimizer"], as_index=False)
        .agg(mean_value=(metric, "mean"), std_value=(metric, "std"), runs=(metric, "count"))
        .fillna(0.0)
        .sort_values(["node_count", "optimizer"])
    )


def _plot_nodes_figure(runs_df: pd.DataFrame, scenario_prefix: str, metric: str, ylabel: str, title: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    summary = _summary_by_scenario(runs_df, scenario_prefix, metric)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for optimizer in ALGORITHMS:
        sub = summary[summary["optimizer"].eq(optimizer)]
        if sub.empty:
            continue
        ax.errorbar(
            sub["node_count"],
            sub["mean_value"],
            yerr=sub["std_value"],
            label=LABELS[optimizer],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linewidth=2.0,
            capsize=3,
        )
    ax.set_title(title)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(ncol=3, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _scenario_bars(runs_df: pd.DataFrame, scenario_prefixes: Iterable[str], metric: str) -> pd.DataFrame:
    frames = []
    for prefix in scenario_prefixes:
        df = runs_df[
            runs_df["status"].eq("ok") & runs_df["scenario_id"].str.startswith(prefix)
        ].copy()
        if df.empty:
            continue
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df["scenario_group"] = prefix
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    return (
        all_df.groupby(["scenario_group", "optimizer"], as_index=False)
        .agg(mean_value=(metric, "mean"), std_value=(metric, "std"))
        .fillna(0.0)
    )


def _plot_average_bar(runs_df: pd.DataFrame, metric: str, ylabel: str, title: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    scenario_prefixes = ["fig05_10", "fig06_11", "fig09"]
    labels = {
        "fig05_10": "R=200m, N=100",
        "fig06_11": "R=150m, N=100",
        "fig09": "Node sweep",
    }
    summary = _scenario_bars(runs_df, scenario_prefixes, metric)
    if summary.empty:
        return
    groups = list(scenario_prefixes)
    x = np.arange(len(groups), dtype=float)
    width = 0.15
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx, optimizer in enumerate(ALGORITHMS):
        sub = summary[summary["optimizer"].eq(optimizer)].set_index("scenario_group")
        values = [float(sub.loc[g, "mean_value"]) if g in sub.index else 0.0 for g in groups]
        errs = [float(sub.loc[g, "std_value"]) if g in sub.index else 0.0 for g in groups]
        ax.bar(
            x + (idx - 2) * width,
            values,
            width=width,
            yerr=errs,
            label=LABELS[optimizer],
            color=COLORS[optimizer],
            capsize=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[g] for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.32)
    ax.legend(ncol=3, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_statistical_analysis(runs_df: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    df = runs_df[runs_df["status"].eq("ok")].copy()
    if df.empty:
        return
    metrics = [
        ("final_residual_energy_pct", "Residual energy (%)"),
        ("final_dead_nodes", "Dead nodes"),
        ("pdr_pct", "PDR (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        data = [df[df["optimizer"].eq(opt)][metric].dropna().to_numpy() for opt in ALGORITHMS]
        ax.boxplot(data, tick_labels=[LABELS[opt] for opt in ALGORITHMS], showmeans=True)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", linestyle="--", alpha=0.32)
    fig.suptitle("Figure 17: Statistical analysis across model runs")
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_notes(runs_df: pd.DataFrame, output: Path) -> None:
    ok = runs_df[runs_df["status"].eq("ok")].copy()
    lines = [
        "# Paper Result Reproduction Notes",
        "",
        "Figures are regenerated with the current simulator/model, using PSO-EULC, EEUMC, EBREC, EULC and LEACH.",
        "Each algorithm uses the same topology seed for a given scenario/seed.",
        "",
        f"Successful simulation runs: {len(ok)}",
    ]
    if not ok.empty:
        ok["final_residual_energy_pct"] = pd.to_numeric(ok["final_residual_energy_pct"], errors="coerce")
        ok["final_dead_nodes"] = pd.to_numeric(ok["final_dead_nodes"], errors="coerce")
        ok["pdr_pct"] = pd.to_numeric(ok["pdr_pct"], errors="coerce")
        grouped = ok.groupby("optimizer").agg(
            residual_pct=("final_residual_energy_pct", "mean"),
            dead_nodes=("final_dead_nodes", "mean"),
            pdr_pct=("pdr_pct", "mean"),
        )
        lines.append("")
        lines.append("Mean over all generated scenarios:")
        for optimizer in ALGORITHMS:
            if optimizer not in grouped.index:
                continue
            row = grouped.loc[optimizer]
            lines.append(
                f"- {LABELS[optimizer]}: residual={row['residual_pct']:.2f}%, "
                f"dead={row['dead_nodes']:.2f}, PDR={row['pdr_pct']:.2f}%"
            )
    output.write_text("\n".join(lines), encoding="utf-8")


def plot_figures(rounds: int | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RUNS_CSV.exists() or not ROUNDS_CSV.exists():
        print("No cached data yet; run experiments first.")
        return
    _migrate_rounds_csv_schema()
    runs_df = pd.read_csv(RUNS_CSV)
    rounds_df = pd.read_csv(ROUNDS_CSV)
    if rounds is not None:
        runs_df = runs_df[pd.to_numeric(runs_df["rounds_requested"], errors="coerce").eq(rounds)].copy()
        rounds_df = rounds_df[
            pd.to_numeric(rounds_df["rounds_requested"], errors="coerce").eq(rounds)
        ].copy()
        if runs_df.empty or rounds_df.empty:
            print(f"No cached data for rounds={rounds}; run experiments first.")
            return

    _plot_round_figure(
        rounds_df,
        "fig05_10",
        "residual_energy_j",
        "Residual energy (J)",
        "Figure 5: Residual energy vs rounds\nE0=1J, packet=6400 bits, R=200m, N=100",
        OUT_DIR / "figure_05_residual_energy_vs_rounds.png",
    )
    _plot_round_figure(
        rounds_df,
        "fig06_11",
        "residual_energy_j",
        "Residual energy (J)",
        "Figure 6: Residual energy vs rounds\nE0=1J, packet=4000 bits, R=150m, N=100",
        OUT_DIR / "figure_06_residual_energy_vs_rounds.png",
    )
    _plot_nodes_figure(
        runs_df,
        "fig07_12",
        "final_residual_energy_j",
        "Residual energy (J)",
        "Figure 7: Residual energy vs number of nodes\nE0=0.5J, packet=4000 bits, R=150m",
        OUT_DIR / "figure_07_residual_energy_vs_nodes.png",
    )
    _plot_nodes_figure(
        runs_df,
        "fig08_13",
        "final_residual_energy_j",
        "Residual energy (J)",
        "Figure 8: Residual energy vs number of nodes\nE0=0.5J, packet=6400 bits, R=200m",
        OUT_DIR / "figure_08_residual_energy_vs_nodes.png",
    )
    _plot_nodes_figure(
        runs_df,
        "fig09",
        "final_dead_nodes",
        "Dead nodes",
        "Figure 9: Dead nodes vs number of nodes\nE0=0.5J, packet=6400 bits, R=200m",
        OUT_DIR / "figure_09_dead_nodes_vs_nodes.png",
    )
    _plot_round_figure(
        rounds_df,
        "fig05_10",
        "dead_nodes",
        "Dead nodes",
        "Figure 10: Dead nodes vs rounds\nE0=1J, packet=6400 bits, R=200m, N=100",
        OUT_DIR / "figure_10_dead_nodes_vs_rounds.png",
    )
    _plot_round_figure(
        rounds_df,
        "fig06_11",
        "dead_nodes",
        "Dead nodes",
        "Figure 11: Dead nodes vs rounds\nE0=1J, packet=4000 bits, R=150m, N=100",
        OUT_DIR / "figure_11_dead_nodes_vs_rounds.png",
    )
    _plot_nodes_figure(
        runs_df,
        "fig07_12",
        "final_dead_nodes",
        "Dead nodes",
        "Figure 12: Dead nodes vs number of nodes\nE0=0.5J, packet=4000 bits, R=150m",
        OUT_DIR / "figure_12_dead_nodes_vs_nodes.png",
    )
    _plot_nodes_figure(
        runs_df,
        "fig08_13",
        "final_dead_nodes",
        "Dead nodes",
        "Figure 13: Dead nodes vs number of nodes\nE0=0.5J, packet=6400 bits, R=200m",
        OUT_DIR / "figure_13_dead_nodes_vs_nodes.png",
    )
    _plot_nodes_figure(
        runs_df,
        "fig14",
        "pdr_pct",
        "PDR / packet delivery (%)",
        "Figure 14: Packet delivery/PDR vs number of nodes\nE0=1J, packet=6400 bits, R=200m",
        OUT_DIR / "figure_14_pdr_vs_nodes.png",
    )
    _plot_average_bar(
        runs_df,
        "final_residual_energy_pct",
        "Residual energy (%)",
        "Figure 15: Average residual energy analysis",
        OUT_DIR / "figure_15_average_residual_energy_analysis.png",
    )
    _plot_average_bar(
        runs_df,
        "final_dead_nodes",
        "Dead nodes",
        "Figure 16: Average dead node analysis",
        OUT_DIR / "figure_16_average_dead_node_analysis.png",
    )
    _plot_statistical_analysis(runs_df, OUT_DIR / "figure_17_statistical_analysis.png")
    _write_notes(runs_df, OUT_DIR / "README_paper_result_reproduction.md")
    print(f"Saved figures to {OUT_DIR.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate paper-style result figures with the current model.")
    parser.add_argument("--runs", type=int, default=2, help="Seed count per scenario/algorithm.")
    parser.add_argument("--rounds", type=int, default=500, help="Rounds per simulation.")
    parser.add_argument("--max-node", type=int, default=500, help="Maximum node count included in node-sweep figures.")
    parser.add_argument("--base-seed", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many missing simulations.")
    parser.add_argument("--plot-only", action="store_true", help="Only redraw figures from cached CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.plot_only:
        run_missing_experiments(
            runs=max(1, args.runs),
            rounds=max(1, args.rounds),
            max_node=max(50, args.max_node),
            base_seed=args.base_seed,
            limit=args.limit,
        )
    plot_figures(rounds=max(1, args.rounds))


if __name__ == "__main__":
    main()
