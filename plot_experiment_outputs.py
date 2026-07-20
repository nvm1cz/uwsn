from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
    1000: (500, 1000, 1500, 2000),
}
OPTIMIZERS = ("pso", "ga", "de", "leach", "eulc", "eeumc", "ebrec")
ITERATIVE_OPTIMIZERS = ("pso", "ga", "de")
COLORS = {
    "pso": "#4C78A8",
    "ga": "#E45756",
    "de": "#54A24B",
    "leach": "#9467BD",
    "eulc": "#F58518",
    "eeumc": "#B279A2",
    "ebrec": "#72B7B2",
}
MARKERS = {
    "pso": "o",
    "ga": "s",
    "de": "^",
    "leach": "D",
    "eulc": "v",
    "eeumc": "P",
    "ebrec": "X",
}
FIGURE_NOTES = {
    "ft5": "FT5: vòng khi 5% node đầu tiên chết.\nSo sánh tuổi thọ mạng: cao hơn tốt hơn.",
    "convergence": "Hội tụ: best cost / initial best cost.\nSo sánh PSO, GA, DE: giảm nhanh hơn tốt hơn.",
    "ft5_std": "Độ lệch chuẩn FT5 trong cùng nhóm.\nSo sánh độ ổn định: thấp hơn ổn định hơn.",
    "runtime": "Runtime: thời gian chạy một input.\nSo sánh chi phí tính toán: thấp hơn nhanh hơn.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot report figures from fixed experiment input/output CSVs."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/csv/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data/csv/output"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures/experiment_outputs"))
    parser.add_argument("--distribution", default="all")
    parser.add_argument("--parameter-set", default="all")
    parser.add_argument("--transmission-range-m", type=float, default=None)
    parser.add_argument("--packet-size-bits", type=int, default=None)
    parser.add_argument("--initial-energy-j", type=float, default=None)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--c1", type=float, default=None)
    parser.add_argument("--c2", type=float, default=None)
    parser.add_argument(
        "--include-leach",
        action="store_true",
        help="Include LEACH/EULC/EEUMC/EBREC direct baselines in non-convergence plots.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def nice_upper(max_value: float, tick_step: float) -> float:
    if max_value <= 0:
        return tick_step
    return max(tick_step, math.ceil(max_value * 1.10 / tick_step) * tick_step)


def runtime_tick_step(max_value: float) -> float:
    if max_value <= 10:
        return 2.5
    if max_value <= 60:
        return 10
    return 25


def add_figure_note(fig, note: str) -> None:
    fig.text(
        0.985,
        0.5,
        note,
        ha="right",
        va="center",
        fontsize=8,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ffffff",
            "edgecolor": "#cccccc",
            "alpha": 0.95,
        },
    )


def row_matches(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if args.distribution != "all" and row["distribution"] != args.distribution:
        return False
    if args.parameter_set != "all" and row["parameter_set"] != args.parameter_set:
        return False
    if args.transmission_range_m is not None and abs(float(row["transmission_range_m"]) - args.transmission_range_m) >= 1e-9:
        return False
    if args.packet_size_bits is not None and int(float(row["packet_size_bits"])) != args.packet_size_bits:
        return False
    if args.initial_energy_j is not None and abs(float(row["initial_energy_j"]) - args.initial_energy_j) >= 1e-9:
        return False
    if args.particles is not None and int(float(row["pso_particles"])) != args.particles:
        return False
    if args.c1 is not None and abs(float(row["pso_c1"]) - args.c1) >= 1e-9:
        return False
    if args.c2 is not None and abs(float(row["pso_c2"]) - args.c2) >= 1e-9:
        return False
    return int(row["space_m"]) in NODE_SETS and int(row["node_count"]) in NODE_SETS[int(row["space_m"])]


def optimizer_list(include_leach: bool) -> Tuple[str, ...]:
    return OPTIMIZERS if include_leach else ITERATIVE_OPTIMIZERS


def filter_rows(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    optimizers = set(optimizer_list(args.include_leach))
    return [row for row in rows if row_matches(row, args) and row["optimizer"] in optimizers]


def input_completion_report(
    expected_rows: Sequence[Dict[str, str]],
    output_rows: Sequence[Dict[str, str]],
) -> Tuple[int, int]:
    expected_ids = {row["input_id"] for row in expected_rows}
    completed_ids = {row["input_id"] for row in output_rows if row["input_id"] in expected_ids}
    return len(completed_ids), len(expected_ids)


def filter_label(args: argparse.Namespace) -> str:
    range_label = "all" if args.transmission_range_m is None else f"{args.transmission_range_m:g} m"
    return f"distribution={args.distribution}, parameter_set={args.parameter_set}, R={range_label}"


def build_summary(rows: List[Dict[str, str]], include_leach: bool) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int], List[Dict[str, str]]] = {}
    for row in rows:
        optimizer = row["optimizer"]
        if optimizer not in optimizer_list(include_leach):
            continue
        grouped.setdefault(
            (optimizer, int(row["space_m"]), int(row["node_count"])),
            [],
        ).append(row)

    summary: List[Dict[str, object]] = []
    for optimizer in optimizer_list(include_leach):
        for space_m, node_counts in NODE_SETS.items():
            for node_count in node_counts:
                group = grouped.get((optimizer, space_m, node_count), [])
                if not group:
                    continue
                ft5_values = [float(row["ft5_round"]) for row in group if row.get("ft5_round")]
                runtime_values = [float(row["runtime_sec"]) for row in group if row.get("runtime_sec")]
                if not ft5_values:
                    continue
                avg_ft5 = mean(ft5_values)
                ft5_std = std(ft5_values)
                summary.append(
                    {
                        "optimizer": optimizer,
                        "space_m": space_m,
                        "node_count": node_count,
                        "runs": len(ft5_values),
                        "ft5_mean": avg_ft5,
                        "ft5_std": ft5_std,
                        "ft5_cv_pct": (ft5_std / avg_ft5 * 100.0) if len(ft5_values) >= 2 and avg_ft5 else None,
                        "runtime_mean_sec": mean(runtime_values) if runtime_values else 0.0,
                        "runtime_std_sec": std(runtime_values) if runtime_values else 0.0,
                    }
                )
    return summary


def run_count_label(summary: Sequence[Dict[str, object]]) -> str:
    runs = sorted({int(row["runs"]) for row in summary})
    if not runs:
        return "0 seed/case"
    if len(runs) == 1:
        return f"{runs[0]} seed/case"
    return f"{runs[0]}-{runs[-1]} seed/case"


def missing_nodes_for_panel(
    summary: Sequence[Dict[str, object]],
    include_leach: bool,
    space_m: int,
) -> List[int]:
    available = {
        int(row["node_count"])
        for row in summary
        if int(row["space_m"]) == space_m and str(row["optimizer"]) in optimizer_list(include_leach)
    }
    return [node for node in NODE_SETS[space_m] if node not in available]


def report_coverage(summary: Sequence[Dict[str, object]], include_leach: bool) -> None:
    available = {
        (str(row["optimizer"]), int(row["space_m"]), int(row["node_count"]))
        for row in summary
    }
    missing: List[str] = []
    for optimizer in optimizer_list(include_leach):
        for space_m, node_counts in NODE_SETS.items():
            missing_nodes = [
                str(node_count)
                for node_count in node_counts
                if (optimizer, space_m, node_count) not in available
            ]
            if missing_nodes:
                missing.append(f"{optimizer.upper()} space={space_m}: n={','.join(missing_nodes)}")
    print(f"Matched plotted groups: {len(available)}")
    if missing:
        print("Missing groups:")
        for item in missing:
            print(f"  - {item}")


def setup_node_axis(ax, space_m: int, y_upper: float, y_step: float, ylabel: str) -> None:
    ticks = list(NODE_SETS[space_m])
    pad = max(5, int((max(ticks) - min(ticks)) * 0.10))
    ax.set_title(f"Không gian {space_m} m")
    ax.set_xlabel("Số nút cảm biến")
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(ticks) - pad, max(ticks) + pad)
    ax.set_xticks(ticks)
    ax.set_ylim(0, y_upper)
    ax.set_yticks([y_step * idx for idx in range(int(round(y_upper / y_step)) + 1)])
    ax.tick_params(axis="y", labelleft=True)
    ax.grid(True, linestyle="--", alpha=0.35)


def panel_rows(summary: List[Dict[str, object]], optimizer: str, space_m: int) -> List[Dict[str, object]]:
    rows = [
        row for row in summary
        if row["optimizer"] == optimizer and int(row["space_m"]) == space_m
    ]
    rows.sort(key=lambda row: int(row["node_count"]))
    return rows


def available_nodes_for_panel(summary: Sequence[Dict[str, object]], space_m: int) -> List[int]:
    nodes = sorted({int(row["node_count"]) for row in summary if int(row["space_m"]) == space_m})
    return nodes


def plot_line_metric(
    summary: List[Dict[str, object]],
    args: argparse.Namespace,
    completion: Tuple[int, int],
    metric: str,
    std_metric: str | None,
    ylabel: str,
    title: str,
    filename: str,
    y_step: float,
    note: str,
) -> Path:
    import matplotlib.pyplot as plt

    optimizers = optimizer_list(args.include_leach)
    has_metric_data = any(row[metric] is not None for row in summary)
    max_value = 0.0
    for row in summary:
        if row[metric] is None:
            continue
        value = float(row[metric])
        if std_metric is not None:
            value += float(row[std_metric])
        max_value = max(max_value, value)
    y_upper = nice_upper(max_value, y_step)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, (100, 500, 1000)):
        available_nodes = available_nodes_for_panel(summary, space_m)
        for optimizer in optimizers:
            rows = panel_rows(summary, optimizer, space_m)
            rows = [row for row in rows if row[metric] is not None]
            if not rows:
                continue
            xs = [int(row["node_count"]) for row in rows]
            ys = [float(row[metric]) for row in rows]
            yerr = [float(row[std_metric]) for row in rows] if std_metric is not None else None
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker=MARKERS[optimizer],
                linewidth=1.9,
                markersize=7.5 if len(available_nodes) <= 1 else 4.8,
                capsize=3,
                color=COLORS[optimizer],
                label=optimizer.upper(),
            )
            if len(available_nodes) == 1:
                ax.annotate(
                    optimizer.upper(),
                    (xs[0], ys[0]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    color=COLORS[optimizer],
                )
        setup_node_axis(ax, space_m, y_upper, y_step, ylabel)
        if not has_metric_data:
            ax.text(
                0.5,
                0.5,
                "Can >= 2 seed/case de tinh chi so nay",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
            )
        missing_nodes = missing_nodes_for_panel(summary, args.include_leach, space_m)
        if missing_nodes:
            if len(available_nodes) == 1:
                ax.text(
                    0.02,
                    0.88,
                    f"Da chay n={available_nodes[0]}",
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                    color="#333333",
                )
            ax.text(
                0.02,
                0.95,
                "Thieu n=" + ",".join(str(node) for node in missing_nodes),
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                color="#666666",
            )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", framealpha=0.92, fontsize=8)
    fig.suptitle(
        f"{title}\n{filter_label(args)}; completed outputs: {completion[0]}/{completion[1]}; "
        f"group runs: {run_count_label(summary)}",
        fontsize=11,
    )
    add_figure_note(fig, note)
    output_path = args.figure_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def convergence_groups(rows: Iterable[Dict[str, str]], args: argparse.Namespace) -> Dict[Tuple[str, int, int], List[float]]:
    grouped: Dict[Tuple[str, int, int], List[float]] = {}
    for row in rows:
        if row.get("relative_best_score") in ("", None):
            continue
        if not row_matches(row, args):
            continue
        optimizer = row["optimizer"]
        if optimizer not in ITERATIVE_OPTIMIZERS:
            continue
        if int(row.get("refresh_index", 1)) != 1:
            continue
        grouped.setdefault(
            (optimizer, int(row["space_m"]), int(row["iteration"])),
            [],
        ).append(float(row["relative_best_score"]))
    return grouped


def plot_convergence(rows: List[Dict[str, str]], args: argparse.Namespace, completion: Tuple[int, int]) -> Path | None:
    import matplotlib.pyplot as plt

    grouped = convergence_groups(rows, args)
    if not grouped:
        return None
    max_iteration = max(iteration for _, _, iteration in grouped)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, (100, 500, 1000)):
        for optimizer in ITERATIVE_OPTIMIZERS:
            iterations = sorted(
                iteration for opt, space, iteration in grouped
                if opt == optimizer and space == space_m
            )
            if not iterations:
                continue
            means = [mean(grouped[(optimizer, space_m, iteration)]) for iteration in iterations]
            stds = [std(grouped[(optimizer, space_m, iteration)]) for iteration in iterations]
            lower = [max(0.0, avg - sd) for avg, sd in zip(means, stds)]
            upper = [min(1.1, avg + sd) for avg, sd in zip(means, stds)]
            ax.plot(
                iterations,
                means,
                marker=MARKERS[optimizer],
                markersize=2.7,
                linewidth=1.8,
                color=COLORS[optimizer],
                label=optimizer.upper(),
            )
            ax.fill_between(iterations, lower, upper, color=COLORS[optimizer], alpha=0.10)
        ax.set_title(f"Không gian {space_m} m")
        ax.set_xlabel("Số vòng lặp")
        ax.set_ylabel("Chi phí tốt nhất / chi phí ban đầu")
        ax.set_xlim(0, max(50, max_iteration))
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)
        if not any(space == space_m for _, space, _ in grouped):
            ax.text(
                0.5,
                0.5,
                "Chua co du lieu hoi tu",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
            )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", framealpha=0.92, fontsize=8)
    fig.suptitle("So sánh PSO, GA và DE theo mức hội tụ")
    add_figure_note(fig, FIGURE_NOTES["convergence"])
    output_path = args.figure_dir / "02_convergence_from_experiment_inputs.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    expected_rows = filter_rows(load_csv(args.input_dir / "experiment_inputs.csv"), args)
    if not expected_rows:
        raise SystemExit("No matching experiment inputs found.")
    expected_ids = {row["input_id"] for row in expected_rows}
    output_rows = [
        row
        for row in load_csv(args.output_dir / "experiment_outputs.csv")
        if row["input_id"] in expected_ids
    ]
    if not output_rows:
        raise SystemExit(
            f"No matching experiment outputs found for {len(expected_rows)} input rows. "
            "Run run_experiment_inputs.py for this filter first."
        )
    completion = input_completion_report(expected_rows, output_rows)
    print(f"Experiment input rows matched: {len(expected_rows)}")
    print(f"Completed output rows: {completion[0]}/{completion[1]}")
    summary = build_summary(output_rows, args.include_leach)
    if not summary:
        raise SystemExit("Matching outputs exist, but no complete FT5 rows were found.")
    report_coverage(summary, args.include_leach)

    runtime_max = max(float(row["runtime_mean_sec"]) + float(row["runtime_std_sec"]) for row in summary)
    outputs = [
        plot_line_metric(
            summary,
            args,
            completion,
            "ft5_mean",
            "ft5_std",
            "Vòng FT5",
            "So sánh thuật toán theo vòng FT5",
            "01_ft5_from_experiment_inputs.png",
            250,
            FIGURE_NOTES["ft5"],
        ),
        plot_line_metric(
            summary,
            args,
            completion,
            "ft5_std",
            None,
            "Độ lệch chuẩn vòng FT5",
            "So sánh thuật toán theo độ lệch chuẩn vòng FT5",
            "03_ft5_std_from_experiment_inputs.png",
            25,
            FIGURE_NOTES["ft5_std"],
        ),
        plot_line_metric(
            summary,
            args,
            completion,
            "runtime_mean_sec",
            "runtime_std_sec",
            "Thời gian chạy (s)",
            "So sánh thuật toán theo thời gian chạy",
            "04_runtime_from_experiment_inputs.png",
            runtime_tick_step(runtime_max),
            FIGURE_NOTES["runtime"],
        ),
    ]
    convergence_rows = [
        row
        for row in load_csv(args.output_dir / "experiment_convergence.csv")
        if row["input_id"] in expected_ids
    ]
    convergence_output = plot_convergence(convergence_rows, args, completion)
    if convergence_output is not None:
        outputs.insert(1, convergence_output)

    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
