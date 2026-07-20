from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


SUMMARY_CSV = Path("outputs/data/csv/summary/report_tuned_optimizer_summary.csv")
CONVERGENCE_CSV = Path("outputs/data/csv/raw/report_tuned_optimizer_convergence_50iter.csv")
OUTPUT_DIR = Path("outputs/figures/report_approved_optimizer_comparison")

NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
    1000: (500, 1000, 1500, 2000),
}
OPTIMIZERS = ("pso", "ga", "de")
COLORS = {
    "pso": "#4C78A8",
    "ga": "#E45756",
    "de": "#54A24B",
}
MARKERS = {
    "pso": "o",
    "ga": "s",
    "de": "^",
}


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _nice_upper(max_value: float, tick_step: float) -> float:
    if max_value <= 0:
        return tick_step
    return max(tick_step, math.ceil(max_value * 1.10 / tick_step) * tick_step)


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _panel_rows(rows: List[Dict[str, str]], optimizer: str, space_m: int) -> List[Dict[str, str]]:
    panel = [
        row
        for row in rows
        if row["optimizer"] == optimizer
        and int(row["space_m"]) == space_m
        and int(row["node_count"]) in NODE_SETS[space_m]
    ]
    panel.sort(key=lambda row: int(row["node_count"]))
    return panel


def _setup_node_axis(ax, space_m: int, y_upper: float, y_step: float, ylabel: str) -> None:
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


def _legend(ax) -> None:
    ax.legend(loc="best", framealpha=0.92, fontsize=8)


def plot_ft5_mean_std(rows: List[Dict[str, str]]) -> Path:
    import matplotlib.pyplot as plt

    max_y = max(float(row["ft5_mean"]) + float(row["ft5_std"]) for row in rows)
    y_upper = _nice_upper(max_y, 250)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)

    for ax, space_m in zip(axes, [100, 500, 1000]):
        for optimizer in OPTIMIZERS:
            panel = _panel_rows(rows, optimizer, space_m)
            xs = [int(row["node_count"]) for row in panel]
            means = [float(row["ft5_mean"]) for row in panel]
            stds = [float(row["ft5_std"]) for row in panel]
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker=MARKERS[optimizer],
                linewidth=1.9,
                capsize=3,
                color=COLORS[optimizer],
                label=optimizer.upper(),
            )
        _setup_node_axis(ax, space_m, y_upper, 250, "Vòng FT5")
        _legend(ax)

    fig.suptitle("So sánh PSO, GA và DE theo vòng FT5")
    output_path = OUTPUT_DIR / "01_ft5_mean_std_pso_ga_de.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ft5_cv(rows: List[Dict[str, str]]) -> Path:
    import matplotlib.pyplot as plt

    max_y = max(float(row["ft5_cv_pct"]) for row in rows)
    y_upper = _nice_upper(max_y, 2.5)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)

    for ax, space_m in zip(axes, [100, 500, 1000]):
        for optimizer in OPTIMIZERS:
            panel = _panel_rows(rows, optimizer, space_m)
            xs = [int(row["node_count"]) for row in panel]
            cvs = [float(row["ft5_cv_pct"]) for row in panel]
            ax.plot(
                xs,
                cvs,
                marker=MARKERS[optimizer],
                linewidth=1.9,
                color=COLORS[optimizer],
                label=optimizer.upper(),
            )
        _setup_node_axis(ax, space_m, y_upper, 2.5, "Hệ số biến thiên FT5 (%)")
        _legend(ax)

    fig.suptitle("So sánh PSO, GA và DE theo độ ổn định FT5")
    output_path = OUTPUT_DIR / "03_ft5_cv_pso_ga_de.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_runtime_mean_std(rows: List[Dict[str, str]]) -> Path:
    import matplotlib.pyplot as plt

    max_y = max(float(row["runtime_mean_sec"]) + float(row["runtime_std_sec"]) for row in rows)
    if max_y <= 10:
        y_step = 2.5
    elif max_y <= 60:
        y_step = 10
    else:
        y_step = 25
    y_upper = _nice_upper(max_y, y_step)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)

    for ax, space_m in zip(axes, [100, 500, 1000]):
        for optimizer in OPTIMIZERS:
            panel = _panel_rows(rows, optimizer, space_m)
            xs = [int(row["node_count"]) for row in panel]
            means = [float(row["runtime_mean_sec"]) for row in panel]
            stds = [float(row["runtime_std_sec"]) for row in panel]
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker=MARKERS[optimizer],
                linewidth=1.9,
                capsize=3,
                color=COLORS[optimizer],
                label=optimizer.upper(),
            )
        _setup_node_axis(ax, space_m, y_upper, y_step, "Thời gian chạy (s)")
        _legend(ax)

    fig.suptitle("So sánh PSO, GA và DE theo thời gian chạy")
    output_path = OUTPUT_DIR / "06_runtime_mean_std_pso_ga_de.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _convergence_groups(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, int, int], List[float]]:
    grouped: Dict[Tuple[str, int, int], List[float]] = {}
    for row in rows:
        if row.get("relative_best_score") in ("", None):
            continue
        optimizer = row["optimizer"]
        if optimizer not in OPTIMIZERS:
            continue
        space_m = int(row["space_m"])
        if space_m not in NODE_SETS:
            continue
        iteration = int(row["iteration"])
        grouped.setdefault((optimizer, space_m, iteration), []).append(float(row["relative_best_score"]))
    return grouped


def plot_convergence(rows: List[Dict[str, str]]) -> Path:
    import matplotlib.pyplot as plt

    grouped = _convergence_groups(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)

    for ax, space_m in zip(axes, [100, 500, 1000]):
        for optimizer in OPTIMIZERS:
            iterations = sorted(iteration for opt, space, iteration in grouped if opt == optimizer and space == space_m)
            means = [mean(grouped[(optimizer, space_m, iteration)]) for iteration in iterations]
            stds = [_std(grouped[(optimizer, space_m, iteration)]) for iteration in iterations]
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
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)
        _legend(ax)

    fig.suptitle("So sánh PSO, GA và DE theo mức hội tụ")
    output_path = OUTPUT_DIR / "02_convergence_pso_ga_de.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    summary_rows = _load_csv(SUMMARY_CSV)
    convergence_rows = _load_csv(CONVERGENCE_CSV)
    outputs = [
        plot_ft5_mean_std(summary_rows),
        plot_convergence(convergence_rows),
        plot_ft5_cv(summary_rows),
        plot_runtime_mean_std(summary_rows),
    ]
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
