from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List


SUMMARY_LOG = Path("outputs/data/csv/summary/report_node_set_smooth_summary.csv")
FIGURE_DIR = Path("outputs/figures/report_node_set_smooth")
ALGORITHM_LABEL = "PSO-EULC (PSO)"
SPACE_NODE_TICKS = {
    100: [50, 100, 150, 200],
    500: [100, 200, 300, 400, 500],
    1000: [200, 300, 400, 500],
}


def _load_rows() -> List[Dict[str, str]]:
    with SUMMARY_LOG.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _nice_upper(max_value: float, tick_step: float) -> float:
    if max_value <= 0:
        return tick_step
    return max(tick_step, math.ceil(max_value * 1.10 / tick_step) * tick_step)


def plot_ft5_line(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    max_y = max(float(row["ft5_mean"]) + float(row["ft5_std"]) for row in rows)
    y_upper = _nice_upper(max_y, 250)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, [100, 500, 1000]):
        panel = [row for row in rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        xs = [int(row["node_count"]) for row in panel]
        means = [float(row["ft5_mean"]) for row in panel]
        stds = [float(row["ft5_std"]) for row in panel]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2.0, capsize=4, color="#4C78A8")
        for x_value, mean_value, std_value in zip(xs, means, stds):
            ax.text(x_value, mean_value + std_value + y_upper * 0.025, f"{mean_value:.0f}", ha="center", fontsize=8)
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("FT5 round")
        ticks = SPACE_NODE_TICKS[space_m]
        pad = max(10, int((max(ticks) - min(ticks)) * 0.12))
        ax.set_xlim(min(ticks) - pad, max(ticks) + pad)
        ax.set_ylim(0, y_upper)
        ax.set_xticks(ticks)
        ax.set_yticks([250 * idx for idx in range(int(round(y_upper / 250)) + 1)])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(f"{ALGORITHM_LABEL} - FT5 mean +/- std across 30 seeds, contention factor=0.01")
    output_path = FIGURE_DIR / "report_node_set_ft5_line_errorbar.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_relative_ft5_line(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, [100, 500, 1000]):
        panel = [row for row in rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        baseline = float(panel[0]["ft5_mean"])
        xs = [int(row["node_count"]) for row in panel]
        rel_values = [float(row["ft5_mean"]) / baseline * 100.0 for row in panel]
        rel_stds = [float(row["ft5_std"]) / baseline * 100.0 for row in panel]
        ax.errorbar(xs, rel_values, yerr=rel_stds, marker="o", linewidth=2.0, capsize=4, color="#4C78A8")
        for x_value, value, std_value in zip(xs, rel_values, rel_stds):
            ax.text(x_value, value + std_value + 1.8, f"{value:.0f}%", ha="center", fontsize=8)
        ax.axhline(100, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("FT5 relative to lowest-node case (%)")
        ticks = SPACE_NODE_TICKS[space_m]
        pad = max(10, int((max(ticks) - min(ticks)) * 0.12))
        ax.set_xlim(min(ticks) - pad, max(ticks) + pad)
        ax.set_ylim(0, 125)
        ax.set_xticks(ticks)
        ax.set_yticks([0, 25, 50, 75, 100, 125])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(f"{ALGORITHM_LABEL} - Relative FT5 trend within each fixed space, 30 seeds")
    output_path = FIGURE_DIR / "report_node_set_ft5_relative_line_errorbar.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_cv_line(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    max_y = max(float(row["ft5_cv_pct"]) for row in rows)
    y_upper = _nice_upper(max_y, 2.5)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, [100, 500, 1000]):
        panel = [row for row in rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        xs = [int(row["node_count"]) for row in panel]
        cvs = [float(row["ft5_cv_pct"]) for row in panel]
        ax.plot(xs, cvs, marker="o", linewidth=2.0, color="#4C78A8")
        for x_value, cv_value in zip(xs, cvs):
            ax.text(x_value, cv_value + y_upper * 0.025, f"{cv_value:.1f}%", ha="center", fontsize=8)
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("FT5 CV (%)")
        ticks = SPACE_NODE_TICKS[space_m]
        pad = max(10, int((max(ticks) - min(ticks)) * 0.12))
        ax.set_xlim(min(ticks) - pad, max(ticks) + pad)
        ax.set_ylim(0, y_upper)
        ax.set_xticks(ticks)
        ax.set_yticks([2.5 * idx for idx in range(int(round(y_upper / 2.5)) + 1)])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(f"{ALGORITHM_LABEL} - FT5 coefficient of variation across 30 seeds")
    output_path = FIGURE_DIR / "report_node_set_ft5_cv_line.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_ft5_vs_neighbor_line(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    max_y = max(float(row["ft5_mean"]) + float(row["ft5_std"]) for row in rows)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for space_m in [100, 500, 1000]:
        panel = [row for row in rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: float(row["avg_neighbor_count_mean"]))
        xs = [float(row["avg_neighbor_count_mean"]) for row in panel]
        means = [float(row["ft5_mean"]) for row in panel]
        stds = [float(row["ft5_std"]) for row in panel]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2.0, capsize=4, label=f"space={space_m}m")
        for row, x_value, mean_value, std_value in zip(panel, xs, means, stds):
            ax.text(x_value, mean_value + std_value + 18, f"n={row['node_count']}", ha="center", fontsize=8)

    ax.set_title(f"{ALGORITHM_LABEL} - FT5 vs average neighbor count, 30 seeds")
    ax.set_xlabel("Average neighbors within transmission range")
    ax.set_ylabel("FT5 round")
    ax.set_ylim(0, _nice_upper(max_y, 250))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.92)
    output_path = FIGURE_DIR / "report_node_set_ft5_vs_avg_neighbor_count_line.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_runtime_line(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    max_y = max(float(row["runtime_mean_sec"]) for row in rows)
    y_upper = _nice_upper(max_y, 5)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True, constrained_layout=True)
    for ax, space_m in zip(axes, [100, 500, 1000]):
        panel = [row for row in rows if int(row["space_m"]) == space_m]
        panel.sort(key=lambda row: int(row["node_count"]))
        xs = [int(row["node_count"]) for row in panel]
        runtimes = [float(row["runtime_mean_sec"]) for row in panel]
        ax.plot(xs, runtimes, marker="o", linewidth=2.0, color="#4C78A8")
        for x_value, runtime_value in zip(xs, runtimes):
            ax.text(x_value, runtime_value + y_upper * 0.025, f"{runtime_value:.1f}s", ha="center", fontsize=8)
        ax.set_title(f"space={space_m}m")
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("Runtime per seed (s)")
        ticks = SPACE_NODE_TICKS[space_m]
        pad = max(10, int((max(ticks) - min(ticks)) * 0.12))
        ax.set_xlim(min(ticks) - pad, max(ticks) + pad)
        ax.set_ylim(0, y_upper)
        ax.set_xticks(ticks)
        ax.set_yticks([5 * idx for idx in range(int(round(y_upper / 5)) + 1)])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(f"{ALGORITHM_LABEL} - Runtime mean across 30 seeds")
    output_path = FIGURE_DIR / "report_node_set_runtime_line.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    rows = _load_rows()
    plot_ft5_line(rows)
    plot_relative_ft5_line(rows)
    plot_cv_line(rows)
    plot_ft5_vs_neighbor_line(rows)
    plot_runtime_line(rows)


if __name__ == "__main__":
    main()
