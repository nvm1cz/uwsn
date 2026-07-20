from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple


CONVERGENCE_CSV = Path("outputs/data/csv/raw/report_tuned_optimizer_convergence_50iter.csv")
OUTPUT_DIR = Path("outputs/figures/report_approved_optimizer_comparison")

NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
    1000: (500, 1000, 1500, 2000),
}
OPTIMIZERS = ("pso", "ga", "de")
NODE_COLORS = {
    20: "#4C78A8",
    50: "#F58518",
    100: "#E45756",
    150: "#E45756",
    200: "#F58518",
    300: "#54A24B",
    500: "#4C78A8",
    1000: "#F58518",
    1500: "#54A24B",
    2000: "#E45756",
}


def _load_rows() -> List[Dict[str, str]]:
    with CONVERGENCE_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return pstdev(values) if len(values) > 1 else 0.0


def _group_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[str, int, int, int], List[float]]:
    grouped: Dict[Tuple[str, int, int, int], List[float]] = {}
    for row in rows:
        if row.get("relative_best_score") in ("", None):
            continue
        optimizer = row["optimizer"]
        space_m = int(row["space_m"])
        node_count = int(row["node_count"])
        iteration = int(row["iteration"])
        if optimizer not in OPTIMIZERS:
            continue
        if space_m not in NODE_SETS or node_count not in NODE_SETS[space_m]:
            continue
        grouped.setdefault((optimizer, space_m, node_count, iteration), []).append(
            float(row["relative_best_score"])
        )
    return grouped


def plot_convergence_by_node() -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    grouped = _group_rows(_load_rows())
    fig, axes = plt.subplots(
        1,
        len(NODE_SETS),
        figsize=(16.4, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    optimizer_styles = {
        "pso": {"linestyle": "-", "marker": "o", "label": "PSO"},
        "ga": {"linestyle": "--", "marker": "s", "label": "GA"},
        "de": {"linestyle": ":", "marker": "^", "label": "DE"},
    }

    for col_idx, space_m in enumerate([100, 500, 1000]):
        ax = axes[col_idx]
        for optimizer in OPTIMIZERS:
            for node_count in NODE_SETS[space_m]:
                iterations = sorted(
                    iteration
                    for opt, space, node, iteration in grouped
                    if opt == optimizer and space == space_m and node == node_count
                )
                if not iterations:
                    continue
                means = [
                    mean(grouped[(optimizer, space_m, node_count, iteration)])
                    for iteration in iterations
                ]
                color = NODE_COLORS.get(node_count)
                style = optimizer_styles[optimizer]
                ax.plot(
                    iterations,
                    means,
                    marker=style["marker"],
                    markevery=5,
                    markersize=3.0,
                    linewidth=1.45,
                    linestyle=style["linestyle"],
                    color=color,
                    alpha=0.95,
                )

        ax.set_title(f"Không gian {space_m} m")
        if col_idx == 0:
            ax.set_ylabel("Chi phí tốt nhất / chi phí ban đầu")
        ax.set_xlabel("Số vòng lặp")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.35)

        node_handles = [
            Line2D([0], [0], color=NODE_COLORS[node_count], lw=2.0, label=f"n={node_count}")
            for node_count in NODE_SETS[space_m]
        ]
        optimizer_handles = [
            Line2D(
                [0],
                [0],
                color="#333333",
                lw=1.7,
                marker=style["marker"],
                linestyle=style["linestyle"],
                label=style["label"],
            )
            for style in optimizer_styles.values()
        ]
        first_legend = ax.legend(
            handles=node_handles,
            loc="upper right",
            framealpha=0.90,
            fontsize=7,
            title="Số nút",
            title_fontsize=8,
        )
        ax.add_artist(first_legend)
        ax.legend(
            handles=optimizer_handles,
            loc="lower left",
            framealpha=0.90,
            fontsize=7,
            title="Thuật toán",
            title_fontsize=8,
        )

    fig.suptitle("So sánh PSO, GA và DE theo mức hội tụ trên từng số nút")
    output_path = OUTPUT_DIR / "05_convergence_3panels_node_color_optimizer_style.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    print(f"Saved {plot_convergence_by_node()}")


if __name__ == "__main__":
    main()
