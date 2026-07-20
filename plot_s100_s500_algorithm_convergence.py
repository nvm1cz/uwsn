from __future__ import annotations

from pathlib import Path

import pandas as pd


CONVERGENCE_CSV = Path("outputs/data/csv/raw/s100_s500_representative_optimizer_convergence.csv")
OUTPUT_DIR = Path("outputs/figures/s100_s500_all_parameter_sets/00_algorithm_convergence_reference")

SPACES = (100, 500)
NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
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


def _load_summary() -> tuple[pd.DataFrame, str]:
    usecols = [
        "optimizer",
        "space_m",
        "node_count",
        "distribution",
        "parameter_set",
        "packet_size_bits",
        "initial_energy_j",
        "transmission_range_m",
        "cluster_head_ratio",
        "pso_particles",
        "pso_iterations",
        "seed",
        "iteration",
        "relative_best_score",
    ]
    df = pd.read_csv(CONVERGENCE_CSV, usecols=usecols)
    valid_pairs = {(space, node) for space, nodes in NODE_SETS.items() for node in nodes}
    pairs = list(zip(df["space_m"].astype(int), df["node_count"].astype(int)))
    df = df[
        df["optimizer"].isin(OPTIMIZERS)
        & pd.Series([pair in valid_pairs for pair in pairs], index=df.index)
        & df["relative_best_score"].notna()
    ].copy()
    if df.empty:
        raise SystemExit(f"No matching convergence rows in {CONVERGENCE_CSV}")

    note = (
        "Aggregated by space across node counts, distributions, and seeds; "
        "LEACH is not applicable to convergence"
    )
    summary = (
        df.groupby(["space_m", "optimizer", "iteration"], as_index=False)
        .agg(relative_mean=("relative_best_score", "mean"), relative_std=("relative_best_score", "std"))
        .fillna(0.0)
    )
    return summary, note


def plot() -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    summary, note = _load_summary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "algorithm_convergence_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharex=True, sharey=True)
    for idx, (ax, space_m) in enumerate(zip(axes, SPACES)):
        panel = summary[summary["space_m"].eq(space_m)]
        for optimizer in OPTIMIZERS:
            series = panel[panel["optimizer"].eq(optimizer)].sort_values("iteration")
            if series.empty:
                continue
            x = series["iteration"].to_numpy()
            y = series["relative_mean"].to_numpy()
            std = series["relative_std"].to_numpy()
            ax.plot(
                x,
                y,
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                markevery=5,
                markersize=3.8,
                linewidth=2.2,
                alpha=0.98,
                label=optimizer.upper(),
            )
            ax.fill_between(
                x,
                (y - std).clip(min=0.0),
                (y + std).clip(max=1.05),
                color=COLORS[optimizer],
                alpha=0.10,
                linewidth=0,
            )
        ax.set_title(f"Space {space_m} m", fontsize=11)
        ax.set_xlim(0, 50)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best cost / initial cost" if idx == 0 else "")
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.32)

    optimizer_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[optimizer],
            lw=2,
            marker=MARKERS[optimizer],
            label=optimizer.upper(),
        )
        for optimizer in OPTIMIZERS
    ]
    fig.legend(
        handles=optimizer_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(f"Convergence Comparison Between PSO, GA, and DE\n{note}", fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.07, right=0.99, wspace=0.10)
    output_path = OUTPUT_DIR / "convergence_pso_ga_de_s100_s500.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    print(f"Saved: {plot()}")


if __name__ == "__main__":
    main()
