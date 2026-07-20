from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


OUTPUT_CSV = Path("outputs/data/csv/output/experiment_outputs.csv")
CONVERGENCE_CSV = Path("outputs/data/csv/output/experiment_convergence.csv")
FIGURE_DIR = Path("outputs/figures/s100_s500_baseline_report")
SUMMARY_CSV = FIGURE_DIR / "s100_s500_baseline_summary.csv"

SPACES = (100, 500)
NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
}
OPTIMIZERS = ("pso", "ga", "de", "leach")
ITERATIVE_OPTIMIZERS = ("pso", "ga", "de")

COLORS = {
    "pso": "#4C78A8",
    "ga": "#E45756",
    "de": "#54A24B",
    "leach": "#9467BD",
}
MARKERS = {
    "pso": "o",
    "ga": "s",
    "de": "^",
    "leach": "D",
}
NODE_COLORS = {
    20: "#4C78A8",
    50: "#F58518",
    100: "#E45756",
    150: "#72B7B2",
    200: "#F58518",
    300: "#54A24B",
}
OPTIMIZER_STYLES = {
    "pso": "-",
    "ga": "--",
    "de": ":",
}

PARAM_NOTE = (
    "baseline; R=200m; packet=6400 bit; E0=0.5J; K_CH=0.2; "
    "candidate=0.4; particle=20; iter=50; c1=c2=1.5; "
    "3 phân phối x 15 seed = 45 mẫu/điểm"
)


def _load_metric_rows() -> pd.DataFrame:
    usecols = [
        "input_id",
        "distribution",
        "optimizer",
        "parameter_set",
        "space_m",
        "node_count",
        "seed",
        "transmission_range_m",
        "packet_size_bits",
        "initial_energy_j",
        "cluster_head_ratio",
        "eulc_candidate_ratio",
        "pso_particles",
        "pso_iterations",
        "pso_c1",
        "pso_c2",
        "ft5_round",
        "residual_energy_pct",
        "runtime_sec",
    ]
    df = pd.read_csv(OUTPUT_CSV, usecols=usecols)
    return df[
        df["space_m"].isin(SPACES)
        & df["node_count"].isin([node for nodes in NODE_SETS.values() for node in nodes])
        & df["optimizer"].isin(OPTIMIZERS)
        & df["parameter_set"].eq("baseline")
        & df["transmission_range_m"].eq(200)
        & df["packet_size_bits"].eq(6400)
        & df["initial_energy_j"].eq(0.5)
        & df["cluster_head_ratio"].eq(0.2)
        & df["eulc_candidate_ratio"].eq(0.4)
        & df["pso_particles"].eq(20)
        & df["pso_iterations"].eq(50)
        & df["pso_c1"].eq(1.5)
        & df["pso_c2"].eq(1.5)
    ].copy()


def _metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["space_m", "node_count", "optimizer"], as_index=False)
        .agg(
            runs=("input_id", "count"),
            ft5_mean=("ft5_round", "mean"),
            ft5_std=("ft5_round", "std"),
            residual_energy_pct_mean=("residual_energy_pct", "mean"),
            residual_energy_pct_std=("residual_energy_pct", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
        )
        .fillna(0.0)
    )
    return summary


def _nice_upper(value: float, step: float) -> float:
    if value <= 0:
        return step
    return math.ceil(value * 1.08 / step) * step


def _metric_ylim(summary: pd.DataFrame, mean_col: str, std_col: str, step: float) -> tuple[float, float]:
    upper = float((summary[mean_col] + summary[std_col]).max())
    return 0.0, _nice_upper(upper, step)


def _setup_common_metric_axis(
    ax,
    space_m: int,
    y_label: str,
    y_lim: tuple[float, float],
    y_step: float,
    show_ylabel: bool,
) -> None:
    from matplotlib.ticker import MultipleLocator

    ticks = list(NODE_SETS[space_m])
    pad = max(8, (max(ticks) - min(ticks)) * 0.08)
    ax.set_xlim(min(ticks) - pad, max(ticks) + pad)
    ax.set_xticks(ticks)
    ax.set_ylim(*y_lim)
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.set_xlabel("Số nút cảm biến")
    ax.set_ylabel(y_label if show_ylabel else "")
    ax.tick_params(axis="y", labelleft=True)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.32)


def _plot_metric(
    summary: pd.DataFrame,
    mean_col: str,
    std_col: str,
    title: str,
    y_label: str,
    y_step: float,
    output_name: str,
) -> Path:
    import matplotlib.pyplot as plt

    y_lim = _metric_ylim(summary, mean_col, std_col, y_step)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharex=False, sharey=True)
    for idx, (ax, space_m) in enumerate(zip(axes, SPACES)):
        panel = summary[summary["space_m"].eq(space_m)]
        for optimizer in OPTIMIZERS:
            series = panel[panel["optimizer"].eq(optimizer)].sort_values("node_count")
            if series.empty:
                continue
            ax.errorbar(
                series["node_count"],
                series[mean_col],
                yerr=series[std_col],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                markersize=5,
                linewidth=2,
                capsize=3,
                label=optimizer.upper(),
            )
        ax.set_title(f"Không gian {space_m} m", fontsize=11)
        _setup_common_metric_axis(ax, space_m, y_label, y_lim, y_step, show_ylabel=(idx == 0))
        expected = set(NODE_SETS[space_m])
        available = set(panel["node_count"].astype(int).unique())
        missing = sorted(expected - available)
        if missing:
            ax.text(
                0.04,
                0.93,
                "Thiếu n=" + ",".join(map(str, missing)),
                transform=ax.transAxes,
                fontsize=8,
                color="#666666",
                va="top",
            )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"{title}\n{PARAM_NOTE}", fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.07, right=0.99, wspace=0.10)
    output_path = FIGURE_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _iter_convergence_chunks() -> Iterable[pd.DataFrame]:
    usecols = [
        "optimizer",
        "parameter_set",
        "space_m",
        "node_count",
        "transmission_range_m",
        "packet_size_bits",
        "initial_energy_j",
        "cluster_head_ratio",
        "eulc_candidate_ratio",
        "pso_particles",
        "pso_iterations",
        "pso_c1",
        "pso_c2",
        "refresh_index",
        "iteration",
        "relative_best_score",
    ]
    for chunk in pd.read_csv(CONVERGENCE_CSV, usecols=usecols, chunksize=500_000):
        filtered = chunk[
            chunk["space_m"].isin(SPACES)
            & chunk["node_count"].isin([node for nodes in NODE_SETS.values() for node in nodes])
            & chunk["optimizer"].isin(ITERATIVE_OPTIMIZERS)
            & chunk["parameter_set"].eq("baseline")
            & chunk["transmission_range_m"].eq(200)
            & chunk["packet_size_bits"].eq(6400)
            & chunk["initial_energy_j"].eq(0.5)
            & chunk["cluster_head_ratio"].eq(0.2)
            & chunk["eulc_candidate_ratio"].eq(0.4)
            & chunk["pso_particles"].eq(20)
            & chunk["pso_iterations"].eq(50)
            & chunk["pso_c1"].eq(1.5)
            & chunk["pso_c2"].eq(1.5)
            & chunk["refresh_index"].eq(1)
            & chunk["relative_best_score"].notna()
        ]
        if not filtered.empty:
            yield filtered


def _load_convergence_summary() -> pd.DataFrame:
    frames = list(_iter_convergence_chunks())
    if not frames:
        return pd.DataFrame(
            columns=[
                "space_m",
                "node_count",
                "optimizer",
                "iteration",
                "relative_mean",
                "relative_std",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    return (
        df.groupby(["space_m", "node_count", "optimizer", "iteration"], as_index=False)
        .agg(
            relative_mean=("relative_best_score", "mean"),
            relative_std=("relative_best_score", "std"),
        )
        .fillna(0.0)
    )


def _plot_convergence(summary: pd.DataFrame) -> Path | None:
    if summary.empty:
        return None

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    available_optimizers = [opt for opt in ITERATIVE_OPTIMIZERS if opt in set(summary["optimizer"])]
    title_opts = ", ".join(opt.upper() for opt in available_optimizers)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharex=True, sharey=True)
    for idx, (ax, space_m) in enumerate(zip(axes, SPACES)):
        panel = summary[summary["space_m"].eq(space_m)]
        for optimizer in available_optimizers:
            for node_count in NODE_SETS[space_m]:
                series = panel[
                    panel["optimizer"].eq(optimizer) & panel["node_count"].eq(node_count)
                ].sort_values("iteration")
                if series.empty:
                    continue
                ax.plot(
                    series["iteration"],
                    series["relative_mean"],
                    color=NODE_COLORS[node_count],
                    linestyle=OPTIMIZER_STYLES[optimizer],
                    linewidth=2,
                    marker=MARKERS[optimizer],
                    markevery=5,
                    markersize=3.5,
                    alpha=0.95,
                )
        ax.set_title(f"Không gian {space_m} m", fontsize=11)
        ax.set_xlim(0, 50)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Số vòng lặp")
        ax.set_ylabel("Chi phí tốt nhất / chi phí ban đầu" if idx == 0 else "")
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.32)
        if panel.empty:
            ax.text(
                0.5,
                0.5,
                "Chưa có dữ liệu hội tụ",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#666666",
            )

    node_handles = [
        Line2D([0], [0], color=NODE_COLORS[n], lw=2, label=f"n={n}")
        for n in sorted({n for nodes in NODE_SETS.values() for n in nodes})
    ]
    optimizer_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=2,
            linestyle=OPTIMIZER_STYLES[opt],
            marker=MARKERS[opt],
            label=opt.upper(),
        )
        for opt in available_optimizers
    ]
    fig.legend(
        handles=node_handles + optimizer_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"So sánh mức hội tụ ({title_opts})\n{PARAM_NOTE}; refresh_index=1",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.07, right=0.99, wspace=0.10)
    output_path = FIGURE_DIR / "03_convergence_s100_s500.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = _load_metric_rows()
    if metric_rows.empty:
        raise SystemExit("No matching rows found in experiment_outputs.csv")

    summary = _metric_summary(metric_rows)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    outputs = [
        _plot_metric(
            summary,
            "ft5_mean",
            "ft5_std",
            "So sánh thuật toán theo vòng FT5",
            "Vòng FT5",
            250,
            "01_ft5_s100_s500.png",
        ),
        _plot_metric(
            summary,
            "residual_energy_pct_mean",
            "residual_energy_pct_std",
            "So sánh thuật toán theo năng lượng dư tại điểm dừng",
            "Năng lượng dư (%)",
            5,
            "02_residual_energy_s100_s500.png",
        ),
    ]

    convergence_output = _plot_convergence(_load_convergence_summary())
    if convergence_output is not None:
        outputs.append(convergence_output)

    outputs.append(
        _plot_metric(
            summary,
            "runtime_sec_mean",
            "runtime_sec_std",
            "So sánh thuật toán theo thời gian chạy",
            "Thời gian chạy (s)",
            25,
            "04_runtime_s100_s500.png",
        )
    )

    print(f"Metric rows: {len(metric_rows)}")
    print(f"Summary saved: {SUMMARY_CSV}")
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
