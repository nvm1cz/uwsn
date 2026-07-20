from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ALL_PARAM_ROOT = Path("outputs/figures/s100_s500_all_parameter_sets")
CONVERGENCE_SUMMARY_CSV = (
    ALL_PARAM_ROOT / "00_algorithm_convergence_reference" / "algorithm_convergence_summary.csv"
)
OUTPUT_DIR = Path("outputs/figures/s100_s500_best_by_metric")

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


def _load_metric_summaries() -> pd.DataFrame:
    index_path = ALL_PARAM_ROOT / "config_index.csv"
    if not index_path.exists():
        raise SystemExit(f"Missing {index_path}. Run plot_s100_s500_all_parameter_sets.py first.")

    index_df = pd.read_csv(index_path)
    frames: list[pd.DataFrame] = []
    for folder in sorted(path for path in ALL_PARAM_ROOT.iterdir() if path.is_dir()):
        summary_path = folder / "summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        summary["folder_name"] = folder.name
        frames.append(summary)
    if not frames:
        raise SystemExit(f"No summary.csv files found in {ALL_PARAM_ROOT}")

    rows = pd.concat(frames, ignore_index=True)
    return rows.merge(
        index_df[["folder_name", "config_note"]],
        on="folder_name",
        how="left",
    )


def _best_metric_rows(rows: pd.DataFrame, metric_col: str, better: str) -> pd.DataFrame:
    ascending = better == "min"
    sort_cols = ["space_m", "node_count", "optimizer", metric_col]
    sorted_rows = rows.sort_values(sort_cols, ascending=[True, True, True, ascending])
    best = sorted_rows.groupby(["space_m", "node_count", "optimizer"], as_index=False).head(1)
    return best.sort_values(["space_m", "node_count", "optimizer"]).reset_index(drop=True)


def _nice_upper(value: float, step: float) -> float:
    if value <= 0:
        return step
    return math.ceil(value * 1.08 / step) * step


def _nice_lower(value: float, step: float) -> float:
    if value >= 0:
        return 0.0
    return math.floor(value * 1.08 / step) * step


def _metric_ylim(best: pd.DataFrame, mean_col: str, std_col: str, step: float, force_zero: bool = True) -> tuple[float, float]:
    lower = float((best[mean_col] - best[std_col]).min())
    upper = float((best[mean_col] + best[std_col]).max())
    y_min = 0.0 if force_zero else _nice_lower(lower, step)
    y_max = _nice_upper(upper, step)
    return y_min, y_max


def _setup_node_axis(ax, space_m: int, y_label: str, y_lim: tuple[float, float], y_step: float, show_ylabel: bool) -> None:
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


def _plot_best_metric(
    best: pd.DataFrame,
    mean_col: str,
    std_col: str,
    title: str,
    subtitle: str,
    y_label: str,
    y_step: float,
    output_name: str,
) -> Path:
    import matplotlib.pyplot as plt

    y_lim = _metric_ylim(best, mean_col, std_col, y_step)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharex=False, sharey=True)
    for idx, (ax, space_m) in enumerate(zip(axes, SPACES)):
        panel = best[best["space_m"].eq(space_m)]
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
        _setup_node_axis(ax, space_m, y_label, y_lim, y_step, show_ylabel=(idx == 0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.07, right=0.99, wspace=0.10)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_convergence_best() -> pd.DataFrame:
    if not CONVERGENCE_SUMMARY_CSV.exists():
        raise SystemExit(
            f"Missing {CONVERGENCE_SUMMARY_CSV}. Run plot_s100_s500_algorithm_convergence.py first."
        )
    rows = pd.read_csv(CONVERGENCE_SUMMARY_CSV)
    rows = rows[
        rows["space_m"].isin(SPACES)
        & rows["optimizer"].isin(ITERATIVE_OPTIMIZERS)
        & rows.apply(lambda row: int(row["node_count"]) in NODE_SETS[int(row["space_m"])], axis=1)
    ].copy()
    final_iteration = int(rows["iteration"].max())
    final_rows = rows[rows["iteration"].eq(final_iteration)].copy()
    return final_rows.rename(
        columns={
            "relative_mean": "convergence_final_mean",
            "relative_std": "convergence_final_std",
        }
    )


def _plot_convergence_final(best: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    y_lim = _metric_ylim(best, "convergence_final_mean", "convergence_final_std", 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharex=False, sharey=True)
    for idx, (ax, space_m) in enumerate(zip(axes, SPACES)):
        panel = best[best["space_m"].eq(space_m)]
        for optimizer in ITERATIVE_OPTIMIZERS:
            series = panel[panel["optimizer"].eq(optimizer)].sort_values("node_count")
            if series.empty:
                continue
            ax.errorbar(
                series["node_count"],
                series["convergence_final_mean"],
                yerr=series["convergence_final_std"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                markersize=5,
                linewidth=2,
                capsize=3,
                label=optimizer.upper(),
            )
        ax.set_title(f"Không gian {space_m} m", fontsize=11)
        _setup_node_axis(
            ax,
            space_m,
            "Chi phí cuối / chi phí ban đầu",
            y_lim,
            0.1,
            show_ylabel=(idx == 0),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle(
        "So sánh thuật toán theo mức hội tụ tốt nhất\n"
        "Giá trị tại iteration cuối; thấp hơn tốt hơn; LEACH bỏ qua vì không có vòng lặp tối ưu",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.07, right=0.99, wspace=0.10)
    output_path = OUTPUT_DIR / "04_best_convergence_final.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_metric_summaries()

    ft5_best = _best_metric_rows(rows, "ft5_mean", "max")
    energy_best = _best_metric_rows(rows, "residual_energy_pct_mean", "max")
    runtime_best = _best_metric_rows(rows, "runtime_sec_mean", "min")
    convergence_best = _load_convergence_best()

    ft5_best.to_csv(OUTPUT_DIR / "best_ft5_details.csv", index=False, encoding="utf-8-sig")
    energy_best.to_csv(OUTPUT_DIR / "best_residual_energy_details.csv", index=False, encoding="utf-8-sig")
    runtime_best.to_csv(OUTPUT_DIR / "best_runtime_details.csv", index=False, encoding="utf-8-sig")
    convergence_best.to_csv(OUTPUT_DIR / "best_convergence_final_details.csv", index=False, encoding="utf-8-sig")

    outputs = [
        _plot_best_metric(
            ft5_best,
            "ft5_mean",
            "ft5_std",
            "So sánh cấu hình tốt nhất theo vòng FT5",
            "Mỗi điểm chọn bộ tham số có FT5 trung bình cao nhất; cao hơn tốt hơn",
            "Vòng FT5",
            250,
            "01_best_ft5.png",
        ),
        _plot_best_metric(
            energy_best,
            "residual_energy_pct_mean",
            "residual_energy_pct_std",
            "So sánh cấu hình tốt nhất theo năng lượng dư",
            "Mỗi điểm chọn bộ tham số có năng lượng dư trung bình cao nhất tại điểm dừng",
            "Năng lượng dư (%)",
            5,
            "02_best_residual_energy.png",
        ),
        _plot_best_metric(
            runtime_best,
            "runtime_sec_mean",
            "runtime_sec_std",
            "So sánh cấu hình tốt nhất theo thời gian chạy",
            "Mỗi điểm chọn bộ tham số có runtime trung bình thấp nhất; thấp hơn tốt hơn",
            "Thời gian chạy (s)",
            10,
            "03_best_runtime.png",
        ),
        _plot_convergence_final(convergence_best),
    ]

    readme = [
        "# S100-S500 best-by-metric figures",
        "",
        "Mỗi ảnh đặt `space=100m` và `space=500m` cạnh nhau.",
        "Trục hoành chỉ hiện đúng các số node được xét trong từng space.",
        "Trục tung dùng chung giữa hai panel.",
        "",
        "- `01_best_ft5.png`: chọn cấu hình có FT5 cao nhất cho từng thuật toán, space và số node.",
        "- `02_best_residual_energy.png`: chọn cấu hình có năng lượng dư cao nhất tại điểm dừng mô phỏng.",
        "- `03_best_runtime.png`: chọn cấu hình có runtime thấp nhất.",
        "- `04_best_convergence_final.png`: so sánh PSO/GA/DE bằng chi phí tương đối tại iteration cuối; LEACH không có hội tụ.",
        "",
        "Các file `best_*_details.csv` ghi rõ mỗi điểm dữ liệu lấy từ folder tham số nào.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")

    for output in outputs:
        print(f"Saved: {output}")
    print(f"Details saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
