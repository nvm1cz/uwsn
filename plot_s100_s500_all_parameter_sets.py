from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

import pandas as pd


OUTPUT_CSV = Path("outputs/data/csv/output/experiment_outputs.csv")
CONVERGENCE_CSV = Path("outputs/data/csv/output/experiment_convergence.csv")
FIGURE_ROOT = Path("outputs/figures/s100_s500_all_parameter_sets")

SPACES = (100, 500)
NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
}
OPTIMIZERS = ("pso", "ga", "de", "leach")
ITERATIVE_OPTIMIZERS = ("pso", "ga", "de")

CONFIG_COLS = [
    "parameter_set",
    "transmission_range_m",
    "packet_size_bits",
    "initial_energy_j",
    "alpha_weight",
    "beta_weight",
    "gamma_weight",
    "cluster_head_ratio",
    "eulc_candidate_ratio",
    "pso_particles",
    "pso_iterations",
    "pso_inertia",
    "pso_c1",
    "pso_c2",
    "pso_omega",
]

METRIC_COLS = CONFIG_COLS + [
    "input_id",
    "distribution",
    "optimizer",
    "space_m",
    "node_count",
    "seed",
    "ft5_round",
    "residual_energy_pct",
    "runtime_sec",
]

CONVERGENCE_COLS = CONFIG_COLS + [
    "optimizer",
    "space_m",
    "node_count",
    "refresh_index",
    "iteration",
    "relative_best_score",
]

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


def _fmt_num(value: float | int) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _slug_num(value: float | int) -> str:
    return _fmt_num(value).replace(".", "p")


def _safe_slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _config_slug(row: pd.Series) -> str:
    parts = [
        _safe_slug(str(row["parameter_set"])),
        f"k{_slug_num(row['cluster_head_ratio'])}",
        f"p{int(row['pso_particles'])}",
        f"it{int(row['pso_iterations'])}",
        f"w{_slug_num(row['pso_omega'])}",
        f"a{_slug_num(row['alpha_weight'])}",
        f"b{_slug_num(row['beta_weight'])}",
        f"g{_slug_num(row['gamma_weight'])}",
    ]
    return "_".join(parts)


def _config_note(row: pd.Series) -> str:
    return (
        f"{row['parameter_set']}; R={_fmt_num(row['transmission_range_m'])}m; "
        f"packet={int(row['packet_size_bits'])} bit; E0={_fmt_num(row['initial_energy_j'])}J; "
        f"K_CH={_fmt_num(row['cluster_head_ratio'])}; "
        f"alpha/beta/gamma={_fmt_num(row['alpha_weight'])}/"
        f"{_fmt_num(row['beta_weight'])}/{_fmt_num(row['gamma_weight'])}; "
        f"candidate={_fmt_num(row['eulc_candidate_ratio'])}; "
        f"particle={int(row['pso_particles'])}; iter={int(row['pso_iterations'])}; "
        f"inertia={_fmt_num(row['pso_inertia'])}; c1=c2={_fmt_num(row['pso_c1'])}; "
        f"omega={_fmt_num(row['pso_omega'])}"
    )


def _load_metric_rows() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_CSV, usecols=METRIC_COLS)
    return df[
        df["space_m"].isin(SPACES)
        & df["node_count"].isin([node for nodes in NODE_SETS.values() for node in nodes])
        & df["optimizer"].isin(OPTIMIZERS)
    ].copy()


def _build_configs(metric_rows: pd.DataFrame) -> pd.DataFrame:
    configs = metric_rows[CONFIG_COLS].drop_duplicates().copy()
    configs = configs.sort_values(
        ["parameter_set", "cluster_head_ratio", "pso_particles", "pso_omega"],
        kind="stable",
    ).reset_index(drop=True)
    configs["config_slug"] = configs.apply(_config_slug, axis=1)
    configs["folder_name"] = [
        f"{index:02d}_{slug}" for index, slug in enumerate(configs["config_slug"], start=1)
    ]
    configs["config_note"] = configs.apply(_config_note, axis=1)
    return configs


def _attach_config(rows: pd.DataFrame, configs: pd.DataFrame) -> pd.DataFrame:
    return rows.merge(configs[CONFIG_COLS + ["folder_name", "config_note"]], on=CONFIG_COLS, how="inner")


def _metric_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["folder_name", "space_m", "node_count", "optimizer"], as_index=False)
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


def _run_label(runs: pd.Series) -> str:
    values = sorted(int(value) for value in runs.dropna().unique())
    if not values:
        return "0 mẫu/điểm"
    if len(values) == 1:
        return f"{values[0]} mẫu/điểm"
    return f"{values[0]}-{values[-1]} mẫu/điểm"


def _nice_upper(value: float, step: float) -> float:
    if value <= 0:
        return step
    return math.ceil(value * 1.08 / step) * step


def _metric_ylim(summary: pd.DataFrame, mean_col: str, std_col: str, step: float) -> tuple[float, float]:
    upper = float((summary[mean_col] + summary[std_col]).max())
    return 0.0, _nice_upper(upper, step)


def _setup_metric_axis(
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
    folder: Path,
    summary: pd.DataFrame,
    config_note: str,
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
        _setup_metric_axis(ax, space_m, y_label, y_lim, y_step, show_ylabel=(idx == 0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle(f"{title}\n{config_note}; {_run_label(summary['runs'])}", fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.07, right=0.99, wspace=0.10)
    output_path = folder / output_name
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_convergence_summary(configs: pd.DataFrame) -> pd.DataFrame:
    if not CONVERGENCE_CSV.exists() or CONVERGENCE_CSV.stat().st_size == 0:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(CONVERGENCE_CSV, usecols=CONVERGENCE_COLS, chunksize=500_000):
        filtered = chunk[
            chunk["space_m"].isin(SPACES)
            & chunk["node_count"].isin([node for nodes in NODE_SETS.values() for node in nodes])
            & chunk["optimizer"].isin(ITERATIVE_OPTIMIZERS)
            & chunk["refresh_index"].eq(1)
            & chunk["relative_best_score"].notna()
        ]
        if filtered.empty:
            continue
        merged = _attach_config(filtered, configs)
        if merged.empty:
            continue
        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    rows = pd.concat(frames, ignore_index=True)
    return (
        rows.groupby(["folder_name", "space_m", "node_count", "optimizer", "iteration"], as_index=False)
        .agg(relative_mean=("relative_best_score", "mean"), relative_std=("relative_best_score", "std"))
        .fillna(0.0)
    )


def _plot_convergence(
    folder: Path,
    summary: pd.DataFrame,
    config_note: str,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharex=True, sharey=True)
    available_optimizers = [opt for opt in ITERATIVE_OPTIMIZERS if opt in set(summary.get("optimizer", []))]
    for idx, (ax, space_m) in enumerate(zip(axes, SPACES)):
        panel = summary[summary["space_m"].eq(space_m)] if not summary.empty else summary
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

    handles = [
        Line2D([0], [0], color=NODE_COLORS[n], lw=2, label=f"n={n}")
        for n in sorted({node for nodes in NODE_SETS.values() for node in nodes})
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=2,
            linestyle=OPTIMIZER_STYLES[optimizer],
            marker=MARKERS[optimizer],
            label=optimizer.upper(),
        )
        for optimizer in available_optimizers
    )
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), frameon=False)
    optimizer_label = ", ".join(opt.upper() for opt in available_optimizers) if available_optimizers else "không có"
    fig.suptitle(
        f"So sánh mức hội tụ ({optimizer_label})\n{config_note}; refresh_index=1",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.07, right=0.99, wspace=0.10)
    output_path = folder / "03_convergence.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_readme(index_rows: pd.DataFrame) -> None:
    lines = [
        "# S100-S500 all parameter-set figures",
        "",
        "Mỗi thư mục con tương ứng một bộ tham số. Trong từng thư mục:",
        "- `01_ft5.png`: so sánh FT5 giữa các thuật toán.",
        "- `02_residual_energy.png`: so sánh năng lượng dư tại điểm dừng mô phỏng.",
        "- `03_convergence.png`: hội tụ theo iteration nếu file convergence có dữ liệu; nếu không có sẽ ghi rõ trong hình.",
        "- `04_runtime.png`: so sánh thời gian chạy.",
        "",
        "Biểu đồ hội tụ so sánh PSO/GA/DE được đặt riêng trong `00_algorithm_convergence_reference` vì LEACH không có vòng lặp tối ưu và file convergence theo input chính không lưu đủ GA/DE cho mọi bộ tham số.",
        "",
        "`config_index.csv` là bảng tra cứu folder và tham số đầy đủ.",
        "",
        "Các điểm dữ liệu gộp 3 phân phối node và các seed tương ứng trong cùng bộ tham số.",
    ]
    (FIGURE_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    index_rows.to_csv(FIGURE_ROOT / "config_index.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    metric_rows = _load_metric_rows()
    if metric_rows.empty:
        raise SystemExit("No s100/s500 metric rows found.")

    configs = _build_configs(metric_rows)
    metric_rows = _attach_config(metric_rows, configs)
    metric_summary = _metric_summary(metric_rows)
    convergence_summary = _load_convergence_summary(configs)

    if FIGURE_ROOT.exists():
        shutil.rmtree(FIGURE_ROOT)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    index_rows = []
    saved_count = 0
    for _, config in configs.iterrows():
        folder = FIGURE_ROOT / str(config["folder_name"])
        folder.mkdir(parents=True, exist_ok=True)
        config_summary = metric_summary[metric_summary["folder_name"].eq(config["folder_name"])]
        config_convergence = (
            convergence_summary[convergence_summary["folder_name"].eq(config["folder_name"])]
            if not convergence_summary.empty
            else pd.DataFrame()
        )
        config_summary.to_csv(folder / "summary.csv", index=False, encoding="utf-8-sig")
        _plot_metric(
            folder,
            config_summary,
            str(config["config_note"]),
            "ft5_mean",
            "ft5_std",
            "So sánh thuật toán theo vòng FT5",
            "Vòng FT5",
            250,
            "01_ft5.png",
        )
        _plot_metric(
            folder,
            config_summary,
            str(config["config_note"]),
            "residual_energy_pct_mean",
            "residual_energy_pct_std",
            "So sánh thuật toán theo năng lượng dư tại điểm dừng",
            "Năng lượng dư (%)",
            5,
            "02_residual_energy.png",
        )
        _plot_convergence(folder, config_convergence, str(config["config_note"]))
        _plot_metric(
            folder,
            config_summary,
            str(config["config_note"]),
            "runtime_sec_mean",
            "runtime_sec_std",
            "So sánh thuật toán theo thời gian chạy",
            "Thời gian chạy (s)",
            25,
            "04_runtime.png",
        )
        saved_count += 4
        algorithms = ",".join(
            sorted(config_summary["optimizer"].dropna().str.upper().unique().tolist())
        )
        index_rows.append(
            {
                "folder_name": config["folder_name"],
                "algorithms": algorithms,
                "runs_per_point": _run_label(config_summary["runs"]),
                "has_convergence": not config_convergence.empty,
                "config_note": config["config_note"],
            }
        )

    index_df = pd.DataFrame(index_rows)
    _write_readme(index_df)
    print(f"Configs: {len(configs)}")
    print(f"Figures saved: {saved_count}")
    print(f"Output root: {FIGURE_ROOT}")
    print(f"Index: {FIGURE_ROOT / 'config_index.csv'}")


if __name__ == "__main__":
    main()
