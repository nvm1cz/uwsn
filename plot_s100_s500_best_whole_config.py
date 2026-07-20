from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pandas as pd


ALL_PARAM_ROOT = Path("outputs/figures/s100_s500_all_parameter_sets")
EXPERIMENT_OUTPUTS_CSV = Path("outputs/data/csv/output/experiment_outputs.csv")
CONVERGENCE_CSV = Path("outputs/data/csv/raw/s100_s500_representative_optimizer_convergence.csv")
OUTPUT_DIR = Path("outputs/figures/s100_s500_best_whole_config")

SPACES = (100, 500)
NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
}
OPTIMIZERS = ("pso", "ga", "de", "leach")
TUNED_OPTIMIZERS = ("pso", "ga", "de")
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
PARAM_COLS = [
    "optimizer",
    "parameter_set",
    "space_m",
    "node_count",
    "distribution",
    "seed",
    "transmission_range_m",
    "packet_size_bits",
    "initial_energy_j",
    "max_rounds",
    "min_alive_ratio",
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
    "contention_penalty_factor",
    "contention_reference_degree",
    "contention_max_multiplier",
]


def _load_all_summaries() -> pd.DataFrame:
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
    return pd.concat(frames, ignore_index=True).merge(
        index_df[["folder_name", "config_note", "algorithms"]],
        on="folder_name",
        how="left",
    )


def _load_experiment_outputs() -> pd.DataFrame:
    if not EXPERIMENT_OUTPUTS_CSV.exists():
        return pd.DataFrame(columns=PARAM_COLS)
    return pd.read_csv(EXPERIMENT_OUTPUTS_CSV, usecols=lambda col: col in PARAM_COLS)


def _select_whole_config(rows: pd.DataFrame, metric_col: str, better: str) -> pd.Series:
    # Select by PSO/GA/DE only so LEACH does not bias a tuned-optimizer choice.
    scoped = rows[rows["optimizer"].isin(TUNED_OPTIMIZERS)]
    scores = (
        scoped.groupby("folder_name", as_index=False)
        .agg(
            score=(metric_col, "mean"),
            config_note=("config_note", "first"),
            algorithms=("algorithms", "first"),
        )
        .sort_values("score", ascending=(better == "min"))
        .reset_index(drop=True)
    )
    return scores.iloc[0]


def _nice_upper(value: float, step: float) -> float:
    if value <= 0:
        return step
    return math.ceil(value * 1.08 / step) * step


def _metric_ylim(summary: pd.DataFrame, mean_col: str, std_col: str, step: float) -> tuple[float, float]:
    upper = float((summary[mean_col] + summary[std_col]).max())
    return 0.0, _nice_upper(upper, step)


def _setup_node_axis(ax, space_m: int, y_label: str, y_lim: tuple[float, float], y_step: float, show_ylabel: bool) -> None:
    from matplotlib.ticker import MultipleLocator

    ticks = list(NODE_SETS[space_m])
    pad = max(8, (max(ticks) - min(ticks)) * 0.08)
    ax.set_xlim(min(ticks) - pad, max(ticks) + pad)
    ax.set_xticks(ticks)
    ax.set_ylim(*y_lim)
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.set_xlabel("Number of sensor nodes")
    ax.set_ylabel(y_label if show_ylabel else "")
    ax.tick_params(axis="y", labelleft=True)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.32)


def _run_label(summary: pd.DataFrame) -> str:
    runs = sorted(int(value) for value in summary["runs"].dropna().unique())
    if not runs:
        return "0 runs/point"
    if len(runs) == 1:
        return f"{runs[0]} runs/point"
    return f"{runs[0]}-{runs[-1]} runs/point"


def _plot_metric_config(
    summary: pd.DataFrame,
    selected: pd.Series,
    mean_col: str,
    std_col: str,
    title: str,
    selection_note: str,
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
        ax.set_title(f"Space {space_m} m", fontsize=11)
        _setup_node_axis(ax, space_m, y_label, y_lim, y_step, show_ylabel=(idx == 0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle(
        f"{title}\n{selection_note}; see the parameter table for full settings",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(top=0.75, bottom=0.18, left=0.07, right=0.99, wspace=0.10)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_convergence_rows() -> pd.DataFrame:
    if not CONVERGENCE_CSV.exists():
        raise SystemExit(f"Missing {CONVERGENCE_CSV}. Run run_representative_convergence.py first.")
    rows = pd.read_csv(CONVERGENCE_CSV)
    rows = rows[
        rows["space_m"].isin(SPACES)
        & rows["optimizer"].isin(TUNED_OPTIMIZERS)
        & rows.apply(lambda row: int(row["node_count"]) in NODE_SETS[int(row["space_m"])], axis=1)
        & rows["relative_best_score"].notna()
    ].copy()
    return rows


def _load_convergence_summary(convergence_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        convergence_rows.groupby(["space_m", "optimizer", "iteration"], as_index=False)
        .agg(relative_mean=("relative_best_score", "mean"), relative_std=("relative_best_score", "std"))
        .fillna(0.0)
    )


def _plot_convergence(summary: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharex=True, sharey=True)
    for idx, (ax, space_m) in enumerate(zip(axes, SPACES)):
        panel = summary[summary["space_m"].eq(space_m)]
        for optimizer in TUNED_OPTIMIZERS:
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
        for optimizer in TUNED_OPTIMIZERS
    ]
    fig.legend(handles=optimizer_handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "Representative Convergence Comparison\nAggregated by space; see the parameter table for full settings",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.07, right=0.99, wspace=0.10)
    output_path = OUTPUT_DIR / "04_representative_convergence.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_config_note(config_note: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    parts = [part.strip() for part in str(config_note).split(";") if part.strip()]
    if parts:
        parsed["parameter_set"] = parts[0]
    for part in parts[1:]:
        if part.startswith("alpha/beta/gamma="):
            values = part.split("=", 1)[1].split("/")
            parsed["alpha_weight"], parsed["beta_weight"], parsed["gamma_weight"] = values
        elif part.startswith("c1=c2="):
            value = part.split("=", 2)[2]
            parsed["pso_c1"] = value
            parsed["pso_c2"] = value
        elif "=" in part:
            key, value = part.split("=", 1)
            mapped_key = {
                "R": "transmission_range_m",
                "packet": "packet_size_bits",
                "E0": "initial_energy_j",
                "K_CH": "cluster_head_ratio",
                "candidate": "eulc_candidate_ratio",
                "particle": "pso_particles",
                "iter": "pso_iterations",
                "inertia": "pso_inertia",
                "omega": "pso_omega",
            }.get(key.strip(), key.strip())
            parsed[mapped_key] = value.strip().replace("m", "").replace(" bit", "").replace("J", "")
    return parsed


def _matches_config(rows: pd.DataFrame, selected: pd.Series) -> pd.Series:
    parsed = _parse_config_note(selected["config_note"])
    mask = pd.Series(True, index=rows.index)
    text_cols = {"parameter_set"}
    numeric_cols = {
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
    }
    for key, value in parsed.items():
        if key not in rows.columns:
            continue
        if key in text_cols:
            mask &= rows[key].astype(str).eq(str(value))
        elif key in numeric_cols:
            mask &= (pd.to_numeric(rows[key], errors="coerce") - float(value)).abs().lt(1e-9)
    mask &= rows["space_m"].isin(SPACES)
    mask &= rows.apply(lambda row: int(row["node_count"]) in NODE_SETS[int(row["space_m"])], axis=1)
    return mask


def _format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


def _compact_values(values: pd.Series) -> str:
    unique = sorted(pd.Series(values).dropna().unique().tolist())
    if not unique:
        return ""
    if all(isinstance(value, (int, float)) or str(value).replace(".", "", 1).isdigit() for value in unique):
        nums = [int(float(value)) if abs(float(value) - round(float(value))) < 1e-9 else float(value) for value in unique]
        if len(nums) > 3 and all(isinstance(value, int) for value in nums):
            contiguous = nums == list(range(nums[0], nums[-1] + 1))
            if contiguous:
                return f"{nums[0]}-{nums[-1]} ({len(nums)} values)"
        return ", ".join(_format_number(value) for value in nums)
    return ", ".join(str(value) for value in unique)


def _node_set_label(frame: pd.DataFrame) -> str:
    parts = []
    for space_m in SPACES:
        nodes = sorted(frame[frame["space_m"].eq(space_m)]["node_count"].dropna().astype(int).unique())
        if nodes:
            parts.append(f"s{space_m}: " + ",".join(str(node) for node in nodes))
    return "; ".join(parts)


def _settings_from_frame(
    frame: pd.DataFrame,
    chart_label: str,
    selection_rule: str,
    fallback_algorithms: str = "",
) -> dict[str, str]:
    first = frame.iloc[0] if not frame.empty else pd.Series(dtype=object)
    if "optimizer" in frame and not frame.empty:
        algorithms = ", ".join(str(value).upper() for value in sorted(frame["optimizer"].dropna().unique()))
    else:
        algorithms = fallback_algorithms
    distributions = _compact_values(frame["distribution"]) if "distribution" in frame and not frame.empty else ""
    seeds = _compact_values(frame["seed"]) if "seed" in frame and not frame.empty else ""
    spaces = _compact_values(frame["space_m"]) if "space_m" in frame and not frame.empty else "100, 500"
    if frame.empty:
        runs_per_point = ""
    elif "iteration" in frame.columns:
        case_frame = frame.drop_duplicates(["space_m", "node_count", "optimizer", "distribution", "seed"])
        case_count = int(case_frame.groupby(["space_m", "node_count", "optimizer"]).size().median())
        runs_per_point = f"{case_count} runs x {frame['iteration'].nunique()} iterations"
    else:
        runs_per_point = str(int(frame.groupby(["space_m", "node_count", "optimizer"]).size().median()))
    return {
        "Chart": chart_label,
        "Selection": selection_rule,
        "Algorithms": algorithms,
        "Spaces (m)": spaces,
        "Node counts": _node_set_label(frame),
        "Distributions": distributions,
        "Seeds": seeds,
        "Runs per point": runs_per_point,
        "Parameter set": str(first.get("parameter_set", "")),
        "Transmission range (m)": _format_number(first.get("transmission_range_m", "")),
        "Packet size (bit)": _format_number(first.get("packet_size_bits", "")),
        "Initial energy (J)": _format_number(first.get("initial_energy_j", "")),
        "Max rounds": _format_number(first.get("max_rounds", "")),
        "Stop alive ratio": _format_number(first.get("min_alive_ratio", "")),
        "alpha": _format_number(first.get("alpha_weight", "")),
        "beta": _format_number(first.get("beta_weight", "")),
        "gamma": _format_number(first.get("gamma_weight", "")),
        "K_CH": _format_number(first.get("cluster_head_ratio", "")),
        "Candidate ratio": _format_number(first.get("eulc_candidate_ratio", "")),
        "Population size": _format_number(first.get("pso_particles", "")),
        "Iterations": _format_number(first.get("pso_iterations", "")),
        "Inertia": _format_number(first.get("pso_inertia", "")),
        "c1": _format_number(first.get("pso_c1", "")),
        "c2": _format_number(first.get("pso_c2", "")),
        "Cost omega": _format_number(first.get("pso_omega", "")),
        "Contention penalty": (
            f"factor={_format_number(first.get('contention_penalty_factor', ''))}; "
            f"ref={_format_number(first.get('contention_reference_degree', ''))}; "
            f"max={_format_number(first.get('contention_max_multiplier', ''))}"
        ),
    }


def _settings_from_selection(
    outputs: pd.DataFrame,
    selected: pd.Series,
    chart_label: str,
    selection_rule: str,
) -> dict[str, str]:
    if outputs.empty:
        parsed = _parse_config_note(selected["config_note"])
        return {"Chart": chart_label, "Selection": selection_rule, "Algorithms": selected.get("algorithms", ""), **parsed}
    frame = outputs[_matches_config(outputs, selected)].copy()
    return _settings_from_frame(frame, chart_label, selection_rule, fallback_algorithms=str(selected.get("algorithms", "")))


def _write_parameter_table(table: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    csv_path = OUTPUT_DIR / "05_representative_parameter_table.csv"
    table.to_csv(csv_path, index=True, index_label="Parameter", encoding="utf-8-sig")

    display = table.astype(str).map(lambda value: textwrap.fill(value, width=34, break_long_words=False))
    display.insert(0, "Parameter", display.index)
    cell_text = display.values.tolist()
    headers = display.columns.tolist()

    fig, ax = plt.subplots(figsize=(18.5, 11.0))
    ax.axis("off")
    ax.set_title("Parameter Settings Used by the Representative Charts", fontsize=16, weight="bold", pad=18)
    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        colWidths=[0.20, 0.20, 0.20, 0.20, 0.20],
        bbox=[0.01, 0.01, 0.98, 0.91],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.2)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#E8EEF7")
            cell.set_text_props(weight="bold", color="#111827")
        elif col == 0:
            cell.set_facecolor("#F6F8FB")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#FBFCFE")
    output_path = OUTPUT_DIR / "05_representative_parameter_table.png"
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_all_summaries()
    experiment_outputs = _load_experiment_outputs()

    selections = {
        "ft5": _select_whole_config(rows, "ft5_mean", "max"),
        "energy": _select_whole_config(rows, "residual_energy_pct_mean", "max"),
        "runtime": _select_whole_config(rows, "runtime_sec_mean", "min"),
    }
    pd.DataFrame(selections).T.to_csv(
        OUTPUT_DIR / "selected_whole_configs.csv",
        index_label="criterion",
        encoding="utf-8-sig",
    )

    outputs: list[Path] = []
    ft5_summary = rows[rows["folder_name"].eq(selections["ft5"]["folder_name"])]
    outputs.append(
        _plot_metric_config(
            ft5_summary,
            selections["ft5"],
            "ft5_mean",
            "ft5_std",
            "Representative FT5 Comparison",
            "Configuration selected by highest mean FT5 over PSO/GA/DE",
            "FT5 round",
            250,
            "01_representative_best_ft5.png",
        )
    )

    energy_summary = rows[rows["folder_name"].eq(selections["energy"]["folder_name"])]
    outputs.append(
        _plot_metric_config(
            energy_summary,
            selections["energy"],
            "residual_energy_pct_mean",
            "residual_energy_pct_std",
            "Representative Residual Energy Comparison",
            "Configuration selected by highest mean residual energy over PSO/GA/DE",
            "Residual energy (%)",
            5,
            "02_representative_best_residual_energy.png",
        )
    )

    runtime_summary = rows[rows["folder_name"].eq(selections["runtime"]["folder_name"])]
    outputs.append(
        _plot_metric_config(
            runtime_summary,
            selections["runtime"],
            "runtime_sec_mean",
            "runtime_sec_std",
            "Representative Runtime Comparison",
            "Configuration selected by lowest mean runtime over PSO/GA/DE",
            "Runtime (s)",
            10,
            "03_representative_best_runtime.png",
        )
    )

    convergence_rows = _load_convergence_rows()
    convergence_summary = _load_convergence_summary(convergence_rows)
    outputs.append(_plot_convergence(convergence_summary))
    convergence_summary.to_csv(OUTPUT_DIR / "representative_convergence_summary.csv", index=False, encoding="utf-8-sig")

    parameter_settings = {
        "FT5": _settings_from_selection(
            experiment_outputs,
            selections["ft5"],
            "FT5",
            "Highest mean FT5 over PSO/GA/DE",
        ),
        "Residual Energy": _settings_from_selection(
            experiment_outputs,
            selections["energy"],
            "Residual Energy",
            "Highest mean residual energy over PSO/GA/DE",
        ),
        "Runtime": _settings_from_selection(
            experiment_outputs,
            selections["runtime"],
            "Runtime",
            "Lowest mean runtime over PSO/GA/DE",
        ),
        "Convergence": _settings_from_frame(
            convergence_rows,
            "Convergence",
            "Baseline convergence-only run; aggregated by space",
        ),
    }
    parameter_table = pd.DataFrame(parameter_settings)
    outputs.append(_write_parameter_table(parameter_table))

    readme = [
        "# S100-S500 representative figures",
        "",
        "Each metric figure uses one selected parameter configuration for the whole chart.",
        "Full parameter values are kept in `05_representative_parameter_table.png` and `05_representative_parameter_table.csv` instead of being printed inside every figure.",
        "",
        "- `01_representative_best_ft5.png`: selected by highest mean FT5 over PSO/GA/DE.",
        "- `02_representative_best_residual_energy.png`: selected by highest mean residual energy over PSO/GA/DE.",
        "- `03_representative_best_runtime.png`: selected by lowest mean runtime over PSO/GA/DE.",
        "- `04_representative_convergence.png`: PSO/GA/DE convergence aggregated by space; LEACH is not applicable.",
        "- `05_representative_parameter_table.png`: full parameter settings used by the four figures.",
        "",
        "`selected_whole_configs.csv` records which configuration was selected for each metric.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")

    for output in outputs:
        print(f"Saved: {output}")
    print(f"Selections: {OUTPUT_DIR / 'selected_whole_configs.csv'}")


if __name__ == "__main__":
    main()

