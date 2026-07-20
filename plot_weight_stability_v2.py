from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


SUMMARY_LOG = Path("outputs/data/csv/summary/stability_weight_v2_summary.csv")
OUTPUT_DIR = Path("outputs/figures/stability")


def _load_rows() -> List[Dict[str, str]]:
    with SUMMARY_LOG.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_weight_stability_chart(rows: List[Dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    labels = [
        (
            f"a={float(row['alpha_weight']):.2f}\n"
            f"b={float(row['beta_weight']):.2f}\n"
            f"g={float(row['gamma_weight']):.2f}"
        )
        for row in rows
    ]
    fnd_means = [float(row["fnd_mean"]) for row in rows]
    fnd_stds = [float(row["fnd_std"]) for row in rows]
    cvs = [float(row["fnd_cv_pct"]) for row in rows]
    x_positions = list(range(len(rows)))

    fig, ax1 = plt.subplots(figsize=(10.8, 5.8))
    bars = ax1.bar(
        x_positions,
        fnd_means,
        yerr=fnd_stds,
        capsize=5,
        color="#4C78A8",
        alpha=0.82,
        label="FND mean +/- std",
    )
    ax1.set_ylabel("FND round")
    ax1.set_xlabel("EULC alpha / beta / gamma")
    ax1.set_xticks(x_positions, labels)
    ax1.set_ylim(0, max(mean + std for mean, std in zip(fnd_means, fnd_stds)) * 1.18)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar, mean, cv in zip(bars, fnd_means, cvs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 24,
            f"{mean:.0f}\nCV={cv:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax1.twinx()
    ax2.plot(x_positions, cvs, marker="o", color="#E45756", linewidth=2.0, label="CV")
    ax2.set_ylabel("CV (%)")
    ax2.set_ylim(0, max(10.0, max(cvs) * 1.25))

    fig.suptitle("EULC alpha/beta/gamma stability sweep (space=100m, n=300, pkt=6400, Eini=0.5J)")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "stability_weight_v2_alpha_beta_gamma.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_weight_stability_chart(_load_rows())


if __name__ == "__main__":
    main()
