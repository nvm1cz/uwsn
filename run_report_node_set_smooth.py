from __future__ import annotations

from pathlib import Path

import run_report_node_set_density as base


base.RAW_LOG = Path("outputs/data/csv/raw/report_node_set_smooth_runs.csv")
base.CONVERGENCE_LOG = Path("outputs/data/csv/raw/report_node_set_smooth_convergence_events.csv")
base.SUMMARY_LOG = Path("outputs/data/csv/summary/report_node_set_smooth_summary.csv")
base.FIGURE_DIR = Path("outputs/figures/report_node_set_smooth")


def parse_args():
    args = base.parse_args()
    args.runs = 30 if args.runs == 10 else args.runs
    args.seed_start = 4000 if args.seed_start == 3042 else args.seed_start
    args.contention_penalty_factor = 0.01 if args.contention_penalty_factor == 0.35 else args.contention_penalty_factor
    return args


if __name__ == "__main__":
    base.run_all(parse_args())
