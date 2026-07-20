from __future__ import annotations

from pathlib import Path

import run_report_node_set_convergence_50iter as base


base.REPORT_NODE_SETS = {
    100: (20, 50, 100, 150),
    500: (100, 200, 300),
    1000: (500, 1000, 1500, 2000),
}
base.RAW_LOG = Path("outputs/data/csv/raw/report_convergence_tuned_nodes_50iter.csv")
base.SUMMARY_LOG = Path("outputs/data/csv/summary/report_convergence_tuned_nodes_50iter_summary.csv")
base.FIGURE_PATH = Path("outputs/figures/report_node_set_smooth/report_node_set_pso_convergence_by_node_count.png")
base.FIGURE_PATH_50 = Path("outputs/figures/report_node_set_smooth/report_convergence_tuned_nodes_50iter.png")


if __name__ == "__main__":
    base.main()
