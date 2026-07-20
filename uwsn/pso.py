"""Backward-compatible exports for CH optimizers.

The implementation is split under ``uwsn.optimizers``. This module remains so
older scripts/tests importing ``uwsn.pso`` keep working.
"""

from __future__ import annotations

from .optimizers.common import (
    PaperCostNormalization,
    _build_paper_cost_normalization,
    _decode_priority_vector,
    _paper_solution_cost,
    _target_cluster_head_count,
    evaluate_solution,
    evaluate_solution_score,
    solution_cost,
)
from .optimizers.de import run_de_cluster_head_selection
from .optimizers.ebrec import run_ebrec_cluster_head_selection
from .optimizers.eeumc import run_eeumc_cluster_head_selection
from .optimizers.eulc import run_eulc_cluster_head_selection
from .optimizers.ga import run_ga_cluster_head_selection
from .optimizers.leach import run_leach_cluster_head_selection
from .optimizers.pso import run_pso_cluster_head_selection

__all__ = [
    "PaperCostNormalization",
    "_build_paper_cost_normalization",
    "_decode_priority_vector",
    "_paper_solution_cost",
    "_target_cluster_head_count",
    "evaluate_solution",
    "evaluate_solution_score",
    "solution_cost",
    "run_pso_cluster_head_selection",
    "run_ga_cluster_head_selection",
    "run_de_cluster_head_selection",
    "run_leach_cluster_head_selection",
    "run_eulc_cluster_head_selection",
    "run_eeumc_cluster_head_selection",
    "run_ebrec_cluster_head_selection",
]
