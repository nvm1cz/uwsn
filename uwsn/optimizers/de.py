from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..cases import SimulationCase
from ..run_config import TunableParams
from .common import (
    _build_paper_cost_normalization,
    _decode_priority_vector,
    _ensure_candidate_coverage,
    _finalize_solution,
    _normalize_solution,
    _target_cluster_head_count,
    evaluate_solution_score,
)


def run_de_cluster_head_selection(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    candidates: Sequence[int],
    rng: np.random.Generator,
    convergence_callback: Callable[[int, float], None] | None = None,
) -> Tuple[List[int], Dict[int, List[int]]]:
    candidates = _ensure_candidate_coverage(
        energies,
        distance_matrix,
        candidates,
        params.transmission_range_m,
        params.dead_energy_threshold_j,
    )
    candidate_count = len(candidates)
    if candidate_count == 0:
        return [], {}

    candidates_array = np.asarray(candidates, dtype=int)
    pop_size = max(4, int(params.pso_particles))
    dims = candidate_count
    active_node_count = int(np.count_nonzero(energies > params.dead_energy_threshold_j))
    target_ch_count = _target_cluster_head_count(
        active_node_count,
        candidate_count,
        params.cluster_head_ratio,
    )
    cost_normalization = _build_paper_cost_normalization(
        distance_matrix,
        energies,
        dist_to_sink,
        candidates,
        params.dead_energy_threshold_j,
    )
    score_cache: Dict[tuple[int, ...], float] = {}

    def decode(vector: np.ndarray) -> List[int]:
        return _decode_priority_vector(vector, candidates_array, target_ch_count)

    def score_vector(vector: np.ndarray) -> float:
        solution = tuple(
            _normalize_solution(
                energies,
                decode(vector),
                candidates,
                params.dead_energy_threshold_j,
            )
        )
        cached = score_cache.get(solution)
        if cached is not None:
            return cached
        score = evaluate_solution_score(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            solution,
            candidates,
            cost_normalization,
        )
        score_cache[solution] = score
        return score

    population = rng.uniform(0.0, 1.0, size=(pop_size, dims))
    scores = np.asarray([score_vector(vector) for vector in population], dtype=float)
    best_idx = int(np.argmin(scores))
    best = population[best_idx].copy()
    best_score = float(scores[best_idx])
    if convergence_callback is not None:
        convergence_callback(0, best_score)

    differential_weight = 0.55
    crossover_rate = 0.90

    for generation_idx in range(1, params.pso_iterations + 1):
        for idx in range(pop_size):
            choices = [pos for pos in range(pop_size) if pos != idx]
            a_idx, b_idx, c_idx = rng.choice(choices, size=3, replace=False)
            mutant = population[a_idx] + differential_weight * (population[b_idx] - population[c_idx])
            mutant = np.clip(mutant, 0.0, 1.0)

            crossover_mask = rng.random(dims) < crossover_rate
            if not np.any(crossover_mask):
                crossover_mask[int(rng.integers(0, dims))] = True
            trial = np.where(crossover_mask, mutant, population[idx])
            trial_score = score_vector(trial)
            if trial_score <= scores[idx]:
                population[idx] = trial
                scores[idx] = trial_score
                if trial_score < best_score:
                    best_score = float(trial_score)
                    best = trial.copy()

        if convergence_callback is not None:
            convergence_callback(generation_idx, best_score)

    return _finalize_solution(
        case,
        params,
        distance_matrix,
        energies,
        layers,
        dist_to_sink,
        decode(best),
        candidates,
        cost_normalization,
    )

