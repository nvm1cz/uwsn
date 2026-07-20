from __future__ import annotations

from typing import Callable, List

import numpy as np

from .cases import SimulationCase
from .clustering import select_eulc_candidates
from .deployment import compute_distances_to_sink, compute_layers, compute_neighbor_degree, compute_pairwise_distances, deploy_nodes, resolve_sink
from .energy import attenuation_array
from .metrics import RunMetrics
from .optimizers import (
    run_de_cluster_head_selection,
    run_ebrec_cluster_head_selection,
    run_eeumc_cluster_head_selection,
    run_eulc_cluster_head_selection,
    run_ga_cluster_head_selection,
    run_leach_cluster_head_selection,
    run_pso_cluster_head_selection,
)
from .run_config import TunableParams
from .routing import execute_round_transmissions


class PsoEulcSimulator:
    def __init__(
        self,
        case: SimulationCase,
        params: TunableParams,
        seed: int,
        verbose: bool = True,
        track_pso_convergence: bool = False,
        initial_positions: np.ndarray | None = None,
    ) -> None:
        self.case = case
        self.params = params
        self.seed = seed
        self.verbose = verbose
        self.track_pso_convergence = track_pso_convergence
        self.pso_convergence: list[dict[str, float]] = []
        self.cluster_snapshots: list[dict[str, object]] = []
        self._optimizer_refresh_index = 0
        self.np_rng = np.random.default_rng(seed)
        if initial_positions is None:
            self.positions = deploy_nodes(case, params, self.np_rng)
        else:
            positions = np.asarray(initial_positions, dtype=float)
            expected_shape = (case.node_count, 3)
            if positions.shape != expected_shape:
                raise ValueError(
                    f"initial_positions must have shape {expected_shape}, got {positions.shape}"
                )
            self.positions = positions.copy()
        self.energies = np.full(case.node_count, case.initial_energy, dtype=float)
        self.layers = compute_layers(self.positions, params)
        self.sink = resolve_sink(params)
        self.dist_to_sink = compute_distances_to_sink(self.positions, self.sink)
        self.distance_matrix = compute_pairwise_distances(self.positions)
        self.neighbor_degree = compute_neighbor_degree(self.positions, params, self.distance_matrix)
        self.neighbor_count = (
            np.count_nonzero(self.distance_matrix <= self.params.transmission_range_m, axis=1) - 1
        ).astype(float)
        e1_j = self.params.electronic_energy_e1_nj * 1e-9
        tx_scale = self.params.energy_calibration_factor * self.params.transmit_power_p0 * e1_j
        self.tx_cost_matrix_per_bit = tx_scale * attenuation_array(self.distance_matrix, self.params)
        self.tx_cost_to_sink_per_bit = tx_scale * attenuation_array(self.dist_to_sink, self.params)
        self.current_chs: list[int] = []
        self.current_assignments: dict[int, list[int]] = {}
        self.last_routing_edges: list[tuple[int, int | None]] = []
        self._clusters_printed = False

    def _alive_mask(self) -> np.ndarray:
        return self.energies > self.params.dead_energy_threshold_j

    def _need_recluster(self, round_idx: int) -> bool:
        if not self.current_assignments:
            return True
        if round_idx == 1:
            return True
        if (round_idx - 1) % max(1, self.params.recluster_interval) == 0:
            return True
        if any(self.energies[ch] <= self.params.dead_energy_threshold_j for ch in self.current_chs):
            return True
        return False

    def _refresh_clusters(self, round_idx: int) -> None:
        candidates = select_eulc_candidates(
            self.case,
            self.params,
            self.energies,
            self.layers,
            self.dist_to_sink,
            self.neighbor_degree,
            self.distance_matrix,
        )
        self._optimizer_refresh_index += 1
        optimizer = getattr(self.params, "optimizer", "pso").lower()

        def record_convergence(iteration: int, best_score: float) -> None:
            if not self.track_pso_convergence:
                return
            self.pso_convergence.append(
                {
                    "round": float(round_idx),
                    "refresh_index": float(self._optimizer_refresh_index),
                    "iteration": float(iteration),
                    "best_score": float(best_score),
                    "candidate_count": float(len(candidates)),
                }
            )

        optimizer_fn = {
            "pso": run_pso_cluster_head_selection,
            "ga": run_ga_cluster_head_selection,
            "de": run_de_cluster_head_selection,
            "leach": run_leach_cluster_head_selection,
            "eulc": run_eulc_cluster_head_selection,
            "eeumc": run_eeumc_cluster_head_selection,
            "ebrec": run_ebrec_cluster_head_selection,
        }.get(optimizer)
        if optimizer_fn is None:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        chs, assignments = optimizer_fn(
            self.case,
            self.params,
            self.distance_matrix,
            self.energies,
            self.layers,
            self.dist_to_sink,
            candidates,
            self.np_rng,
            convergence_callback=record_convergence if self.track_pso_convergence else None,
        )
        self.current_chs = [ch for ch in chs if self.energies[ch] > self.params.dead_energy_threshold_j]
        self.current_assignments = assignments
        self.cluster_snapshots.append(
            {
                "round": round_idx,
                "refresh_index": self._optimizer_refresh_index,
                "cluster_heads": list(self.current_chs),
                "assignments": {int(ch): list(members) for ch, members in assignments.items()},
            }
        )
        
        # Print clustering info only once after initial cluster formation
        if self.verbose and not self._clusters_printed:
            print("\n=== INITIAL CLUSTERING INFO ===")
            print(f"Number of Cluster Heads: {len(self.current_chs)}")
            print(f"Cluster Heads: {sorted(self.current_chs)}\n")
            for ch in sorted(self.current_chs):
                members = self.current_assignments.get(ch, [])
                alive_members = [m for m in members if self.energies[m] > self.params.dead_energy_threshold_j]
                ch_layer = int(self.layers[ch])
                ch_pos = self.positions[ch]
                print(f"CH {ch} (Layer {ch_layer}): Pos=({ch_pos[0]:.2f}, {ch_pos[1]:.2f}, {ch_pos[2]:.2f}), Members = {sorted(alive_members)} (Count: {len(alive_members)})")
            print("================================\n")
            self._clusters_printed = True

    def _run_round(self, round_idx: int, packets_received: int) -> int:
        if self._need_recluster(round_idx):
            self._refresh_clusters(round_idx)

        cluster_heads = [ch for ch in self.current_chs if self.energies[ch] > self.params.dead_energy_threshold_j]
        filtered_assignments = {
            ch: [member for member in members if self.energies[member] > self.params.dead_energy_threshold_j]
            for ch, members in self.current_assignments.items()
            if self.energies[ch] > self.params.dead_energy_threshold_j
        }
        self.last_routing_edges = []
        result = execute_round_transmissions(
            self.case,
            self.params,
            self.distance_matrix,
            self.tx_cost_matrix_per_bit,
            self.tx_cost_to_sink_per_bit,
            self.energies,
            self.layers,
            self.dist_to_sink,
            filtered_assignments,
            cluster_heads,
            packets_received,
            self.params.dead_energy_threshold_j,
            neighbor_count=self.neighbor_count,
            routing_edges=self.last_routing_edges,
        )
        
        if self.verbose and round_idx == 1:
            print("================================\n")
        
        return result

    def run(
        self,
        stop_on_first_dead: bool = True,
        stop_at_ft5: bool = False,
        min_alive_ratio: float | None = None,
        max_rounds: int | None = None,
        round_callback: Callable[["PsoEulcSimulator", int, int], None] | None = None,
    ) -> RunMetrics:
        residual_by_round: List[float] = []
        dead_by_round: List[int] = []
        packets_by_round: List[int] = []
        packets_received = 0
        fnd_round = None
        hnd_round = None
        lnd_round = None
        first_5pct_round = None

        round_idx = 0
        while True:
            round_idx += 1
            if max_rounds is not None and round_idx > max_rounds:
                break
            alive_before_round = int(np.count_nonzero(self._alive_mask()))
            if alive_before_round == 0:
                break

            packets_received = self._run_round(round_idx, packets_received)
            alive_after_round = int(np.count_nonzero(self._alive_mask()))
            dead_nodes = self.case.node_count - alive_after_round
            if first_5pct_round is None and dead_nodes >= 0.05 * self.case.node_count:
                first_5pct_round = round_idx
            if fnd_round is None and dead_nodes >= 1:
                fnd_round = round_idx

            if hnd_round is None and dead_nodes >= (self.case.node_count + 1) // 2:
                hnd_round = round_idx

            if lnd_round is None and alive_after_round == 0:
                lnd_round = round_idx

            residual_by_round.append(float(np.sum(self.energies)))
            dead_by_round.append(dead_nodes)
            packets_by_round.append(packets_received)

            if round_callback is not None:
                round_callback(self, round_idx, packets_received)

            # Stop simulation at the first dead node (FND condition).
            if stop_on_first_dead and dead_nodes >= 1:
                break

            if stop_at_ft5 and first_5pct_round is not None:
                break

            if min_alive_ratio is not None:
                alive_ratio = alive_after_round / max(1, self.case.node_count)
                if alive_ratio <= min_alive_ratio:
                    break

        # Print final cluster formation
        if self.verbose:
            print("\n=== FINAL CLUSTERING INFO (Last Round) ===")
            print(f"Number of Cluster Heads: {len(self.current_chs)}")
            print(f"Cluster Heads: {sorted(self.current_chs)}\n")
            for ch in sorted(self.current_chs):
                members = self.current_assignments.get(ch, [])
                alive_members = [m for m in members if self.energies[m] > self.params.dead_energy_threshold_j]
                ch_layer = int(self.layers[ch])
                ch_pos = self.positions[ch]
                ch_energy = self.energies[ch]
                print(f"CH {ch} (Layer {ch_layer}): Pos=({ch_pos[0]:.2f}, {ch_pos[1]:.2f}, {ch_pos[2]:.2f}), Energy={ch_energy:.6f}J, Members = {sorted(alive_members)} (Count: {len(alive_members)})")
            print("================================\n")

        return RunMetrics(
            seed=self.seed,
            fnd_round=fnd_round or None,
            hnd_round=hnd_round or None,
            lnd_round=lnd_round or None,
            first_5pct_round=first_5pct_round, 
            packets_received=packets_received,
            residual_energy=float(np.sum(self.energies)),
            alive_nodes=int(np.count_nonzero(self._alive_mask())),
            residual_energy_by_round=residual_by_round,
            dead_nodes_by_round=dead_by_round,
            packets_received_by_round=packets_by_round,
            pso_convergence=self.pso_convergence,
        )
