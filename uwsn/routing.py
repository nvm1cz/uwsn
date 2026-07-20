from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .cases import SimulationCase
from .run_config import TunableParams


def contention_multiplier(params: TunableParams, neighbor_count: np.ndarray | float | int) -> np.ndarray:
    factor = float(getattr(params, "contention_penalty_factor", 0.0))
    if factor <= 0:
        return np.ones_like(np.asarray(neighbor_count, dtype=float), dtype=float)

    reference = max(float(getattr(params, "contention_reference_degree", 10.0)), 1.0)
    max_multiplier = max(float(getattr(params, "contention_max_multiplier", 4.0)), 1.0)
    multiplier = 1.0 + factor * (np.asarray(neighbor_count, dtype=float) / reference)
    return np.minimum(max_multiplier, multiplier)


def send_energy_from_distance_cost(distance_cost: float, bits: int) -> float:
    return bits * distance_cost


def receive_energy(params: TunableParams, bits: int) -> float:
    return params.energy_calibration_factor * bits * params.processing_energy_nj * 1e-9


def aggregate_energy(params: TunableParams, bits: int) -> float:
    return params.energy_calibration_factor * bits * params.electronic_energy_e1_nj * 1e-9


def can_transmit_to_sink(params: TunableParams, dist_to_sink_m: float) -> bool:
    return dist_to_sink_m <= params.transmission_range_m


def routing_selection_costs(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    dist_to_sink: np.ndarray,
    current_ch: int,
    candidate_chs: np.ndarray,
) -> np.ndarray:
    """Equation P(i,j) from Algorithm 1; lower is better."""
    epsilon = float(params.balance_factor)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("balance_factor epsilon must be in [0, 1]")

    candidate_chs = np.asarray(candidate_chs, dtype=int)
    if candidate_chs.size == 0:
        return np.asarray([], dtype=float)

    residual_energy = np.maximum(energies[candidate_chs], 1e-12)
    energy_term = epsilon * (float(case.initial_energy) / residual_energy)

    d_i_sink_squared = max(float(dist_to_sink[current_ch]) ** 2, 1e-12)
    d_i_j_squared = np.square(distance_matrix[current_ch, candidate_chs])
    d_j_sink_squared = np.square(dist_to_sink[candidate_chs])
    distance_term = (1.0 - epsilon) * (
        (d_i_j_squared + d_j_sink_squared) / d_i_sink_squared
    )
    return energy_term + distance_term


def pick_next_hop(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    current_ch: int,
    cluster_heads: Sequence[int],
) -> int | None:
    current_layer = layers[current_ch]
    ch_array = np.asarray(cluster_heads, dtype=int)
    if ch_array.size == 0:
        return None

    forwarding_chs = ch_array[
        (ch_array != current_ch)
        & (energies[ch_array] > params.dead_energy_threshold_j)
        & (distance_matrix[current_ch, ch_array] <= params.transmission_range_m)
        & (layers[ch_array] < current_layer)
    ]
    if forwarding_chs.size == 0:
        return None

    scores = routing_selection_costs(
        case,
        params,
        distance_matrix,
        energies,
        dist_to_sink,
        current_ch,
        forwarding_chs,
    )
    return int(forwarding_chs[int(np.argmin(scores))])


def build_route_to_sink(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    source_ch: int,
    cluster_heads: Sequence[int],
) -> list[int | None]:
    """Apply P(i,j) at every CH-to-CH hop until a CH can reach the sink."""
    current_ch = int(source_ch)
    route: list[int | None] = []
    visited = {current_ch}

    for _ in range(len(cluster_heads) + 1):
        if can_transmit_to_sink(params, float(dist_to_sink[current_ch])):
            route.append(None)
            return route

        next_hop = pick_next_hop(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            current_ch,
            cluster_heads,
        )
        if next_hop is None or next_hop in visited:
            return []

        route.append(next_hop)
        visited.add(next_hop)
        current_ch = next_hop

    return []


def execute_round_transmissions(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    tx_cost_matrix_per_bit: np.ndarray,
    tx_cost_to_sink_per_bit: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    assignments: Dict[int, List[int]],
    cluster_heads: Sequence[int],
    packets_received: int,
    dead_energy_threshold_j: float,
    neighbor_count: np.ndarray | None = None,
    routing_edges: List[tuple[int, int | None]] | None = None,
) -> int:
    if neighbor_count is None:
        neighbor_count = np.zeros_like(energies, dtype=float)
    contention = contention_multiplier(params, neighbor_count)

    broadcast_bits = params.broadcast_packet_size_bits
    alive_nodes = np.flatnonzero(energies > dead_energy_threshold_j)
    if alive_nodes.size:
        energies[alive_nodes] = np.maximum(
            0.0,
            energies[alive_nodes] - aggregate_energy(params, broadcast_bits) * contention[alive_nodes],
        )

    packet_bits = case.packet_size_bits
    rx_cost = receive_energy(params, packet_bits)
    aggregate_cost_per_packet = aggregate_energy(params, packet_bits)

    for ch_idx, members in assignments.items():
        if energies[ch_idx] <= dead_energy_threshold_j:
            continue

        members_array = np.asarray(members, dtype=int)
        if members_array.size:
            transmitting_members = members_array[
                (members_array != ch_idx) & (energies[members_array] > dead_energy_threshold_j)
            ]
        else:
            transmitting_members = members_array

        if transmitting_members.size:
            tx_costs = (
                tx_cost_matrix_per_bit[transmitting_members, ch_idx]
                * packet_bits
                * contention[transmitting_members]
            )
            energies[transmitting_members] = np.maximum(
                0.0,
                energies[transmitting_members] - tx_costs,
            )
            energies[ch_idx] = max(
                0.0,
                energies[ch_idx] - rx_cost * transmitting_members.size * contention[ch_idx],
            )

        if energies[ch_idx] <= dead_energy_threshold_j:
            continue

        if members_array.size:
            data_packets = max(1, int(np.count_nonzero(energies[members_array] > dead_energy_threshold_j)))
        else:
            data_packets = 1

        aggregated_bits = packet_bits
        energies[ch_idx] = max(0.0, energies[ch_idx] - data_packets * aggregate_cost_per_packet)
        if energies[ch_idx] <= dead_energy_threshold_j:
            continue

        route = build_route_to_sink(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            ch_idx,
            cluster_heads,
        )
        if not route:
            continue

        current = int(ch_idx)
        delivered = False
        for hop in route:
            if routing_edges is not None:
                routing_edges.append((current, None if hop is None else int(hop)))

            if hop is None:
                tx = (
                    send_energy_from_distance_cost(float(tx_cost_to_sink_per_bit[current]), aggregated_bits)
                    * contention[current]
                )
                energies[current] = max(0.0, energies[current] - tx)
                delivered = energies[current] > dead_energy_threshold_j
                break

            hop = int(hop)
            tx = (
                send_energy_from_distance_cost(float(tx_cost_matrix_per_bit[current, hop]), aggregated_bits)
                * contention[current]
            )
            rx = receive_energy(params, aggregated_bits) * contention[hop]
            energies[current] = max(0.0, energies[current] - tx)
            energies[hop] = max(0.0, energies[hop] - rx)
            if energies[current] <= dead_energy_threshold_j or energies[hop] <= dead_energy_threshold_j:
                break
            current = hop

        if delivered:
            packets_received += 1

    return packets_received
