from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from uwsn.routing import pick_next_hop
from uwsn.run_config import SIMULATION_CASE, SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 3D UWSN snapshot with sensors, CHs, clusters, and routing."
    )
    parser.add_argument("--space", type=float, default=500.0, help="Cube side length in meters.")
    parser.add_argument("--nodes", type=int, default=300, help="Number of sensor nodes.")
    parser.add_argument("--seed", type=int, default=4000, help="Random seed.")
    parser.add_argument(
        "--distribution",
        choices=["uniform", "gaussian", "exponential"],
        default="uniform",
        help="Node deployment distribution.",
    )
    parser.add_argument("--packet-size-bits", type=int, default=6400)
    parser.add_argument("--initial-energy-j", type=float, default=0.5)
    parser.add_argument("--optimizer", choices=["pso", "ga", "de"], default="pso")
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--transmission-range-m", type=float, default=150.0)
    parser.add_argument("--cluster-head-ratio", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "figures" / "network_3d",
    )
    parser.add_argument(
        "--max-member-links",
        type=int,
        default=900,
        help="Maximum member-to-CH helper lines to draw. Points are always fully drawn.",
    )
    return parser.parse_args()


def _build_simulator(args: argparse.Namespace) -> PsoEulcSimulator:
    case = replace(
        SIMULATION_CASE,
        name=f"3d_{args.distribution}_s{int(args.space)}_n{args.nodes}",
        node_count=args.nodes,
        packet_size_bits=args.packet_size_bits,
        initial_energy=args.initial_energy_j,
    )
    params = replace(
        SIMULATION_PARAMS,
        width_m=args.space,
        height_m=args.space,
        depth_m=args.space,
        deployment_distribution=args.distribution,
        optimizer=args.optimizer,
        pso_particles=args.particles,
        pso_iterations=args.iterations,
        transmission_range_m=args.transmission_range_m,
        cluster_head_ratio=args.cluster_head_ratio,
    )
    sim = PsoEulcSimulator(case, params, seed=args.seed, verbose=False)
    sim._refresh_clusters(round_idx=1)
    return sim


def _routing_edges(sim: PsoEulcSimulator) -> list[tuple[int, int | None]]:
    used_edges = getattr(sim, "last_routing_edges", None)
    if used_edges:
        return [(int(ch), None if next_hop is None else int(next_hop)) for ch, next_hop in used_edges]

    edges: list[tuple[int, int | None]] = []
    live_chs = [int(ch) for ch in sim.current_chs if sim.energies[int(ch)] > sim.params.dead_energy_threshold_j]
    for ch in live_chs:
        next_hop = pick_next_hop(
            sim.case,
            sim.params,
            sim.distance_matrix,
            sim.energies,
            sim.layers,
            sim.dist_to_sink,
            ch,
            live_chs,
        )
        edges.append((ch, next_hop))
    return edges


def _iter_member_links(assignments: dict[int, list[int]]) -> Iterable[tuple[int, int]]:
    for ch, members in assignments.items():
        for member in members:
            if int(member) != int(ch):
                yield int(member), int(ch)


def _set_equal_3d_axes(ax, space: float) -> None:
    ax.set_xlim(0, space)
    ax.set_ylim(0, space)
    ax.set_zlim(space, 0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("depth h_i (m), surface=0")


def _draw_water_surface(ax, space: float) -> None:
    grid = np.array([[0.0, space], [0.0, space]])
    ax.plot_surface(
        grid,
        grid.T,
        np.zeros((2, 2), dtype=float),
        color="#8fd3ff",
        alpha=0.10,
        linewidth=0,
        shade=False,
    )


def _save_network_plot(sim: PsoEulcSimulator, args: argparse.Namespace, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions = sim.positions
    chs = [int(ch) for ch in sim.current_chs]
    ch_set = set(chs)
    cmap = plt.colormaps.get_cmap("tab20")
    cluster_count = max(len(chs), 1)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    _draw_water_surface(ax, args.space)

    assigned_nodes: set[int] = set()
    for cluster_idx, ch in enumerate(chs):
        members = [int(m) for m in sim.current_assignments.get(ch, [])]
        assigned_nodes.update(members)
        if not members:
            continue

        color = cmap((cluster_idx % 20) / max(1, min(cluster_count, 20) - 1))
        member_only = [member for member in members if member not in ch_set]
        if member_only:
            member_pos = positions[member_only]
            ax.scatter(
                member_pos[:, 0],
                member_pos[:, 1],
                member_pos[:, 2],
                s=16,
                color=color,
                alpha=0.48,
                depthshade=True,
            )

        ch_pos = positions[ch]
        ax.scatter(
            [ch_pos[0]],
            [ch_pos[1]],
            [ch_pos[2]],
            s=115,
            marker="^",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            depthshade=True,
        )

    unassigned = sorted(set(range(sim.case.node_count)) - assigned_nodes)
    if unassigned:
        unassigned_pos = positions[unassigned]
        ax.scatter(
            unassigned_pos[:, 0],
            unassigned_pos[:, 1],
            unassigned_pos[:, 2],
            s=16,
            color="#999999",
            alpha=0.55,
            depthshade=True,
        )

    for link_idx, (member, ch) in enumerate(_iter_member_links(sim.current_assignments)):
        if link_idx >= args.max_member_links:
            break
        p0 = positions[member]
        p1 = positions[ch]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="#9aa0a6",
            alpha=0.13,
            linewidth=0.55,
        )

    sink = sim.sink
    ax.scatter(
        [sink[0]],
        [sink[1]],
        [sink[2]],
        s=170,
        marker="*",
        color="black",
        edgecolor="white",
        linewidth=0.8,
        depthshade=False,
    )

    for ch, next_hop in _routing_edges(sim):
        p0 = positions[ch]
        p1 = sink if next_hop is None else positions[int(next_hop)]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="#111111",
            linewidth=2.1,
            alpha=0.90,
        )

    _set_equal_3d_axes(ax, args.space)
    ax.view_init(elev=22, azim=-52)
    ax.grid(True, alpha=0.28)
    ax.set_title(
        f"3D UWSN snapshot - {args.distribution}, space={int(args.space)}m, "
        f"n={args.nodes}, seed={args.seed}, CH={len(chs)}",
        pad=18,
    )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C9BD4", markersize=8, label="Member sensor"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#4C9BD4", markeredgecolor="black", markersize=10, label="Cluster Head"),
        Line2D([0], [0], color="#8fd3ff", linewidth=4, alpha=0.45, label="Water surface"),
        Line2D([0], [0], color="#9aa0a6", linewidth=1.2, label="Member to CH"),
        Line2D([0], [0], color="#111111", linewidth=2.2, label="CH routing"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=12, label="Sink"),
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    latest_path = output_path.parent / "network3d_latest.png"
    if output_path != latest_path:
        fig.savefig(latest_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    sim = _build_simulator(args)
    output_path = args.output_dir / (
        f"network3d_{args.distribution}_s{int(args.space)}_n{args.nodes}_seed{args.seed}.png"
    )
    _save_network_plot(sim, args, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
