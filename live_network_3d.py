from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from uwsn.routing import pick_next_hop
from uwsn.run_config import SIMULATION_CASE, SIMULATION_PARAMS
from uwsn.simulator import PsoEulcSimulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the UWSN simulation and update a live 3D plot while it runs."
    )
    parser.add_argument("--space", type=float, default=500.0)
    parser.add_argument("--nodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument(
        "--distribution",
        choices=["uniform", "gaussian", "exponential"],
        default="uniform",
    )
    parser.add_argument("--packet-size-bits", type=int, default=6400)
    parser.add_argument("--initial-energy-j", type=float, default=0.5)
    parser.add_argument(
        "--optimizer",
        choices=["pso", "ga", "de", "leach", "eulc", "eeumc", "ebrec"],
        default="pso",
    )
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--transmission-range-m", type=float, default=150.0)
    parser.add_argument("--cluster-head-ratio", type=float, default=0.2)
    parser.add_argument("--recluster-interval", type=int, default=20)
    parser.add_argument("--draw-every", type=int, default=1)
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument("--max-member-links", type=int, default=900)
    parser.add_argument("--stop-on-first-dead", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "figures" / "network_3d_live",
    )
    return parser.parse_args()


def _build_simulator(args: argparse.Namespace) -> PsoEulcSimulator:
    case = replace(
        SIMULATION_CASE,
        name=f"live3d_{args.distribution}_s{int(args.space)}_n{args.nodes}",
        node_count=args.nodes,
        packet_size_bits=args.packet_size_bits,
        initial_energy=args.initial_energy_j,
        rounds=args.rounds,
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
        recluster_interval=max(1, args.recluster_interval),
    )
    return PsoEulcSimulator(case, params, seed=args.seed, verbose=False)


def _routing_edges(sim: PsoEulcSimulator) -> list[tuple[int, int | None]]:
    used_edges = getattr(sim, "last_routing_edges", None)
    if used_edges:
        return [(int(ch), None if next_hop is None else int(next_hop)) for ch, next_hop in used_edges]

    live_chs = [int(ch) for ch in sim.current_chs if sim.energies[int(ch)] > sim.params.dead_energy_threshold_j]
    edges: list[tuple[int, int | None]] = []
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


class LiveNetwork3DPlot:
    def __init__(self, args: argparse.Namespace) -> None:
        import matplotlib.pyplot as plt

        self.args = args
        self.plt = plt
        self.fig = plt.figure(figsize=(12, 8.5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.cmap = plt.colormaps.get_cmap("tab20")
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir = self.output_dir / "frames"
        if args.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def _set_axes(self) -> None:
        space = self.args.space
        self.ax.set_xlim(0, space)
        self.ax.set_ylim(0, space)
        self.ax.set_zlim(space, 0)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_zlabel("depth h_i (m), surface=0")
        self.ax.view_init(elev=22, azim=-52)
        self.ax.grid(True, alpha=0.28)

    def _draw_member_links(self, sim: PsoEulcSimulator) -> None:
        drawn = 0
        positions = sim.positions
        threshold = sim.params.dead_energy_threshold_j
        for ch, members in sim.current_assignments.items():
            if sim.energies[int(ch)] <= threshold:
                continue
            for member in members:
                member = int(member)
                if member == int(ch) or sim.energies[member] <= threshold:
                    continue
                p0 = positions[member]
                p1 = positions[int(ch)]
                self.ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    color="#9aa0a6",
                    alpha=0.12,
                    linewidth=0.5,
                )
                drawn += 1
                if drawn >= self.args.max_member_links:
                    return

    def _draw_routing(self, sim: PsoEulcSimulator) -> None:
        positions = sim.positions
        for ch, next_hop in _routing_edges(sim):
            p0 = positions[int(ch)]
            p1 = sim.sink if next_hop is None else positions[int(next_hop)]
            self.ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color="#111111",
                linewidth=2.0,
                alpha=0.9,
            )

    def _draw_water_surface(self) -> None:
        space = self.args.space
        grid = np.array([[0.0, space], [0.0, space]])
        self.ax.plot_surface(
            grid,
            grid.T,
            np.zeros((2, 2), dtype=float),
            color="#8fd3ff",
            alpha=0.10,
            linewidth=0,
            shade=False,
        )

    def draw(self, sim: PsoEulcSimulator, round_idx: int, packets_received: int, force: bool = False) -> None:
        if not force and round_idx % max(1, self.args.draw_every) != 0:
            return

        from matplotlib.lines import Line2D

        self.ax.clear()
        self._draw_water_surface()
        positions = sim.positions
        threshold = sim.params.dead_energy_threshold_j
        alive_mask = sim.energies > threshold
        dead_indices = np.flatnonzero(~alive_mask)
        chs = [int(ch) for ch in sim.current_chs if sim.energies[int(ch)] > threshold]
        ch_set = set(chs)
        assigned_alive: set[int] = set()
        cluster_count = max(len(chs), 1)

        for cluster_idx, ch in enumerate(chs):
            members = [
                int(member)
                for member in sim.current_assignments.get(ch, [])
                if sim.energies[int(member)] > threshold
            ]
            assigned_alive.update(members)
            color = self.cmap((cluster_idx % 20) / max(1, min(cluster_count, 20) - 1))
            member_only = [member for member in members if member not in ch_set]
            if member_only:
                member_pos = positions[member_only]
                energy_alpha = 0.28 + 0.35 * np.clip(
                    sim.energies[member_only] / max(sim.case.initial_energy, 1e-9),
                    0.0,
                    1.0,
                )
                self.ax.scatter(
                    member_pos[:, 0],
                    member_pos[:, 1],
                    member_pos[:, 2],
                    s=15,
                    color=color,
                    alpha=float(np.mean(energy_alpha)),
                    depthshade=True,
                )

            ch_pos = positions[ch]
            self.ax.scatter(
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

        unassigned_alive = sorted(set(np.flatnonzero(alive_mask).astype(int).tolist()) - assigned_alive)
        if unassigned_alive:
            unassigned_pos = positions[unassigned_alive]
            self.ax.scatter(
                unassigned_pos[:, 0],
                unassigned_pos[:, 1],
                unassigned_pos[:, 2],
                s=14,
                color="#8f8f8f",
                alpha=0.55,
                depthshade=True,
            )

        if dead_indices.size:
            dead_pos = positions[dead_indices]
            self.ax.scatter(
                dead_pos[:, 0],
                dead_pos[:, 1],
                dead_pos[:, 2],
                s=32,
                marker="x",
                color="#d62728",
                alpha=0.95,
                depthshade=False,
            )

        self._draw_member_links(sim)
        self._draw_routing(sim)

        sink = sim.sink
        self.ax.scatter(
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

        alive_count = int(np.count_nonzero(alive_mask))
        dead_count = int(sim.case.node_count - alive_count)
        residual_energy = float(np.sum(sim.energies))
        self._set_axes()
        self.ax.set_title(
            f"Live 3D UWSN - round={round_idx}, alive={alive_count}/{sim.case.node_count}, "
            f"dead={dead_count}, CH={len(chs)}, residual={residual_energy:.2f}J",
            pad=18,
        )

        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C9BD4", markersize=8, label="Member sensor"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#4C9BD4", markeredgecolor="black", markersize=10, label="Cluster Head"),
            Line2D([0], [0], marker="x", color="#d62728", linestyle="None", markersize=9, label="Dead node"),
            Line2D([0], [0], color="#8fd3ff", linewidth=4, alpha=0.45, label="Water surface"),
            Line2D([0], [0], color="#9aa0a6", linewidth=1.2, label="Member to CH"),
            Line2D([0], [0], color="#111111", linewidth=2.2, label="CH routing"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=12, label="Sink"),
        ]
        self.ax.legend(handles=handles, loc="lower left", framealpha=0.92)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.args.save_frames:
            self.fig.savefig(self.frame_dir / f"round_{round_idx:05d}.png", dpi=130)

        latest_path = self.output_dir / "live_network3d_latest.png"
        self.fig.savefig(latest_path, dpi=150)
        print(
            f"round={round_idx}, alive={alive_count}, dead={dead_count}, "
            f"CH={len(chs)}, packets={packets_received}, saved={latest_path}"
        )


def main() -> None:
    args = _parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    sim = _build_simulator(args)
    plotter = LiveNetwork3DPlot(args)

    if not args.no_show:
        plt.ion()
        plt.show(block=False)

    def on_round(current_sim: PsoEulcSimulator, round_idx: int, packets_received: int) -> None:
        plotter.draw(current_sim, round_idx, packets_received)
        if not args.no_show:
            plt.pause(max(0.001, args.pause))

    metrics = sim.run(
        stop_on_first_dead=args.stop_on_first_dead,
        max_rounds=args.rounds,
        round_callback=on_round,
    )
    final_round = len(metrics.dead_nodes_by_round)
    plotter.draw(sim, final_round, metrics.packets_received, force=True)

    if not args.no_show:
        plt.ioff()
        print("Simulation finished. Close the 3D window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
