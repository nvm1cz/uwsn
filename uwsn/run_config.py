from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .cases import SimulationCase


@dataclass
class TunableParams:
    width_m: float
    height_m: float
    depth_m: float
    # R0: ngÆ°á»¡ng chiá»u sÃ¢u cho lá»›p nÃ´ng nháº¥t + cÆ¡ sá»Ÿ bÃ¡n kÃ­nh cáº¡nh tranh trong EULC (khÃ¡c pháº¡m vi truyá»n).
    layer_r0_m: float
    # Pháº¡m vi truyá»n thÃ´ng tá»‘i Ä‘a giá»¯a cÃ¡c node (theo bÃ i: 150 m hoáº·c 200 m).
    transmission_range_m: float
    layer_spacing_m: float
    cluster_head_ratio: float
    alpha_weight: float
    beta_weight: float
    gamma_weight: float
    balance_factor: float
    recluster_interval: int
    pso_particles: int
    pso_iterations: int
    pso_inertia: float
    pso_c1: float
    pso_c2: float
    pso_omega: float
    optimizer: str
    acoustic_frequency_khz: float
    spreading_factor: float
    transmit_power_p0: float
    electronic_energy_e1_nj: float
    processing_energy_nj: float
    broadcast_packet_size_bits: int
    energy_calibration_factor: float
    contention_penalty_factor: float
    contention_reference_degree: float
    contention_max_multiplier: float
    dead_energy_threshold_j: float
    sink_position: Tuple[float, float, float] | None
    deployment_distribution: str
    gaussian_std_fraction: float
    exponential_scale_fraction: float

    # EULC
    eulc_candidate_ratio: float
    connectivity_penalty_enabled: bool = True

    # ðŸ”¥ AUTO SINK
    def get_sink_position(self) -> Tuple[float, float, float]:
        if self.sink_position is not None:
            return self.sink_position
        return (
            self.width_m / 2,
            self.height_m / 2,
            0.0
        )


# =========================
# SIMULATION CONFIG
# =========================

width = 100.0
height = 100.0
depth = 100.0

SIMULATION_CASE = SimulationCase(
    name="custom",
    initial_energy=0.5,
    packet_size_bits=6400,
    node_count=100,
    rounds=1000,
)

SIMULATION_PARAMS = TunableParams(
    width_m=width,
    height_m=height,
    depth_m=depth,

    layer_r0_m=20.0,
    transmission_range_m=150.0,  # Ä‘á»•i thÃ nh 200.0 cho case thá»© hai
    layer_spacing_m=10.0,

    cluster_head_ratio=0.2,

    alpha_weight=0.6,
    beta_weight=0.25,
    gamma_weight=0.15,

    balance_factor=0.5,
    recluster_interval=20,

    pso_particles=20,
    pso_iterations=50,
    pso_inertia=0.7,
    pso_c1=1.5,
    pso_c2=1.5,
    pso_omega=0.65,
    optimizer="pso",

    acoustic_frequency_khz=10.0,
    spreading_factor=2,

    transmit_power_p0=3.0,
    electronic_energy_e1_nj=5.0,
    processing_energy_nj=50.0,

    broadcast_packet_size_bits=200,
    energy_calibration_factor=1.0,
    contention_penalty_factor=0.0,
    contention_reference_degree=10.0,
    contention_max_multiplier=4.0,
    dead_energy_threshold_j=0.075,

    
    sink_position=None,
    deployment_distribution="uniform",
    gaussian_std_fraction=0.18,
    exponential_scale_fraction=0.35,

    # EULC
    eulc_candidate_ratio=0.4
)

RUNS = 10
BASE_SEED = 42
OUTPUT_DIR = "outputs"

