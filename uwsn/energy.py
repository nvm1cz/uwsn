from __future__ import annotations

import numpy as np

from .run_config import TunableParams

# Cong thuc tinh do am cua thorp
def thorp_absorption_db_per_km(freq_khz: float) -> float:
    f2 = freq_khz * freq_khz
    return (
        0.11 * f2 / (1 + f2)
        + 44 * f2 / (4100 + f2)
        + 2.75e-4 * f2
        + 0.003
    )

## do suy hao theo cong thuc: A(d) = d^k * a^d
def attenuation(distance_m: float, params: TunableParams) -> float:
    distance_km = max(distance_m / 1000.0, 1e-6)
    alpha_db = thorp_absorption_db_per_km(params.acoustic_frequency_khz)
    absorption_linear = 10 ** (alpha_db / 10.0)
    return (distance_km ** params.spreading_factor) * (absorption_linear ** distance_km)


def attenuation_array(distance_m: np.ndarray, params: TunableParams) -> np.ndarray:
    distance_km = np.maximum(np.asarray(distance_m, dtype=float) / 1000.0, 1e-6)
    alpha_db = thorp_absorption_db_per_km(params.acoustic_frequency_khz)
    absorption_linear = 10 ** (alpha_db / 10.0)
    return (distance_km ** params.spreading_factor) * (absorption_linear ** distance_km)

# Khoảng cách Euclidean giữa hai điểm trong không gian 3D
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))
