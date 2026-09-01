from dataclasses import dataclass
import math

import numpy as np


MASS = 1.0
SPRING_CONSTANT = math.pi**2
OMEGA = math.sqrt(SPRING_CONSTANT / MASS)
PERIOD = 2.0 * math.pi / OMEGA


@dataclass(frozen=True)
class Trajectory:
    x0: float
    v0: float
    t_end: float
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray


def derivatives(x: float, v: float) -> tuple[float, float]:
    return v, -(OMEGA**2) * x


def rk4_step(x: float, v: float, dt: float) -> tuple[float, float]:
    k1_x, k1_v = derivatives(x, v)
    k2_x, k2_v = derivatives(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v)
    k3_x, k3_v = derivatives(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v)
    k4_x, k4_v = derivatives(x + dt * k3_x, v + dt * k3_v)

    return (
        x + dt * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) / 6.0,
        v + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0,
    )


def solve_trajectory(
    x0: float,
    v0: float,
    t_end: float,
    dt: float = 0.01,
) -> Trajectory:
    x0_value = float(x0)
    v0_value = float(v0)
    t_end_value = float(t_end)
    dt_value = float(dt)
    values = (x0_value, v0_value, t_end_value, dt_value)

    if not all(math.isfinite(value) for value in values):
        raise ValueError("x0, v0, t_end, and dt must be finite")
    if t_end_value <= 0.0:
        raise ValueError("t_end must be positive")
    if dt_value <= 0.0:
        raise ValueError("dt must be positive")

    normalized_x0 = round(x0_value, 3)
    normalized_v0 = round(v0_value, 3)
    step_count = max(1, math.ceil(t_end_value / dt_value))
    times = np.linspace(0.0, t_end_value, step_count + 1)
    positions = np.empty(step_count + 1, dtype=float)
    velocities = np.empty(step_count + 1, dtype=float)
    positions[0] = normalized_x0
    velocities[0] = normalized_v0

    for index in range(step_count):
        actual_dt = float(times[index + 1] - times[index])
        positions[index + 1], velocities[index + 1] = rk4_step(
            positions[index], velocities[index], actual_dt
        )

    return Trajectory(
        x0=normalized_x0,
        v0=normalized_v0,
        t_end=t_end_value,
        t=times,
        x=positions,
        v=velocities,
    )
