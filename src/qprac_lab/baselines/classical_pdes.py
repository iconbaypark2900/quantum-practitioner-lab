from __future__ import annotations

import numpy as np


def finite_difference_heat_step(u, alpha: float, dx: float, dt: float):
    """Explicit finite-difference baseline for the 1D heat equation.

    Algorithm type:
    - Classical finite difference method.
    """
    next_u = u.copy()
    coeff = alpha * dt / (dx * dx)
    for i in range(1, len(u) - 1):
        next_u[i] = u[i] + coeff * (u[i - 1] - 2 * u[i] + u[i + 1])
    return next_u


def solve_small_linear_system():
    """Classical baseline for HHL tutorial."""
    a = np.array([[1.0, 0.25], [0.25, 1.0]])
    b = np.array([1.0, 0.0])
    x = np.linalg.solve(a, b)
    return a, b, x
