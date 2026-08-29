"""Black-Scholes as a PDE, solved variationally -- and why you would not.

The Black-Scholes equation

    dV/dt + (1/2) sigma^2 S^2 d2V/dS2 + r S dV/dS - r V = 0

is a convection-diffusion PDE. Discretised in ``S`` and stepped backwards from
the payoff with implicit Euler, every timestep is a linear system -- so the
variational solver from
:mod:`qprac_lab.algorithms.pdes.variational_heat_equation` applies directly.

This tutorial exists to make a cost argument concrete rather than to advocate the
method. European options have a **closed-form solution**, computable in
microseconds, so a numerical PDE solve is already unnecessary here and a quantum
one is unnecessary twice over. The value is in the arithmetic: pricing this option
variationally costs on the order of ``10^5`` circuit evaluations against one call
to a formula, and the tutorial reports that ratio instead of hiding it.

Where quantum finance is actually argued to help is Monte Carlo pricing of
path-dependent derivatives via amplitude estimation -- a quadratic speedup on
sampling, not an exponential one on PDEs. That is a different algorithm, and the
distinction is usually lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qprac_lab.algorithms.pdes.variational_heat_equation import (
    solve_variational_linear_system,
)
from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit


@dataclass
class BlackScholesResult:
    """A Black-Scholes PDE solve against the closed-form price."""

    algorithm: str
    use_case: str
    algorithm_type: str
    backend: dict
    parameters: dict
    grid: dict
    finite_difference_prices: list[dict[str, float]]
    max_absolute_error: float
    variational_step: dict
    cost_comparison: dict
    notes: dict[str, Any] = field(default_factory=dict)


def black_scholes_call(spot, strike: float, maturity: float, rate: float, volatility: float):
    """Closed-form European call price -- the exact baseline.

    Any numerical scheme here is checked against this, which is precisely why
    Black-Scholes is a poor advertisement for numerical PDE methods.
    """
    from scipy.stats import norm

    spot = np.asarray(spot, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + (rate + volatility**2 / 2) * maturity) / (
            volatility * np.sqrt(maturity)
        )
        d2 = d1 - volatility * np.sqrt(maturity)
        price = spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)
    # S = 0 is worthless and produces a log singularity above.
    return np.where(spot <= 0, 0.0, price)


def black_scholes_operator(grid, rate: float, volatility: float):
    """Spatial discretisation of the Black-Scholes operator on a price grid."""
    size = len(grid)
    spacing = grid[1] - grid[0]
    operator = np.zeros((size, size))
    for index in range(1, size - 1):
        diffusion = 0.5 * volatility**2 * grid[index] ** 2 / spacing**2
        convection = 0.5 * rate * grid[index] / spacing
        operator[index, index - 1] = diffusion - convection
        operator[index, index] = -2 * diffusion - rate
        operator[index, index + 1] = diffusion + convection
    return operator


def solve_black_scholes_pde(
    strike: float = 100.0,
    maturity: float = 1.0,
    rate: float = 0.05,
    volatility: float = 0.2,
    max_spot: float = 300.0,
    num_qubits: int = 6,
    num_steps: int = 200,
):
    """Implicit-Euler backward solve of the Black-Scholes PDE.

    Implicit rather than explicit because the diffusion coefficient scales as
    ``S^2``, so an explicit scheme's stability limit becomes punishing at the top
    of the price grid.
    """
    size = 2**num_qubits
    grid = np.linspace(0.0, max_spot, size)
    dt = maturity / num_steps
    matrix = np.eye(size) - dt * black_scholes_operator(grid, rate, volatility)
    matrix[0, :] = 0.0
    matrix[0, 0] = 1.0
    matrix[-1, :] = 0.0
    matrix[-1, -1] = 1.0

    values = np.maximum(grid - strike, 0.0)  # terminal payoff
    for step in range(num_steps):
        rhs = values.copy()
        rhs[0] = 0.0
        rhs[-1] = max_spot - strike * np.exp(-rate * (step + 1) * dt)
        values = np.linalg.solve(matrix, rhs)
    return grid, values, matrix


def run_black_scholes_pde_tutorial(
    strike: float = 100.0,
    maturity: float = 1.0,
    rate: float = 0.05,
    volatility: float = 0.2,
    num_qubits: int = 6,
    num_steps: int = 200,
    variational_qubits: int = 4,
    restarts: int = 5,
    seed: int = 42,
) -> BlackScholesResult:
    """Price a European call by PDE, check it against the formula, and cost it."""
    require_qiskit("The Black-Scholes tutorial")
    grid, values, _matrix = solve_black_scholes_pde(
        strike, maturity, rate, volatility, num_qubits=num_qubits, num_steps=num_steps
    )

    prices = []
    for spot in (80.0, 100.0, 120.0, 150.0):
        numerical = float(np.interp(spot, grid, values))
        analytic = float(black_scholes_call(spot, strike, maturity, rate, volatility))
        prices.append(
            {
                "spot": spot,
                "finite_difference": numerical,
                "analytic": analytic,
                "absolute_error": abs(numerical - analytic),
                "relative_error": abs(numerical - analytic) / max(analytic, 1e-12),
            }
        )

    # One variational timestep, on a smaller grid so the demonstration is quick.
    small_size = 2**variational_qubits
    small_grid = np.linspace(0.0, 300.0, small_size)
    small_dt = maturity / num_steps
    small_matrix = np.eye(small_size) - small_dt * black_scholes_operator(
        small_grid, rate, volatility
    )
    small_matrix[0, :] = 0.0
    small_matrix[0, 0] = 1.0
    small_matrix[-1, :] = 0.0
    small_matrix[-1, -1] = 1.0
    payoff = np.maximum(small_grid - strike, 0.0)

    state, cost, evaluations, ansatz = solve_variational_linear_system(
        small_matrix, payoff, variational_qubits, restarts=restarts, seed=seed
    )
    exact_step = np.linalg.solve(small_matrix, payoff)
    exact_direction = exact_step / np.linalg.norm(exact_step)
    state = state * np.sign(np.dot(state, exact_direction) or 1.0)

    return BlackScholesResult(
        algorithm="black_scholes_pde",
        use_case="financial_derivatives_pricing",
        algorithm_type="variational_pde_residual_minimization",
        backend=QiskitBackendAdapter(seed=seed).describe(),
        parameters={
            "strike": strike,
            "maturity": maturity,
            "rate": rate,
            "volatility": volatility,
            "option": "European call",
        },
        grid={
            "num_qubits": num_qubits,
            "grid_points": 2**num_qubits,
            "timesteps": num_steps,
            "scheme": "implicit Euler, backward from the terminal payoff",
        },
        finite_difference_prices=prices,
        max_absolute_error=max(row["absolute_error"] for row in prices),
        variational_step={
            "grid_points": small_size,
            "num_qubits": variational_qubits,
            "num_parameters": int(ansatz.num_parameters),
            "vqls_cost": cost,
            "function_evaluations": evaluations,
            "restarts": restarts,
            "fidelity": float(abs(np.dot(state, exact_direction)) ** 2),
            "note": "one timestep of the 200 the full solve needs",
        },
        cost_comparison={
            "analytic_formula": "1 evaluation, microseconds",
            "finite_difference": f"{num_steps} tridiagonal solves, milliseconds",
            "variational_per_step": evaluations,
            "variational_full_solve_estimate": int(evaluations * num_steps),
            "verdict": (
                "pricing this option variationally would cost about "
                f"{evaluations * num_steps:,} circuit evaluations against one call to a "
                "closed-form formula"
            ),
        },
        notes={
            "closed_form_exists": "European options do not need a numerical solver at all",
            "normalisation": (
                "the quantum state encodes only the shape of the value function; the "
                "price level must be tracked classically"
            ),
            "where_quantum_finance_actually_argues_advantage": (
                "amplitude estimation for Monte Carlo pricing of path-dependent "
                "derivatives -- a quadratic speedup on sampling, not an exponential one "
                "on PDEs"
            ),
        },
    )
