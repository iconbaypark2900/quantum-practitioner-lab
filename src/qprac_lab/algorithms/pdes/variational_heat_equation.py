"""Variational solution of the 1D heat equation.

Each implicit-Euler timestep of a PDE is a linear system, so a PDE solver is a
linear-solver in a loop. Where :mod:`qprac_lab.algorithms.pdes.hhl_intro` solves
one with phase estimation, this solves it variationally (VQLS): prepare
``|psi(theta)>`` with an ansatz and tune ``theta`` until ``A|psi>`` points along
``|b>``. No phase estimation, no postselection, much shallower circuits -- and a
non-convex optimisation instead, with everything that implies.

**The normalisation is not a technicality.** A quantum state is normalised, so
``|psi>`` encodes only the *direction* of ``u``, never its magnitude. For the heat
equation the magnitude is the physics: ``||u||`` decaying is heat leaving the
system. Any quantum PDE solver has to track that scale classically, on the side,
and a demo reporting a beautiful profile match while quietly dropping it is
answering an easier question than the one posed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit


@dataclass
class VariationalPDEResult:
    """A variational PDE step against the exact linear solve."""

    algorithm: str
    use_case: str
    algorithm_type: str
    backend: dict
    problem: dict
    num_qubits: int
    num_parameters: int
    circuit_depth: int
    vqls_cost: float
    function_evaluations: int
    restarts: int
    fidelity: float
    solution: list[float]
    exact_solution: list[float]
    norm_tracking: dict[str, float]
    classical_baseline: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


def heat_equation_system(
    num_qubits: int = 3,
    alpha: float = 0.1,
    dt: float = 0.02,
    length: float = 1.0,
):
    """Implicit-Euler matrix for ``du/dt = alpha d2u/dx2`` and an initial profile.

    Implicit Euler is used rather than explicit because it is unconditionally
    stable -- explicit would impose ``dt < dx^2 / (2 alpha)`` and turn the choice
    of timestep into a stability question rather than an accuracy one.
    """
    size = 2**num_qubits
    grid = np.linspace(0.0, length, size)
    spacing = grid[1] - grid[0]
    laplacian = (
        np.diag(-2.0 * np.ones(size))
        + np.diag(np.ones(size - 1), 1)
        + np.diag(np.ones(size - 1), -1)
    )
    matrix = np.eye(size) - alpha * dt / spacing**2 * laplacian
    initial = np.sin(np.pi * grid)
    initial[0] = initial[-1] = 0.0
    return matrix, initial, grid


def vqls_cost(matrix, target_direction, state) -> float:
    """VQLS global cost: ``1 - |<b|A|psi>|^2 / <psi|A'A|psi>``.

    Zero exactly when ``A|psi>`` is parallel to ``|b>``, which -- because both
    sides are normalised -- is as much as a normalised state can express.
    """
    projected = matrix @ state
    denominator = float(np.dot(projected, projected))
    if denominator < 1e-15:
        return 1.0
    return float(1.0 - abs(np.dot(target_direction, projected)) ** 2 / denominator)


def solve_variational_linear_system(
    matrix,
    rhs,
    num_qubits: int,
    ansatz_reps: int = 3,
    restarts: int = 5,
    maxiter: int = 2000,
    seed: int = 42,
):
    """Solve ``A x = b`` variationally, returning the normalised direction of ``x``.

    Restarted by default: the cost is non-convex, so a single run reports one
    local optimum -- the same lesson the QAOA tutorial had to learn the hard way.
    """
    require_qiskit("Solving a linear system variationally")
    from qiskit.circuit.library import real_amplitudes
    from qiskit.quantum_info import Statevector

    ansatz = real_amplitudes(num_qubits, reps=ansatz_reps)
    target = np.asarray(rhs, dtype=float)
    target = target / np.linalg.norm(target)
    rng = np.random.default_rng(seed)
    evaluations = 0

    def objective(parameters):
        nonlocal evaluations
        evaluations += 1
        state = np.real(Statevector(ansatz.assign_parameters(parameters)).data)
        return vqls_cost(matrix, target, state)

    best = None
    for _ in range(restarts):
        candidate = minimize(
            objective,
            rng.uniform(-np.pi, np.pi, ansatz.num_parameters),
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        if best is None or candidate.fun < best.fun:
            best = candidate

    state = np.real(Statevector(ansatz.assign_parameters(best.x)).data)
    return state, float(best.fun), evaluations, ansatz


def run_variational_heat_equation_tutorial(
    num_qubits: int = 3,
    alpha: float = 0.1,
    dt: float = 0.02,
    ansatz_reps: int = 3,
    restarts: int = 5,
    num_steps: int = 3,
    seed: int = 42,
) -> VariationalPDEResult:
    """Take implicit-Euler heat-equation steps with a variational linear solver."""
    require_qiskit("The variational heat-equation tutorial")
    matrix, initial, grid = heat_equation_system(num_qubits, alpha, dt)
    exact = np.linalg.solve(matrix, initial)

    state, cost, evaluations, ansatz = solve_variational_linear_system(
        matrix, initial, num_qubits, ansatz_reps, restarts, seed=seed
    )
    exact_direction = exact / np.linalg.norm(exact)
    # A statevector has an arbitrary global sign; align before comparing profiles.
    state = state * np.sign(np.dot(state, exact_direction) or 1.0)
    fidelity = float(abs(np.dot(state, exact_direction)) ** 2)

    profile = initial.copy()
    steps = []
    for index in range(num_steps):
        profile = np.linalg.solve(matrix, profile)
        steps.append(
            {
                "step": index + 1,
                "time": (index + 1) * dt,
                "norm": float(np.linalg.norm(profile)),
                "peak": float(profile.max()),
            }
        )

    return VariationalPDEResult(
        algorithm="variational_heat_equation",
        use_case="thermal_diffusion_simulation",
        algorithm_type="variational_pde_residual_minimization",
        backend=QiskitBackendAdapter(seed=seed).describe(),
        problem={
            "equation": "du/dt = alpha d2u/dx2",
            "scheme": "implicit Euler (unconditionally stable)",
            "grid_points": 2**num_qubits,
            "alpha": alpha,
            "dt": dt,
            "condition_number": float(np.linalg.cond(matrix)),
        },
        num_qubits=num_qubits,
        num_parameters=int(ansatz.num_parameters),
        circuit_depth=int(ansatz.decompose(reps=3).depth()),
        vqls_cost=cost,
        function_evaluations=evaluations,
        restarts=restarts,
        fidelity=fidelity,
        solution=state.tolist(),
        exact_solution=exact_direction.tolist(),
        norm_tracking={
            "initial_norm": float(np.linalg.norm(initial)),
            "exact_norm_after_step": float(np.linalg.norm(exact)),
            "norm_lost_to_diffusion": float(
                np.linalg.norm(initial) - np.linalg.norm(exact)
            ),
            "recovered_from_quantum_state": 0.0,
        },
        classical_baseline={
            "method": "numpy.linalg.solve on the same implicit-Euler matrix",
            "exact": True,
            "cost": "O(N^3) dense, O(N) for this tridiagonal system with a banded solver",
        },
        steps=steps,
        notes={
            "normalisation": (
                "|psi> encodes only the direction of u; ||u|| carries the total heat "
                "and must be tracked classically alongside"
            ),
            "non_convex": "restarted by default, since one run reports a local optimum",
            "vs_hhl": "no phase estimation and no postselection, at the price of a "
            "non-convex optimisation",
        },
    )


def run_variational_heat_equation_scaffold():
    """Backwards-compatible alias for :func:`run_variational_heat_equation_tutorial`."""
    return run_variational_heat_equation_tutorial()
