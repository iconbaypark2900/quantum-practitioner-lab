"""PDE tutorials: HHL, the variational heat equation, and Black-Scholes.

Each is checked against an exact classical answer, because all three have one --
which is itself the honest framing of this whole module.
"""

import numpy as np
import pytest

pytest.importorskip("qiskit", reason='needs the quantum stack: pip install -e ".[qiskit]"')

from qprac_lab.algorithms.pdes.black_scholes_pde import (  # noqa: E402
    black_scholes_call,
    run_black_scholes_pde_tutorial,
    solve_black_scholes_pde,
)
from qprac_lab.algorithms.pdes.hhl_intro import (  # noqa: E402
    clock_values,
    eigenvalues_representable,
    measurements_for_precision,
    run_hhl_intro_tutorial,
    solve_hhl,
    suggested_evolution_time,
    well_conditioned_system,
)
from qprac_lab.algorithms.pdes.variational_heat_equation import (  # noqa: E402
    heat_equation_system,
    run_variational_heat_equation_tutorial,
    solve_variational_linear_system,
    vqls_cost,
)

# --------------------------------------------------------------------- HHL


def test_suggested_time_makes_eigenvalues_exactly_representable():
    """Phase estimation only resolves eigenvalues landing on integer register values."""
    matrix, _rhs = well_conditioned_system()
    eigenvalues = np.linalg.eigvalsh(matrix)
    time = suggested_evolution_time(eigenvalues, 2)
    assert eigenvalues_representable(eigenvalues, time, 2)
    assert clock_values(eigenvalues, time, 2) == pytest.approx([1.0, 2.0])


def test_aliasing_is_detected():
    """An evolution time wrapping lambda_max to phase zero must not pass the check."""
    eigenvalues = np.array([1.0, 2.0])
    assert not eigenvalues_representable(eigenvalues, np.pi, 2)  # lambda=2 -> clock 4 == 0


def test_hhl_reproduces_the_classical_solution_exactly():
    matrix, rhs = well_conditioned_system()
    solution, success, _circuit, _time = solve_hhl(matrix, rhs, num_clock_qubits=2)
    exact = np.linalg.solve(matrix, rhs)
    exact /= np.linalg.norm(exact)
    assert abs(np.vdot(solution, exact)) ** 2 == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < success < 1.0, "postselection must discard some runs"


@pytest.fixture(scope="module")
def hhl_result():
    return run_hhl_intro_tutorial()


def test_hhl_tutorial_hits_unit_fidelity(hhl_result):
    assert hhl_result.eigenvalues_exactly_representable
    assert hhl_result.fidelity == pytest.approx(1.0, abs=1e-9)
    assert hhl_result.condition_number == pytest.approx(2.0)


def test_mis_encoded_eigenvalues_degrade_fidelity(hhl_result):
    """The encoding requirement, measured rather than asserted."""
    exact_rows = [row for row in hhl_result.encoding_study if row["exactly_representable"]]
    inexact_rows = [row for row in hhl_result.encoding_study if not row["exactly_representable"]]
    assert exact_rows and inexact_rows
    assert min(row["fidelity"] for row in exact_rows) > max(
        row["fidelity"] for row in inexact_rows
    )


def test_ill_conditioning_hurts_success_probability(hhl_result):
    """HHL's runtime carries kappa^2; postselection is where that shows up."""
    study = sorted(hhl_result.conditioning_study, key=lambda row: row["condition_number"])
    assert study[0]["success_probability"] > study[-1]["success_probability"]
    assert study[0]["fidelity"] > study[-1]["fidelity"]


def test_readout_cost_is_linear_in_dimension():
    """The caveat that dissolves the exponential speedup."""
    assert measurements_for_precision(1024, 0.01) == 10 * measurements_for_precision(102.4, 0.01)
    assert measurements_for_precision(2, 0.001) > measurements_for_precision(2, 0.01)


# ------------------------------------------------------- variational heat


def test_vqls_cost_is_zero_for_the_true_solution():
    matrix, initial, _grid = heat_equation_system(num_qubits=3)
    exact = np.linalg.solve(matrix, initial)
    target = initial / np.linalg.norm(initial)
    assert vqls_cost(matrix, target, exact / np.linalg.norm(exact)) == pytest.approx(0.0, abs=1e-12)


def test_variational_solver_recovers_the_solution_direction():
    matrix, initial, _grid = heat_equation_system(num_qubits=3)
    state, cost, evaluations, _ansatz = solve_variational_linear_system(
        matrix, initial, 3, restarts=3, seed=0
    )
    exact = np.linalg.solve(matrix, initial)
    exact /= np.linalg.norm(exact)
    assert cost < 1e-6
    assert evaluations > 0
    assert abs(np.dot(state, exact)) ** 2 == pytest.approx(1.0, abs=1e-5)


def test_heat_equation_norm_is_not_recoverable_from_the_state():
    """The normalisation caveat, pinned: diffusion loses norm and |psi> cannot carry it."""
    result = run_variational_heat_equation_tutorial(restarts=3)
    tracking = result.norm_tracking
    assert tracking["norm_lost_to_diffusion"] > 0, "diffusion must reduce the norm"
    assert tracking["recovered_from_quantum_state"] == 0.0
    assert result.fidelity > 0.99


def test_heat_decays_monotonically():
    result = run_variational_heat_equation_tutorial(restarts=2, num_steps=4)
    norms = [step["norm"] for step in result.steps]
    assert all(later < earlier for earlier, later in zip(norms, norms[1:], strict=False))


# ---------------------------------------------------------- Black-Scholes


def test_analytic_price_satisfies_known_bounds():
    """A call is worth at least its discounted intrinsic value and never more than spot."""
    price = float(black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2))
    assert price > 100.0 - 100.0 * np.exp(-0.05)
    assert price < 100.0
    assert float(black_scholes_call(0.0, 100.0, 1.0, 0.05, 0.2)) == 0.0


def test_finite_difference_matches_the_closed_form():
    grid, values, _matrix = solve_black_scholes_pde(num_qubits=6, num_steps=200)
    for spot in (80.0, 100.0, 120.0):
        numerical = float(np.interp(spot, grid, values))
        analytic = float(black_scholes_call(spot, 100.0, 1.0, 0.05, 0.2))
        assert abs(numerical - analytic) < 0.15


def test_black_scholes_tutorial_reports_the_cost_ratio():
    """The point of the tutorial: the cost comparison, not the price."""
    result = run_black_scholes_pde_tutorial(restarts=2, num_steps=50)
    assert result.max_absolute_error < 0.5
    assert result.variational_step["fidelity"] > 0.95
    cost = result.cost_comparison
    assert cost["variational_full_solve_estimate"] > 10_000
    assert "closed-form" in cost["verdict"]
