"""The QUBO -> Ising mapping is the correctness-critical step of every quantum
optimisation tutorial: get it wrong and QAOA cheerfully minimises the wrong
Hamiltonian, producing a well-formed answer to a different problem.
"""

import numpy as np
import pytest

from qprac_lab.algorithms.optimization.qubo_builder import (
    QUBO,
    default_penalty,
    maxcut_qubo,
    portfolio_qubo,
)
from qprac_lab.baselines.classical_optimization import brute_force_maxcut, brute_force_portfolio
from qprac_lab.data.synthetic import make_small_portfolio_dataset


def all_bitstrings(n):
    for index in range(2**n):
        yield np.array([(index >> bit) & 1 for bit in range(n)], dtype=int)


def ising_energy(operator, offset, x):
    """Energy of a binary assignment under the mapped Ising Hamiltonian."""
    from qiskit.quantum_info import Statevector

    label = "".join(str(bit) for bit in reversed(x))  # Qiskit is little-endian
    return float(Statevector.from_label(label).expectation_value(operator).real) + offset


@pytest.mark.parametrize(
    "make_qubo",
    [
        pytest.param(lambda: portfolio_qubo(*make_small_portfolio_dataset(5), budget=2), id="p5b2"),
        pytest.param(
            lambda: portfolio_qubo(*make_small_portfolio_dataset(6), budget=3, risk_lambda=1.5),
            id="p6b3",
        ),
        pytest.param(
            lambda: maxcut_qubo(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]), id="maxcut4"
        ),
    ],
)
def test_ising_mapping_matches_the_qubo_on_every_assignment(make_qubo):
    pytest.importorskip("qiskit")
    qubo = make_qubo()
    operator, offset = qubo.to_ising()
    for x in all_bitstrings(qubo.num_variables):
        assert ising_energy(operator, offset, x) == pytest.approx(qubo.objective(x), abs=1e-9)


def test_ising_operator_is_diagonal():
    """A classical cost function must map to Z-terms only, never X or Y."""
    pytest.importorskip("qiskit")
    qubo = portfolio_qubo(*make_small_portfolio_dataset(5), budget=2)
    operator, _ = qubo.to_ising()
    assert all(set(label) <= {"I", "Z"} for label in operator.paulis.to_labels())


def test_portfolio_penalty_reproduces_the_constrained_optimum():
    """With a large enough penalty, the unconstrained QUBO optimum is feasible."""
    returns, covariance = make_small_portfolio_dataset(n_assets=6)
    qubo = portfolio_qubo(returns, covariance, budget=3, risk_lambda=0.5)
    constrained = brute_force_portfolio(returns, covariance, risk_lambda=0.5, budget=3)
    assert np.array_equal(qubo.brute_force()["selection"], constrained["selection"])


def test_weak_penalty_can_break_feasibility():
    """Documents the failure mode: too small a penalty and the optimum cheats."""
    returns, covariance = make_small_portfolio_dataset(n_assets=6)
    weak = portfolio_qubo(returns, covariance, budget=3, risk_lambda=0.5, penalty=1e-6)
    assert weak.brute_force()["selection"].sum() != 3


def test_maxcut_qubo_matches_brute_force_maxcut():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    best = maxcut_qubo(4, edges).brute_force()
    assert -best["objective_value"] == brute_force_maxcut(4, edges)["objective_value"]


def test_bitstring_decoding_is_little_endian():
    qubo = maxcut_qubo(3, [(0, 1)])
    assert qubo.bitstring_to_selection("001").tolist() == [1, 0, 0]
    assert qubo.bitstring_to_selection("100").tolist() == [0, 0, 1]
    with pytest.raises(ValueError):
        qubo.bitstring_to_selection("01")


def test_default_penalty_scales_with_the_objective():
    returns, covariance = make_small_portfolio_dataset(n_assets=6)
    assert default_penalty(returns, covariance) >= 1.0
    assert default_penalty(returns, covariance, risk_lambda=2.0) > default_penalty(
        returns, covariance, risk_lambda=0.5
    )


def test_qubo_rejects_a_non_square_matrix():
    with pytest.raises(ValueError):
        QUBO(matrix=np.zeros((2, 3)))
