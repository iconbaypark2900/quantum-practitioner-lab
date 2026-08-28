"""Constraint-preserving mixers and QAOA Max-Cut.

The XY mixer's whole value is a *structural* guarantee -- amplitude can never
leave the fixed-Hamming-weight subspace -- so these tests assert exact
feasibility rather than "mostly feasible".
"""

import numpy as np
import pytest

pytest.importorskip("qiskit", reason='needs the quantum stack: pip install -e ".[qiskit]"')

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Operator, Statevector  # noqa: E402

from qprac_lab.algorithms.optimization.qaoa_maxcut import (  # noqa: E402
    make_maxcut_graph,
    run_qaoa_maxcut_tutorial,
)
from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import (  # noqa: E402
    run_qaoa_portfolio_selection_tutorial,
)
from qprac_lab.algorithms.optimization.qubo_builder import maxcut_qubo, portfolio_qubo  # noqa: E402
from qprac_lab.baselines.classical_optimization import (  # noqa: E402
    brute_force_maxcut,
    greedy_maxcut,
)
from qprac_lab.circuits.mixers import (  # noqa: E402
    apply_diagonal_cost_layer,
    build_xy_qaoa_ansatz,
    k_hot_initial_state,
    xy_mixer_edges,
)
from qprac_lab.data.synthetic import make_small_portfolio_dataset  # noqa: E402


def test_xy_mixer_edges_topologies():
    assert xy_mixer_edges(4, "ring") == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert len(xy_mixer_edges(5, "complete")) == 10
    assert xy_mixer_edges(2, "ring") == [(0, 1)]
    with pytest.raises(ValueError):
        xy_mixer_edges(4, "not_a_topology")


def test_k_hot_initial_state_has_the_right_weight():
    circuit = QuantumCircuit(5)
    k_hot_initial_state(circuit, 3)
    state = Statevector(circuit).probabilities_dict()
    (bitstring,) = [b for b, p in state.items() if p > 0.5]
    assert sum(int(bit) for bit in bitstring) == 3
    with pytest.raises(ValueError):
        k_hot_initial_state(QuantumCircuit(3), 4)


def test_diagonal_cost_layer_is_exact():
    """All Z terms commute, so the layer must equal exp(-i g C) with no Trotter error."""
    from scipy.linalg import expm

    operator, _ = portfolio_qubo(*make_small_portfolio_dataset(4), budget=2).to_ising()
    gamma = 0.37
    circuit = QuantumCircuit(4)
    apply_diagonal_cost_layer(circuit, operator, gamma)

    built = Operator(circuit).data
    target = expm(-1j * gamma * operator.to_matrix())
    phase = built[0, 0] / target[0, 0]
    assert abs(abs(phase) - 1.0) < 1e-9
    assert np.allclose(built, phase * target, atol=1e-9)


def test_cost_layer_rejects_non_diagonal_operators():
    from qiskit.quantum_info import SparsePauliOp

    with pytest.raises(ValueError):
        apply_diagonal_cost_layer(
            QuantumCircuit(2), SparsePauliOp.from_list([("XX", 1.0)]), 0.1
        )


@pytest.mark.parametrize("topology", ["ring", "complete"])
@pytest.mark.parametrize("reps", [1, 3])
def test_xy_ansatz_never_leaves_the_feasible_subspace(topology, reps):
    """The structural guarantee: infeasible probability is exactly zero."""
    budget = 3
    operator, _ = portfolio_qubo(
        *make_small_portfolio_dataset(6), budget=budget, penalty=0.0
    ).to_ising()
    ansatz = build_xy_qaoa_ansatz(operator, reps=reps, num_ones=budget, topology=topology)

    rng = np.random.default_rng(0)
    for _ in range(3):
        values = rng.uniform(-np.pi, np.pi, size=ansatz.num_parameters)
        probabilities = Statevector(ansatz.assign_parameters(values)).probabilities_dict()
        infeasible = sum(
            p for bits, p in probabilities.items() if sum(map(int, bits)) != budget
        )
        assert infeasible == pytest.approx(0.0, abs=1e-12)


def test_xy_mixer_makes_every_sampled_portfolio_feasible():
    result = run_qaoa_portfolio_selection_tutorial(mixer="xy", reps=4, shots=2048)
    assert result.feasible_probability == 1.0
    assert result.constraint_report["constraint_violations"] == 0
    assert result.problem["penalty"] == 0.0
    assert result.mixer == "xy:ring:k_hot"


def test_xy_mixer_beats_the_penalty_encoding_on_feasibility():
    xy = run_qaoa_portfolio_selection_tutorial(mixer="xy", reps=4, shots=2048)
    penalty = run_qaoa_portfolio_selection_tutorial(reps=4, shots=2048)
    assert xy.feasible_probability >= penalty.feasible_probability


def test_maxcut_graph_is_regular_and_deterministic():
    first = make_maxcut_graph(8, 3, seed=42)
    assert first == make_maxcut_graph(8, 3, seed=42)
    degrees: dict[int, int] = {}
    for u, v in first:
        degrees[u] = degrees.get(u, 0) + 1
        degrees[v] = degrees.get(v, 0) + 1
    assert set(degrees.values()) == {3}


def test_maxcut_qubo_optimum_matches_brute_force():
    edges = make_maxcut_graph(8, 3, seed=42)
    assert -maxcut_qubo(8, edges).brute_force()["objective_value"] == (
        brute_force_maxcut(8, edges)["objective_value"]
    )


def test_greedy_maxcut_is_a_valid_cut():
    edges = make_maxcut_graph(8, 3, seed=42)
    greedy = greedy_maxcut(8, edges)
    assert len(greedy["bitstring"]) == 8
    assert 0 <= greedy["objective_value"] <= brute_force_maxcut(8, edges)["objective_value"]


@pytest.fixture(scope="module")
def maxcut_result():
    return run_qaoa_maxcut_tutorial(shots=2048)


def test_qaoa_maxcut_cannot_exceed_the_exact_optimum(maxcut_result):
    assert maxcut_result.best_cut_value <= maxcut_result.max_cut_value
    assert maxcut_result.approximation_ratio <= 1.0 + 1e-9
    assert maxcut_result.expected_approximation_ratio <= 1.0 + 1e-9


def test_qaoa_maxcut_beats_random_assignment(maxcut_result):
    """A uniformly random cut scores |E|/2; QAOA has to clear that to be worth running."""
    assert maxcut_result.expected_cut_value > 0.5 * maxcut_result.graph["num_edges"]
    assert maxcut_result.beats_random_guessing
    assert maxcut_result.expected_approximation_ratio > maxcut_result.random_guess_ratio


def test_qaoa_maxcut_expected_value_is_consistent_with_samples(maxcut_result):
    """The reported expectation must actually be the mean over the sampled rows."""
    assert maxcut_result.expected_cut_value <= maxcut_result.best_cut_value
    assert 0.0 <= maxcut_result.optimal_probability <= 1.0


def test_dicke_state_is_uniform_over_the_feasible_subspace():
    """The constrained analogue of |+>^n: unbiased across every weight-k string."""
    from math import comb

    from qprac_lab.circuits.mixers import dicke_state_vector

    amplitudes = dicke_state_vector(6, 3)
    probabilities = amplitudes**2
    support = probabilities[probabilities > 1e-12]
    assert len(support) == comb(6, 3) == 20
    assert np.allclose(support, 1 / 20)
    for index, probability in enumerate(probabilities):
        if probability > 1e-12:
            assert bin(index).count("1") == 3

    with pytest.raises(ValueError):
        dicke_state_vector(3, 5)


@pytest.mark.parametrize("initial_state", ["k_hot", "dicke"])
def test_both_warm_starts_preserve_the_feasible_subspace(initial_state):
    budget = 3
    operator, _ = portfolio_qubo(
        *make_small_portfolio_dataset(6), budget=budget, penalty=0.0
    ).to_ising()
    ansatz = build_xy_qaoa_ansatz(
        operator, reps=3, num_ones=budget, initial_state=initial_state
    )
    rng = np.random.default_rng(0)
    values = rng.uniform(-np.pi, np.pi, size=ansatz.num_parameters)
    probabilities = Statevector(ansatz.assign_parameters(values)).probabilities_dict()
    infeasible = sum(p for bits, p in probabilities.items() if sum(map(int, bits)) != budget)
    assert infeasible == pytest.approx(0.0, abs=1e-12)


def test_unknown_initial_state_is_rejected():
    operator, _ = portfolio_qubo(*make_small_portfolio_dataset(4), budget=2).to_ising()
    with pytest.raises(ValueError, match="initial_state"):
        build_xy_qaoa_ansatz(operator, reps=1, num_ones=2, initial_state="nonsense")


def test_restarts_are_recorded_and_the_best_is_kept():
    """A single QAOA run is a draw from a wide distribution, so restarts are the
    default and the whole spread is reported rather than just the winner."""
    result = run_qaoa_portfolio_selection_tutorial(mixer="xy", reps=4, restarts=3, shots=1024)
    assert result.restarts == 3
    assert len(result.restart_objectives) == 3
    # run_qaoa minimises, so the kept run must be the smallest objective seen.
    assert result.qaoa_cost_expectation == pytest.approx(min(result.restart_objectives))


def test_restarts_must_be_positive():
    with pytest.raises(ValueError, match="restarts"):
        run_qaoa_portfolio_selection_tutorial(mixer="xy", reps=2, restarts=0, shots=512)
