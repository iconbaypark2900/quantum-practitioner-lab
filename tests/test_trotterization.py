"""Trotterization: product-formula error and how it scales.

The recurring hazard is measuring *no* error at all. An undecomposed
``PauliEvolutionGate`` evaluates as the exact matrix exponential, so a Trotter
study built on it reports 1e-15 at every step count and looks like a triumph.
That specific failure is pinned below.
"""

import numpy as np
import pytest

pytest.importorskip("qiskit", reason='needs the quantum stack: pip install -e ".[qiskit]"')

from qiskit.quantum_info import Operator  # noqa: E402

from qprac_lab.algorithms.simulation.trotterization import (  # noqa: E402
    exact_evolution_operator,
    noise_tradeoff,
    run_trotterization_tutorial,
    transverse_field_ising_hamiltonian,
    trotter_circuit,
    trotter_operator_error,
)


def test_tfim_has_non_commuting_parts():
    """Without that, Trotterisation would be exact and the tutorial pointless."""
    hamiltonian = transverse_field_ising_hamiltonian(4)
    labels = hamiltonian.paulis.to_labels()
    coefficients = hamiltonian.coeffs
    from qiskit.quantum_info import SparsePauliOp

    diagonal = SparsePauliOp.from_list(
        [(lab, c) for lab, c in zip(labels, coefficients, strict=True) if "X" not in lab]
    )
    transverse = SparsePauliOp.from_list(
        [(lab, c) for lab, c in zip(labels, coefficients, strict=True) if "X" in lab]
    )
    commutator = (diagonal @ transverse - transverse @ diagonal).simplify()
    assert not np.allclose(commutator.to_matrix(), 0)


def test_hamiltonian_validation():
    with pytest.raises(ValueError):
        transverse_field_ising_hamiltonian(1)
    with pytest.raises(ValueError):
        trotter_circuit(transverse_field_ising_hamiltonian(3), 1.0, steps=0)
    with pytest.raises(ValueError):
        trotter_circuit(transverse_field_ising_hamiltonian(3), 1.0, steps=1, order=5)


def test_undecomposed_evolution_gate_hides_all_trotter_error():
    """Regression pin for the trap this module exists to avoid.

    Evaluated undecomposed, a one-step circuit looks exact. Decomposed, its error
    is order 1. Any Trotter benchmark that skips the decomposition is measuring
    ``expm`` and reporting it as a quantum result.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    hamiltonian = transverse_field_ising_hamiltonian(4)
    time = 1.5
    circuit = QuantumCircuit(4)
    circuit.append(
        PauliEvolutionGate(hamiltonian, time=time, synthesis=LieTrotter(reps=1)), range(4)
    )
    exact = exact_evolution_operator(hamiltonian, time)

    hidden = np.linalg.norm(np.asarray(Operator(circuit).data) - exact, 2)
    real = np.linalg.norm(np.asarray(Operator(circuit.decompose(reps=4)).data) - exact, 2)
    assert hidden < 1e-9, "undecomposed gate should evaluate as exact"
    assert real > 0.5, "decomposed one-step Trotter should carry real error"


@pytest.mark.parametrize("order", [1, 2])
def test_error_decreases_monotonically_with_steps(order):
    hamiltonian = transverse_field_ising_hamiltonian(4)
    errors = [trotter_operator_error(hamiltonian, 1.5, steps, order) for steps in (2, 4, 8, 16)]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:], strict=False))


def test_second_order_beats_first_order_at_equal_steps():
    hamiltonian = transverse_field_ising_hamiltonian(4)
    for steps in (4, 8, 16):
        first = trotter_operator_error(hamiltonian, 1.5, steps, order=1)
        second = trotter_operator_error(hamiltonian, 1.5, steps, order=2)
        assert second < first


def test_fitted_scaling_exponents_match_theory():
    """First order should scale as 1/steps and second order as 1/steps^2."""
    result = run_trotterization_tutorial(include_noise_tradeoff=False)
    assert result.fitted_exponents["order_1"] == pytest.approx(1.0, abs=0.25)
    assert result.fitted_exponents["order_2"] == pytest.approx(2.0, abs=0.25)


def test_depth_grows_with_step_count():
    """The cost side of the trade: accuracy is bought with circuit depth."""
    hamiltonian = transverse_field_ising_hamiltonian(4)
    depths = [trotter_circuit(hamiltonian, 1.5, steps, 2).depth() for steps in (1, 2, 4, 8)]
    assert all(later > earlier for earlier, later in zip(depths, depths[1:], strict=False))


def test_noise_reverses_the_benefit_of_more_steps():
    """The headline: under noise the optimum is at finite depth, not maximum depth."""
    pytest.importorskip("qiskit_aer")
    hamiltonian = transverse_field_ising_hamiltonian(4)
    rows = noise_tradeoff(hamiltonian, 1.5, step_counts=(2, 4, 8, 16, 32), order=2, shots=4096)

    ideal = [row["ideal_error"] for row in rows]
    assert ideal[-1] < ideal[0], "ideal error must keep falling with more steps"

    noisy = [row["noisy_error"] for row in rows]
    best = int(np.argmin(noisy))
    assert best < len(noisy) - 1, "under noise the best step count should not be the largest"
    assert noisy[-1] > noisy[best]


def test_tutorial_reports_an_optimum_under_noise():
    pytest.importorskip("qiskit_aer")
    result = run_trotterization_tutorial()
    available = {row["steps"] for row in result.noise_tradeoff}
    assert result.optimal_steps_under_noise["steps"] in available
    assert result.trotter_error > 0
    assert result.circuit_depth > 0
