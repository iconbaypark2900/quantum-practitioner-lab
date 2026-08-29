"""ADAPT-VQE: the ansatz is discovered, so the discovery has to be correct.

A sign error in the gradient would still converge -- just to the wrong ansatz --
so the gradient formula is cross-checked against the parameter-shift rule rather
than assumed.
"""

import numpy as np
import pytest

pytest.importorskip("qiskit", reason='needs the quantum stack: pip install -e ".[qiskit]"')

from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402

from qprac_lab.algorithms.simulation.adapt_vqe_materials import (  # noqa: E402
    build_adapt_circuit,
    operator_gradients,
    qubit_excitation_pool,
    run_adapt_vqe_materials,
)
from qprac_lab.algorithms.simulation.hamiltonian_utils import build_h2_hamiltonian  # noqa: E402
from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter  # noqa: E402


def test_pool_contains_only_odd_y_strings_within_the_weight_cap():
    pool = qubit_excitation_pool(3, max_weight=2)
    assert pool, "pool is empty"
    for label in pool:
        assert label.count("Y") % 2 == 1
        assert 1 <= (len(label) - label.count("I")) <= 2
    assert "III" not in pool


def test_pool_size_grows_with_the_weight_cap():
    assert len(qubit_excitation_pool(3, 1)) < len(qubit_excitation_pool(3, 3))
    with pytest.raises(ValueError):
        qubit_excitation_pool(0)
    with pytest.raises(ValueError):
        qubit_excitation_pool(2, max_weight=0)


def test_analytic_gradient_matches_the_parameter_shift_rule():
    """dE/dtheta = (1/2)<i[P,H]> must equal [E(+pi/2) - E(-pi/2)] / 2, exactly."""
    from scipy.linalg import expm

    hamiltonian = build_h2_hamiltonian(0.735)
    observable = hamiltonian.qubit_operator
    pool = qubit_excitation_pool(hamiltonian.num_qubits, max_weight=2)
    circuit = build_adapt_circuit(
        hamiltonian.num_qubits, hamiltonian.hartree_fock_bitstring, []
    )
    estimator = QiskitBackendAdapter().estimator()
    analytic = operator_gradients(estimator, circuit, [], observable, pool)

    reference = Statevector(circuit).data
    matrix = observable.to_matrix()
    for index, label in enumerate(pool):
        generator = SparsePauliOp.from_list([(label, 1.0)]).to_matrix()

        def energy(angle, generator=generator):
            state = expm(-1j * angle / 2 * generator) @ reference
            return float(np.real(state.conj() @ matrix @ state))

        shift = (energy(np.pi / 2) - energy(-np.pi / 2)) / 2
        assert analytic[index] == pytest.approx(shift, abs=1e-9), f"mismatch on {label}"


@pytest.fixture(scope="module")
def adapt_result():
    return run_adapt_vqe_materials()


def test_adapt_discovers_the_known_h2_generator(adapt_result):
    """The hand-derived UCC ansatz uses an XY-type generator; ADAPT must find it."""
    assert adapt_result.selected_operators == ["XY"]
    assert adapt_result.num_parameters == 1
    assert adapt_result.converged


def test_adapt_reaches_the_exact_energy(adapt_result):
    assert adapt_result.absolute_error < 1e-6
    assert adapt_result.chemical_accuracy_reached
    assert adapt_result.energy >= adapt_result.exact_baseline_energy - 1e-9
    assert adapt_result.energy < adapt_result.hartree_fock_baseline_energy


def test_adapt_uses_far_fewer_parameters_than_a_fixed_ansatz(adapt_result):
    comparison = adapt_result.comparison
    assert comparison["adapt_num_parameters"] < comparison["fixed_num_parameters"]
    assert comparison["parameter_reduction"] >= 10


def test_gradients_fall_below_tolerance_on_convergence(adapt_result):
    assert adapt_result.final_max_gradient < 1e-4
    assert len(adapt_result.iterations) == 1


def test_a_pool_without_the_right_generator_cannot_leave_hartree_fock():
    """Pool choice decides what is reachable -- weight-1 operators have zero gradient."""
    restricted = run_adapt_vqe_materials(max_operator_weight=1)
    assert restricted.selected_operators == []
    assert restricted.energy == pytest.approx(
        restricted.hartree_fock_baseline_energy, abs=1e-9
    )
    assert not restricted.chemical_accuracy_reached


def test_adapt_holds_up_at_a_stretched_bond():
    """Where correlation is strongest and a fixed hardware-efficient ansatz struggles."""
    stretched = run_adapt_vqe_materials(bond_length_angstrom=2.5)
    assert stretched.selected_operators == ["XY"]
    assert stretched.absolute_error < 1e-6
    # Correlation energy is far larger here than at equilibrium.
    correlation = stretched.hartree_fock_baseline_energy - stretched.exact_baseline_energy
    assert correlation > 0.2
