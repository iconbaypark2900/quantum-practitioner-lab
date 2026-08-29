"""PennyLane cross-check backend.

The point of this backend is catching the failure mode that recurred throughout
this project: a bug that produces plausible numbers instead of an error. An
unseeded estimator, an undecomposed evolution gate, a mirror-image bitstring --
all silent from inside one library. These tests confirm the second stack actually
disagrees when it should.
"""

import numpy as np
import pytest

from qprac_lab.backends.pennylane_adapter import (
    PennyLaneBackendAdapter,
    PennyLaneNotInstalledError,
    pennylane_available,
    pennylane_version,
)

needs_pennylane = pytest.mark.skipif(
    not pennylane_available(),
    reason='needs the cross-check backend: pip install -e ".[pennylane]"',
)
def test_describe_works_without_pennylane_installed():
    described = PennyLaneBackendAdapter().describe()
    assert described["framework"] == "pennylane"
    assert described["pennylane_installed"] == pennylane_available()
    assert described["version"] == pennylane_version()


def test_adapter_validates_shots():
    with pytest.raises(ValueError):
        PennyLaneBackendAdapter(shots=0)


def test_missing_backend_raises_an_actionable_error():
    error = PennyLaneNotInstalledError("Cross-checking")
    assert "pennylane" in str(error)
    assert "pip install" in str(error)


@needs_pennylane
def test_hamiltonian_conversion_preserves_the_spectrum():
    pytest.importorskip("qiskit")
    import pennylane as qml
    from qiskit.quantum_info import SparsePauliOp

    from qprac_lab.backends.pennylane_adapter import to_pennylane_hamiltonian

    operator = SparsePauliOp.from_list(
        [("II", -1.0523732), ("IZ", 0.3979374), ("ZI", -0.3979374),
         ("ZZ", -0.0112801), ("XX", 0.1809312)]
    )
    converted = to_pennylane_hamiltonian(operator)
    qiskit_spectrum = np.sort(np.linalg.eigvalsh(operator.to_matrix()))
    pennylane_spectrum = np.sort(
        np.linalg.eigvalsh(qml.matrix(converted, wire_order=[0, 1]))
    )
    assert np.allclose(qiskit_spectrum, pennylane_spectrum, atol=1e-9)


@needs_pennylane
def test_conversion_gets_qubit_ordering_right():
    """A flipped mapping has an identical spectrum -- only an asymmetric operator
    catches it, which is why this test exists separately from the one above."""
    pytest.importorskip("qiskit")
    import pennylane as qml
    from qiskit.quantum_info import SparsePauliOp, Statevector

    from qprac_lab.backends.pennylane_adapter import to_pennylane_hamiltonian

    operator = SparsePauliOp.from_list([("IZ", 1.0)])  # Z on qubit 0 only
    converted = to_pennylane_hamiltonian(operator)
    device = qml.device("default.qubit", wires=2)

    @qml.qnode(device)
    def flipped_qubit_zero():
        qml.PauliX(0)
        return qml.expval(converted)

    qiskit_value = float(Statevector.from_label("01").expectation_value(operator).real)
    assert float(flipped_qubit_zero()) == pytest.approx(qiskit_value, abs=1e-9)
    assert qiskit_value == pytest.approx(-1.0)


@needs_pennylane
def test_independent_vqe_agrees_with_qiskit():
    """Two unrelated stacks reaching the same energy is the whole point."""
    pytest.importorskip("qiskit")
    from qprac_lab.backends.pennylane_adapter import cross_check_vqe

    result = cross_check_vqe()
    assert result["frameworks_agree"]
    assert result["absolute_difference"] < 1e-6
    assert result["pennylane_energy"] == pytest.approx(-1.137306, abs=1e-5)
    # And both must respect the variational bound.
    assert result["pennylane_energy"] >= result["exact_energy"] - 1e-9


@needs_pennylane
@pytest.mark.parametrize("bond_length", [1.0, 2.5])
def test_cross_check_holds_across_geometries(bond_length):
    pytest.importorskip("qiskit")
    from qprac_lab.backends.pennylane_adapter import cross_check_vqe

    assert cross_check_vqe(bond_length)["frameworks_agree"]


@needs_pennylane
def test_ising_mapping_agrees_across_frameworks():
    """Guards the decoding path, where a mirror image looks entirely plausible."""
    pytest.importorskip("qiskit")
    from qprac_lab.backends.pennylane_adapter import cross_check_ising_mapping

    result = cross_check_ising_mapping()
    assert result["assignments_checked"] == 64
    assert result["frameworks_agree"]
    assert result["max_absolute_difference"] < 1e-9


@needs_pennylane
def test_expectation_matches_a_known_value():
    pytest.importorskip("qiskit")
    import pennylane as qml
    from qiskit.quantum_info import SparsePauliOp

    operator = SparsePauliOp.from_list([("ZZ", 1.0)])
    adapter = PennyLaneBackendAdapter()
    assert adapter.expectation(operator, lambda: None) == pytest.approx(1.0)
    assert adapter.expectation(operator, lambda: qml.PauliX(0)) == pytest.approx(-1.0)
