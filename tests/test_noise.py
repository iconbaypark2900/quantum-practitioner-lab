"""Device noise models.

The recurring hazard with simulated noise is that it silently fails to apply --
a wrong option key, or gates outside the model's basis -- and the run then
reports ideal results under a noisy label. These tests assert that noise is
actually doing something, not merely configured.
"""

import numpy as np
import pytest

from qprac_lab.backends.noise import (
    NOISE_PRESETS,
    build_noise_model,
    noise_basis_gates,
    noise_spec,
)
from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter


def test_presets_are_ordered_by_severity():
    light, moderate, heavy = (NOISE_PRESETS[n] for n in ("light", "moderate", "heavy"))
    assert light.two_qubit_error < moderate.two_qubit_error < heavy.two_qubit_error
    assert light.readout_error < moderate.readout_error < heavy.readout_error
    assert all(spec.single_qubit_error < spec.two_qubit_error for spec in NOISE_PRESETS.values())


def test_unknown_preset_is_rejected_with_options_listed():
    with pytest.raises(ValueError, match="light"):
        noise_spec("not_a_preset")


def test_noise_requires_the_aer_backend():
    """Statevector has no noise support; silently ignoring it would be worse."""
    with pytest.raises(ValueError, match="backend='aer'"):
        QiskitBackendAdapter(backend="statevector", noise="moderate")


def test_adapter_validates_the_preset_at_construction():
    with pytest.raises(ValueError):
        QiskitBackendAdapter(backend="aer", noise="not_a_preset")


def test_describe_reports_the_noise_configuration():
    ideal = QiskitBackendAdapter(backend="aer").describe()
    assert ideal["ideal_device"] is True
    assert ideal["noise"] is None

    noisy = QiskitBackendAdapter(backend="aer", noise="moderate").describe()
    assert noisy["ideal_device"] is False
    assert noisy["noise"]["preset"] == "moderate"
    assert noisy["noise"]["two_qubit_error"] > 0


def test_prepare_is_a_no_op_without_noise():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    assert QiskitBackendAdapter().prepare(circuit) is circuit


def test_prepare_transpiles_into_the_noise_basis():
    """Gates outside the basis would be applied perfectly, under-reporting noise."""
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.7, 1)

    prepared = QiskitBackendAdapter(backend="aer", noise="moderate").prepare(circuit)
    allowed = set(noise_basis_gates("moderate")) | {"barrier", "measure"}
    assert set(prepared.count_ops()) <= allowed
    assert "h" not in prepared.count_ops()


def _expectation(**kwargs):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.7, 1)
    observable = SparsePauliOp.from_list([("ZZ", 1.0)])
    adapter = QiskitBackendAdapter(seed=42, **kwargs)
    prepared = adapter.prepare(circuit)
    return float(adapter.estimator().run([(prepared, observable)]).result()[0].data.evs)


def test_noise_degrades_expectation_values_monotonically():
    """The core check: more noise must move the answer further from ideal."""
    pytest.importorskip("qiskit_aer")
    ideal = _expectation(backend="aer")
    errors = [
        abs(_expectation(backend="aer", noise=preset) - ideal)
        for preset in ("light", "moderate", "heavy")
    ]
    assert errors[0] < errors[1] < errors[2]
    assert errors[0] > 0, "light noise changed nothing -- the model is not being applied"


def test_noisy_runs_are_reproducible():
    pytest.importorskip("qiskit_aer")
    values = [_expectation(backend="aer", noise="moderate", shots=4096) for _ in range(3)]
    assert values[0] == values[1] == values[2]


def test_build_noise_model_covers_one_and_two_qubit_gates():
    pytest.importorskip("qiskit_aer")
    model = build_noise_model("moderate")
    assert {"cx", "sx", "rz"} <= set(model.basis_gates)
    assert model.noise_instructions


@pytest.mark.parametrize("noise", ["light", "heavy"])
def test_self_fidelity_falls_below_one_under_noise(noise):
    """K(x,x) is 1 by definition; the shortfall measures lost circuit fidelity."""
    pytest.importorskip("qiskit_machine_learning")
    from qprac_lab.algorithms.qml.quantum_kernel_biomedical import measure_self_fidelity

    sample = np.linspace(0.1, np.pi - 0.1, 12).reshape(3, 4)
    measured = measure_self_fidelity(sample, embedding_dim=4, noise=noise, shots=1024)
    assert 0.0 < measured["mean_self_fidelity"] < 1.0
    assert measured["min_self_fidelity"] <= measured["mean_self_fidelity"]


def test_vqe_loses_chemical_accuracy_under_noise():
    """The headline hardware finding, pinned so it cannot regress silently."""
    pytest.importorskip("qiskit_aer")
    from qprac_lab.algorithms.simulation.vqe_molecular_energy import (
        run_vqe_molecular_energy_tutorial,
    )

    ideal = run_vqe_molecular_energy_tutorial(
        backend="aer", include_dissociation_curve=False, maxiter=100
    )
    noisy = run_vqe_molecular_energy_tutorial(
        backend="aer", noise="moderate", include_dissociation_curve=False, maxiter=100
    )
    assert ideal.chemical_accuracy_reached
    assert not noisy.chemical_accuracy_reached
    assert noisy.absolute_error > ideal.absolute_error


def test_xy_mixer_feasibility_guarantee_does_not_survive_noise():
    """The XY mixer's 100% feasibility is a property of the *ideal* unitary.

    Depolarizing and readout errors move amplitude out of the fixed-Hamming-weight
    subspace, so the structural guarantee degrades on hardware. Pinned here
    because the ideal result is strong enough to be over-claimed.
    """
    pytest.importorskip("qiskit_aer")
    from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import (
        run_qaoa_portfolio_selection_tutorial,
    )

    ideal = run_qaoa_portfolio_selection_tutorial(
        mixer="xy", reps=4, shots=1024, maxiter=60, backend="aer"
    )
    noisy = run_qaoa_portfolio_selection_tutorial(
        mixer="xy", reps=4, shots=1024, maxiter=60, backend="aer", noise="heavy"
    )
    assert ideal.feasible_probability == 1.0
    assert noisy.feasible_probability < 1.0
