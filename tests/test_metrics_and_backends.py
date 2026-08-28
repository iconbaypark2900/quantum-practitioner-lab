"""Metric and backend-adapter checks that do not need the quantum stack."""

import numpy as np
import pytest

from qprac_lab.backends.qiskit_adapter import (
    SUPPORTED_BACKENDS,
    QiskitBackendAdapter,
    qiskit_available,
)
from qprac_lab.metrics.classification import kernel_target_alignment
from qprac_lab.metrics.optimization import (
    constraint_report,
    normalized_approximation_ratio,
)


def test_perfectly_aligned_kernel_scores_higher_than_a_noise_kernel():
    labels = np.array([0, 0, 0, 1, 1, 1])
    signed = np.where(labels == 0, -1.0, 1.0)
    ideal = np.outer(signed, signed)
    assert kernel_target_alignment(ideal, labels) == pytest.approx(1.0, abs=1e-9)
    assert kernel_target_alignment(np.eye(6), labels) < kernel_target_alignment(ideal, labels)


def test_kernel_alignment_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        kernel_target_alignment(np.eye(3), np.array([0, 1]))
    with pytest.raises(ValueError):
        kernel_target_alignment(np.zeros((2, 3)), np.array([0, 1]))


def test_normalized_approximation_ratio_spans_zero_to_one():
    assert normalized_approximation_ratio(5.0, 5.0, -3.0) == pytest.approx(1.0)
    assert normalized_approximation_ratio(-3.0, 5.0, -3.0) == pytest.approx(0.0)
    assert normalized_approximation_ratio(1.0, 5.0, -3.0) == pytest.approx(0.5)
    # Degenerate range must not divide by zero.
    assert normalized_approximation_ratio(2.0, 2.0, 2.0) == pytest.approx(1.0)


def test_normalized_ratio_handles_negative_objectives_correctly():
    """The raw candidate/optimal ratio is misleading when values go negative."""
    better, worse, optimum, floor = -0.1, -1.0, 0.3, -1.5
    assert normalized_approximation_ratio(better, optimum, floor) > (
        normalized_approximation_ratio(worse, optimum, floor)
    )


def test_constraint_report_counts_violations():
    assert constraint_report([1, 1, 0, 0], 2)["budget_constraint_satisfied"]
    report = constraint_report([1, 1, 1, 0], 2)
    assert not report["budget_constraint_satisfied"]
    assert report["constraint_violations"] == 1


def test_adapter_precision_reflects_the_shot_budget():
    assert QiskitBackendAdapter(shots=None).precision == 0.0
    assert QiskitBackendAdapter(shots=10_000).precision == pytest.approx(0.01)
    assert QiskitBackendAdapter(shots=None).describe()["exact_expectation_values"]


def test_adapter_rejects_bad_configuration():
    with pytest.raises(ValueError):
        QiskitBackendAdapter(backend="not_a_backend")
    with pytest.raises(ValueError):
        QiskitBackendAdapter(shots=0)


def test_adapter_describe_works_without_qiskit_installed():
    """describe() is a plain probe and must never require the optional extra."""
    described = QiskitBackendAdapter().describe()
    assert described["backend"] in SUPPORTED_BACKENDS
    assert described["qiskit_installed"] == qiskit_available()
    assert isinstance(described["versions"], dict)


@pytest.mark.parametrize("backend", ["statevector", "aer"])
def test_shot_based_estimation_is_reproducible(backend):
    """Regression: Aer's EstimatorV2 ignores a plain ``seed`` option.

    It only honours ``run_options["seed_simulator"]``. Passing the wrong key is
    accepted silently and leaves every run unseeded, which quietly makes any
    published shot-based number irreproducible.
    """
    pytest.importorskip("qiskit")
    if backend == "aer":
        pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    circuit = QuantumCircuit(2)
    circuit.ry(0.7, 0)
    circuit.cx(0, 1)
    observable = SparsePauliOp.from_list([("ZI", 1.0), ("XX", 0.5)])

    values = []
    for _ in range(3):
        estimator = QiskitBackendAdapter(backend=backend, shots=4096, seed=42).estimator()
        values.append(float(estimator.run([(circuit, observable)]).result()[0].data.evs))
    assert values[0] == values[1] == values[2]


def test_shot_based_sampling_is_reproducible():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    counts = []
    for _ in range(2):
        sampler = QiskitBackendAdapter(shots=1024, seed=7).sampler()
        counts.append(sampler.run([circuit]).result()[0].data.meas.get_counts())
    assert counts[0] == counts[1]
