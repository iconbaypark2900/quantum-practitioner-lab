"""Variational quantum classifier.

VQC is the non-convex counterpart to the quantum kernel, so these check the
training machinery is sound (broadcast evaluation matches per-sample evaluation,
loss actually decreases) rather than that the accuracy is good -- on this data no
model is far from chance, and asserting otherwise would encode noise.
"""

import numpy as np
import pytest

pytest.importorskip("qiskit", reason='needs the quantum stack: pip install -e ".[qiskit]"')

from qprac_lab.algorithms.qml.vqc_classifier import (  # noqa: E402
    barren_plateau_scan,
    binary_cross_entropy,
    build_vqc_circuit,
    run_vqc_classifier_tutorial,
    train_vqc,
    vqc_expectations,
)
from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter  # noqa: E402


def test_circuit_splits_data_and_weight_parameters():
    circuit, num_data, num_weights, observable = build_vqc_circuit(4, 2, 2)
    assert num_data == 4, "one data parameter per feature"
    assert num_weights > 0
    assert circuit.num_parameters == num_data + num_weights
    assert observable.paulis.to_labels() == ["ZZZZ"]


def test_broadcast_evaluation_matches_per_sample_evaluation():
    """Training is only practical batched, so the batching must be exact."""
    circuit, num_data, num_weights, observable = build_vqc_circuit(4, 2, 2)
    estimator = QiskitBackendAdapter(seed=1).estimator()
    rng = np.random.default_rng(0)
    features = rng.uniform(0, np.pi, size=(6, num_data))
    weights = rng.uniform(-np.pi, np.pi, size=num_weights)

    batched = vqc_expectations(estimator, circuit, observable, features, weights)
    assert batched.shape == (6,)
    for index in range(6):
        single = vqc_expectations(
            estimator, circuit, observable, features[index : index + 1], weights
        )
        assert batched[index] == pytest.approx(single[0], abs=1e-12)


def test_expectations_stay_within_the_pauli_range():
    circuit, num_data, num_weights, observable = build_vqc_circuit(4, 2, 2)
    estimator = QiskitBackendAdapter(seed=1).estimator()
    rng = np.random.default_rng(3)
    values = vqc_expectations(
        estimator,
        circuit,
        observable,
        rng.uniform(0, np.pi, size=(10, num_data)),
        rng.uniform(-np.pi, np.pi, size=num_weights),
    )
    assert np.all(values >= -1 - 1e-9)
    assert np.all(values <= 1 + 1e-9)


def test_cross_entropy_rewards_confident_correct_predictions():
    labels = np.array([1, 0])
    assert binary_cross_entropy(np.array([0.99, 0.01]), labels) < binary_cross_entropy(
        np.array([0.6, 0.4]), labels
    )
    assert binary_cross_entropy(np.array([0.01, 0.99]), labels) > binary_cross_entropy(
        np.array([0.5, 0.5]), labels
    )


def test_training_reduces_the_loss():
    """A separable toy problem, so failure means the training loop is broken."""
    rng = np.random.default_rng(0)
    features = np.vstack(
        [rng.uniform(0.0, 0.6, size=(12, 4)), rng.uniform(2.5, 3.1, size=(12, 4))]
    )
    labels = np.r_[np.zeros(12), np.ones(12)].astype(int)
    _weights, threshold, history, evaluations, _parts = train_vqc(
        features, labels, maxiter=40, seed=0
    )
    assert evaluations > 1
    assert len(history) == evaluations
    assert min(history) < history[0], "optimiser never improved on its starting loss"
    assert np.isfinite(threshold)


def test_barren_plateau_variance_shrinks_with_width():
    """The defining symptom: gradients vanish as the circuit gets wider."""
    scan = barren_plateau_scan(qubit_counts=(2, 4, 6), samples=15, seed=0)
    variances = scan["gradient_variances"]
    assert variances[0] > variances[-1]
    assert all(factor > 1 for factor in scan["decay_factor_per_two_qubits"])


@pytest.fixture(scope="module")
def vqc_result():
    return run_vqc_classifier_tutorial(
        dataset="synthetic", n_pairs=60, n_splits=3, maxiter=40,
        include_barren_plateau_scan=False,
    )


def test_all_four_models_are_scored_on_shared_folds(vqc_result):
    assert vqc_result.evaluation["shared_folds"] is True
    assert vqc_result.evaluation["n_splits"] == 3
    for metrics in (
        vqc_result.vqc_metrics,
        vqc_result.quantum_kernel_metrics,
        vqc_result.rbf_svm_metrics,
        vqc_result.random_forest_metrics,
    ):
        assert metrics["n_evaluations"] == 3
        assert {"accuracy", "f1", "roc_auc"} <= set(metrics)


def test_ranking_is_ordered_and_names_the_best(vqc_result):
    ranks = [row["rank"] for row in vqc_result.ranking]
    assert ranks == sorted(ranks)
    assert vqc_result.ranking[0]["model"] == vqc_result.best_model
    scores = [row["roc_auc"] for row in vqc_result.ranking]
    assert scores == sorted(scores, reverse=True)


def test_paired_comparison_against_the_kernel(vqc_result):
    comparison = vqc_result.vqc_vs_kernel
    assert comparison["comparable_folds"] == 3
    assert -1.0 <= comparison["mean_roc_auc_difference"] <= 1.0
    assert 0 <= comparison["vqc_wins"] <= 3


def test_training_cost_shows_the_per_iteration_penalty(vqc_result):
    """The kernel pays once and quadratically; VQC pays linearly every iteration."""
    cost = vqc_result.training_cost
    assert cost["vqc_circuit_evaluations"] > cost["kernel_circuit_evaluations"]
    assert cost["vqc_objective_evaluations"] > 0
