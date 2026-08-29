"""Variational quantum classifier on the same data as the quantum kernel.

Two quantum approaches to the same supervised problem, and they differ in a way
that matters more than either paper suggests:

**Quantum kernel** (tutorial 3) computes pairwise similarities, then hands them to
an SVM. The classical half is a *convex* problem -- one global optimum, found
reliably. Cost is quadratic in dataset size and paid once.

**VQC** (here) trains a parameterised circuit directly against a loss. The
optimisation is *non-convex* in the circuit parameters, so it inherits every
problem VQE has: local minima, initialisation sensitivity, and barren plateaus.
Cost is linear per epoch and paid on every iteration.

Same feature map, same dataset, same folds -- so the comparison isolates the
learning strategy rather than the encoding. Whether the training difficulty is
worth it is the question this tutorial exists to answer, and the answer here is
not flattering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from qprac_lab.algorithms.qml.quantum_kernel_biomedical import load_dataset
from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit
from qprac_lab.baselines.classical_ml import (
    predict_scores,
    train_precomputed_svm,
    train_random_forest,
    train_rbf_svm,
)
from qprac_lab.metrics.classification import compute_classification_metrics


@dataclass
class VQCClassificationReport:
    """VQC against the quantum kernel and classical baselines on identical folds."""

    algorithm: str
    use_case: str
    algorithm_type: str
    backend: dict
    dataset: dict
    circuit: dict
    evaluation: dict
    vqc_metrics: dict
    quantum_kernel_metrics: dict
    rbf_svm_metrics: dict
    random_forest_metrics: dict
    ranking: list[dict[str, Any]]
    best_model: str
    vqc_vs_kernel: dict
    training_cost: dict
    barren_plateau: dict = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def build_vqc_circuit(num_features: int, feature_reps: int = 2, ansatz_reps: int = 2):
    """Feature map followed by a trainable ansatz.

    Returns ``(circuit, num_data_parameters, num_weight_parameters, observable)``.
    The observable is the full-register parity ``Z...Z``; its expectation lies in
    ``[-1, 1]`` and maps to a class probability by ``(1 + <P>) / 2``.
    """
    require_qiskit("Building a VQC circuit")
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
    from qiskit.quantum_info import SparsePauliOp

    feature_map = zz_feature_map(feature_dimension=num_features, reps=feature_reps)
    ansatz = real_amplitudes(num_features, reps=ansatz_reps)
    circuit = feature_map.compose(ansatz)
    observable = SparsePauliOp.from_list([("Z" * num_features, 1.0)])
    return circuit, feature_map.num_parameters, ansatz.num_parameters, observable


def vqc_expectations(estimator, circuit, observable, features, weights) -> np.ndarray:
    """Circuit expectation for every sample, in one broadcast primitive call.

    V2 primitives accept a parameter *array*, so all samples go in a single pub
    rather than one call each -- measured at 337 ms for 200 samples versus
    seconds for the naive loop. Without this, training is not practical.
    """
    features = np.atleast_2d(features)
    bound = np.hstack([features, np.tile(np.asarray(weights), (len(features), 1))])
    result = estimator.run([(circuit, observable, bound)]).result()
    return np.atleast_1d(np.asarray(result[0].data.evs, dtype=float))


def _probabilities(expectations: np.ndarray) -> np.ndarray:
    return np.clip((1.0 + expectations) / 2.0, 1e-7, 1 - 1e-7)


def binary_cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    return float(
        -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
    )


def train_vqc(
    features,
    labels,
    feature_reps: int = 2,
    ansatz_reps: int = 2,
    maxiter: int = 120,
    backend: str = "statevector",
    seed: int = 42,
    noise: str | None = None,
    shots: int | None = None,
):
    """Train a VQC by minimising binary cross-entropy over the circuit weights.

    Returns ``(weights, threshold, loss_history, evaluations, parts)`` where
    ``threshold`` is fitted on the *training* expectations -- the circuit has no
    trainable bias, so the decision boundary has to come from somewhere, and
    taking it from training data keeps the test split clean.
    """
    require_qiskit("Training a VQC")
    features = np.atleast_2d(features)
    labels = np.asarray(labels)
    circuit, num_data, num_weights, observable = build_vqc_circuit(
        features.shape[1], feature_reps, ansatz_reps
    )
    adapter = QiskitBackendAdapter(backend=backend, shots=shots, seed=seed, noise=noise)
    estimator = adapter.estimator()
    prepared = adapter.prepare(circuit)

    rng = np.random.default_rng(seed)
    initial = rng.uniform(-0.1, 0.1, size=num_weights)
    history: list[float] = []
    evaluations = 0

    def objective(weights):
        nonlocal evaluations
        evaluations += 1
        loss = binary_cross_entropy(
            _probabilities(
                vqc_expectations(estimator, prepared, observable, features, weights)
            ),
            labels,
        )
        history.append(loss)
        return loss

    result = minimize(objective, x0=initial, method="COBYLA", options={"maxiter": maxiter})
    weights = np.atleast_1d(result.x)
    train_expectations = vqc_expectations(
        estimator, prepared, observable, features, weights
    )
    threshold = float(np.median(train_expectations))
    parts = {
        "circuit": prepared,
        "observable": observable,
        "estimator": estimator,
        "num_data_parameters": num_data,
        "num_weight_parameters": num_weights,
    }
    return weights, threshold, history, evaluations, parts


def barren_plateau_scan(
    qubit_counts: tuple[int, ...] = (2, 4, 6, 8),
    ansatz_reps: int = 2,
    samples: int = 30,
    seed: int = 42,
) -> dict:
    """Variance of a loss gradient component against qubit count.

    The barren-plateau diagnostic. If the variance decays exponentially with
    width, a randomly-initialised deep ansatz has essentially no gradient to
    follow, and no optimiser fixes that.
    """
    require_qiskit("Scanning for barren plateaus")
    rng = np.random.default_rng(seed)
    variances = []
    for num_qubits in qubit_counts:
        circuit, num_data, num_weights, observable = build_vqc_circuit(
            num_qubits, feature_reps=1, ansatz_reps=ansatz_reps
        )
        estimator = QiskitBackendAdapter(seed=seed).estimator()
        data = rng.uniform(0, np.pi, size=(1, num_data))
        gradients = []
        for _ in range(samples):
            weights = rng.uniform(-np.pi, np.pi, size=num_weights)
            shifted_plus = weights.copy()
            shifted_minus = weights.copy()
            shifted_plus[0] += np.pi / 2
            shifted_minus[0] -= np.pi / 2
            plus = vqc_expectations(estimator, circuit, observable, data, shifted_plus)[0]
            minus = vqc_expectations(estimator, circuit, observable, data, shifted_minus)[0]
            gradients.append((plus - minus) / 2)
        variances.append(float(np.var(gradients)))

    return {
        "qubit_counts": list(qubit_counts),
        "gradient_variances": variances,
        "decay_factor_per_two_qubits": [
            float(variances[i] / variances[i + 1]) if variances[i + 1] else float("inf")
            for i in range(len(variances) - 1)
        ],
        "interpretation": (
            "variance shrinking with width is the barren-plateau signature: a "
            "randomly-initialised ansatz has no gradient to follow"
        ),
    }


def run_vqc_classifier_tutorial(
    dataset: str = "hetionet",
    n_pairs: int = 200,
    embedding_dim: int = 4,
    feature_reps: int = 2,
    ansatz_reps: int = 2,
    n_splits: int = 5,
    maxiter: int = 120,
    backend: str = "statevector",
    seed: int = 42,
    include_barren_plateau_scan: bool = True,
    allow_download: bool = True,
) -> VQCClassificationReport:
    """Compare VQC, the quantum kernel, and classical baselines on identical folds."""
    require_qiskit("The VQC tutorial")
    from qprac_lab.algorithms.qml.quantum_kernel_biomedical import build_quantum_kernel

    x, y, dataset_metadata = load_dataset(
        dataset=dataset,
        n_pairs=n_pairs,
        embedding_dim=embedding_dim,
        seed=seed,
        allow_download=allow_download,
    )
    angles = MinMaxScaler(feature_range=(0.0, np.pi)).fit_transform(x)

    kernel, feature_map_descriptor = build_quantum_kernel(
        feature_dimension=embedding_dim, reps=feature_reps, backend=backend, seed=seed
    )
    kernel_start = time.perf_counter()
    full_kernel = np.asarray(kernel.evaluate(x_vec=angles))
    kernel_seconds = time.perf_counter() - kernel_start

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    vqc_scores, kernel_scores, rbf_scores, forest_scores = [], [], [], []
    vqc_seconds = 0.0
    vqc_evaluations = 0

    for train_index, test_index in splitter.split(x, y):
        y_train, y_test = y[train_index], y[test_index]

        start = time.perf_counter()
        weights, threshold, _history, evaluations, parts = train_vqc(
            angles[train_index],
            y_train,
            feature_reps=feature_reps,
            ansatz_reps=ansatz_reps,
            maxiter=maxiter,
            backend=backend,
            seed=seed,
        )
        vqc_seconds += time.perf_counter() - start
        vqc_evaluations += evaluations

        test_expectations = vqc_expectations(
            parts["estimator"], parts["circuit"], parts["observable"],
            angles[test_index], weights,
        )
        vqc_scores.append(
            compute_classification_metrics(
                y_test, (test_expectations > threshold).astype(int), test_expectations
            ).__dict__
        )

        quantum_svm = train_precomputed_svm(
            full_kernel[np.ix_(train_index, train_index)], y_train
        )
        block = full_kernel[np.ix_(test_index, train_index)]
        kernel_scores.append(
            compute_classification_metrics(
                y_test, quantum_svm.predict(block), quantum_svm.decision_function(block)
            ).__dict__
        )

        x_train, x_test = x[train_index], x[test_index]
        svm = train_rbf_svm(x_train, y_train)
        rbf_scores.append(
            compute_classification_metrics(
                y_test, svm.predict(x_test), predict_scores(svm, x_test)
            ).__dict__
        )
        forest = train_random_forest(x_train, y_train)
        forest_scores.append(
            compute_classification_metrics(
                y_test, forest.predict(x_test), predict_scores(forest, x_test)
            ).__dict__
        )

    def summarise(scores):
        summary = {"n_evaluations": len(scores)}
        for key in ("accuracy", "f1", "roc_auc"):
            values = [s[key] for s in scores if s.get(key) is not None]
            summary[key] = float(np.mean(values)) if values else None
            summary[f"{key}_std"] = float(np.std(values)) if values else None
        return summary

    models = {
        "vqc": summarise(vqc_scores),
        "quantum_kernel_svm": summarise(kernel_scores),
        "rbf_svm": summarise(rbf_scores),
        "random_forest": summarise(forest_scores),
    }
    ranking = sorted(
        (
            {
                "model": name,
                "roc_auc": metrics["roc_auc"],
                "roc_auc_std": metrics["roc_auc_std"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
            }
            for name, metrics in models.items()
        ),
        key=lambda row: (row["roc_auc"] or 0.0, row["f1"] or 0.0),
        reverse=True,
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    paired = [
        (v["roc_auc"], k["roc_auc"])
        for v, k in zip(vqc_scores, kernel_scores, strict=True)
        if v["roc_auc"] is not None and k["roc_auc"] is not None
    ]
    differences = np.array([v - k for v, k in paired]) if paired else np.array([])
    comparison = {
        "comparable_folds": len(paired),
        "mean_roc_auc_difference": float(differences.mean()) if len(differences) else 0.0,
        "std_roc_auc_difference": float(differences.std()) if len(differences) else 0.0,
        "vqc_wins": int((differences > 0).sum()),
        "difference_exceeds_noise": bool(
            len(differences)
            and abs(differences.mean()) > differences.std() / np.sqrt(len(differences))
        ),
        "interpretation": "positive mean difference means the VQC scored higher than the kernel",
    }

    _circuit, num_data, num_weights, _observable = build_vqc_circuit(
        embedding_dim, feature_reps, ansatz_reps
    )
    n_train = len(y) - len(y) // n_splits

    return VQCClassificationReport(
        algorithm="vqc_classifier",
        use_case="biomedical_kg_link_prediction",
        algorithm_type="variational_quantum_classifier",
        backend=QiskitBackendAdapter(backend=backend, seed=seed).describe(),
        dataset={"n_pairs": int(len(y)), "embedding_dim": embedding_dim, **dataset_metadata},
        circuit={
            "feature_map": feature_map_descriptor["name"],
            "feature_map_reps": feature_reps,
            "ansatz": "real_amplitudes",
            "ansatz_reps": ansatz_reps,
            "num_data_parameters": num_data,
            "num_weight_parameters": num_weights,
            "observable": "full-register parity Z...Z",
            "loss": "binary cross-entropy",
            "optimizer": "COBYLA",
        },
        evaluation={
            "scheme": "StratifiedKFold",
            "n_splits": n_splits,
            "shared_folds": True,
            "why": "identical folds for every model, so the comparison is paired",
        },
        vqc_metrics=models["vqc"],
        quantum_kernel_metrics=models["quantum_kernel_svm"],
        rbf_svm_metrics=models["rbf_svm"],
        random_forest_metrics=models["random_forest"],
        ranking=ranking,
        best_model=ranking[0]["model"],
        vqc_vs_kernel=comparison,
        training_cost={
            "vqc_seconds": vqc_seconds,
            "quantum_kernel_seconds": kernel_seconds,
            "vqc_objective_evaluations": vqc_evaluations,
            "vqc_circuit_evaluations": int(vqc_evaluations * n_train),
            "kernel_circuit_evaluations": int(len(y) * (len(y) - 1) / 2),
            "note": (
                "the kernel pays a one-off quadratic cost; the VQC pays a linear "
                "cost on every optimiser iteration, and there are many iterations"
            ),
        },
        barren_plateau=barren_plateau_scan(seed=seed) if include_barren_plateau_scan else {},
        notes={
            "convexity": (
                "the kernel SVM is convex and finds its global optimum; VQC training "
                "is non-convex and inherits VQE's local minima and initialisation "
                "sensitivity"
            ),
            "threshold": "fitted on training expectations; the circuit has no trainable bias",
            "controlled_comparison": "same feature map, same data, same folds",
        },
    )
