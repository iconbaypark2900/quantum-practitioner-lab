"""Tutorial 3: quantum kernel classification for biomedical link prediction.

A quantum kernel measures the overlap between two data-encoded quantum states,

    K(x, x') = |<phi(x') | phi(x)>|^2

where ``phi`` is a parameterised feature-map circuit. The quantum computer only
ever produces that similarity number; the classifier on top is an ordinary SVM
over the precomputed kernel matrix. That is the whole idea -- the quantum part
is a similarity measure, not a learner.

The comparison is deliberately unflattering to the quantum method. A ZZ feature
map on classically-generated Gaussian-blob features has no reason to beat an RBF
kernel, and usually does not. Reporting that plainly, alongside kernel-target
alignment which predicts it before any classifier is trained, is more useful
than a demo tuned until the quantum line wins.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit
from qprac_lab.baselines.classical_ml import (
    predict_scores,
    train_precomputed_svm,
    train_random_forest,
    train_rbf_svm,
    train_xgboost_classifier,
)
from qprac_lab.data.synthetic import make_biomedical_pair_features
from qprac_lab.metrics.classification import (
    compute_classification_metrics,
    kernel_target_alignment,
)


@dataclass
class BiomedicalKernelClassificationReport:
    """Quantum kernel results next to every classical baseline."""

    algorithm: str
    use_case: str
    algorithm_type: str
    backend: dict
    dataset: dict
    feature_map: dict
    quantum_kernel_metrics: dict
    rbf_svm_metrics: dict
    random_forest_metrics: dict
    xgboost_metrics: dict | str
    kernel_alignment: dict
    ranking: list[dict[str, Any]]
    best_model: str
    quantum_beats_all_classical: bool
    kernel_matrix_preview: list[list[float]]
    runtime_seconds: dict
    notes: dict[str, Any] = field(default_factory=dict)


def build_quantum_kernel(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str = "linear",
    backend: str = "statevector",
    seed: int = 42,
):
    """Build a ZZ-feature-map fidelity kernel.

    The ZZ feature map is the Havlicek et al. construction: single-qubit
    rotations encode each feature, and entangling ZZ rotations encode products of
    features. Its classical simulation is believed hard at depth, which is the
    (conjectured) source of any advantage.
    """
    require_qiskit("Building a quantum kernel")
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    feature_map = zz_feature_map(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement=entanglement,
    )
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    descriptor = {
        "name": "zz_feature_map",
        "feature_dimension": feature_dimension,
        "reps": reps,
        "entanglement": entanglement,
        "num_qubits": feature_map.num_qubits,
        "circuit_depth": feature_map.decompose().depth(),
        "fidelity": "ComputeUncompute",
        "backend": QiskitBackendAdapter(backend=backend, seed=seed).describe(),
    }
    return kernel, descriptor


def run_quantum_kernel_biomedical_tutorial(
    n_pairs: int = 80,
    embedding_dim: int = 4,
    reps: int = 2,
    test_size: float = 0.3,
    backend: str = "statevector",
    seed: int = 42,
    max_preview: int = 12,
) -> BiomedicalKernelClassificationReport:
    """Run tutorial 3 end to end: quantum kernel SVM vs classical baselines.

    ``n_pairs`` drives the cost quadratically -- the kernel needs one circuit pair
    per entry of an ``n x n`` matrix -- which is itself the tutorial's main
    practical lesson about quantum kernels.
    """
    require_qiskit("The quantum-kernel tutorial")
    x, y = make_biomedical_pair_features(n_pairs=n_pairs, embedding_dim=embedding_dim)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    kernel, feature_map_descriptor = build_quantum_kernel(
        feature_dimension=embedding_dim, reps=reps, backend=backend, seed=seed
    )

    start = time.perf_counter()
    kernel_train = kernel.evaluate(x_vec=x_train)
    kernel_test = kernel.evaluate(x_vec=x_test, y_vec=x_train)
    quantum_kernel_seconds = time.perf_counter() - start

    quantum_svm = train_precomputed_svm(kernel_train, y_train)
    quantum_metrics = compute_classification_metrics(
        y_test,
        quantum_svm.predict(kernel_test),
        quantum_svm.decision_function(kernel_test),
    ).__dict__

    start = time.perf_counter()
    svm = train_rbf_svm(x_train, y_train)
    rbf_metrics = compute_classification_metrics(
        y_test, svm.predict(x_test), predict_scores(svm, x_test)
    ).__dict__
    rbf_seconds = time.perf_counter() - start

    forest = train_random_forest(x_train, y_train)
    forest_metrics = compute_classification_metrics(
        y_test, forest.predict(x_test), predict_scores(forest, x_test)
    ).__dict__

    boosted = train_xgboost_classifier(x_train, y_train)
    if boosted is None:
        xgboost_metrics: dict | str = "install_optional_extra_xgboost"
    else:
        xgboost_metrics = compute_classification_metrics(
            y_test, boosted.predict(x_test), predict_scores(boosted, x_test)
        ).__dict__

    alignment = _alignment_report(kernel_train, x_train, y_train)
    ranking = _rank_models(
        {
            "quantum_kernel_svm": quantum_metrics,
            "rbf_svm": rbf_metrics,
            "random_forest": forest_metrics,
            **({"xgboost": xgboost_metrics} if isinstance(xgboost_metrics, dict) else {}),
        }
    )
    best_model = ranking[0]["model"]

    return BiomedicalKernelClassificationReport(
        algorithm="quantum_kernel_biomedical_classification",
        use_case="biomedical_kg_link_prediction",
        algorithm_type="kernel_method_qsvc",
        backend=QiskitBackendAdapter(backend=backend, seed=seed).describe(),
        dataset={
            "n_pairs": n_pairs,
            "embedding_dim": embedding_dim,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "positive_rate": float(np.mean(y)),
            "source": "synthetic KG-style pair features",
        },
        feature_map=feature_map_descriptor,
        quantum_kernel_metrics=quantum_metrics,
        rbf_svm_metrics=rbf_metrics,
        random_forest_metrics=forest_metrics,
        xgboost_metrics=xgboost_metrics,
        kernel_alignment=alignment,
        ranking=ranking,
        best_model=best_model,
        quantum_beats_all_classical=best_model == "quantum_kernel_svm",
        kernel_matrix_preview=np.asarray(kernel_train)[:max_preview, :max_preview]
        .round(4)
        .tolist(),
        runtime_seconds={
            "quantum_kernel_matrices": quantum_kernel_seconds,
            "rbf_svm_total": rbf_seconds,
            "quantum_overhead_factor": (
                quantum_kernel_seconds / rbf_seconds if rbf_seconds else float("inf")
            ),
        },
        notes={
            "kernel_definition": "K(x, x') = |<phi(x')|phi(x)>|^2",
            "classifier": "SVC(kernel='precomputed') -- equivalent to qiskit QSVC",
            "circuit_evaluations": int(len(x_train) * (len(x_train) - 1) / 2)
            + int(len(x_test) * len(x_train)),
            "scaling": "kernel cost is quadratic in dataset size",
            "alignment": "kernel-target alignment predicts separability before training",
        },
    )


def _alignment_report(kernel_train, x_train, y_train) -> dict:
    """Compare quantum and RBF kernel-target alignment on the same training split."""
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(x_train)
    classical_kernel = rbf_kernel(scaled)
    quantum = kernel_target_alignment(kernel_train, y_train)
    classical = kernel_target_alignment(classical_kernel, y_train)
    return {
        "quantum_kernel": quantum,
        "rbf_kernel": classical,
        "quantum_higher": bool(quantum > classical),
        "interpretation": (
            "higher alignment means the kernel separates the classes better before "
            "any classifier is fitted"
        ),
    }


def _rank_models(metrics_by_model: dict[str, dict]) -> list[dict[str, Any]]:
    """Rank models by ROC-AUC, falling back to F1 when AUC is unavailable."""
    rows = [
        {
            "model": name,
            "roc_auc": metrics.get("roc_auc"),
            "f1": metrics.get("f1"),
            "accuracy": metrics.get("accuracy"),
        }
        for name, metrics in metrics_by_model.items()
    ]
    rows.sort(key=lambda row: (row["roc_auc"] or 0.0, row["f1"] or 0.0), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
