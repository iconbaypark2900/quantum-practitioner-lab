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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit
from qprac_lab.baselines.classical_ml import (
    predict_scores,
    train_precomputed_svm,
    train_random_forest,
    train_rbf_svm,
    train_xgboost_classifier,
)
from qprac_lab.data.hetionet import make_hetionet_link_prediction_dataset
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
    evaluation: dict
    quantum_kernel_metrics: dict
    rbf_svm_metrics: dict
    random_forest_metrics: dict
    xgboost_metrics: dict | str
    kernel_alignment: dict
    quantum_vs_rbf: dict
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


def load_dataset(
    dataset: str = "hetionet",
    n_pairs: int = 200,
    embedding_dim: int = 4,
    seed: int = 42,
    allow_download: bool = True,
):
    """Load pair features and labels, returning ``(x, y, metadata)``.

    ``hetionet`` is real drug--disease link prediction; ``synthetic`` is the old
    Gaussian-blob generator, kept only as an offline fallback for environments
    without the data. It is not a substitute -- an RBF kernel is near-ideal for
    that geometry, so results on it say little.
    """
    if dataset == "hetionet":
        built = make_hetionet_link_prediction_dataset(
            n_pairs=n_pairs,
            embedding_dim=embedding_dim,
            seed=seed,
            allow_download=allow_download,
        )
        return (
            built.features,
            built.labels,
            {
                "name": "hetionet_ctd_link_prediction",
                "real_data": True,
                "feature_names": list(built.feature_names),
                **built.metadata,
            },
        )
    if dataset == "synthetic":
        x, y = make_biomedical_pair_features(n_pairs=n_pairs, embedding_dim=embedding_dim)
        return (
            x,
            y,
            {
                "name": "synthetic_gaussian_blobs",
                "real_data": False,
                "warning": (
                    "make_classification blobs are close to ideal RBF geometry; any "
                    "quantum-vs-classical conclusion drawn here is an artifact of the "
                    "generator, not a finding"
                ),
            },
        )
    raise ValueError(f"Unknown dataset {dataset!r}; expected 'hetionet' or 'synthetic'")


def _summarise(scores: list[dict]) -> dict:
    """Mean and standard deviation of per-fold metrics."""
    keys = ("accuracy", "f1", "roc_auc")
    summary: dict[str, Any] = {"n_evaluations": len(scores)}
    for key in keys:
        values = [s[key] for s in scores if s.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
        summary[f"{key}_std"] = float(np.std(values)) if values else None
    return summary


def run_quantum_kernel_biomedical_tutorial(
    dataset: str = "hetionet",
    n_pairs: int = 200,
    embedding_dim: int = 4,
    reps: int = 2,
    n_splits: int = 5,
    n_repeats: int = 4,
    backend: str = "statevector",
    seed: int = 42,
    max_preview: int = 12,
    allow_download: bool = True,
) -> BiomedicalKernelClassificationReport:
    """Run tutorial 3 end to end: quantum kernel SVM vs classical baselines.

    Evaluation is **repeated stratified cross-validation**, not a single split.
    That is not statistical fussiness -- at the dataset sizes a quantum kernel can
    afford, one split is genuinely uninformative. On this data a single 80-sample
    split produced test ROC-AUCs anywhere from 0.54 to 0.85 depending only on the
    split seed, which is wider than any difference between the models.

    The quantum kernel makes this affordable: the full ``n x n`` matrix is
    computed **once** up front, and every fold then reuses submatrices of it, so
    20 evaluations cost the same quantum time as one.
    """
    require_qiskit("The quantum-kernel tutorial")
    x, y, dataset_metadata = load_dataset(
        dataset=dataset,
        n_pairs=n_pairs,
        embedding_dim=embedding_dim,
        seed=seed,
        allow_download=allow_download,
    )

    # Angle encoding needs bounded inputs. The Hetionet features span three
    # orders of magnitude (a Jaccard index of 0.07 beside an overlap count of
    # 102), and feeding those in raw would wrap the feature-map rotations many
    # times over. Fitted on the whole feature matrix because the kernel is
    # computed once for all folds; it uses no labels, so no target information
    # crosses the split.
    angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(x)
    x_angles = angle_scaler.transform(x)

    kernel, feature_map_descriptor = build_quantum_kernel(
        feature_dimension=embedding_dim, reps=reps, backend=backend, seed=seed
    )

    start = time.perf_counter()
    full_kernel = np.asarray(kernel.evaluate(x_vec=x_angles))
    quantum_kernel_seconds = time.perf_counter() - start

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    quantum_scores: list[dict] = []
    rbf_scores: list[dict] = []
    forest_scores: list[dict] = []
    boosted_scores: list[dict] = []
    xgboost_available = True

    classical_start = time.perf_counter()
    for train_index, test_index in splitter.split(x, y):
        y_train, y_test = y[train_index], y[test_index]

        quantum_svm = train_precomputed_svm(
            full_kernel[np.ix_(train_index, train_index)], y_train
        )
        test_block = full_kernel[np.ix_(test_index, train_index)]
        quantum_scores.append(
            compute_classification_metrics(
                y_test, quantum_svm.predict(test_block), quantum_svm.decision_function(test_block)
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

        boosted = train_xgboost_classifier(x_train, y_train)
        if boosted is None:
            xgboost_available = False
        else:
            boosted_scores.append(
                compute_classification_metrics(
                    y_test, boosted.predict(x_test), predict_scores(boosted, x_test)
                ).__dict__
            )
    classical_seconds = time.perf_counter() - classical_start

    quantum_metrics = _summarise(quantum_scores)
    rbf_metrics = _summarise(rbf_scores)
    forest_metrics = _summarise(forest_scores)
    xgboost_metrics: dict | str = (
        _summarise(boosted_scores) if xgboost_available else "install_optional_extra_xgboost"
    )

    alignment = _alignment_report(full_kernel, x, y)
    models = {
        "quantum_kernel_svm": quantum_metrics,
        "rbf_svm": rbf_metrics,
        "random_forest": forest_metrics,
    }
    if isinstance(xgboost_metrics, dict):
        models["xgboost"] = xgboost_metrics
    ranking = _rank_models(models)
    best_model = ranking[0]["model"]

    return BiomedicalKernelClassificationReport(
        algorithm="quantum_kernel_biomedical_classification",
        use_case="biomedical_kg_link_prediction",
        algorithm_type="kernel_method_qsvc",
        backend=QiskitBackendAdapter(backend=backend, seed=seed).describe(),
        dataset={
            "n_pairs": int(len(y)),
            "embedding_dim": embedding_dim,
            "positive_rate": float(np.mean(y)),
            "angle_encoding_range": [0.0, float(np.pi)],
            **dataset_metadata,
        },
        feature_map=feature_map_descriptor,
        evaluation={
            "scheme": "RepeatedStratifiedKFold",
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "n_evaluations": n_splits * n_repeats,
            "why": (
                "a single split on a dataset this small swings ROC-AUC by more than "
                "the gap between the models"
            ),
        },
        quantum_kernel_metrics=quantum_metrics,
        rbf_svm_metrics=rbf_metrics,
        random_forest_metrics=forest_metrics,
        xgboost_metrics=xgboost_metrics,
        kernel_alignment=alignment,
        quantum_vs_rbf=_paired_comparison(quantum_scores, rbf_scores),
        ranking=ranking,
        best_model=best_model,
        quantum_beats_all_classical=best_model == "quantum_kernel_svm",
        kernel_matrix_preview=full_kernel[:max_preview, :max_preview].round(4).tolist(),
        runtime_seconds={
            "quantum_kernel_matrix": quantum_kernel_seconds,
            "classical_models_total": classical_seconds,
            "quantum_overhead_factor": (
                quantum_kernel_seconds / classical_seconds if classical_seconds else float("inf")
            ),
        },
        notes={
            "kernel_definition": "K(x, x') = |<phi(x')|phi(x)>|^2",
            "classifier": "SVC(kernel='precomputed') -- equivalent to qiskit QSVC",
            "circuit_evaluations": int(len(y) * (len(y) - 1) / 2),
            "scaling": "kernel cost is quadratic in dataset size",
            "alignment": "kernel-target alignment predicts separability before training",
            "kernel_reuse": (
                "the full kernel is computed once; every CV fold reuses submatrices, "
                "so repeated evaluation costs no extra quantum time"
            ),
        },
    )


def _paired_comparison(quantum_scores: list[dict], rbf_scores: list[dict]) -> dict:
    """Compare the quantum kernel against RBF on the *same* folds.

    A paired comparison is the only fair one here: both models saw identical
    splits, so fold-to-fold noise cancels. ``wins`` counts folds where the
    quantum kernel scored higher, which is far more informative than comparing
    two means that each carry a large standard deviation.
    """
    pairs = [
        (q["roc_auc"], r["roc_auc"])
        for q, r in zip(quantum_scores, rbf_scores, strict=True)
        if q["roc_auc"] is not None and r["roc_auc"] is not None
    ]
    if not pairs:
        return {"comparable_folds": 0}
    differences = np.array([q - r for q, r in pairs])
    wins = int((differences > 0).sum())
    return {
        "comparable_folds": len(pairs),
        "mean_roc_auc_difference": float(differences.mean()),
        "std_roc_auc_difference": float(differences.std()),
        "quantum_wins": wins,
        "quantum_win_rate": wins / len(pairs),
        "difference_exceeds_noise": bool(
            abs(differences.mean()) > differences.std() / np.sqrt(len(differences))
        ),
        "interpretation": "positive mean difference means the quantum kernel scored higher",
    }


def _alignment_report(kernel_matrix, x, y) -> dict:
    """Compare quantum and RBF kernel-target alignment on identical data."""
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(x)
    classical_kernel = rbf_kernel(scaled)
    quantum = kernel_target_alignment(kernel_matrix, y)
    classical = kernel_target_alignment(classical_kernel, y)
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
    """Rank models by mean ROC-AUC, falling back to F1 when AUC is unavailable."""
    rows = [
        {
            "model": name,
            "roc_auc": metrics.get("roc_auc"),
            "roc_auc_std": metrics.get("roc_auc_std"),
            "f1": metrics.get("f1"),
            "accuracy": metrics.get("accuracy"),
        }
        for name, metrics in metrics_by_model.items()
    ]
    rows.sort(key=lambda row: (row["roc_auc"] or 0.0, row["f1"] or 0.0), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
