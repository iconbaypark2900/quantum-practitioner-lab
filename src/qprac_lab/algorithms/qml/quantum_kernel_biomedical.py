from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split

from qprac_lab.data.synthetic import make_biomedical_pair_features
from qprac_lab.baselines.classical_ml import (
    train_rbf_svm,
    train_random_forest,
    train_xgboost_classifier,
)
from qprac_lab.metrics.classification import compute_classification_metrics


@dataclass
class BiomedicalKernelClassificationReport:
    algorithm: str
    use_case: str
    algorithm_type: str
    rbf_svm_metrics: dict
    random_forest_metrics: dict
    xgboost_metrics: dict | str
    kernel_matrix_preview: list[list[float]]


def simple_rbf_kernel_preview(x, gamma: float = 0.5, max_points: int = 10):
    """Classical kernel preview used until quantum kernel matrix is implemented."""
    subset = x[:max_points]
    kernel = np.zeros((len(subset), len(subset)))
    for i in range(len(subset)):
        for j in range(len(subset)):
            diff = subset[i] - subset[j]
            kernel[i, j] = np.exp(-gamma * float(diff @ diff))
    return kernel


def run_quantum_kernel_biomedical_tutorial() -> BiomedicalKernelClassificationReport:
    """Run the biomedical quantum-kernel tutorial scaffold."""
    x, y = make_biomedical_pair_features(n_pairs=120, embedding_dim=4)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    svm = train_rbf_svm(x_train, y_train)
    svm_preds = svm.predict(x_test)
    svm_scores = svm.predict_proba(x_test)[:, 1]
    svm_metrics = compute_classification_metrics(y_test, svm_preds, svm_scores).__dict__

    rf = train_random_forest(x_train, y_train)
    rf_preds = rf.predict(x_test)
    rf_scores = rf.predict_proba(x_test)[:, 1]
    rf_metrics = compute_classification_metrics(y_test, rf_preds, rf_scores).__dict__

    xgb = train_xgboost_classifier(x_train, y_train)
    if xgb is None:
        xgb_metrics: dict | str = "install_optional_extra_xgboost"
    else:
        xgb_preds = xgb.predict(x_test)
        xgb_scores = xgb.predict_proba(x_test)[:, 1]
        xgb_metrics = compute_classification_metrics(y_test, xgb_preds, xgb_scores).__dict__

    kernel_preview = simple_rbf_kernel_preview(x_train)

    return BiomedicalKernelClassificationReport(
        algorithm="quantum_kernel_biomedical_classification",
        use_case="biomedical_kg_link_prediction",
        algorithm_type="kernel_method_qsvc",
        rbf_svm_metrics=svm_metrics,
        random_forest_metrics=rf_metrics,
        xgboost_metrics=xgb_metrics,
        kernel_matrix_preview=kernel_preview.round(4).tolist(),
    )
