from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1: float
    roc_auc: float | None = None


def compute_classification_metrics(y_true, y_pred, y_score=None) -> ClassificationMetrics:
    roc_auc = None
    if y_score is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc_auc = None

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
        roc_auc=roc_auc,
    )


def kernel_target_alignment(kernel_matrix, labels) -> float:
    """Alignment between a kernel and the ideal kernel ``y y^T``, in ``[-1, 1]``.

    Defined as the normalised Frobenius inner product

        A(K, y y^T) = y^T K y / (n * ||K||_F)

    with labels mapped to +/-1. It measures how well a kernel separates the
    classes *before* any classifier is trained, which makes it the cheapest way
    to tell whether a quantum feature map is doing anything useful -- a quantum
    kernel with lower alignment than RBF will not out-classify it.
    """
    kernel_matrix = np.asarray(kernel_matrix, dtype=float)
    labels = np.asarray(labels)
    if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
        raise ValueError(f"kernel matrix must be square, got {kernel_matrix.shape}")
    if len(labels) != kernel_matrix.shape[0]:
        raise ValueError(
            f"got {len(labels)} labels for a {kernel_matrix.shape[0]}x"
            f"{kernel_matrix.shape[1]} kernel matrix"
        )

    signed = np.where(labels == np.unique(labels)[0], -1.0, 1.0)
    frobenius_norm = float(np.linalg.norm(kernel_matrix, "fro"))
    if frobenius_norm == 0:
        return 0.0
    return float(signed @ kernel_matrix @ signed / (len(signed) * frobenius_norm))
