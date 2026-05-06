from __future__ import annotations

from dataclasses import dataclass
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
