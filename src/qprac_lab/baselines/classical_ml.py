from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def predict_scores(model, x):
    """Continuous scores for ROC-AUC, whatever the estimator offers.

    ``decision_function`` is preferred over ``predict_proba``: for SVMs the
    probabilities come from an extra Platt-scaling fit that costs a cross
    validation and changes nothing about the ranking ROC-AUC measures. Asking
    for them is also deprecated in scikit-learn 1.9.
    """
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x))
    return np.asarray(model.predict_proba(x))[:, 1]


def train_rbf_svm(x_train, y_train):
    """Classical baseline for quantum kernel tutorials.

    Algorithm type:
    - Support Vector Machine with RBF kernel.
    """
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf")),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, random_state: int = 42):
    """Classical tree ensemble baseline."""
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def train_xgboost_classifier(x_train, y_train, random_state: int = 42):
    """Gradient boosting baseline when ``xgboost`` is installed; otherwise ``None``."""
    try:
        import xgboost as xgb
    except ImportError:
        return None
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=80,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=random_state,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_precomputed_svm(kernel_train, y_train, c: float = 1.0):
    """SVM over a precomputed kernel matrix.

    This is what turns any kernel -- quantum included -- into a classifier: the
    SVM only ever sees pairwise similarities, never the feature vectors.
    """
    model = SVC(kernel="precomputed", C=c)
    model.fit(kernel_train, y_train)
    return model
