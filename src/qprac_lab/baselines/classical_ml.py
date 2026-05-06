from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_rbf_svm(x_train, y_train):
    """Classical baseline for quantum kernel tutorials.

    Algorithm type:
    - Support Vector Machine with RBF kernel.
    """
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, random_state: int = 42):
    """Classical tree ensemble baseline."""
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)
    return model
