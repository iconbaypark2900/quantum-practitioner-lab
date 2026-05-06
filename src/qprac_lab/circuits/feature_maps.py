from __future__ import annotations


def describe_zz_feature_map(num_features: int, reps: int = 2):
    """Backend-neutral placeholder for a ZZFeatureMap description.

    Replace with Qiskit ZZFeatureMap in the Qiskit implementation phase.
    """
    return {
        "name": "ZZFeatureMap",
        "num_features": num_features,
        "reps": reps,
        "target_backend": "qiskit",
    }
