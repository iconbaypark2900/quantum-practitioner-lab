from __future__ import annotations


def run_qsvc_classifier_scaffold():
    """QSVC scaffold.

    Algorithm type:
    - Quantum kernel support vector classifier.
    """
    return {
        "algorithm": "qsvc_classifier",
        "status": "scaffold",
        "next_step": "Add Qiskit FidelityQuantumKernel and SVC(kernel='precomputed').",
    }
