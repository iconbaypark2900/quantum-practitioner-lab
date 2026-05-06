from __future__ import annotations


def run_vqc_classifier_scaffold():
    """VQC scaffold.

    Algorithm type:
    - Variational quantum classifier.
    """
    return {
        "algorithm": "vqc_classifier",
        "status": "scaffold",
        "next_step": "Add parameterized circuit, expectation output, loss, and optimizer.",
    }
