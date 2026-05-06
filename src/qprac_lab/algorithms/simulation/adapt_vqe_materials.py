from __future__ import annotations


def run_adapt_vqe_materials_scaffold():
    """ADAPT-VQE scaffold.

    Algorithm type:
    - Adaptive variational eigensolver.

    Build loop:
    1. Define operator pool.
    2. Measure gradients.
    3. Add best operator.
    4. Re-optimize.
    5. Stop when gradients are small.
    """
    operator_pool = ["X0", "Y0", "X0Y1", "Y0X1"]
    selected_operators = ["X0"]
    return {
        "algorithm": "adapt_vqe_materials",
        "status": "scaffold",
        "operator_pool": operator_pool,
        "selected_operators": selected_operators,
    }
