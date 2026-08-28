from __future__ import annotations

from qprac_lab.baselines.classical_pdes import solve_small_linear_system


def run_hhl_intro_scaffold():
    """HHL intro scaffold.

    Algorithm type:
    - Quantum linear systems algorithm.

    Current implementation:
    - Classical linear solve baseline.
    """
    a, b, x = solve_small_linear_system()
    return {
        "algorithm": "hhl_intro",
        "use_case": "quantum_linear_systems_for_pdes",
        "algorithm_type": "quantum_linear_systems_algorithm",
        "status": "classical_baseline_ready",
        "matrix": a.tolist(),
        "rhs": b.tolist(),
        "solution": x.tolist(),
    }
