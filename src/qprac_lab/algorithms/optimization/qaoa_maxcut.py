from __future__ import annotations

from qprac_lab.baselines.classical_optimization import brute_force_maxcut


def run_qaoa_maxcut_scaffold():
    """QAOA Max-Cut scaffold.

    Algorithm type:
    - Hybrid variational combinatorial optimization.
    """
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    baseline = brute_force_maxcut(n_nodes=4, edges=edges)
    return {
        "algorithm": "qaoa_maxcut",
        "status": "classical_baseline_ready",
        "baseline": baseline,
    }
