from __future__ import annotations

import numpy as np


def portfolio_qubo_matrix(expected_returns, covariance, risk_lambda: float = 0.5):
    """Create a simple QUBO matrix for portfolio selection.

    Objective intuition:
    minimize risk_lambda * x.T C x - returns.T x
    """
    q = risk_lambda * np.asarray(covariance).copy()
    for i, ret in enumerate(expected_returns):
        q[i, i] -= ret
    return q


def ising_mapping_note():
    return {
        "binary_variable": "x in {0,1}",
        "spin_variable": "z in {-1,+1}",
        "mapping": "x = (1 - z) / 2",
    }
