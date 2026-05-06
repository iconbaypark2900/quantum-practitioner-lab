from __future__ import annotations

import numpy as np


def exact_lowest_eigenvalue(matrix) -> float:
    """Exact diagonalization baseline for small Hamiltonians.

    Algorithm type:
    - Classical dense eigensolver.
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    return float(np.min(eigenvalues))
