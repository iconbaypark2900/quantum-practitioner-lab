from __future__ import annotations

import numpy as np


def exact_lowest_eigenvalue(matrix) -> float:
    """Exact diagonalization baseline for small Hamiltonians.

    Algorithm type:
    - Classical dense eigensolver.

    The input is Hermitian but not necessarily real. An earlier version cast with
    ``dtype=float``, which discarded the imaginary part instead of refusing it:
    handed the Pauli-Y matrix it returned ``0.0`` rather than ``-1.0``, with only a
    ``ComplexWarning`` to show for it. Nothing caught that because nothing called
    this module -- which is the argument for wiring a baseline in rather than
    leaving it beside the code that reimplements it.
    """
    matrix = np.asarray(matrix)
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("exact diagonalisation expects a Hermitian matrix")
    return float(np.linalg.eigvalsh(matrix)[0])
