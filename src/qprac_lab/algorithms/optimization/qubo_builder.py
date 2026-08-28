"""QUBO construction and the QUBO -> Ising mapping.

This is the bridge every quantum optimisation tutorial crosses. A combinatorial
problem is first written as a QUBO over binary variables,

    minimise  x^T Q x + offset,   x in {0, 1}^n

and then mapped onto qubits by substituting ``x = (1 - z) / 2`` with ``z`` the
+/-1 eigenvalue of a Pauli-Z. The result is an Ising Hamiltonian whose ground
state is the QUBO optimum, which is what QAOA actually minimises.

Hard constraints do not survive that translation -- an Ising Hamiltonian has no
notion of "subject to". They have to be folded into the objective as penalty
terms, which is why :func:`portfolio_qubo` takes a ``penalty`` and why the
resulting samples still need a feasibility check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from qprac_lab.backends.qiskit_adapter import require_qiskit


@dataclass
class QUBO:
    """A quadratic unconstrained binary optimisation problem.

    ``matrix`` need not be symmetric; only the combination ``Q[i, j] + Q[j, i]``
    affects the objective, and the Ising mapping symmetrises internally.
    """

    matrix: np.ndarray
    offset: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"QUBO matrix must be square, got shape {self.matrix.shape}")

    @property
    def num_variables(self) -> int:
        return int(self.matrix.shape[0])

    def objective(self, x) -> float:
        """Evaluate ``x^T Q x + offset`` for a binary vector."""
        x = np.asarray(x, dtype=float)
        return float(x @ self.matrix @ x + self.offset)

    def brute_force(self) -> dict[str, Any]:
        """Exact minimiser by exhaustive enumeration -- for small ``n`` only."""
        best_x, best_value = None, float("inf")
        for bits in product([0, 1], repeat=self.num_variables):
            candidate = np.array(bits, dtype=int)
            value = self.objective(candidate)
            if value < best_value:
                best_value, best_x = value, candidate
        return {"selection": best_x, "objective_value": best_value}

    def to_ising(self):
        """Map to ``(SparsePauliOp, offset)`` with ``x = (1 - z) / 2``.

        Substituting into ``x^T Q x`` and collecting terms gives, for the
        symmetrised couplings ``S[i, j] = (Q[i, j] + Q[j, i]) / 2``:

            constant  = sum_i Q[i, i] / 2 + sum_{i<j} S[i, j] / 2
            h_i       = -Q[i, i] / 2 - sum_{j != i} S[i, j] / 2
            J_ij      = S[i, j] / 2                       (i < j)

        The diagonal is handled as ``x_i`` rather than ``x_i^2`` -- they are equal
        for binary variables, and folding it into the linear term is what keeps
        the mapping exact.
        """
        require_qiskit("Mapping a QUBO to an Ising Hamiltonian")
        from qiskit.quantum_info import SparsePauliOp

        n = self.num_variables
        q = self.matrix
        symmetric = (q + q.T) / 2.0

        constant = float(np.trace(q)) / 2.0 + self.offset
        linear = -np.diag(q) / 2.0
        quadratic: list[tuple[str, list[int], float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                coupling = symmetric[i, j]
                if coupling == 0.0:
                    continue
                constant += coupling / 2.0
                linear[i] -= coupling / 2.0
                linear[j] -= coupling / 2.0
                quadratic.append(("ZZ", [i, j], coupling / 2.0))

        terms: list[tuple[str, list[int], float]] = [
            ("Z", [i], float(linear[i])) for i in range(n) if linear[i] != 0.0
        ]
        terms.extend(quadratic)
        if not terms:
            # Degenerate but valid: a constant objective still needs an operator.
            terms = [("I", [0], 0.0)]

        operator = SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()
        return operator, constant

    def bitstring_to_selection(self, bitstring: str) -> np.ndarray:
        """Decode a Qiskit measurement bitstring into a binary selection vector.

        Qiskit prints bitstrings little-endian: the leftmost character is the
        highest-index qubit. Getting this backwards silently produces a
        mirror-image answer that still looks plausible, so it is done in one
        place rather than at every call site.
        """
        if len(bitstring) != self.num_variables:
            raise ValueError(
                f"bitstring {bitstring!r} has length {len(bitstring)}, "
                f"expected {self.num_variables}"
            )
        return np.array([int(bit) for bit in reversed(bitstring)], dtype=int)


def portfolio_qubo(
    expected_returns,
    covariance,
    budget: int,
    risk_lambda: float = 0.5,
    penalty: float | None = None,
) -> QUBO:
    """Budget-constrained mean-variance portfolio selection as a QUBO.

    The constrained problem is

        maximise  mu^T x - lambda * x^T Sigma x    subject to  sum(x) = budget

    QUBO form flips the sign to a minimisation and absorbs the cardinality
    constraint into a quadratic penalty ``P * (sum(x) - budget)^2``:

        minimise  -mu^T x + lambda * x^T Sigma x + P * (sum(x) - budget)^2

    Expanding the penalty and using ``x_i^2 = x_i`` gives a diagonal contribution
    of ``P - 2 * P * budget`` per asset, an off-diagonal ``P`` per pair, and a
    constant ``P * budget^2``.
    """
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = len(expected_returns)
    if covariance.shape != (n, n):
        raise ValueError(f"covariance shape {covariance.shape} does not match {n} assets")
    if not 0 <= budget <= n:
        raise ValueError(f"budget {budget} outside 0..{n}")

    if penalty is None:
        penalty = default_penalty(expected_returns, covariance, risk_lambda)

    q = risk_lambda * covariance.copy()
    q += penalty * np.ones((n, n))
    np.fill_diagonal(
        q,
        np.diag(risk_lambda * covariance) - expected_returns + penalty - 2 * penalty * budget,
    )

    return QUBO(
        matrix=q,
        offset=penalty * budget**2,
        metadata={
            "problem": "budget_constrained_mean_variance",
            "budget": budget,
            "risk_lambda": risk_lambda,
            "penalty": float(penalty),
        },
    )


def default_penalty(expected_returns, covariance, risk_lambda: float = 0.5) -> float:
    """A penalty large enough that violating the budget is never worthwhile.

    Bounded by the total swing the objective can produce: no rearrangement of the
    portfolio can gain more than ``sum|mu| + lambda * sum|Sigma|``, so a penalty
    above that makes any constraint violation strictly unprofitable.
    """
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    swing = float(np.abs(expected_returns).sum() + risk_lambda * np.abs(covariance).sum())
    return max(1.0, swing)


def maxcut_qubo(num_nodes: int, edges: list[tuple[int, int]]) -> QUBO:
    """Max-Cut as a QUBO.

    Cutting edge ``(i, j)`` scores ``x_i + x_j - 2 x_i x_j``; minimising the
    negation maximises the cut.
    """
    q = np.zeros((num_nodes, num_nodes), dtype=float)
    for i, j in edges:
        q[i, i] -= 1.0
        q[j, j] -= 1.0
        q[i, j] += 1.0
        q[j, i] += 1.0
    return QUBO(matrix=q, metadata={"problem": "maxcut", "num_edges": len(edges)})


def ising_mapping_note() -> dict[str, str]:
    """Human-readable summary of the substitution used by :meth:`QUBO.to_ising`."""
    return {
        "binary_variable": "x in {0,1}",
        "spin_variable": "z in {-1,+1}",
        "mapping": "x = (1 - z) / 2",
        "operator": "z is the +/-1 eigenvalue of Pauli-Z",
        "constraints": "hard constraints must become penalty terms before mapping",
    }
