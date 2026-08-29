"""HHL: the quantum linear systems algorithm, and why its speedup is conditional.

HHL solves ``Ax = b`` in time polylogarithmic in the system size ``N`` -- an
exponential speedup over any classical solver, on paper. It is the algorithm most
often cited as the reason quantum computers will transform scientific computing,
and it is also the one whose fine print does the most work.

It does not return ``x``. It returns a quantum state ``|x>`` whose *amplitudes*
encode the solution. Reading them out costs ``O(N)`` measurements, which alone
erases the exponential advantage. The speedup survives only when every one of
these holds:

1. ``A`` is sparse and well conditioned (runtime carries ``kappa^2``).
2. ``|b>`` can be prepared efficiently -- often as hard as the original problem.
3. You need a *summary statistic* of ``x``, never ``x`` itself.
4. ``A`` can be exponentiated efficiently to build ``exp(iAt)``.

This module implements HHL honestly at 2x2, verifies it reaches fidelity 1.0
against the classical solution, and then measures each caveat rather than
gesturing at it. Aaronson's "Read the fine print" is the canonical statement of
the case; this is the arithmetic behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import expm

from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit


@dataclass
class HHLResult:
    """An HHL solve, its fidelity against the exact answer, and its real costs."""

    algorithm: str
    use_case: str
    algorithm_type: str
    backend: dict
    matrix: list[list[float]]
    rhs: list[float]
    eigenvalues: list[float]
    condition_number: float
    num_clock_qubits: int
    evolution_time: float
    clock_values: list[float]
    eigenvalues_exactly_representable: bool
    quantum_solution: list[float]
    classical_solution: list[float]
    fidelity: float
    success_probability: float
    circuit_depth: int
    two_qubit_gates: int
    encoding_study: list[dict[str, Any]] = field(default_factory=list)
    conditioning_study: list[dict[str, Any]] = field(default_factory=list)
    caveats: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def well_conditioned_system(gap: float = 0.5):
    """A 2x2 Hermitian system with eigenvalues ``1 +/- gap`` scaled to ``1`` and ``2``.

    Built eigenvalue-first, because HHL's accuracy depends on the eigenvalues
    landing exactly on clock-register values -- a matrix chosen for looks rather
    than spectrum makes the algorithm look broken when it is only mis-encoded.
    """
    low, high = 1.0, 1.0 + 2 * gap
    matrix = np.array([[(low + high) / 2, (high - low) / 2], [(high - low) / 2, (low + high) / 2]])
    return matrix, np.array([1.0, 0.0])


def clock_values(eigenvalues, evolution_time: float, num_clock_qubits: int) -> list[float]:
    """Where each eigenvalue lands in the phase register.

    Quantum phase estimation resolves ``lambda * t / (2 pi)`` as an
    ``n``-bit fraction, so this must be an integer in ``1 .. 2^n - 1`` for the
    eigenvalue to be represented exactly. Zero means the eigenvalue aliased away.
    """
    scale = evolution_time / (2 * np.pi) * 2**num_clock_qubits
    return [float(value * scale) for value in eigenvalues]


def eigenvalues_representable(eigenvalues, evolution_time, num_clock_qubits, tol=1e-9) -> bool:
    values = clock_values(eigenvalues, evolution_time, num_clock_qubits)
    return all(
        abs(value - round(value)) < tol and 1 <= round(value) <= 2**num_clock_qubits - 1
        for value in values
    )


def suggested_evolution_time(eigenvalues, num_clock_qubits: int) -> float:
    """Pick ``t`` so every eigenvalue lands exactly on a clock-register value.

    Two constraints. ``lambda_max * t / (2 pi)`` must stay below 1, or the largest
    eigenvalue wraps to phase zero and its part of the solution is silently lost.
    And each ``lambda * t / (2 pi) * 2^n`` should be an integer, or phase
    estimation spreads that eigenvalue over neighbouring registers and the
    reconstructed solution degrades.

    Scans the candidate times that map ``lambda_max`` onto each integer register
    value and returns the largest that represents *every* eigenvalue exactly,
    falling back to the largest non-aliasing time when the spectrum is
    incommensurate -- which is the common case, and a real limitation rather than
    a tuning inconvenience.
    """
    largest = float(np.max(np.abs(eigenvalues)))
    fallback = float(2 * np.pi / largest * (2**num_clock_qubits - 1) / 2**num_clock_qubits)
    for value in range(2**num_clock_qubits - 1, 0, -1):
        candidate = 2 * np.pi * value / (largest * 2**num_clock_qubits)
        if eigenvalues_representable(eigenvalues, candidate, num_clock_qubits):
            return float(candidate)
    return fallback


def hhl_circuit(matrix, rhs, num_clock_qubits: int = 2, evolution_time: float | None = None):
    """Build the HHL circuit: QPE, controlled rotation, inverse QPE.

    The ancilla is qubit 0, the clock register next, the solution register last.
    """
    require_qiskit("Building an HHL circuit")
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import QFTGate, RYGate, UnitaryGate

    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if evolution_time is None:
        evolution_time = suggested_evolution_time(eigenvalues, num_clock_qubits)

    num_solution_qubits = int(np.log2(len(rhs)))
    clock = QuantumRegister(num_clock_qubits, "clock")
    solution = QuantumRegister(num_solution_qubits, "x")
    ancilla = QuantumRegister(1, "anc")
    circuit = QuantumCircuit(ancilla, clock, solution)

    circuit.initialize(rhs / np.linalg.norm(rhs), solution[:])
    circuit.h(clock[:])
    for index in range(num_clock_qubits):
        power = expm(1j * matrix * evolution_time * (2**index))
        circuit.append(UnitaryGate(power).control(1), [clock[index], *solution[:]])
    circuit.append(QFTGate(num_clock_qubits).inverse(), clock[:])

    # Rotate the ancilla by arcsin(C / lambda) so its |1> branch carries 1/lambda.
    smallest = min(
        value * 2 * np.pi / (evolution_time * 2**num_clock_qubits)
        for value in range(1, 2**num_clock_qubits)
    )
    for value in range(1, 2**num_clock_qubits):
        eigenvalue = value * 2 * np.pi / (evolution_time * 2**num_clock_qubits)
        angle = 2 * np.arcsin(min(smallest / eigenvalue, 1.0))
        circuit.append(
            RYGate(angle).control(
                num_clock_qubits, ctrl_state=format(value, f"0{num_clock_qubits}b")
            ),
            clock[:] + [ancilla[0]],
        )

    circuit.append(QFTGate(num_clock_qubits), clock[:])
    for index in reversed(range(num_clock_qubits)):
        power = expm(-1j * matrix * evolution_time * (2**index))
        circuit.append(UnitaryGate(power).control(1), [clock[index], *solution[:]])
    circuit.h(clock[:])
    return circuit, evolution_time


def solve_hhl(matrix, rhs, num_clock_qubits: int = 2, evolution_time: float | None = None):
    """Run HHL and postselect the ancilla, returning the normalised solution.

    HHL yields ``|x>`` only in the branch where the ancilla measures 1. Everything
    else is discarded, which is why the success probability is a headline cost
    rather than a footnote.
    """
    require_qiskit("Solving with HHL")
    from qiskit.quantum_info import Statevector

    circuit, evolution_time = hhl_circuit(matrix, rhs, num_clock_qubits, evolution_time)
    state = Statevector(circuit)
    dimension = len(rhs)
    num_solution_qubits = int(np.log2(dimension))

    # Ancilla = 1 and clock back to |0...0>; ancilla is the lowest-index qubit.
    amplitudes = np.array(
        [
            state.data[int(f"{index:0{num_solution_qubits}b}" + "0" * num_clock_qubits + "1", 2)]
            for index in range(dimension)
        ]
    )
    success_probability = float(np.linalg.norm(amplitudes) ** 2)
    if success_probability < 1e-12:
        return np.zeros(dimension), 0.0, circuit, evolution_time
    return amplitudes / np.linalg.norm(amplitudes), success_probability, circuit, evolution_time


def measurements_for_precision(dimension: int, precision: float = 0.01) -> int:
    """Shots needed to read the full solution vector to a given precision.

    The number that dissolves the exponential speedup. Each amplitude is estimated
    from sampling, so its error falls as ``1/sqrt(shots)``; recovering all ``N``
    of them costs ``O(N / precision^2)``. HHL's advantage survives only when you
    never do this.
    """
    return int(dimension / precision**2)


def run_hhl_intro_tutorial(
    num_clock_qubits: int = 2,
    gap: float = 0.5,
    precision: float = 0.01,
) -> HHLResult:
    """Solve a 2x2 system with HHL and measure every caveat in its fine print."""
    require_qiskit("The HHL tutorial")
    matrix, rhs = well_conditioned_system(gap)
    eigenvalues = np.linalg.eigvalsh(matrix)
    classical = np.linalg.solve(matrix, rhs)
    classical_normalised = classical / np.linalg.norm(classical)

    quantum, success, circuit, evolution_time = solve_hhl(matrix, rhs, num_clock_qubits)
    fidelity = float(abs(np.vdot(quantum, classical_normalised)) ** 2)
    decomposed = circuit.decompose(reps=4)

    encoding_study = []
    for divisor in (1, 2, 3, 4):
        candidate_time = suggested_evolution_time(eigenvalues, num_clock_qubits) / divisor
        candidate, candidate_success, _c, _t = solve_hhl(
            matrix, rhs, num_clock_qubits, candidate_time
        )
        encoding_study.append(
            {
                "evolution_time": candidate_time,
                "clock_values": clock_values(eigenvalues, candidate_time, num_clock_qubits),
                "exactly_representable": eigenvalues_representable(
                    eigenvalues, candidate_time, num_clock_qubits
                ),
                "fidelity": float(abs(np.vdot(candidate, classical_normalised)) ** 2),
                "success_probability": candidate_success,
            }
        )

    conditioning_study = []
    for candidate_gap in (0.5, 1.5, 3.5):
        candidate_matrix, candidate_rhs = well_conditioned_system(candidate_gap)
        candidate_eigenvalues = np.linalg.eigvalsh(candidate_matrix)
        candidate_exact = np.linalg.solve(candidate_matrix, candidate_rhs)
        candidate, candidate_success, _c, _t = solve_hhl(
            candidate_matrix, candidate_rhs, num_clock_qubits
        )
        conditioning_study.append(
            {
                "condition_number": float(
                    candidate_eigenvalues.max() / candidate_eigenvalues.min()
                ),
                "eigenvalues": candidate_eigenvalues.tolist(),
                "fidelity": float(
                    abs(np.vdot(candidate, candidate_exact / np.linalg.norm(candidate_exact))) ** 2
                ),
                "success_probability": candidate_success,
            }
        )

    dimension = len(rhs)
    return HHLResult(
        algorithm="hhl_intro",
        use_case="quantum_linear_systems_for_pdes",
        algorithm_type="quantum_linear_systems_algorithm",
        backend=QiskitBackendAdapter().describe(),
        matrix=matrix.tolist(),
        rhs=rhs.tolist(),
        eigenvalues=eigenvalues.tolist(),
        condition_number=float(eigenvalues.max() / eigenvalues.min()),
        num_clock_qubits=num_clock_qubits,
        evolution_time=evolution_time,
        clock_values=clock_values(eigenvalues, evolution_time, num_clock_qubits),
        eigenvalues_exactly_representable=eigenvalues_representable(
            eigenvalues, evolution_time, num_clock_qubits
        ),
        quantum_solution=np.real(quantum).tolist(),
        classical_solution=classical_normalised.tolist(),
        fidelity=fidelity,
        success_probability=success,
        circuit_depth=int(decomposed.depth()),
        two_qubit_gates=int(
            sum(v for k, v in decomposed.count_ops().items() if k in {"cx", "cz", "ecr"})
        ),
        encoding_study=encoding_study,
        conditioning_study=conditioning_study,
        caveats={
            "returns_a_state_not_a_vector": (
                "HHL produces |x>, whose amplitudes encode the solution; it does not "
                "produce x"
            ),
            "shots_to_read_full_solution": measurements_for_precision(dimension, precision),
            "readout_precision": precision,
            "readout_scaling": "O(N / precision^2) -- linear in N, which is the speedup",
            "postselection_cost": (
                f"only {success:.1%} of runs succeed; the rest are discarded, and this "
                "falls as 1/kappa^2"
            ),
            "state_preparation": (
                "|b> is assumed loadable in polylog time; for general b that is as hard "
                "as the original problem"
            ),
            "sparsity": "the speedup assumes A is sparse and exp(iAt) is efficiently implementable",
        },
        notes={
            "classical_baseline": "numpy.linalg.solve, exact and instant at this size",
            "verification": "fidelity 1.0 against the classical solution when eigenvalues "
            "are exactly representable",
            "reference": "Aaronson, 'Read the fine print' (2015)",
        },
    )


def run_hhl_intro_scaffold():
    """Backwards-compatible alias for :func:`run_hhl_intro_tutorial`."""
    return run_hhl_intro_tutorial()
