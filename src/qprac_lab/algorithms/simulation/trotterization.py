"""Trotterization: simulating time evolution with product formulas.

Simulating ``exp(-iHt)`` directly needs a matrix exponential of a ``2^n x 2^n``
matrix -- the thing a quantum computer is supposed to avoid. Product formulas
split it into pieces that *are* implementable:

    first order  (Lie-Trotter):  exp(-iHt) ~ [ prod_k exp(-i H_k t/r) ]^r
    second order (Suzuki):       symmetrised, forward then reverse per step

The approximation is only necessary because the pieces do not commute. If every
``H_k`` commuted the product would be exact at ``r = 1``, and there would be no
error to trade against depth.

That trade is the whole practical story, and it is the reason this module exists
alongside the noise benchmark: **more Trotter steps mean less algorithmic error
and more circuit depth**. On an ideal simulator error falls monotonically with
step count. On noisy hardware it does not -- past some point the noise from the
extra depth costs more than the Trotter error it removes.

.. warning::
   Evaluate the **decomposed** circuit. ``Operator(circuit)`` on an undecomposed
   ``PauliEvolutionGate`` returns the exact matrix exponential and ignores the
   synthesis entirely, reporting a Trotter error of ``1e-15`` no matter the step
   count. Measured here: 1.05e-15 undecomposed against 1.95 decomposed, for the
   same ``reps=1`` circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import expm

from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit

TROTTER_ORDERS = (1, 2)


@dataclass
class TrotterizationResult:
    """Trotter error against exact evolution, and how it scales."""

    algorithm: str
    use_case: str
    algorithm_type: str
    backend: dict
    hamiltonian: dict
    num_qubits: int
    evolution_time: float
    order: int
    steps: int
    trotter_error: float
    circuit_depth: int
    two_qubit_gates: int
    scaling: list[dict[str, Any]]
    fitted_exponents: dict[str, float]
    noise_tradeoff: list[dict[str, Any]] = field(default_factory=list)
    optimal_steps_under_noise: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def transverse_field_ising_hamiltonian(
    num_qubits: int,
    coupling: float = 1.0,
    field: float = 1.0,
    periodic: bool = False,
):
    """Transverse-field Ising model: ``H = -J sum Z_i Z_{i+1} - h sum X_i``.

    The standard testbed for Hamiltonian simulation. Its two parts deliberately
    do not commute -- ``ZZ`` is diagonal, ``X`` is not -- which is exactly what
    makes the Trotter error nonzero and therefore worth measuring.
    """
    require_qiskit("Building a TFIM Hamiltonian")
    from qiskit.quantum_info import SparsePauliOp

    if num_qubits < 2:
        raise ValueError(f"need at least 2 qubits for a coupling term, got {num_qubits}")

    terms: list[tuple[str, float]] = []
    pairs = list(zip(range(num_qubits), range(1, num_qubits), strict=False))
    if periodic and num_qubits > 2:
        pairs.append((num_qubits - 1, 0))
    for left, right in pairs:
        terms.append(
            ("".join("Z" if q in (left, right) else "I" for q in range(num_qubits)), -coupling)
        )
    for qubit in range(num_qubits):
        terms.append(("".join("X" if q == qubit else "I" for q in range(num_qubits)), -field))
    return SparsePauliOp.from_list(terms)


def trotter_circuit(hamiltonian, time: float, steps: int, order: int = 1):
    """Circuit approximating ``exp(-i H t)`` with ``steps`` product-formula steps.

    Returned already decomposed, because an undecomposed ``PauliEvolutionGate``
    evaluates as the exact exponential and would silently report zero error.
    """
    require_qiskit("Building a Trotter circuit")
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter, SuzukiTrotter

    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if order not in TROTTER_ORDERS:
        raise ValueError(f"order must be one of {TROTTER_ORDERS}, got {order}")

    synthesis = LieTrotter(reps=steps) if order == 1 else SuzukiTrotter(order=2, reps=steps)
    circuit = QuantumCircuit(hamiltonian.num_qubits)
    circuit.append(
        PauliEvolutionGate(hamiltonian, time=time, synthesis=synthesis),
        range(hamiltonian.num_qubits),
    )
    return circuit.decompose(reps=4)


def exact_evolution_operator(hamiltonian, time: float) -> np.ndarray:
    """Classical baseline: the exact ``exp(-i H t)`` by dense matrix exponential.

    Costs ``O(8^n)`` time and ``O(4^n)`` memory, which is precisely why it stops
    being the answer somewhere around 15 qubits on a workstation.
    """
    return expm(-1j * time * np.asarray(hamiltonian.to_matrix()))


def trotter_operator_error(hamiltonian, time: float, steps: int, order: int = 1) -> float:
    """Spectral-norm distance between the Trotter circuit and exact evolution.

    State-independent, unlike a fidelity on one particular input, so it measures
    the approximation rather than a lucky choice of initial state.
    """
    require_qiskit("Measuring Trotter error")
    from qiskit.quantum_info import Operator

    circuit = trotter_circuit(hamiltonian, time, steps, order)
    approximate = np.asarray(Operator(circuit).data)
    return float(np.linalg.norm(approximate - exact_evolution_operator(hamiltonian, time), 2))


def _fit_exponent(step_counts, errors) -> float:
    """Fit ``error ~ steps^(-p)`` and return ``p``, using the asymptotic tail."""
    steps = np.asarray(step_counts, dtype=float)
    values = np.asarray(errors, dtype=float)
    keep = values > 1e-12
    # The leading-order law only holds once the step is small; the first couple of
    # points sit outside that regime and would drag the fit.
    if keep.sum() > 3:
        keep &= steps >= steps[keep][1]
    if keep.sum() < 2:
        return float("nan")
    slope = np.polyfit(np.log(steps[keep]), np.log(values[keep]), 1)[0]
    return float(-slope)


def run_trotterization_tutorial(
    num_qubits: int = 4,
    evolution_time: float = 1.5,
    steps: int = 8,
    order: int = 2,
    coupling: float = 1.0,
    field: float = 1.0,
    step_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    backend: str = "statevector",
    seed: int = 42,
    include_noise_tradeoff: bool = True,
    noise: str = "moderate",
    shots: int = 4096,
) -> TrotterizationResult:
    """Measure Trotter error, its scaling law, and the depth/noise trade."""
    require_qiskit("The Trotterization tutorial")
    from qiskit.quantum_info import Operator

    hamiltonian = transverse_field_ising_hamiltonian(num_qubits, coupling, field)
    adapter = QiskitBackendAdapter(backend=backend, seed=seed)
    exact = exact_evolution_operator(hamiltonian, evolution_time)

    scaling: list[dict[str, Any]] = []
    for candidate_order in TROTTER_ORDERS:
        for count in step_counts:
            circuit = trotter_circuit(hamiltonian, evolution_time, count, candidate_order)
            error = float(np.linalg.norm(np.asarray(Operator(circuit).data) - exact, 2))
            scaling.append(
                {
                    "order": candidate_order,
                    "steps": count,
                    "error": error,
                    "depth": int(circuit.depth()),
                    "two_qubit_gates": int(
                        sum(v for k, v in circuit.count_ops().items() if k in {"cx", "cz", "ecr"})
                    ),
                }
            )

    fitted = {
        f"order_{candidate}": _fit_exponent(
            [row["steps"] for row in scaling if row["order"] == candidate],
            [row["error"] for row in scaling if row["order"] == candidate],
        )
        for candidate in TROTTER_ORDERS
    }

    chosen = trotter_circuit(hamiltonian, evolution_time, steps, order)
    tradeoff: list[dict[str, Any]] = []
    optimum: dict[str, Any] = {}
    if include_noise_tradeoff:
        tradeoff = noise_tradeoff(
            hamiltonian,
            evolution_time,
            step_counts=step_counts,
            order=order,
            noise=noise,
            shots=shots,
            seed=seed,
        )
        if tradeoff:
            best = min(tradeoff, key=lambda row: row["noisy_error"])
            optimum = {
                "steps": best["steps"],
                "noisy_error": best["noisy_error"],
                "ideal_error_at_that_step": best["ideal_error"],
                "noise": noise,
                "interpretation": (
                    "on an ideal simulator more steps is always better; under noise "
                    "the extra depth eventually costs more than the Trotter error it "
                    "removes"
                ),
            }

    return TrotterizationResult(
        algorithm="trotterization_time_evolution",
        use_case="quantum_many_body_dynamics",
        algorithm_type="product_formula_hamiltonian_simulation",
        backend=adapter.describe(),
        hamiltonian={
            "model": "transverse_field_ising",
            "num_qubits": num_qubits,
            "coupling_j": coupling,
            "field_h": field,
            "num_terms": len(hamiltonian),
            "non_commuting": True,
        },
        num_qubits=num_qubits,
        evolution_time=evolution_time,
        order=order,
        steps=steps,
        trotter_error=trotter_operator_error(hamiltonian, evolution_time, steps, order),
        circuit_depth=int(chosen.depth()),
        two_qubit_gates=int(
            sum(v for k, v in chosen.count_ops().items() if k in {"cx", "cz", "ecr"})
        ),
        scaling=scaling,
        fitted_exponents=fitted,
        noise_tradeoff=tradeoff,
        optimal_steps_under_noise=optimum,
        notes={
            "error_metric": "spectral norm ||U_trotter - U_exact||, state-independent",
            "expected_scaling": "first order ~ 1/steps, second order ~ 1/steps^2",
            "classical_baseline": "dense matrix exponential, O(8^n)",
            "decomposition_warning": (
                "Operator() on an undecomposed PauliEvolutionGate returns the exact "
                "exponential and reports zero Trotter error"
            ),
        },
    )


def noise_tradeoff(
    hamiltonian,
    evolution_time: float,
    step_counts: tuple[int, ...],
    order: int = 2,
    noise: str = "moderate",
    shots: int = 4096,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Trotter error versus total error under device noise, across step counts.

    Measures an observable -- total magnetisation ``sum Z_i`` from an all-up
    initial state -- rather than the operator norm, because that is what an
    experiment can actually read out.
    """
    require_qiskit("Measuring the Trotter noise tradeoff")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    num_qubits = hamiltonian.num_qubits
    observable = SparsePauliOp.from_list(
        [
            ("".join("Z" if q == i else "I" for q in range(num_qubits)), 1.0)
            for i in range(num_qubits)
        ]
    )

    # All spins up. |+...+> would look like a natural choice and is a trap: it is
    # an eigenstate of the model's spin-flip symmetry, so the magnetisation stays
    # pinned at zero and every step count scores an identical, meaningless error.
    preparation = QuantumCircuit(num_qubits)
    initial = Statevector(preparation)

    exact_state = exact_evolution_operator(hamiltonian, evolution_time) @ initial.data
    exact_value = float(
        np.real(exact_state.conj() @ np.asarray(observable.to_matrix()) @ exact_state)
    )

    ideal_adapter = QiskitBackendAdapter(backend="statevector", seed=seed)
    noisy_adapter = QiskitBackendAdapter(backend="aer", shots=shots, seed=seed, noise=noise)
    ideal_estimator = ideal_adapter.estimator()
    noisy_estimator = noisy_adapter.estimator()

    rows = []
    for count in step_counts:
        evolution = preparation.compose(
            trotter_circuit(hamiltonian, evolution_time, count, order)
        )
        ideal = float(
            ideal_estimator.run([(evolution, observable)]).result()[0].data.evs
        )
        noisy = float(
            noisy_estimator.run(
                [(noisy_adapter.prepare(evolution), observable)]
            ).result()[0].data.evs
        )
        rows.append(
            {
                "steps": count,
                "depth": int(evolution.decompose(reps=4).depth()),
                "exact_observable": exact_value,
                "ideal_observable": ideal,
                "noisy_observable": noisy,
                "ideal_error": abs(ideal - exact_value),
                "noisy_error": abs(noisy - exact_value),
            }
        )
    return rows
