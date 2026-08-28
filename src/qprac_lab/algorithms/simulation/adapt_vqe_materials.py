"""ADAPT-VQE: build the ansatz instead of guessing it.

Fixed ansaetze force a bad trade. A chemically-motivated one is compact but has to
be derived per problem; a hardware-efficient one is generic but spends parameters
freely -- ``efficient_su2`` needs 12 of them for a *two-qubit* H2 problem that one
parameter solves exactly. Both matter more than they look, because the noise
benchmark showed VQE failing on a depth-10 circuit: every superfluous parameter is
circuit depth, and depth is what hardware cannot afford.

ADAPT-VQE (Grimsley et al., 2019) grows the ansatz one operator at a time:

1. Start from the Hartree-Fock reference.
2. For every operator ``P`` in a pool, compute the energy gradient at zero angle.
3. Append the operator with the largest ``|gradient|``, giving it a new parameter.
4. Re-optimise **all** parameters.
5. Stop when the largest remaining gradient falls below a threshold.

The result is the shortest ansatz that reaches the answer, discovered rather than
assumed. The gradient at zero angle has a closed form,

    dE/dtheta = (1/2) <psi| i[P, H] |psi>

so ranking the pool costs one expectation value per operator -- no optimisation
per candidate. The test suite cross-checks this against the parameter-shift rule,
since a sign error here would still converge, just to the wrong ansatz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import minimize

from qprac_lab.algorithms.simulation.hamiltonian_utils import (
    MolecularHamiltonian,
    build_h2_hamiltonian,
)
from qprac_lab.algorithms.simulation.vqe_molecular_energy import (
    CHEMICAL_ACCURACY_HARTREE,
    build_ansatz,
)
from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit

#: Default stopping threshold on the largest pool gradient, in hartree.
DEFAULT_GRADIENT_TOLERANCE = 1e-4


@dataclass
class AdaptVQEResult:
    """Outcome of an ADAPT-VQE run, with the fixed-ansatz comparison alongside."""

    algorithm: str
    use_case: str
    algorithm_type: str
    backend: dict
    hamiltonian: dict
    pool_size: int
    max_operator_weight: int
    selected_operators: list[str]
    num_parameters: int
    circuit_depth: int
    objective_evaluations: int
    iterations: list[dict[str, Any]]
    energy: float
    exact_baseline_energy: float
    hartree_fock_baseline_energy: float
    absolute_error: float
    chemical_accuracy_reached: bool
    converged: bool
    final_max_gradient: float
    convergence_history: list[float]
    comparison: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def qubit_excitation_pool(num_qubits: int, max_weight: int = 2) -> list[str]:
    """Pauli-string operator pool for qubit-ADAPT.

    Restricted to strings with an **odd number of Y factors**. Those are exactly
    the generators producing real-valued rotations, which is what a real
    Hamiltonian's ground state needs; even-Y generators can only add phases.
    Capping the weight keeps the pool to hardware-implementable 1- and 2-local
    operators instead of the full ``4^n`` set.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be positive, got {num_qubits}")
    if max_weight < 1:
        raise ValueError(f"max_weight must be positive, got {max_weight}")

    pool = []
    for factors in product("IXYZ", repeat=num_qubits):
        label = "".join(factors)
        weight = num_qubits - label.count("I")
        if weight == 0 or weight > max_weight:
            continue
        if label.count("Y") % 2 == 1:
            pool.append(label)
    return pool


def operator_gradients(estimator, circuit, parameters, hamiltonian_op, pool: list[str]):
    """Energy gradient of every pool operator at zero angle.

    Uses ``dE/dtheta = (1/2) <i[P, H]>``, one expectation value per operator, so
    ranking the whole pool costs no optimisation at all.
    """
    require_qiskit("Computing ADAPT gradients")
    from qiskit.quantum_info import SparsePauliOp

    observables = []
    for label in pool:
        pauli = SparsePauliOp.from_list([(label, 1.0)])
        commutator = (pauli @ hamiltonian_op - hamiltonian_op @ pauli).simplify()
        observables.append((0.5 * (1j * commutator)).simplify())

    values = list(parameters)
    pubs = [(circuit, observable, values) for observable in observables]
    results = estimator.run(pubs).result()
    return np.array([float(np.real(result.data.evs)) for result in results])


def build_adapt_circuit(num_qubits: int, hf_bitstring: str, operators: list[str]):
    """Hartree-Fock reference followed by one Pauli rotation per selected operator."""
    require_qiskit("Building an ADAPT ansatz")
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    circuit = QuantumCircuit(num_qubits)
    for qubit, bit in enumerate(reversed(hf_bitstring)):
        if bit == "1":
            circuit.x(qubit)

    if operators:
        thetas = ParameterVector("theta", len(operators))
        for theta, label in zip(thetas, operators, strict=True):
            pauli = SparsePauliOp.from_list([(label, 1.0)])
            # PauliEvolutionGate(P, t) is exp(-i t P); we want exp(-i theta P / 2).
            circuit.append(PauliEvolutionGate(pauli, time=theta / 2), range(num_qubits))
    # Decomposed once here: PauliEvolutionGate re-synthesises on every estimator
    # call otherwise, which is a ~400x slowdown inside an optimisation loop.
    return circuit.decompose(reps=3)


def run_adapt_vqe(
    hamiltonian: MolecularHamiltonian,
    max_iterations: int = 8,
    gradient_tolerance: float = DEFAULT_GRADIENT_TOLERANCE,
    max_operator_weight: int = 2,
    optimizer: str = "COBYLA",
    maxiter: int = 300,
    backend: str = "statevector",
    shots: int | None = None,
    seed: int = 42,
    noise: str | None = None,
):
    """Grow an ansatz operator by operator until the pool gradients vanish.

    Returns ``(operators, parameters, energy, iterations, max_gradient)``.
    """
    require_qiskit("Running ADAPT-VQE")
    adapter = QiskitBackendAdapter(backend=backend, shots=shots, seed=seed, noise=noise)
    estimator = adapter.estimator()
    observable = hamiltonian.qubit_operator
    pool = qubit_excitation_pool(hamiltonian.num_qubits, max_weight=max_operator_weight)

    operators: list[str] = []
    parameters: list[float] = []
    iterations: list[dict[str, Any]] = []
    max_gradient = float("inf")

    for iteration in range(1, max_iterations + 1):
        circuit = adapter.prepare(build_adapt_circuit(
            hamiltonian.num_qubits, hamiltonian.hartree_fock_bitstring, operators
        ))
        gradients = operator_gradients(estimator, circuit, parameters, observable, pool)
        best = int(np.argmax(np.abs(gradients)))
        max_gradient = float(abs(gradients[best]))

        if max_gradient < gradient_tolerance:
            break

        operators.append(pool[best])
        parameters.append(0.0)
        grown = adapter.prepare(build_adapt_circuit(
            hamiltonian.num_qubits, hamiltonian.hartree_fock_bitstring, operators
        ))

        evaluations = 0

        def objective(values, circuit=grown):
            nonlocal evaluations
            evaluations += 1
            job = estimator.run([(circuit, observable, list(values))])
            return float(job.result()[0].data.evs)

        result = minimize(
            objective,
            x0=np.array(parameters, dtype=float),
            method=optimizer,
            options={"maxiter": maxiter},
        )
        parameters = [float(v) for v in np.atleast_1d(result.x)]
        iterations.append(
            {
                "iteration": iteration,
                "selected_operator": pool[best],
                "gradient": float(gradients[best]),
                "num_parameters": len(parameters),
                "objective_evaluations": evaluations,
                "electronic_energy": float(result.fun),
                "total_energy": hamiltonian.total_energy(float(result.fun)),
            }
        )

    energy = (
        iterations[-1]["electronic_energy"]
        if iterations
        else hamiltonian.hartree_fock_electronic_energy()
    )
    return operators, parameters, energy, iterations, max_gradient


def run_adapt_vqe_materials(
    bond_length_angstrom: float = 0.735,
    max_iterations: int = 8,
    gradient_tolerance: float = DEFAULT_GRADIENT_TOLERANCE,
    max_operator_weight: int = 2,
    backend: str = "statevector",
    shots: int | None = None,
    seed: int = 42,
    noise: str | None = None,
    compare_fixed_ansatz: bool = True,
) -> AdaptVQEResult:
    """Run ADAPT-VQE on H2 and compare it against a fixed hardware-efficient ansatz."""
    require_qiskit("The ADAPT-VQE tutorial")
    hamiltonian = build_h2_hamiltonian(bond_length_angstrom)
    adapter = QiskitBackendAdapter(backend=backend, shots=shots, seed=seed, noise=noise)

    operators, parameters, electronic, iterations, max_gradient = run_adapt_vqe(
        hamiltonian,
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
        max_operator_weight=max_operator_weight,
        backend=backend,
        shots=shots,
        seed=seed,
        noise=noise,
    )

    total = hamiltonian.total_energy(electronic)
    exact = hamiltonian.exact_total_energy()
    hartree_fock = hamiltonian.hartree_fock_total_energy()
    circuit = build_adapt_circuit(
        hamiltonian.num_qubits, hamiltonian.hartree_fock_bitstring, operators
    )

    comparison: dict[str, Any] = {}
    if compare_fixed_ansatz:
        fixed = build_ansatz(hamiltonian.num_qubits, kind="efficient_su2")
        comparison = {
            "fixed_ansatz": "efficient_su2",
            "fixed_num_parameters": int(fixed.num_parameters),
            "fixed_circuit_depth": int(fixed.decompose(reps=3).depth()),
            "adapt_num_parameters": len(parameters),
            "adapt_circuit_depth": int(circuit.depth()),
            "parameter_reduction": (
                int(fixed.num_parameters) / len(parameters) if parameters else float("inf")
            ),
            "adapt_objective_evaluations": sum(
                step["objective_evaluations"] for step in iterations
            ),
        }

    return AdaptVQEResult(
        algorithm="adapt_vqe_materials",
        use_case="materials_discovery_refinement",
        algorithm_type="adaptive_variational_eigensolver",
        backend=adapter.describe(),
        hamiltonian=hamiltonian.describe(),
        pool_size=len(qubit_excitation_pool(hamiltonian.num_qubits, max_operator_weight)),
        max_operator_weight=max_operator_weight,
        selected_operators=operators,
        num_parameters=len(parameters),
        circuit_depth=int(circuit.depth()),
        objective_evaluations=sum(step["objective_evaluations"] for step in iterations),
        iterations=iterations,
        energy=total,
        exact_baseline_energy=exact,
        hartree_fock_baseline_energy=hartree_fock,
        absolute_error=abs(total - exact),
        chemical_accuracy_reached=abs(total - exact) < CHEMICAL_ACCURACY_HARTREE,
        converged=max_gradient < gradient_tolerance,
        final_max_gradient=max_gradient,
        convergence_history=[step["total_energy"] for step in iterations],
        comparison=comparison,
        notes={
            "gradient_formula": "dE/dtheta = (1/2) <i[P, H]> at theta = 0",
            "pool": "Pauli strings with an odd number of Y factors, weight <= max_weight",
            "stopping_rule": f"largest pool gradient < {gradient_tolerance}",
            "why_it_matters": (
                "every parameter is circuit depth, and the noise benchmark shows depth "
                "is what current hardware cannot afford"
            ),
        },
    )


def run_adapt_vqe_materials_scaffold():
    """Backwards-compatible alias for :func:`run_adapt_vqe_materials`."""
    return run_adapt_vqe_materials()
