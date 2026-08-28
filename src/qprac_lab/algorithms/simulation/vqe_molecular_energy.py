"""Tutorial 1: VQE for molecular ground-state energy.

A real variational quantum eigensolver: a parameterised circuit is run through a
Qiskit V2 ``Estimator`` to get ``<psi(theta)|H|psi(theta)>``, and a classical
optimiser drives ``theta`` downhill. That expectation value is the only thing the
quantum computer provides; the optimisation loop is classical, which is what
makes VQE a *hybrid* algorithm.

Two classical baselines, both mandatory under the project's rules:

exact diagonalisation
    The variational principle guarantees VQE cannot go below this. It is the
    correctness check -- if VQE lands under it, something is wrong.
hartree-fock
    The mean-field reference. VQE is only interesting to the extent it beats HF;
    the gap between them is the correlation energy, and recovering it is the
    entire point of the exercise.

Success is measured against *chemical accuracy* (1.6 mHa, i.e. 1 kcal/mol), the
threshold at which computed energies start being useful for chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from qprac_lab.algorithms.simulation.hamiltonian_utils import (
    MolecularHamiltonian,
    build_h2_hamiltonian,
    nature_available,
)
from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit

#: 1 kcal/mol in hartree -- the accuracy chemistry actually cares about.
CHEMICAL_ACCURACY_HARTREE = 1.6e-3

ANSATZ_CHOICES = ("two_qubit_uccsd", "efficient_su2", "real_amplitudes")


@dataclass
class VQEMolecularEnergyResult:
    """Outcome of one VQE run, with both classical baselines alongside."""

    algorithm: str
    use_case: str
    algorithm_type: str
    backend: dict
    hamiltonian: dict
    ansatz: str
    num_parameters: int
    optimizer: str
    function_evaluations: int
    optimizer_converged: bool
    optimal_parameters: list[float]
    vqe_electronic_energy: float
    vqe_energy: float
    exact_baseline_energy: float
    hartree_fock_baseline_energy: float
    correlation_energy_available: float
    correlation_energy_recovered: float
    correlation_recovery_fraction: float
    absolute_error: float
    chemical_accuracy_reached: bool
    beats_hartree_fock: bool
    convergence_history: list[float]
    # Retained so older callers keep working; for the one-parameter ansatz
    # ``best_theta`` is the single variational angle.
    best_theta: float
    best_energy: float
    dissociation_curve: list[dict[str, float]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


def build_ansatz(
    num_qubits: int,
    kind: str = "two_qubit_uccsd",
    reps: int = 2,
    hf_bitstring: str = "01",
):
    """Build the parameterised trial state.

    ``two_qubit_uccsd``
        The one-parameter unitary-coupled-cluster ansatz for parity-mapped H2.
        A single ``Ry``-style excitation on top of the Hartree-Fock determinant
        spans the whole relevant subspace, so it reaches the exact ground state.
    ``efficient_su2`` / ``real_amplitudes``
        Generic hardware-efficient ansaetze. They make no chemistry assumptions,
        need many more parameters, and are the realistic choice for molecules
        where a compact chemically-motivated ansatz is unavailable.
    """
    require_qiskit("Building a VQE ansatz")
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import EfficientSU2, RealAmplitudes

    if kind == "two_qubit_uccsd":
        if num_qubits != 2:
            raise ValueError(
                "the two_qubit_uccsd ansatz is specific to the 2-qubit parity-mapped "
                f"problem, got {num_qubits} qubits; use efficient_su2 instead"
            )
        theta = Parameter("theta")
        circuit = QuantumCircuit(2)
        _prepare_basis_state(circuit, hf_bitstring)
        # Exp(-i theta/2 * X0 Y1): a single-excitation rotation out of the HF state.
        circuit.rx(np.pi / 2, 0)
        circuit.h(1)
        circuit.cx(0, 1)
        circuit.rz(theta, 1)
        circuit.cx(0, 1)
        circuit.rx(-np.pi / 2, 0)
        circuit.h(1)
        return circuit

    if kind == "efficient_su2":
        return EfficientSU2(num_qubits, reps=reps, entanglement="linear")
    if kind == "real_amplitudes":
        return RealAmplitudes(num_qubits, reps=reps, entanglement="linear")

    raise ValueError(f"Unknown ansatz {kind!r}; expected one of {ANSATZ_CHOICES}")


def _prepare_basis_state(circuit, bitstring: str) -> None:
    """Apply X gates so the circuit starts in ``bitstring`` (Qiskit little-endian)."""
    for qubit, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            circuit.x(qubit)


def run_vqe(
    hamiltonian: MolecularHamiltonian,
    ansatz_kind: str = "two_qubit_uccsd",
    optimizer: str = "COBYLA",
    maxiter: int = 300,
    backend: str = "statevector",
    shots: int | None = None,
    seed: int = 42,
    initial_point: np.ndarray | None = None,
):
    """Minimise the Hamiltonian expectation value over the ansatz parameters.

    Returns ``(scipy_result, history, ansatz)`` where ``history`` is the
    electronic energy at every objective evaluation -- the raw material for the
    convergence plot.
    """
    require_qiskit("Running VQE")
    adapter = QiskitBackendAdapter(backend=backend, shots=shots, seed=seed)
    estimator = adapter.estimator()
    ansatz = build_ansatz(
        hamiltonian.num_qubits,
        kind=ansatz_kind,
        hf_bitstring=hamiltonian.hartree_fock_bitstring,
    )
    observable = hamiltonian.qubit_operator
    history: list[float] = []

    def objective(parameters: np.ndarray) -> float:
        job = estimator.run([(ansatz, observable, list(parameters))])
        energy = float(job.result()[0].data.evs)
        history.append(energy)
        return energy

    num_parameters = ansatz.num_parameters
    if initial_point is None:
        rng = np.random.default_rng(seed)
        # Start at zero for the chemically-motivated ansatz -- that is exactly the
        # Hartree-Fock state, so VQE provably starts at the HF energy and can only
        # improve. Hardware-efficient ansaetze need a random start to avoid the
        # all-zero saddle point where every gradient vanishes.
        if ansatz_kind == "two_qubit_uccsd":
            initial_point = np.zeros(num_parameters)
        else:
            initial_point = rng.uniform(-0.1, 0.1, size=num_parameters)

    result = minimize(
        objective,
        x0=np.asarray(initial_point, dtype=float),
        method=optimizer,
        options={"maxiter": maxiter},
    )
    return result, history, ansatz


def scan_dissociation_curve(
    bond_lengths: list[float] | None = None,
    ansatz_kind: str = "two_qubit_uccsd",
    backend: str = "statevector",
) -> list[dict[str, float]]:
    """VQE vs exact vs Hartree-Fock across a range of bond lengths.

    This is where the tutorial's use case actually shows up. Restricted HF gets
    steadily worse as the bond stretches -- the textbook static-correlation
    failure -- while VQE tracks the exact curve throughout.

    Requires the chemistry stack; returns an empty list without it.
    """
    if not nature_available():
        return []
    bond_lengths = bond_lengths or [0.5, 0.735, 1.0, 1.5, 2.0, 2.5]
    curve: list[dict[str, float]] = []
    for bond_length in bond_lengths:
        hamiltonian = build_h2_hamiltonian(bond_length)
        result, _, _ = run_vqe(hamiltonian, ansatz_kind=ansatz_kind, backend=backend)
        curve.append(
            {
                "bond_length_angstrom": float(bond_length),
                "vqe_energy": hamiltonian.total_energy(float(result.fun)),
                "exact_energy": hamiltonian.exact_total_energy(),
                "hartree_fock_energy": hamiltonian.hartree_fock_total_energy(),
            }
        )
    return curve


def run_vqe_molecular_energy_tutorial(
    bond_length_angstrom: float = 0.735,
    ansatz_kind: str = "two_qubit_uccsd",
    backend: str = "statevector",
    shots: int | None = None,
    optimizer: str = "COBYLA",
    maxiter: int = 300,
    seed: int = 42,
    include_dissociation_curve: bool = True,
) -> VQEMolecularEnergyResult:
    """Run tutorial 1 end to end: VQE for the H2 ground state."""
    require_qiskit("The VQE tutorial")
    hamiltonian = build_h2_hamiltonian(bond_length_angstrom)
    adapter = QiskitBackendAdapter(backend=backend, shots=shots, seed=seed)

    result, history, ansatz = run_vqe(
        hamiltonian,
        ansatz_kind=ansatz_kind,
        optimizer=optimizer,
        maxiter=maxiter,
        backend=backend,
        shots=shots,
        seed=seed,
    )

    electronic = float(result.fun)
    total = hamiltonian.total_energy(electronic)
    exact = hamiltonian.exact_total_energy()
    hartree_fock = hamiltonian.hartree_fock_total_energy()

    correlation_available = hartree_fock - exact
    correlation_recovered = hartree_fock - total
    recovery_fraction = (
        correlation_recovered / correlation_available if correlation_available else 0.0
    )
    optimal_parameters = [float(v) for v in np.atleast_1d(result.x)]

    curve: list[dict[str, float]] = []
    if include_dissociation_curve and ansatz_kind == "two_qubit_uccsd":
        curve = scan_dissociation_curve(ansatz_kind=ansatz_kind, backend=backend)

    return VQEMolecularEnergyResult(
        algorithm="vqe_molecular_energy",
        use_case="materials_discovery_refinement",
        algorithm_type="hybrid_variational_eigensolver",
        backend=adapter.describe(),
        hamiltonian=hamiltonian.describe(),
        ansatz=ansatz_kind,
        num_parameters=int(ansatz.num_parameters),
        optimizer=optimizer,
        function_evaluations=len(history),
        optimizer_converged=bool(getattr(result, "success", False)),
        optimal_parameters=optimal_parameters,
        vqe_electronic_energy=electronic,
        vqe_energy=total,
        exact_baseline_energy=exact,
        hartree_fock_baseline_energy=hartree_fock,
        correlation_energy_available=correlation_available,
        correlation_energy_recovered=correlation_recovered,
        correlation_recovery_fraction=float(recovery_fraction),
        absolute_error=abs(total - exact),
        chemical_accuracy_reached=abs(total - exact) < CHEMICAL_ACCURACY_HARTREE,
        beats_hartree_fock=total < hartree_fock,
        convergence_history=[hamiltonian.total_energy(e) for e in history],
        best_theta=optimal_parameters[0] if optimal_parameters else 0.0,
        best_energy=total,
        dissociation_curve=curve,
        notes={
            "energy_units": "hartree",
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "convergence_history_units": "total energy (electronic + nuclear repulsion)",
            "variational_principle": "VQE energy is an upper bound on the exact energy",
        },
    )
