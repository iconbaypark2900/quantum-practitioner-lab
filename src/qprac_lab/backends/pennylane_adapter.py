"""PennyLane backend adapter, for independent cross-checking.

This exists for a different reason than the Qiskit adapter. Qiskit runs the
algorithms; PennyLane **re-runs them through an unrelated stack** so the answers
can be checked against something other than themselves.

That matters here more than usual. Several results in this repository came within
one silent bug of being wrong -- an unseeded Aer estimator, an undecomposed
evolution gate that reported the exact answer, a bitstring decoded in the wrong
endianness. Every one of those failed *quietly*, producing plausible numbers. A
second implementation that agrees to 1e-9 is the cheapest evidence that a result
is a property of the physics rather than of one library's conventions.

It replaces the planned IBM Runtime adapter, which needed credentials, could not
be tested in CI, and would have exercised the same Qiskit stack anyway.

**Qubit ordering is the trap.** Qiskit Pauli labels are little-endian -- the
leftmost character is the *highest* qubit index -- while PennyLane addresses wires
explicitly. :func:`to_pennylane_hamiltonian` maps by index rather than position,
and a test pins an asymmetric operator, which is the only kind that catches a flip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

INSTALL_HINT = 'install the cross-check backend with: pip install -e ".[pennylane]"'


class PennyLaneNotInstalledError(RuntimeError):
    """Raised when a PennyLane code path is reached without the extra installed."""

    def __init__(self, what: str = "This code path") -> None:
        super().__init__(f"{what} requires PennyLane; {INSTALL_HINT}")


def pennylane_available() -> bool:
    try:
        import pennylane  # noqa: F401
    except ImportError:
        return False
    return True


def require_pennylane(what: str = "This code path") -> None:
    if not pennylane_available():
        raise PennyLaneNotInstalledError(what)


def pennylane_version() -> str | None:
    try:
        import pennylane
    except ImportError:
        return None
    return getattr(pennylane, "__version__", "unknown")


def to_pennylane_hamiltonian(operator):
    """Convert a Qiskit ``SparsePauliOp`` into a PennyLane Hamiltonian.

    Mapped by explicit qubit index, not string position: Qiskit's label ``"XY"``
    means ``X`` on qubit 1 and ``Y`` on qubit 0, so character ``i`` of an
    ``n``-qubit label addresses wire ``n - 1 - i``. Getting this backwards
    produces a mirror-image Hamiltonian with an identical spectrum, which is why
    the test uses an asymmetric operator.
    """
    require_pennylane("Converting a Hamiltonian")
    import pennylane as qml

    paulis = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    num_qubits = operator.num_qubits
    coefficients: list[float] = []
    observables = []

    for label, coefficient in zip(operator.paulis.to_labels(), operator.coeffs, strict=True):
        factors = [
            paulis[character](num_qubits - 1 - position)
            for position, character in enumerate(label)
            if character != "I"
        ]
        if not factors:
            term = qml.Identity(0)
        elif len(factors) == 1:
            term = factors[0]
        else:
            term = qml.prod(*factors)
        coefficients.append(float(np.real(coefficient)))
        observables.append(term)

    return qml.Hamiltonian(coefficients, observables)


@dataclass
class PennyLaneBackendAdapter:
    """Resolve a PennyLane device, mirroring :class:`QiskitBackendAdapter`."""

    device: str = "default.qubit"
    shots: int | None = None
    seed: int | None = 42

    name = "pennylane_adapter"

    def __post_init__(self) -> None:
        if self.shots is not None and self.shots <= 0:
            raise ValueError(f"shots must be positive or None, got {self.shots}")

    def device_handle(self, wires: int):
        """Build the PennyLane device for a given qubit count."""
        require_pennylane("Building a PennyLane device")
        import pennylane as qml

        if self.shots is None:
            return qml.device(self.device, wires=wires)
        return qml.device(self.device, wires=wires, shots=self.shots, seed=self.seed)

    def expectation(self, operator, prepare) -> float:
        """Expectation of a Qiskit operator under a PennyLane state preparation.

        ``prepare`` is a callable applying gates to the wires; it takes no
        arguments and returns nothing, matching PennyLane's QNode convention.
        """
        require_pennylane("Computing an expectation value")
        import pennylane as qml

        hamiltonian = to_pennylane_hamiltonian(operator)
        device = self.device_handle(operator.num_qubits)

        @qml.qnode(device)
        def circuit():
            prepare()
            return qml.expval(hamiltonian)

        return float(circuit())

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.device,
            "framework": "pennylane",
            "role": "independent cross-check of the Qiskit results",
            "shots": self.shots,
            "exact_expectation_values": self.shots is None,
            "seed": self.seed,
            "pennylane_installed": pennylane_available(),
            "version": pennylane_version(),
        }


def pennylane_h2_vqe(
    operator,
    hf_bitstring: str = "01",
    maxiter: int = 300,
    shots: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Independent VQE for the 2-qubit H2 problem, written natively in PennyLane.

    Deliberately *not* a translation of the Qiskit circuit -- it is the same
    algorithm expressed in another framework's idiom. A translation would carry
    over any convention error it was meant to detect.
    """
    require_pennylane("Running the PennyLane VQE")
    import pennylane as qml
    from scipy.optimize import minimize

    if operator.num_qubits != 2:
        raise ValueError(f"this ansatz is specific to 2 qubits, got {operator.num_qubits}")

    hamiltonian = to_pennylane_hamiltonian(operator)
    adapter = PennyLaneBackendAdapter(shots=shots, seed=seed)
    device = adapter.device_handle(2)
    excited = [wire for wire, bit in enumerate(reversed(hf_bitstring)) if bit == "1"]

    @qml.qnode(device)
    def energy(theta):
        for wire in excited:
            qml.PauliX(wire)
        # exp(-i theta/2 X0 Y1), the same single excitation the Qiskit ansatz uses.
        qml.RX(np.pi / 2, wires=0)
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(theta, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(-np.pi / 2, wires=0)
        qml.Hadamard(wires=1)
        return qml.expval(hamiltonian)

    history: list[float] = []

    def objective(values):
        value = float(energy(values[0]))
        history.append(value)
        return value

    result = minimize(objective, x0=[0.0], method="COBYLA", options={"maxiter": maxiter})
    return {
        "electronic_energy": float(result.fun),
        "optimal_theta": float(np.atleast_1d(result.x)[0]),
        "function_evaluations": len(history),
        "convergence_history": history,
        "framework": "pennylane",
        "version": pennylane_version(),
    }


def cross_check_vqe(bond_length_angstrom: float = 0.735, tolerance: float = 1e-6) -> dict[str, Any]:
    """Run H2 VQE through Qiskit and PennyLane and compare.

    Agreement between two unrelated stacks is what distinguishes a physical result
    from a convention that happens to be self-consistent.
    """
    require_pennylane("Cross-checking VQE")
    from qprac_lab.algorithms.simulation.hamiltonian_utils import build_h2_hamiltonian
    from qprac_lab.algorithms.simulation.vqe_molecular_energy import (
        run_vqe_molecular_energy_tutorial,
    )

    hamiltonian = build_h2_hamiltonian(bond_length_angstrom)
    qiskit_result = run_vqe_molecular_energy_tutorial(
        bond_length_angstrom=bond_length_angstrom, include_dissociation_curve=False
    )
    pennylane_result = pennylane_h2_vqe(
        hamiltonian.qubit_operator, hf_bitstring=hamiltonian.hartree_fock_bitstring
    )
    pennylane_total = hamiltonian.total_energy(pennylane_result["electronic_energy"])
    difference = abs(pennylane_total - qiskit_result.vqe_energy)

    return {
        "bond_length_angstrom": bond_length_angstrom,
        "qiskit_energy": qiskit_result.vqe_energy,
        "pennylane_energy": pennylane_total,
        "exact_energy": hamiltonian.exact_total_energy(),
        "absolute_difference": float(difference),
        "tolerance": tolerance,
        "frameworks_agree": bool(difference < tolerance),
        "qiskit_evaluations": qiskit_result.function_evaluations,
        "pennylane_evaluations": pennylane_result["function_evaluations"],
        "interpretation": (
            "two unrelated simulation stacks reaching the same energy is evidence "
            "the result is physics rather than one library's conventions"
        ),
    }


def cross_check_ising_mapping(num_assets: int = 6, budget: int = 3) -> dict[str, Any]:
    """Verify the QUBO -> Ising mapping through a second framework.

    The endianness check that matters most. A bitstring decoded in the wrong
    order still produces a valid portfolio and a plausible objective value, so
    the failure is invisible from inside one library's conventions. This
    evaluates every assignment's energy in both stacks and compares.
    """
    require_pennylane("Cross-checking the Ising mapping")
    import pennylane as qml
    from qiskit.quantum_info import Statevector

    from qprac_lab.algorithms.optimization.qubo_builder import portfolio_qubo
    from qprac_lab.data.synthetic import make_small_portfolio_dataset

    returns, covariance = make_small_portfolio_dataset(n_assets=num_assets)
    qubo = portfolio_qubo(returns, covariance, budget=budget)
    operator, offset = qubo.to_ising()
    hamiltonian = to_pennylane_hamiltonian(operator)
    device = qml.device("default.qubit", wires=num_assets)

    @qml.qnode(device)
    def energy(selection):
        for wire, bit in enumerate(selection):
            if bit:
                qml.PauliX(wire)
        return qml.expval(hamiltonian)

    worst = 0.0
    for index in range(2**num_assets):
        selection = [(index >> wire) & 1 for wire in range(num_assets)]
        label = "".join(str(bit) for bit in reversed(selection))
        qiskit_energy = (
            float(Statevector.from_label(label).expectation_value(operator).real) + offset
        )
        pennylane_energy = float(energy(selection)) + offset
        worst = max(
            worst,
            abs(qiskit_energy - qubo.objective(selection)),
            abs(pennylane_energy - qubo.objective(selection)),
        )

    return {
        "num_assets": num_assets,
        "budget": budget,
        "assignments_checked": 2**num_assets,
        "max_absolute_difference": float(worst),
        "frameworks_agree": bool(worst < 1e-9),
        "interpretation": (
            "a mirror-image decoding produces a valid-looking portfolio and a "
            "plausible objective, so only a second framework catches it"
        ),
    }
