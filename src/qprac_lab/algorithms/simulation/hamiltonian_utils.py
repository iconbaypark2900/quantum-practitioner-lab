"""Molecular qubit Hamiltonians for the simulation tutorials.

Two ways to get an H2 Hamiltonian, both returning the same
:class:`MolecularHamiltonian` container:

``"nature"``
    Run PySCF through Qiskit Nature and parity-map the result, giving a real
    Hamiltonian at an arbitrary bond length. Needs the ``nature`` extra.

``"builtin"``
    A single hard-coded bond length (0.735 A) so the tutorial still runs with no
    quantum-chemistry stack installed. The coefficients are the canonical H2 /
    STO-3G parity-tapered set and were checked against the Nature path: both
    diagonalise to an electronic energy of -1.857275030 Ha, which with the
    nuclear repulsion term gives the known FCI total of -1.137306 Ha.

Every energy here follows the same split, which is the usual source of confusion
when comparing against published numbers:

    E_total = E_electronic (what the qubit operator measures) + E_nuclear_repulsion

The qubit operator encodes the *electronic* structure only. Diagonalising it
alone gives about -1.857 Ha, not the -1.137 Ha quoted for H2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qprac_lab.backends.qiskit_adapter import require_qiskit
from qprac_lab.baselines.exact_diagonalization import exact_lowest_eigenvalue

#: 1 bohr expressed in angstrom (CODATA 2018).
ANGSTROM_PER_BOHR = 0.529177210903

#: Canonical H2 / STO-3G parity-tapered coefficients at R = 0.735 A.
H2_BUILTIN_BOND_LENGTH = 0.735
H2_BUILTIN_TERMS: tuple[tuple[str, float], ...] = (
    ("II", -1.052373245772859),
    ("IZ", 0.397937424843179),
    ("ZI", -0.397937424843179),
    ("ZZ", -0.011280104256235),
    ("XX", 0.180931199784231),
)
#: Hartree-Fock occupation for the 2-qubit parity-mapped H2 problem.
H2_BUILTIN_HF_BITSTRING = "01"


def hydrogen_nuclear_repulsion(bond_length_angstrom: float) -> float:
    """Nuclear repulsion energy of H2 in hartree.

    For two unit charges this is simply ``1 / R`` in atomic units. Checked
    against Qiskit Nature at R = 0.735 A: both give 0.719968994 Ha.
    """
    if bond_length_angstrom <= 0:
        raise ValueError(f"bond length must be positive, got {bond_length_angstrom}")
    return ANGSTROM_PER_BOHR / bond_length_angstrom


@dataclass
class MolecularHamiltonian:
    """A qubit-mapped electronic Hamiltonian plus the constants needed to read it."""

    qubit_operator: Any
    nuclear_repulsion_energy: float
    hartree_fock_bitstring: str
    bond_length_angstrom: float
    basis: str
    source: str
    molecule: str = "H2"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        return int(self.qubit_operator.num_qubits)

    def to_matrix(self) -> np.ndarray:
        """Dense matrix of the electronic Hamiltonian."""
        return np.asarray(self.qubit_operator.to_matrix())

    def total_energy(self, electronic_energy: float) -> float:
        """Add nuclear repulsion to an electronic energy."""
        return float(electronic_energy) + self.nuclear_repulsion_energy

    def exact_electronic_energy(self) -> float:
        """Lowest eigenvalue of the electronic Hamiltonian (exact diagonalisation)."""
        return exact_lowest_eigenvalue(self.to_matrix())

    def exact_total_energy(self) -> float:
        """Full-CI ground-state energy within the chosen basis."""
        return self.total_energy(self.exact_electronic_energy())

    def hartree_fock_state(self):
        """Statevector of the Hartree-Fock reference determinant."""
        require_qiskit("Building the Hartree-Fock reference state")
        from qiskit.quantum_info import Statevector

        return Statevector.from_label(self.hartree_fock_bitstring)

    def hartree_fock_electronic_energy(self) -> float:
        """``<HF|H|HF>`` -- the mean-field reference, with no correlation."""
        state = self.hartree_fock_state()
        return float(state.expectation_value(self.qubit_operator).real)

    def hartree_fock_total_energy(self) -> float:
        return self.total_energy(self.hartree_fock_electronic_energy())

    def correlation_energy(self) -> float:
        """Exact minus Hartree-Fock: the energy mean-field theory cannot reach."""
        return self.exact_total_energy() - self.hartree_fock_total_energy()

    def describe(self) -> dict[str, Any]:
        """Serialisable summary for demo output and benchmark rows."""
        return {
            "molecule": self.molecule,
            "basis": self.basis,
            "bond_length_angstrom": self.bond_length_angstrom,
            "num_qubits": self.num_qubits,
            "source": self.source,
            "hartree_fock_bitstring": self.hartree_fock_bitstring,
            "nuclear_repulsion_energy": self.nuclear_repulsion_energy,
            "terms": [
                {"pauli": label, "coefficient": float(np.real(coeff))}
                for label, coeff in zip(
                    self.qubit_operator.paulis.to_labels(),
                    self.qubit_operator.coeffs,
                    strict=True,
                )
            ],
            **self.metadata,
        }


def nature_available() -> bool:
    """Return ``True`` when Qiskit Nature and a PySCF driver are importable."""
    try:
        import pyscf  # noqa: F401
        from qiskit_nature.second_q.drivers import PySCFDriver  # noqa: F401
    except ImportError:
        return False
    return True


def h2_hamiltonian_builtin() -> MolecularHamiltonian:
    """H2 at R = 0.735 A from stored coefficients -- no chemistry stack required."""
    require_qiskit("Building a molecular Hamiltonian")
    from qiskit.quantum_info import SparsePauliOp

    return MolecularHamiltonian(
        qubit_operator=SparsePauliOp.from_list(list(H2_BUILTIN_TERMS)),
        nuclear_repulsion_energy=hydrogen_nuclear_repulsion(H2_BUILTIN_BOND_LENGTH),
        hartree_fock_bitstring=H2_BUILTIN_HF_BITSTRING,
        bond_length_angstrom=H2_BUILTIN_BOND_LENGTH,
        basis="sto3g",
        source="builtin_table",
    )


def h2_hamiltonian_from_nature(
    bond_length_angstrom: float = H2_BUILTIN_BOND_LENGTH,
    basis: str = "sto3g",
) -> MolecularHamiltonian:
    """Build H2 at an arbitrary bond length via PySCF + Qiskit Nature.

    Uses a parity mapping with two-qubit reduction, which exploits particle-number
    and spin symmetry to bring H2 / STO-3G down from four qubits to two.
    """
    require_qiskit("Building a molecular Hamiltonian")
    if not nature_available():
        raise ImportError(
            "Qiskit Nature and PySCF are required for arbitrary bond lengths; "
            'install them with: pip install -e ".[nature]"'
        )
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import ParityMapper

    driver = PySCFDriver(atom=f"H 0 0 0; H 0 0 {bond_length_angstrom}", basis=basis)
    problem = driver.run()
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_operator = mapper.map(problem.hamiltonian.second_q_op())

    hf_circuit = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    hf_bitstring = _statevector_basis_label(hf_circuit)

    return MolecularHamiltonian(
        qubit_operator=qubit_operator,
        nuclear_repulsion_energy=float(problem.nuclear_repulsion_energy),
        hartree_fock_bitstring=hf_bitstring,
        bond_length_angstrom=bond_length_angstrom,
        basis=basis,
        source="qiskit_nature_pyscf",
        metadata={"nature_reference_energy": float(problem.reference_energy)},
    )


def _statevector_basis_label(circuit) -> str:
    """Read the computational-basis label out of a product-state circuit."""
    from qiskit.quantum_info import Statevector

    amplitudes = np.abs(Statevector(circuit).data)
    index = int(np.argmax(amplitudes))
    return format(index, f"0{circuit.num_qubits}b")


def build_h2_hamiltonian(
    bond_length_angstrom: float = H2_BUILTIN_BOND_LENGTH,
    basis: str = "sto3g",
    prefer_nature: bool = True,
) -> MolecularHamiltonian:
    """Return an H2 Hamiltonian, preferring the real chemistry path when available.

    Falls back to the stored coefficients only at the built-in bond length; any
    other geometry genuinely needs the chemistry stack, so asking for one without
    it raises rather than silently returning the wrong molecule.
    """
    if prefer_nature and nature_available():
        return h2_hamiltonian_from_nature(bond_length_angstrom, basis=basis)

    if not np.isclose(bond_length_angstrom, H2_BUILTIN_BOND_LENGTH):
        raise ImportError(
            f"Bond length {bond_length_angstrom} A requires Qiskit Nature and PySCF "
            f'(install with: pip install -e ".[nature]"); without them only '
            f"R = {H2_BUILTIN_BOND_LENGTH} A is available."
        )
    if basis != "sto3g":
        raise ImportError(
            f"Basis {basis!r} requires Qiskit Nature and PySCF; the built-in "
            f"table covers sto3g only."
        )
    return h2_hamiltonian_builtin()


def describe_pauli_hamiltonian() -> dict[str, Any]:
    """Backend-neutral description of the built-in H2 Hamiltonian."""
    return {
        "terms": [
            {"coefficient": coefficient, "pauli": pauli} for pauli, coefficient in H2_BUILTIN_TERMS
        ],
        "molecule": "H2",
        "basis": "sto3g",
        "bond_length_angstrom": H2_BUILTIN_BOND_LENGTH,
        "mapping": "parity mapping with two-qubit reduction",
        "note": "Electronic Hamiltonian; add nuclear repulsion for total energy.",
        "nuclear_repulsion_energy": hydrogen_nuclear_repulsion(H2_BUILTIN_BOND_LENGTH),
    }
