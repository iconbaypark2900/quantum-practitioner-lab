"""Molecular Hamiltonian checks against published H2 / STO-3G values."""

import numpy as np
import pytest

pytest.importorskip("qiskit", reason='needs the quantum stack: pip install -e ".[qiskit]"')

from qprac_lab.algorithms.simulation.hamiltonian_utils import (  # noqa: E402
    ANGSTROM_PER_BOHR,
    build_h2_hamiltonian,
    describe_pauli_hamiltonian,
    h2_hamiltonian_builtin,
    hydrogen_nuclear_repulsion,
    nature_available,
)

H2_ELECTRONIC_ENERGY = -1.857275030
H2_NUCLEAR_REPULSION = 0.719968994
H2_FCI_ENERGY = -1.137306036
H2_HARTREE_FOCK_ENERGY = -1.116998997


def test_nuclear_repulsion_is_one_over_r_in_atomic_units():
    assert hydrogen_nuclear_repulsion(0.735) == pytest.approx(H2_NUCLEAR_REPULSION, abs=1e-8)
    assert hydrogen_nuclear_repulsion(ANGSTROM_PER_BOHR) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        hydrogen_nuclear_repulsion(0.0)


def test_builtin_hamiltonian_reproduces_known_h2_energies():
    hamiltonian = h2_hamiltonian_builtin()
    assert hamiltonian.num_qubits == 2
    assert hamiltonian.exact_electronic_energy() == pytest.approx(H2_ELECTRONIC_ENERGY, abs=1e-7)
    assert hamiltonian.exact_total_energy() == pytest.approx(H2_FCI_ENERGY, abs=1e-7)
    assert hamiltonian.hartree_fock_total_energy() == pytest.approx(
        H2_HARTREE_FOCK_ENERGY, abs=1e-7
    )


def test_correlation_energy_is_negative_and_small():
    """Correlation lowers the energy below Hartree-Fock, by about 20 mHa for H2."""
    hamiltonian = h2_hamiltonian_builtin()
    assert hamiltonian.correlation_energy() < 0
    assert hamiltonian.correlation_energy() == pytest.approx(-0.020307, abs=1e-5)


def test_electronic_energy_alone_is_not_the_quoted_total():
    """Guards the split that trips up every comparison against the literature."""
    hamiltonian = h2_hamiltonian_builtin()
    assert hamiltonian.exact_electronic_energy() < H2_FCI_ENERGY
    assert hamiltonian.total_energy(
        hamiltonian.exact_electronic_energy()
    ) == pytest.approx(H2_FCI_ENERGY, abs=1e-7)


@pytest.mark.skipif(
    not nature_available(), reason="requires the nature extra (qiskit-nature+pyscf)"
)
def test_nature_path_agrees_with_the_builtin_table():
    """The stored coefficients must match what PySCF actually computes."""
    builtin = h2_hamiltonian_builtin()
    computed = build_h2_hamiltonian(0.735, prefer_nature=True)
    assert computed.source == "qiskit_nature_pyscf"
    assert computed.nuclear_repulsion_energy == pytest.approx(
        builtin.nuclear_repulsion_energy, abs=1e-7
    )
    assert computed.exact_total_energy() == pytest.approx(builtin.exact_total_energy(), abs=1e-7)
    assert computed.hartree_fock_total_energy() == pytest.approx(
        builtin.hartree_fock_total_energy(), abs=1e-7
    )
    assert np.allclose(
        np.sort(np.linalg.eigvalsh(computed.to_matrix())),
        np.sort(np.linalg.eigvalsh(builtin.to_matrix())),
        atol=1e-7,
    )


@pytest.mark.skipif(
    not nature_available(), reason="requires the nature extra (qiskit-nature+pyscf)"
)
def test_hartree_fock_error_grows_as_the_bond_stretches():
    """Restricted HF fails on a breaking bond; that failure is the tutorial's point."""
    near = build_h2_hamiltonian(0.735)
    stretched = build_h2_hamiltonian(2.5)
    assert abs(stretched.correlation_energy()) > abs(near.correlation_energy()) * 5


def test_builtin_fallback_refuses_other_geometries():
    """Without the chemistry stack, silently returning the wrong molecule is worse
    than failing."""
    if nature_available():
        pytest.skip("nature is installed, so arbitrary bond lengths are supported")
    with pytest.raises(ImportError):
        build_h2_hamiltonian(1.5)


def test_describe_pauli_hamiltonian_is_serialisable():
    described = describe_pauli_hamiltonian()
    assert described["molecule"] == "H2"
    assert len(described["terms"]) == 5
    assert described["nuclear_repulsion_energy"] == pytest.approx(H2_NUCLEAR_REPULSION, abs=1e-8)
