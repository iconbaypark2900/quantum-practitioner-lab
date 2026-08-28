"""QAOA mixers, including constraint-preserving XY mixers.

The standard QAOA mixer is a transverse field, ``sum_i X_i``, which moves
amplitude between *all* bitstrings. Under it a hard constraint has to be encoded
as a penalty in the cost operator -- and that penalty then competes with the
objective for the optimiser's attention. In the portfolio tutorial the penalty
wins: QAOA learns feasibility and ends up close to uniform over feasible states.

An **XY mixer** fixes this structurally. Each term ``(X_i X_j + Y_i Y_j) / 2``
commutes with the total number operator, so evolution under it cannot change
Hamming weight. Start in a state with exactly ``k`` ones and the whole
optimisation stays inside the ``k``-hot subspace: feasibility becomes free, the
penalty term disappears, and every angle works on the objective instead.

The tradeoff is a deeper circuit -- two-qubit XY rotations instead of
single-qubit X rotations -- which is a real cost on hardware.
"""

from __future__ import annotations

import numpy as np

from qprac_lab.backends.qiskit_adapter import require_qiskit

#: Coupling graphs an XY mixer can be built over.
XY_TOPOLOGIES = ("ring", "complete")


def xy_mixer_edges(num_qubits: int, topology: str = "ring") -> list[tuple[int, int]]:
    """Coupling pairs for an XY mixer.

    ``ring`` uses ``n`` nearest-neighbour couplings and is cheap but only
    *partially* mixing; ``complete`` uses all ``n(n-1)/2`` pairs and mixes the
    whole fixed-weight subspace at the cost of a quadratically deeper layer.
    """
    if topology == "ring":
        if num_qubits < 2:
            return []
        if num_qubits == 2:
            return [(0, 1)]
        return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    if topology == "complete":
        return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    raise ValueError(f"Unknown XY topology {topology!r}; expected one of {XY_TOPOLOGIES}")


def apply_xy_mixer(circuit, beta, topology: str = "ring") -> None:
    """Append one XY mixer layer, ``exp(-i beta sum_edges (XX + YY) / 2)``.

    ``XXPlusYYGate(theta)`` implements ``exp(-i (theta/2) (XX + YY) / 2)``, so the
    angle passed is ``2 * beta``. Getting that factor wrong does not break
    Hamming-weight preservation -- it just silently rescales every mixer angle,
    which is exactly the kind of bug an optimiser hides by re-tuning around it.
    """
    require_qiskit("Building an XY mixer")
    from qiskit.circuit.library import XXPlusYYGate

    for i, j in xy_mixer_edges(circuit.num_qubits, topology):
        circuit.append(XXPlusYYGate(2 * beta), [i, j])


def apply_transverse_field_mixer(circuit, beta) -> None:
    """Append the standard QAOA mixer, ``exp(-i beta sum_i X_i)``."""
    for qubit in range(circuit.num_qubits):
        circuit.rx(2 * beta, qubit)


def apply_diagonal_cost_layer(circuit, cost_operator, gamma) -> None:
    """Append ``exp(-i gamma C)`` for a diagonal (Z-only) cost operator.

    Because every term commutes, this is *exact* -- no Trotter error. Single-Z
    terms become ``RZ(2 * c * gamma)`` and ZZ terms ``RZZ(2 * c * gamma)``;
    identity terms contribute only a global phase and are skipped.
    """
    require_qiskit("Building a cost layer")
    for pauli, coefficient in zip(cost_operator.paulis, cost_operator.coeffs, strict=True):
        if pauli.x.any():
            raise ValueError(
                "cost operator must be diagonal (Z terms only) to use an exact cost "
                f"layer, got Pauli {pauli}"
            )
        qubits = np.flatnonzero(pauli.z)
        angle = 2.0 * float(np.real(coefficient)) * gamma
        if len(qubits) == 0:
            continue
        if len(qubits) == 1:
            circuit.rz(angle, int(qubits[0]))
        elif len(qubits) == 2:
            circuit.rzz(angle, int(qubits[0]), int(qubits[1]))
        else:
            raise ValueError(
                f"cost layer supports at most 2-local Z terms, got {len(qubits)}-local"
            )


def k_hot_initial_state(circuit, num_ones: int) -> None:
    """Prepare ``|1...10...0>`` -- one feasible basis state of Hamming weight k.

    A Dicke state (the uniform superposition over *all* weight-k bitstrings)
    would be the ideal warm start, but needs a dedicated preparation circuit. A
    single feasible basis state is the standard cheap substitute: the XY mixer
    spreads amplitude across the subspace from there.
    """
    if not 0 <= num_ones <= circuit.num_qubits:
        raise ValueError(f"num_ones {num_ones} outside 0..{circuit.num_qubits}")
    for qubit in range(num_ones):
        circuit.x(qubit)


def dicke_state_vector(num_qubits: int, num_ones: int) -> np.ndarray:
    """Amplitudes of the Dicke state ``|D^n_k>``.

    The uniform superposition over *every* bitstring of Hamming weight ``k``. It
    is the constrained analogue of ``|+>^n``: the unbiased starting point over the
    feasible subspace, where :func:`k_hot_initial_state` picks one arbitrary
    member of it.
    """
    from itertools import combinations

    if not 0 <= num_ones <= num_qubits:
        raise ValueError(f"num_ones {num_ones} outside 0..{num_qubits}")
    amplitudes = np.zeros(2**num_qubits)
    for chosen in combinations(range(num_qubits), num_ones):
        amplitudes[sum(1 << qubit for qubit in chosen)] = 1.0
    return amplitudes / np.linalg.norm(amplitudes)


def dicke_initial_state(circuit, num_ones: int) -> None:
    """Prepare a Dicke state by exact state preparation.

    Exact and simple, but **not scalable**: generic state preparation synthesises
    into depth that grows fast with qubit count -- measured at depth 272 for
    ``n=6, k=3`` and 1241 for ``n=8, k=4``, which is more than the QAOA circuit it
    is warming up. Baertschi and Eidenbenz (2019) give a dedicated ``O(kn)``
    construction; that is the route for anything hardware-bound. Implemented this
    way here because the *physics* is identical either way, and the question this
    warm start exists to answer is about solution quality, not gate counts.
    """
    require_qiskit("Preparing a Dicke state")
    from qiskit.circuit.library import StatePreparation

    amplitudes = dicke_state_vector(circuit.num_qubits, num_ones)
    circuit.append(StatePreparation(amplitudes), range(circuit.num_qubits))


#: How the fixed-weight subspace is entered before the first cost layer.
INITIAL_STATES = ("k_hot", "dicke")


def build_xy_qaoa_ansatz(
    cost_operator,
    reps: int,
    num_ones: int,
    topology: str = "ring",
    initial_state: str = "k_hot",
):
    """Constraint-preserving QAOA ansatz over the fixed-Hamming-weight subspace.

    Parameters are ordered ``[beta_0..beta_{p-1}, gamma_0..gamma_{p-1}]``, matching
    Qiskit's own ``QAOAAnsatz`` so the same initial-point helper works for both.

    ``initial_state`` selects how the feasible subspace is entered: ``k_hot`` picks
    one arbitrary feasible bitstring, ``dicke`` starts unbiased across all of them.
    """
    require_qiskit("Building an XY-mixer QAOA ansatz")
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    num_qubits = cost_operator.num_qubits
    betas = ParameterVector("beta", reps)
    gammas = ParameterVector("gamma", reps)

    circuit = QuantumCircuit(num_qubits)
    if initial_state == "dicke":
        dicke_initial_state(circuit, num_ones)
    elif initial_state == "k_hot":
        k_hot_initial_state(circuit, num_ones)
    else:
        raise ValueError(
            f"Unknown initial_state {initial_state!r}; expected one of {INITIAL_STATES}"
        )
    for layer in range(reps):
        apply_diagonal_cost_layer(circuit, cost_operator, gammas[layer])
        apply_xy_mixer(circuit, betas[layer], topology=topology)
    return circuit
