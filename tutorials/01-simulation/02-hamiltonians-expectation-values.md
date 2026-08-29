# Concept: Hamiltonians and expectation values

> **Short note, not a standalone tutorial.** The material lives in
> [VQE for Molecular Energy](01-vqe-molecular-energy.md) and
> [`hamiltonian_utils.py`](../../src/qprac_lab/algorithms/simulation/hamiltonian_utils.py).

Every variational algorithm reduces to one question a quantum computer can
answer: what is `⟨ψ|H|ψ⟩`? The circuit prepares `|ψ⟩`, the estimator returns that
number, and a classical optimiser does the rest.

Two things about that reduction are worth stating once, because both cause real
confusion downstream.

**A molecular Hamiltonian is electronic only.** Diagonalising the H₂ qubit
operator gives `−1.857275 Ha`, not the `−1.137306 Ha` quoted in the literature.
Nuclear repulsion is a classical constant added afterwards:

```text
E_total = E_electronic + E_nuclear,    E_nuclear = 1/R in atomic units
```

Comparing a raw qubit-operator eigenvalue against a published total energy is the
most common way to conclude your Hamiltonian is wrong when it is fine.

**Which terms matter is legible from the operator.** For 2-qubit H₂, every `Z`
term is diagonal, so a computational basis state diagonalises them — that state
*is* Hartree-Fock. Only the `XX` term mixes determinants, so `XX` alone carries
the electron correlation. That is why
[ADAPT-VQE](03-adapt-vqe-materials.md) selects an `XY`-type generator with
gradient `0.180931`: exactly the `XX` coefficient.

## Where this is used

- [VQE](01-vqe-molecular-energy.md) — the estimator loop over `⟨H⟩`
- [ADAPT-VQE](03-adapt-vqe-materials.md) — gradients as commutator expectations
- [Trotterization](04-trotterization-time-evolution.md) — `exp(−iHt)` instead of `⟨H⟩`
