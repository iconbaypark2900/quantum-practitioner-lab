# Verification: re-running the results through PennyLane

```bash
pip install -e ".[pennylane]"
python scripts/run_cross_check.py
```

## Why this exists

Every serious bug found while building this project produced **plausible numbers
rather than an error**:

- Aer's `EstimatorV2` silently ignored a `seed` option, so "reproducible" results
  were not.
- An undecomposed `PauliEvolutionGate` evaluated as the exact matrix exponential,
  reporting a Trotter error of `1e-15` at every step count.
- A quantum kernel's diagonal was *assumed* to be 1 rather than measured, hiding
  26% fidelity loss under noise.
- A QUBO bitstring decoded in the wrong endianness yields a valid-looking
  portfolio with a plausible objective value.

None of those raise. None are visible from inside a single library's conventions.
An independent implementation is the cheapest thing that catches them.

This is also why the planned IBM Runtime adapter was dropped in favour of
PennyLane: Runtime needs credentials, cannot be exercised in CI, and would have
run the same Qiskit stack anyway — so it would have verified nothing.

## What is checked

The PennyLane implementations are written **natively**, not translated from the
Qiskit circuits. A translation carries over exactly the convention errors it is
supposed to detect.

### VQE on H₂

| Bond length | Qiskit | PennyLane | \|difference\| |
| --- | --- | --- | --- |
| 0.735 Å | `-1.137306035` | `-1.137306035` | `4.4e-16` |
| 1.000 Å | `-1.101150330` | `-1.101150330` | `4.4e-16` |
| 2.500 Å | `-0.936054920` | `-0.936054920` | `2.2e-16` |

Machine precision, across the whole dissociation range including the strongly
correlated stretched bond.

### QUBO → Ising mapping

All 64 assignments of the 6-asset portfolio problem, evaluated in both stacks and
against the original QUBO objective: max difference `1.8e-14`.

This is the check that matters most, because **a mirror-image decoding produces a
valid portfolio and a plausible objective value**. Nothing inside Qiskit would
flag it. The mapping is verified exhaustively within Qiskit already; this confirms
the *conventions* too.

## The qubit-ordering trap

Qiskit Pauli labels are little-endian — the leftmost character is the **highest**
qubit index — while PennyLane addresses wires explicitly. So character `i` of an
`n`-qubit label maps to wire `n - 1 - i`.

Get it backwards and you produce a mirror-image Hamiltonian with an **identical
spectrum**. Every eigenvalue test passes. `to_pennylane_hamiltonian` maps by index,
and the test suite pins an *asymmetric* operator (`Z` on qubit 0 only) — the only
kind that catches a flip.

## What this does not prove

Agreement means the two stacks share no *implementation* error. It says nothing
about a shared *conceptual* one: if the Hamiltonian is wrong, both will faithfully
reproduce the wrong answer. That is what the literature comparisons are for — H₂
at `-1.137306` Ha is checked against published FCI values, not just against
itself.

Verification is a layered argument, and this is one layer.
