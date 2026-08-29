# Concept: QUBO and the Ising mapping

> **Short note, not a standalone tutorial.** The full derivation, with an
> exhaustive correctness check, is in
> [QAOA for Portfolio Selection](02-qaoa-portfolio-selection.md) and
> [`qubo_builder.py`](../../src/qprac_lab/algorithms/optimization/qubo_builder.py).

Combinatorial problems reach a quantum computer through two translations, and
both can fail silently.

**Constrained problem → QUBO.** An Ising Hamiltonian has no notion of "subject
to", so hard constraints become penalty terms:

```text
minimise  f(x) + P · (constraint violation)²
```

The penalty is not free. If it dominates the objective the optimiser learns
feasibility and nothing else — measured at a 1.05× lift over random guessing in
the portfolio tutorial.

**QUBO → Ising.** Substitute `x = (1 − z)/2`, with `z` the ±1 eigenvalue of a
Pauli-Z:

```text
constant = Σᵢ Q[i,i]/2 + Σᵢ<ⱼ S[i,j]/2        S = (Q + Qᵀ)/2
hᵢ       = −Q[i,i]/2 − Σⱼ≠ᵢ S[i,j]/2
Jᵢⱼ      = S[i,j]/2
```

A classical cost function must map to a **diagonal** Hamiltonian. If yours
contains `X` or `Y` terms, the mapping is wrong.

**Bitstrings back to variables.** Qiskit prints bitstrings little-endian — the
leftmost character is the *highest* qubit index. Reversing this yields a
mirror-image solution that is still valid-looking, which is why it is handled in
one place (`QUBO.bitstring_to_selection`) and
[verified against PennyLane](../05-benchmarking/cross_framework_verification.md).

The test suite checks the mapping over **every** `2ⁿ` assignment rather than a
sample, because a partially-correct mapping is indistinguishable from a correct
one on most inputs.

## Where this is used

- [QAOA Max-Cut](01-qaoa-maxcut.md) — unconstrained, so no penalty needed
- [QAOA portfolio selection](02-qaoa-portfolio-selection.md) — penalty vs XY mixer
