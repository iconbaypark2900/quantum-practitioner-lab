# Use Case: Materials Discovery Refinement

## Pipeline

```text
Materials dataset
  → classical ML surrogate
  → shortlist candidates
  → VQE / ADAPT-VQE refinement
  → energy and stability ranking
```

## Why quantum might help here

The state of a correlated electronic system takes exponentially many classical
numbers to write down, and a quantum register holds it natively. Where a
mean-field method like Hartree-Fock systematically misses correlation energy, a
variational quantum eigensolver can in principle recover it. The
[VQE tutorial](../01-vqe-molecular-energy.md) shows that recovery on H2, including
the dissociation curve where restricted Hartree-Fock fails and VQE tracks the
exact answer.

Note the shape of the argument: quantum enters *after* classical screening, on a
shortlist. Nothing here proposes replacing the surrogate.

## What is actually implemented

H2 in a minimal basis, parity-mapped to **2 qubits**, on a simulator. That is a
problem exact diagonalisation solves instantly, which is precisely why it works as
a teaching case — the right answer is known, so the method can be checked rather
than trusted.

## What the real use case would additionally need

- **Active-space selection.** Real materials need orbital selection to fit a
  device; that choice dominates the answer and is not modelled here.
- **Qubit counts one to two orders of magnitude larger**, with the measurement
  cost growing as the number of Pauli terms.
- **Error mitigation.** The [noise benchmark](../../05-benchmarking/noise_benchmark.md)
  shows VQE missing chemical accuracy at even the optimistic preset on a
  *two-qubit* circuit. It is the most fragile method in this repository.

## When not to use it

If a classical method already reaches chemical accuracy on your system, it wins —
today, on every system this repository can simulate. The case for quantum here is
about a regime none of these tutorials reach.
