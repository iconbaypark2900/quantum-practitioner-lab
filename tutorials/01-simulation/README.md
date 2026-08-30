# 01 Quantum Simulation

Algorithms for molecules, materials and time evolution — the domain where quantum
computers have the clearest theoretical case, because the thing being simulated is
itself a quantum system.

## Tutorials

| # | Tutorial | What it shows |
| --- | --- | --- |
| 1 | [VQE for Molecular Energy](01-vqe-molecular-energy.md) | H2 ground state to `5.6e-10` Ha against exact diagonalisation. The finding is that **shot noise, not the algorithm, is the binding constraint** — at 8192 shots the same circuit misses chemical accuracy. |
| 2 | [Hamiltonians and Expectation Values](02-hamiltonians-expectation-values.md) | What the Estimator actually computes, and why which Pauli terms matter is legible from the operator before you run anything. |
| 3 | [ADAPT-VQE for Materials](03-adapt-vqe-materials.md) | Growing the ansatz from a gradient-ranked pool instead of fixing it. Reaches the same energy with **1 parameter where `efficient_su2` needs 12**. |
| 4 | [Trotterization for Time Evolution](04-trotterization-time-evolution.md) | TFIM evolution with the error scaling *verified*, not asserted: fitted exponents **1.09** and **2.09** against a theory of 1 and 2. And the depth/noise tradeoff that makes the second-order product formula lose on hardware. |

## Primary use case

[Materials discovery refinement](use_cases/materials_discovery_refinement.md) —
classical screening first, quantum refinement on the shortlist.

## Not covered here

**Quantum phase estimation** has no standalone tutorial. It is not skipped: the
[HHL tutorial](../03-pdes/01-hhl-linear-systems-intro.md) builds and runs a QPE
subroutine, which is where its cost becomes visible rather than abstract. A
separate page would repeat that circuit without the problem that motivates it.

## Papers

[Source papers for this section](papers.md) — generated from
`configs/papers.yaml`, every entry carrying a resolvable DOI or arXiv id.
