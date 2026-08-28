# Project Memory

## Current state

The three priority tutorials plus QAOA Max-Cut are **real quantum
implementations**, not scaffolds. Remaining secondary tutorials (PDEs,
ADAPT-VQE, Trotter, VQC) are still classical scaffolds and are labelled as such
by `qprac-lab list`.

## Stack decision

Qiskit **2.x with V2 primitives only**. Qiskit 2.0 removed the V1
`Estimator`/`Sampler`, and VQE/QAOA no longer ship in qiskit core. This project
implements its own optimisation loops on top of `StatevectorEstimator` /
`StatevectorSampler` (and the Aer equivalents) plus `scipy.optimize`, so it does
not depend on `qiskit-algorithms`.

## Priority

1. VQE for Molecular Energy — done
2. QAOA for Portfolio Selection — done
3. Quantum Kernel for Biomedical Classification — done
4. QAOA Max-Cut — done (unconstrained reference problem)

## Important design decisions

- Every quantum algorithm must include a classical baseline.
- Qiskit is an **optional extra**. The package imports and tests without it;
  quantum code paths raise `QiskitNotInstalledError` with an install hint rather
  than failing deep in a stack trace. CI covers both configurations.
- Results are reported honestly. Two of three tutorials show the quantum method
  failing to beat its classical baseline, and both say so in their headline
  metrics rather than in a footnote.

## Verified reference values

H2 / STO-3G at R = 0.735 A, parity-mapped to 2 qubits. Cross-checked between the
built-in coefficient table and a live PySCF calculation:

```text
electronic energy   -1.857275030 Ha    (what the qubit operator measures)
nuclear repulsion   +0.719968994 Ha    (= 1/R in atomic units)
exact (FCI) total   -1.137306036 Ha
Hartree-Fock total  -1.116998997 Ha
correlation energy  -0.020307039 Ha
```

The electronic/nuclear split is the usual source of confusion when comparing
against published numbers — the qubit operator alone gives -1.857, not -1.137.

## Measured findings worth keeping

- **VQE**: one-parameter UCC ansatz reaches the exact energy to `5.6e-10` in 23
  evaluations; hardware-efficient `efficient_su2` needs 12 parameters and 268.
  At 8192 shots neither reaches chemical accuracy — shot noise is the binding
  constraint, and a noisy estimate can fall *below* the exact energy without
  violating the variational principle.
- **QAOA portfolio**: the penalty encoding is only ~**1.05x** better than uniform
  sampling over feasible portfolios — it learns feasibility and little else,
  because the penalty (6.17) dwarfs the objective spread (~2.1). Lowering it to
  0.5 doubles the lift but drops feasibility to 55%.
- **XY mixer removes that tradeoff.** `(XX+YY)/2` commutes with total Hamming
  weight, so feasibility becomes structural: **exactly 100%** at every depth and
  topology, with no penalty term at all. Optimality lift reaches 20x (ring, p=6).
  But it is **non-monotonic in depth** — ring gives 3.52% / 4.74% / 100% / 16.67%
  at p = 3/4/6/8. That is local optima from a fixed linear-ramp warm start, not
  evidence that p=6 is special. The complete topology is far more predictable
  (1.47x -> 7.50x) since it mixes the whole subspace in one layer.
- **QAOA Max-Cut**: 0.905 expected approximation ratio vs 0.600 for random
  assignment, 49.7% of shots optimal. Greedy also ties the exact optimum
  instantly, which is what an 8-vertex problem looks like.
- **Quantum kernel**: loses to RBF-SVM (ROC-AUC 0.694 vs 0.826) at ~1675x the
  compute. Kernel-target alignment (0.0906 vs 0.1356) predicted this before any
  classifier was fitted — use it as a cheap go/no-go check.

## Performance trap worth remembering

`QAOAAnsatz` holds `PauliEvolutionGate`s whose synthesis is redone on **every**
estimator call. Measured at 2.56s per call versus 0.006s after a one-time
`.decompose(reps=3)` on an 8-qubit graph — a ~400x difference that turned a
2-second optimisation into 13 minutes. `run_qaoa` decomposes up front.

Similarly, Aer's `EstimatorV2` seeds only via `run_options["seed_simulator"]`;
a plain `seed` key is accepted and ignored, leaving runs irreproducible.

## Next milestone

- Notebook walkthroughs for the finished tutorials.
- A genuinely KG-derived biomedical dataset; the current features are synthetic
  Gaussian blobs, which is exactly the geometry RBF is best at. This is the
  biggest remaining integrity gap — tutorial 3's conclusion is currently an
  artifact of the data generator.
- Dicke-state warm start for the XY mixer, to replace the single k-hot basis
  state and reduce the depth sensitivity seen above.
- Noise models, then IBM Runtime and CUDA-Q adapters.
