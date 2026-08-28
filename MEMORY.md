# Project Memory

## Current state

The three priority tutorials are **real quantum implementations**, not scaffolds.
Secondary tutorials (PDEs, ADAPT-VQE, Trotter, VQC, Max-Cut) are still classical
scaffolds and are labelled as such by `qprac-lab list`.

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
- **QAOA**: only **1.12x** better than uniform sampling over feasible portfolios
  at the default penalty. Large penalty buys feasibility (99.3%) but flattens the
  distribution; small penalty (0.5) reaches 1.85x lift but drops feasibility to
  77%. The fix is a constraint-preserving XY mixer, not more tuning.
- **Quantum kernel**: loses to RBF-SVM (ROC-AUC 0.694 vs 0.826) at ~1675x the
  compute. Kernel-target alignment (0.0906 vs 0.1356) predicted this before any
  classifier was fitted — use it as a cheap go/no-go check.

## Next milestone

- Notebook walkthroughs for the three tutorials.
- XY mixer for QAOA so cardinality is preserved by construction.
- A genuinely KG-derived biomedical dataset; the current features are synthetic
  Gaussian blobs, which is exactly the geometry RBF is best at.
- Noise models, then IBM Runtime and CUDA-Q adapters.
