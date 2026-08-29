# Concept: QSVC

> **Short note, not a standalone tutorial.** QSVC is implemented and benchmarked
> in [Quantum Kernel for Biomedical Classification](03-quantum-kernel-biomedical-classification.md).

QSVC is not a separate algorithm. It is a support vector classifier over a
quantum kernel matrix, and Qiskit's `QSVC` is a thin wrapper around exactly that:

```python
SVC(kernel="precomputed").fit(quantum_kernel_matrix, y)
```

This project uses the explicit form, for two reasons that matter in practice:

- **The kernel matrix is computed once** and reused across every
  cross-validation fold, so 20 evaluations cost the quantum time of one.
- **The matrix stays inspectable.** Its diagonal is the direct readout of circuit
  fidelity — `K(x,x)` is 1 by definition, and under moderate noise it measures
  0.734. Qiskit's default `evaluate_duplicates="off_diagonal"` *assumes* that
  diagonal rather than measuring it, hiding exactly the damage worth seeing.

## Where this is used

- [Quantum kernel classification](03-quantum-kernel-biomedical-classification.md)
- [Noise benchmark](../05-benchmarking/noise_benchmark.md) — self-fidelity under noise
