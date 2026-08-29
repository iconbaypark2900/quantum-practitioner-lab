# Concept: quantum kernels

> **Short note, not a standalone tutorial.** The worked comparison on real data
> is [Quantum Kernel for Biomedical Classification](03-quantum-kernel-biomedical-classification.md).

A kernel method never sees your data — only pairwise similarities. That is the
opening a quantum computer fits through:

```text
K(x, x') = |⟨φ(x') | φ(x)⟩|²
```

Encode each point into a quantum state with a feature-map circuit, measure the
squared overlap, hand the matrix to a classical SVM. **The quantum computer
supplies a similarity measure, not a learner.**

Three facts that decide whether this is worth doing:

- **A feature map is a prior.** The ZZ map imposes oscillatory, periodic
  structure. If your data does not have that structure, it is a *worse* prior
  than RBF, not a neutral one.
- **Alignment predicts the outcome for one matrix multiply.** Kernel-target
  alignment `A(K, yyᵀ) = yᵀKy / (n‖K‖_F)` measures class separation before any
  classifier is fitted. It correctly predicted both the loss on blob data and the
  tie on real data.
- **Cost is quadratic.** One circuit pair per matrix entry, with no
  approximation. 10,000 samples is 50 million circuit pairs.

## Where this is used

- [Quantum kernel classification](03-quantum-kernel-biomedical-classification.md) — the full comparison
- [VQC classifier](04-vqc-classifier.md) — the variational alternative, and why the kernel's convexity matters
