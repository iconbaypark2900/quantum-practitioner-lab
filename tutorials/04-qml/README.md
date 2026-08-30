# 04 Quantum Machine Learning

Quantum kernels and variational classifiers, evaluated on real data with paired
folds. The headline result here is a **statistical tie**, reported as one.

## Tutorials

| # | Tutorial | What it shows |
| --- | --- | --- |
| 1 | [Quantum Kernels](01-quantum-kernels.md) | The idea: the quantum computer produces a *similarity number*, `K(x,x') = |⟨φ(x')|φ(x)⟩|²`, and an ordinary SVM learns on top. The quantum part is not the learner. |
| 2 | [QSVC Classifier](02-qsvc-classifier.md) | What `QSVC` actually wraps — `SVC(kernel="precomputed")` — and why using it directly is clearer. |
| 3 | [Quantum Kernel for Biomedical Classification](03-quantum-kernel-biomedical-classification.md) | Real Hetionet Compound–treats–Disease edges. Quantum ranks first (**0.587 ± 0.096** vs RBF **0.577 ± 0.076**) but wins only **12/20 paired folds** — a tie, reported as `difference_exceeds_noise: false`. |
| 4 | [VQC Classifier](04-vqc-classifier.md) | A controlled comparison against the kernel: same data, same folds, same seeds. Its ranking sits inside its own noise, and says so. |

## Use case

[Biomedical KG link prediction](use_cases/biomedical_kg_link_prediction.md) — the
task tutorials 3 and 4 both run on.

## Honesty check built into the data

Negatives are **degree-matched**, because well-studied compounds have high degree
and a model can score well by learning popularity instead of biology. With
matching, degree alone scores **0.538** ROC-AUC; without it, **0.689**. The loader
reports `degree_only_roc_auc` so this is checked rather than asserted.

## Papers

[Source papers for this section](papers.md).
