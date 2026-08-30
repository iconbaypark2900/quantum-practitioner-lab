# Use Case: Biomedical KG Link Prediction

Predicting which compounds treat which diseases from a biomedical knowledge graph —
drug repurposing, framed as link prediction.

## Pipeline

```text
Hetionet (CC0)
  → Compound-treats-Disease edges as positives
  → degree-matched negatives          ← the step that makes this honest
  → pairwise features
  → quantum kernel / VQC vs RBF-SVM / Random Forest
  → paired 5x4 cross-validated comparison
```

## What is implemented

Real [Hetionet](https://het.io) data, 200 pairs, 4 features, run by both the
[quantum kernel](../03-quantum-kernel-biomedical-classification.md) and the
[VQC](../04-vqc-classifier.md) on identical folds and seeds.

## The result

Quantum kernel ranks first at **0.587 ± 0.096** ROC-AUC against RBF's
**0.577 ± 0.076** — and wins only **12 of 20 paired folds**. That is a statistical
tie, and it is reported as one: `difference_exceeds_noise: false`.

The evaluation is `5x4` repeated stratified CV because a single split on 200 pairs
swings ROC-AUC by more than the gap between the models. A single-split number here
would have been noise presented as a finding.

## Why degree matching is the load-bearing step

Well-studied compounds have many edges. Sample negatives at random and node degree
alone predicts the label, so a model can score well by learning *popularity*
instead of biology. With degree-matched negatives, degree alone scores **0.538**;
without matching, **0.689**. The dataset reports `degree_only_roc_auc` so this is
measured rather than assumed.

This is also the cautionary tale of the section. An earlier version used synthetic
Gaussian blobs and showed RBF beating the quantum kernel clearly — a conclusion
that turned out to be an artifact of the generator, not a property of either
method. Changing to real data reversed it into a tie.

## Metrics

ROC-AUC as the headline (it is threshold-free and the classes are balanced by
construction), plus F1 and accuracy per model, and `kernel_alignment` as a
pre-training go/no-go. Precision@K and Recall@K would matter for a deployed
repurposing shortlist and are not implemented — with 200 pairs they would be
dominated by fold noise.

## When not to use it

At 4 features and 200 pairs, an RBF kernel is a tie and a Random Forest is
competitive, both in milliseconds. The quantum kernel costs `O(n²)` circuit pairs
to build its matrix. There is no regime in this tutorial where that is the right
trade — the value is the honest comparison, not the method.
