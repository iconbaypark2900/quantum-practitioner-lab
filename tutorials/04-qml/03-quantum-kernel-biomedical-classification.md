# Tutorial 3: Quantum Kernel for Biomedical Classification

## Concept

A kernel method never looks at your data directly -- it only ever asks "how
similar are these two points?" An SVM given a matrix of pairwise similarities can
classify without seeing a single feature vector.

That is the opening a quantum computer can fit through. Encode each data point
into a quantum state `|phi(x)>` with a parameterised circuit, and define
similarity as the squared overlap:

```text
K(x, x') = |<phi(x') | phi(x)>|^2
```

Measure that for every pair, hand the matrix to a classical SVM, done. The
quantum computer contributes **a similarity measure, not a learner**.

## The data is real, and that took work

This tutorial predicts genuine **`Compound--treats--Disease`** edges from
[Hetionet](https://het.io) (Himmelstein et al., eLife 2017), the canonical
drug-repurposing knowledge graph: 47k nodes, 2.25M edges, CC0-licensed. There are
755 known treatment edges over 1552 compounds and 137 diseases.

An earlier version of this tutorial used Gaussian blobs from
`make_classification`, and concluded that the quantum kernel loses. **That
conclusion was an artifact of the data generator** -- blob geometry is close to
ideal for an RBF kernel, so the comparison was rigged before it started. The
conclusion changed once the data became real. That is the whole reason this
section exists.

Two construction choices do the real work, and both exist to stop the benchmark
being accidentally *easy* -- the opposite failure, and just as misleading:

**No label leakage.** Features come only from compound--gene and gene--disease
edges. The `CtD` edge being predicted is never traversed, and neither is `CpD`
(*palliates*), which is close enough to *treats* to act as a label in disguise.
Because no feature path touches either, there is nothing to mask per split --
the separation is structural, and a test asserts it.

**Degree-matched negatives.** The naive negative -- a random compound and a
random disease -- makes degree alone predictive: well-studied compounds have both
more edges and more known treatments, so a classifier can score well while
learning no biology at all. Instead, each positive `(compound, disease)` is
paired with a negative `(other_compound, disease)`. The disease is held fixed, so
its degree matches *exactly*, and the substitute compound is drawn from the 15
nearest by degree.

Does it work? The dataset reports it rather than asserting it:

| Negative sampling | Degree-only ROC-AUC | Biology-features ROC-AUC |
| --- | --- | --- |
| Uniform random pairs | **0.689** | 0.729 |
| Degree-matched (this dataset) | **0.538** | 0.616 |

Read the first column. With uniform random negatives, **node degree alone scores
0.689** -- a classifier given nothing but "how well studied is this compound"
does most of the job, and the 0.729 next to it is largely that same shortcut
wearing a lab coat. Degree matching drops it to 0.538, near a coin flip, and the
biology score falls to 0.616 because the easy signal is gone.

The task got harder and the numbers got worse. That is what fixing a benchmark
looks like.

## Math intuition

The ZZ feature map (Havlicek et al.) encodes `x` in two stages, repeated `reps`
times:

```text
H on every qubit                       -> uniform superposition
P(2 x_i) on qubit i                    -> encode each feature
P(2 (pi - x_i)(pi - x_j)) on (i,j)     -> encode feature products
```

The entangling stage matters. Without it the map is a product state and the
kernel factorises into something classically trivial.

Overlaps use the **compute-uncompute** trick: run `phi(x)`, then `phi(x')`
inverted, and measure the probability of returning to `|0...0>`.

**Scaling is not optional on real data.** These features are raw counts and a
Jaccard index, spanning three orders of magnitude -- an overlap count up to 102
beside a Jaccard index of 0.07. Feature values become rotation *angles*, so
feeding those in raw wraps the rotations many times over and destroys the kernel.
They are min-max scaled to `[0, pi]`. Blob data hid this problem by arriving
pre-standardised.

## Minimal example

```python
from qprac_lab.data.hetionet import make_hetionet_link_prediction_dataset

dataset = make_hetionet_link_prediction_dataset(n_pairs=200)
print(dataset.feature_names)
# ('bind_gene_overlap', 'any_gene_overlap', 'jaccard_gene', 'adamic_adar_gene')
print(dataset.metadata["degree_only_roc_auc"])   # 0.5376 -- no degree shortcut
```

The features are interpretable graph quantities, not embeddings: how many genes
the compound binds that the disease implicates, raw and Jaccard-normalised
overlap, and an Adamic-Adar score that weights rare shared genes more heavily.

## Runnable implementation

```bash
python scripts/download_data.py                                  # ~12 MB, once
python scripts/run_demo.py --algorithm quantum_kernel_biomedical
```

## Evaluation: why a single split will not do

At the dataset sizes a quantum kernel can afford, **one train/test split is
uninformative**. Measured on this data, an 80-sample split gave test ROC-AUCs
anywhere from 0.54 to 0.85 depending only on the split seed:

| Samples | Test size | RBF ROC-AUC over 20 splits |
| --- | --- | --- |
| 80 | 24 | 0.702 ± 0.074 (range 0.54–0.85) |
| 200 | 60 | 0.609 ± 0.056 |
| 400 | 120 | 0.635 ± 0.040 |
| 1510 | 453 | 0.598 ± 0.026 |

That spread is far wider than any gap between the models. So this tutorial uses
**repeated stratified cross-validation** (5 folds × 4 repeats = 20 evaluations).

The quantum kernel makes that affordable in a way worth internalising: the full
`n x n` matrix is computed **once**, and every fold reuses submatrices of it. All
20 evaluations cost the same quantum time as one.

## Benchmark

200 pairs, 4 features, ZZ feature map (`reps=2`, 4 qubits, depth 19), 19,900
circuit pairs, 20 cross-validation folds:

| Rank | Model | ROC-AUC | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| 1 | **Quantum kernel SVM** | **0.587 ± 0.096** | 0.506 | 0.554 |
| 2 | RBF-SVM | 0.577 ± 0.076 | 0.457 | 0.546 |
| 3 | Random Forest | 0.501 ± 0.075 | 0.469 | 0.531 |

The quantum kernel ranks first. **It has not won.**

Both models saw identical folds, so the fair comparison is paired:

| Paired quantum vs RBF | Value |
| --- | --- |
| Mean ROC-AUC difference | **+0.0100** |
| Standard deviation of difference | 0.0858 |
| Folds where quantum scored higher | 12 / 20 (60%) |
| **Difference exceeds noise** | **No** |

A +0.01 edge against a fold-to-fold spread of 0.09, winning 12 of 20 folds. That
is a **statistical tie**, and `difference_exceeds_noise` is reported as a field
so it cannot be quietly dropped from a summary. The comparison plot draws the
error bars for the same reason.

Kernel-target alignment agrees, and again predicts the ranking before any
classifier is fitted:

| Kernel | Alignment |
| --- | --- |
| Quantum (ZZ) | **0.0266** |
| RBF | 0.0101 |

On blobs this was reversed (0.091 quantum vs 0.136 RBF) and correctly predicted
an RBF win. It costs one matrix multiply, so run it before committing to a full
comparison.

And the cost:

| Measure | Value |
| --- | --- |
| Quantum kernel matrix (200 × 200) | `77.6 s` |
| All 20 folds of all 3 classical models | `2.3 s` |
| Overhead | **~33x** |

That 33x is *flattering* to the quantum method -- it counts one kernel
computation against 20 full classical fits. Per fit, the gap is far wider.

## What to take from this

The honest summary changed when the data did. On blob data the quantum kernel
lost clearly; on real biomedical graph features it ties. **A tie is not a win,**
and this instance is small (200 pairs, 4 features), so the result should not be
over-read in either direction. But it does show the earlier negative result was
about the benchmark, not the method -- which is exactly why the dataset was worth
replacing.

Note also that every model here is close to chance (0.50–0.59). Drug repurposing
from graph topology alone, with degree-matched negatives, is genuinely hard. A
tutorial reporting 0.95 on this task would be reporting a leak.

## Visualization

- `results/kernel_matrix.png` -- the quantum kernel matrix. Block structure means
  the map is separating classes; a uniform matrix means it is not.
- `results/kernel_model_comparison.png` -- ROC-AUC with error bars.

## Real use case

```text
Hetionet knowledge graph
  -> compound-gene and gene-disease edges
  -> pair features (shared targets, Adamic-Adar, Jaccard)
  -> kernel classifier -> P(compound treats disease)
  -> ranked repurposing hypotheses
  -> literature check or wet-lab validation
```

This is the real drug-repurposing pipeline, and the regime suits kernel methods:
validated links are expensive, so datasets stay small -- which is also the only
regime where a quadratic-cost kernel is affordable at all.

## When not to use this

- **Datasets beyond a few hundred points.** Quadratic circuit cost with no
  approximation available. 10,000 samples is 50 million circuit pairs.
- **Without checking alignment first.** One matrix multiply tells you whether the
  full comparison is worth running.
- **Without repeated evaluation.** A single split on 200 samples cannot resolve a
  0.01 difference. Any quantum-advantage claim from one split is noise.
- **When you need calibrated probabilities.** Precomputed-kernel SVMs give
  decision-function scores; probabilities need an extra Platt-scaling fit.
- **On noisy hardware without checking self-fidelity.** `K(x, x)` is 1 by
  definition, so measuring it reads out how much of the circuit survived: 0.936 /
  0.734 / 0.388 across the light / moderate / heavy presets. Note that
  `FidelityQuantumKernel` *assumes* that diagonal by default
  (`evaluate_duplicates="off_diagonal"`) and silently repairs the resulting
  non-PSD matrix (`enforce_psd=True`), so both defaults hide exactly the damage
  you would want to see. See
  [the noise benchmark](../05-benchmarking/noise_benchmark.md).
- **Expecting an advantage.** There is still no demonstrated quantum-kernel
  advantage on practical data. A tie on 200 samples is not one.

## Source papers

- Havlicek et al., "Supervised learning with quantum-enhanced feature spaces"
  (2019) -- the ZZ feature map and the quantum kernel method.
- Himmelstein et al., "Systematic integration of biomedical knowledge prioritizes
  drugs for repurposing" (eLife 2017) -- Hetionet and the `CtD` task.
- Schuld and Killoran, "Quantum machine learning in feature Hilbert spaces"
  (2019) -- the kernel view of quantum ML.
- Huang et al., "Power of data in quantum machine learning" (2021) -- when
  quantum kernels can and cannot beat classical ones.
- Cristianini et al., "On Kernel-Target Alignment" (2001) -- the alignment metric.
