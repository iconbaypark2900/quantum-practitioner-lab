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
quantum computer contributes **a similarity measure, not a learner**. All the
learning is still the SVM's.

The hoped-for advantage is that some feature maps produce similarity structures
that are hard to compute classically. Whether that structure is ever *useful* for
real data is an open question, and this tutorial is built to answer it honestly
for the case at hand rather than to advertise.

## Math intuition

The ZZ feature map (Havlicek et al.) encodes `x` in two stages, repeated
`reps` times:

```text
H on every qubit                       -> uniform superposition
P(2 x_i) on qubit i                    -> encode each feature
P(2 (pi - x_i)(pi - x_j)) on (i,j)     -> encode feature products
```

The entangling stage is what matters. Without it the map is a product state and
the kernel factorises into something classically trivial. The products
`(pi - x_i)(pi - x_j)` are conjectured hard to simulate at depth -- that
conjecture is the entire basis for expecting an advantage.

Overlaps are computed by the **compute-uncompute** trick: run `phi(x)`, then run
`phi(x')` inverted, and measure the probability of returning to `|0...0>`. That
probability *is* `|<phi(x')|phi(x)>|^2`.

The cost is the catch. A kernel matrix needs one circuit per pair:

```text
train: n(n-1)/2 circuits      test: n_test * n_train circuits
```

That is **quadratic in dataset size**, before any learning happens.

## Minimal example

```python
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel

kernel = FidelityQuantumKernel(feature_map=zz_feature_map(feature_dimension=4, reps=2))
matrix = kernel.evaluate(x_vec=x_train)   # symmetric, diagonal exactly 1
```

The diagonal is exactly 1 because `|<phi(x)|phi(x)>|^2 = 1` -- every point is
perfectly similar to itself. If your diagonal is not 1, the kernel is broken.

## Runnable implementation

```bash
python scripts/run_demo.py --algorithm quantum_kernel_biomedical
```

The classifier is `SVC(kernel="precomputed")`, which is exactly what Qiskit's
`QSVC` wraps.

## Classical baselines

RBF-SVM, Random Forest, and optionally XGBoost (`pip install -e ".[xgboost]"`),
all on the same train/test split.

## Benchmark

56 train / 24 test, 4 features, ZZ feature map with `reps=2` (4 qubits, depth 19),
2884 circuit evaluations:

| Rank | Model | ROC-AUC | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| 1 | RBF-SVM | **0.826** | 0.632 | 0.708 |
| 2 | Random Forest | 0.729 | 0.583 | 0.583 |
| 3 | Quantum kernel SVM | 0.694 | 0.714 | 0.667 |

**The quantum kernel loses.** That is the correct and expected outcome, and it is
reported rather than tuned away.

Kernel-target alignment predicted it before a single classifier was fitted:

| Kernel | Alignment |
| --- | --- |
| RBF | **0.1356** |
| Quantum (ZZ) | 0.0906 |

Alignment measures how well a kernel separates the classes on its own,
`A(K, yy^T) = y^T K y / (n ||K||_F)`. It costs one matrix multiply. **A quantum
kernel with lower alignment than RBF will not out-classify it**, so this is the
cheapest possible go/no-go check -- run it before investing in a full comparison.

And the cost:

| Measure | Value |
| --- | --- |
| Quantum kernel matrices | `10.49 s` |
| RBF-SVM (fit + predict) | `0.0063 s` |
| Overhead | **~1675x** |

On a simulator, for 80 samples. The overhead grows quadratically.

Why the loss is unsurprising: these features are Gaussian blobs from
`make_classification`. RBF similarity is close to an ideal inductive bias for
that geometry, while the ZZ map imposes an oscillatory, periodic structure that
matches nothing in the data. **A quantum feature map is a prior, and a prior that
does not match the data hurts.**

## Visualization

- `results/kernel_matrix.png` -- the quantum kernel matrix. Block structure along
  the diagonal means the map is separating classes; a fairly uniform matrix means
  it is not.
- `results/kernel_model_comparison.png` -- ROC-AUC across all models.

## Real use case

Biomedical knowledge-graph link prediction:

```text
Knowledge graph (drug, gene, disease, protein nodes)
  -> node embeddings
  -> candidate pair features
  -> kernel classifier -> P(link exists)
  -> ranked hypotheses
  -> literature check or wet-lab validation
```

Applied to drug-disease repurposing, gene-disease association, protein-protein
interaction, and compound-target prediction. Kernel methods suit this well: the
datasets are often small (validated links are expensive), which is the regime
where SVMs beat deep models -- and, conveniently, the only regime where a
quadratic-cost kernel is affordable at all.

## When not to use this

- **Datasets beyond a few hundred points.** Quadratic circuit cost with no
  approximation available. 10,000 samples is 50 million circuit pairs.
- **When features have obvious classical geometry.** As measured above: if RBF
  fits your data's structure, a quantum map is a worse prior at 1000x the cost.
- **Without checking alignment first.** One matrix multiply tells you whether the
  full comparison is worth running.
- **When you need calibrated probabilities.** Precomputed-kernel SVMs give
  decision-function scores; probabilities need an extra Platt-scaling fit.

The honest summary: quantum kernels are a mathematically clean idea with a real
conjectured hardness result behind them, and no demonstrated advantage on
practical data. This tutorial is set up to show you which of those you are
looking at.

## Source papers

- Havlicek et al., "Supervised learning with quantum-enhanced feature spaces"
  (2019) -- the ZZ feature map and the quantum kernel method.
- Schuld and Killoran, "Quantum machine learning in feature Hilbert spaces"
  (2019) -- the kernel view of quantum ML.
- Huang et al., "Power of data in quantum machine learning" (2021) -- when
  quantum kernels can and cannot beat classical ones. Read this before expecting
  an advantage.
- Cristianini et al., "On Kernel-Target Alignment" (2001) -- the alignment metric.
