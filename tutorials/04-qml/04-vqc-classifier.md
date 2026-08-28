# Tutorial: Variational Quantum Classifier

## Concept

Two quantum routes to the same supervised problem, and the difference between
them matters more than either paper suggests.

The [quantum kernel](03-quantum-kernel-biomedical-classification.md) computes
pairwise similarities and hands them to an SVM. The classical half is a **convex**
problem — one global optimum, found reliably every time. Its cost is quadratic in
dataset size and paid **once**.

A VQC trains the circuit itself. Encode the data, apply a parameterised ansatz,
measure an observable, map it to a class probability, and minimise a loss over the
circuit weights:

```text
x → feature map → trainable ansatz → ⟨Z…Z⟩ → p = (1 + ⟨Z…Z⟩)/2 → cross-entropy
```

That optimisation is **non-convex**, so VQC inherits everything VQE struggles
with: local minima, initialisation sensitivity, and barren plateaus. Its cost is
linear per epoch and paid on **every iteration**, of which there are many.

This tutorial runs both on the same data, with the same feature map, on the same
folds — so what is being compared is the learning strategy, not the encoding.

## Math intuition

The observable is the full-register parity `Z⊗Z⊗Z⊗Z`, whose expectation lies in
`[-1, 1]` and maps to a probability by `(1 + ⟨P⟩)/2`. Training minimises binary
cross-entropy over the 12 ansatz weights; the 4 feature-map parameters are data,
not trainable.

**Batching is not an optimisation detail, it is what makes this runnable.** Qiskit
V2 primitives accept a parameter *array*, so all samples go into a single pub:

```python
bound = np.hstack([features, np.tile(weights, (len(features), 1))])
estimator.run([(circuit, observable, bound)])     # 200 samples, 337 ms
```

One call for 200 samples instead of 200 calls. A test asserts the batched result
matches per-sample evaluation exactly, because a broadcasting bug here would
produce plausible-looking garbage.

The circuit has **no trainable bias**, so the decision threshold is fitted on the
training expectations (their median) and applied to test. ROC-AUC — the headline
metric — is threshold-free either way.

## Minimal example

```python
from qprac_lab.algorithms.qml.vqc_classifier import build_vqc_circuit

circuit, num_data, num_weights, observable = build_vqc_circuit(4, feature_reps=2, ansatz_reps=2)
print(num_data, num_weights)   # 4 data parameters, 12 trainable weights
```

## Runnable implementation

```bash
python scripts/download_data.py                  # ~12 MB, once
python scripts/run_demo.py --algorithm vqc_classifier
```

Takes about 4 minutes, most of it VQC training.

## Benchmark

Hetionet drug–disease link prediction, 200 pairs, 4 features, 5-fold stratified
CV. **Every model sees identical folds**, so the comparison is paired.

| Rank | Model | ROC-AUC | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| 1 | **VQC** | **0.600 ± 0.121** | 0.579 | 0.590 |
| 2 | RBF-SVM | 0.580 ± 0.099 | 0.414 | 0.525 |
| 3 | Quantum kernel SVM | 0.555 ± 0.079 | 0.514 | 0.560 |
| 4 | Random Forest | 0.471 ± 0.071 | 0.447 | 0.500 |

VQC ranks first. **Do not read that as a win.** It also has the largest standard
deviation of the four (0.121), and every model sits within one standard deviation
of every other. The paired comparison against the kernel — same folds, so
fold-to-fold noise cancels — reports `difference_exceeds_noise: false`.

Four models, all between 0.47 and 0.60 on a hard task, separated by less than
their own spread. The honest summary is a **four-way tie**, and the ranking order
is not stable information.

## The cost is not a tie

| | VQC | Quantum kernel |
| --- | --- | --- |
| Circuit evaluations | **96,000** | 19,900 |
| When paid | every optimiser iteration | once |
| Wall clock (5 folds) | 166 s | 87 s |
| Optimisation | non-convex, 600 iterations | convex, closed form |

The kernel's cost is quadratic in dataset size but **paid once**; the VQC's is
linear per iteration and paid 600 times. At this size that is 4.8× more circuit
evaluations for a statistically indistinguishable result — and the kernel gets
its global optimum by construction, where the VQC gets whatever COBYLA reached.

## Barren plateaus

The structural reason VQC training gets harder rather than easier at scale.
Measuring the variance of a loss-gradient component over random initialisations:

| Qubits | Gradient variance |
| --- | --- |
| 2 | `6.85e-02` |
| 4 | `1.66e-02` |
| 6 | `6.89e-03` |
| 8 | `1.51e-03` |

Roughly a **4× decay per two qubits added** — exponential in circuit width. At 8
qubits the gradient is already 45× smaller than at 2; extrapolate to 50 qubits and
a randomly-initialised deep ansatz has no gradient to follow at all, and no
choice of optimiser fixes that.

This is the sharpest structural difference between the two methods. The quantum
kernel has no trainable circuit, so it has no barren plateau. Whatever its own
limits are, they are not this one.

## Visualization

`result.ranking` feeds the same comparison plot as tutorial 3:

```python
from qprac_lab.visualization.tutorial_outputs import plot_model_comparison
plot_model_comparison(result.ranking, output_path="results/vqc_model_comparison.png")
```

Error bars matter here more than usual — see the spreads in the table above.

## Real use case

Same pipeline as the quantum kernel tutorial: biomedical knowledge-graph link
prediction for drug repurposing. The VQC's one structural advantage is that
inference is **cheap** — a trained circuit classifies a new point with one circuit
evaluation, where a kernel method must compute similarities against every training
point. If you train once and serve many predictions, that inverts the cost
comparison above.

## When not to use this

- **When a quantum kernel will do.** Convex, global optimum, no barren plateau,
  and here 4.8× cheaper for an indistinguishable result.
- **At scale, with a random initialisation.** The gradient variance data above is
  the whole argument.
- **When you cannot afford many iterations.** 600 optimiser steps × the training
  set, every fold. Each is a circuit execution on hardware.
- **Expecting the ranking above to hold.** It is within noise. A different seed
  could reorder the top three.

## Source papers

- Havlíček et al., "Supervised learning with quantum-enhanced feature spaces"
  (2019) — introduces both the variational classifier and the kernel method, and
  is the reason they are compared here rather than separately.
- Schuld et al., "Circuit-centric quantum classifiers" (PRA, 2020) — the VQC
  architecture used here.
- McClean et al., "Barren plateaus in quantum neural network training landscapes"
  (Nature Communications, 2018) — the effect measured above.
- Cerezo et al., "Cost function dependent barren plateaus in shallow
  parametrized quantum circuits" (2021) — when they can be avoided.
