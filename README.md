# Quantum Practitioner Lab

[![CI](https://github.com/iconbaypark2900/quantum-practitioner-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/iconbaypark2900/quantum-practitioner-lab/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.13-blue.svg)](pyproject.toml)
[![Qiskit](https://img.shields.io/badge/qiskit-2.x%20V2%20primitives-6929c4.svg)](https://www.ibm.com/quantum/qiskit)

A practical quantum engineering lab for simulation, optimization, partial
differential equations, quantum machine learning, benchmarking, and
source-paper-driven examples.

Every quantum method here is compared against a classical baseline, and the
comparison is reported honestly — including when the classical method wins.

## Naming convention

```text
Project/docs title: Quantum Practitioner Lab
Repo/folder:        quantum-practitioner-lab
Python package:     qprac_lab
CLI:                qprac-lab
```

## Status

| Area | State |
| --- | --- |
| VQE for molecular energy | **Implemented** — Qiskit V2 `Estimator`, real H2 Hamiltonian |
| QAOA for portfolio selection | **Implemented** — QUBO → Ising → `QAOAAnsatz` → sampling, with a constraint-preserving XY mixer |
| QAOA for Max-Cut | **Implemented** — the unconstrained reference problem |
| Quantum kernel classification | **Implemented** — `zz_feature_map` + `FidelityQuantumKernel`, on real Hetionet data |
| ADAPT-VQE | **Implemented** — grows the ansatz from a gradient-ranked pool |
| VQC classifier | **Implemented** — same data and folds as the quantum kernel |
| Trotterization | **Implemented** — TFIM time evolution, error scaling verified |
| PennyLane cross-check | **Implemented** — independent verification of key results |
| Device noise models | **Implemented** — depolarizing + readout, all tutorials benchmarked |
| PDEs (HHL, heat equation, Black-Scholes) | **Implemented** — with the cost caveats measured |
| IBM Runtime / CUDA-Q backends | Dropped — see `TASKS.md` |

`qprac-lab list` prints which is which. Nothing is labelled quantum unless it
actually runs a circuit.

Targets **Qiskit 2.x and its V2 primitives**. The V1 `Estimator`/`Sampler` were
removed in Qiskit 2.0, so this project does not use them.

## What this does not show

Every circuit here is small and every result is simulated. That is the right size
for teaching and the wrong thing to leave implicit, so here it is in one place.

| Tutorial | Qubits | Problem size | Shots |
| --- | --- | --- | --- |
| VQE (H2) | 2 | STO-3G, parity-mapped | exact statevector |
| ADAPT-VQE | 2 | same H2 Hamiltonian | exact statevector |
| Trotterization | 4 | TFIM, 8 steps | 4096 |
| QAOA portfolio | 6 | 6 assets, budget 3, p=3 | 4096 |
| QAOA Max-Cut | 8 | 8-vertex 3-regular graph, p=3 | 4096 |
| Quantum kernel | 4 | 200 Hetionet pairs, 4 features | 2048 |
| VQC | 4 | same 200 pairs and folds | 2048 |
| HHL | 4 | 2x2 system, 2 clock qubits | exact statevector |
| Variational heat equation | 3 | 8-point grid, 3 implicit steps | exact statevector |
| Black-Scholes | 4 | variational; 6-qubit classical FD grid | exact statevector |

Three limits follow from that, and they bound what any number here can mean.

**Scale.** 2 to 8 qubits. Everything is classically simulable by construction --
that is what makes the classical baselines computable at all, and it is why the
baselines so often win. Nothing here demonstrates quantum advantage, and results
at this size do not extrapolate.

**Simulation only.** There is no hardware execution anywhere in this repository.
The noise presets are a depolarizing-plus-readout *model*: no coherent error, no
crosstalk, no drift within a run, no leakage, and no real device topology. A
preset is a controlled approximation of a device, not a measurement of one.

**What that means for the findings.** This project can say *"here is how these
algorithms behave at small scale under simulated device noise."* It cannot say
*"here is how quantum methods perform."* Where a finding is expected to survive
scale -- shot noise binding before algorithmic error, methods needing a precise
number degrading faster than methods needing only an ordering -- the tutorial says
why. Where it is an artifact of this size, it says that too.

See [`tutorials/05-benchmarking/benchmark_methodology.md`](tutorials/05-benchmarking/benchmark_methodology.md)
for what the presets model and what they leave out.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate

# Core install: scaffolds and classical baselines only.
pip install -e ".[dev]"

# The quantum stack: required by the three implemented tutorials.
pip install -e ".[dev,qiskit]"

# Optional: real molecular Hamiltonians at any bond length (PySCF).
pip install -e ".[nature]"

# Optional: real Hetionet data for the quantum-kernel tutorial (~12 MB).
python scripts/download_data.py

pytest
python scripts/run_demo.py --algorithm qaoa_portfolio_selection
python scripts/run_benchmarks.py
```

The package stays importable and testable without the quantum extra — the
quantum tutorials skip themselves rather than failing.

## The three priority tutorials

Every tutorial follows the same arc:

```text
Concept → math intuition → minimal example → runnable implementation
→ classical baseline → benchmark → visualization → real use case
→ when not to use it → source papers
```

### 1. VQE for Molecular Energy

Ground-state energy of H2 (STO-3G, parity-mapped to 2 qubits) with a
one-parameter UCC ansatz and a `StatevectorEstimator` loop.

- Baselines: exact diagonalisation and Hartree-Fock
- Result: `-1.137306 Ha` vs exact `-1.137306 Ha` — error `5.6e-10`, well inside
  chemical accuracy, recovering 100% of the correlation energy HF misses
- Finding: shot noise, not the algorithm, is the binding constraint — at 8192
  shots the same circuit misses chemical accuracy
- Papers: Peruzzo (VQE), O'Malley (2-qubit H2), Grimsley (ADAPT-VQE)

```bash
python scripts/run_demo.py --algorithm vqe_molecular_energy
```

### 2. QAOA for Portfolio Selection

Budget-constrained mean-variance selection, encoded as a QUBO with a penalty
term, mapped to an Ising Hamiltonian, solved with `QAOAAnsatz`.

- Baselines: brute force, greedy, simulated annealing
- Result: finds the exact optimum
- Finding: with the textbook penalty encoding, only a **~1.1x lift over uniform
  sampling of feasible portfolios** — it learns feasibility and little else.
  An **XY mixer** (`mixer="xy"`) makes feasibility structural: **exactly 100%**
  at every depth, because `(XX+YY)/2` commutes with the number operator.
- Caveat, measured: optimality is a **lottery**. The same `p=6` configuration
  ranges from 0.1% to 100% probability on the optimum depending only on the
  optimiser's opening angles (s.d. 43%). `restarts=5` is therefore the default,
  and a Dicke warm start (`xy_initial_state="dicke"`) trades peak for
  reliability — a third of the variance, but never the 20x peak.
- Papers: Farhi (QAOA), Hadfield (constraint-preserving mixers)

```bash
python scripts/run_demo.py --algorithm qaoa_portfolio_selection
```

### 3. Quantum Kernel for Biomedical Classification

ZZ feature map + fidelity kernel + precomputed-kernel SVM, predicting real
`Compound–treats–Disease` edges from [Hetionet](https://het.io) (CC0).

- Baselines: RBF-SVM, Random Forest, optional XGBoost
- Evaluation: 5×4 repeated stratified CV — a single split on data this small
  swings ROC-AUC by more than the gap between the models
- Result: quantum kernel ranks first (0.587 ± 0.096 vs RBF 0.577 ± 0.076) but
  wins only 12/20 paired folds. **A statistical tie**, reported as
  `difference_exceeds_noise: false`.
- Finding: this reversed the earlier Gaussian-blob result, which had RBF winning
  clearly — that conclusion was an artifact of the generator, not the method
- Honesty check: negatives are degree-matched, so degree alone scores 0.538
  instead of 0.689. The dataset reports this rather than asserting it.
- Papers: Havlíček (quantum kernels), Himmelstein (Hetionet), Huang (when they
  help), Cristianini (alignment)

```bash
python scripts/download_data.py    # ~12 MB, once
python scripts/run_demo.py --algorithm quantum_kernel_biomedical
```

### 4. QAOA for Max-Cut

The unconstrained reference problem — positive objective, no penalty term, so the
plain approximation ratio is meaningful. Start here before the portfolio tutorial.

- Baselines: brute force, greedy, random assignment (`|E|/2`)
- Result: **0.905 expected approximation ratio** vs 0.600 for random guessing,
  with 49.7% of shots on an optimal cut
- Finding: greedy also hits the exact optimum, instantly. On an 8-vertex graph
  the quantum method has nothing to offer — which is what that size looks like.
- Papers: Farhi (QAOA), Goemans-Williamson (the 0.878 classical guarantee)

```bash
python scripts/run_demo.py --algorithm qaoa_maxcut
```

## Every tutorial under device noise

```bash
python scripts/run_noise_sweep.py     # ~6 minutes
```

| Noise | VQE error (Ha) | Max-Cut E[ratio] | XY feasibility | Kernel self-fidelity |
| --- | --- | --- | --- | --- |
| ideal | `5.6e-10` | 0.886 | **100%** | 1.000 |
| light | `2.2e-03` | 0.877 | 82.7% | 0.936 |
| moderate | `1.1e-02` | 0.824 | 46.2% | 0.734 |
| heavy | `3.7e-02` | 0.756 | 33.2% | 0.388 |

Three findings worth the run:

- **VQE is the most fragile thing here.** Even the optimistic preset misses
  chemical accuracy (`2.2e-3` vs a `1.6e-3` threshold) on a *two-qubit* circuit.
- **Max-Cut is the most robust**, losing only 15% of its ideal quality at heavy
  noise. The pattern: methods needing a precise *number* break first; methods
  needing only an *ordering* last longest.
- **A structural guarantee is not preserved by noise.** The XY mixer's exact 100%
  feasibility is a property of the ideal unitary; it measures 46% at moderate
  device noise, and its lift falls below random guessing at heavy.

Full write-up: [`tutorials/05-benchmarking/noise_benchmark.md`](tutorials/05-benchmarking/noise_benchmark.md).

## Generating artifacts

```bash
python scripts/run_first_three_tutorial_outputs.py
```

Writes to `results/`:

- `vqe_energy_convergence.png` — convergence with exact/HF reference lines and a
  log-scale error panel
- `vqe_dissociation_curve.png` — H2 potential energy surface; needs `[nature]`
- `qaoa_sampling_distribution.png` — sampled bitstrings with the
  uniform-over-feasible line
- `portfolio_constraint_report.json`
- `kernel_matrix.png`, `kernel_model_comparison.png`
- `first_three_tutorial_outputs.json`

## Notebooks

Executable walkthroughs for every implemented tutorial, with outputs committed so
they read on GitHub without running anything:

| Notebook | Covers |
| --- | --- |
| [`01-vqe-molecular-energy`](notebooks/01-vqe-molecular-energy.ipynb) | Hamiltonian, both baselines, convergence, dissociation curve, noise |
| [`02-qaoa-portfolio-selection`](notebooks/02-qaoa-portfolio-selection.ipynb) | QUBO → Ising, penalty vs XY mixer, depth sweep, noise |
| [`03-quantum-kernel-biomedical-classification`](notebooks/03-quantum-kernel-biomedical-classification.ipynb) | Hetionet data, leakage and degree checks, cross-validated comparison, self-fidelity |
| [`04-qaoa-maxcut`](notebooks/04-qaoa-maxcut.ipynb) | The unconstrained reference problem — **start here** |
| [`05-hhl-linear-systems-intro`](notebooks/05-hhl-linear-systems-intro.ipynb) | Exact solve, then every caveat measured: encoding, conditioning, readout |
| [`06-variational-heat-equation`](notebooks/06-variational-heat-equation.ipynb) | VQLS at 1/8 the depth of HHL, and the norm it cannot represent |

```bash
jupyter lab notebooks/
```

## Independent verification

```bash
pip install -e ".[pennylane]"
python scripts/run_cross_check.py
```

Every serious bug found while building this produced *plausible numbers rather
than an error* — an ignored seed, an undecomposed gate returning the exact
answer, an assumed kernel diagonal. A second, unrelated stack is the cheapest
thing that catches those. VQE agrees to `4e-16` across the dissociation range;
the QUBO→Ising mapping agrees over all 64 assignments to `1.8e-14`.

Details: [`tutorials/05-benchmarking/cross_framework_verification.md`](tutorials/05-benchmarking/cross_framework_verification.md).

## CLI

```bash
qprac-lab list                                  # demos and implementation level
qprac-lab env                                   # installed quantum stack versions
qprac-lab cross-check                           # verify results against PennyLane
qprac-lab demo --algorithm vqe_molecular_energy
```

## Full module map

```text
tutorials/
  01-simulation/  02-optimization/  03-pdes/  04-qml/  05-benchmarking/

src/qprac_lab/
  algorithms/{simulation,optimization,pdes,qml}/
  backends/       # Qiskit V2 adapter, noise presets, PennyLane cross-check
  baselines/      # exact diagonalisation, classical optimisation, classical ML
  benchmarks/     # runner and result schema
  circuits/  data/  metrics/  papers/  visualization/
```

## Development philosophy

A tutorial is not complete until it has a runnable example, a classical
baseline, metrics, a real use case, a source-paper trail, and a clear
explanation of when *not* to use the method.

The last one is load-bearing. Two of the three implemented tutorials show the
quantum method failing to beat its classical baseline. That is the useful
result, and burying it would make the other one worthless.
