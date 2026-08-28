# Quantum Practitioner Lab

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
| Quantum kernel classification | **Implemented** — `zz_feature_map` + `FidelityQuantumKernel` |
| PDEs, ADAPT-VQE, Trotter, VQC | Classical scaffolds only |
| IBM Runtime / CUDA-Q backends | Placeholders |

`qprac-lab list` prints which is which. Nothing is labelled quantum unless it
actually runs a circuit.

Targets **Qiskit 2.x and its V2 primitives**. The V1 `Estimator`/`Sampler` were
removed in Qiskit 2.0, so this project does not use them.

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
- Finding: with the textbook penalty encoding, only a **1.05x lift over uniform
  sampling of feasible portfolios** — it learns feasibility and little else.
  Switching to an **XY mixer** (`mixer="xy"`) makes feasibility structural:
  **exactly 100%** at every depth, and up to a 20x lift on optimality. Reported
  by default so neither result can be oversold.
- Papers: Farhi (QAOA), Hadfield (constraint-preserving mixers)

```bash
python scripts/run_demo.py --algorithm qaoa_portfolio_selection
```

### 3. Quantum Kernel for Biomedical Classification

ZZ feature map + fidelity kernel + precomputed-kernel SVM for KG link prediction.

- Baselines: RBF-SVM, Random Forest, optional XGBoost
- Result: **RBF-SVM wins** (ROC-AUC 0.826 vs 0.694), at ~1675x less compute
- Finding: kernel-target alignment (0.0906 quantum vs 0.1356 RBF) predicted the
  loss before any classifier was fitted — one matrix multiply as a go/no-go check
- Papers: Havlíček (quantum kernels), Huang (when they help), Cristianini
  (alignment)

```bash
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

## CLI

```bash
qprac-lab list                                  # demos and implementation level
qprac-lab env                                   # installed quantum stack versions
qprac-lab demo --algorithm vqe_molecular_energy
```

## Full module map

```text
tutorials/
  01-simulation/  02-optimization/  03-pdes/  04-qml/  05-benchmarking/

src/qprac_lab/
  algorithms/{simulation,optimization,pdes,qml}/
  backends/       # Qiskit V2 primitive adapter; CUDA-Q and Runtime placeholders
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
