# Quantum Practitioner Lab

A practical quantum engineering lab for simulation, optimization, partial differential equations, quantum machine learning, benchmarking, and source-paper-driven examples.

## Naming convention

```text
Project/docs title: Quantum Practitioner Lab
Repo/folder:        quantum-practitioner-lab
Python package:     qprac_lab
CLI:                qprac-lab
Release artifact:   quantum-practitioner-lab-v0.3.0-scaffold.zip
```

A practical quantum engineering lab for simulation, optimization, partial differential equations, quantum machine learning, benchmarking, and source-paper-driven examples.

## Naming convention

```text
Project name:  Quantum Practitioner Lab
Repo/folder:   qprac_lab
Package:       qprac_lab
CLI:           qprac-lab
Release:       qprac_lab-v0.3.0-scaffold.zip
```

A comprehensive tutorial and example platform for practical quantum algorithms across:

1. Quantum simulation
2. Quantum optimization
3. Quantum partial differential equations
4. Quantum machine learning
5. Benchmarking and source-paper tracking

The project is designed for builders. Every tutorial follows:

```text
Concept
→ math intuition
→ minimal example
→ runnable implementation
→ classical baseline
→ benchmark
→ visualization
→ real use case
→ source papers
```

## Priority tutorials

The first three first-class tutorials are:

### 1. VQE for Molecular Energy

- Use case: materials discovery refinement
- Algorithm type: hybrid variational eigensolver
- Classical baseline: exact diagonalization / Hartree-Fock
- Papers: Peruzzo VQE + Grimsley ADAPT-VQE
- Required output: energy convergence plot

Run:

```bash
python scripts/run_demo.py --algorithm vqe_molecular_energy
```

### 2. QAOA for Portfolio Selection

- Use case: quantum-hybrid portfolio optimizer
- Algorithm type: hybrid combinatorial optimization
- Classical baseline: brute force, simulated annealing, greedy selection
- Paper: Farhi QAOA
- Required output: selected assets + objective value + constraint report

Run:

```bash
python scripts/run_demo.py --algorithm qaoa_portfolio_selection
```

### 3. Quantum Kernel for Biomedical Classification

- Use case: biomedical KG link prediction
- Algorithm type: kernel method / QSVC
- Classical baseline: RBF-SVM, XGBoost, Random Forest
- Paper: Havlíček quantum-enhanced feature spaces
- Required output: ROC-AUC, F1, kernel matrix visualization

Run:

```bash
python scripts/run_demo.py --algorithm quantum_kernel_biomedical
```

Generate required tutorial artifacts:

```bash
python scripts/run_first_three_tutorial_outputs.py
```

## Full module map

```text
tutorials/
  01-simulation/
  02-optimization/
  03-pdes/
  04-qml/
  05-benchmarking/

src/qprac_lab/
  algorithms/
  baselines/
  benchmarks/
  circuits/
  data/
  metrics/
  visualization/
  backends/
  papers/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pytest
python scripts/run_demo.py --algorithm qaoa_portfolio_selection
python scripts/run_all_benchmarks.py
```

## Development philosophy

Every quantum method must be compared against a classical baseline. A tutorial is not complete until it has:

- a runnable example
- a baseline
- metrics
- a real use case
- a source-paper trail
- a clear explanation of when not to use the method# quantum-practitioner-lab
