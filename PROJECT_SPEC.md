# Project Spec: Quantum Practitioner Lab

## Mission

Build a practical, research-grounded tutorial platform for quantum algorithms that connects theory, implementation, benchmarks, and real use cases.

## Core areas

1. Quantum simulation
2. Quantum optimization
3. Quantum PDEs
4. Quantum machine learning
5. Benchmarking
6. Source-paper library
7. Backend adapters for Qiskit, CUDA-Q, and IBM Runtime

## Tutorial standard

Every tutorial must include:

```text
Concept
Math intuition
Minimal example
Runnable implementation
Classical baseline
Benchmark
Visualization
Real use case
Source papers
```

## Priority tutorials

### Tutorial 1: VQE for Molecular Energy

- Use case: materials discovery refinement
- Algorithm type: hybrid variational eigensolver
- Baseline: exact diagonalization / Hartree-Fock
- Papers: Peruzzo VQE + Grimsley ADAPT-VQE
- Output: energy convergence plot

### Tutorial 2: QAOA for Portfolio Selection

- Use case: quantum-hybrid portfolio optimizer
- Algorithm type: hybrid combinatorial optimization
- Baseline: brute force, simulated annealing, greedy selection
- Paper: Farhi QAOA
- Output: selected assets, objective value, constraint report

### Tutorial 3: Quantum Kernel for Biomedical Classification

- Use case: biomedical KG link prediction
- Algorithm type: kernel method / QSVC
- Baseline: RBF-SVM, XGBoost, Random Forest
- Paper: Havlíček quantum-enhanced feature spaces
- Output: ROC-AUC, F1, kernel matrix visualization

## Secondary tutorials

### Simulation

- Hamiltonians and expectation values
- ADAPT-VQE for materials refinement
- Trotterization for time evolution
- Quantum phase estimation overview

### Optimization

- QAOA for Max-Cut
- QUBO and Ising mapping
- Logistics routing QUBO
- Scheduling QUBO
- Quantum annealing comparison

### PDEs

- HHL linear systems intro
- Variational heat equation
- Black-Scholes PDE demo
- Poisson equation demo

### QML

- Quantum kernels
- QSVC classifier
- VQC classifier
- Biomedical KG link prediction
- Kernel alignment

## Success criteria

A user should be able to:

1. Install the project.
2. Run the first three tutorial demos.
3. Generate output artifacts.
4. Read module-specific source papers.
5. Extend the project with real Qiskit/CUDA-Q implementations.
6. Compare each quantum method against classical baselines.
