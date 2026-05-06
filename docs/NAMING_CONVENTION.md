# Naming Convention

## Final convention

```text
Project/docs title: Quantum Practitioner Lab
Repo/folder:        quantum-practitioner-lab
Python package:     qprac_lab
CLI:                qprac-lab
Release artifact:   quantum-practitioner-lab-v0.3.0-scaffold.zip
```

## Versioned artifacts

```text
quantum-practitioner-lab-v0.1.0-initial-scaffold.zip
quantum-practitioner-lab-v0.2.0-first-three-tutorials.zip
quantum-practitioner-lab-v0.3.0-scaffold.zip
quantum-practitioner-lab-v0.4.0-qiskit.zip
quantum-practitioner-lab-v0.5.0-backends.zip
quantum-practitioner-lab-v1.0.0-complete-platform.zip
```

## Version meaning

```text
v0.1.0 = initial scaffold
v0.2.0 = first three tutorials
v0.3.0 = full comprehensive scaffold
v0.4.0 = Qiskit implementations
v0.5.0 = CUDA-Q / IBM Runtime adapters
v1.0.0 = complete tutorial platform
```

## Tutorial folders

```text
tutorials/
  01-simulation/
  02-optimization/
  03-pdes/
  04-qml/
  05-benchmarking/
```

## Tutorial files

Use kebab-case with numeric ordering:

```text
01-vqe-molecular-energy.md
02-qaoa-portfolio-selection.md
03-quantum-kernel-biomedical-classification.md
```

## Source package

```text
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

## Algorithm IDs

Use snake_case for machine-readable IDs:

```text
vqe_molecular_energy
qaoa_portfolio_selection
quantum_kernel_biomedical
hhl_linear_systems_intro
variational_heat_equation
black_scholes_pde
adapt_vqe_materials
qaoa_maxcut
qubo_ising_mapping
qsvc_classifier
vqc_classifier
```

## Config files

```text
project.yaml
algorithms.yaml
tutorials.yaml
benchmarks.yaml
backends.yaml
qiskit.yaml
cudaq.yaml
papers.yaml
```
