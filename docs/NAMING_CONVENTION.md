# Naming Convention

## Final convention

```text
Project/docs title: Quantum Practitioner Lab
Repo/folder:        quantum-practitioner-lab
Python package:     qprac_lab
CLI:                qprac-lab
Release artifact:   quantum-practitioner-lab-v0.3.0.zip
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
adapt_vqe_materials
trotterization
qaoa_portfolio_selection
qaoa_maxcut
quantum_kernel_biomedical
vqc_classifier
hhl_intro
variational_heat_equation
black_scholes_pde
logistics_routing_qubo        # still a scaffold
```

These are the keys of `qprac_lab.demo_registry.DEMOS`; `qprac-lab list` prints
them with their implementation level.

## Config files

```text
project.yaml
algorithms.yaml
tutorials.yaml
benchmarks.yaml
backends.yaml
experiment.yaml
noise_model.yaml
qiskit.yaml
papers.yaml
```

Loaded by `qprac_lab.config` and cross-checked against the code by
`tests/test_configs.py`, which also fails if a new YAML is added without being
registered.
