# Architecture

## High-level architecture

```text
Tutorial Markdown
  → Example script / notebook
  → Algorithm module
  → Classical baseline
  → Benchmark runner
  → Visualization
  → Results artifact
```

## Source package

```text
src/qprac_lab/
  algorithms/
    simulation/
    optimization/
    pdes/
    qml/
  baselines/
  benchmarks/
  circuits/
  data/
  metrics/
  visualization/
  backends/
  papers/
```

## Backend strategy

```text
Phase 1: scaffold / classical baselines
Phase 2: Qiskit Aer implementations
Phase 3: noisy simulation
Phase 4: IBM Runtime
Phase 5: CUDA-Q
```

## Benchmark strategy

Every benchmark row should include:

- algorithm
- use case
- algorithm type
- backend
- baseline
- metric values
- runtime
- notes
