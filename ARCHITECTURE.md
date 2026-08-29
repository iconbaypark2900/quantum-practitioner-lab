# Architecture

## High-level flow

```text
Tutorial markdown
  → algorithm module (src/qprac_lab/algorithms/…)
  → classical baseline (src/qprac_lab/baselines/…)
  → backend adapter (Qiskit V2 primitives)
  → benchmark runner + metrics
  → visualization
  → results artifact
```

## Source package

```text
src/qprac_lab/
  algorithms/
    simulation/     VQE, ADAPT-VQE, Trotterization, Hamiltonian construction
    optimization/   QAOA portfolio, QAOA Max-Cut, QUBO/Ising builder
    pdes/           HHL, variational heat equation, Black-Scholes
    qml/            quantum kernel, VQC
  backends/         Qiskit V2 adapter, noise presets, PennyLane cross-check
  baselines/        exact diagonalisation, classical optimisation, classical ML
  benchmarks/       runner and result schema
  circuits/         mixers (XY, transverse field), ansatz and feature-map helpers
  config.py         loads configs/, validated against the code by tests
  data/             synthetic generators and the Hetionet loader
  metrics/          classification, optimisation, PDE metrics
  papers/           source-paper registry
  visualization/    the required output artifacts
```

## Backend strategy

Qiskit **2.x with V2 primitives only**. Qiskit 2.0 removed the V1
`Estimator`/`Sampler`, and VQE/QAOA no longer ship in qiskit core, so this project
implements its own optimisation loops on top of the primitives plus
`scipy.optimize` rather than depending on `qiskit-algorithms`.

| Backend | Status |
| --- | --- |
| `statevector` | Exact expectation values; the default everywhere |
| `aer` | Shot sampling and device noise models |
| `pennylane` | Independent cross-check, not a runtime |
| IBM Runtime | **Dropped** — needs credentials, untestable in CI, and drives the same Qiskit stack, so it would verify nothing |
| CUDA-Q | **Dropped** — no GPU access on the target machine |

`QiskitBackendAdapter` owns three things that fail silently if skipped: Aer's
`seed_simulator` convention, transpilation into the noise basis, and decomposing
`PauliEvolutionGate` before it enters an optimisation loop.

## Optionality

Qiskit is an **extra, not a core dependency**. The package imports and tests
without it; quantum paths raise `QiskitNotInstalledError` with an install hint
rather than an `ImportError` from deep in a call stack. CI runs both
configurations, so the no-quantum-stack path cannot rot unnoticed.

## Benchmark row schema

Every benchmark row carries: algorithm, use case, algorithm type, backend,
runtime, metric values, and the full payload. `run_benchmarks.py` writes flat CSV
for reading and full JSON for analysis.

## Verification layers

1. **Unit and property tests** — e.g. the QUBO→Ising mapping over every `2ⁿ`
   assignment, not a sample.
2. **Literature values** — H₂ energies checked against published FCI numbers, not
   only against the code's own output.
3. **Cross-framework** — VQE and the Ising mapping re-run through PennyLane,
   written natively rather than translated.
4. **Config validation** — `configs/` cross-checked against the code, because
   files nobody loads drift silently.
