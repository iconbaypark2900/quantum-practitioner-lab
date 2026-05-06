# Architecture Decisions

## Decision 1: Classical baselines are mandatory

No quantum method should be presented without a classical comparison.

## Decision 2: First three tutorials are priority implementations

The project starts with:

1. VQE for Molecular Energy
2. QAOA for Portfolio Selection
3. Quantum Kernel for Biomedical Classification

## Decision 3: PDEs are included but treated as advanced research

PDE tutorials are part of the platform, but implementation priority comes after simulation, optimization, and QML.

## Decision 4: Backend adapters are isolated

Qiskit, CUDA-Q, and IBM Runtime integrations should live under `src/qprac_lab/backends/`.

## Decision 5: Source papers are treated as project assets

Paper lists should be maintained in both Markdown and machine-readable JSON/YAML.
