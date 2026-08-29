# Architecture Decisions

## 1. Classical baselines are mandatory

No quantum method is presented without a classical comparison. Every algorithm
module returns its baselines in the same result object as the quantum answer, so
they cannot be omitted from a summary.

## 2. Negative results are headline metrics, not footnotes

Where the quantum method loses or ties, the comparison that shows it is a
first-class field: `optimal_probability_lift`, `difference_exceeds_noise`,
`quantum_beats_all_classical`, `beats_random_guessing`. This is a structural
choice — a result you have to compute to see is a result that gets dropped.

## 3. Qiskit 2.x with V2 primitives only

Qiskit 2.0 removed the V1 `Estimator`/`Sampler`, and VQE/QAOA no longer ship in
qiskit core. Rather than depend on `qiskit-algorithms`, this project implements
its own optimisation loops on the V2 primitives plus `scipy.optimize`. One code
path, no version shims.

## 4. Qiskit is an optional extra

The package imports and tests without it. Quantum paths raise
`QiskitNotInstalledError` with an install hint. CI runs both configurations so the
core path cannot rot.

## 5. Backend adapters are isolated

Everything backend-specific lives in `src/qprac_lab/backends/`, including the
conventions that fail silently: Aer's `seed_simulator`, transpilation into the
noise basis, and decomposing `PauliEvolutionGate` before an optimisation loop.

## 6. Source papers are project assets

Paper lists are maintained in Markdown and in machine-readable YAML/JSON, and
every tutorial ends with its references.

## 7. IBM Runtime and CUDA-Q are dropped, not deferred

IBM Runtime needs credentials, cannot be exercised in CI, and drives the same
Qiskit stack — so it would verify nothing. CUDA-Q needs a GPU the target machine
does not have. Both were replaced by a **PennyLane cross-check**, which is free,
CI-testable, and genuinely a different implementation.

## 8. Verification is layered, and cross-framework

Every serious bug found in this project produced *plausible numbers rather than
an error*. So: property tests over exhaustive input sets rather than samples,
comparison against published literature values, and an independent PennyLane
implementation written natively rather than translated — a translation carries
over the conventions it is meant to check.

## 9. Configs are loaded and validated, or deleted

`configs/` drifted through the entire scaffold phase because nothing read it:
dropped backends stayed advertised, noise stayed marked disabled after it shipped,
and a tutorial path pointed at a renamed file. Editing them once only resets that
clock. They are now loaded by `qprac_lab.config` and cross-checked against the
code by `tests/test_configs.py`. A config nobody reads is a claim with no one to
contradict it.

## 10. Optimiser results are reported as distributions

A single run of a variational algorithm is one draw from a wide distribution. The
QAOA XY-mixer result swung from 0.1× to 100× on warm start alone while looking
perfectly reproducible across sampling seeds. Restarts are the default and the
spread is reported.

## Superseded

- **PDEs are advanced research, lower priority.** They are implemented (HHL,
  variational heat equation, Black-Scholes), each with its costs measured. The
  conclusion is that they are honest about being poor value, which is a more
  useful outcome than deferring them.
