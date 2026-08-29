# Context

A quantum tutorial and examples platform for practitioners: every method paired
with a classical baseline, a real use case, source papers, and an explicit
statement of when *not* to use it.

## Coverage

Simulation (VQE, ADAPT-VQE, Trotterization), optimization (QAOA on portfolio
selection and Max-Cut), PDEs (HHL, variational heat equation, Black-Scholes), and
QML (quantum kernels, VQC) — each with runnable code, benchmarks, and output
artifacts.

## Stack

- **Qiskit 2.x + V2 primitives** — the only supported interface; V1 was removed in 2.0
- **Qiskit Aer** — shot sampling and device noise models
- **Qiskit Machine Learning** — fidelity quantum kernels
- **Qiskit Nature + PySCF** *(optional)* — molecular Hamiltonians at arbitrary bond lengths
- **PennyLane** *(optional)* — independent cross-check of the Qiskit results
- Local simulation only; no cloud runtime and no GPU backend

## Editorial stance

The distinguishing commitment is that **negative and null results are reported as
headline metrics**, not footnotes. As implemented:

- QAOA on the portfolio problem beats random feasible guessing by ~1.1× with the
  textbook penalty encoding.
- The quantum kernel ties RBF on real data — and the earlier "it loses" result
  turned out to be an artifact of the synthetic dataset.
- The VQC ranking is inside its own noise, and says so.
- Black-Scholes would cost ~2,000,000 circuit evaluations against one closed-form
  call.

Where a result is fragile, the fragility is measured rather than smoothed: the
QAOA XY-mixer lift ranges from 0.1× to 100× on optimiser warm start alone, so
restarts are the default and the spread is reported.

## Non-goals

- Cloud QPU execution (needs credentials, untestable in CI)
- GPU-accelerated simulation (no GPU on the target machine)
- Beating classical methods. Where quantum loses, that *is* the result.
