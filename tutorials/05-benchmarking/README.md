# 05 Benchmarking

Every quantum method here is measured against a classical baseline, and the
comparison is reported as it came out — including when the classical method wins.
Two of the three headline tutorials publish a quantum loss or tie.

## Start here

| Page | What it answers |
| --- | --- |
| [`benchmark_methodology.md`](benchmark_methodology.md) | How the numbers are produced, what the noise presets model, and — the load-bearing part — what they leave out |
| [`noise_benchmark.md`](noise_benchmark.md) | What actually survives simulated device noise, across all four levels |
| [`cross_framework_verification.md`](cross_framework_verification.md) | Why a second, independently written PennyLane implementation exists, and what it caught |

Read the methodology first if you intend to trust any number on the other two
pages. It is where the limits are.

## Required comparison dimensions

A benchmark is not complete until it reports all of these:

- **result quality** against a named classical baseline
- **runtime** for both sides
- **circuit depth** and two-qubit gate count — the cost the quantum side pays
- **shot count**, and whether the result is shot-limited or algorithm-limited
- **noise sensitivity** across the presets
- **run-to-run spread**, published as a range whenever a single number is not stable

## The two questions that come before reporting a win

1. **Is the comparison paired?** Same folds, same data, same seeds, same budget.
2. **Is the benchmark easy for the wrong reason?** Measure the shortcut before
   claiming the method found signal — the Hetionet loader reports
   `degree_only_roc_auc` for exactly this reason.

## Reproducing

```bash
python scripts/run_noise_sweep.py          # ~6 min, writes results/noise_sweep.*
python scripts/run_cross_check.py          # verifies key results against PennyLane
python scripts/run_benchmarks.py           # the benchmark runner
```
