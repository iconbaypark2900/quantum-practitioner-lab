# Prompt Library

## Generate a tutorial

Write a tutorial with these sections, in order:

```text
Concept
Math intuition
Minimal example
Runnable implementation
Classical baseline
Benchmark
Visualization
Real use case
When not to use this
Source papers
```

"When not to use this" is not optional and not a disclaimer. It is where the
measured limits go: the noise level that breaks the method, the dataset size past
which it stops being affordable, the classical method that already does the job.

## Implement a quantum algorithm

Implement it with:

- a clear algorithm type and use case
- classical baselines returned in the same result object as the quantum answer
- metrics that would show the method failing, not only succeeding
- tests that verify the thing under test can actually fail
- output artifacts

## Report a benchmark

Before reporting a win, ask:

- Is the comparison paired? Same folds, same data, same seeds?
- Is the difference larger than the run-to-run spread? Report both.
- Could the benchmark be easy for the wrong reason? Measure the shortcut.
- Does this number reproduce from the documented command? If not, publish the
  range and the cause.

## Add a paper summary

Summarise: problem, method, contribution, limitations, and how it maps to this
project. Add it to `configs/papers.yaml` as well as the module's `papers.md`.

## Investigate a suspicious result

A result that looks too good usually is. Check, in order:

1. Is the seed actually taking effect? (Aer ignores a plain `seed`.)
2. Is the circuit decomposed? (An undecomposed `PauliEvolutionGate` evaluates
   exactly and reports zero approximation error.)
3. Is a default hiding the damage? (`evaluate_duplicates`, `enforce_psd`.)
4. Is the initial state accidentally symmetric, making every configuration score
   identically?
5. Does a second framework agree? (`python scripts/run_cross_check.py`)
