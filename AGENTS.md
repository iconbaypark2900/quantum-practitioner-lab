# Agent Guide

## Roles

### Tutorial Writer
Writes tutorials to the nine-part standard: concept, math intuition, minimal
example, runnable implementation, classical baseline, benchmark, visualization,
real use case, **when not to use it**, source papers.

### Algorithm Engineer
Implements modules under `src/qprac_lab/algorithms`, returning a dataclass that
carries the quantum result *and* its baselines in one object.

### Benchmark Engineer
Adds metrics, runners, and result schemas. Reports distributions, not single runs.

### Research Curator
Maintains the paper references in Markdown and in `configs/papers.yaml`.

### Backend Engineer
Maintains `src/qprac_lab/backends/`, including the library conventions that fail
silently if unhandled.

## Rules

- No quantum tutorial without a classical baseline.
- No algorithm without a source-paper reference.
- No benchmark without a result schema.
- Keep examples small enough to run locally.
- **Report the result you got.** If the quantum method loses or ties, that is the
  finding; put it in the headline metric, not a footnote.
- **Make the honest comparison, not the flattering one.** Check whether a
  benchmark is easy for the wrong reason before reporting a win — see the
  degree-shortcut check in the Hetionet loader.
- **Publish reproducible numbers.** If a value is not stable, publish the range
  and the cause.
- **Assume a silent failure is possible.** Every serious bug here produced
  plausible numbers rather than an error. Test that the thing under test can
  actually fail.
