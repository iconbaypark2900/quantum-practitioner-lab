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

---

# Maintenance Prompts

The prompts above are for building. These are for the failure mode this project
keeps rediscovering: **work that was finished, and then quietly stopped being
true.** Configs that drifted because nothing loaded them. Modules that outlived
their scaffold. Citations that lead nowhere. A stated Qiskit position with no test
watching the next release.

Each has a runnable command in `.claude/commands/`, and most have a test that
fails when the work regresses. A prompt with neither is a wish.

## Make the work reachable

Before anything else, ask what can currently see this. A repository with no
license is not open by default, it is all-rights-reserved by default — and every
judgement about who this is for is a hypothesis that cannot be tested while
nothing can reach it.

Ship the license, the description and the topics *before* the next improvement.
Improvements to something unreachable compound at zero.

→ `/license-and-publish`

## Prune a module nothing imports

`DECISIONS.md` #9 says configs are loaded and validated, or deleted. Code is not
exempt. Before deleting, ask which kind of dead it is:

- **Superseded** — a real implementation shipped elsewhere. Delete it.
- **Orphaned but right** — the thing it does is still done, just inlined somewhere
  else. Prefer wiring it back up. A baseline in a named module is harder to
  quietly drop than one buried in a result function, which is the whole point of
  the rule that every tutorial has one.
- **Deliberate placeholder** — a recorded decision, like the dropped backends.
  Leave it, and make sure its own source says so.

Delete the module and its ratchet entry in the same change, or the exemption
outlives the debt.

→ `/prune-dead-modules`, `tests/test_module_reachability.py`

## State the operating range

Every result here is 2–8 qubits on a simulator. That is the right size for
teaching and the wrong thing to leave implicit.

Say the limit where a reader meets it, not where they discover it. The project's
credibility rests on volunteering unflattering facts; an operating range found by
a sceptic in the source reads as one that was hidden, and costs more than the
disclosure ever would.

Scope is not a disclaimer. Do not apologise for it.

→ `/scope-disclosure`

## Migrate off a deprecated API

A stated version position — Qiskit 2.x, V2 primitives, no shims — is a good
position that removes the layer that would otherwise absorb a major release. It
has to be paid for with attention instead.

Deprecation warnings are a dated invoice. Read them on a schedule, not when the
build breaks. When a migration changes a circuit rather than just its
construction, re-run every published number it could touch and say which moved.

→ `/qiskit-migration`, `.github/workflows/forward-compat.yml`

## Document a benchmark's method

"Report a benchmark" asks whether a number reproduces from the documented command.
When it does not, the missing piece is usually the method page, not the number.

The load-bearing half is **what the model does not capture**. A noise preset that
lists its error rates and omits that it has no coherent error, crosstalk or drift
invites the reader to over-trust it — the same failure the project catches
everywhere else.

→ `/benchmark-methodology`

## Make a citation resolvable

An author and a year are an assertion. A DOI is a citation.

Keep one source of truth and generate the rest; a paper recorded in five places is
recorded in none. Verify every identifier resolves before committing it — a
fabricated DOI is worse than a missing one, because it looks like traceability.

→ `/backfill-papers`, `tests/test_paper_citations.py`

## Promote a simulated result to hardware

Dropping a backend as a *dependency* is not an argument against running on it
once. Credentials, CI and stack-duplication all argue against the adapter; none
argue against a one-off whose output is committed as data.

Run it paired — same ansatz, same optimiser, same shots — and record what a
simulator never has to: backend, calibration time, qubits, their error rates,
transpiled depth.

Expect the device to be worse than the pessimistic preset. **Do not retune the
presets to match it afterwards.** That converts a measurement into a fit, and the
project exists to not do that.

→ `/hardware-run`
