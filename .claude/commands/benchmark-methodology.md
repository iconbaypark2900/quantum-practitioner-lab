---
description: Write the methodology page the noise benchmark currently rests on.
allowed-tools: Read, Write, Edit, Bash(grep:*), Bash(python scripts/run_noise_sweep.py:*)
---

# Write benchmark_methodology.md

`tutorials/05-benchmarking/benchmark_methodology.md` is 73 words. It sits under
the section that produces the most transferable finding in the project.

## Why this page and not another

The noise sweep yields the one result here that generalises past its own tutorial:
**methods that need a precise number break first; methods that need only an
ordering last longest** — with the sharper corollary that a structural guarantee
is not preserved by noise (the XY mixer's exact 100% feasibility is a property of
the ideal unitary, and measures 46% at moderate device noise).

That finding is currently unreproducible by a reader, because the page explaining
what the presets actually model is a stub. `PROMPTS.md` → "Report a benchmark"
asks whether a number reproduces from the documented command. Here it does not,
for want of the document.

## Do

Read the preset definitions in `src/qprac_lab/backends/noise.py` and
`configs/noise_model.yaml` first — describe what is implemented, not what is
typical. Then cover:

1. **What a preset is.** Depolarizing plus readout error at named strengths. The
   actual one- and two-qubit error rates for light / moderate / heavy, and where
   those numbers came from.
2. **What it does not model.** Be specific and complete: no coherent errors, no
   crosstalk, no drift over a run, no leakage, no real device topology or
   connectivity constraints, no measurement of the transpilation the real backend
   would do. This list is the honest core of the page.
3. **How the sweep is run.** The command, the seeds, the shot counts, and what is
   held fixed across noise levels so the comparison is paired.
4. **Which comparisons are legitimate.** Across noise levels within one tutorial,
   yes. Across tutorials, only for *shape* — VQE error in hartree and Max-Cut
   approximation ratio are not the same kind of number, and the cross-tutorial
   claim is about how each degrades, not which is larger.
5. **Run-to-run spread.** Report it, per AGENTS.md. If a preset's result moves
   between seeds, publish the range and the cause rather than one draw.
6. **What would change the conclusions.** Name the specific thing — real device
   data — and what it would most likely change first.

## Done when

- A reader can reproduce the noise table from the documented command.
- Every number in `tutorials/05-benchmarking/noise_benchmark.md` traces to a
  method described here.
- The section README links to it, and the scope note in the root README does too.
