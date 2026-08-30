---
description: State the project's operating range in the README, up front.
allowed-tools: Read, Edit, Bash(grep:*)
---

# Say what this does not show

Add a short "what this does not show" section near the top of `README.md`.

## Why

The project is scrupulously honest about everything except its own operating
range — and the problem is collection, not omission. Two of the four sizes are
already in the README: H2 is "parity-mapped to 2 qubits" (line 79) and Max-Cut runs
"on an 8-vertex graph" (line 146). But they sit in the tutorial write-ups where each
happens to be relevant, while the portfolio's six assets and the TFIM's four sites
appear only in the source. No single place states the range, so a reader assembles it
by accumulation or not at all — and a sceptical reader who assembles it themselves
reads the scattering as evasion. Stating it costs nothing the project has not already earned, and it is the
same move the Max-Cut tutorial already makes beautifully ("greedy also hits the
exact optimum, instantly").

## Do

1. Pull the real numbers from the source rather than repeating these — check each
   tutorial's default arguments, because the point is to be accurate:
   - qubit count per tutorial
   - problem size per tutorial (assets, vertices, sites, pairs)
   - shots, and the default optimiser budget
2. Write the section. It belongs directly after the Status table, before Quick
   start — a reader deciding whether to invest attention should hit it early.
3. Cover three limits explicitly:
   - **Scale.** Every circuit is 2–8 qubits. These are pedagogical sizes.
   - **Simulation only.** No hardware execution anywhere. The noise presets are a
     depolarizing-plus-readout *model*, not a device.
   - **What that means for the findings.** The project can say "this is how these
     algorithms behave at small scale under simulated device noise." It cannot say
     "this is how quantum methods perform." Where a finding is expected to hold at
     scale, say so and say why; where it is not, say that too.
4. Link onward to `tutorials/05-benchmarking/benchmark_methodology.md` for what
   the presets do and do not capture.

## Done when

- The section is in the README's first screen after Status.
- Every number in it was read from source this session, not carried over.
- Nothing in it apologises. It is a specification of scope, not a disclaimer.
