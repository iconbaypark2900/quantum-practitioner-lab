---
description: Run the two-qubit VQE once on real hardware and commit the result as data.
allowed-tools: Read, Write, Edit, Bash(python scripts/:*), Bash(git status:*)
---

# One hardware run

Run the H2 VQE on a real device once, by hand, and commit what came back.

## Why this does not contradict DECISIONS.md #7

IBM Runtime was dropped for good reasons: it needs credentials CI cannot hold, it
cannot be exercised in CI, and it drives the same Qiskit stack, so as a *backend
adapter* it would verify nothing. All three arguments are about a **dependency**.
None of them apply to a one-off run whose output is committed as data.

The payoff is the largest credibility gain available per hour of work here. It
converts the entire noise story from "simulated" to "simulated, and here is what
the device actually did" — and the circuit is two qubits, so the run itself is
minutes.

## Do

1. Use a free-tier account. **No credentials enter the repository**, and no CI job
   depends on this. If a script is needed, it reads from the environment and lives
   in `scripts/` with a docstring saying it is run by hand.
2. Run the `two_qubit_uccsd` H2 ground state at the default bond length. Keep
   everything else identical to the simulated run — same ansatz, same optimiser,
   same shot count — or the comparison is not paired, per `PROMPTS.md` → "Report a
   benchmark".
3. Record what a simulator run does not have to record: the backend name, the
   calibration timestamp, the qubits used, their reported error rates, and the
   transpiled depth. Without these the number is not reproducible even in
   principle.
4. Commit the raw result to `results/hardware_vqe_h2.json`.
5. Write the comparison into `tutorials/05-benchmarking/noise_benchmark.md`:
   measured hardware error against the light and moderate presets. State plainly
   which preset the device actually resembled.

## Expect to be wrong about the presets

The simulated result already shows VQE missing chemical accuracy at the *lightest*
preset (2.2e-3 against a 1.6e-3 threshold) on a two-qubit circuit. The hardware
number will probably be worse than "light" and may be worse than "heavy", because
the presets model depolarizing and readout error and a real device also brings
coherent error, crosstalk and drift.

**That is the finding, and it is a good one.** Report it. Do not retune the presets
to match the device afterwards — that would convert a measurement into a fit, and
the whole project exists to not do that. If the presets turn out to be optimistic,
say so in `benchmark_methodology.md` and leave them.

## Done when

- `results/hardware_vqe_h2.json` exists, with backend and calibration metadata.
- The noise benchmark states which preset the device resembled.
- No credential, token, or account id is anywhere in the repository.
- No CI job depends on hardware access.
