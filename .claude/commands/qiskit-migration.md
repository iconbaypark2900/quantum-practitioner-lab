---
description: Move off the deprecated Qiskit constructors before the next major removes them.
allowed-tools: Bash(python -m pytest:*), Bash(python -m ruff:*), Bash(grep:*), Read, Edit
---

# Migrate off the deprecated Qiskit APIs

The suite passes on Qiskit 2.5.2 today and emits deprecation warnings for APIs
scheduled for removal in 3.0. Move the call sites before the removal, not after.

## Why this one carries extra weight

`DECISIONS.md` #3 makes version discipline a *stated position*: Qiskit 2.x and its
V2 primitives, own optimisation loops on top of `scipy.optimize`, no
`qiskit-algorithms` dependency, one code path, no shims. That is a good position
and it has a cost — there is no shim layer to absorb a major release. Being caught
flat-footed by 3.0 would undercut the claim the project makes loudest.

## Status: the project side is done

Executed 2026-08-30 against Qiskit 2.5.2. All three call sites migrated; project-code
Qiskit deprecations went 5 → 0, the suite stayed at 232 passed, every published number
held (HHL bit-identical at depth 86 / 40 two-qubit gates), and the QAOA path got 11%
faster with half the gate count.

**One deprecation survives and is not ours.** `qiskit-nature` constructs a
`BlueprintCircuit` inside `second_q/circuit/library/initial_states/hartree_fock.py:53`.
Nothing in this repository can fix that — it needs a qiskit-nature release. It is
reached only through the optional `[nature]` extra, so the core and `[qiskit]` installs
are already 3.0-clean, but the dissociation-curve path will break on Qiskit 3.0 until
upstream moves. Watch it via the `forward-compat` workflow rather than re-deriving it.

Re-run the steps below if a future Qiskit release deprecates something new.

## The call sites

Confirm the current list rather than trusting this one — run
`pytest -q -W "always::DeprecationWarning"` and read the warnings summary, or pull
the `deprecation-inventory` artifact from the `forward-compat` workflow.

As of this writing:

| Deprecated | Since | Where |
| --- | --- | --- |
| `EfficientSU2`, `RealAmplitudes` (classes) | 2.1 | `algorithms/simulation/vqe_molecular_energy.py` |
| `NLocal` / `BlueprintCircuit` behind `QAOAAnsatz` | 2.1 | `algorithms/optimization/qaoa_portfolio_selection.py` |
| `Gate.control(annotated=None)` | 2.3 | `algorithms/pdes/hhl_intro.py` |

## Do

1. **The pattern is already in the repo.** `algorithms/qml/quantum_kernel_biomedical.py`
   uses the functional `zz_feature_map` rather than the deprecated `ZZFeatureMap`
   class. Follow it: `efficient_su2(...)`, `real_amplitudes(...)`.
2. For `Gate.control`, pass `annotated=` explicitly. Read the new default
   (`annotated=True`) before choosing — it defers construction to the transpiler,
   which changes the circuit the HHL tutorial builds. **Check the tutorial's
   reported depth and its measured solution error both before and after**; this is
   the one change here that can move a published number.
3. `QAOAAnsatz` → `qaoa_ansatz(cost_operator, reps=...)`. Same class-to-function
   move as step 1, and the biggest win of the three. Measured here on a 4-qubit Ising
   operator at `reps=2`: identical parameter count, depth 8 against 18 for
   `QAOAAnsatz(...).decompose(reps=3)`, and zero deprecation warnings. The functional
   form returns a flattened `QuantumCircuit` of `{h, rx, rzz}` — no
   `PauliEvolutionGate` anywhere in it.
4. **That makes the `.decompose(reps=3)` at `qaoa_portfolio_selection.py:145`
   unnecessary**, because there is no longer an unsynthesised gate to re-expand on
   every estimator call. Confirm it on the real cost operator before deleting the
   call, and keep the measurement in a comment either way: 2.56 s versus 0.006 s per
   call is *why* it was there, and a later refactor that reintroduces an
   unsynthesised gate needs to be able to find that reasoning.

## Done when

- `pytest -q -W "error::DeprecationWarning"` surfaces no *Qiskit* deprecations.
  (Other libraries' warnings are not this task.)
- The `forward-compat` pre-release job is green.
- Every published number that the HHL change could touch has been re-run and
  either matches or has been updated in both the tutorial and `results/`.
