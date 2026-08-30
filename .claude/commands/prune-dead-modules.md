---
description: Delete scaffold-era modules nothing imports, and tighten the reachability ratchet.
allowed-tools: Bash(git:*), Bash(python -m pytest:*), Bash(python -m ruff:*), Read, Edit, Bash(rm:*)
---

# Prune the dead scaffolds

`tests/test_module_reachability.py` records five modules that no entry point
reaches. Empty that list by deleting them.

## Why

`DECISIONS.md` #9 already set the standard for the configs: *loaded and validated,
or deleted*, because a config nobody reads is a claim with no one to contradict it.
The same rot reached the code. `circuits/ansatz.py` still carries the docstring
"Replace with Qiskit RealAmplitudes or EfficientSU2 in implementation phase" —
long after the real implementations shipped elsewhere — and it sits in a directory
the README advertises.

## The list, and what each one actually is

Read `KNOWN_DEAD` in `tests/test_module_reachability.py`. Two groups:

**Plain scaffold leftovers** — delete outright:
- `circuits/ansatz.py`, `circuits/feature_maps.py` — placeholder descriptors,
  superseded by the real Qiskit builders used in the algorithm modules.

**The more interesting group** — decide, do not just delete:
- `baselines/exact_diagonalization.py`, `baselines/classical_pdes.py`,
  `metrics/pdes.py`

These are not obviously junk. The PDE and VQE tutorials *do* compute their
classical baselines — inline, in the algorithm modules. So the modules that
advertise themselves as the baseline and metric home are never called, while
AGENTS.md's "no quantum tutorial without a classical baseline" rule is satisfied
by code living somewhere else. For each: either wire the tutorial up to the
module, or delete the module and let the inline computation stand. **Wiring is
usually the better answer for baselines** — a baseline in a named module is
easier to reuse and harder to quietly drop than one inlined in a result function.
Say which you chose and why in the commit message.

## Do

1. Work through `KNOWN_DEAD` one module at a time.
2. For each, remove the entry from `KNOWN_DEAD` **in the same change** as the
   deletion or the wiring-up. The set records outstanding debt; a stale entry
   silently widens the exemption, which is exactly what the ratchet exists to stop.
3. Leave `DOCUMENTED_PLACEHOLDERS` alone — `cudaq_adapter` and
   `ibm_runtime_adapter` are a recorded decision (DECISIONS.md #7), not rot.
4. `pytest -q` and `ruff check src tests scripts`.

## Done when

- `KNOWN_DEAD` is empty, and `test_no_module_is_orphaned` passes without it.
- The full suite is still green (214+ tests).
- `ARCHITECTURE.md`'s module map no longer lists anything that was deleted.
