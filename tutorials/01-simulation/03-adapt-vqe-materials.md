# Tutorial: ADAPT-VQE — building the ansatz instead of guessing it

## Concept

Ordinary VQE makes you choose an ansatz up front, and the choice is a bad trade:

- A **chemically-motivated** ansatz is compact, but has to be derived by hand for
  each problem. The one-parameter UCC circuit in
  [tutorial 1](01-vqe-molecular-energy.md) is exact for H₂ — and useless for
  anything else.
- A **hardware-efficient** ansatz is generic, but spends parameters freely.
  `efficient_su2` needs **12 parameters** for a two-qubit problem that one
  parameter solves exactly.

ADAPT-VQE (Grimsley et al., 2019) removes the choice. It grows the ansatz one
operator at a time, letting the Hamiltonian decide what comes next:

1. Start from the Hartree-Fock reference.
2. For every operator `P` in a pool, compute the energy gradient at zero angle.
3. Append the operator with the largest `|gradient|`, giving it a new parameter.
4. Re-optimise **all** parameters.
5. Stop when the largest remaining gradient falls below a threshold.

The noise benchmark is why this matters rather than being merely elegant: VQE
already failed on a *depth-10, two-qubit* circuit. Every superfluous parameter is
circuit depth, and depth is exactly what current hardware cannot afford.

## Math intuition

The gradient of a Pauli rotation `exp(-i θ P / 2)` at `θ = 0` has a closed form:

```text
dE/dθ = (1/2) ⟨ψ| i[P, H] |ψ⟩
```

One expectation value per pool operator. Ranking the entire pool therefore costs
no optimisation at all — which is what makes ADAPT affordable.

A sign error here would still converge, just to the wrong ansatz, so the test
suite cross-checks this against the parameter-shift rule
`dE/dθ = [E(π/2) − E(−π/2)] / 2`, which is exact for a single-Pauli rotation.

**The pool.** Operators are Pauli strings with an **odd number of Y factors**.
Those are exactly the generators producing real rotations, which is what a real
Hamiltonian's ground state needs; even-Y generators only add phases. Weight is
capped so the pool stays 1- and 2-local rather than the full `4ⁿ`.

## Minimal example

```python
from qprac_lab.algorithms.simulation.adapt_vqe_materials import qubit_excitation_pool

print(qubit_excitation_pool(2, max_weight=2))
# ['IY', 'XY', 'YI', 'YX', 'YZ', 'ZY']
```

Only two of those six have any gradient at the Hartree-Fock state: `XY` and `YX`,
at `±0.180931`. That number is exactly the `XX` coefficient of the H₂
Hamiltonian — and it should be. `XX` is the only term that mixes determinants, so
it is the only source of correlation, and only an XY-type generator can act on it.

## Runnable implementation

```bash
python scripts/run_demo.py --algorithm adapt_vqe_materials
```

## Benchmark

H₂ / STO-3G, pool of 6 operators, gradient tolerance `1e-4`:

| Metric | ADAPT-VQE | `efficient_su2` | Hand-derived UCC |
| --- | --- | --- | --- |
| Parameters | **1** | 12 | 1 |
| Operators selected | `['XY']` | — (fixed) | — (derived by hand) |
| Error at R = 0.735 Å | `5.6e-10` | `3.4e-10` | `5.6e-10` |
| Optimiser evaluations | 23 | 268 | 23 |
| Pool-gradient scans | 12 | 0 | 0 |
| **Total expectation values** | **35** | 268 | 23 |
| Circuit depth | 10 | **8** | 10 |

The pool-scan row is ADAPT's overhead and belongs in the comparison: two scans
(one to select, one to confirm convergence) over six operators. Counting it,
ADAPT costs 35 expectation values against `efficient_su2`'s 268 — still 7.7x
cheaper — and 35 against 23 for the hand-derived ansatz, which is the price of
not having to know the answer in advance.

**ADAPT rediscovered the hand-derived ansatz.** Given only the Hamiltonian and a
generic pool, it selected exactly the generator a human derived from chemistry —
and stopped there, because after one operator every remaining gradient had fallen
to `3e-5`.

Across bond lengths, with the fixed ansatz for comparison:

| R (Å) | ADAPT operators | Params | ADAPT error | `efficient_su2` error |
| --- | --- | --- | --- | --- |
| 0.735 | `['XY']` | 1 | `5.6e-10` | `3.4e-10` |
| 1.50 | `['XY']` | 1 | `1.4e-11` | `5.8e-08` |
| 2.50 | `['XY']` | 1 | `2.8e-11` | **`2.5e-03`** |

**At the stretched bond, the fixed ansatz fails.** At R = 2.5 Å — where
correlation is strongest and Hartree-Fock is worst — `efficient_su2` exhausts its
300-evaluation budget and lands at `2.5e-03` Ha, outside chemical accuracy. ADAPT
reaches `2.8e-11` with a single parameter. Twelve parameters optimised badly beat
by one parameter chosen well.

### Two things this does *not* show

**It is not shallower here.** ADAPT's circuit is depth 10 against
`efficient_su2`'s 8. Fewer parameters did not buy a shorter circuit on a
two-qubit problem, because a single `XY` rotation still decomposes into a
comparable number of native gates. The win at this size is **optimisation cost**
(23 evaluations versus 268) and **reliability at strong correlation** — not
depth. The depth advantage is a claim about larger molecules, and this problem is
too small to demonstrate it.

**The pool decides what is reachable.** Restrict it to weight-1 operators and
every gradient is zero, so ADAPT selects nothing and returns exactly the
Hartree-Fock energy:

| Pool | Operators selected | Energy | Error |
| --- | --- | --- | --- |
| weight ≤ 2 (6 ops) | `['XY']` | `-1.137306` | `5.6e-10` |
| weight ≤ 1 (2 ops) | `[]` | `-1.116999` | `2.0e-02` |

That second row is not a bug. It is ADAPT correctly reporting that nothing in its
pool can lower the energy. A pool without the right generator cannot be rescued
by more iterations.

## Visualization

The per-iteration record — selected operator, its gradient, and the energy after
re-optimisation — is returned in `result.iterations` and is the natural thing to
plot for larger molecules, where the ansatz grows over many steps. For H₂ it is a
single row.

## Real use case

```text
Candidate material or ligand
  → Hamiltonian construction
  → ADAPT-VQE with a hardware-native operator pool
  → shortest ansatz reaching chemical accuracy
  → depth estimate → is this runnable on available hardware?
```

That last step is the point. ADAPT does not just find an energy; it tells you the
**minimum circuit** that reaches it, which is what decides whether a molecule is
tractable on a given device at all.

## When not to use this

- **When a good ansatz is already known.** For H₂ the hand-derived UCC circuit
  matches ADAPT exactly, in one step, with no pool scan.
- **On small problems.** The pool scan costs one expectation value per operator
  per iteration. Here that is cheap; the pool grows as `4ⁿ` before the weight cap,
  and gradient measurement becomes the dominant cost on real hardware.
- **When gradients are noisy.** The selection rule compares gradients that are
  `1e-4`-scale near convergence. Shot noise at 8192 shots is `~1e-2`, which would
  swamp the ranking entirely — ADAPT's operator choice is far more noise-sensitive
  than the energy it produces.
- **Expecting a depth win at small scale.** See the benchmark above.

## Source papers

- Grimsley, Economou, Barnes, Mayhall, "An adaptive variational algorithm for
  exact molecular simulations on a quantum computer" (Nature Communications,
  2019) — introduces ADAPT-VQE.
- Tang et al., "Qubit-ADAPT-VQE: An adaptive algorithm for constructing
  hardware-efficient ansätze" (PRX Quantum, 2021) — the Pauli-string pool used
  here, in place of fermionic excitations.
- Peruzzo et al., "A variational eigenvalue solver on a photonic quantum
  processor" (2014) — the VQE this builds on.
