# Tutorial: HHL and the fine print

## Concept

HHL solves `Ax = b` in time polylogarithmic in the system size `N` — an
exponential speedup over any classical solver. It is the algorithm most often
cited as the reason quantum computers will transform scientific computing.

It is also the one whose fine print does the most work, and this tutorial is
mostly about measuring that fine print rather than admiring the speedup.

**HHL does not return `x`.** It returns a quantum state `|x⟩` whose *amplitudes*
encode the solution. The speedup survives only if all of these hold:

1. `A` is sparse and well conditioned — runtime carries `κ²`.
2. `|b⟩` can be prepared efficiently — often as hard as the original problem.
3. You need a *summary statistic* of `x`, never `x` itself.
4. `exp(iAt)` is efficiently implementable.

Break any one and the exponential advantage is gone.

## Math intuition

Three stages, and the middle one is where the inverse appears:

```text
QPE:                |b⟩|0⟩  →  Σⱼ βⱼ|uⱼ⟩|λ̃ⱼ⟩          eigenvalues into a clock register
controlled rotation: RY(2 arcsin(C/λ̃ⱼ)) on an ancilla   amplitude ∝ 1/λⱼ
inverse QPE + postselect ancilla = 1                     →  Σⱼ (βⱼ/λⱼ)|uⱼ⟩  =  |A⁻¹b⟩
```

**The eigenvalues must land exactly on clock-register values.** Phase estimation
resolves `λt/2π` as an `n`-bit fraction, so `λ·t/(2π)·2ⁿ` has to be an integer in
`1 … 2ⁿ−1`. Too large a `t` and `λ_max` wraps to phase zero, silently deleting
that component of the solution.

## Runnable implementation

```bash
python scripts/run_demo.py --algorithm hhl_intro
```

## Benchmark

`A = [[1.5, 0.5], [0.5, 1.5]]`, `b = [1, 0]`, eigenvalues 1 and 2, κ = 2, two
clock qubits. Built eigenvalue-first, so they land exactly on clock values 1 and 2.

| | Result |
| --- | --- |
| HHL solution | `[0.948683, −0.316228]` |
| Classical (`numpy.linalg.solve`) | `[0.948683, −0.316228]` |
| **Fidelity** | **1.000000000000** |
| P(ancilla = 1) | 0.625 |
| Circuit depth | 86 |

Exact, to twelve digits. Now the costs.

### Encoding: eigenvalues must be representable

Varying only the evolution time:

| `t` | Clock values | Exact? | Fidelity |
| --- | --- | --- | --- |
| 1.5708 | `[1.0, 2.0]` | **yes** | **1.000000** |
| 0.7854 | `[0.5, 1.0]` | no | 0.630555 |
| 0.5236 | `[0.333, 0.667]` | no | 0.480871 |
| 0.3927 | `[0.25, 0.5]` | no | 0.443705 |

One parameter, and fidelity falls from 1.0 to 0.44. In practice you do not know
the spectrum in advance — that is what you are trying to compute — so choosing `t`
well is a chicken-and-egg problem, and the general case is *worse* than these rows
because a real spectrum is incommensurate and no `t` makes every eigenvalue exact.

### Conditioning: the κ² penalty

| κ | Fidelity | P(success) |
| --- | --- | --- |
| 2 | 1.000000 | 0.625 |
| 4 | 0.983968 | 0.421 |
| 8 | 0.558323 | 0.098 |

At κ = 8 barely a tenth of runs survive postselection, and the surviving answer
has lost half its fidelity. This is HHL's `κ²` showing up as wasted shots.

### Readout: the caveat that dissolves the speedup

Reading all `N` amplitudes to precision `ε` costs `O(N/ε²)` measurements. For this
2-element system at 1% precision: **20,000 shots** — to solve a 2×2 system that
`numpy` solves exactly and instantly.

The scaling is `O(N)` — linear in the problem size the algorithm was supposed to
be logarithmic in. **If you read out the solution, you have thrown the speedup
away.** HHL is only interesting when you want something like `⟨x|M|x⟩`, a single
number, and never `x`.

## Classical baseline

`numpy.linalg.solve`: exact, microseconds, and unbothered by conditioning at this
size. For any system a current quantum computer can hold, the classical solver
wins on every axis.

## Real use case

Discretised PDEs are the standard motivation — a finite-difference Poisson or
heat equation is a large sparse linear system, and sparsity is exactly HHL's
requirement. See [the variational heat equation](02-variational-heat-equation.md)
for the shallower alternative.

The realistic framing: HHL is a **subroutine**, useful inside a larger quantum
algorithm where its output stays quantum. As a standalone solver it is not
competitive and is not close.

## When not to use this

- **When you want the solution vector.** Readout is `O(N)`. This is the big one.
- **When `A` is ill-conditioned.** See the κ table; 90% of runs discarded at κ = 8.
- **When `|b⟩` is not cheaply preparable.** Loading a general vector is `O(N)`,
  which again erases the advantage before the algorithm starts.
- **When you do not know the spectrum.** The encoding table shows what a bad `t`
  costs, and the general case has no good `t`.
- **On current hardware, at all.** Depth 86 for a 2×2 system, and QPE depth grows
  with the precision you need.

## Source papers

- Harrow, Hassidim, Lloyd, "Quantum algorithm for linear systems of equations"
  (PRL, 2009) — the algorithm.
- Aaronson, "Read the fine print" (Nature Physics, 2015) — the caveats above,
  stated by someone who wanted the result to be true.
- Childs, Kothari, Somma (2017) — improved dependence on precision and κ.
