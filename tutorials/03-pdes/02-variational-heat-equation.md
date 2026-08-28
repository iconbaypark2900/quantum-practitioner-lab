# Tutorial: Variational solution of the heat equation

## Concept

Every implicit timestep of a PDE is a linear system, so a PDE solver is a linear
solver in a loop. Discretising `∂u/∂t = α ∂²u/∂x²` with implicit Euler gives

```text
(I − α Δt L) uⁿ⁺¹ = uⁿ
```

one solve per timestep. [HHL](01-hhl-linear-systems-intro.md) attacks that with
phase estimation. This tutorial attacks it **variationally** (VQLS): prepare
`|ψ(θ)⟩` with an ansatz and tune `θ` until `A|ψ⟩` points along `|b⟩`.

No phase estimation, no postselection, and circuits an order of magnitude
shallower — depth 10 here against HHL's 86. In exchange you get a non-convex
optimisation, with every problem that implies.

Implicit rather than explicit Euler because it is unconditionally stable;
explicit would impose `Δt < Δx²/2α` and make the timestep a stability question
rather than an accuracy one.

## Math intuition

The VQLS cost is the misalignment between `A|ψ⟩` and `|b⟩`:

```text
C(θ) = 1 − |⟨b|A|ψ(θ)⟩|² / ⟨ψ(θ)|A†A|ψ(θ)⟩
```

Zero exactly when `A|ψ⟩` is parallel to `|b⟩`. **Parallel, not equal** — and that
distinction is the whole caveat below.

## Runnable implementation

```bash
python scripts/run_demo.py --algorithm variational_heat_equation
```

## Benchmark

3 qubits (8 grid points), `α = 0.1`, `Δt = 0.02`, `RealAmplitudes(reps=3)`:

| | Value |
| --- | --- |
| Parameters | 12 |
| Circuit depth | **10** (HHL: 86) |
| VQLS cost | `1.3e-09` |
| **Fidelity vs exact direction** | **0.999999999** |
| Objective evaluations | 1308 (5 restarts) |

The profile is recovered essentially exactly. Restarts are the default here for
the same reason as in [the QAOA tutorial](../02-optimization/02-qaoa-portfolio-selection.md):
the cost is non-convex, so a single run reports one local optimum.

## The caveat that is not a technicality

A quantum state is normalised. `|ψ⟩` encodes the **direction** of `u`, never its
magnitude — and for the heat equation the magnitude *is* the physics:

| | Value |
| --- | --- |
| `‖u‖` before the step | 1.870829 |
| `‖u‖` after the step | 1.837448 |
| Norm lost to diffusion | **0.033381** |
| Norm recovered from `\|ψ⟩` | **0.0** |

Heat leaving the system is the entire phenomenon being simulated, and the quantum
state cannot represent it. Any quantum PDE solver has to track that scale
classically on the side.

This matters beyond bookkeeping. A demo reporting "fidelity 0.9999 against the
exact solution" while quietly normalising both sides is answering an easier
question than the one posed. The fidelity above is real, and so is the zero in
that last row.

## Classical baseline

`numpy.linalg.solve` on the same matrix: exact, and `O(N)` for a tridiagonal
system with a banded solver. The 8-point problem here is solved in microseconds,
and the 1308 circuit evaluations bought a normalised copy of that answer.

## Real use case

```text
Thermal / diffusion model
  → spatial discretisation
  → implicit timestep = linear system
  → variational solve
  → track ‖u‖ classically
  → observable (average temperature, flux, hot-spot location)
```

The realistic framing is the same as HHL's: worthwhile only when you want a
*summary statistic* of `u` rather than `u`, and when the system is large enough
that the classical solve is genuinely hard. Neither is true at 8 grid points.

## When not to use this

- **When you need the magnitude**, not just the shape — see the table above.
- **When a banded solver applies.** Tridiagonal systems are `O(N)` classically;
  this is the easiest class of linear system there is.
- **Without restarts.** The cost is non-convex; one run is one local optimum.
- **At small `N`.** Everything here is faster and exact classically.

## Source papers

- Bravo-Prieto et al., "Variational Quantum Linear Solver" (Quantum, 2023) — the
  VQLS cost function used here.
- Lubasch et al., "Variational quantum algorithms for nonlinear problems"
  (PRA, 2020) — variational PDE solving more broadly.
- Cerezo et al., "Variational Quantum Algorithms" (Nature Reviews Physics, 2021).
