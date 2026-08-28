# Tutorial: Black-Scholes as a PDE — and the cost of solving it quantumly

## Concept

The Black-Scholes equation

```text
∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S − rV = 0
```

is a convection-diffusion PDE. Discretise in `S`, step backwards from the payoff
with implicit Euler, and each timestep is a linear system — so the
[variational solver](02-variational-heat-equation.md) applies directly.

**This tutorial exists to make a cost argument, not to recommend the method.**
European options have a *closed-form solution* computable in microseconds. A
numerical PDE solve is already unnecessary here; a quantum one is unnecessary
twice over. What is worth having is the arithmetic.

## Runnable implementation

```bash
python scripts/run_demo.py --algorithm black_scholes_pde
```

## Benchmark

European call, `K = 100`, `T = 1`, `r = 0.05`, `σ = 0.2`, 64 grid points, 200
implicit timesteps:

| Spot | Finite difference | Closed form | Rel. error |
| --- | --- | --- | --- |
| 80 | 1.8746 | 1.8594 | 0.82% |
| 100 | 10.3886 | 10.4506 | 0.59% |
| 120 | 26.1556 | 26.1690 | 0.05% |
| 150 | 54.9724 | 54.9701 | 0.00% |

The finite-difference scheme is correct — sub-1% against the analytic price, with
the residual being discretisation error that shrinks with grid and timestep
refinement.

One variational timestep on a 16-point grid reaches fidelity **0.9987** against
the exact solve of that same step, at a cost of **10,000 circuit evaluations**.

## The number this tutorial is for

| Method | Cost |
| --- | --- |
| Closed-form formula | **1 evaluation, microseconds** |
| Finite difference | 200 tridiagonal solves, milliseconds |
| Variational, per timestep | 10,000 circuit evaluations |
| **Variational, full solve** | **≈ 2,000,000 circuit evaluations** |

Two million circuit evaluations against one call to a formula, for a worse answer.

That ratio is the honest content of "quantum computing for finance" applied to
vanilla option pricing, and it does not improve with better engineering — the
closed form is not going to get slower.

## Where quantum finance actually argues an advantage

Not here, and the distinction is routinely lost in the literature.

The real claim is **amplitude estimation for Monte Carlo pricing** of
path-dependent derivatives — Asian options, barriers, portfolio risk measures —
where no closed form exists and classical Monte Carlo converges as `1/√M`.
Amplitude estimation converges as `1/M`, a **quadratic** speedup on sampling.

That is a different algorithm, a defensible claim, and a much more modest one
than "exponential speedup for PDEs". A quadratic speedup on a genuinely hard
sampling problem is worth more than an exponential speedup on a problem with a
closed-form answer.

## Classical baseline

The Black-Scholes formula itself, via `scipy.stats.norm`. Exact, closed-form, and
the reason every number above is checkable.

## When not to use this

- **For vanilla European options.** There is a closed form. This is the whole point.
- **For American options**, where early exercise makes it a free-boundary problem
  the linear-system formulation does not capture.
- **When you want the price**, a single number: reading one amplitude out of a
  normalised state loses the price *level*, which must be tracked classically.
- **As evidence for quantum advantage in finance.** Point at amplitude estimation
  for path-dependent Monte Carlo instead — a real claim, honestly quadratic.

## Source papers

- Black and Scholes, "The Pricing of Options and Corporate Liabilities" (1973).
- Stamatopoulos et al., "Option Pricing using Quantum Computers" (Quantum, 2020)
  — amplitude estimation for derivative pricing, i.e. the approach that actually
  has an argument.
- Chakrabarti et al., "A Threshold for Quantum Advantage in Derivative Pricing"
  (Quantum, 2021) — what hardware would have to reach for that to pay off.
- Aaronson, "Read the fine print" (2015).
