# Use Case: Financial PDE Pricing

## Pipeline

```text
Market inputs
  → PDE model (Black-Scholes)
  → discretization or variational residual
  → quantum solver prototype
  → compare against finite difference / closed form
```

## What is implemented

[Black-Scholes PDE](../03-black-scholes-pde.md): implicit time stepping on a
discretised grid, with the linear solve at each step done variationally, checked
against both a finite-difference solver and the closed-form European call price.

## The cost, stated plainly

| Method | Work for a full solve |
| --- | --- |
| Closed-form Black-Scholes | **1 evaluation**, microseconds |
| Finite difference | ~200 tridiagonal solves, milliseconds |
| Variational quantum | **≈2,000,000 circuit evaluations** |

That is the result. A European option has an analytic price, so the numerical
solver is unnecessary before the quantum one is; the tutorial exists to make the
comparison concrete rather than to propose the method.

## Two structural problems this use case has to face

**Readout.** The solution lives in a normalised statevector's amplitudes.
Extracting the price at every grid point costs a number of measurements that
scales with the grid, which erases the speedup that motivated the approach.

**The norm.** A statevector cannot represent the solution's scale — only its
direction. Price *level* must be tracked classically alongside, and the
[heat equation tutorial](../02-variational-heat-equation.md) measures exactly how
much norm each diffusion step loses (`0.033381`).

## Where it could plausibly matter

Path-dependent, high-dimensional derivatives with no closed form, where classical
methods fall back on Monte Carlo and the dimension is the problem. Nothing in this
repository reaches that regime, and this page does not claim otherwise.
