# Use Case: Quantum-Hybrid Portfolio Optimizer

## Pipeline

```text
Market data
  → returns/covariance
  → QUBO binary selection      ← the only quantum step
  → classical weight optimizer
  → backtest
```

The quantum part is **selection**, not allocation. Which assets to hold is a
binary problem and maps to a QUBO; how much of each to hold is continuous and
belongs to a classical optimiser. Conflating the two is the most common way this
use case gets oversold.

## What is implemented

[QAOA for portfolio selection](../02-qaoa-portfolio-selection.md): 6 assets,
budget 3, mean-variance objective, against brute force, greedy and simulated
annealing baselines.

## What it measures

| Metric | Why it is the one that matters |
| --- | --- |
| `optimal_probability_lift` | QAOA returns a *distribution*. A uniform draw over feasible portfolios already hits the optimum with probability `1/C(n,k)`. Beating that is the bar. Measured: **~1.1x** with a penalty encoding. |
| `feasible_probability` | With an XY mixer, **100%** by construction — `(XX+YY)/2` commutes with the number operator. Under moderate device noise, **46%**. |
| `restart_objectives` | Optimality is a lottery: the same `p=6` run ranged 0.1% to 100% on opening angles alone, s.d. 43 points. `restarts=5` is the default and every attempt is recorded. |

Sharpe, Sortino and max drawdown belong to the *backtest* stage, which this
repository does not implement — reporting them here would attribute portfolio
performance to a selection step that beat random guessing by 1.1x.

## When not to use it

At 6 assets, brute force is instant and exact. The interesting question is not
whether QAOA wins at this size — it does not — but whether the penalty encoding
ever stops flattening the distribution as size grows. Nothing here answers that.
