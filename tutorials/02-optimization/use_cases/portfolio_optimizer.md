# Use Case: Quantum-Hybrid Portfolio Optimizer

## Pipeline

```text
Market data
  → returns/covariance
  → QUBO binary selection
  → QAOA or classical QUBO solver
  → classical weight optimizer
  → backtest
```

## Metrics

- Sharpe ratio
- Sortino ratio
- Max drawdown
- Objective value
- Constraint violations
