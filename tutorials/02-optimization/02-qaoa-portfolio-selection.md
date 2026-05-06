# Tutorial 2: QAOA for Portfolio Selection

## Use case

Quantum-hybrid portfolio optimizer.

## Algorithm type

Hybrid combinatorial optimization.

## Classical baselines

- Brute force for small N
- Greedy selection
- Simulated annealing

## Source paper

- Farhi, Goldstone, Gutmann, "A Quantum Approximate Optimization Algorithm"

## Required output

- Selected assets
- Objective value
- Constraint report
- Approximation ratio
- Baseline comparison table
- Risk/return decomposition

## Objective

```text
maximize expected_return - risk_penalty - constraint_penalty
```

Equivalent QUBO form:

```text
minimize xᵀQx
```
