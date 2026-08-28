# Tutorial 2: QAOA for Portfolio Selection

## Concept

Pick exactly `B` assets from `n` to maximise return minus risk. The search space
is combinatorial, and the constraint makes it worse: `C(6,3) = 20` here, but
`C(50,25)` is about `1.3e14`.

QAOA prepares a superposition over all `2^n` candidate portfolios, then applies
`p` alternating layers of a *cost* operator (which phases each bitstring by its
objective) and a *mixer* (which moves amplitude between bitstrings). Classical
optimisation of the `2p` layer angles is supposed to concentrate amplitude on
good solutions. You then sample.

The critical thing to internalise: **QAOA returns a probability distribution,
not an answer.** Reporting "the best of 4096 samples" while ignoring the shape of
that distribution is how QAOA demos end up looking better than they are.

## Math intuition

Three translations, each of which can silently break the result.

**1. Constrained problem to QUBO.** An Ising Hamiltonian has no notion of
"subject to", so the cardinality constraint becomes a penalty:

```text
maximise   mu^T x - lambda * x^T Sigma x     subject to sum(x) = B
minimise  -mu^T x + lambda * x^T Sigma x + P * (sum(x) - B)^2
```

Expanding the penalty and using `x_i^2 = x_i` (true for binary variables) gives

```text
Q[i,i] = -mu_i + lambda * Sigma[i,i] + P - 2PB
Q[i,j] =         lambda * Sigma[i,j] + P          (i != j)
offset =  P * B^2
```

**2. QUBO to Ising.** Substitute `x = (1 - z) / 2`, where `z` is the +/-1
eigenvalue of a Pauli-Z:

```text
constant = sum_i Q[i,i]/2 + sum_{i<j} S[i,j]/2
h_i      = -Q[i,i]/2 - sum_{j!=i} S[i,j]/2           S = (Q + Q^T)/2
J_ij     = S[i,j]/2
```

The resulting operator contains only `I` and `Z` terms -- a classical cost
function must map to a diagonal Hamiltonian. If yours has `X` or `Y` in it, the
mapping is wrong.

**3. Bitstrings back to selections.** Qiskit prints bitstrings little-endian:
the leftmost character is the *highest*-index qubit. Reversing this silently
produces a mirror-image portfolio that still looks plausible. It is handled once
in `QUBO.bitstring_to_selection` rather than at each call site.

The test suite verifies step 2 exhaustively -- every one of the `2^n` assignments
must give the same energy under the QUBO and under the mapped Ising operator.

## Minimal example

```python
from qprac_lab.algorithms.optimization.qubo_builder import portfolio_qubo
from qprac_lab.data.synthetic import make_small_portfolio_dataset

mu, sigma = make_small_portfolio_dataset(n_assets=6)
qubo = portfolio_qubo(mu, sigma, budget=3, risk_lambda=0.5)
operator, offset = qubo.to_ising()
print(operator.num_qubits, len(operator))     # 6 qubits, 21 terms
print(qubo.brute_force()["selection"])        # [1 0 1 1 0 0] -- feasible
```

## Runnable implementation

```bash
python scripts/run_demo.py --algorithm qaoa_portfolio_selection
```

## Classical baselines

| Baseline | Selection | Objective |
| --- | --- | --- |
| Brute force (exact) | `[1,0,1,1,0,0]` | `+0.333631` |
| Simulated annealing | `[1,0,1,1,0,0]` | `+0.333631` |
| Greedy (top-3 by return) | `[1,0,1,0,0,1]` | `-0.305651` |

Greedy fails badly, and instructively: ranking by return alone ignores the
covariance term entirely, so it happily picks three correlated assets.

## Benchmark

`n = 6`, `budget = 3`, 4096 shots, statevector simulator:

| Penalty | p | Feasible | Hits optimum | Uniform baseline | Lift |
| --- | --- | --- | --- | --- | --- |
| 6.17 (auto) | 3 | 99.3% | 5.59% | 5.00% | **1.12x** |
| 6.17 (auto) | 6 | 86.0% | 7.06% | 5.00% | 1.41x |
| 2.00 | 6 | 98.8% | 5.52% | 5.00% | 1.10x |
| 0.50 | 6 | 77.2% | 9.23% | 5.00% | **1.85x** |

Reproduce with `run_qaoa_portfolio_selection_tutorial(penalty=..., reps=...)`;
all runs use the default `seed=42`. Every row found the exact optimum among its
samples — the differences are entirely in how much probability mass landed there.

The `Uniform baseline` column is the honest comparison and is the reason it is
reported by default. There are 20 feasible portfolios, so **guessing a feasible
portfolio at random hits the optimum 5% of the time.** Any QAOA configuration
scoring near 5% has learned feasibility and nothing else.

At the safe default penalty, QAOA achieves 99.3% feasibility and a lift of
**1.12x** -- it is barely better than random guessing among feasible states. The
sampling-distribution plot shows this directly: every bar sits on the uniform
line.

This is a real tension, not a tuning failure:

- **A large penalty dominates the cost landscape.** Feasibility is worth `P`
  while the entire spread of objective values across feasible portfolios is about
  `2.1`. With `P = 6.17`, QAOA spends its expressive power on the constraint and
  is left nearly flat over what remains.
- **A small penalty sharpens the distribution and breaks feasibility.** At
  `P = 0.5` the lift rises to 1.85x, but 23% of samples violate the budget.

Both configurations *find* the optimum among their samples. Neither concentrates
much probability on it.

## Visualization

- `results/qaoa_sampling_distribution.png` -- sampled bitstrings coloured by
  feasibility, with the uniform-over-feasible line drawn in. This is the plot
  that tells you whether QAOA worked.
- `results/portfolio_constraint_report.json` -- budget compliance of the
  returned selection.

## Real use case

```text
Asset universe
  -> covariance estimation
  -> cardinality-constrained mean-variance QUBO
  -> QAOA / annealer / classical solver
  -> feasibility filter
  -> risk report and human review
```

The feasibility filter is not optional. Penalty-encoded constraints are soft, so
a fraction of returned portfolios will violate the budget and must be discarded
before anything downstream sees them.

## When not to use this

- **When brute force fits.** At `n = 6` exhaustive search takes microseconds and
  is exact. QAOA is only interesting past roughly `n = 30`, where the classical
  comparison is a good heuristic rather than an exact solver.
- **When simulated annealing already ties.** It matched the exact optimum here
  in milliseconds. That is the bar a real quantum advantage claim has to clear,
  not brute force.
- **Hard constraints.** Penalty encoding makes constraints soft. If a violated
  budget is unacceptable rather than merely undesirable, you need a
  constraint-preserving mixer (XY mixers keep the state in the fixed-cardinality
  subspace) or a different method.
- **Shallow depth.** `p = 3` on a strongly-penalised landscape is close to
  uniform sampling. QAOA quality improves with `p`, and depth is exactly what
  noisy hardware cannot afford.

## Source papers

- Farhi, Goldstone, Gutmann, "A Quantum Approximate Optimization Algorithm"
  (2014) -- introduces QAOA.
- Glover, Kochenberger, Du, "A Tutorial on Formulating and Using QUBO Models"
  (2019) -- the penalty-encoding recipe used above.
- Hadfield et al., "From the Quantum Approximate Optimization Algorithm to a
  Quantum Alternating Operator Ansatz" (2019) -- constraint-preserving mixers,
  the principled fix for the feasibility/optimality tension measured here.
