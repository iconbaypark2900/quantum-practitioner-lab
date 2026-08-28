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

`n = 6`, `budget = 3`, 4096 shots, statevector simulator. There are 20 feasible
portfolios, so **random guessing among feasible states hits the optimum 5% of the
time** -- that is the bar, and the `Lift` column measures against it.

| Mixer | Penalty | p | Feasible | Hits optimum | Lift |
| --- | --- | --- | --- | --- | --- |
| Transverse field | 6.17 (auto) | 3 | 94.7% | 5.27% | 1.05x |
| Transverse field | 6.17 (auto) | 6 | 99.7% | 5.20% | 1.04x |
| Transverse field | 0.50 | 6 | 54.8% | 9.91% | 1.98x |
| XY ring | none | 3 | **100%** | 3.52% | 0.70x |
| XY ring | none | 4 | **100%** | 4.74% | 0.95x |
| XY ring | none | 6 | **100%** | **100%** | **20.0x** |
| XY ring | none | 8 | **100%** | 16.67% | 3.33x |
| XY complete | none | 3 | **100%** | 7.35% | 1.47x |
| XY complete | none | 6 | **100%** | 37.52% | 7.50x |

Every row found the optimum somewhere in its samples. The differences are
entirely in how much probability mass landed there.

### The penalty encoding barely works

At the safe automatic penalty, QAOA reaches ~1.05x. It is **doing almost nothing
beyond learning feasibility**. The reason is a scale mismatch: feasibility is
worth `P = 6.17`, while the entire spread of objective values across all feasible
portfolios is about 2.1. The penalty dominates the landscape, and the optimiser
spends its angles on the constraint.

Lowering the penalty to 0.5 doubles the lift to 1.98x -- and drops feasibility to
55%. You are trading one failure for another.

### The XY mixer removes the tradeoff

`(X_i X_j + Y_i Y_j)/2` commutes with the total number operator, so evolution
under it cannot change Hamming weight. Start in a 3-hot state and the
optimisation *cannot leave* the feasible subspace. The penalty term disappears
entirely, and every angle works on the objective.

**Feasibility is exactly 100% at every depth and topology** -- not "mostly
feasible", but a structural guarantee. The test suite asserts infeasible
probability is exactly zero rather than merely small.

Optimality improves too, dramatically at the best depth. But read the ring rows
carefully:

```text
p = 3 -> 3.52%      p = 4 -> 4.74%      p = 6 -> 100%      p = 8 -> 16.67%
```

**That is not monotonic, and `p = 6` is not special.** All rows start from the
same fixed linear-ramp warm start, and COBYLA settles into different local optima
from it at different depths. The 20x row is real and reproducible across sampling
seeds, but it is a property of *this warm start on this instance*, not a
depth you should expect to transfer. The complete-graph topology behaves far more
predictably (1.47x -> 7.50x) because it mixes the whole feasible subspace in a
single layer, where the ring needs several.

The cost is circuit depth: two-qubit XY rotations instead of single-qubit X
rotations, and `n(n-1)/2` of them per layer for the complete topology. On real
hardware that is exactly the resource you do not have.

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
- **Hard constraints with the default mixer.** Penalty encoding makes constraints
  soft. If a violated budget is unacceptable rather than merely undesirable, use
  `mixer="xy"` -- it makes feasibility structural rather than incentivised.
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
