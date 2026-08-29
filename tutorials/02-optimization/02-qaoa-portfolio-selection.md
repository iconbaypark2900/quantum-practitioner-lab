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

`n = 6`, `budget = 3`, 4096 shots, **5 optimiser restarts**, statevector
simulator. There are 20 feasible portfolios, so random guessing among them hits
the optimum 5% of the time — that is the bar the `Lift` column measures against.

| Mixer | Warm start | p | Feasible | Hits optimum | Lift |
| --- | --- | --- | --- | --- | --- |
| Transverse field + penalty | — | 3 | 95.5% | 5.93% | 1.19x |
| Transverse field + penalty | — | 6 | 99.7% | 5.20% | 1.04x |
| XY ring | k-hot | 3 | **100%** | 1.93% | 0.39x |
| XY ring | k-hot | 4 | **100%** | 1.46% | 0.29x |
| XY ring | k-hot | 6 | **100%** | 100% | 20.0x |
| XY ring | k-hot | 8 | **100%** | 100% | 20.0x |
| XY ring | Dicke | 3 | **100%** | 6.32% | 1.26x |
| XY ring | Dicke | 4 | **100%** | 11.55% | 2.31x |
| XY ring | Dicke | 6 | **100%** | 41.21% | 8.24x |
| XY ring | Dicke | 8 | **100%** | 13.18% | 2.64x |

### The penalty encoding barely works

At the automatic penalty QAOA reaches ~1.05–1.19x. It is **doing almost nothing
beyond learning feasibility**: feasibility is worth `P = 6.17` while the entire
spread of objective values across feasible portfolios is about 2.1, so the
penalty dominates the landscape.

Lowering the penalty to 0.5 roughly doubles the lift and drops feasibility to
55%. One failure traded for another.

### The XY mixer makes feasibility structural

`(X_i X_j + Y_i Y_j)/2` commutes with the total number operator, so evolution
under it cannot change Hamming weight. Start in a 3-hot state and the
optimisation *cannot leave* the feasible subspace. The penalty term disappears
and every angle works on the objective.

**Feasibility is exactly 100% at every depth, topology and warm start** — not
"mostly feasible", but a structural guarantee, asserted in tests as exact zero
infeasible probability.

> **On an ideal simulator.** That guarantee is a property of the ideal unitary,
> and noise does not respect it. Measured feasibility falls to 82.7% (light),
> 46.2% (moderate) and 33.2% (heavy). See
> [the noise benchmark](../05-benchmarking/noise_benchmark.md).

### Optimality is a lottery, and that is the real finding

Look at the k-hot rows: 0.39x, 0.29x, **20.0x**, **20.0x**. That is not a depth
trend. An earlier version of this tutorial reported the `p = 6` 20x figure as
"real and reproducible across sampling seeds" — technically true and thoroughly
misleading, because sampling seeds only affect shot noise. The optimiser's
*opening angles* are what matter, and the result is wildly sensitive to them.

Perturbing the initial `gamma` scale at `p = 6`, single restart, changing nothing
else:

| gamma multiplier | k-hot | Dicke |
| --- | --- | --- |
| 0.50 | 0.1% | 5.6% |
| 0.75 | 8.4% | 5.5% |
| 1.00 | **100%** | 35.6% |
| 1.25 | 23.0% | 1.4% |
| 1.50 | **100%** | 0.6% |
| 2.00 | 7.1% | 31.8% |
| **mean ± s.d.** | **39.8% ± 43.1%** | **13.4% ± 14.5%** |

The same configuration ranges from 0.1% — *twenty times worse than random
guessing* — to 100%. A single QAOA run on this problem reports a draw from that
distribution, not a property of the algorithm.

So `restarts=5` is now the default, and `restart_objectives` returns the whole
spread rather than just the winner.

### What the Dicke warm start actually buys

A Dicke state `|D^n_k>` is the uniform superposition over *all* weight-k
bitstrings — the constrained analogue of `|+>^n`, where the k-hot state picks one
arbitrary member of the feasible subspace.

It does **not** simply beat the k-hot start. It trades peak for reliability:

- **Better at low depth**, where k-hot is worse than random guessing (`p = 3`:
  1.26x vs 0.39x; `p = 4`: 2.31x vs 0.29x).
- **Lower peak**, never reaching k-hot's 20x.
- **Roughly a third of the variance** under warm-start perturbation (s.d. 14.5%
  vs 43.1%).

Its cost is depth: naive state preparation is depth 272 for `n = 6, k = 3` and
1241 for `n = 8, k = 4` — more than the QAOA circuit it warms up. Bärtschi and
Eidenbenz (2019) give a dedicated `O(kn)` construction, which is the route for
anything hardware-bound; this project uses exact preparation because the physics
is identical and the question here was about solution quality.

### A mismatch worth knowing about

Restarts keep the run with the best `<C>`, because that is what you can measure
without knowing the answer. But `<C>` is the *expected* cost, and a distribution
spread over several good portfolios can score better on it than one peaked on the
single best. So selecting by `<C>` does not reliably maximise `P(optimum)` — visible
in the k-hot `p = 3, 4` rows, which restarts made *worse*.

**QAOA optimises the objective you can measure, not the one you want.** For
Max-Cut those coincide (expected cut ratio *is* `<C>`), and restarts improve it
from 0.886 to 0.935. For "probability of the single best portfolio", they do not.

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
- **On noisy hardware, without a feasibility filter.** The XY mixer's guarantee
  is ideal-simulator-only; measured feasibility is 46% at moderate device noise.
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
