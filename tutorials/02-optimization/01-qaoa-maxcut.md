# Tutorial: QAOA for Max-Cut

## Concept

Split a graph's vertices into two sets so that as many edges as possible run
*between* the sets. Max-Cut is NP-hard, and it is QAOA's reference problem --
where the algorithm was introduced and where nearly all of its theory lives.

It is also the cleanest problem in this repository to measure, for two reasons:

- **The objective is positive**, so the ordinary approximation ratio
  `cut / cut_max` means exactly what it looks like. The portfolio tutorial cannot
  use it, because a mean-variance objective can go negative and a worse solution
  can then produce a larger ratio.
- **It is unconstrained.** Every bitstring is a legal cut. No penalty term, no
  feasibility filter, no mixer subtleties -- what remains is purely solution
  quality.

Start here before the portfolio tutorial. It isolates one variable.

## Math intuition

Assign each vertex `x_i ∈ {0,1}`. Edge `(i,j)` is cut when `x_i ≠ x_j`, which the
polynomial `x_i + x_j - 2 x_i x_j` scores as exactly 1 (cut) or 0 (uncut). So

```text
maximise  sum_{(i,j) in E} (x_i + x_j - 2 x_i x_j)
```

Negate it to get the QUBO this repo's `maxcut_qubo` builds. Under the standard
`x = (1 - z)/2` substitution each edge becomes `(1 - z_i z_j)/2`, giving the
Ising form every QAOA paper writes down:

```text
C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2
```

`Z_i Z_j` is `+1` when the endpoints agree and `-1` when they differ, so each
term contributes 1 precisely for a cut edge.

## Minimal example

```python
from qprac_lab.algorithms.optimization.qaoa_maxcut import make_maxcut_graph
from qprac_lab.algorithms.optimization.qubo_builder import maxcut_qubo

edges = make_maxcut_graph(num_nodes=8, degree=3, seed=42)
qubo = maxcut_qubo(8, edges)
print(-qubo.brute_force()["objective_value"])   # 10 -- the true maximum cut
```

## Runnable implementation

```bash
python scripts/run_demo.py --algorithm qaoa_maxcut
```

## Classical baselines

| Baseline | Cut | Ratio |
| --- | --- | --- |
| Brute force (exact) | 10 | 1.000 |
| Greedy | 10 | 1.000 |
| Random assignment (expected) | 6.0 | 0.600 |

The random-assignment row is the one that matters, and it is not arbitrary: a
uniformly random cut separates each edge with probability 1/2, so it scores
`|E|/2` in expectation. **Any method that cannot beat `|E|/2` has done nothing.**

## Benchmark

8-node 3-regular graph, `p = 3`, 4096 shots, statevector simulator:

| Metric | Value |
| --- | --- |
| Maximum cut (exact) | 10 |
| QAOA best sample | 10 (ratio **1.000**) |
| **QAOA expected cut** | **9.05 (ratio 0.905)** |
| Random assignment | 6.0 (ratio 0.600) |
| P(sampling an optimal cut) | 49.7% |

**Read the expected value, not the best sample.** QAOA returns a distribution,
and its theoretical guarantees are statements about `<C>`. "Best of 4096 shots"
measures your shot budget as much as your algorithm -- it improves indefinitely
just by sampling more, and it reaches 1.000 here for exactly that reason.

The honest headline is **0.905 expected approximation ratio**, comfortably above
the 0.600 random baseline, with half of all shots landing on an optimal cut. That
is a genuinely good QAOA result.

And then the deflating line in the baseline table: **greedy also found a cut of
10**, instantly, on a laptop. On a graph this size the quantum method has nothing
to offer. That is not a flaw in the implementation -- it is what an 8-vertex
problem looks like.

## Visualization

The sampled distribution can be plotted with the same helper the portfolio
tutorial uses:

```python
from qprac_lab.visualization.tutorial_outputs import plot_qaoa_distribution
plot_qaoa_distribution(result.top_bitstrings)
```

## Real use case

```text
Network / interaction graph
  -> Max-Cut or community-detection objective
  -> partition into two groups
  -> load balancing, circuit layout, image segmentation, clustering
```

Max-Cut shows up as VLSI circuit partitioning, statistical-physics spin-glass
ground states (its original home), image segmentation, and as a subroutine inside
larger graph-partitioning pipelines.

## When not to use this

- **On small graphs.** Greedy tied the exact optimum here in microseconds.
- **On graphs where good heuristics exist**, which is most of them. The real bar
  is not brute force — it is Goemans-Williamson, whose SDP relaxation guarantees
  a 0.878 approximation ratio *in the worst case*. QAOA at `p = 3` reaching 0.905
  on one easy 8-vertex graph is not evidence of beating it, and matching a
  guarantee on a favourable instance is not the same as having one.
- **At shallow depth on hard instances.** QAOA's quality grows with `p`, and
  depth is precisely what noisy hardware cannot afford.

## Source papers

- Farhi, Goldstone, Gutmann, "A Quantum Approximate Optimization Algorithm"
  (2014) -- introduces QAOA on Max-Cut.
- Goemans, Williamson, "Improved approximation algorithms for maximum cut and
  satisfiability problems using semidefinite programming" (1995) -- the 0.878
  classical guarantee that any quantum claim has to be measured against.
- Crooks, "Performance of the Quantum Approximate Optimization Algorithm on the
  Maximum Cut Problem" (2018) -- empirical QAOA-vs-GW comparison.
