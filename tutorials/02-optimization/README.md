# 02 Quantum Optimization

QUBO/Ising formulation and QAOA. This is the section where the classical baselines
win most often, and the tutorials are written around that rather than despite it.

## Tutorials

| # | Tutorial | What it shows |
| --- | --- | --- |
| 1 | [QAOA for Max-Cut](01-qaoa-maxcut.md) | **Start here.** The unconstrained reference problem, so the approximation ratio means something: **0.905 expected** against 0.600 for random. Then the honest part — greedy also hits the exact optimum, instantly, on 8 vertices. |
| 2 | [QAOA for Portfolio Selection](02-qaoa-portfolio-selection.md) | A real constraint. With the textbook penalty encoding it beats uniform sampling over feasible portfolios by **~1.1x** — it learns feasibility and little else. An XY mixer makes feasibility structural (100%), and optimality remains a **lottery**: 0.1% to 100% on opening angles alone. |
| 3 | [QUBO and Ising Mapping](03-qubo-ising-mapping.md) | The substitution `x = (1 − z)/2` and why it is exact. Checked over **every** `2ⁿ` assignment rather than sampled. |

## Use cases

- [Quantum-hybrid portfolio optimizer](use_cases/portfolio_optimizer.md) — implemented
- [Logistics routing](use_cases/logistics_routing.md) — **classical scaffold only**, see that page

## Not covered here

**Scheduling QUBO** and a quantum **logistics routing** solver do not exist. The
routing module (`algorithms/optimization/logistics_routing_qubo.py`) is a classical
scaffold that reports `status: "scaffold"`, and `qprac-lab list` marks it as such.
Nothing here is labelled quantum unless it runs a circuit.

## Papers

[Source papers for this section](papers.md).
