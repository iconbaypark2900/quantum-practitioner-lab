# Use Case: Logistics Routing

> **Status: classical scaffold.** `algorithms/optimization/logistics_routing_qubo.py`
> returns `status: "scaffold"` and runs no circuit. `qprac-lab list` marks it
> `classical_scaffold`. This page describes what the use case would involve, not
> something implemented.

## Problem

Assign vehicles, jobs or routes under constraints — capacity, time windows,
precedence — minimising distance or cost.

## Quantum formulation

Routing constraints become QUBO penalty terms, exactly as the budget constraint
does in the [portfolio tutorial](../02-qaoa-portfolio-selection.md). The
[QUBO/Ising mapping tutorial](../03-qubo-ising-mapping.md) covers the substitution.

## Why this is not implemented

The honest reason, and it generalises. A routing problem needs one binary variable
per (vehicle, job, position) triple, so a toy instance — 4 vehicles, 8 jobs —
already exceeds what any simulator here can run, and the penalty terms multiply
faster than the objective.

The portfolio tutorial already shows what happens next at a size that *does* fit:
with a textbook penalty encoding, QAOA beat uniform sampling over feasible
solutions by **~1.1x**. It learned feasibility and little else. Routing has
strictly more constraint structure and would be expected to do worse, not better.

Building it would demonstrate the encoding, not the method. The
[XY mixer](../02-qaoa-portfolio-selection.md) is the more useful direction:
enforcing a constraint by construction rather than by penalty.

## If you want to build it anyway

Start from `qubo_builder.py`, add the assignment constraints as penalties, and
report `optimal_probability_lift` before anything else. If it sits near 1.0, the
penalty term has flattened the landscape and the rest of the pipeline is measuring
nothing.
