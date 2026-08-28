"""QAOA for Max-Cut.

Max-Cut is QAOA's reference problem: it is where the algorithm was introduced and
where essentially all of its theory lives. It is also the cleanest problem in
this repository to measure, for two reasons.

* **The objective is positive**, so the ordinary approximation ratio
  ``cut / cut_max`` means what it looks like it means. (The portfolio objective
  can go negative, which is why that tutorial needs a normalised ratio instead.)
* **It is unconstrained.** No penalty term, no feasibility filter, no mixer
  subtleties -- every bitstring is a legal cut. What is left is purely a question
  of solution quality.

The headline number is the **expected** approximation ratio over the sampled
distribution, not the best sample. QAOA's guarantees are statements about
``<C>``; taking the best of many shots measures your shot budget as much as the
algorithm, and improves indefinitely just by sampling more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import run_qaoa
from qprac_lab.algorithms.optimization.qubo_builder import maxcut_qubo
from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit
from qprac_lab.baselines.classical_optimization import (
    brute_force_maxcut,
    greedy_maxcut,
    maxcut_value,
)
from qprac_lab.metrics.optimization import approximation_ratio


@dataclass
class MaxCutReport:
    """QAOA Max-Cut results with exact and heuristic classical baselines."""

    algorithm: str
    use_case: str
    algorithm_type: str
    backend: dict
    graph: dict
    qaoa_reps: int
    optimizer: str
    function_evaluations: int
    optimal_parameters: list[float]
    best_bitstring: str
    best_cut_value: int
    expected_cut_value: float
    max_cut_value: int
    approximation_ratio: float
    expected_approximation_ratio: float
    random_guess_ratio: float
    optimal_probability: float
    beats_random_guessing: bool
    matches_brute_force: bool
    top_bitstrings: list[dict[str, Any]] = field(default_factory=list)
    baseline_report: dict = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def make_maxcut_graph(num_nodes: int = 8, degree: int = 3, seed: int = 42):
    """Deterministic random regular graph, as an edge list."""
    import networkx as nx

    graph = nx.random_regular_graph(degree, num_nodes, seed=seed)
    return [(int(u), int(v)) for u, v in graph.edges()]


def run_qaoa_maxcut_tutorial(
    num_nodes: int = 8,
    degree: int = 3,
    reps: int = 3,
    backend: str = "statevector",
    shots: int = 4096,
    optimizer: str = "COBYLA",
    maxiter: int = 300,
    seed: int = 42,
) -> MaxCutReport:
    """Run QAOA on a random regular graph and score it against the exact optimum."""
    require_qiskit("The QAOA Max-Cut tutorial")
    edges = make_maxcut_graph(num_nodes=num_nodes, degree=degree, seed=seed)
    qubo = maxcut_qubo(num_nodes, edges)

    result, counts, history, _offset = run_qaoa(
        qubo,
        reps=reps,
        optimizer=optimizer,
        maxiter=maxiter,
        backend=backend,
        shots=shots,
        seed=seed,
    )

    exact = brute_force_maxcut(num_nodes, edges)
    max_cut = int(exact["objective_value"])
    greedy = greedy_maxcut(num_nodes, edges)

    total_shots = sum(counts.values())
    decoded = []
    for bitstring, count in counts.items():
        selection = qubo.bitstring_to_selection(bitstring)
        assignment = "".join(str(bit) for bit in selection)
        decoded.append(
            {
                "bitstring": assignment,
                "count": int(count),
                "probability": count / total_shots,
                "cut_value": maxcut_value(assignment, edges),
            }
        )
    decoded.sort(key=lambda row: -row["count"])

    best_row = max(decoded, key=lambda row: row["cut_value"])
    expected_cut = sum(row["probability"] * row["cut_value"] for row in decoded)
    optimal_probability = sum(
        row["probability"] for row in decoded if row["cut_value"] == max_cut
    )
    # A uniformly random assignment cuts each edge with probability 1/2.
    random_ratio = (0.5 * len(edges)) / max_cut

    return MaxCutReport(
        algorithm="qaoa_maxcut",
        use_case="graph_partitioning_and_network_clustering",
        algorithm_type="hybrid_combinatorial_optimization",
        backend=QiskitBackendAdapter(backend=backend, shots=shots, seed=seed).describe(),
        graph={
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "degree": degree,
            "edges": edges,
        },
        qaoa_reps=reps,
        optimizer=optimizer,
        function_evaluations=len(history),
        optimal_parameters=[float(v) for v in np.atleast_1d(result.x)],
        best_bitstring=best_row["bitstring"],
        best_cut_value=int(best_row["cut_value"]),
        expected_cut_value=float(expected_cut),
        max_cut_value=max_cut,
        approximation_ratio=approximation_ratio(best_row["cut_value"], max_cut),
        expected_approximation_ratio=approximation_ratio(expected_cut, max_cut),
        random_guess_ratio=float(random_ratio),
        optimal_probability=float(optimal_probability),
        beats_random_guessing=bool(expected_cut > 0.5 * len(edges)),
        matches_brute_force=bool(best_row["cut_value"] == max_cut),
        top_bitstrings=decoded[:10],
        baseline_report={
            "brute_force": {
                "bitstring": exact["bitstring"],
                "cut_value": max_cut,
                "ratio": 1.0,
            },
            "greedy": {
                "bitstring": greedy["bitstring"],
                "cut_value": int(greedy["objective_value"]),
                "ratio": approximation_ratio(greedy["objective_value"], max_cut),
            },
            "random_assignment": {
                "expected_cut_value": 0.5 * len(edges),
                "ratio": float(random_ratio),
            },
        },
        notes={
            "headline_metric": (
                "expected_approximation_ratio -- QAOA's guarantees are about <C>, "
                "and best-of-N samples improves with shot budget alone"
            ),
            "random_baseline": "a uniformly random assignment cuts half the edges",
            "unconstrained": "every bitstring is a valid cut, so no penalty term is needed",
        },
    )


def run_qaoa_maxcut_scaffold():
    """Backwards-compatible alias for :func:`run_qaoa_maxcut_tutorial`."""
    return run_qaoa_maxcut_tutorial()
