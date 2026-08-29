from __future__ import annotations


def approximation_ratio(candidate_value: float, optimal_value: float) -> float:
    """Raw ``candidate / optimal`` ratio.

    Only meaningful for strictly positive objectives such as Max-Cut. For an
    objective that can go negative -- mean-variance portfolio scores, for one --
    use :func:`normalized_approximation_ratio` instead, since a worse solution
    can otherwise produce a larger ratio.
    """
    if optimal_value == 0:
        return 0.0
    return float(candidate_value / optimal_value)


def normalized_approximation_ratio(
    candidate_value: float,
    best_value: float,
    worst_value: float,
) -> float:
    """Position of a candidate within the achievable objective range, in ``[0, 1]``.

    ``1.0`` is the optimum and ``0.0`` the worst feasible solution, which keeps
    the score interpretable for objectives that are negative or span zero.
    """
    span = best_value - worst_value
    if span == 0:
        return 1.0
    return float((candidate_value - worst_value) / span)


def constraint_report(selection, budget: int):
    """Report whether a selection satisfies its cardinality budget."""
    selected_count = int(sum(selection))
    return {
        "budget": budget,
        "selected_count": selected_count,
        "budget_constraint_satisfied": selected_count == budget,
        "constraint_violations": abs(selected_count - budget),
    }
