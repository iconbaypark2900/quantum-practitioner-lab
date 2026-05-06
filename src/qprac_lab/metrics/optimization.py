from __future__ import annotations


def approximation_ratio(candidate_value: float, optimal_value: float) -> float:
    if optimal_value == 0:
        return 0.0
    return float(candidate_value / optimal_value)


def constraint_report(selection, budget: int):
    selected_count = int(sum(selection))
    return {
        "budget": budget,
        "selected_count": selected_count,
        "budget_constraint_satisfied": selected_count == budget,
        "constraint_violations": abs(selected_count - budget),
    }
