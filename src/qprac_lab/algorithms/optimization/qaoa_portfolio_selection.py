from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from qprac_lab.data.synthetic import make_small_portfolio_dataset
from qprac_lab.baselines.classical_optimization import (
    brute_force_portfolio,
    greedy_portfolio_selection,
    simulated_annealing_portfolio,
)
from qprac_lab.metrics.optimization import constraint_report


@dataclass
class PortfolioSelectionReport:
    algorithm: str
    use_case: str
    algorithm_type: str
    selected_assets: list[int]
    objective_value: float
    constraint_report: dict
    baseline_report: dict


def run_qaoa_portfolio_selection_tutorial(
    n_assets: int = 6,
    budget: int = 3,
    risk_lambda: float = 0.5,
) -> PortfolioSelectionReport:
    """Run the QAOA portfolio-selection tutorial scaffold.

    Current state:
    - Uses brute-force, greedy, and simulated annealing baselines.
    - QAOA implementation should replace or supplement selected_assets.
    """
    expected_returns, covariance = make_small_portfolio_dataset(n_assets=n_assets)
    exact = brute_force_portfolio(expected_returns, covariance, risk_lambda=risk_lambda, budget=budget)
    greedy = greedy_portfolio_selection(expected_returns, covariance, budget=budget, risk_lambda=risk_lambda)
    annealed = simulated_annealing_portfolio(
        expected_returns,
        covariance,
        budget=budget,
        risk_lambda=risk_lambda,
    )

    selected_assets = np.where(exact["selection"] == 1)[0].tolist()

    baseline_report = {
        "brute_force": {
            "selection": exact["selection"].tolist(),
            "objective_value": float(exact["objective_value"]),
        },
        "greedy": {
            "selection": greedy["selection"].tolist(),
            "objective_value": float(greedy["objective_value"]),
        },
        "simulated_annealing": {
            "selection": annealed["selection"].tolist(),
            "objective_value": float(annealed["objective_value"]),
        },
    }

    return PortfolioSelectionReport(
        algorithm="qaoa_portfolio_selection",
        use_case="quantum_hybrid_portfolio_optimizer",
        algorithm_type="hybrid_combinatorial_optimization",
        selected_assets=selected_assets,
        objective_value=float(exact["objective_value"]),
        constraint_report=constraint_report(exact["selection"], budget),
        baseline_report=baseline_report,
    )
