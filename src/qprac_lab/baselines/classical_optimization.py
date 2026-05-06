from __future__ import annotations

from itertools import product
import math
import random
import numpy as np


def maxcut_value(bitstring: str, edges: list[tuple[int, int]]) -> int:
    """Classical Max-Cut objective."""
    return sum(1 for i, j in edges if bitstring[i] != bitstring[j])


def brute_force_maxcut(n_nodes: int, edges: list[tuple[int, int]]):
    """Solve small Max-Cut instances exactly by exhaustive search."""
    best_bitstring = None
    best_value = -1
    for bits in product("01", repeat=n_nodes):
        bitstring = "".join(bits)
        value = maxcut_value(bitstring, edges)
        if value > best_value:
            best_value = value
            best_bitstring = bitstring
    return {"bitstring": best_bitstring, "objective_value": best_value}


def portfolio_objective(x, expected_returns, covariance, risk_lambda: float = 0.5) -> float:
    """Portfolio binary objective: return minus risk penalty."""
    x = np.asarray(x)
    return float(expected_returns @ x - risk_lambda * (x.T @ covariance @ x))


def brute_force_portfolio(expected_returns, covariance, risk_lambda: float = 0.5, budget: int = 3):
    """Exact small-N portfolio-selection baseline.

    Algorithm type:
    - Exhaustive combinatorial search.
    """
    n_assets = len(expected_returns)
    best_x = None
    best_value = float("-inf")

    for bits in product([0, 1], repeat=n_assets):
        x = np.array(bits)
        if x.sum() != budget:
            continue
        value = portfolio_objective(x, expected_returns, covariance, risk_lambda)
        if value > best_value:
            best_value = value
            best_x = x

    return {"selection": best_x, "objective_value": best_value}


def greedy_portfolio_selection(expected_returns, covariance, budget: int = 3, risk_lambda: float = 0.5):
    """Greedy baseline using return ranking.

    Algorithm type:
    - Greedy heuristic combinatorial optimization.
    """
    selected = np.argsort(expected_returns)[-budget:]
    x = np.zeros(len(expected_returns), dtype=int)
    x[selected] = 1
    return {"selection": x, "objective_value": portfolio_objective(x, expected_returns, covariance, risk_lambda)}


def simulated_annealing_portfolio(
    expected_returns,
    covariance,
    risk_lambda: float = 0.5,
    budget: int = 3,
    steps: int = 500,
    seed: int = 42,
):
    """Simple simulated annealing baseline for binary portfolio selection.

    Algorithm type:
    - Classical stochastic local search / simulated annealing.
    """
    rng = random.Random(seed)
    n = len(expected_returns)
    current = np.zeros(n, dtype=int)
    selected = rng.sample(range(n), budget)
    current[selected] = 1
    current_value = portfolio_objective(current, expected_returns, covariance, risk_lambda)

    best = current.copy()
    best_value = current_value

    for step in range(steps):
        temperature = max(1e-6, 1.0 - step / steps)
        ones = np.where(current == 1)[0].tolist()
        zeros = np.where(current == 0)[0].tolist()
        if not ones or not zeros:
            continue

        proposal = current.copy()
        proposal[rng.choice(ones)] = 0
        proposal[rng.choice(zeros)] = 1
        proposal_value = portfolio_objective(proposal, expected_returns, covariance, risk_lambda)

        delta = proposal_value - current_value
        if delta > 0 or rng.random() < math.exp(delta / temperature):
            current = proposal
            current_value = proposal_value

        if current_value > best_value:
            best = current.copy()
            best_value = current_value

    return {"selection": best, "objective_value": float(best_value)}
