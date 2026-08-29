"""Tutorial 2: QAOA for budget-constrained portfolio selection.

The full pipeline, with nothing faked:

    mean-variance problem
      -> QUBO with the cardinality constraint as a penalty term
      -> Ising Hamiltonian (x = (1 - z) / 2)
      -> QAOA ansatz, p alternating cost/mixer layers
      -> classical optimisation of the 2p angles against <C>
      -> sample the optimised state, decode bitstrings, keep the feasible ones

The honest metrics are the sampled ones. QAOA returns a *distribution*, not an
answer, so what matters is how much probability mass lands on feasible
portfolios and on the true optimum -- not merely whether the best of many
samples happens to be right. Those numbers are reported explicitly, alongside
brute-force, greedy, and simulated-annealing baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Any

import numpy as np
from scipy.optimize import minimize

from qprac_lab.algorithms.optimization.qubo_builder import QUBO, portfolio_qubo
from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, require_qiskit
from qprac_lab.baselines.classical_optimization import (
    brute_force_portfolio,
    greedy_portfolio_selection,
    portfolio_objective,
    simulated_annealing_portfolio,
)
from qprac_lab.circuits.mixers import build_xy_qaoa_ansatz
from qprac_lab.metrics.optimization import constraint_report, normalized_approximation_ratio


@dataclass
class PortfolioSelectionReport:
    """QAOA's answer, its sampling statistics, and the classical baselines."""

    algorithm: str
    use_case: str
    algorithm_type: str
    backend: dict
    problem: dict
    qaoa_reps: int
    mixer: str
    restarts: int
    restart_objectives: list[float]
    optimizer: str
    function_evaluations: int
    optimal_parameters: list[float]
    qaoa_cost_expectation: float
    selected_assets: list[int]
    selection: list[int]
    objective_value: float
    constraint_report: dict
    shots: int
    feasible_probability: float
    optimal_probability: float
    uniform_feasible_probability: float
    optimal_probability_lift: float
    normalized_approximation_ratio: float
    objective_gap_to_optimum: float
    matches_brute_force: bool
    top_bitstrings: list[dict[str, Any]] = field(default_factory=list)
    baseline_report: dict = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def qaoa_initial_point(reps: int, gamma_scale: float = 0.1) -> np.ndarray:
    """Linear-ramp initial angles, ordered as Qiskit's ``[beta..., gamma...]``.

    Betas ramp down and gammas ramp up, the discretised-adiabatic schedule that
    is the standard warm start for QAOA. Gammas are scaled down because the cost
    operator carries the (large) constraint penalty, and an unscaled gamma
    rotates far past the useful regime.
    """
    steps = np.arange(1, reps + 1)
    betas = (1.0 - steps / (reps + 1)) * (np.pi / 4)
    gammas = (steps / (reps + 1)) * gamma_scale
    return np.concatenate([betas, gammas])


def run_qaoa(
    qubo: QUBO,
    reps: int = 3,
    optimizer: str = "COBYLA",
    maxiter: int = 250,
    backend: str = "statevector",
    shots: int = 4096,
    seed: int = 42,
    noise: str | None = None,
    mixer: str = "transverse_field",
    num_ones: int | None = None,
    xy_topology: str = "ring",
    xy_initial_state: str = "k_hot",
    gamma_scale: float | None = None,
    restarts: int = 1,
):
    """Optimise QAOA angles for a QUBO, then sample the optimised state.

    ``mixer`` selects how constraints are handled:

    ``transverse_field``
        The standard ``sum_i X_i`` mixer. Explores every bitstring, so hard
        constraints must already be penalties inside ``qubo``.
    ``xy``
        A Hamming-weight-preserving XY mixer over a ``num_ones``-hot subspace.
        Feasibility holds by construction, so ``qubo`` should carry **no**
        penalty term.

    ``restarts`` repeats the optimisation from perturbed warm starts and keeps the
    best. This is not optional rigour: measured on this problem, the *same*
    configuration at ``p=6`` produced anywhere from 0.1% to 100% probability on the
    optimum depending only on the opening angles. A single restart reports a draw
    from that distribution, not a property of the algorithm.

    Returns ``(scipy_result, counts, history, offset, restart_objectives)``.
    """
    require_qiskit("Running QAOA")

    cost_operator, offset = qubo.to_ising()
    if mixer == "xy":
        if num_ones is None:
            raise ValueError("the xy mixer needs num_ones (the Hamming weight to preserve)")
        ansatz = build_xy_qaoa_ansatz(
            cost_operator,
            reps=reps,
            num_ones=num_ones,
            topology=xy_topology,
            initial_state=xy_initial_state,
        )
    elif mixer == "transverse_field":
        from qiskit.circuit.library import QAOAAnsatz

        # Decompose once, up front. QAOAAnsatz holds PauliEvolutionGates whose
        # synthesis is otherwise redone on every estimator call -- measured at
        # 2.56s per call versus 0.006s pre-decomposed on an 8-qubit graph, a
        # ~400x difference that turns a 2-second optimisation into 13 minutes.
        ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=reps).decompose(reps=3)
    else:
        raise ValueError(f"Unknown mixer {mixer!r}; expected 'transverse_field' or 'xy'")
    adapter = QiskitBackendAdapter(backend=backend, shots=None, seed=seed, noise=noise)
    estimator = adapter.estimator()
    ansatz = adapter.prepare(ansatz)
    history: list[float] = []

    def objective(parameters: np.ndarray) -> float:
        job = estimator.run([(ansatz, cost_operator, list(parameters))])
        value = float(job.result()[0].data.evs) + offset
        history.append(value)
        return value

    if gamma_scale is None:
        # Scale the opening gamma to the cost operator's magnitude. A gamma tuned
        # for a penalty-dominated operator is far too small once the penalty is
        # removed, and the optimiser starts on a flat patch of landscape.
        largest = max(float(np.max(np.abs(np.real(cost_operator.coeffs)))), 1e-12)
        gamma_scale = 1.0 / largest

    if restarts < 1:
        raise ValueError(f"restarts must be at least 1, got {restarts}")

    rng = np.random.default_rng(seed)
    best_result = None
    restart_objectives: list[float] = []
    for attempt in range(restarts):
        # First restart uses the documented linear ramp; the rest perturb its
        # scale, which is the axis the outcome proved most sensitive to.
        multiplier = 1.0 if attempt == 0 else float(rng.uniform(0.5, 2.0))
        candidate = minimize(
            objective,
            x0=qaoa_initial_point(reps, gamma_scale=gamma_scale * multiplier),
            method=optimizer,
            options={"maxiter": maxiter},
        )
        restart_objectives.append(float(candidate.fun))
        if best_result is None or candidate.fun < best_result.fun:
            best_result = candidate
    result = best_result

    measured = ansatz.assign_parameters(result.x)
    measured.measure_all()
    sampling_adapter = QiskitBackendAdapter(
        backend=backend, shots=shots, seed=seed, noise=noise
    )
    counts = (
        sampling_adapter.sampler()
        .run([sampling_adapter.prepare(measured)])
        .result()[0]
        .data.meas.get_counts()
    )
    return result, counts, history, offset, restart_objectives


def feasible_objective_range(expected_returns, covariance, budget: int, risk_lambda: float):
    """Best and worst objective over portfolios that satisfy the budget.

    Needed for a meaningful approximation ratio: the raw ``candidate / optimum``
    ratio is meaningless here because the mean-variance objective can be
    negative, where a worse solution can produce a larger ratio.
    """
    n = len(expected_returns)
    values = []
    for chosen in combinations(range(n), budget):
        x = np.zeros(n, dtype=int)
        x[list(chosen)] = 1
        values.append(portfolio_objective(x, expected_returns, covariance, risk_lambda))
    return float(max(values)), float(min(values))


def run_qaoa_portfolio_selection_tutorial(
    n_assets: int = 6,
    budget: int = 3,
    risk_lambda: float = 0.5,
    reps: int = 3,
    mixer: str = "transverse_field",
    xy_topology: str = "ring",
    xy_initial_state: str = "k_hot",
    penalty: float | None = None,
    backend: str = "statevector",
    shots: int = 4096,
    optimizer: str = "COBYLA",
    maxiter: int = 300,
    seed: int = 42,
    noise: str | None = None,
    restarts: int = 5,
) -> PortfolioSelectionReport:
    """Run tutorial 2 end to end: QAOA portfolio selection against three baselines."""
    require_qiskit("The QAOA portfolio-selection tutorial")
    from qprac_lab.data.synthetic import make_small_portfolio_dataset

    expected_returns, covariance = make_small_portfolio_dataset(n_assets=n_assets)
    if mixer == "xy" and penalty is None:
        # The XY mixer enforces the budget structurally, so a penalty term would
        # only add a constant over the feasible subspace while distorting the
        # landscape the optimiser sees.
        penalty = 0.0
    qubo = portfolio_qubo(
        expected_returns,
        covariance,
        budget=budget,
        risk_lambda=risk_lambda,
        penalty=penalty,
    )

    result, counts, history, _offset, restart_objectives = run_qaoa(
        qubo,
        reps=reps,
        optimizer=optimizer,
        maxiter=maxiter,
        backend=backend,
        shots=shots,
        seed=seed,
        noise=noise,
        mixer=mixer,
        num_ones=budget if mixer == "xy" else None,
        xy_topology=xy_topology,
        xy_initial_state=xy_initial_state,
        restarts=restarts,
    )

    total_shots = sum(counts.values())
    decoded = []
    for bitstring, count in counts.items():
        selection = qubo.bitstring_to_selection(bitstring)
        feasible = int(selection.sum()) == budget
        decoded.append(
            {
                "bitstring": bitstring,
                "selection": selection.tolist(),
                "count": int(count),
                "probability": count / total_shots,
                "feasible": feasible,
                "objective_value": portfolio_objective(
                    selection, expected_returns, covariance, risk_lambda
                ),
            }
        )
    decoded.sort(key=lambda row: -row["count"])

    feasible_rows = [row for row in decoded if row["feasible"]]
    feasible_probability = sum(row["probability"] for row in feasible_rows)

    exact = brute_force_portfolio(
        expected_returns, covariance, risk_lambda=risk_lambda, budget=budget
    )
    optimal_selection = np.asarray(exact["selection"])

    if feasible_rows:
        best_row = max(feasible_rows, key=lambda row: row["objective_value"])
        selection = np.asarray(best_row["selection"])
    else:
        # QAOA produced nothing satisfying the budget. Report that honestly
        # rather than quietly substituting a classical answer.
        best_row = None
        selection = np.zeros(n_assets, dtype=int)

    objective_value = portfolio_objective(selection, expected_returns, covariance, risk_lambda)
    best_feasible, worst_feasible = feasible_objective_range(
        expected_returns, covariance, budget, risk_lambda
    )
    optimal_probability = sum(
        row["probability"]
        for row in decoded
        if np.array_equal(np.asarray(row["selection"]), optimal_selection)
    )

    # The number that actually decides whether QAOA earned its keep: a uniform
    # draw over feasible portfolios already hits the optimum with probability
    # 1 / C(n, budget). Beating that is the bar; matching it means the penalty
    # term dominated the cost landscape and QAOA only learned feasibility.
    num_feasible = comb(n_assets, budget)
    uniform_feasible_probability = 1.0 / num_feasible if num_feasible else 0.0
    optimal_probability_lift = (
        optimal_probability / uniform_feasible_probability
        if uniform_feasible_probability
        else 0.0
    )

    greedy = greedy_portfolio_selection(
        expected_returns, covariance, budget=budget, risk_lambda=risk_lambda
    )
    annealed = simulated_annealing_portfolio(
        expected_returns, covariance, budget=budget, risk_lambda=risk_lambda
    )
    baseline_report = {
        name: {
            "selection": np.asarray(payload["selection"]).tolist(),
            "objective_value": float(payload["objective_value"]),
        }
        for name, payload in (
            ("brute_force", exact),
            ("greedy", greedy),
            ("simulated_annealing", annealed),
        )
    }

    return PortfolioSelectionReport(
        algorithm="qaoa_portfolio_selection",
        use_case="quantum_hybrid_portfolio_optimizer",
        algorithm_type="hybrid_combinatorial_optimization",
        backend=QiskitBackendAdapter(
            backend=backend, shots=shots, seed=seed, noise=noise
        ).describe(),
        problem={
            "n_assets": n_assets,
            "budget": budget,
            "risk_lambda": risk_lambda,
            "penalty": qubo.metadata["penalty"],
            "num_ising_terms": len(qubo.to_ising()[0]),
            "expected_returns": np.asarray(expected_returns).round(6).tolist(),
        },
        qaoa_reps=reps,
        restarts=restarts,
        restart_objectives=restart_objectives,
        mixer=(
            f"{mixer}:{xy_topology}:{xy_initial_state}" if mixer == "xy" else mixer
        ),
        optimizer=optimizer,
        function_evaluations=len(history),
        optimal_parameters=[float(v) for v in np.atleast_1d(result.x)],
        qaoa_cost_expectation=float(result.fun),
        selected_assets=np.where(selection == 1)[0].tolist(),
        selection=selection.tolist(),
        objective_value=float(objective_value),
        constraint_report=constraint_report(selection, budget),
        shots=int(total_shots),
        feasible_probability=float(feasible_probability),
        optimal_probability=float(optimal_probability),
        uniform_feasible_probability=float(uniform_feasible_probability),
        optimal_probability_lift=float(optimal_probability_lift),
        normalized_approximation_ratio=normalized_approximation_ratio(
            objective_value, best_feasible, worst_feasible
        ),
        objective_gap_to_optimum=float(best_feasible - objective_value),
        matches_brute_force=bool(np.array_equal(selection, optimal_selection)),
        top_bitstrings=decoded[:10],
        baseline_report=baseline_report,
        notes={
            "constraint_handling": "cardinality constraint encoded as a quadratic penalty",
            "feasible_probability": "share of shots satisfying the budget constraint",
            "optimal_probability": "share of shots landing exactly on the brute-force optimum",
            "sampled_answer": best_row is not None,
            "objective_convention": "maximise expected return minus risk_lambda * variance",
            "optimal_probability_lift": (
                "optimal_probability divided by uniform sampling over feasible "
                "portfolios; 1.0 means QAOA did no better than random feasible guessing"
            ),
            "mixer": (
                "transverse_field explores all bitstrings and needs a penalty term; "
                "xy preserves Hamming weight so feasibility is guaranteed by construction"
            ),
            "penalty_tradeoff": (
                "a large penalty buys feasibility but flattens the distribution over "
                "feasible states; a small penalty sharpens it but samples infeasible "
                "portfolios. Tune `penalty` to trade these off."
            ),
        },
    )
