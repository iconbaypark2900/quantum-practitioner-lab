"""End-to-end checks for the three priority tutorials.

These assert on physics and optimisation properties, not just that the functions
return without raising -- a scaffold that returns a plausible-looking dataclass
would pass a smoke test while being completely wrong.
"""

import pytest

pytest.importorskip("qiskit", reason='needs the quantum stack: pip install -e ".[qiskit]"')

import numpy as np  # noqa: E402

from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import (  # noqa: E402
    run_qaoa_portfolio_selection_tutorial,
)
from qprac_lab.algorithms.qml.quantum_kernel_biomedical import (  # noqa: E402
    run_quantum_kernel_biomedical_tutorial,
)
from qprac_lab.algorithms.simulation.vqe_molecular_energy import (  # noqa: E402
    CHEMICAL_ACCURACY_HARTREE,
    run_vqe_molecular_energy_tutorial,
)
from qprac_lab.data.hetionet import hetionet_available  # noqa: E402

#: Reference H2 / STO-3G values at R = 0.735 A, cross-checked against PySCF.
H2_FCI_ENERGY = -1.137306
H2_HARTREE_FOCK_ENERGY = -1.116999


@pytest.fixture(scope="module")
def vqe_result():
    return run_vqe_molecular_energy_tutorial(include_dissociation_curve=False)


def test_vqe_reaches_the_known_h2_ground_state(vqe_result):
    assert vqe_result.algorithm == "vqe_molecular_energy"
    assert vqe_result.vqe_energy == pytest.approx(H2_FCI_ENERGY, abs=1e-5)
    assert vqe_result.exact_baseline_energy == pytest.approx(H2_FCI_ENERGY, abs=1e-5)
    assert vqe_result.hartree_fock_baseline_energy == pytest.approx(
        H2_HARTREE_FOCK_ENERGY, abs=1e-5
    )


def test_vqe_respects_the_variational_principle(vqe_result):
    """No exact-simulation VQE energy may fall below the true ground state."""
    assert vqe_result.vqe_energy >= vqe_result.exact_baseline_energy - 1e-9


def test_vqe_beats_hartree_fock_and_reaches_chemical_accuracy(vqe_result):
    assert vqe_result.beats_hartree_fock
    assert vqe_result.absolute_error < CHEMICAL_ACCURACY_HARTREE
    assert vqe_result.chemical_accuracy_reached
    assert vqe_result.correlation_recovery_fraction > 0.99


def test_vqe_convergence_history_starts_at_hartree_fock(vqe_result):
    """The chemically-motivated ansatz begins at the HF determinant by construction."""
    assert len(vqe_result.convergence_history) > 1
    assert vqe_result.convergence_history[0] == pytest.approx(
        vqe_result.hartree_fock_baseline_energy, abs=1e-6
    )
    assert min(vqe_result.convergence_history) <= vqe_result.convergence_history[0]


@pytest.fixture(scope="module")
def qaoa_result():
    return run_qaoa_portfolio_selection_tutorial(shots=2048)


def test_qaoa_portfolio_selection_respects_the_budget(qaoa_result):
    assert qaoa_result.algorithm == "qaoa_portfolio_selection"
    assert "constraint_violations" in qaoa_result.constraint_report
    assert qaoa_result.constraint_report["budget_constraint_satisfied"]
    assert qaoa_result.constraint_report["constraint_violations"] == 0
    assert len(qaoa_result.selected_assets) == qaoa_result.problem["budget"]


def test_qaoa_never_beats_the_exact_optimum(qaoa_result):
    """Brute force is exact, so QAOA can match it but never exceed it."""
    optimum = qaoa_result.baseline_report["brute_force"]["objective_value"]
    assert qaoa_result.objective_value <= optimum + 1e-9
    assert 0.0 <= qaoa_result.normalized_approximation_ratio <= 1.0 + 1e-9
    assert qaoa_result.objective_gap_to_optimum >= -1e-9


def test_qaoa_penalty_produces_mostly_feasible_samples(qaoa_result):
    """The whole point of the penalty term is to concentrate on feasible states."""
    assert qaoa_result.feasible_probability > 0.5
    assert 0.0 <= qaoa_result.optimal_probability <= 1.0
    assert qaoa_result.uniform_feasible_probability == pytest.approx(1 / 20)


@pytest.fixture(scope="module")
def kernel_result():
    """Kernel *properties* are dataset-independent, so use the fast offline path.

    Behaviour on the real Hetionet data is covered by
    ``test_quantum_kernel_runs_on_real_hetionet_data`` and by
    ``tests/test_hetionet_dataset.py``.
    """
    return run_quantum_kernel_biomedical_tutorial(
        dataset="synthetic", n_pairs=40, n_splits=3, n_repeats=1
    )


def test_quantum_kernel_tutorial_reports_every_baseline(kernel_result):
    assert kernel_result.algorithm == "quantum_kernel_biomedical_classification"
    for metrics in (
        kernel_result.quantum_kernel_metrics,
        kernel_result.rbf_svm_metrics,
        kernel_result.random_forest_metrics,
    ):
        assert {"accuracy", "f1", "roc_auc"} <= set(metrics)
    assert isinstance(kernel_result.xgboost_metrics, (dict, str))


def test_quantum_kernel_matrix_is_a_valid_kernel(kernel_result):
    """Fidelities are in [0, 1], symmetric, and exactly 1 on the diagonal."""
    matrix = np.asarray(kernel_result.kernel_matrix_preview)
    assert matrix.shape[0] == matrix.shape[1]
    assert np.allclose(matrix, matrix.T, atol=1e-6)
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-6)
    assert matrix.min() >= -1e-9
    assert matrix.max() <= 1.0 + 1e-9


def test_quantum_kernel_evaluation_is_repeated_not_a_single_split(kernel_result):
    """A single split on data this small is uninformative; the report must say how
    many evaluations back its numbers."""
    assert kernel_result.evaluation["n_evaluations"] == 3
    assert kernel_result.quantum_kernel_metrics["n_evaluations"] == 3
    assert kernel_result.quantum_kernel_metrics["roc_auc_std"] is not None


def test_synthetic_dataset_is_labelled_as_not_real(kernel_result):
    """The blob fallback must never be mistaken for the real benchmark."""
    assert kernel_result.dataset["real_data"] is False
    assert "artifact" in kernel_result.dataset["warning"]


@pytest.mark.skipif(
    not hetionet_available(),
    reason="Hetionet not cached; run `python scripts/download_data.py`",
)
def test_quantum_kernel_runs_on_real_hetionet_data():
    result = run_quantum_kernel_biomedical_tutorial(n_pairs=60, n_splits=3, n_repeats=1)
    assert result.dataset["real_data"] is True
    assert result.dataset["target_edge_type"].startswith("CtD")
    assert result.dataset["degree_only_roc_auc"] < 0.60
    comparison = result.quantum_vs_rbf
    assert comparison["comparable_folds"] == 3
    assert -1.0 <= comparison["mean_roc_auc_difference"] <= 1.0
    assert 0.0 <= comparison["quantum_win_rate"] <= 1.0


def test_quantum_kernel_ranking_is_ordered_and_complete(kernel_result):
    ranks = [row["rank"] for row in kernel_result.ranking]
    assert ranks == sorted(ranks)
    assert kernel_result.ranking[0]["model"] == kernel_result.best_model
    scores = [row["roc_auc"] for row in kernel_result.ranking]
    assert scores == sorted(scores, reverse=True)
    assert kernel_result.quantum_beats_all_classical == (
        kernel_result.best_model == "quantum_kernel_svm"
    )
