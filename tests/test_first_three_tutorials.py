from qprac_lab.algorithms.simulation.vqe_molecular_energy import run_vqe_molecular_energy_tutorial
from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import run_qaoa_portfolio_selection_tutorial
from qprac_lab.algorithms.qml.quantum_kernel_biomedical import run_quantum_kernel_biomedical_tutorial


def test_vqe_molecular_energy_tutorial_runs():
    result = run_vqe_molecular_energy_tutorial()
    assert result.algorithm == "vqe_molecular_energy"
    assert len(result.convergence_history) > 0


def test_qaoa_portfolio_selection_tutorial_runs():
    result = run_qaoa_portfolio_selection_tutorial()
    assert result.algorithm == "qaoa_portfolio_selection"
    assert "constraint_violations" in result.constraint_report


def test_quantum_kernel_biomedical_tutorial_runs():
    result = run_quantum_kernel_biomedical_tutorial()
    assert result.algorithm == "quantum_kernel_biomedical_classification"
    assert "f1" in result.rbf_svm_metrics
    assert isinstance(result.xgboost_metrics, (dict, str))
