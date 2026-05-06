from qprac_lab.algorithms.pdes.hhl_intro import run_hhl_intro_scaffold
from qprac_lab.algorithms.pdes.variational_heat_equation import run_variational_heat_equation_scaffold
from qprac_lab.algorithms.optimization.qaoa_maxcut import run_qaoa_maxcut_scaffold


def test_hhl_intro_runs():
    result = run_hhl_intro_scaffold()
    assert result["algorithm"] == "hhl_intro"


def test_variational_heat_equation_runs():
    result = run_variational_heat_equation_scaffold()
    assert result["algorithm"] == "variational_heat_equation"


def test_qaoa_maxcut_runs():
    result = run_qaoa_maxcut_scaffold()
    assert result["algorithm"] == "qaoa_maxcut"
