"""Scaffolds that are still classical placeholders.

These run without the quantum stack by design. Anything promoted to a real
quantum implementation moves out of this file -- QAOA Max-Cut now lives in
``test_mixers_and_maxcut.py``.
"""

from qprac_lab.algorithms.pdes.hhl_intro import run_hhl_intro_scaffold
from qprac_lab.algorithms.pdes.variational_heat_equation import (
    run_variational_heat_equation_scaffold,
)


def test_hhl_intro_runs():
    result = run_hhl_intro_scaffold()
    assert result["algorithm"] == "hhl_intro"
    assert result["use_case"]
    assert result["algorithm_type"]


def test_variational_heat_equation_runs():
    result = run_variational_heat_equation_scaffold()
    assert result["algorithm"] == "variational_heat_equation"
    assert result["use_case"]
    assert result["algorithm_type"]
