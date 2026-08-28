from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qprac_lab.algorithms.optimization.logistics_routing_qubo import (
    run_logistics_routing_qubo_scaffold,
)
from qprac_lab.algorithms.optimization.qaoa_maxcut import run_qaoa_maxcut_scaffold
from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import (
    run_qaoa_portfolio_selection_tutorial,
)
from qprac_lab.algorithms.pdes.black_scholes_pde import run_black_scholes_pde_scaffold
from qprac_lab.algorithms.pdes.hhl_intro import run_hhl_intro_scaffold
from qprac_lab.algorithms.pdes.variational_heat_equation import (
    run_variational_heat_equation_scaffold,
)
from qprac_lab.algorithms.qml.qsvc_classifier import run_qsvc_classifier_scaffold
from qprac_lab.algorithms.qml.quantum_kernel_biomedical import (
    run_quantum_kernel_biomedical_tutorial,
)
from qprac_lab.algorithms.qml.vqc_classifier import run_vqc_classifier_scaffold
from qprac_lab.algorithms.simulation.adapt_vqe_materials import run_adapt_vqe_materials_scaffold
from qprac_lab.algorithms.simulation.trotterization import run_trotterization_scaffold
from qprac_lab.algorithms.simulation.vqe_molecular_energy import run_vqe_molecular_energy_tutorial

#: Demos backed by real quantum implementations; these need the ``qiskit`` extra.
#: Everything else is still a classical scaffold and runs with the core deps alone.
QUANTUM_DEMOS: frozenset[str] = frozenset(
    {
        "vqe_molecular_energy",
        "qaoa_portfolio_selection",
        "quantum_kernel_biomedical",
    }
)

DEMOS: dict[str, Callable[[], Any]] = {
    "vqe_molecular_energy": run_vqe_molecular_energy_tutorial,
    "adapt_vqe_materials": run_adapt_vqe_materials_scaffold,
    "trotterization": run_trotterization_scaffold,
    "qaoa_portfolio_selection": run_qaoa_portfolio_selection_tutorial,
    "qaoa_maxcut": run_qaoa_maxcut_scaffold,
    "logistics_routing_qubo": run_logistics_routing_qubo_scaffold,
    "hhl_intro": run_hhl_intro_scaffold,
    "variational_heat_equation": run_variational_heat_equation_scaffold,
    "black_scholes_pde": run_black_scholes_pde_scaffold,
    "quantum_kernel_biomedical": run_quantum_kernel_biomedical_tutorial,
    "qsvc_classifier": run_qsvc_classifier_scaffold,
    "vqc_classifier": run_vqc_classifier_scaffold,
}


def requires_qiskit(algorithm: str) -> bool:
    """Whether a demo needs the optional quantum stack."""
    return algorithm in QUANTUM_DEMOS


def describe_demos() -> list[dict[str, Any]]:
    """List every demo with its implementation level, for CLI help and docs."""
    return [
        {
            "algorithm": name,
            "implementation": "quantum" if requires_qiskit(name) else "classical_scaffold",
            "requires_qiskit": requires_qiskit(name),
        }
        for name in sorted(DEMOS)
    ]
