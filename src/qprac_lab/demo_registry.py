from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qprac_lab.algorithms.optimization.logistics_routing_qubo import (
    run_logistics_routing_qubo_scaffold,
)
from qprac_lab.algorithms.optimization.qaoa_maxcut import run_qaoa_maxcut_tutorial
from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import (
    run_qaoa_portfolio_selection_tutorial,
)
from qprac_lab.algorithms.pdes.black_scholes_pde import run_black_scholes_pde_tutorial
from qprac_lab.algorithms.pdes.hhl_intro import run_hhl_intro_tutorial
from qprac_lab.algorithms.pdes.variational_heat_equation import (
    run_variational_heat_equation_tutorial,
)
from qprac_lab.algorithms.qml.quantum_kernel_biomedical import (
    run_quantum_kernel_biomedical_tutorial,
)
from qprac_lab.algorithms.qml.vqc_classifier import run_vqc_classifier_tutorial
from qprac_lab.algorithms.simulation.adapt_vqe_materials import run_adapt_vqe_materials
from qprac_lab.algorithms.simulation.trotterization import run_trotterization_tutorial
from qprac_lab.algorithms.simulation.vqe_molecular_energy import (
    run_vqe_molecular_energy_tutorial,
)

#: Demos backed by real quantum implementations; these need the ``qiskit`` extra.
#: Anything absent is still a classical scaffold and runs on the core deps alone.
#: ``tests/test_configs.py`` asserts this matches ``configs/algorithms.yaml``.
QUANTUM_DEMOS: frozenset[str] = frozenset(
    {
        "vqe_molecular_energy",
        "adapt_vqe_materials",
        "trotterization",
        "qaoa_portfolio_selection",
        "qaoa_maxcut",
        "quantum_kernel_biomedical",
        "vqc_classifier",
        "hhl_intro",
        "variational_heat_equation",
        "black_scholes_pde",
    }
)

DEMOS: dict[str, Callable[[], Any]] = {
    "vqe_molecular_energy": run_vqe_molecular_energy_tutorial,
    "adapt_vqe_materials": run_adapt_vqe_materials,
    "trotterization": run_trotterization_tutorial,
    "qaoa_portfolio_selection": run_qaoa_portfolio_selection_tutorial,
    "qaoa_maxcut": run_qaoa_maxcut_tutorial,
    "logistics_routing_qubo": run_logistics_routing_qubo_scaffold,
    "hhl_intro": run_hhl_intro_tutorial,
    "variational_heat_equation": run_variational_heat_equation_tutorial,
    "black_scholes_pde": run_black_scholes_pde_tutorial,
    "quantum_kernel_biomedical": run_quantum_kernel_biomedical_tutorial,
    "vqc_classifier": run_vqc_classifier_tutorial,
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
