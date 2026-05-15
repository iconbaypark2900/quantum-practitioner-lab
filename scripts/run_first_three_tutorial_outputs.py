from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from qprac_lab.algorithms.simulation.vqe_molecular_energy import run_vqe_molecular_energy_tutorial
from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import run_qaoa_portfolio_selection_tutorial
from qprac_lab.algorithms.qml.quantum_kernel_biomedical import run_quantum_kernel_biomedical_tutorial
from qprac_lab.visualization.tutorial_outputs import (
    plot_energy_convergence,
    plot_kernel_matrix,
    save_constraint_report,
)


def main():
    Path("results").mkdir(exist_ok=True)

    vqe = run_vqe_molecular_energy_tutorial()
    qaoa = run_qaoa_portfolio_selection_tutorial()
    qkernel = run_quantum_kernel_biomedical_tutorial()

    energy_plot = plot_energy_convergence(vqe.convergence_history)
    kernel_plot = plot_kernel_matrix(qkernel.kernel_matrix_preview)
    constraint_report_path = save_constraint_report(qaoa.constraint_report)

    payload = {
        "vqe_molecular_energy": asdict(vqe),
        "qaoa_portfolio_selection": asdict(qaoa),
        "quantum_kernel_biomedical": asdict(qkernel),
        "outputs": {
            "energy_convergence_plot": energy_plot,
            "kernel_matrix_plot": kernel_plot,
            "constraint_report": constraint_report_path,
        },
    }

    output_file = Path("results/first_three_tutorial_outputs.json")
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
