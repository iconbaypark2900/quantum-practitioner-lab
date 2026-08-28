"""Generate every required artifact for the three priority tutorials.

Outputs land in ``results/``:

* ``vqe_energy_convergence.png``     -- required output for tutorial 1
* ``vqe_dissociation_curve.png``     -- use-case plot (needs the ``nature`` extra)
* ``qaoa_sampling_distribution.png`` -- what QAOA actually sampled
* ``portfolio_constraint_report.json`` -- required output for tutorial 2
* ``kernel_matrix.png``              -- required output for tutorial 3
* ``kernel_model_comparison.png``    -- quantum kernel vs classical baselines
* ``first_three_tutorial_outputs.json`` -- the full structured payload
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import (
    run_qaoa_portfolio_selection_tutorial,
)
from qprac_lab.algorithms.qml.quantum_kernel_biomedical import (
    run_quantum_kernel_biomedical_tutorial,
)
from qprac_lab.algorithms.simulation.vqe_molecular_energy import run_vqe_molecular_energy_tutorial
from qprac_lab.backends.qiskit_adapter import QiskitNotInstalledError, qiskit_versions
from qprac_lab.visualization.tutorial_outputs import (
    plot_dissociation_curve,
    plot_energy_convergence,
    plot_kernel_matrix,
    plot_model_comparison,
    plot_qaoa_distribution,
    save_constraint_report,
)


def main() -> int:
    Path("results").mkdir(exist_ok=True)

    print("[1/3] VQE for molecular energy ...", flush=True)
    vqe = run_vqe_molecular_energy_tutorial()
    print(
        f"      VQE {vqe.vqe_energy:.6f} Ha vs exact {vqe.exact_baseline_energy:.6f} Ha "
        f"(error {vqe.absolute_error:.2e}, chemical accuracy: {vqe.chemical_accuracy_reached})"
    )

    print("[2/3] QAOA for portfolio selection ...", flush=True)
    qaoa = run_qaoa_portfolio_selection_tutorial()
    print(
        f"      assets {qaoa.selected_assets} objective {qaoa.objective_value:.6f} "
        f"(matches brute force: {qaoa.matches_brute_force}, "
        f"feasible {qaoa.feasible_probability:.1%}, "
        f"lift over uniform {qaoa.optimal_probability_lift:.2f}x)"
    )

    print("[3/3] Quantum kernel for biomedical classification ...", flush=True)
    kernel = run_quantum_kernel_biomedical_tutorial()
    print(
        f"      best model: {kernel.best_model} "
        f"(quantum beats all classical: {kernel.quantum_beats_all_classical})"
    )

    outputs = {
        "energy_convergence_plot": plot_energy_convergence(
            vqe.convergence_history,
            exact_energy=vqe.exact_baseline_energy,
            hartree_fock_energy=vqe.hartree_fock_baseline_energy,
        ),
        "dissociation_curve_plot": plot_dissociation_curve(vqe.dissociation_curve),
        "qaoa_distribution_plot": plot_qaoa_distribution(
            qaoa.top_bitstrings,
            uniform_feasible_probability=qaoa.uniform_feasible_probability,
        ),
        "constraint_report": save_constraint_report(qaoa.constraint_report),
        "kernel_matrix_plot": plot_kernel_matrix(kernel.kernel_matrix_preview),
        "model_comparison_plot": plot_model_comparison(kernel.ranking),
    }

    payload = {
        "environment": {"qiskit_versions": qiskit_versions()},
        "vqe_molecular_energy": asdict(vqe),
        "qaoa_portfolio_selection": asdict(qaoa),
        "quantum_kernel_biomedical": asdict(kernel),
        "outputs": outputs,
    }

    output_file = Path("results/first_three_tutorial_outputs.json")
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWrote {output_file}")
    for name, path in outputs.items():
        print(f"  {name}: {path if path else 'skipped'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QiskitNotInstalledError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
