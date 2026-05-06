from __future__ import annotations

from qprac_lab.algorithms.simulation.vqe_molecular_energy import run_vqe_molecular_energy_tutorial
from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import run_qaoa_portfolio_selection_tutorial
from qprac_lab.algorithms.qml.quantum_kernel_biomedical import run_quantum_kernel_biomedical_tutorial
from qprac_lab.algorithms.pdes.hhl_intro import run_hhl_intro_scaffold
from qprac_lab.algorithms.pdes.variational_heat_equation import run_variational_heat_equation_scaffold
from qprac_lab.benchmarks.runner import run_and_time, save_benchmark_results


def main():
    jobs = [
        ("vqe_molecular_energy", run_vqe_molecular_energy_tutorial),
        ("qaoa_portfolio_selection", run_qaoa_portfolio_selection_tutorial),
        ("quantum_kernel_biomedical", run_quantum_kernel_biomedical_tutorial),
        ("hhl_intro", run_hhl_intro_scaffold),
        ("variational_heat_equation", run_variational_heat_equation_scaffold),
    ]

    results = [run_and_time(name, fn) for name, fn in jobs]
    df = save_benchmark_results(
        results,
        output_csv="results/benchmark_results.csv",
        output_json="results/benchmark_results.json",
    )
    print(df)


if __name__ == "__main__":
    main()
