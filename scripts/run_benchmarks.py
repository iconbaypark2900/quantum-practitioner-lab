"""Benchmark every runnable tutorial and write a comparison table.

Writes ``results/benchmark_results.csv`` (flat, readable) and
``results/benchmark_results.json`` (full payloads).
"""

from __future__ import annotations

import sys

from qprac_lab.algorithms.optimization.qaoa_maxcut import run_qaoa_maxcut_tutorial
from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import (
    run_qaoa_portfolio_selection_tutorial,
)
from qprac_lab.algorithms.pdes.hhl_intro import run_hhl_intro_scaffold
from qprac_lab.algorithms.pdes.variational_heat_equation import (
    run_variational_heat_equation_scaffold,
)
from qprac_lab.algorithms.qml.quantum_kernel_biomedical import (
    run_quantum_kernel_biomedical_tutorial,
)
from qprac_lab.algorithms.simulation.adapt_vqe_materials import run_adapt_vqe_materials
from qprac_lab.algorithms.simulation.vqe_molecular_energy import run_vqe_molecular_energy_tutorial
from qprac_lab.backends.qiskit_adapter import qiskit_available
from qprac_lab.benchmarks.runner import run_and_time, save_benchmark_results

#: Tutorials backed by real quantum implementations; these need the qiskit extra.
QUANTUM_JOBS = [
    ("vqe_molecular_energy", run_vqe_molecular_energy_tutorial),
    ("adapt_vqe_materials", run_adapt_vqe_materials),
    ("qaoa_portfolio_selection", run_qaoa_portfolio_selection_tutorial),
    ("qaoa_maxcut", run_qaoa_maxcut_tutorial),
    ("quantum_kernel_biomedical", run_quantum_kernel_biomedical_tutorial),
]

#: Still classical scaffolds; they run without the quantum stack.
SCAFFOLD_JOBS = [
    ("hhl_intro", run_hhl_intro_scaffold),
    ("variational_heat_equation", run_variational_heat_equation_scaffold),
]


def main() -> int:
    jobs = list(SCAFFOLD_JOBS)
    if qiskit_available():
        jobs = QUANTUM_JOBS + jobs
    else:
        print(
            'warning: qiskit is not installed, so the three quantum tutorials are '
            'skipped. Install them with: pip install -e ".[qiskit]"',
            file=sys.stderr,
        )

    scaffold_names = {name for name, _ in SCAFFOLD_JOBS}
    results = []
    for name, fn in jobs:
        print(f"running {name} ...", flush=True)
        # Scaffolds have no quantum backend to report; saying "statevector" for
        # them would overstate what the row actually measured.
        backend = "classical_scaffold" if name in scaffold_names else "statevector"
        results.append(run_and_time(name, fn, backend=backend))

    frame = save_benchmark_results(
        results,
        output_csv="results/benchmark_results.csv",
        output_json="results/benchmark_results.json",
    )
    print()
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
