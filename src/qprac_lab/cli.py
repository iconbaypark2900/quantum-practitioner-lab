from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass

from qprac_lab.algorithms.simulation.vqe_molecular_energy import run_vqe_molecular_energy_tutorial
from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import run_qaoa_portfolio_selection_tutorial
from qprac_lab.algorithms.qml.quantum_kernel_biomedical import run_quantum_kernel_biomedical_tutorial


DEMOS = {
    "vqe_molecular_energy": run_vqe_molecular_energy_tutorial,
    "qaoa_portfolio_selection": run_qaoa_portfolio_selection_tutorial,
    "quantum_kernel_biomedical": run_quantum_kernel_biomedical_tutorial,
}


def to_jsonable(result):
    if is_dataclass(result):
        return asdict(result)
    return result


def main():
    parser = argparse.ArgumentParser(prog="qprac-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--algorithm", choices=DEMOS.keys(), required=True)

    args = parser.parse_args()

    if args.command == "demo":
        result = to_jsonable(DEMOS[args.algorithm]())
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
