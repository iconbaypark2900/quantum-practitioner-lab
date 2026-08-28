from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass

from qprac_lab.backends.qiskit_adapter import (
    QiskitNotInstalledError,
    qiskit_available,
    qiskit_versions,
)
from qprac_lab.demo_registry import DEMOS, describe_demos


def to_jsonable(result):
    if is_dataclass(result):
        return asdict(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qprac-lab",
        description="Quantum Practitioner Lab: runnable tutorials with classical baselines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="run a single tutorial demo")
    demo_parser.add_argument("--algorithm", choices=sorted(DEMOS.keys()), required=True)

    subparsers.add_parser("list", help="list demos and their implementation level")
    subparsers.add_parser("env", help="report the installed quantum stack")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        for entry in describe_demos():
            marker = "quantum" if entry["requires_qiskit"] else "scaffold"
            print(f"{entry['algorithm']:32s} [{marker}]")
        return 0

    if args.command == "env":
        print(
            json.dumps(
                {"qiskit_available": qiskit_available(), "versions": qiskit_versions()},
                indent=2,
            )
        )
        return 0

    try:
        result = to_jsonable(DEMOS[args.algorithm]())
    except QiskitNotInstalledError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
