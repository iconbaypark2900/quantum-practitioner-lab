"""Run a single tutorial demo and print its structured result as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass

from qprac_lab.backends.qiskit_adapter import QiskitNotInstalledError
from qprac_lab.demo_registry import DEMOS


def to_jsonable(result):
    if is_dataclass(result):
        return asdict(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=sorted(DEMOS.keys()), required=True)
    args = parser.parse_args()

    try:
        result = to_jsonable(DEMOS[args.algorithm]())
    except QiskitNotInstalledError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
