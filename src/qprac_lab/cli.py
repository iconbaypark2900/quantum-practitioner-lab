from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass

from qprac_lab.demo_registry import DEMOS


def to_jsonable(result):
    if is_dataclass(result):
        return asdict(result)
    return result


def main():
    parser = argparse.ArgumentParser(prog="qprac-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--algorithm", choices=sorted(DEMOS.keys()), required=True)

    args = parser.parse_args()

    if args.command == "demo":
        result = to_jsonable(DEMOS[args.algorithm]())
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
