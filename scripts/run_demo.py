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
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=sorted(DEMOS.keys()), required=True)
    args = parser.parse_args()

    result = to_jsonable(DEMOS[args.algorithm]())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
