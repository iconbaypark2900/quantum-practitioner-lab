"""Re-run key results through PennyLane and compare against Qiskit.

Independent verification. Several results in this project came within one silent
bug of being wrong, and every one of those bugs produced plausible numbers rather
than an error. A second, unrelated stack agreeing to machine precision is the
cheapest evidence that a result is physics rather than one library's conventions.

Writes ``results/cross_check.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from qprac_lab.backends.pennylane_adapter import (
    PennyLaneBackendAdapter,
    cross_check_ising_mapping,
    cross_check_vqe,
    pennylane_available,
)


def main() -> int:
    if not pennylane_available():
        print(
            'error: PennyLane is not installed. Install it with: pip install -e ".[pennylane]"',
            file=sys.stderr,
        )
        return 1

    print("VQE (H2) across bond lengths ...", flush=True)
    vqe_checks = [cross_check_vqe(bond_length) for bond_length in (0.735, 1.0, 2.5)]
    for check in vqe_checks:
        print(
            f"  R = {check['bond_length_angstrom']:.3f} A: "
            f"qiskit {check['qiskit_energy']:.9f} vs pennylane {check['pennylane_energy']:.9f} "
            f"-> |diff| {check['absolute_difference']:.2e}, agree: {check['frameworks_agree']}"
        )

    print("\nQUBO -> Ising mapping over every assignment ...", flush=True)
    ising = cross_check_ising_mapping()
    print(
        f"  {ising['assignments_checked']} assignments, "
        f"max |diff| {ising['max_absolute_difference']:.2e}, agree: {ising['frameworks_agree']}"
    )

    payload = {
        "backend": PennyLaneBackendAdapter().describe(),
        "vqe": vqe_checks,
        "ising_mapping": ising,
        "all_agree": all(c["frameworks_agree"] for c in vqe_checks) and ising["frameworks_agree"],
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/cross_check.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nall checks agree: {payload['all_agree']}")
    print("Wrote results/cross_check.json")
    return 0 if payload["all_agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
