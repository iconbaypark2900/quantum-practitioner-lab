"""Run the two-qubit H2 VQE on real IBM hardware, once, by hand.

This does **not** contradict ``DECISIONS.md`` #7. That decision dropped IBM
Runtime as a *backend adapter*: it needs credentials CI cannot hold, cannot be
exercised in CI, and drives the same Qiskit stack, so as a dependency it would
verify nothing. All three arguments are about a dependency. None apply to a
one-off run whose output is committed as data.

Nothing in the repository imports this script, no CI job runs it, and no
credential is read from disk -- the token comes from the environment only.

Usage
-----

    pip install -e ".[qiskit,hardware]"
    export QISKIT_IBM_TOKEN=...            # never commit this
    python scripts/run_hardware_vqe.py

    # Exercise the whole path with no credentials, against a local noise preset:
    python scripts/run_hardware_vqe.py --dry-run moderate

Modes
-----

``energy`` (default)
    Optimise on the simulator, then evaluate the energy **once** on hardware at
    the optimal angle. One job, queue-friendly, and it isolates the quantity that
    matters: how much of the prepared state the device actually delivers.

``vqe``
    Run the whole optimisation loop on hardware. Apples-to-apples with the noise
    sweep, which also optimises under noise -- and far slower, because every
    objective evaluation is a queued job.

The distinction is reported in the output, because the two measure different
things and conflating them would overstate whichever is more flattering.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from qprac_lab.algorithms.simulation.hamiltonian_utils import build_h2_hamiltonian
from qprac_lab.algorithms.simulation.vqe_molecular_energy import (
    CHEMICAL_ACCURACY_HARTREE,
    build_ansatz,
    run_vqe,
)
from qprac_lab.backends.noise import NOISE_PRESETS

#: Held equal to scripts/run_noise_sweep.py so the comparison is paired.
MAXITER = 150
SHOTS = 2048
SEED = 42
BOND_LENGTH = 0.735

#: The real run writes here, and this path is exempted from .gitignore because the
#: result exists to be committed as data.
OUTPUT = Path("results/hardware_vqe_h2.json")

#: A dry run writes somewhere else, and stays ignored. A simulated number sitting
#: at the hardware path would be committed as though a device produced it, which is
#: the one mistake this whole script is arranged to avoid.
DRY_RUN_OUTPUT = Path("results/hardware_vqe_h2.dryrun.json")


class MissingCredentials(RuntimeError):
    """Raised with an actionable message rather than a stack trace."""


def _load_service():
    """Connect to IBM Quantum, reading the token from the environment only."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as error:  # pragma: no cover - optional extra
        raise MissingCredentials(
            "qiskit-ibm-runtime is not installed. It is deliberately not a core "
            'dependency (DECISIONS.md #7); install it for this run with:\n'
            '    pip install -e ".[qiskit,hardware]"'
        ) from error

    token = os.environ.get("QISKIT_IBM_TOKEN")
    if not token:
        raise MissingCredentials(
            "QISKIT_IBM_TOKEN is not set. Export it for this shell only:\n"
            "    export QISKIT_IBM_TOKEN=...\n"
            "Do not write it into a file inside this repository."
        )
    return QiskitRuntimeService(channel="ibm_quantum", token=token)


def backend_metadata(backend) -> dict:
    """Everything a simulator never has to record, and a device run is worthless without.

    A hardware number with no calibration timestamp is not reproducible even in
    principle: the same circuit on the same backend a day later is a different
    experiment.
    """
    meta: dict = {"backend": getattr(backend, "name", str(backend))}
    try:
        properties = backend.properties()
    except Exception:  # pragma: no cover - simulators have no properties
        properties = None
    if properties is None:
        return meta

    meta["calibration_timestamp"] = str(getattr(properties, "last_update_date", None))
    errors = {}
    for qubit in range(min(2, backend.num_qubits)):
        try:
            errors[f"q{qubit}"] = {
                "readout_error": properties.readout_error(qubit),
                "t1_us": properties.t1(qubit) * 1e6,
                "t2_us": properties.t2(qubit) * 1e6,
            }
        except Exception:  # pragma: no cover
            pass
    meta["qubit_properties"] = errors
    return meta


def _simulated_optimum(hamiltonian):
    """The angle the simulator says is optimal, and the energy it predicts there."""
    result, history, _ansatz = run_vqe(
        hamiltonian,
        ansatz_kind="two_qubit_uccsd",
        maxiter=MAXITER,
        backend="statevector",
        seed=SEED,
    )
    return np.atleast_1d(result.x), float(result.fun), len(history)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=("energy", "vqe"), default="energy")
    parser.add_argument(
        "--dry-run",
        metavar="PRESET",
        choices=sorted(NOISE_PRESETS),
        help="use a local Aer noise preset instead of hardware; no credentials needed",
    )
    parser.add_argument("--backend", help="force a specific backend name")
    parser.add_argument("--shots", type=int, default=SHOTS)
    args = parser.parse_args(argv)

    hamiltonian = build_h2_hamiltonian(BOND_LENGTH)
    exact = hamiltonian.exact_total_energy()
    theta, simulated_electronic, evaluations = _simulated_optimum(hamiltonian)
    simulated_total = hamiltonian.total_energy(simulated_electronic)

    ansatz = build_ansatz(
        hamiltonian.num_qubits,
        kind="two_qubit_uccsd",
        hf_bitstring=hamiltonian.hartree_fock_bitstring,
    )

    if args.dry_run:
        from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter

        adapter = QiskitBackendAdapter(
            backend="aer", shots=args.shots, seed=SEED, noise=args.dry_run
        )
        estimator = adapter.estimator()
        prepared = adapter.prepare(ansatz)
        observable = hamiltonian.qubit_operator
        meta = {
            "backend": f"aer:{args.dry_run}",
            "simulated_preset": NOISE_PRESETS[args.dry_run].describe(),
        }
    else:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        # _load_service() first: it turns a missing package or token into an
        # actionable message, and a bare import above would pre-empt it with a
        # ModuleNotFoundError traceback.
        service = _load_service()
        from qiskit_ibm_runtime import EstimatorV2
        backend = (
            service.backend(args.backend)
            if args.backend
            else service.least_busy(operational=True, simulator=False, min_num_qubits=2)
        )
        print(f"backend: {backend.name}")
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        prepared = pass_manager.run(ansatz)
        observable = hamiltonian.qubit_operator.apply_layout(prepared.layout)
        estimator = EstimatorV2(mode=backend)
        estimator.options.default_shots = args.shots
        meta = backend_metadata(backend)

    def energy_at(parameters) -> float:
        job = estimator.run([(prepared, observable, list(np.atleast_1d(parameters)))])
        return float(job.result()[0].data.evs)

    if args.mode == "energy":
        measured_electronic = energy_at(theta)
        device_evaluations = 1
    else:
        outcome = minimize(
            lambda p: energy_at(p),
            x0=np.zeros(ansatz.num_parameters),
            method="COBYLA",
            options={"maxiter": MAXITER},
        )
        measured_electronic = float(outcome.fun)
        device_evaluations = int(outcome.nfev)

    measured_total = hamiltonian.total_energy(measured_electronic)
    error = abs(measured_total - exact)

    payload = {
        "hardware": not args.dry_run,
        "mode": args.mode,
        "measures": (
            "energy at the simulator-optimal angle -- device state fidelity, NOT the "
            "full-loop error the noise sweep reports"
            if args.mode == "energy"
            else "full optimisation on device -- comparable to the noise sweep"
        ),
        "molecule": "H2",
        "bond_length_angstrom": BOND_LENGTH,
        "num_qubits": hamiltonian.num_qubits,
        "ansatz": "two_qubit_uccsd",
        "shots": args.shots,
        "seed": SEED,
        "optimal_theta_from_simulator": [float(v) for v in theta],
        "simulator_evaluations": evaluations,
        "device_evaluations": device_evaluations,
        "transpiled_depth": prepared.depth(),
        "transpiled_two_qubit_gates": sum(
            count for gate, count in prepared.count_ops().items() if gate in {"cx", "cz", "ecr"}
        ),
        "exact_energy": exact,
        "simulated_energy": simulated_total,
        "measured_energy": measured_total,
        "absolute_error_hartree": error,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "chemical_accuracy_reached": bool(error < CHEMICAL_ACCURACY_HARTREE),
        **meta,
    }

    destination = DRY_RUN_OUTPUT if args.dry_run else OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"  exact      {exact:.9f} Ha")
    print(f"  simulator  {simulated_total:.9f} Ha")
    print(f"  measured   {measured_total:.9f} Ha")
    print(f"  error      {error:.3e} Ha  (chemical accuracy: {CHEMICAL_ACCURACY_HARTREE:.1e})")
    print(f"  depth {payload['transpiled_depth']}, "
          f"{payload['transpiled_two_qubit_gates']} two-qubit gates")
    print()
    print("Which preset did the device resemble?")
    for name, spec in NOISE_PRESETS.items():
        print(f"  {name:9s} 2q error {spec.two_qubit_error:<8g} readout {spec.readout_error}")
    print()
    print("Do NOT retune the presets to match this. They are a measurement, and")
    print("fitting them to a device converts one into the other.")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MissingCredentials as error:
        raise SystemExit(f"error: {error}") from None
