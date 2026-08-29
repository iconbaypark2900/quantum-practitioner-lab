"""Re-run every implemented tutorial under simulated device noise.

Answers the question each tutorial's "when not to use this" section can only
gesture at on an ideal simulator: *how much of this survives real hardware?*

Writes ``results/noise_sweep.json``, ``results/noise_sweep.csv``, and
``results/noise_sweep.png``.

Noisy simulation is 60-90x slower than statevector -- Aer has to propagate a
density matrix -- so the optimisation budget is deliberately smaller here than in
the tutorials themselves. The comparison across noise levels is what matters, and
it stays fair because every level uses the same budget.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from qprac_lab.algorithms.optimization.qaoa_maxcut import run_qaoa_maxcut_tutorial
from qprac_lab.algorithms.optimization.qaoa_portfolio_selection import (
    run_qaoa_portfolio_selection_tutorial,
)
from qprac_lab.algorithms.qml.quantum_kernel_biomedical import (
    load_dataset,
    measure_self_fidelity,
)
from qprac_lab.algorithms.simulation.vqe_molecular_energy import (
    CHEMICAL_ACCURACY_HARTREE,
    run_vqe_molecular_energy_tutorial,
)
from qprac_lab.backends.noise import NOISE_PRESETS

NOISE_LEVELS = [None, "light", "moderate", "heavy"]


def _backend_kwargs(noise):
    return {"backend": "aer", "noise": noise} if noise else {"backend": "aer"}


def sweep(maxiter: int = 150, shots: int = 2048, kernel_points: int = 10) -> list[dict]:
    rows = []
    for noise in NOISE_LEVELS:
        label = noise or "ideal"
        kwargs = _backend_kwargs(noise)
        print(f"--- noise: {label} ---", flush=True)
        row: dict = {"noise": label}
        if noise:
            row.update(
                {
                    "single_qubit_error": NOISE_PRESETS[noise].single_qubit_error,
                    "two_qubit_error": NOISE_PRESETS[noise].two_qubit_error,
                    "readout_error": NOISE_PRESETS[noise].readout_error,
                }
            )

        start = time.perf_counter()
        vqe = run_vqe_molecular_energy_tutorial(
            include_dissociation_curve=False, maxiter=maxiter, **kwargs
        )
        row["vqe_energy"] = vqe.vqe_energy
        row["vqe_error_hartree"] = vqe.absolute_error
        row["vqe_chemical_accuracy"] = vqe.chemical_accuracy_reached
        row["vqe_beats_hartree_fock"] = vqe.beats_hartree_fock
        print(f"  VQE      error {vqe.absolute_error:.2e} Ha", flush=True)

        maxcut = run_qaoa_maxcut_tutorial(shots=shots, maxiter=maxiter, **kwargs)
        row["maxcut_expected_ratio"] = maxcut.expected_approximation_ratio
        row["maxcut_random_ratio"] = maxcut.random_guess_ratio
        row["maxcut_optimal_probability"] = maxcut.optimal_probability
        print(f"  Max-Cut  E[ratio] {maxcut.expected_approximation_ratio:.4f}", flush=True)

        portfolio = run_qaoa_portfolio_selection_tutorial(
            mixer="xy", reps=6, shots=shots, maxiter=maxiter, **kwargs
        )
        row["portfolio_feasible_probability"] = portfolio.feasible_probability
        row["portfolio_optimal_probability"] = portfolio.optimal_probability
        row["portfolio_lift"] = portfolio.optimal_probability_lift
        print(
            f"  QAOA XY  feasible {portfolio.feasible_probability:.1%} "
            f"lift {portfolio.optimal_probability_lift:.2f}x",
            flush=True,
        )

        x, _y, _meta = load_dataset(n_pairs=40, embedding_dim=4)
        angles = MinMaxScaler(feature_range=(0.0, np.pi)).fit_transform(x)
        fidelity = measure_self_fidelity(
            angles[:kernel_points], embedding_dim=4, noise=noise, shots=shots
        )
        row["kernel_self_fidelity"] = fidelity["mean_self_fidelity"]
        row["kernel_min_eigenvalue"] = fidelity["min_eigenvalue_without_psd_projection"]
        print(f"  Kernel   self-fidelity {fidelity['mean_self_fidelity']:.4f}", flush=True)

        row["runtime_seconds"] = time.perf_counter() - start
        rows.append(row)
    return rows


def plot(rows: list[dict], output_path: str = "results/noise_sweep.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["noise"] for r in rows]
    positions = range(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].semilogy(positions, [r["vqe_error_hartree"] for r in rows], "o-", color="#1f77b4")
    axes[0].axhline(
        CHEMICAL_ACCURACY_HARTREE, color="gray", linestyle=":", label="Chemical accuracy"
    )
    axes[0].set_ylabel("VQE error (Ha)")
    axes[0].set_title("VQE accuracy")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        positions, [r["maxcut_expected_ratio"] for r in rows], "o-", color="#1f77b4", label="QAOA"
    )
    axes[1].axhline(
        rows[0]["maxcut_random_ratio"], color="#d62728", linestyle="--", label="Random guessing"
    )
    axes[1].set_ylabel("Expected approximation ratio")
    axes[1].set_title("QAOA Max-Cut quality")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        positions,
        [r["portfolio_feasible_probability"] for r in rows],
        "o-",
        color="#2ca02c",
        label="XY feasibility",
    )
    axes[2].plot(
        positions,
        [r["kernel_self_fidelity"] for r in rows],
        "s-",
        color="#9467bd",
        label="Kernel self-fidelity",
    )
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Probability / fidelity")
    axes[2].set_title("Structural guarantees under noise")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(list(positions))
        axis.set_xticklabels(labels)
        axis.set_xlabel("Device noise")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> int:
    rows = sweep()
    Path("results").mkdir(exist_ok=True)
    Path("results/noise_sweep.json").write_text(json.dumps(rows, indent=2, default=str))
    frame = pd.DataFrame(rows)
    frame.to_csv("results/noise_sweep.csv", index=False)
    print(f"\n{frame.to_string(index=False)}")
    print(f"\nWrote results/noise_sweep.json, .csv, and {plot(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
