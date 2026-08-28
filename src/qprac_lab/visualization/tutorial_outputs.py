"""Required output artifacts for the three priority tutorials.

Every plot shows the classical baseline alongside the quantum result. A
convergence curve on its own says nothing about whether the method worked; the
same curve with the exact and Hartree-Fock lines on it says everything.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# Chosen before pyplot is imported: these run headless in CI and in scripts.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _prepare(output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_energy_convergence(
    history: list[float],
    output_path: str = "results/vqe_energy_convergence.png",
    exact_energy: float | None = None,
    hartree_fock_energy: float | None = None,
    chemical_accuracy: float = 1.6e-3,
):
    """VQE convergence against the exact and Hartree-Fock reference energies.

    Two panels, because one is not enough. The optimiser's early exploration
    spans a far wider energy range than the converged region, so on a linear axis
    the part that matters collapses onto the reference line. The log-scale error
    panel underneath is where convergence is actually readable, and it is the
    panel that shows whether chemical accuracy was reached.
    """
    _prepare(output_path)
    steps = range(len(history))

    if exact_energy is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        axes_error = None
    else:
        fig, (ax, axes_error) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
        )

    ax.plot(steps, history, label="VQE energy", color="#1f77b4", linewidth=1.8)
    if exact_energy is not None:
        ax.axhline(exact_energy, color="#2ca02c", linestyle="--", label="Exact (diagonalisation)")
    if hartree_fock_energy is not None:
        ax.axhline(
            hartree_fock_energy, color="#d62728", linestyle=":", label="Hartree-Fock baseline"
        )
    ax.set_ylabel("Energy (hartree)")
    ax.set_title("VQE energy convergence")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    if axes_error is not None:
        errors = [max(abs(value - exact_energy), 1e-16) for value in history]
        axes_error.semilogy(steps, errors, color="#1f77b4", linewidth=1.8, label="|E - E_exact|")
        axes_error.axhline(
            chemical_accuracy, color="gray", linestyle=":", label="Chemical accuracy (1.6 mHa)"
        )
        if hartree_fock_energy is not None:
            axes_error.axhline(
                abs(hartree_fock_energy - exact_energy),
                color="#d62728",
                linestyle="--",
                label="Hartree-Fock error",
            )
        axes_error.set_ylabel("Error (Ha)")
        axes_error.legend(loc="best", fontsize=8)
        axes_error.grid(alpha=0.3)
        axes_error.set_xlabel("Objective evaluation")
    else:
        ax.set_xlabel("Objective evaluation")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_dissociation_curve(
    curve: list[dict],
    output_path: str = "results/vqe_dissociation_curve.png",
):
    """H2 potential energy surface: VQE vs exact vs Hartree-Fock.

    The point of the plot is the growing gap between the HF and exact curves as
    the bond stretches -- restricted Hartree-Fock cannot describe a breaking
    bond, and VQE can.
    """
    if not curve:
        return None
    _prepare(output_path)
    bond_lengths = [row["bond_length_angstrom"] for row in curve]

    fig, (ax, ax_err) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(bond_lengths, [r["exact_energy"] for r in curve], "-", color="#2ca02c", label="Exact")
    ax.plot(
        bond_lengths,
        [r["vqe_energy"] for r in curve],
        "o",
        color="#1f77b4",
        markersize=6,
        label="VQE",
    )
    ax.plot(
        bond_lengths,
        [r["hartree_fock_energy"] for r in curve],
        "--",
        color="#d62728",
        label="Hartree-Fock",
    )
    ax.set_ylabel("Total energy (hartree)")
    ax.set_title("H$_2$ dissociation curve (STO-3G)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax_err.semilogy(
        bond_lengths,
        [max(abs(r["vqe_energy"] - r["exact_energy"]), 1e-16) for r in curve],
        "o-",
        color="#1f77b4",
        label="|VQE - exact|",
    )
    ax_err.semilogy(
        bond_lengths,
        [abs(r["hartree_fock_energy"] - r["exact_energy"]) for r in curve],
        "s--",
        color="#d62728",
        label="|HF - exact|",
    )
    ax_err.axhline(1.6e-3, color="gray", linestyle=":", label="Chemical accuracy")
    ax_err.set_xlabel("Bond length (angstrom)")
    ax_err.set_ylabel("Error (Ha)")
    ax_err.legend(loc="best", fontsize=8)
    ax_err.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_qaoa_distribution(
    top_bitstrings: list[dict],
    output_path: str = "results/qaoa_sampling_distribution.png",
    uniform_feasible_probability: float | None = None,
):
    """Sampled QAOA bitstrings, coloured by whether they satisfy the budget.

    QAOA returns a distribution rather than an answer, so this -- not the single
    best sample -- is the honest picture of what the algorithm produced.
    """
    if not top_bitstrings:
        return None
    _prepare(output_path)
    labels = [row["bitstring"] for row in top_bitstrings]
    probabilities = [row["probability"] for row in top_bitstrings]
    colors = ["#1f77b4" if row["feasible"] else "#d62728" for row in top_bitstrings]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(labels)), probabilities, color=colors)
    if uniform_feasible_probability:
        ax.axhline(
            uniform_feasible_probability,
            color="gray",
            linestyle="--",
            label="Uniform over feasible states",
        )
        ax.legend(loc="best", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8, family="monospace")
    ax.set_xlabel("Sampled bitstring (blue = satisfies budget, red = violates)")
    ax.set_ylabel("Probability")
    ax.set_title("QAOA sampling distribution")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_kernel_matrix(
    kernel_matrix,
    output_path: str = "results/kernel_matrix.png",
    title: str = "Quantum kernel matrix",
):
    """Heatmap of a kernel matrix."""
    _prepare(output_path)
    matrix = np.asarray(kernel_matrix)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="$|\\langle\\phi(x')|\\phi(x)\\rangle|^2$")
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_model_comparison(
    ranking: list[dict],
    output_path: str = "results/kernel_model_comparison.png",
):
    """ROC-AUC of the quantum kernel against every classical baseline.

    Error bars are not decoration here. The gap between the quantum kernel and
    RBF is around 0.01 ROC-AUC while the fold-to-fold standard deviation is
    nearly 0.10, so a bare bar chart would imply a decisive win the data does not
    support. Drawing the spread makes the overlap impossible to miss.
    """
    if not ranking:
        return None
    _prepare(output_path)
    models = [row["model"] for row in ranking]
    scores = [row["roc_auc"] or 0.0 for row in ranking]
    errors = [row.get("roc_auc_std") or 0.0 for row in ranking]
    colors = ["#1f77b4" if m == "quantum_kernel_svm" else "#7f7f7f" for m in models]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(range(len(models)), scores, xerr=errors, color=colors, capsize=4,
            error_kw={"ecolor": "#333333", "lw": 1.2})
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.axvline(0.5, color="black", linestyle=":", label="Random classifier")
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC-AUC (mean $\\pm$ s.d. over cross-validation folds)")
    ax.set_title("Quantum kernel vs classical baselines")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    for index, (score, error) in enumerate(zip(scores, errors, strict=True)):
        ax.text(min(score + error + 0.02, 0.97), index, f"{score:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_trotter_tradeoff(
    scaling: list[dict],
    noise_tradeoff: list[dict],
    output_path: str = "results/trotter_tradeoff.png",
):
    """Trotter error scaling, and where device noise reverses it.

    The left panel is the textbook result: more steps, less error, at both
    orders. The right panel is the one that matters for hardware -- the ideal
    curve keeps falling while the noisy curve turns around, because past some
    depth the added noise costs more than the Trotter error it removes.
    """
    if not scaling:
        return None
    _prepare(output_path)
    fig, (ax, ax_noise) = plt.subplots(1, 2, figsize=(11, 4.2))

    for order, colour, marker in ((1, "#d62728", "s"), (2, "#1f77b4", "o")):
        rows = [r for r in scaling if r["order"] == order]
        if not rows:
            continue
        ax.loglog(
            [r["steps"] for r in rows],
            [max(r["error"], 1e-16) for r in rows],
            marker=marker,
            color=colour,
            label=f"order {order}",
        )
    ax.set_xlabel("Trotter steps")
    ax.set_ylabel(r"$\||U_{\mathrm{trotter}} - U_{\mathrm{exact}}\||_2$")
    ax.set_title("Trotter error scaling (ideal)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    if noise_tradeoff:
        steps = [r["steps"] for r in noise_tradeoff]
        ax_noise.loglog(
            steps,
            [max(r["ideal_error"], 1e-16) for r in noise_tradeoff],
            "o-",
            color="#1f77b4",
            label="Ideal simulator",
        )
        ax_noise.loglog(
            steps,
            [max(r["noisy_error"], 1e-16) for r in noise_tradeoff],
            "s-",
            color="#d62728",
            label="With device noise",
        )
        best = min(noise_tradeoff, key=lambda r: r["noisy_error"])
        ax_noise.axvline(
            best["steps"], color="gray", linestyle=":", label=f"Optimum ({best['steps']} steps)"
        )
        ax_noise.set_xlabel("Trotter steps")
        ax_noise.set_ylabel("Observable error")
        ax_noise.set_title("More steps stops helping under noise")
        ax_noise.legend(fontsize=8)
        ax_noise.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_constraint_report(
    report: dict,
    output_path: str = "results/portfolio_constraint_report.json",
):
    """Persist the portfolio constraint report."""
    _prepare(output_path)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
