from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_energy_convergence(history: list[float], output_path: str = "results/vqe_energy_convergence.png"):
    """Create the required VQE energy convergence plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(len(history)), history)
    plt.xlabel("Iteration / parameter step")
    plt.ylabel("Energy")
    plt.title("VQE Energy Convergence")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_kernel_matrix(kernel_matrix, output_path: str = "results/kernel_matrix.png"):
    """Create the required kernel matrix visualization."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(kernel_matrix)
    plt.figure()
    plt.imshow(matrix)
    plt.colorbar()
    plt.title("Kernel Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_constraint_report(report: dict, output_path: str = "results/portfolio_constraint_report.json"):
    import json
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
