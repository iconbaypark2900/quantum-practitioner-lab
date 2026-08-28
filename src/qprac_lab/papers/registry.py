from __future__ import annotations

PAPER_REGISTRY = {
    "vqe": [
        "Peruzzo et al. — A variational eigenvalue solver on a photonic quantum processor",
        "Grimsley et al. — An adaptive variational algorithm for exact molecular "
        "simulations on a quantum computer",
    ],
    "qaoa": [
        "Farhi, Goldstone, Gutmann — A Quantum Approximate Optimization Algorithm",
    ],
    "hhl": [
        "Harrow, Hassidim, Lloyd — Quantum algorithm for linear systems of equations",
    ],
    "quantum_kernels": [
        "Havlíček et al. — Supervised learning with quantum-enhanced feature spaces",
    ],
}


def get_papers(topic: str):
    return PAPER_REGISTRY.get(topic, [])
