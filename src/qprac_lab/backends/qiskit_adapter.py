from __future__ import annotations


class QiskitBackendAdapter:
    """Placeholder for future Qiskit Aer / Runtime integration."""

    name = "qiskit_adapter"

    def describe(self):
        return {
            "backend": self.name,
            "status": "placeholder",
            "next_step": "Implement Aer simulator, Sampler, Estimator, and Runtime options.",
        }
