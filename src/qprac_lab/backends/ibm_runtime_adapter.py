from __future__ import annotations


class IBMRuntimeAdapter:
    """Placeholder for IBM Quantum Runtime integration."""

    name = "ibm_runtime_adapter"

    def describe(self):
        return {
            "backend": self.name,
            "status": "placeholder",
            "next_step": "Add service authentication, backend selection, Sampler, and Estimator.",
        }
