from __future__ import annotations


class CudaQBackendAdapter:
    """Placeholder for future CUDA-Q integration."""

    name = "cudaq_adapter"

    def describe(self):
        return {
            "backend": self.name,
            "status": "placeholder",
            "next_step": "Implement CPU target first, then GPU target where available.",
        }
