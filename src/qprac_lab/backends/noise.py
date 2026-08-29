"""Device noise models for the Aer simulator.

Shot noise -- already supported through ``QiskitBackendAdapter(shots=...)`` -- is
only the statistical part of the story: it shrinks as ``1/sqrt(shots)`` and
vanishes given enough sampling. **Device noise does not.** Gates misapply,
qubits decohere, and readout misreports, and no amount of sampling averages that
away. It is the difference between "this algorithm needs a lot of shots" and
"this algorithm does not work on current hardware".

Presets are depolarizing gate errors plus readout error, with rates chosen to
bracket current superconducting hardware. ``moderate`` is roughly a present-day
IBM device. Amplitude damping and crosstalk are deliberately excluded -- they
need gate-duration and topology modelling that would obscure the one variable
these tutorials are trying to isolate.

**Transpilation matters here.** Noise attaches to specific gate names, so a
circuit containing gates outside the model's basis has those gates applied
*perfectly*. That failure is not loud: the run succeeds and simply reports less
noise than the device would. :meth:`QiskitBackendAdapter.prepare` transpiles to
the noise basis to avoid it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NoiseSpec:
    """Error rates for a depolarizing + readout noise model."""

    name: str
    single_qubit_error: float
    two_qubit_error: float
    readout_error: float
    description: str = ""
    single_qubit_gates: tuple[str, ...] = field(
        default=("id", "rz", "sx", "x", "u", "u1", "u2", "u3")
    )
    two_qubit_gates: tuple[str, ...] = field(default=("cx", "cz", "ecr", "rzz"))

    def describe(self) -> dict:
        return {
            "preset": self.name,
            "single_qubit_error": self.single_qubit_error,
            "two_qubit_error": self.two_qubit_error,
            "readout_error": self.readout_error,
            "description": self.description,
        }


#: Error rates bracketing current superconducting hardware.
NOISE_PRESETS: dict[str, NoiseSpec] = {
    "light": NoiseSpec(
        "light",
        single_qubit_error=1e-4,
        two_qubit_error=1e-3,
        readout_error=5e-3,
        description="optimistic near-term hardware, better than most devices today",
    ),
    "moderate": NoiseSpec(
        "moderate",
        single_qubit_error=3e-4,
        two_qubit_error=6e-3,
        readout_error=1.5e-2,
        description="roughly a present-day IBM superconducting device",
    ),
    "heavy": NoiseSpec(
        "heavy",
        single_qubit_error=1e-3,
        two_qubit_error=2e-2,
        readout_error=4e-2,
        description="a poorly calibrated device, or deep circuits on a mediocre one",
    ),
}


def noise_spec(preset: str) -> NoiseSpec:
    """Look up a preset, with an actionable error for unknown names."""
    try:
        return NOISE_PRESETS[preset]
    except KeyError:
        raise ValueError(
            f"Unknown noise preset {preset!r}; expected one of {sorted(NOISE_PRESETS)} "
            "(or None for an ideal simulator)"
        ) from None


def build_noise_model(preset: str):
    """Build an Aer ``NoiseModel`` from a preset name."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    spec = noise_spec(preset)
    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(spec.single_qubit_error, 1), list(spec.single_qubit_gates)
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error(spec.two_qubit_error, 2), list(spec.two_qubit_gates)
    )
    if spec.readout_error > 0:
        probability = spec.readout_error
        model.add_all_qubit_readout_error(
            ReadoutError([[1 - probability, probability], [probability, 1 - probability]])
        )
    return model


def noise_basis_gates(preset: str) -> list[str]:
    """Gate set a circuit must be transpiled into for the noise to fully apply."""
    spec = noise_spec(preset)
    return sorted({"id", "rz", "sx", "x", "cx"} | set(spec.two_qubit_gates) - {"rzz"})
