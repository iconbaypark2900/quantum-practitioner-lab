"""Qiskit 2.x backend adapter.

Qiskit is an optional extra, not a core dependency, so this module is written to
be importable without it. ``qiskit_available()`` is the cheap probe; asking for a
primitive on a machine without the extra raises :class:`QiskitNotInstalledError`
with an actionable hint instead of an ``ImportError`` from deep in a call stack.

Everything here targets the Qiskit 2.x **V2** primitives. The V1 ``Estimator`` and
``Sampler`` were removed in Qiskit 2.0, so V2 is the only interface available.
"""

from __future__ import annotations

from dataclasses import dataclass

from qprac_lab.backends.noise import (
    build_noise_model,
    noise_basis_gates,
    noise_spec,
)

INSTALL_HINT = 'install the quantum stack with: pip install -e ".[qiskit]"'

#: A universal basis Aer executes natively. Transpiling into it is exact -- it
#: only rewrites gates, never approximates them.
AER_BASIS_GATES = ["id", "rz", "sx", "x", "cx"]

#: Backends this adapter knows how to build primitives for.
SUPPORTED_BACKENDS = ("statevector", "aer")


class QiskitNotInstalledError(RuntimeError):
    """Raised when a quantum code path is reached without the ``qiskit`` extra."""

    def __init__(self, what: str = "This code path") -> None:
        super().__init__(f"{what} requires Qiskit; {INSTALL_HINT}")


def qiskit_available() -> bool:
    """Return ``True`` when the ``qiskit`` extra is importable."""
    try:
        import qiskit  # noqa: F401
    except ImportError:
        return False
    return True


def require_qiskit(what: str = "This code path") -> None:
    """Raise :class:`QiskitNotInstalledError` unless Qiskit is importable."""
    if not qiskit_available():
        raise QiskitNotInstalledError(what)


def qiskit_versions() -> dict[str, str]:
    """Report installed versions of the quantum stack, for benchmark provenance."""
    versions: dict[str, str] = {}
    for module_name, label in (
        ("qiskit", "qiskit"),
        ("qiskit_aer", "qiskit-aer"),
        ("qiskit_machine_learning", "qiskit-machine-learning"),
        ("qiskit_optimization", "qiskit-optimization"),
        ("qiskit_nature", "qiskit-nature"),
    ):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        versions[label] = getattr(module, "__version__", "unknown")
    return versions


@dataclass
class QiskitBackendAdapter:
    """Resolve Qiskit V2 primitives for a named local simulation backend.

    Parameters
    ----------
    backend:
        ``"statevector"`` for Qiskit's reference simulators, ``"aer"`` for the
        higher-performance Aer simulators.
    shots:
        Sampling budget. ``None`` means *exact* expectation values from the
        statevector: the estimator runs at zero precision rather than sampling.
        An integer turns on shot noise, which is what makes a benchmark against
        hardware meaningful.
    seed:
        Seed for reproducible sampling. Results are deterministic given a seed.
    """

    backend: str = "statevector"
    shots: int | None = None
    seed: int | None = 42
    noise: str | None = None

    name = "qiskit_adapter"

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend {self.backend!r}; expected one of {SUPPORTED_BACKENDS}"
            )
        if self.shots is not None and self.shots <= 0:
            raise ValueError(f"shots must be positive or None, got {self.shots}")
        if self.noise is not None:
            # Validates the preset name up front rather than at run time.
            noise_spec(self.noise)
            if self.backend != "aer":
                raise ValueError(
                    f"noise={self.noise!r} requires backend='aer'; the statevector "
                    "simulator has no noise support. Silently ignoring the noise "
                    "model would report ideal results under a noisy label."
                )

    @property
    def precision(self) -> float:
        """Estimator precision implied by ``shots`` (``0.0`` means exact)."""
        if self.shots is None:
            return 0.0
        return 1.0 / (self.shots**0.5)

    def noise_model(self):
        """Aer ``NoiseModel`` for this adapter, or ``None`` when running ideally."""
        if self.noise is None:
            return None
        require_qiskit("Building a noise model")
        return build_noise_model(self.noise)

    def prepare(self, circuit):
        """Transpile a circuit into a basis this backend can actually run.

        Needed for two distinct reasons, both of which fail late and confusingly
        if skipped:

        * **Noise attaches to gate names.** Any gate outside the noise model's
          basis is applied perfectly, so the run succeeds and quietly
          under-reports the noise.
        * **Aer cannot execute every Qiskit gate.** The XY mixer's
          ``XXPlusYYGate`` raises ``AerError: unknown instruction: xx_plus_yy``
          rather than being decomposed automatically.

        A no-op on the statevector backend, which handles arbitrary gates and has
        no noise to attach.
        """
        if self.backend != "aer":
            return circuit
        require_qiskit("Transpiling for the Aer backend")
        from qiskit import transpile

        basis = noise_basis_gates(self.noise) if self.noise else AER_BASIS_GATES
        return transpile(
            circuit,
            basis_gates=basis,
            optimization_level=1,
            seed_transpiler=self.seed,
        )

    def pass_manager(self):
        """Transpiler pass manager matching :meth:`prepare`, or ``None`` if ideal.

        Needed by library code that builds and runs its own circuits -- the
        quantum-kernel fidelity, for instance -- where there is no single circuit
        to hand to :meth:`prepare`.
        """
        if self.noise is None:
            return None
        require_qiskit("Building a pass manager")
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        return generate_preset_pass_manager(
            optimization_level=1,
            basis_gates=noise_basis_gates(self.noise),
            seed_transpiler=self.seed,
        )

    def estimator(self):
        """Return a V2 estimator primitive for expectation values."""
        require_qiskit("Building an estimator")
        if self.backend == "aer":
            from qiskit_aer.primitives import EstimatorV2

            # Aer's EstimatorV2 seeds only via run_options["seed_simulator"].
            # Passing "seed" (or backend_options["seed_simulator"]) is silently
            # accepted and ignored, leaving results non-reproducible.
            options: dict = {
                "default_precision": self.precision,
                "run_options": {"seed_simulator": self.seed},
            }
            noise = self.noise_model()
            if noise is not None:
                options["backend_options"] = {"noise_model": noise}
            return EstimatorV2(options=options)

        from qiskit.primitives import StatevectorEstimator

        return StatevectorEstimator(default_precision=self.precision, seed=self.seed)

    def sampler(self, shots: int | None = None):
        """Return a V2 sampler primitive for bitstring sampling.

        Sampling is inherently shot-based, so unlike :meth:`estimator` this falls
        back to ``default_shots`` when the adapter was built for exact expectation
        values.
        """
        require_qiskit("Building a sampler")
        default_shots = shots or self.shots or 1024
        if self.backend == "aer":
            from qiskit_aer.primitives import SamplerV2

            noise = self.noise_model()
            options = {"backend_options": {"noise_model": noise}} if noise is not None else None
            return SamplerV2(default_shots=default_shots, seed=self.seed, options=options)

        from qiskit.primitives import StatevectorSampler

        return StatevectorSampler(default_shots=default_shots, seed=self.seed)

    def describe(self) -> dict:
        """Summarise the configured backend for benchmark rows and demo output."""
        return {
            "backend": self.backend,
            "primitives": "qiskit_v2",
            "shots": self.shots,
            "estimator_precision": self.precision,
            "exact_expectation_values": self.shots is None,
            "seed": self.seed,
            "noise": noise_spec(self.noise).describe() if self.noise else None,
            "ideal_device": self.noise is None,
            "qiskit_installed": qiskit_available(),
            "versions": qiskit_versions(),
        }
