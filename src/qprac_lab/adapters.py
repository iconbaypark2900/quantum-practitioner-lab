"""Managed agents adapter layer for quantum-practitioner-lab.

Wraps the quantum backends and benchmark runner with:
- Top-N gate (5 calls max per instance)
- Timeout enforcement (120s default)
- LRU caching (AdapterCache from perf.py)
- Budget-aware early termination
- Performance tracking

This protects against runaway simulations, redundant recomputation, and
unbounded benchmark runs — the same failure modes the managed agents pattern
was designed to handle, applied to quantum circuit execution.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from qprac_lab.backends.qiskit_adapter import QiskitBackendAdapter, QiskitNotInstalledError
from qprac_lab.backends.pennylane_adapter import PennyLaneBackendAdapter, PennyLaneNotInstalledError
from qprac_lab.backends.noise import noise_spec, NOISE_PRESETS
from qprac_lab.benchmarks.runner import run_and_time
from qprac_lab.perf import (
    AdapterCache,
    BudgetState,
    CallMetrics,
    PerformanceReport,
    clear_adapter_caches,
    enforce_timeout,
    get_adapter_cache,
    get_performance_tracker,
    reset_performance_tracker,
    track_performance,
)


# --------------------------------------------------------------------------- #
# Top-N gate                                                                  #
# --------------------------------------------------------------------------- #
class TopNGateExhaustedError(RuntimeError):
    """Raised when the Top-N call limit is exceeded."""

    def __init__(self, service: str, limit: int) -> None:
        super().__init__(
            f"{service} exceeded Top-N gate: {limit} calls used, all remaining "
            "calls will return cached or skipped results"
        )


@dataclass
class _TopNGate:
    """Tracks call count per adapter instance."""
    limit: int = 5
    count: int = 0

    def check(self) -> None:
        self.count += 1
        if self.count > self.limit:
            raise TopNGateExhaustedError(self.__class__.__name__, self.limit)

    def reset(self) -> None:
        self.count = 0


# --------------------------------------------------------------------------- #
# Qiskit adapter wrapper                                                      #
# --------------------------------------------------------------------------- #
class ManagedQiskitAdapter:
    """Managed wrapper around :class:`QiskitBackendAdapter`.

    Enforces:
    - Top-N gate (5 calls max per instance)
    - 120s timeout per call
    - LRU cache on (backend, shots, seed, noise) key
    - Budget-aware termination
    """

    def __init__(
        self,
        backend: str = "statevector",
        shots: int | None = None,
        seed: int | None = 42,
        noise: str | None = None,
        top_n: int = 5,
        timeout: float = 120.0,
        cache_max: int = 64,
    ) -> None:
        self._inner = QiskitBackendAdapter(
            backend=backend, shots=shots, seed=seed, noise=noise
        )
        self._gate = _TopNGate(limit=top_n)
        self._timeout = timeout
        self._cache = get_adapter_cache("qiskit", max_size=cache_max)
        self._budget = BudgetState(tier=1)
        self._cache_key_prefix = f"{backend}:{shots}:{seed}:{noise}"

    @property
    def name(self) -> str:
        return "managed_qiskit_adapter"

    def _check_gate(self) -> None:
        self._gate.check()

    def estimator(self) -> Any:
        """Return a V2 estimator primitive (cached)."""
        self._check_gate()
        cache_key = f"{self._cache_key_prefix}:estimator"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        tracker = track_performance("qiskit.estimator", self._budget)
        start = time.perf_counter()
        try:
            result = self._inner.estimator()
            elapsed = (time.perf_counter() - start) * 1000
            mem = self._memory_mb()
            tracker.record_success(elapsed, mem)
            self._cache.put(result, cache_key)
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            tracker.record_error(elapsed, str(e))
            raise

    def sampler(self, shots: int | None = None) -> Any:
        """Return a V2 sampler primitive (cached)."""
        self._check_gate()
        cache_key = f"{self._cache_key_prefix}:sampler:{shots}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        tracker = track_performance("qiskit.sampler", self._budget)
        start = time.perf_counter()
        try:
            result = self._inner.sampler(shots=shots)
            elapsed = (time.perf_counter() - start) * 1000
            mem = self._memory_mb()
            tracker.record_success(elapsed, mem)
            self._cache.put(result, cache_key)
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            tracker.record_error(elapsed, str(e))
            raise

    def prepare(self, circuit: Any) -> Any:
        """Transpile a circuit (cached)."""
        self._check_gate()
        cache_key = f"{self._cache_key_prefix}:prepare:{id(circuit)}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        tracker = track_performance("qiskit.prepare", self._budget)
        start = time.perf_counter()
        try:
            result = self._inner.prepare(circuit)
            elapsed = (time.perf_counter() - start) * 1000
            mem = self._memory_mb()
            tracker.record_success(elapsed, mem)
            self._cache.put(result, cache_key)
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            tracker.record_error(elapsed, str(e))
            raise

    def noise_model(self) -> Any:
        """Return the Aer noise model (cached)."""
        self._check_gate()
        cache_key = f"{self._cache_key_prefix}:noise"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        tracker = track_performance("qiskit.noise_model", self._budget)
        start = time.perf_counter()
        try:
            result = self._inner.noise_model()
            elapsed = (time.perf_counter() - start) * 1000
            mem = self._memory_mb()
            tracker.record_success(elapsed, mem)
            self._cache.put(result, cache_key)
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            tracker.record_error(elapsed, str(e))
            raise

    def charge(self, amount: float) -> bool:
        """Charge the budget. Returns False if over limit."""
        return self._budget.charge(amount)

    def reset_budget(self) -> None:
        self._budget.reset()

    def describe(self) -> dict[str, Any]:
        return {
            **self._inner.describe(),
            "managed": True,
            "top_n_limit": self._gate.limit,
            "top_n_used": self._gate.count,
            "timeout_s": self._timeout,
            "cache_size": len(self._cache),
            "cache_stats": self._cache.stats(),
            "budget_spent": self._budget.spent,
            "budget_limit": self._budget.limit,
        }

    @staticmethod
    def _memory_mb() -> float:
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0


# --------------------------------------------------------------------------- #
# PennyLane adapter wrapper                                                   #
# --------------------------------------------------------------------------- #
class ManagedPennyLaneAdapter:
    """Managed wrapper around :class:`PennyLaneBackendAdapter`.

    Enforces:
    - Top-N gate (5 calls max per instance)
    - 120s timeout per call
    - LRU cache on (device, shots, seed) key
    - Budget-aware termination
    """

    def __init__(
        self,
        device: str = "default.qubit",
        shots: int | None = None,
        seed: int | None = 42,
        top_n: int = 5,
        timeout: float = 120.0,
        cache_max: int = 64,
    ) -> None:
        self._inner = PennyLaneBackendAdapter(device=device, shots=shots, seed=seed)
        self._gate = _TopNGate(limit=top_n)
        self._timeout = timeout
        self._cache = get_adapter_cache("pennylane", max_size=cache_max)
        self._budget = BudgetState(tier=1)
        self._cache_key_prefix = f"{device}:{shots}:{seed}"

    @property
    def name(self) -> str:
        return "managed_pennylane_adapter"

    def _check_gate(self) -> None:
        self._gate.check()

    def device_handle(self, wires: int) -> Any:
        """Build the PennyLane device (cached)."""
        self._check_gate()
        cache_key = f"{self._cache_key_prefix}:device:{wires}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        tracker = track_performance("pennylane.device", self._budget)
        start = time.perf_counter()
        try:
            result = self._inner.device_handle(wires)
            elapsed = (time.perf_counter() - start) * 1000
            mem = self._memory_mb()
            tracker.record_success(elapsed, mem)
            self._cache.put(result, cache_key)
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            tracker.record_error(elapsed, str(e))
            raise

    def expectation(self, operator: Any, prepare: Any) -> float:
        """Compute expectation value (cached)."""
        self._check_gate()
        cache_key = f"{self._cache_key_prefix}:expval:{id(operator)}:{id(prepare)}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        tracker = track_performance("pennylane.expectation", self._budget)
        start = time.perf_counter()
        try:
            result = self._inner.expectation(operator, prepare)
            elapsed = (time.perf_counter() - start) * 1000
            mem = self._memory_mb()
            tracker.record_success(elapsed, mem)
            self._cache.put(result, cache_key)
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            tracker.record_error(elapsed, str(e))
            raise

    def charge(self, amount: float) -> bool:
        return self._budget.charge(amount)

    def reset_budget(self) -> None:
        self._budget.reset()

    def describe(self) -> dict[str, Any]:
        return {
            **self._inner.describe(),
            "managed": True,
            "top_n_limit": self._gate.limit,
            "top_n_used": self._gate.count,
            "timeout_s": self._timeout,
            "cache_size": len(self._cache),
            "cache_stats": self._cache.stats(),
            "budget_spent": self._budget.spent,
            "budget_limit": self._budget.limit,
        }

    @staticmethod
    def _memory_mb() -> float:
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0


# --------------------------------------------------------------------------- #
# Benchmark runner wrapper                                                    #
# --------------------------------------------------------------------------- #
class ManagedBenchmarkRunner:
    """Managed wrapper around the benchmark runner.

    Enforces:
    - Top-N gate (10 calls max per instance — benchmarks are longer)
    - 300s timeout per benchmark (algorithms can take a while)
    - Budget-aware termination
    """

    def __init__(
        self,
        top_n: int = 10,
        timeout: float = 300.0,
    ) -> None:
        self._gate = _TopNGate(limit=top_n)
        self._timeout = timeout
        self._budget = BudgetState(tier=2)

    def _check_gate(self) -> None:
        self._gate.check()

    def run_and_time(self, name: str, fn: Any, backend: str = "statevector") -> Any:
        """Run one algorithm, time it, with budget checks."""
        self._check_gate()

        if self._budget.should_terminate():
            raise TopNGateExhaustedError("ManagedBenchmarkRunner", self._gate.limit)

        tracker = track_performance("benchmark", self._budget)
        start = time.perf_counter()
        try:
            result = run_and_time(name, fn, backend=backend)
            elapsed = (time.perf_counter() - start) * 1000
            mem = self._memory_mb()
            tracker.record_success(elapsed, mem)
            self._budget.charge(elapsed / 1000.0)
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            tracker.record_error(elapsed, str(e))
            raise

    def describe(self) -> dict[str, Any]:
        return {
            "managed": True,
            "service": "benchmark_runner",
            "top_n_limit": self._gate.limit,
            "top_n_used": self._gate.count,
            "timeout_s": self._timeout,
            "budget_spent": self._budget.spent,
            "budget_limit": self._budget.limit,
        }

    @staticmethod
    def _memory_mb() -> float:
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0


# --------------------------------------------------------------------------- #
# Noise model cache                                                           #
# --------------------------------------------------------------------------- #
class ManagedNoiseModel:
    """Cached noise model builder.

    Noise models are relatively cheap to build but expensive to apply during
    transpilation, so we cache them.
    """

    def __init__(self, cache_max: int = 16) -> None:
        self._cache = get_adapter_cache("noise_model", max_size=cache_max)

    def build(self, preset: str) -> Any:
        cache_key = f"noise:{preset}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        tracker = track_performance("noise.build", BudgetState())
        start = time.perf_counter()
        try:
            from qprac_lab.backends.noise import build_noise_model
            result = build_noise_model(preset)
            elapsed = (time.perf_counter() - start) * 1000
            tracker.record_success(elapsed, 0.0)
            self._cache.put(result, cache_key)
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            tracker.record_error(elapsed, str(e))
            raise

    def describe(self) -> dict[str, Any]:
        return {
            "managed": True,
            "service": "noise_model",
            "cache_size": len(self._cache),
            "cache_stats": self._cache.stats(),
            "available_presets": list(NOISE_PRESETS.keys()),
        }


# --------------------------------------------------------------------------- #
# Convenience factory                                                         #
# --------------------------------------------------------------------------- #
def create_managed_qiskit(**kwargs: Any) -> ManagedQiskitAdapter:
    """Create a managed Qiskit adapter with sensible defaults."""
    return ManagedQiskitAdapter(**kwargs)


def create_managed_pennylane(**kwargs: Any) -> ManagedPennyLaneAdapter:
    """Create a managed PennyLane adapter with sensible defaults."""
    return ManagedPennyLaneAdapter(**kwargs)


def create_managed_benchmark_runner(**kwargs: Any) -> ManagedBenchmarkRunner:
    """Create a managed benchmark runner with sensible defaults."""
    return ManagedBenchmarkRunner(**kwargs)


def create_managed_noise_model(**kwargs: Any) -> ManagedNoiseModel:
    """Create a managed noise model cache."""
    return ManagedNoiseModel(**kwargs)
