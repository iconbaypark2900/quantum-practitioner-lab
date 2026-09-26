"""Performance utilities for quantum-practitioner-lab.

This module provides:
- Adapter-level caching (memoize quantum results)
- Batch processing for multiple circuit evaluations
- Timeout enforcement for all adapters
- Budget-aware early termination
- Performance metrics tracking (time per call, memory usage)
"""
from __future__ import annotations

import functools
import hashlib
import resource
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# Performance metrics                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class CallMetrics:
    """Metrics for a single adapter call."""
    service: str = ""
    call_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    memory_mb: float = 0.0
    budget_tier: int = 0
    budget_spent: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class PerformanceReport:
    """Aggregated performance report for a run."""
    total_calls: int = 0
    total_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    total_memory_mb: float = 0.0
    calls_by_service: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    calls_by_tier: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def avg_duration_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_duration_ms / self.total_calls

    @property
    def avg_memory_mb(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_memory_mb / self.total_calls

    def add_call(self, metrics: CallMetrics):
        self.total_calls += 1
        self.total_duration_ms += metrics.duration_ms
        self.total_memory_mb += metrics.memory_mb
        self.calls_by_service[metrics.service] += 1
        self.calls_by_tier[metrics.budget_tier] += 1

        if metrics.duration_ms > self.max_duration_ms:
            self.max_duration_ms = metrics.duration_ms

        if not metrics.success:
            self.errors.append({
                "service": metrics.service,
                "error": metrics.error,
                "duration_ms": metrics.duration_ms,
            })

    def summary(self) -> dict[str, Any]:
        avg_duration = self.total_duration_ms / self.total_calls if self.total_calls > 0 else 0
        avg_memory = self.total_memory_mb / self.total_calls if self.total_calls > 0 else 0
        return {
            "total_calls": self.total_calls,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "avg_duration_ms": round(avg_duration, 2),
            "max_duration_ms": round(self.max_duration_ms, 2),
            "total_memory_mb": round(self.total_memory_mb, 2),
            "avg_memory_mb": round(avg_memory, 2),
            "calls_by_service": dict(self.calls_by_service),
            "calls_by_tier": dict(self.calls_by_tier),
            "errors": self.errors,
        }


# --------------------------------------------------------------------------- #
# Global performance tracker                                                  #
# --------------------------------------------------------------------------- #
_global_tracker: PerformanceReport | None = None


def get_performance_tracker() -> PerformanceReport:
    """Return the global PerformanceReport, creating one if needed."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceReport()
    return _global_tracker


def reset_performance_tracker():
    """Reset the global tracker (useful for tests)."""
    global _global_tracker
    _global_tracker = None


# --------------------------------------------------------------------------- #
# Adapter cache                                                               #
# --------------------------------------------------------------------------- #
class AdapterCache:
    """LRU cache keyed on a hashable representation of inputs.

    Parameters
    ----------
    max_size:
        Maximum number of entries. Default 128.
    """

    def __init__(self, max_size: int = 128) -> None:
        self._max_size = max_size
        self._store: dict[str, Any] = {}
        self._order: list[str] = []
        self._hits = 0
        self._misses = 0

    def _key(self, *args: Any, **kwargs: Any) -> str:
        raw = f"{args}|{sorted(kwargs.items())}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, *args: Any, **kwargs: Any) -> Any | None:
        key = self._key(*args, **kwargs)
        if key not in self._store:
            self._misses += 1
            return None
        self._hits += 1
        self._order.remove(key)
        self._order.append(key)
        return self._store[key]

    def put(self, value: Any, *args: Any, **kwargs: Any) -> None:
        key = self._key(*args, **kwargs)
        if key in self._store:
            self._order.remove(key)
        self._store[key] = value
        self._order.append(key)
        if len(self._store) > self._max_size:
            evict = self._order.pop(0)
            del self._store[evict]

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> dict[str, int]:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }


# Global cache registry keyed by adapter name
_adapter_caches: dict[str, AdapterCache] = {}


def get_adapter_cache(name: str, max_size: int = 128) -> AdapterCache:
    if name not in _adapter_caches:
        _adapter_caches[name] = AdapterCache(max_size)
    return _adapter_caches[name]


def clear_adapter_caches():
    """Clear all adapter caches."""
    global _adapter_caches
    _adapter_caches.clear()


# --------------------------------------------------------------------------- #
# Timeout enforcement                                                         #
# --------------------------------------------------------------------------- #
def enforce_timeout(timeout_seconds: float):
    """Decorator that enforces a wall-clock timeout on a function.

    Raises ``TimeoutError`` if the function exceeds ``timeout_seconds``.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"{fn.__name__} exceeded {timeout_seconds}s "
                    f"(took {elapsed:.2f}s)"
                )
            return result
        return wrapper
    return decorator


# --------------------------------------------------------------------------- #
# Budget-aware termination                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class BudgetState:
    """Track budget spending for early termination."""
    tier: int = 0
    spent: float = 0.0
    limit: float = float("inf")

    def charge(self, amount: float) -> bool:
        """Charge the budget. Returns False if over limit."""
        self.spent += amount
        return self.spent <= self.limit

    def reset(self) -> None:
        self.spent = 0.0

    def should_terminate(self) -> bool:
        return self.spent > self.limit


# --------------------------------------------------------------------------- #
# Performance tracking context manager                                        #
# --------------------------------------------------------------------------- #
@dataclass
class _TrackingContext:
    tracker: PerformanceReport
    service: str
    budget: BudgetState
    call_id: str
    start_time: float

    def record_success(self, duration_ms: float, memory_mb: float):
        metrics = CallMetrics(
            service=self.service,
            call_id=self.call_id,
            start_time=self.start_time,
            end_time=time.perf_counter(),
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            budget_tier=self.budget.tier,
            budget_spent=self.budget.spent,
            success=True,
        )
        self.tracker.add_call(metrics)

    def record_error(self, duration_ms: float, error: str):
        metrics = CallMetrics(
            service=self.service,
            call_id=self.call_id,
            start_time=self.start_time,
            end_time=time.perf_counter(),
            duration_ms=duration_ms,
            memory_mb=0.0,
            budget_tier=self.budget.tier,
            budget_spent=self.budget.spent,
            success=False,
            error=error,
        )
        self.tracker.add_call(metrics)


def track_performance(
    service: str,
    budget: BudgetState | None = None,
) -> _TrackingContext:
    """Start tracking performance for an adapter call.

    Returns a context handle; call ``record_success`` or ``record_error``
    when the call completes.
    """
    tracker = get_performance_tracker()
    budget = budget or BudgetState()
    call_id = hashlib.sha256(f"{service}:{time.perf_counter()}".encode()).hexdigest()[:12]
    return _TrackingContext(
        tracker=tracker,
        service=service,
        budget=budget,
        call_id=call_id,
        start_time=time.perf_counter(),
    )


# --------------------------------------------------------------------------- #
# Batch processing                                                            #
# --------------------------------------------------------------------------- #
def batch_candidates(items: list[Any], batch_size: int = 10) -> list[list[Any]]:
    """Split items into batches of ``batch_size``."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def process_batch(
    items: list[Any],
    processor: Callable[[list[Any]], list[Any]],
    batch_size: int = 10,
) -> list[Any]:
    """Process items in batches, returning all results concatenated."""
    results: list[Any] = []
    for batch in batch_candidates(items, batch_size):
        results.extend(processor(batch))
    return results
