"""Tests for the managed agents adapter layer.

Verifies:
- Top-N gate enforcement
- LRU caching behavior
- Budget-aware termination
- Performance tracking
- Adapter creation and describe()
"""
import pytest

from qprac_lab.adapters import (
    ManagedQiskitAdapter,
    ManagedPennyLaneAdapter,
    ManagedBenchmarkRunner,
    ManagedNoiseModel,
    TopNGateExhaustedError,
    create_managed_qiskit,
    create_managed_pennylane,
    create_managed_benchmark_runner,
    create_managed_noise_model,
)
from qprac_lab.perf import (
    AdapterCache,
    BudgetState,
    PerformanceReport,
    clear_adapter_caches,
    get_adapter_cache,
    get_performance_tracker,
    reset_performance_tracker,
)
from qprac_lab.backends.qiskit_adapter import qiskit_available as _qiskit_available

QISKIT_INSTALLED = _qiskit_available()


class TestAdapterCache:
    def test_put_get(self):
        cache = AdapterCache(max_size=10)
        cache.put("value", key="test")
        assert cache.get(key="test") == "value"

    def test_miss(self):
        cache = AdapterCache()
        assert cache.get(key="nonexistent") is None

    def test_eviction(self):
        cache = AdapterCache(max_size=2)
        cache.put("a", key="a")
        cache.put("b", key="b")
        cache.put("c", key="c")  # evicts "a"
        assert cache.get(key="a") is None
        assert cache.get(key="b") == "b"
        assert cache.get(key="c") == "c"

    def test_stats(self):
        cache = AdapterCache()
        cache.put("v", key="k1")
        cache.get(key="k1")
        cache.get(key="k2")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_clear(self):
        cache = AdapterCache()
        cache.put("v", key="k")
        cache.clear()
        assert cache.get(key="k") is None
        assert cache.stats()["hits"] == 0


class TestBudgetState:
    def test_charge_under_limit(self):
        budget = BudgetState(limit=10.0)
        assert budget.charge(5.0) is True
        assert budget.spent == 5.0

    def test_charge_over_limit(self):
        budget = BudgetState(limit=10.0)
        assert budget.charge(11.0) is False
        assert budget.spent == 11.0

    def test_reset(self):
        budget = BudgetState(limit=10.0)
        budget.charge(5.0)
        budget.reset()
        assert budget.spent == 0.0

    def test_should_terminate(self):
        budget = BudgetState(limit=10.0)
        assert budget.should_terminate() is False
        budget.charge(11.0)
        assert budget.should_terminate() is True


class TestPerformanceTracker:
    def setup_method(self):
        reset_performance_tracker()

    def test_get_tracker(self):
        tracker = get_performance_tracker()
        assert isinstance(tracker, PerformanceReport)

    def test_add_call(self):
        tracker = get_performance_tracker()
        from qprac_lab.perf import CallMetrics
        metrics = CallMetrics(
            service="test",
            call_id="abc",
            start_time=0.0,
            end_time=1.0,
            duration_ms=1000.0,
            memory_mb=100.0,
            success=True,
        )
        tracker.add_call(metrics)
        assert tracker.total_calls == 1
        assert tracker.total_duration_ms == 1000.0
        assert tracker.avg_duration_ms == 1000.0

    def test_summary(self):
        tracker = get_performance_tracker()
        from qprac_lab.perf import CallMetrics
        metrics = CallMetrics(
            service="test",
            call_id="abc",
            start_time=0.0,
            end_time=1.0,
            duration_ms=500.0,
            memory_mb=50.0,
            success=True,
        )
        tracker.add_call(metrics)
        summary = tracker.summary()
        assert summary["total_calls"] == 1
        assert summary["total_duration_ms"] == 500.0


class TestTopNGate:
    def test_allows_under_limit(self):
        from qprac_lab.adapters import _TopNGate
        gate = _TopNGate(limit=3)
        for _ in range(3):
            gate.check()  # should not raise

    def test_raises_over_limit(self):
        from qprac_lab.adapters import _TopNGate
        gate = _TopNGate(limit=2)
        gate.check()
        gate.check()
        with pytest.raises(TopNGateExhaustedError):
            gate.check()

    def test_reset(self):
        from qprac_lab.adapters import _TopNGate
        gate = _TopNGate(limit=1)
        gate.check()
        gate.reset()
        gate.check()  # should not raise after reset


class TestManagedQiskitAdapter:
    def setup_method(self):
        clear_adapter_caches()

    def test_create(self):
        adapter = create_managed_qiskit(backend="statevector")
        assert adapter.name == "managed_qiskit_adapter"
        desc = adapter.describe()
        assert desc["managed"] is True
        assert desc["top_n_limit"] == 5

    @pytest.mark.skipif(not QISKIT_INSTALLED, reason="Qiskit not installed")
    def test_top_n_gate(self):
        adapter = create_managed_qiskit(backend="statevector", top_n=2)
        # First two calls should work (estimator is cached after first call)
        adapter.estimator()
        adapter.estimator()
        with pytest.raises(TopNGateExhaustedError):
            adapter.estimator()

    def test_budget_charge(self):
        adapter = create_managed_qiskit()
        assert adapter.charge(5.0) is True
        assert adapter._budget.spent == 5.0

    def test_reset_budget(self):
        adapter = create_managed_qiskit()
        adapter.charge(5.0)
        adapter.reset_budget()
        assert adapter._budget.spent == 0.0


class TestManagedPennyLaneAdapter:
    def setup_method(self):
        clear_adapter_caches()

    def test_create(self):
        adapter = create_managed_pennylane()
        assert adapter.name == "managed_pennylane_adapter"
        desc = adapter.describe()
        assert desc["managed"] is True


class TestManagedBenchmarkRunner:
    def setup_method(self):
        clear_adapter_caches()

    def test_create(self):
        runner = create_managed_benchmark_runner()
        desc = runner.describe()
        assert desc["managed"] is True
        assert desc["top_n_limit"] == 10
        assert desc["timeout_s"] == 300.0

    def test_top_n_gate(self):
        runner = create_managed_benchmark_runner(top_n=2)
        # Mock fn to avoid actual quantum execution
        def fake_fn():
            return type("Result", (), {"__dict__": {"algorithm": "test"}})()
        
        runner.run_and_time("test", fake_fn)
        runner.run_and_time("test", fake_fn)
        with pytest.raises(TopNGateExhaustedError):
            runner.run_and_time("test", fake_fn)


class TestManagedNoiseModel:
    def setup_method(self):
        clear_adapter_caches()

    def test_create(self):
        model = create_managed_noise_model()
        desc = model.describe()
        assert desc["managed"] is True
        assert "available_presets" in desc
        assert "light" in desc["available_presets"]
        assert "moderate" in desc["available_presets"]
        assert "heavy" in desc["available_presets"]

    @pytest.mark.skipif(not QISKIT_INSTALLED, reason="Qiskit not installed")
    def test_build(self):
        model = create_managed_noise_model()
        # First build should cache
        noise1 = model.build("light")
        # Second build should return cached
        noise2 = model.build("light")
        assert noise1 is noise2  # same object from cache


class TestClearAdapterCaches:
    def test_clear(self):
        cache1 = get_adapter_cache("test1")
        cache2 = get_adapter_cache("test2")
        cache1.put("v", key="k")
        clear_adapter_caches()
        cache3 = get_adapter_cache("test1")
        assert cache3.get(key="k") is None
