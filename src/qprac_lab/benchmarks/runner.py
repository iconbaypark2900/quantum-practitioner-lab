from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from qprac_lab.benchmarks.schemas import BenchmarkResult

#: Metric fields worth lifting out of a result payload into the benchmark table.
#: Anything absent from a given algorithm is simply skipped.
METRIC_FIELDS = (
    # VQE
    "vqe_energy",
    "exact_baseline_energy",
    "hartree_fock_baseline_energy",
    "absolute_error",
    "chemical_accuracy_reached",
    "correlation_recovery_fraction",
    # QAOA
    "objective_value",
    "matches_brute_force",
    "feasible_probability",
    "optimal_probability",
    "optimal_probability_lift",
    "normalized_approximation_ratio",
    "constraint_report",
    # Quantum kernel
    "best_model",
    "quantum_beats_all_classical",
    "quantum_kernel_metrics",
    "rbf_svm_metrics",
    "random_forest_metrics",
)


def run_and_time(name: str, fn, backend: str = "statevector") -> BenchmarkResult:
    """Run one algorithm, time it, and normalise the result into a benchmark row."""
    start = time.perf_counter()
    result = fn()
    runtime = time.perf_counter() - start

    payload = result.__dict__ if hasattr(result, "__dict__") else result
    resolved_backend = payload.get("backend", backend)
    if isinstance(resolved_backend, dict):
        resolved_backend = resolved_backend.get("backend", backend)

    return BenchmarkResult(
        algorithm=payload.get("algorithm", name),
        use_case=payload.get("use_case", "unknown"),
        algorithm_type=payload.get("algorithm_type", "unknown"),
        backend=str(resolved_backend),
        runtime_seconds=float(runtime),
        metrics=_extract_metrics(payload),
        payload=payload,
    )


def _extract_metrics(payload: dict) -> dict:
    return {key: payload[key] for key in METRIC_FIELDS if key in payload}


def save_benchmark_results(results: list[BenchmarkResult], output_csv: str, output_json: str):
    """Persist benchmark rows as full JSON plus a flat, readable CSV.

    The JSON keeps the complete payload for later analysis; the CSV drops it and
    flattens the metrics into columns, since a nested dict rendered into a CSV
    cell is unreadable in every tool that opens one.
    """
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in results]
    Path(output_json).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    flat_rows = []
    for row in rows:
        flat = {
            "algorithm": row["algorithm"],
            "use_case": row["use_case"],
            "algorithm_type": row["algorithm_type"],
            "backend": row["backend"],
            "runtime_seconds": round(row["runtime_seconds"], 4),
        }
        for key, value in row["metrics"].items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, (dict, list)):
                        flat[f"{key}.{sub_key}"] = sub_value
            elif not isinstance(value, list):
                flat[key] = value
        flat_rows.append(flat)

    frame = pd.DataFrame(flat_rows)
    frame.to_csv(output_csv, index=False)
    return frame
