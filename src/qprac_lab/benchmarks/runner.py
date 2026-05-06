from __future__ import annotations

import time
import json
from pathlib import Path
import pandas as pd

from qprac_lab.benchmarks.schemas import BenchmarkResult


def run_and_time(name: str, fn, backend: str = "local_scaffold") -> BenchmarkResult:
    start = time.perf_counter()
    result = fn()
    runtime = time.perf_counter() - start

    payload = result.__dict__ if hasattr(result, "__dict__") else result
    return BenchmarkResult(
        algorithm=payload.get("algorithm", name),
        use_case=payload.get("use_case", "unknown"),
        algorithm_type=payload.get("algorithm_type", "unknown"),
        backend=backend,
        runtime_seconds=float(runtime),
        metrics=_extract_metrics(payload),
        payload=payload,
    )


def _extract_metrics(payload: dict):
    metrics = {}
    for key in [
        "absolute_error",
        "objective_value",
        "constraint_report",
        "rbf_svm_metrics",
        "random_forest_metrics",
    ]:
        if key in payload:
            metrics[key] = payload[key]
    return metrics


def save_benchmark_results(results: list[BenchmarkResult], output_csv: str, output_json: str):
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in results]
    Path(output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df
