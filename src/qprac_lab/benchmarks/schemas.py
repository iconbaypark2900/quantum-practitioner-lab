from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BenchmarkResult:
    algorithm: str
    use_case: str
    algorithm_type: str
    backend: str
    runtime_seconds: float
    metrics: dict[str, Any]
    payload: dict[str, Any]

    def to_dict(self):
        return asdict(self)
