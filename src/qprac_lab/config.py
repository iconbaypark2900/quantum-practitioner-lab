"""Loading and resolving the YAML files under ``configs/``.

These files spent the whole scaffold phase as decoration: nothing imported them,
so nothing noticed when they drifted. By the time anyone looked, ``backends.yaml``
still advertised two backends that had been dropped, ``noise_model.yaml`` said
noise was disabled after it was implemented, and ``tutorials.yaml`` pointed at a
notebook that had been renumbered out of existence.

Editing them once only resets that clock. The fix is that they are now *loaded*
and *cross-checked against the code* by ``tests/test_configs.py`` -- every
tutorial path must resolve, every algorithm named must be registered, every noise
preset must match the implementation. A config file nobody reads is not
documentation, it is a claim with no one to contradict it.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

#: Repository root, resolved from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

CONFIG_NAMES = (
    "algorithms",
    "backends",
    "benchmarks",
    "experiment",
    "noise_model",
    "papers",
    "project",
    "qiskit",
    "tutorials",
)


def config_path(name: str) -> Path:
    """Path to a named config, without loading it."""
    if name not in CONFIG_NAMES:
        raise ValueError(f"Unknown config {name!r}; expected one of {CONFIG_NAMES}")
    return CONFIG_DIR / f"{name}.yaml"


@functools.lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load and cache one config file."""
    path = config_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Config {name!r} not found at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_all() -> dict[str, dict[str, Any]]:
    """Every config, keyed by name."""
    return {name: load_config(name) for name in CONFIG_NAMES}


def resolve(relative: str) -> Path:
    """Resolve a repository-relative path from a config against the project root."""
    return PROJECT_ROOT / relative


def tutorial_entries() -> list[dict[str, Any]]:
    """Tutorial records from ``tutorials.yaml``."""
    return list(load_config("tutorials")["tutorials"])


def benchmark_entries() -> list[dict[str, Any]]:
    return list(load_config("tutorials")["benchmarks"])


def concept_entries() -> list[dict[str, Any]]:
    """Short concept notes that point at the tutorials carrying the material."""
    return list(load_config("tutorials").get("concepts", []))


def algorithm_entries() -> dict[str, dict[str, Any]]:
    """Flatten ``algorithms.yaml`` into ``{algorithm_id: record}``.

    The file is grouped by module for readability; callers want it flat, and the
    module is carried through so nothing is lost.
    """
    flattened: dict[str, dict[str, Any]] = {}
    for module, algorithms in load_config("algorithms").items():
        for name, record in algorithms.items():
            flattened[name] = {"module": module, **record}
    return flattened


def implemented_algorithms() -> set[str]:
    """Algorithms marked as having a real quantum implementation."""
    return {
        name
        for name, record in algorithm_entries().items()
        if record.get("status") == "quantum"
    }


def dropped_backends() -> dict[str, str]:
    """Backends deliberately not implemented, with the reason recorded."""
    return {
        name: record["reason"]
        for name, record in load_config("backends").get("dropped", {}).items()
    }
