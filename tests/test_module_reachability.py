"""Assert that every module under ``src/`` is reachable from an entry point.

``DECISIONS.md`` #9 sets the standard for the YAML configs: they are *loaded and
validated, or deleted*, because a config nobody reads is a claim with no one to
contradict it. The same failure mode reaches code. ``circuits/ansatz.py`` and
``circuits/feature_maps.py`` survived the entire scaffold phase in a directory the
README advertises, still carrying docstrings that read "Replace with Qiskit
RealAmplitudes or EfficientSU2 in implementation phase" -- long after the real
implementations had shipped elsewhere.

Deleting them once only resets that clock, which is the same argument
``tests/test_configs.py`` makes about the configs. This test is the fix: it walks
the actual import graph from the real entry points and fails when a module is
reachable from nothing.

``KNOWN_DEAD`` is a **ratchet, not an exemption**. It records the orphans that
existed when this test was written, and the test asserts the set can only shrink:
new rot fails immediately, and each entry is deleted as the module is. The suite
stays honest about the debt instead of hiding it.

Needs no quantum stack.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from qprac_lab.config import PROJECT_ROOT

PACKAGE_ROOT = PROJECT_ROOT / "src" / "qprac_lab"

#: Where execution actually begins. ``cli`` is the ``qprac-lab`` console script
#: declared in ``pyproject.toml``; ``demo_registry`` is what the CLI and
#: ``scripts/run_demo.py`` dispatch through. The other six runner scripts import
#: their algorithm, backend and data modules directly, which is why
#: ``_reachable_modules`` also seeds its roots from everything ``scripts/`` imports.
ENTRY_POINTS = frozenset({"qprac_lab.cli", "qprac_lab.demo_registry"})

#: Deliberately unreferenced, and documented as such in ``DECISIONS.md`` #7.
#: IBM Runtime needs credentials CI cannot hold and drives the same Qiskit stack;
#: CUDA-Q needs a GPU the target machine does not have. Both were replaced by the
#: PennyLane cross-check. Each survives as an inert descriptor whose ``describe()``
#: reports ``status: placeholder`` -- note it does *not* raise, so nothing stops a
#: caller using one by mistake. That is a recorded decision rather than rot, which is
#: what this exemption is for.
DOCUMENTED_PLACEHOLDERS = frozenset(
    {
        "qprac_lab.backends.cudaq_adapter",
        "qprac_lab.backends.ibm_runtime_adapter",
    }
)

#: Empty, and meant to stay that way. Shrink this set by deleting the module or
#: wiring it in, never by adding to it. See the ``/prune-dead-modules`` command.
#:
#: It held five entries when this test was written. Four were deleted: two scaffold
#: descriptors superseded by the real Qiskit builders, plus a PDE baseline and a PDE
#: metrics module whose functions did not match what the tutorials actually do --
#: an *explicit* finite-difference step against the implicit Euler the heat equation
#: really takes, and an L2/residual pair against the VQLS cost it really minimises.
#: Wiring those in would have changed the numerical method to suit the helper.
#:
#: The fifth, ``baselines.exact_diagonalization``, was wired in instead, because it
#: genuinely was the baseline ``hamiltonian_utils`` had reimplemented inline. Doing
#: so surfaced a bug that had sat there unrun: a ``dtype=float`` cast that discarded
#: the imaginary part of a Hermitian matrix and returned 0.0 for Pauli-Y instead of
#: -1.0 -- a plausible number rather than an error, which is the failure mode
#: DECISIONS.md #8 is about.
KNOWN_DEAD: frozenset[str] = frozenset()

SEARCH_ROOTS = ("src", "tests", "scripts", "notebooks")


def _module_name(path: Path) -> str:
    """Dotted module name for a file under ``src/``."""
    parts = path.relative_to(PROJECT_ROOT / "src").with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _all_modules() -> set[str]:
    """Every importable module in the package, excluding empty ``__init__`` files."""
    modules = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        modules.add(_module_name(path))
    return modules


def _source_files() -> list[tuple[Path, str]]:
    """Every Python source in the repo, with notebook code cells flattened."""
    collected: list[tuple[Path, str]] = []
    for root in SEARCH_ROOTS:
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            collected.append((path, path.read_text(encoding="utf-8", errors="ignore")))
        for path in base.rglob("*.ipynb"):
            if ".ipynb_checkpoints" in path.parts:
                continue
            try:
                cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
            except (json.JSONDecodeError, KeyError):
                continue
            code = "\n".join(
                "".join(cell.get("source", ""))
                for cell in cells
                if cell.get("cell_type") == "code"
            )
            # Notebook cells may carry IPython magics that are not valid Python;
            # drop those lines rather than losing the whole notebook's imports.
            code = "\n".join(
                line for line in code.splitlines() if not line.lstrip().startswith(("%", "!"))
            )
            collected.append((path, code))
    return collected


def _imports_from(source: str, known: set[str]) -> set[str]:
    """Package modules imported by one source string."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in known:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if not node.module or not node.module.startswith("qprac_lab"):
                continue
            if node.module in known:
                found.add(node.module)
            # ``from qprac_lab.pkg import mod`` imports a submodule, not a symbol.
            for alias in node.names:
                candidate = f"{node.module}.{alias.name}"
                if candidate in known:
                    found.add(candidate)
    return found


def _import_graph(known: set[str]) -> dict[str, set[str]]:
    """Map each package module to the package modules it imports."""
    graph: dict[str, set[str]] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        graph[_module_name(path)] = _imports_from(source, known)
    return graph


def _reachable_modules() -> set[str]:
    """Everything reachable from the console script, the registry, tests, or scripts."""
    known = _all_modules()
    graph = _import_graph(known)

    roots = set(ENTRY_POINTS)
    for path, source in _source_files():
        if PACKAGE_ROOT in path.parents:
            continue  # package-internal edges come from the graph, not the roots
        roots |= _imports_from(source, known)

    seen: set[str] = set()
    queue = [module for module in roots if module in known]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        queue.extend(graph.get(module, set()) - seen)
    return seen


def test_no_module_is_orphaned():
    """Fail when a module is imported by nothing -- the ansatz.py failure mode."""
    orphaned = _all_modules() - _reachable_modules() - DOCUMENTED_PLACEHOLDERS
    unexpected = orphaned - KNOWN_DEAD
    assert not unexpected, (
        "these modules are reachable from no entry point and are not recorded as "
        f"known-dead: {sorted(unexpected)}. Wire them in, or delete them -- a module "
        "nobody imports is a claim with no one to contradict it (DECISIONS.md #9)."
    )


def test_known_dead_list_only_shrinks():
    """The ratchet: an entry stays only while the module it names still exists."""
    stale = {module for module in KNOWN_DEAD if module not in _all_modules()}
    assert not stale, (
        f"{sorted(stale)} no longer exist -- delete them from KNOWN_DEAD. The set "
        "records outstanding debt, so a stale entry silently widens the exemption."
    )


def test_known_dead_modules_are_genuinely_unreachable():
    """Guard the other direction: a module that got wired up must leave the list."""
    revived = KNOWN_DEAD & _reachable_modules()
    assert not revived, (
        f"{sorted(revived)} are now imported -- remove them from KNOWN_DEAD so the "
        "ratchet tightens."
    )


@pytest.mark.parametrize("module", sorted(DOCUMENTED_PLACEHOLDERS))
def test_documented_placeholders_still_declare_themselves(module):
    """A dropped backend must say so in its own source, not only in DECISIONS.md."""
    path = PROJECT_ROOT / "src" / Path(*module.split(".")).with_suffix(".py")
    assert path.exists(), f"{module} is exempted but missing"
    assert "placeholder" in path.read_text(encoding="utf-8").lower(), (
        f"{module} is exempted from the reachability check as a documented "
        "placeholder, so its source must say that plainly."
    )
