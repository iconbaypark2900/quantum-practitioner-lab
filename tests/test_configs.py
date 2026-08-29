"""Cross-check the YAML configs against the code and the filesystem.

These files drifted for the entire scaffold phase precisely because nothing read
them: a config nobody loads is a claim with no one to contradict it. By the time
anyone looked, ``backends.yaml`` advertised two dropped backends and
``tutorials.yaml`` pointed at a notebook that had been renumbered away.

Editing them once would only reset that clock. These tests are the actual fix.
They need no quantum stack.
"""


import pytest
import tomllib

from qprac_lab.config import (
    CONFIG_NAMES,
    PROJECT_ROOT,
    algorithm_entries,
    benchmark_entries,
    concept_entries,
    config_path,
    dropped_backends,
    implemented_algorithms,
    load_config,
    resolve,
    tutorial_entries,
)
from qprac_lab.demo_registry import DEMOS, QUANTUM_DEMOS


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_every_config_exists_and_parses(name):
    assert config_path(name).exists()
    assert isinstance(load_config(name), dict)


def test_unknown_config_is_rejected():
    with pytest.raises(ValueError):
        config_path("not_a_config")


def test_no_config_file_is_orphaned():
    """A new YAML must be registered in CONFIG_NAMES, or it drifts unwatched."""
    on_disk = {path.stem for path in (PROJECT_ROOT / "configs").glob("*.yaml")}
    assert on_disk == set(CONFIG_NAMES)


# ------------------------------------------------------------------ paths


def test_every_tutorial_path_resolves():
    """The check that would have caught the renumbered HHL notebook."""
    for entry in tutorial_entries():
        assert resolve(entry["path"]).exists(), f"{entry['id']}: missing {entry['path']}"
        if entry.get("notebook"):
            assert resolve(entry["notebook"]).exists(), (
                f"{entry['id']}: missing notebook {entry['notebook']}"
            )


def test_every_concept_note_resolves_and_points_somewhere():
    """A signpost that does not point anywhere is just a thinner duplicate."""
    for entry in concept_entries():
        path = resolve(entry["path"])
        assert path.exists(), f"{entry['id']}: missing {entry['path']}"
        text = path.read_text(encoding="utf-8")
        assert "Short note, not a standalone tutorial" in text, f"{entry['id']}: no signpost"
        assert "## Where this is used" in text, f"{entry['id']}: no onward links"


def test_every_benchmark_doc_resolves():
    for entry in benchmark_entries():
        assert resolve(entry["path"]).exists(), f"{entry['id']}: missing {entry['path']}"


def test_declared_paths_exist():
    for key, relative in load_config("project")["paths"].items():
        assert resolve(relative).exists(), f"project.paths.{key} -> missing {relative}"


def test_every_tutorial_markdown_is_registered():
    """A tutorial nobody lists is a tutorial nobody finds."""
    registered = {entry["path"] for entry in tutorial_entries()}
    registered |= {entry["path"] for entry in benchmark_entries()}
    registered |= {entry["path"] for entry in concept_entries()}
    on_disk = {
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "tutorials").rglob("*.md")
        if path.name not in {"README.md", "papers.md"} and "use_cases" not in path.parts
    }
    assert on_disk <= registered, f"unregistered tutorials: {sorted(on_disk - registered)}"


# ------------------------------------------------------- code cross-checks


def test_algorithms_config_matches_the_demo_registry():
    """`status: quantum` in YAML must mean `in QUANTUM_DEMOS` in code."""
    assert implemented_algorithms() == set(QUANTUM_DEMOS)


def test_every_configured_algorithm_is_runnable():
    for name in algorithm_entries():
        assert name in DEMOS, f"{name} is configured but not registered in DEMOS"


def test_every_registered_demo_is_configured():
    for name in DEMOS:
        assert name in algorithm_entries(), f"{name} is registered but not in algorithms.yaml"


def test_experiment_runs_are_the_implemented_algorithms():
    configured = {run["algorithm"] for run in load_config("experiment")["runs"]}
    assert configured == set(QUANTUM_DEMOS)


def test_scaffold_entries_declare_why():
    for name, record in algorithm_entries().items():
        if record.get("status") == "scaffold":
            assert record.get("note"), f"{name} is a scaffold with no explanation"


# --------------------------------------------------------------- backends


def test_backends_config_matches_the_adapter():
    from qprac_lab.backends.qiskit_adapter import SUPPORTED_BACKENDS

    configured = load_config("backends")["backends"]
    for backend in SUPPORTED_BACKENDS:
        assert backend in configured, f"{backend} is supported in code but not configured"
    assert configured["pennylane"]["enabled"] is True


def test_dropped_backends_record_a_reason():
    dropped = dropped_backends()
    assert set(dropped) == {"ibm_runtime", "cudaq"}
    for name, reason in dropped.items():
        assert len(reason) > 30, f"{name} was dropped without a real explanation"


def test_dropped_backends_are_not_also_offered():
    configured = load_config("backends")["backends"]
    assert not set(configured) & set(dropped_backends())


def test_noise_presets_match_the_implementation():
    """Mirrored values must not drift from the code they describe."""
    from qprac_lab.backends.noise import NOISE_PRESETS

    configured = load_config("noise_model")["presets"]
    assert set(configured) == set(NOISE_PRESETS)
    for name, spec in NOISE_PRESETS.items():
        assert configured[name]["single_qubit_error"] == pytest.approx(spec.single_qubit_error)
        assert configured[name]["two_qubit_error"] == pytest.approx(spec.two_qubit_error)
        assert configured[name]["readout_error"] == pytest.approx(spec.readout_error)


def test_noise_is_marked_enabled():
    """It was left as `enabled: false` for a whole release after being implemented."""
    assert load_config("noise_model")["noise_model"]["enabled"] is True


# ---------------------------------------------------------------- project


def test_project_version_matches_pyproject():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert load_config("project")["project"]["version"] == pyproject["project"]["version"]


def test_project_names_match_packaging():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = load_config("project")["project"]
    assert project["name"] == pyproject["project"]["name"]
    assert project["cli"] in pyproject["project"]["scripts"]


def test_declared_extras_exist_in_pyproject():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    available = set(pyproject["project"]["optional-dependencies"])
    for extra in load_config("project")["extras"]:
        assert extra in available, f"configs claims an extra {extra!r} that pyproject lacks"


def test_qiskit_config_reflects_the_v2_pin():
    qiskit_config = load_config("qiskit")["qiskit"]
    assert qiskit_config["primitives"] == "v2"
    assert qiskit_config["minimum_version"].startswith("2")
    assert set(load_config("qiskit")["gotchas"]) >= {"aer_seeding", "undecomposed_evolution"}


def test_paper_topics_cover_the_implemented_modules():
    papers = load_config("papers")["papers"]
    assert {"simulation", "optimization"} <= set(papers)
    for topic, entries in papers.items():
        for entry in entries:
            assert {"title", "authors", "year"} <= set(entry), f"{topic}: incomplete entry"


def test_naming_convention_doc_lists_every_demo():
    """It listed a deleted module and a deleted config for an entire release."""
    text = (PROJECT_ROOT / "docs" / "NAMING_CONVENTION.md").read_text(encoding="utf-8")
    for name in DEMOS:
        assert name in text, f"NAMING_CONVENTION.md does not list {name}"
    for name in CONFIG_NAMES:
        assert f"{name}.yaml" in text, f"NAMING_CONVENTION.md does not list {name}.yaml"


def test_naming_convention_doc_lists_nothing_that_was_deleted():
    text = (PROJECT_ROOT / "docs" / "NAMING_CONVENTION.md").read_text(encoding="utf-8")
    for gone in ("qsvc_classifier", "cudaq.yaml"):
        assert gone not in text, f"NAMING_CONVENTION.md still lists the removed {gone}"
