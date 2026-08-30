"""Cross-check the paper citations against the machine-readable registry.

``DECISIONS.md`` #6 declares source papers to be project assets, maintained in
Markdown *and* in machine-readable YAML/JSON. In practice the project grew five
places to record a paper -- ``configs/papers.yaml``, ``papers/index.json``,
``papers/<section>.md``, ``tutorials/<section>/papers.md``, and
``src/qprac_lab/papers/registry.py`` -- and nothing compared them. They drifted
exactly the way the configs did, and for the same reason.

The drift this test was written to catch: Kadowaki & Nishimori and Sarma et al.
are cited in the prose but appear in neither machine-readable store, so a reader
following the citation trail from the registry finds two fewer papers than the
tutorials actually lean on.

The identifier floor is the second half. Every entry currently carries a title,
authors and a year, but nothing that resolves to the paper itself -- no DOI, no
arXiv id, no URL. For a project whose whole appeal is traceability that is the
weakest link in it, so coverage is ratcheted rather than merely wished for: the
floor rises as ``/backfill-papers`` fills them in, and can never fall.

Needs no quantum stack.
"""

from __future__ import annotations

import json
import re

import pytest

from qprac_lab.config import PROJECT_ROOT, load_config
from qprac_lab.papers.registry import PAPER_REGISTRY

TOPICS = ("simulation", "optimization", "pdes", "qml")

#: ``tutorials/<dir>/papers.md`` -> registry topic.
TUTORIAL_SECTIONS = {
    "01-simulation": "simulation",
    "02-optimization": "optimization",
    "03-pdes": "pdes",
    "04-qml": "qml",
}

#: Papers cited in prose that no machine-readable store lists yet.
#:
#: A ratchet, not an exemption: shrink it by adding the paper to
#: ``configs/papers.yaml`` and ``papers/index.json``, never by adding a new title.
#: See the ``/backfill-papers`` command.
KNOWN_UNREGISTERED = frozenset(
    {
        "Quantum annealing in the transverse Ising model",
        "Quantum Variational Solving of Nonlinear and Multi-Dimensional "
        "Partial Differential Equations",
    }
)

#: Fields every registry entry must already carry.
REQUIRED_FIELDS = ("title", "authors", "year", "topic")

#: Fields that make a citation *resolvable*. At least one is what the floor counts.
IDENTIFIER_FIELDS = ("doi", "arxiv", "url")

#: How many registry entries currently resolve to the paper itself. Raise this as
#: identifiers are backfilled; the test fails if it is ever set above reality, and
#: fails if reality drops below it.
IDENTIFIER_COVERAGE_FLOOR = 0

QUOTED_TITLE = re.compile(r'"([^"]+)"')


def _registry() -> dict[str, list[dict]]:
    """The YAML registry, which is the store ``qprac_lab.config`` actually loads."""
    return load_config("papers")["papers"]


def _index() -> dict[str, list[dict]]:
    """The JSON registry shipped alongside the prose library."""
    path = PROJECT_ROOT / "papers" / "index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise(title: str) -> str:
    """Collapse whitespace so a line-wrapped title still matches."""
    return " ".join(title.split()).strip().lower()


def _registered_titles() -> set[str]:
    return {
        _normalise(entry["title"]) for entries in _registry().values() for entry in entries
    }


def _cited_titles(path) -> set[str]:
    """Quoted paper titles in one Markdown file."""
    if not path.exists():
        return set()
    return {_normalise(match) for match in QUOTED_TITLE.findall(path.read_text(encoding="utf-8"))}


def _all_citation_files():
    """Every Markdown file that cites papers, paired with its topic."""
    for topic in TOPICS:
        yield topic, PROJECT_ROOT / "papers" / f"{topic}.md"
    for directory, topic in TUTORIAL_SECTIONS.items():
        yield topic, PROJECT_ROOT / "tutorials" / directory / "papers.md"


# ------------------------------------------------------------------ stores agree


@pytest.mark.parametrize("topic", TOPICS)
def test_yaml_and_json_registries_list_the_same_papers(topic):
    """Two machine-readable stores that disagree are worse than one."""
    yaml_titles = {_normalise(entry["title"]) for entry in _registry().get(topic, [])}
    json_titles = {_normalise(entry["title"]) for entry in _index().get(topic, [])}
    assert yaml_titles == json_titles, (
        f"{topic}: configs/papers.yaml and papers/index.json disagree. "
        f"only in yaml={sorted(yaml_titles - json_titles)}, "
        f"only in json={sorted(json_titles - yaml_titles)}"
    )


def test_module_registry_titles_are_all_registered():
    """``papers/registry.py`` is a third store; keep it pinned to the first."""
    registered = _registered_titles()
    unknown = set()
    for citations in PAPER_REGISTRY.values():
        for citation in citations:
            # Entries read "Authors -- Title"; the title is what the registry keys on.
            _, _, title = citation.partition("—")
            title = title or citation
            if _normalise(title) not in registered:
                unknown.add(_normalise(title))
    assert not unknown, (
        f"titles in src/qprac_lab/papers/registry.py that no registry lists: {sorted(unknown)}"
    )


# ------------------------------------------------------------------ prose resolves


def test_every_cited_paper_is_registered():
    """The check that catches a citation trail leading nowhere."""
    registered = _registered_titles()
    allowed = {_normalise(title) for title in KNOWN_UNREGISTERED}
    unresolved: dict[str, set[str]] = {}
    for _topic, path in _all_citation_files():
        missing = _cited_titles(path) - registered - allowed
        if missing:
            unresolved[str(path.relative_to(PROJECT_ROOT))] = missing
    assert not unresolved, (
        "these papers are cited in prose but listed in no machine-readable store: "
        f"{ {k: sorted(v) for k, v in unresolved.items()} }. Add them to "
        "configs/papers.yaml and papers/index.json (DECISIONS.md #6)."
    )


def test_known_unregistered_list_only_shrinks():
    """A ratchet entry stays only while the paper is still both cited and missing."""
    registered = _registered_titles()
    cited = set().union(*(_cited_titles(path) for _topic, path in _all_citation_files()))

    for title in KNOWN_UNREGISTERED:
        normalised = _normalise(title)
        assert normalised in cited, (
            f"{title!r} is on the ratchet list but is cited nowhere -- remove it."
        )
        assert normalised not in registered, (
            f"{title!r} is now registered -- remove it from KNOWN_UNREGISTERED so the "
            "ratchet tightens."
        )


# ------------------------------------------------------------------ entries resolve


@pytest.mark.parametrize("topic", TOPICS)
def test_registry_entries_carry_the_required_fields(topic):
    entries = _registry().get(topic, [])
    assert entries, f"{topic} has no papers registered"
    for entry in entries:
        missing = [field for field in REQUIRED_FIELDS if not entry.get(field)]
        assert not missing, f"{topic}: {entry.get('title')!r} is missing {missing}"


def test_any_identifier_present_is_well_formed():
    """Backfilling is only worth it if the identifiers actually resolve."""
    for topic, entries in _registry().items():
        for entry in entries:
            if doi := entry.get("doi"):
                assert str(doi).startswith("10."), (
                    f"{topic}: {entry['title']!r} has doi={doi!r}, which is not a DOI"
                )
            if arxiv := entry.get("arxiv"):
                assert re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?|[a-z-]+/\d{7}", str(arxiv)), (
                    f"{topic}: {entry['title']!r} has arxiv={arxiv!r}, which is not an arXiv id"
                )
            if url := entry.get("url"):
                assert str(url).startswith("https://"), (
                    f"{topic}: {entry['title']!r} has a non-https url"
                )


def test_identifier_coverage_does_not_regress():
    """The ratchet that turns 'we should add DOIs' into something that can fail."""
    entries = [entry for entries in _registry().values() for entry in entries]
    resolvable = [
        entry for entry in entries if any(entry.get(field) for field in IDENTIFIER_FIELDS)
    ]
    assert len(resolvable) >= IDENTIFIER_COVERAGE_FLOOR, (
        f"{len(resolvable)} of {len(entries)} papers carry a DOI, arXiv id or URL, "
        f"below the floor of {IDENTIFIER_COVERAGE_FLOOR}."
    )
    assert IDENTIFIER_COVERAGE_FLOOR <= len(entries), (
        "the floor exceeds the number of registered papers, so it can never be met"
    )
