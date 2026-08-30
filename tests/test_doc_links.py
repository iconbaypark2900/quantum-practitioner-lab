"""Every relative link in the docs must resolve, and every tutorial must be listed.

The section READMEs spent the whole scaffold phase advertising tutorials that were
never written: `01-simulation` promised a "Quantum Phase Estimation Overview",
`02-optimization` a "Scheduling QUBO", `03-pdes` a "Poisson Equation Demo". None
existed. `configs/tutorials.yaml` was honest the whole time, because
`tests/test_configs.py` checks its paths -- the READMEs drifted precisely because
nothing checked *them*.

That is the same argument `DECISIONS.md` #9 makes about the configs, applied to
prose. Two directions are enforced here:

* a link that points at nothing fails (advertising what does not exist);
* a tutorial its section README never mentions fails (hiding what does).

Needs no quantum stack.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from qprac_lab.config import PROJECT_ROOT

#: Directories that are not ours to validate.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
    }
)

#: ``[text](target)`` -- deliberately not matching image or reference syntax.
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#")

TUTORIAL_SECTIONS = ("01-simulation", "02-optimization", "03-pdes", "04-qml", "05-benchmarking")


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*.md")
        if not any(part in SKIP_DIRS for part in path.parts)
    )


def _label(path: Path) -> str:
    """Repo-relative label that does not explode on a path from outside the tree."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _relative_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [
        target
        for target in MARKDOWN_LINK.findall(text)
        if not target.startswith(EXTERNAL_PREFIXES)
    ]


def test_markdown_files_are_discovered():
    """Guard the guard: an empty sweep would make every check below vacuous."""
    files = _markdown_files()
    assert len(files) > 20, f"only found {len(files)} markdown files; the sweep is wrong"


@pytest.mark.parametrize("path", _markdown_files(), ids=_label)
def test_every_relative_link_resolves(path):
    """A link to a file that does not exist is a doc advertising fiction."""
    broken = []
    for target in _relative_links(path):
        # Strip any #anchor; we check the file exists, not the heading.
        destination = (path.parent / target.split("#")[0]).resolve()
        if not destination.exists():
            broken.append(target)
    assert not broken, f"{_label(path)} links to missing files: {broken}"


@pytest.mark.parametrize("section", TUTORIAL_SECTIONS)
def test_every_tutorial_is_listed_in_its_section_readme(section):
    """The other direction: a tutorial nobody links to is a tutorial nobody finds."""
    directory = PROJECT_ROOT / "tutorials" / section
    readme = directory / "README.md"
    assert readme.exists(), f"{section} has no README"

    listed = set(_relative_links(readme))
    unlisted = [
        page.name
        for page in sorted(directory.glob("*.md"))
        if page.name not in {"README.md", "papers.md"} and page.name not in listed
    ]
    assert not unlisted, (
        f"tutorials/{section}/README.md never links to {unlisted} -- either link "
        "them or delete them."
    )
