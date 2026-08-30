"""Source-paper lookup, derived from ``configs/papers.yaml``.

This module used to keep its own hand-maintained dict of citation strings. That
made it the fifth place a paper was recorded, and the one most likely to drift,
because nothing but itself referred to it -- so when Kadowaki & Nishimori and
Sarma et al. were cited in the tutorials, no store here learned about them.

There is now one source of truth. This module reads it, ``papers/index.json`` is
generated from it by ``scripts/generate_paper_index.py``, and
``tests/test_paper_citations.py`` fails if any of them disagree.
"""

from __future__ import annotations

from typing import Any

from qprac_lab.config import load_config

#: Sections, in tutorial order.
SECTIONS = ("simulation", "optimization", "pdes", "qml")


def papers_by_section() -> dict[str, list[dict[str, Any]]]:
    """Every registered paper, grouped by tutorial section."""
    return dict(load_config("papers")["papers"])


def all_papers() -> list[dict[str, Any]]:
    """Every registered paper, flattened, with its section attached."""
    return [
        {**entry, "section": section}
        for section, entries in papers_by_section().items()
        for entry in entries
    ]


def get_papers(topic: str) -> list[dict[str, Any]]:
    """Papers whose ``topic`` matches, case-insensitively.

    Kept as the module's original entry point. It now returns records rather than
    formatted strings, because a citation you cannot resolve is the problem this
    module was rebuilt to fix.
    """
    wanted = topic.strip().lower()
    return [paper for paper in all_papers() if paper.get("topic", "").lower() == wanted]


def resolve(paper: dict[str, Any]) -> str | None:
    """A URL that reaches the paper, preferring the most durable identifier.

    DOI first because it is the one that survives a publisher reorganising its
    site; arXiv next; an explicit URL last, for the papers that predate both.
    """
    if doi := paper.get("doi"):
        return f"https://doi.org/{doi}"
    if arxiv := paper.get("arxiv"):
        return f"https://arxiv.org/abs/{arxiv}"
    return paper.get("url")


def citation(paper: dict[str, Any]) -> str:
    """One-line human citation: ``Authors, "Title", Year. Venue.``"""
    parts = [f'{paper["authors"]}, "{paper["title"]}", {paper["year"]}.']
    if venue := paper.get("venue"):
        parts.append(f"{venue}.")
    return " ".join(parts)


def markdown_citation(paper: dict[str, Any]) -> str:
    """Citation as a Markdown bullet, linked when the paper resolves."""
    link = resolve(paper)
    text = citation(paper)
    return f"- [{text}]({link})" if link else f"- {text}"


#: Backwards-compatible view: ``{topic: [citation strings]}``. Derived, not stored.
PAPER_REGISTRY: dict[str, list[str]] = {}
for _paper in all_papers():
    PAPER_REGISTRY.setdefault(_paper.get("topic", "uncategorised"), []).append(
        citation(_paper)
    )
del _paper
