"""Structural checks on the tutorial notebooks.

Notebooks rot quietly: a renamed function leaves the narrative intact and the
code broken, and nobody notices until someone runs it. These checks are cheap and
need no quantum stack -- they parse every notebook, compile every code cell, and
confirm the executed ones actually carry output.

They do not execute anything. Full execution lives in
``jupyter nbconvert --execute``, which is far too slow for a test run.
"""

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

#: Every notebook backs an implemented tutorial and is committed with outputs.
EXECUTED_NOTEBOOKS = (
    "01-vqe-molecular-energy.ipynb",
    "02-qaoa-portfolio-selection.ipynb",
    "03-quantum-kernel-biomedical-classification.ipynb",
    "04-qaoa-maxcut.ipynb",
    "05-hhl-linear-systems-intro.ipynb",
    "06-variational-heat-equation.ipynb",
)

#: No placeholders remain. Kept so the distinction survives if one is added.
PLACEHOLDER_NOTEBOOKS: tuple[str, ...] = ()


def load(name):
    return json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))


def code_cells(notebook):
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def source_of(cell):
    return "".join(cell["source"])


@pytest.mark.parametrize("name", EXECUTED_NOTEBOOKS + PLACEHOLDER_NOTEBOOKS)
def test_notebook_is_valid_json_and_well_formed(name):
    notebook = load(name)
    assert notebook["nbformat"] == 4
    assert notebook["cells"], f"{name} has no cells"
    assert notebook["cells"][0]["cell_type"] == "markdown", "should open with a title"


@pytest.mark.parametrize("name", EXECUTED_NOTEBOOKS)
def test_every_code_cell_compiles(name):
    """Catches renamed or removed API without paying to execute the notebook."""
    for index, cell in enumerate(code_cells(load(name))):
        source = source_of(cell)
        try:
            compile(source, f"{name}[cell {index}]", "exec")
        except SyntaxError as error:  # pragma: no cover - failure path
            pytest.fail(f"{name} cell {index} does not compile: {error}")


@pytest.mark.parametrize("name", EXECUTED_NOTEBOOKS)
def test_executed_notebooks_carry_output(name):
    """Committed with outputs so they read on GitHub without being run."""
    cells = code_cells(load(name))
    assert cells, f"{name} has no code cells"
    with_output = [c for c in cells if c.get("outputs")]
    assert len(with_output) >= len(cells) // 2, (
        f"{name}: only {len(with_output)}/{len(cells)} code cells have output; "
        "re-run `jupyter nbconvert --to notebook --execute --inplace`"
    )


@pytest.mark.parametrize("name", EXECUTED_NOTEBOOKS)
def test_no_cell_raised_an_error(name):
    """A stored traceback means the notebook was committed broken."""
    for index, cell in enumerate(code_cells(load(name))):
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"{name} cell {index} raised "
                f"{output.get('ename')}: {output.get('evalue')}"
            )


@pytest.mark.parametrize("name", EXECUTED_NOTEBOOKS)
def test_notebook_links_to_its_tutorial(name):
    """Each notebook is a companion to a markdown tutorial, not a replacement."""
    text = "".join(
        source_of(c) for c in load(name)["cells"] if c["cell_type"] == "markdown"
    )
    assert "../tutorials/" in text, f"{name} does not link to its tutorial"


def test_all_notebooks_are_accounted_for():
    """A new notebook must be classified, so it cannot skip these checks."""
    on_disk = {p.name for p in NOTEBOOK_DIR.glob("*.ipynb")}
    assert on_disk == set(EXECUTED_NOTEBOOKS) | set(PLACEHOLDER_NOTEBOOKS)


@pytest.mark.parametrize("name", EXECUTED_NOTEBOOKS)
def test_notebook_is_a_walkthrough_not_a_stub(name):
    """Guards the gap that let two 160-character stubs pass as notebooks."""
    notebook = load(name)
    characters = sum(len("".join(c["source"]).strip()) for c in notebook["cells"])
    assert characters > 1000, f"{name} is {characters} characters -- a stub"
    assert len(code_cells(notebook)) >= 4, f"{name} has too few code cells to walk through anything"
