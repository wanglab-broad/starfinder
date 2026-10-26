"""The Get Started pages: each equals a fresh conversion of its demo notebook, and their figures stay small (W-602).

A docs-contract module (docs/contributing.md, "Test markers"): these tests, from ``test_get_started_demos.py``, read
the pages and figures under ``docs/get-started/`` here, so that the ``slow`` execution of the demos reads no page.
``docs/examples/render_demos.py`` converts each executed notebook of ``example/introduction/`` to its page; the bound
on the figures is decision 8 of W-587, at most 2 MB under ``docs/get-started/``. Both compare whole files and split
no page on a heading.
"""

import runpy
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
RENDER = runpy.run_path(str(ROOT / "docs" / "examples" / "render_demos.py"))
DEMOS = RENDER["DEMOS"]
FIGURE_BYTES = 2_000_000


@pytest.mark.parametrize("stem", sorted(DEMOS))
def test_each_page_equals_a_fresh_conversion_of_its_notebook(stem):
    pytest.importorskip("nbconvert")
    fresh = RENDER["render"](stem)
    committed = RENDER["written"](stem)
    assert sorted(committed) == sorted(fresh)
    for name, content in fresh.items():
        assert committed[name] == content, name


def test_the_figures_of_the_get_started_pages_stay_within_two_megabytes():
    figures = [p for stem in DEMOS for p in (ROOT / "docs" / "get-started" / f"{DEMOS[stem]}_files").iterdir()]
    assert figures and sum(p.stat().st_size for p in figures) <= FIGURE_BYTES
