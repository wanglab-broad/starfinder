"""The Get Started demos: their rendered pages, their saved outputs, and a fresh execution.

The demos are the executed notebooks of ``example/introduction/``; ``docs/examples/render_demos.py``
converts each to a page with its figures under ``docs/get-started/``. The bounds are decision 8
of W-587: at most 2 minutes and 2 GiB on one CPU, at most 1.5 MB per notebook, and at most 2 MB
of figures under ``docs/get-started/``.
"""

import json
import runpy
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
RENDER = runpy.run_path(str(ROOT / "docs" / "examples" / "render_demos.py"))
DEMOS = RENDER["DEMOS"]
NOTEBOOK_BYTES = 1_500_000
FIGURE_BYTES = 2_000_000
SECONDS = 120
RSS_KIB = 2 * 1024 * 1024
# The extras each demo needs beyond the package; without them the execution test skips.
EXTRAS = {"single_fov_demo": (), "sample_demo": ("multiview_stitcher", "anndata")}


def notebook_path(stem):
    return ROOT / "example" / "introduction" / f"{stem}.ipynb"


def errors(notebook):
    return [output for cell in notebook["cells"] if cell["cell_type"] == "code"
            for output in cell["outputs"] if output["output_type"] == "error"]


@pytest.mark.parametrize("stem", sorted(DEMOS))
def test_each_page_equals_a_fresh_conversion_of_its_notebook(stem):
    pytest.importorskip("nbconvert")
    fresh = RENDER["render"](stem)
    committed = RENDER["written"](stem)
    assert sorted(committed) == sorted(fresh)
    for name, content in fresh.items():
        assert committed[name] == content, name


@pytest.mark.parametrize("stem", sorted(DEMOS))
def test_each_notebook_keeps_its_outputs_within_the_size_bound(stem):
    path = notebook_path(stem)
    notebook = json.loads(path.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert path.stat().st_size <= NOTEBOOK_BYTES
    assert all(cell["outputs"] and cell["execution_count"] for cell in code)
    assert [cell["execution_count"] for cell in code] == list(range(1, len(code) + 1))
    assert not errors(notebook)


def test_the_figures_of_the_get_started_pages_stay_within_two_megabytes():
    figures = [p for stem in DEMOS for p in (ROOT / "docs" / "get-started" / f"{DEMOS[stem]}_files").iterdir()]
    assert figures and sum(p.stat().st_size for p in figures) <= FIGURE_BYTES


@pytest.mark.slow
@pytest.mark.parametrize("stem", sorted(DEMOS))
def test_each_demo_runs_without_error_within_its_bounds(stem):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    for module in EXTRAS[stem]:
        pytest.importorskip(module, reason=f"{stem} needs the {module} extra")
    notebook = nbformat.read(notebook_path(stem), as_version=4)
    # One more cell, not saved, reads the kernel's own peak resident memory at the end.
    notebook.cells.append(nbformat.v4.new_code_cell(
        "import resource\nprint(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"))
    start = time.perf_counter()
    nbclient.NotebookClient(notebook, timeout=SECONDS, kernel_name="python3", record_timing=False,
                            resources={"metadata": {"path": str(notebook_path(stem).parent)}}).execute()
    seconds = time.perf_counter() - start
    peak_kib = int(notebook.cells.pop().outputs[0]["text"])
    assert not errors(notebook)
    assert seconds <= SECONDS, seconds
    assert peak_kib <= RSS_KIB, peak_kib
    assert len(nbformat.writes(notebook).encode()) <= NOTEBOOK_BYTES
    assert notebook.cells[-1].outputs[-1]["text"].strip().endswith("workspace removed: True")
