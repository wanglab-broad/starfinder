"""The documentation pages that the synthetic tests read: the API page and the specification (W-602).

A docs-contract module (docs/contributing.md, "Test markers"): these tests read the pages here, so that no ``slow``
synthetic test reads them. Both read whole pages, by content, and split on no heading:

* ``test_ga2_every_new_name_is_on_the_api_page``: the page part of GA2's
  ``test_ga2_every_new_name_is_exported_and_on_the_api_pages`` (``test_synthetic_appearance_coverage.py``), which
  checks the export of the same names (``appearance_names``);
* ``test_no_artifact_contract_or_issue_references``, from ``test_formed_streams.py``.
"""
from pathlib import Path
import re

import pytest

from . import docs_pages
from .appearance_names import NAMES

pytestmark = [pytest.mark.synthetic, pytest.mark.contract]

PACKAGE = Path(__file__).resolve().parents[1] / 'starfinder/synthetic'
DOCS = docs_pages.DOCS


def test_ga2_every_new_name_is_on_the_api_page():
    """Each of GA2's fifteen new names has its autosummary line on the API page."""
    page = docs_pages.read("api/synthetic.rst")
    assert len(NAMES) == 15
    for name in NAMES:
        assert re.search(rf"^   {name}$", page, re.M), name


def test_no_artifact_contract_or_issue_references():
    sources = list(PACKAGE.glob('*.py'))
    docs = [DOCS / 'synthetic-specification.md', DOCS / 'api/synthetic.rst',
            *(DOCS / 'examples' / f'{name}.py' for name in
              ('amplicon_scene', 'readout_effects', 'background_noise', 'amplicon_geometry'))]
    for path in sources + docs:
        text = path.read_text()
        assert 'starfinder.artifacts/1' not in text, path
        assert 'starfinder.provenance' not in text and 'RunRecorder' not in text, path
    for path in docs:
        assert not re.search(r'\bW-\d+', path.read_text()), path
