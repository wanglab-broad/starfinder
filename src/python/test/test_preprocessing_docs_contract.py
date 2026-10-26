"""The entry "Preprocessing recipe" of docs/migration.md, against the fields of ``PipelineConfig`` (W-602).

A docs-contract module (docs/contributing.md, "Test markers"): the test, from ``test_preprocessing_recipe.py``, reads
the page here and finds the entry's table by its label ``(migration-preprocessing-recipe)=`` (``docs_pages``), not by
its heading.
"""
from dataclasses import fields

import pytest

from starfinder.dataset import PipelineConfig

from . import docs_pages

pytestmark = [pytest.mark.preprocessing, pytest.mark.contract]


def test_pipeline_config_has_no_preprocessing_slots_or_projection():
    removed = {"normalization", "histogram", "histogram_reference_channel", "reconstruction",
               "reconstruction_after_registration", "tophat", "projection"}
    names = {f.name for f in fields(PipelineConfig)}
    assert "preprocessing" in names and not removed & names
    for name in removed:
        with pytest.raises(TypeError):
            PipelineConfig(**{name: None})
    section = docs_pages.table(docs_pages.read("migration.md"), "migration-preprocessing-recipe")
    for name in removed:
        assert f"`PipelineConfig.{name}`" in section
