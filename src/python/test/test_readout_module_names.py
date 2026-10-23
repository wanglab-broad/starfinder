"""The module renames of W-556 (W-526, NM-12 and NM-13): the old module paths are gone, with no alias."""
import importlib

import pytest

pytestmark = [pytest.mark.readout, pytest.mark.contract]


@pytest.mark.parametrize("module", ["starfinder.barcode", "starfinder.sample_output", "starfinder.evaluation.barcode"])
def test_old_module_paths_do_not_import(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("module", ["starfinder.readout", "starfinder.export", "starfinder.evaluation.readout"])
def test_new_module_paths_import(module):
    assert importlib.import_module(module).__name__ == module
