"""Config option tables read descriptions and enforce field completeness."""
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.registration]


@dataclass
class CompleteConfig:
    """Complete fixture.

    Parameters
    ----------
    count : int
        Positive item count.
        Wrapped description with ``code``.
    label, method : str
        Display labels.

    Notes
    -----
    This section is not part of the last description.
    """
    count: int = 1
    label: str = "items"
    method: str = field(default="complete", init=False)


@dataclass
class MissingConfig:
    """Missing fixture.

    Parameters
    ----------
    count : int
        Item count.
    """
    count: int = 1
    method: str = field(default="missing", init=False)


@dataclass
class UnknownConfig:
    """Unknown fixture.

    Parameters
    ----------
    count : int
        Item count.
    typo : str
        An unknown option.
    """
    count: int = 1


@dataclass
class LegacyConfig:
    """A class without a Parameters section is rejected."""
    count: int = 1


@pytest.fixture
def tables():
    pytest.importorskip("docutils", reason="documentation extra required")
    path = Path(__file__).resolve().parents[3] / "docs/_ext/method_tables.py"
    spec = spec_from_file_location("config_options_under_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parameter_descriptions(tables):
    from starfinder.dataset.types import SubtileConfig

    assert tables._parameter_descriptions(SubtileConfig) == {
        "sqrt_pieces": "Positive number of tiles per axis.",
        "overlap_ratio": "Fractional overlap; default 0.1.",
        "windows": "Computed CropWindow list; default new empty list. Call compute_windows before extraction.",
    }
    assert tables._parameter_descriptions(CompleteConfig) == {
        "count": "Positive item count. Wrapped description with ``code``.",
        "label": "Display labels.", "method": "Display labels.",
    }


@pytest.mark.parametrize("config, field_name", [
    (LegacyConfig, "count"), (MissingConfig, "method"), (UnknownConfig, "typo"),
])
def test_parameter_completeness(tables, config, field_name):
    with pytest.raises(ValueError, match=rf"{config.__name__}: .*field '{field_name}'"):
        tables._parameter_descriptions(config)


@pytest.mark.parametrize("config", [CompleteConfig, LegacyConfig, MissingConfig, UnknownConfig])
def test_config_options_directive(tables, monkeypatch, config):
    from docutils import nodes
    from docutils.core import publish_doctree
    from docutils.parsers.rst import directives
    from docutils.utils import SystemMessage

    monkeypatch.setattr(tables, "_resolve", lambda path: config)
    monkeypatch.setitem(directives._directives, "config-options", tables.ConfigOptions)
    source = ".. config-options:: fixture.Config\n   :exclude: method\n"
    if config in (LegacyConfig, MissingConfig, UnknownConfig):
        # Excluding a field from the table does not disable completeness checks.
        with pytest.raises(SystemMessage, match=rf"{config.__name__}: .*field '"):
            publish_doctree(source, settings_overrides={"halt_level": 3})
        return
    tree = publish_doctree(source)
    rows = [[cell.astext() for cell in row.children] for row in tree.findall(nodes.row)]
    assert rows[0] == ["Option", "Default", "Description"]
    assert rows[1:] == [
        ["count", "1", "Positive item count. Wrapped description with code."],
        ["label", "'items'", "Display labels."],
    ]
    assert [node.astext() for node in tree.findall(nodes.literal)][-4:] == [
        "1", "code", "label", "'items'",
    ]
