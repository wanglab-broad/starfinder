"""The shared method-registry helper starfinder._registry (method registry page, move 0)."""
import sys
from dataclasses import dataclass, field
from importlib import metadata

import pytest

from starfinder._registry import (
    Dependency,
    check_name,
    check_shared,
    config_type_for,
    names,
    provenance,
    require,
    spec_for,
)

pytestmark = [pytest.mark.core, pytest.mark.contract]


@dataclass(frozen=True)
class AlphaConfig:
    size: int = 3
    scales: tuple[float, ...] = (1.0, 2.0)
    method: str = field(default="alpha", init=False)


@dataclass(frozen=True)
class BetaConfig:
    method: str = field(default="beta_two", init=False)


@dataclass(frozen=True)
class SubAlphaConfig(AlphaConfig):
    pass


def run_alpha(image, config):
    return image


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    run: object
    requires: tuple = ()
    min_shape_zyx: tuple = (1, 1, 1)


def registry():
    return {AlphaConfig: FixtureSpec("alpha", run_alpha),
            BetaConfig: FixtureSpec("beta_two", run_alpha, (Dependency("json", "numpy", "extra-name"),))}


@pytest.mark.parametrize("name", ["alpha", "beta_two", "b2", "a_1_b"])
def test_snake_case_names_are_accepted(name):
    check_name(name, "fixture method")


@pytest.mark.parametrize("name", ["Alpha", "_alpha", "alpha_", "alpha__two", "2alpha", "alpha-two", "", None, 3])
def test_other_names_raise_with_the_stage_noun(name):
    with pytest.raises(ValueError) as info:
        check_name(name, "fixture method")
    assert str(info.value) == "fixture method name must be lowercase snake_case"


def test_shared_fields_are_validated():
    check_shared(FixtureSpec("alpha", run_alpha), "fixture method")
    with pytest.raises(TypeError, match="^fixture method run must be callable$"):
        check_shared(FixtureSpec("alpha", None), "fixture method")
    with pytest.raises(TypeError, match="requires must be a tuple of Dependency"):
        check_shared(FixtureSpec("alpha", run_alpha, ("json",)), "fixture method")
    for shape in ((1, 1), (0, 1, 1), (1.0, 1, 1), (True, 1, 1), [1, 1, 1]):
        with pytest.raises(ValueError, match="min_shape_zyx must be three positive integers"):
            check_shared(FixtureSpec("alpha", run_alpha, min_shape_zyx=shape), "fixture method")


def test_lookup_uses_the_exact_config_type():
    table = registry()
    assert spec_for(table, AlphaConfig(), "fixture method") is table[AlphaConfig]
    with pytest.raises(TypeError) as info:
        spec_for(table, SubAlphaConfig(), "fixture method")
    assert str(info.value) == ("no fixture method is registered for SubAlphaConfig "
                               "(lookup uses the exact config type)")
    with pytest.raises(KeyError) as info:
        spec_for(table, "alpha", "fixture method", KeyError, "expected a typed fixture config")
    assert info.value.args == ("expected a typed fixture config",)


def test_name_lookup_and_names_follow_the_mapping(monkeypatch):
    table = registry()
    assert names(table) == ("alpha", "beta_two")
    assert config_type_for(table, "beta_two", "fixture method") is BetaConfig
    with pytest.raises(ValueError) as info:
        config_type_for(table, "gamma", "fixture method")
    assert str(info.value) == "unknown fixture method 'gamma'"
    with pytest.raises(LookupError) as info:
        config_type_for(table, "gamma", "fixture method", LookupError, "unsupported fixture method: gamma")
    assert str(info.value) == "unsupported fixture method: gamma"
    # Nothing is cached: an entry inserted later is found by every lookup.
    monkeypatch.setitem(table, SubAlphaConfig, FixtureSpec("sub_alpha", run_alpha))
    assert names(table) == ("alpha", "beta_two", "sub_alpha")
    assert config_type_for(table, "sub_alpha", "fixture method") is SubAlphaConfig
    assert spec_for(table, SubAlphaConfig(), "fixture method").name == "sub_alpha"


def test_duplicate_names_are_rejected_at_lookup():
    table = registry()
    table[SubAlphaConfig] = FixtureSpec("alpha", run_alpha)
    assert names(table) == ("alpha", "beta_two", "alpha")
    with pytest.raises(ValueError) as info:
        config_type_for(table, "alpha", "fixture method")
    assert str(info.value) == "fixture method name 'alpha' is registered more than once"
    assert config_type_for(table, "beta_two", "fixture method") is BetaConfig


def test_dependency_fields():
    assert Dependency("SimpleITK", "SimpleITK") == Dependency("SimpleITK", "SimpleITK", None)
    for bad in (dict(module=""), dict(distribution=None), dict(extra="")):
        with pytest.raises(TypeError):
            Dependency(**{**dict(module="m", distribution="d", extra="x"), **bad})


def test_missing_lazy_dependency_names_the_module_and_the_extra(monkeypatch):
    spec = FixtureSpec("alpha", run_alpha, (Dependency("fixture_missing_module", "fixture-missing", "fixture-extra"),))
    monkeypatch.setitem(sys.modules, "fixture_missing_module", None)
    with pytest.raises(RuntimeError) as info:
        require(spec, "fixture method", RuntimeError)
    assert str(info.value) == ("fixture method 'alpha' requires fixture_missing_module; "
                               "install the 'fixture-extra' extra (starfinder[fixture-extra])")
    assert isinstance(info.value.__cause__, ImportError)
    spec = FixtureSpec("alpha", run_alpha, (Dependency("fixture_missing_module", "fixture-missing"),))
    with pytest.raises(RuntimeError) as info:
        require(spec, "fixture method", RuntimeError)
    assert str(info.value) == "fixture method 'alpha' requires fixture_missing_module"
    # Declaring a missing dependency costs nothing until require runs; present modules import without error.
    FixtureSpec("alpha", run_alpha, (Dependency("fixture_never_imported", "fixture-missing"),))
    assert "fixture_never_imported" not in sys.modules
    require(registry()[BetaConfig], "fixture method", RuntimeError)


def test_provenance_entry():
    table = registry()
    entry = provenance(table[BetaConfig], BetaConfig(), "fixture_stage")
    assert entry == {"stage": "fixture_stage", "method": "beta_two", "config_type": f"{__name__}.BetaConfig",
                     "implementation": f"{__name__}.run_alpha", "config": {"method": "beta_two"},
                     "requires": {"numpy": metadata.version("numpy")}, "artifacts": []}
    entry = provenance(table[AlphaConfig], AlphaConfig(size=5), "fixture_stage")
    assert entry["config"] == {"size": 5, "scales": [1.0, 2.0], "method": "alpha"}
    assert entry["requires"] == {}
    # Private module components are dropped from the config type; a missing distribution has no version.
    spec = FixtureSpec("alpha", run_alpha, (Dependency("json", "fixture-not-installed"),))
    assert provenance(spec, AlphaConfig(), "fixture_stage")["requires"] == {"fixture-not-installed": None}
    from starfinder.registration import DemonsConfig
    assert provenance(spec, DemonsConfig(), "registration")["config_type"] == "starfinder.registration.DemonsConfig"
