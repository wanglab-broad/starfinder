"""REGISTRATION_METHODS and the registration places derived from it (method registry page, move 2)."""
import json
import sys
import typing
from dataclasses import dataclass, field

import numpy as np
import pytest

from starfinder._registry import Dependency
from starfinder.benchmark._adapters import _config as benchmark_config
from starfinder.dataset import RecoveryConfig, RegistrationStep, from_workflow_config
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io._checkpoint import _registration_results, write_registered_header
from starfinder.registration import (
    REGISTRATION_METHODS,
    AffineConfig,
    BSplineConfig,
    CpdConfig,
    DemonsConfig,
    InvalidRegistrationConfigError,
    RegistrationBackendUnavailableError,
    RegistrationEstimationError,
    RegistrationSpec,
    RigidConfig,
    TpsConfig,
    TranslationConfig,
    TranslationTransform,
    WarpConfig,
    estimate_transform,
)
from starfinder.registration._methods import RegistrationConfig

REF = ImageMetadata("reference")
MOV = ImageMetadata("moving")


def estimate(shape, config):
    return estimate_transform(np.ones(shape), np.ones(shape), config=config, reference_metadata=REF,
                              moving_metadata=MOV)


def test_registry_table_holds_the_registered_methods():
    table = {config_type: (spec.name, spec.step_kind, spec.dimensions, spec.min_shape_zyx, spec.transform_kind,
                           spec.space, tuple((d.module, d.distribution, d.extra) for d in spec.requires))
             for config_type, spec in REGISTRATION_METHODS.items()}
    elastix = ("itk", "itk-elastix", "registration-elastix")
    simpleitk = ("SimpleITK", "SimpleITK", "local-registration")
    assert table == {
        TranslationConfig: ("translation", "global", {2, 3}, (1, 1, 1), "translation", "index", ()),
        RigidConfig: ("rigid", "global", {2, 3}, (4, 16, 16), "affine", "physical", (elastix,)),
        AffineConfig: ("affine", "global", {2, 3}, (4, 16, 16), "affine", "physical", (elastix,)),
        BSplineConfig: ("bspline", "local", {2, 3}, (4, 16, 16), "bspline", "physical", (elastix, simpleitk)),
        DemonsConfig: ("demons", "local", {2, 3}, (4, 4, 4), "dense", "index", (simpleitk,)),
        TpsConfig: ("tps", "local", {3}, (2, 2, 2), "dense", "index", ()),
        CpdConfig: ("cpd", "local", {3}, (2, 2, 2), "dense", "index", ()),
    }
    assert all(isinstance(spec.dimensions, frozenset) for spec in REGISTRATION_METHODS.values())


def test_method_discriminators_equal_spec_names():
    for config_type, spec in REGISTRATION_METHODS.items():
        assert config_type().method == spec.name


def test_config_alias_members_are_the_registry_keys():
    assert set(typing.get_args(RegistrationConfig)) == set(REGISTRATION_METHODS)


def test_registration_spec_validates_its_declarations():
    ok = dict(name="fixture_shift", run=lambda *a: None, step_kind="global", dimensions=frozenset({2, 3}),
              transform_kind="translation", space="index")
    RegistrationSpec(**ok)
    for bad in (dict(name="FixtureShift"), dict(step_kind="middle"), dict(dimensions={2, 3}),
                dict(dimensions=frozenset()), dict(dimensions=frozenset({1})), dict(transform_kind="rigid"),
                dict(space="world"), dict(min_shape_zyx=(0, 1, 1))):
        with pytest.raises(ValueError):
            RegistrationSpec(**{**ok, **bad})
    with pytest.raises(TypeError):
        RegistrationSpec(**{**ok, "run": None})


@dataclass(frozen=True)
class FixtureShiftConfig:
    offset: float = 0.0
    method: str = field(default="fixture_shift", init=False)

    def __post_init__(self):
        if not isinstance(self.offset, (int, float)):
            raise InvalidRegistrationConfigError("offset must be a number")


def estimate_fixture_shift(reference, moving, config, geometry):
    return TranslationTransform((config.offset, 0, 0), **geometry), "fixture", WarpConfig()


FIXTURE_SPEC = RegistrationSpec("fixture_shift", estimate_fixture_shift, step_kind="global",
                                dimensions=frozenset({2, 3}), transform_kind="translation", space="index",
                                min_shape_zyx=(1, 5, 5))


def workflow_config(tmp_path, **global_registration):
    return dict(root_input_path=str(tmp_path), root_output_path=str(tmp_path / "out"), dataset_id="data",
                sample_id="sample", output_id="run", n_rounds=2, ref_round="round1",
                seq_channel_order=["a", "b"], rotate_angle=0, img_row=12, img_col=14,
                rules={"rsf_single_fov": {"parameters": {"global_registration": {"run": True, **global_registration},
                                                          "local_registration": {"run": False}}}})


def rejected_everywhere(tmp_path):
    """Each derived place rejects an unregistered config or name with its current error."""
    with pytest.raises(InvalidRegistrationConfigError, match="^expected a typed registration config$"):
        estimate((1, 8, 8), FixtureShiftConfig())
    with pytest.raises(TypeError, match="^unsupported registration config$"):
        RegistrationStep(FixtureShiftConfig())
    with pytest.raises(TypeError, match="^invalid recovery configuration$"):
        RecoveryConfig((RegistrationEstimationError,), (FixtureShiftConfig(),))
    with pytest.raises(ValueError, match="^unknown registration method fixture_shift$"):
        from_workflow_config(workflow_config(tmp_path, method="fixture_shift"))
    with pytest.raises(ValueError, match="^unsupported registration method: fixture_shift$"):
        benchmark_config({"method": "fixture_shift"})


def test_every_registration_place_accepts_a_method_registered_only_in_the_registry(tmp_path, monkeypatch):
    rejected_everywhere(tmp_path)
    monkeypatch.setitem(REGISTRATION_METHODS, FixtureShiftConfig, FIXTURE_SPEC)
    # estimate_transform dispatches to the spec's estimator.
    result = estimate((1, 8, 8), FixtureShiftConfig(offset=0.5))
    assert result.transform.correction_zyx == (0.5, 0, 0)
    assert (result.diagnostics.method, result.diagnostics.backend) == ("fixture_shift", "fixture")
    # RegistrationStep and RecoveryConfig accept it.
    recovery = RecoveryConfig((RegistrationEstimationError,), (FixtureShiftConfig(),))
    assert RegistrationStep(TranslationConfig(), recovery=recovery).recovery.alternatives == (FixtureShiftConfig(),)
    assert RegistrationStep(FixtureShiftConfig()).config == FixtureShiftConfig()
    # The workflow adapter resolves its name and accepts its init fields.
    adapted = from_workflow_config(workflow_config(tmp_path, method="fixture_shift", offset=2.0))
    assert [step.config for step in adapted.pipeline.registration] == [FixtureShiftConfig(offset=2.0)]
    # The checkpoint reader rebuilds its saved config from the saved method name.
    write_registered_header(tmp_path, {}, {"round2": [result]})
    saved = json.loads((tmp_path / "registered" / "transforms.json").read_text())
    restored = _registration_results(tmp_path, saved["transforms"])["round2"][0]
    assert restored.diagnostics.effective_config == FixtureShiftConfig(offset=0.5)
    assert restored.transform == result.transform
    # The benchmark adapter builds it from a case's method name.
    assert benchmark_config({"method": "fixture_shift", "offset": 1.0}) == FixtureShiftConfig(offset=1.0)
    monkeypatch.undo()
    rejected_everywhere(tmp_path)


def test_subclasses_of_registered_configs_are_not_matched():
    @dataclass(frozen=True)
    class SubTranslation(TranslationConfig):
        pass

    for call in (lambda: RegistrationStep(SubTranslation()),
                 lambda: RecoveryConfig((RegistrationEstimationError,), (SubTranslation(),))):
        with pytest.raises(TypeError):
            call()
    with pytest.raises(InvalidRegistrationConfigError):
        estimate((1, 8, 8), SubTranslation())


def test_geometry_checks_follow_the_declared_dimensions_and_minimum_shape(monkeypatch):
    monkeypatch.setitem(REGISTRATION_METHODS, FixtureShiftConfig, FIXTURE_SPEC)
    estimate((1, 5, 5), FixtureShiftConfig())
    estimate((2, 5, 5), FixtureShiftConfig())
    with pytest.raises(IncompatibleGeometryError, match="Y and X of at least"):
        estimate((1, 4, 8), FixtureShiftConfig())
    with pytest.raises(IncompatibleGeometryError, match="every axis at least"):
        estimate((3, 8, 4), FixtureShiftConfig())
    monkeypatch.setitem(REGISTRATION_METHODS, FixtureShiftConfig,
                        RegistrationSpec("fixture_shift", estimate_fixture_shift, step_kind="global",
                                         dimensions=frozenset({2}), transform_kind="translation", space="index"))
    with pytest.raises(IncompatibleGeometryError, match="only Z=1"):
        estimate((2, 5, 5), FixtureShiftConfig())
    # The current local methods keep rejecting small axes, and Z=1 as 3D-only methods.
    for config, shape in ((DemonsConfig(), (3, 8, 8)), (DemonsConfig(), (8, 8, 3)), (TpsConfig(), (8, 1, 8)),
                          (CpdConfig(), (1, 8, 8))):
        with pytest.raises(IncompatibleGeometryError, match="3D"):
            estimate(shape, config)


def test_declared_dependencies_are_required_before_the_estimator_runs(monkeypatch):
    calls = []
    spec = RegistrationSpec("fixture_shift", lambda *a: calls.append(a), step_kind="global",
                            dimensions=frozenset({2, 3}), transform_kind="translation", space="index",
                            requires=(Dependency("fixture_missing_backend", "fixture-backend", "fixture-extra"),))
    monkeypatch.setitem(REGISTRATION_METHODS, FixtureShiftConfig, spec)
    monkeypatch.setitem(sys.modules, "fixture_missing_backend", None)
    with pytest.raises(RegistrationBackendUnavailableError) as info:
        estimate((1, 8, 8), FixtureShiftConfig())
    assert str(info.value) == ("registration method 'fixture_shift' requires fixture_missing_backend; "
                               "install the 'fixture-extra' extra (starfinder[fixture-extra])")
    assert calls == []
