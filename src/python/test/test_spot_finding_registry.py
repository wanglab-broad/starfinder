"""SPOT_FINDING_METHODS (registry move 4), the dependency error, the execution device and the detection
provenance entry in run.json (registry move 5) (W-270, docs/spot-finding-contract.md)."""
import json
import subprocess
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import pytest

from starfinder._registry import Dependency, config_type_for, names
from starfinder.dataset import CheckpointConfig, ExecutionConfig, PipelineConfig
from starfinder.dataset.workflow import from_workflow_config
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io._checkpoint import _detectors, _jsonable
from starfinder.spot_finding import (SPOT_FINDING_METHODS, LocalMaximaConfig, NoiseLandmarkConfig,
    PercentileCentroidConfig, SpotFindingBackendUnavailableError, SpotFindingPlan, SpotFindingResult, SpotFindingSpec,
    find_spots)
from starfinder.spot_finding import _methods
from starfinder.spot_finding._methods import SpotFindingConfig

from .test_spot_finding_golden import CHANNELS, fixture_image, fov_with_fixture, golden_dataset

META = ImageMetadata("registry")
NAMESPACE = "registry/test"


@dataclass(frozen=True)
class FixtureConfig:
    """A fixture detection config: one spot at the brightest voxel of each channel above level."""
    level: float = 0.0
    channel_labels: tuple[str, ...] | None = None
    method: str = field(default="fixture_detector", init=False)

    def __post_init__(self):
        if self.level < 0:
            raise ValueError("level must be nonnegative")


def fixture_run(image, config, context):
    rows = []
    for c in context.channels:
        channel = image[..., c] if image.ndim == 4 else image
        if channel.max() > config.level:
            z, y, x = np.unravel_index(int(np.argmax(channel)), channel.shape)
            rows.append((float(z), float(y), float(x), c))
    table = pd.DataFrame(rows, columns=["z", "y", "x", "channel"]).astype(
        {"z": "float64", "y": "float64", "x": "float64", "channel": "int64"})
    return table, {"thresholds": tuple(config.level for _ in context.channels)}


FIXTURE_SPEC = SpotFindingSpec("fixture_detector", fixture_run, pipeline=True, dimensions=frozenset({2, 3}),
                               output_columns=("z", "y", "x", "channel"))


@pytest.fixture
def fixture_method(monkeypatch):
    monkeypatch.setitem(SPOT_FINDING_METHODS, FixtureConfig, FIXTURE_SPEC)
    return FixtureConfig


class SubLocalMaxima(LocalMaximaConfig):
    pass


def image():
    volume = np.zeros((5, 9, 9, 2), dtype=np.uint16)
    volume[2, 4, 4, 0], volume[1, 3, 5, 1] = 900, 400
    return volume


# --- The registry table -----------------------------------------------------------------------------

def test_registry_holds_the_three_existing_methods_with_the_contract_fields():
    table = {config_type: (spec.name, spec.pipeline, spec.dimensions, spec.min_shape_zyx, spec.output_columns,
                           spec.requires, spec.weights)
             for config_type, spec in SPOT_FINDING_METHODS.items()}
    assert table == {
        LocalMaximaConfig: ("local_maxima", True, frozenset({2, 3}), (1, 1, 1),
                            ("z", "y", "x", "channel", "peak_intensity?"), (), False),
        NoiseLandmarkConfig: ("noise_landmark", False, frozenset({2, 3}), (1, 1, 1), ("z", "y", "x"), (), False),
        PercentileCentroidConfig: ("percentile_centroid", False, frozenset({2, 3}), (1, 1, 1), ("z", "y", "x"),
                                   (), False),
    }
    assert names(SPOT_FINDING_METHODS) == ("local_maxima", "noise_landmark", "percentile_centroid")


def test_discriminators_equal_spec_names_and_the_alias_matches_the_registry():
    for config_type, spec in SPOT_FINDING_METHODS.items():
        assert config_type().method == spec.name
    assert set(SpotFindingConfig.__args__) == set(SPOT_FINDING_METHODS)


@pytest.mark.parametrize("change, message", [
    (dict(name="Bad-Name"), "snake_case"), (dict(dimensions=frozenset({1})), "dimensions"),
    (dict(output_columns=("y", "z", "x")), "output_columns"), (dict(output_columns=("z", "y", "x", "spot_id")),
                                                               "output_columns"),
    (dict(min_shape_zyx=(0, 1, 1)), "min_shape_zyx")])
def test_spec_validation(change, message):
    with pytest.raises(ValueError, match=message):
        replace(FIXTURE_SPEC, **change)
    with pytest.raises(TypeError):
        replace(FIXTURE_SPEC, pipeline="yes")


# --- Every lookup sees an inserted method ------------------------------------------------------------

def test_an_inserted_method_is_seen_by_every_lookup(fixture_method, tmp_path):
    config = FixtureConfig(level=100.0)
    result = find_spots(image(), config=config, metadata=META, spot_namespace=NAMESPACE)
    assert result.spots[["z", "y", "x", "channel"]].values.tolist() == [[2, 4, 4, 0], [1, 3, 5, 1]]
    assert result.diagnostics["method"] == "fixture_detector"
    assert SpotFindingResult(result.spots, META, NAMESPACE, config, {}).config == config
    assert PipelineConfig(detection=config).detection == config
    assert PipelineConfig(detection=SpotFindingPlan(config)).detection.config == config
    assert _detectors()["fixture_detector"] is FixtureConfig
    assert config_type_for(SPOT_FINDING_METHODS, "fixture_detector", "spot-finding method") is FixtureConfig
    fov = fov_with_fixture(golden_dataset(tmp_path), "3d")
    fov.find_spots(config=config)
    assert fov.spot_result.config == FixtureConfig(level=100.0, channel_labels=CHANNELS)
    assert len(fov.spot_result.spots) == 4
    workflow = {"n_rounds": 1, "ref_round": "round1", "dataset_id": "d", "sample_id": "s", "output_id": "o",
                "root_input_path": "in", "root_output_path": "out", "seq_channel_order": list(CHANNELS),
                "rules": {"rsf_single_fov": {"parameters": {"load_raw_images": {"run": False}, "spot_finding": {
                    "run": True, "method": "fixture_detector", "level": 3.0}}}}}
    assert from_workflow_config(workflow).pipeline.detection == FixtureConfig(level=3.0)


def test_a_method_removed_from_the_registry_is_rejected_everywhere(monkeypatch, tmp_path):
    monkeypatch.delitem(SPOT_FINDING_METHODS, LocalMaximaConfig)
    with pytest.raises(TypeError, match="unsupported detection config"):
        find_spots(image(), config=LocalMaximaConfig(), metadata=META, spot_namespace=NAMESPACE)
    with pytest.raises(TypeError):
        PipelineConfig(detection=LocalMaximaConfig())
    with pytest.raises(TypeError):
        fov_with_fixture(golden_dataset(tmp_path), "3d").find_spots(config=LocalMaximaConfig())
    assert "local_maxima" not in _detectors()


def test_a_subclass_of_a_detection_config_is_rejected(tmp_path):
    config = SubLocalMaxima()
    with pytest.raises(TypeError, match="^unsupported detection config$"):
        find_spots(image(), config=config, metadata=META, spot_namespace=NAMESPACE)
    table = find_spots(image(), config=LocalMaximaConfig(), metadata=META, spot_namespace=NAMESPACE).spots
    with pytest.raises(TypeError, match="^unsupported detection config$"):
        SpotFindingResult(table, META, NAMESPACE, config, {})
    with pytest.raises(TypeError, match="^unsupported detection config$"):
        SpotFindingPlan(config)
    with pytest.raises(TypeError, match="detection requires its typed operation config"):
        PipelineConfig(detection=config)
    with pytest.raises(TypeError, match="FOV detection requires LocalMaximaConfig"):
        fov_with_fixture(golden_dataset(tmp_path), "3d").find_spots(config=config)


def test_only_pipeline_methods_are_accepted_by_the_pipeline(tmp_path):
    for config in (NoiseLandmarkConfig(), PercentileCentroidConfig()):
        with pytest.raises(TypeError, match="detection requires its typed operation config"):
            PipelineConfig(detection=config)
        with pytest.raises(TypeError, match="detection requires its typed operation config"):
            PipelineConfig(detection=SpotFindingPlan(config))
        with pytest.raises(TypeError, match="FOV detection requires LocalMaximaConfig"):
            fov_with_fixture(golden_dataset(tmp_path), "3d").find_spots(config=config)


# --- Dependency error, geometry and columns ----------------------------------------------------------

def test_missing_dependency_raises_the_spot_finding_error(fixture_method, monkeypatch):
    missing = Dependency("starfinder_fixture_missing_module", "starfinder-fixture-missing", "fixture-extra")
    monkeypatch.setitem(SPOT_FINDING_METHODS, FixtureConfig, replace(FIXTURE_SPEC, requires=(missing,)))
    with pytest.raises(SpotFindingBackendUnavailableError) as raised:
        find_spots(image(), config=FixtureConfig(), metadata=META, spot_namespace=NAMESPACE)
    assert isinstance(raised.value, ImportError)
    assert str(raised.value) == ("spot-finding method 'fixture_detector' requires starfinder_fixture_missing_module; "
                                 "install the 'fixture-extra' extra (starfinder[fixture-extra])")
    monkeypatch.setattr(_methods, "_PY314", True)
    with pytest.raises(SpotFindingBackendUnavailableError, match=r"not available on Python 3\.14 and later\)$"):
        find_spots(image(), config=FixtureConfig(), metadata=META, spot_namespace=NAMESPACE)


def test_importing_spot_finding_imports_no_optional_dependency():
    code = ("import sys, starfinder.spot_finding, starfinder.dataset; "
            "print(sorted(m for m in ('torch', 'spotiflow', 'piscis', 'starfinder.spot_finding._fetch') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_dimensions_and_minimum_shape_are_checked_before_the_method_runs(fixture_method, monkeypatch):
    calls = []
    spec = replace(FIXTURE_SPEC, run=lambda *a: calls.append(a), dimensions=frozenset({3}), min_shape_zyx=(4, 6, 6))
    monkeypatch.setitem(SPOT_FINDING_METHODS, FixtureConfig, spec)
    with pytest.raises(IncompatibleGeometryError, match="Z=1"):
        find_spots(np.ones((1, 9, 9)), config=FixtureConfig(), metadata=META, spot_namespace=NAMESPACE)
    with pytest.raises(IncompatibleGeometryError, match="at least"):
        find_spots(np.ones((3, 9, 9)), config=FixtureConfig(), metadata=META, spot_namespace=NAMESPACE)
    monkeypatch.setitem(SPOT_FINDING_METHODS, FixtureConfig, replace(spec, dimensions=frozenset({2, 3})))
    with pytest.raises(IncompatibleGeometryError, match="Y and X"):
        find_spots(np.ones((1, 5, 9)), config=FixtureConfig(), metadata=META, spot_namespace=NAMESPACE)
    assert calls == []


def test_undeclared_columns_are_rejected(fixture_method, monkeypatch):
    def extra_column(image, config, context):
        table, details = fixture_run(image, config, context)
        return table.assign(score=1.0), details
    monkeypatch.setitem(SPOT_FINDING_METHODS, FixtureConfig, replace(FIXTURE_SPEC, run=extra_column))
    with pytest.raises(ValueError, match="declared"):
        find_spots(image(), config=FixtureConfig(), metadata=META, spot_namespace=NAMESPACE)


# --- Execution device and the run.json provenance entry ---------------------------------------------

def test_device_accepts_only_cpu():
    assert ExecutionConfig().device == "cpu"
    for device in ("cuda", "CPU", None):
        with pytest.raises(ValueError, match="^device must be 'cpu'; §2.7 runs on CPU only$"):
            ExecutionConfig(device=device)
        with pytest.raises(ValueError, match="^device must be 'cpu'"):
            find_spots(image(), config=LocalMaximaConfig(), metadata=META, spot_namespace=NAMESPACE, device=device)


def test_execution_entry_records_device_and_threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    execution = find_spots(image(), config=LocalMaximaConfig(), metadata=META,
                           spot_namespace=NAMESPACE).diagnostics["execution"]
    assert execution["device"] == "cpu" and execution["framework"] is None
    threads = execution["threads"]
    assert set(threads) == {"torch_num_threads", "torch_num_interop_threads", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"}
    assert threads["OMP_NUM_THREADS"] == "1" and threads["NUMBA_NUM_THREADS"] is None


def test_run_json_records_the_device_and_the_detection_provenance(tmp_path):
    dataset = golden_dataset(tmp_path)
    config = LocalMaximaConfig(threshold_mode="adaptive", threshold_value=0.2)
    checkpoints = CheckpointConfig(stages=("candidates",), directory=tmp_path / "checkpoints")
    fov = fov_with_fixture(dataset, "3d").run(PipelineConfig(detection=config), execution=ExecutionConfig(),
                                             checkpoints=checkpoints)
    data = json.loads((tmp_path / "checkpoints" / "FOV_001" / "run.json").read_text())
    assert data["config"]["execution"] == {"mode": "batch", "retain_images": False, "device": "cpu"}
    detection = [step for step in data["steps"] if step["name"] == "find_spots"]
    assert len(detection) == 1 and detection[0]["status"] == "succeeded"
    (entry,) = detection[0]["methods"]
    assert list(entry) == ["stage", "method", "config_type", "implementation", "config", "requires", "artifacts",
                           "execution"]
    assert entry["stage"] == "spot_finding" and entry["method"] == "local_maxima"
    assert entry["config_type"] == "starfinder.spot_finding.LocalMaximaConfig"
    assert entry["implementation"] == "starfinder.spot_finding._methods._local_maxima"
    assert entry["config"] == _jsonable(replace(config, channel_labels=CHANNELS))
    assert (entry["requires"], entry["artifacts"]) == ({}, [])
    assert entry["execution"] == json.loads(json.dumps(fov.spot_result.diagnostics["execution"]))
    assert all("methods" not in step for step in data["steps"] if step["name"] != "find_spots")
