"""The native Starfish LoG, starfish_log (W-271; docs/spot-finding-algorithms.md, "Starfish LoG", checks
S1, S2 and S12; docs/spot-finding-contract.md).

Bounds and their sources:
* S12 parity: exact equality with the six W-266 starfish BlobDetector tables in
  data/starfish_blob_parity (W-266 parity result, prototype equal with check_exact=True).
* S1 and S2 on the W-266 isolated-spot scenes, seeds 100-102 (W-266 detectors.csv and
  localization-per-axis.csv): iso3d recall 1.0, precision >= 0.98 and 3D distance <= 0.9 voxels
  (W-266 max 0.781); iso_z1 recall in [0.40, 0.60] (W-266 0.45, 0.51, 0.48), precision >= 0.98 and
  lateral distance <= 0.9 px (W-266 max 0.546).
* Intensity rules: exact equality with a run on the explicitly dtype-scaled float32 image
  (the dtype-scaled images of W-266's LoG runs and parity fixtures); the memory estimate is
  10.4 bytes x num_sigma x voxels (W-266 measurement).
"""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder.dataset import CheckpointConfig, PipelineConfig
from starfinder.evaluation.spot_finding import evaluate_spots, localization_errors
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.spot_finding import (SPOT_FINDING_METHODS, ChannelOverride, SpotFindingPlan, StarfishLogConfig,
    find_spots)
from starfinder.spot_finding import _starfish_log
from starfinder.spot_finding._starfish_log import starfish_view

from .spot_finding_scenes import SEEDS, isolated_scene
from .test_spot_finding_golden import CHANNELS, fixture_image, fov_with_fixture, golden_dataset
from .test_spot_finding_workflow_key import detection

pytestmark = pytest.mark.spot_finding

PARITY = Path(__file__).parent / "data" / "starfish_blob_parity"
META = ImageMetadata("starfish-log")
NAMESPACE = "starfish-log/test"
# The starfish ISS tutorial values W-266 used; an example, not a default.
ISS = StarfishLogConfig(min_sigma=1, max_sigma=10, num_sigma=30, threshold=0.01)
W266_MATCH = dict(policy="greedy", threshold=3.0, boundary="inclusive", units="voxel")


def _fixtures():
    spec = importlib.util.spec_from_file_location("starfish_blob_parity_fixtures", PARITY / "fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = _fixtures()
RECORD = json.loads((PARITY / "starfish-tables.json").read_text())
TABLES = [(case, key) for case, entry in RECORD["cases"].items() for key in entry["tables"]]


def detect(image, config, **options):
    return find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE, **options)


def parity_image(name):
    """The W-266 fixture as a ZYXC float32 image in [0, 1] (its single round)."""
    array, _ = FIXTURES.fixture(name)
    return np.moveaxis(array[0], 0, -1)


def starfish_table(entry):
    """One saved starfish table with its recorded column order and dtypes (W-266 check_prototype.load_table)."""
    frame = pd.read_csv(PARITY / entry["csv"])[entry["columns"]]
    if len(frame):
        return frame.astype(entry["dtypes"])
    return pd.DataFrame({k: pd.Series([], dtype=v) for k, v in entry["dtypes"].items()})[entry["columns"]]


# --- S12: starfish parity -----------------------------------------------------------------------------

def test_the_six_parity_tables_and_their_fixtures_are_the_w266_ones():
    assert len(TABLES) == 6
    assert [RECORD["cases"][case]["tables"][key]["rows"] for case, key in TABLES] == [9, 8, 9, 8, 11, 0]
    for case, (name, settings) in FIXTURES.CASES.items():
        array, _ = FIXTURES.fixture(name)
        assert hashlib.sha256(array.tobytes()).hexdigest() == RECORD["cases"][case]["array_sha256"]
        assert json.loads(json.dumps(settings)) == RECORD["cases"][case]["settings"]


@pytest.mark.validation
@pytest.mark.parametrize("case, key", TABLES)
def test_s12_the_table_equals_starfish_blob_detector(case, key):
    name, settings = FIXTURES.CASES[case]
    image = parity_image(name)
    result = detect(image, StarfishLogConfig(**settings))
    channel = int(key.split("_c")[1])
    rows = result.spots[result.spots.channel == channel].drop(columns="spot_id")
    pd.testing.assert_frame_equal(starfish_view(rows, image[..., channel]),
                                  starfish_table(RECORD["cases"][case]["tables"][key]), check_exact=True)


@pytest.mark.validation
def test_s12_a_three_tuple_sigma_on_a_plane_raises():
    name, settings = FIXTURES.ERROR_CASES["plane-anisotropic-3tuple"]
    with pytest.raises(IncompatibleGeometryError, match="Z=1"):
        detect(parity_image(name), StarfishLogConfig(**settings))
    # The same sigmas run on a volume, and numbers run on the plane.
    detect(parity_image("volume-2ch"), StarfishLogConfig(**settings))
    detect(parity_image(name), ISS)


def test_the_table_holds_the_contract_columns_and_starfish_steps():
    image = parity_image("volume-2ch")
    result = detect(image, StarfishLogConfig(**FIXTURES.LOG_SETTINGS_ANISOTROPIC))
    spots = result.spots
    assert list(spots.columns) == ["spot_id", "z", "y", "x", "channel", "peak_intensity", "radius"]
    assert spots.dtypes.drop("spot_id").tolist() == [np.dtype("float64")] * 3 + [np.dtype("int64")] + \
        [np.dtype("float64")] * 2
    assert list(spots.spot_id) == [str(i) for i in range(len(spots))]
    assert (spots[["z", "y", "x"]] == spots[["z", "y", "x"]].round()).all().all()
    z, y, x, c = (spots[a].to_numpy().astype(int) for a in ("z", "y", "x", "channel"))
    assert np.array_equal(spots.peak_intensity.to_numpy(), image[z, y, x, c].astype(np.float64))
    assert result.diagnostics["method"] == "starfish_log" and result.diagnostics["thresholds"] == (0.01, 0.01)
    assert set(result.diagnostics["measurements"]) == {"peak_intensity", "radius"}
    plane = detect(parity_image("plane"), ISS).spots
    assert len(plane) == 11 and (plane.z == 0).all()


# --- S1 and S2: the W-266 isolated-spot scenes ----------------------------------------------------

def isolated(case, seed):
    image, truth = isolated_scene(case, seed)
    detected = detect(image, ISS).spots[["z", "y", "x"]].to_numpy()
    match = evaluate_spots(detected, truth, reference_metadata=META, observed_metadata=META, **W266_MATCH)
    return match, localization_errors(match, detected, truth)


@pytest.mark.validation
@pytest.mark.parametrize("seed", SEEDS)
def test_s1_s2_iso3d_recall_precision_and_distance(seed):
    match, errors = isolated("iso3d", seed)
    assert match.values["recall"] == 1.0
    assert match.values["precision"] >= 0.98
    assert errors.values["dist_max"] <= 0.9


@pytest.mark.validation
@pytest.mark.parametrize("seed", SEEDS)
def test_s1_s2_iso_z1_recall_precision_and_lateral_distance(seed):
    match, errors = isolated("iso_z1", seed)
    assert 0.40 <= match.values["recall"] <= 0.60
    assert match.values["precision"] >= 0.98
    assert errors.values["lateral_max"] <= 0.9


# --- Intensity rules and the memory estimate ------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_integer_images_are_scaled_by_their_dtype_maximum(dtype):
    maximum = np.iinfo(dtype).max
    image = np.rint(parity_image("volume-2ch") * maximum).astype(dtype)
    # img_as_float32 (as a starfish ImageStack holds integer data): float32 times the reciprocal maximum.
    explicit = image.astype(np.float32) * np.float32(1 / maximum)
    config = StarfishLogConfig(**FIXTURES.LOG_SETTINGS_ANISOTROPIC)
    integer, scaled = detect(image, config), detect(explicit, config)
    assert len(integer.spots) > 0
    pd.testing.assert_frame_equal(integer.spots.drop(columns="peak_intensity"),
                                  scaled.spots.drop(columns="peak_intensity"), check_exact=True)
    z, y, x, c = (integer.spots[a].to_numpy().astype(int) for a in ("z", "y", "x", "channel"))
    assert np.array_equal(integer.spots.peak_intensity.to_numpy(), image[z, y, x, c].astype(np.float64))
    assert np.array_equal(scaled.spots.peak_intensity.to_numpy(), explicit[z, y, x, c].astype(np.float64))


@pytest.mark.parametrize("image", [np.full((4, 16, 16), 1.5, dtype=np.float32),
                                   np.full((4, 16, 16), -0.25), np.full((1, 16, 16), -3, dtype=np.int16)])
def test_images_outside_the_unit_range_raise(image):
    image = image.copy()
    image[2 if image.shape[0] > 1 else 0, 8, 8] = 0.5 if image.dtype.kind == "f" else 5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        detect(image, ISS)


def test_a_float_image_in_the_unit_range_is_used_as_float32():
    image = parity_image("volume-2ch")
    pd.testing.assert_frame_equal(detect(image.astype(np.float64), ISS).spots, detect(image, ISS).spots,
                                  check_exact=True)


@pytest.mark.parametrize("shape", [(32, 64, 64), (1, 64, 64)])
def test_the_scale_space_memory_estimate_is_recorded(shape):
    result = detect(np.zeros(shape, dtype=np.uint16), ISS)
    voxels = shape[0] * shape[1] * shape[2]
    geometry = result.diagnostics["geometry"]
    assert geometry["scale_space_bytes_estimate"] == 10.4 * 30 * voxels
    assert (geometry["voxels"], geometry["num_sigma"], geometry["bytes_per_voxel_and_sigma"]) == (voxels, 30, 10.4)


def test_constant_channels_yield_no_rows_without_running_blob_log(monkeypatch):
    calls = []
    original = _starfish_log.blob_log

    def spy(image, **options):
        calls.append(image.shape)
        return original(image, **options)

    monkeypatch.setattr(_starfish_log, "blob_log", spy)
    image = np.zeros((8, 32, 32, 3), dtype=np.uint16)
    image[..., 1] = 100
    image[..., 2] = np.rint(parity_image("empty")[..., 0] * 65535 + 1000)
    image[4, 16, 16, 2] = 30000
    result = detect(image, ISS)
    assert calls == [(8, 32, 32)]
    assert set(result.spots.channel) <= {2}
    empty = detect(np.zeros((8, 32, 32), dtype=np.uint16), ISS)
    assert len(empty.spots) == 0 and list(empty.spots.columns) == list(result.spots.columns)
    assert empty.spots.dtypes.equals(result.spots.dtypes)


def test_a_channel_override_records_its_settings_and_the_largest_estimate():
    image = parity_image("volume-2ch")
    labels = ("a", "b")
    base = StarfishLogConfig(1, 10, 10, 0.01, channel_labels=labels)
    override = replace(base, num_sigma=20, threshold=0.02, channel_labels=None)
    result = detect(image, SpotFindingPlan(base, (ChannelOverride("b", override),)))
    assert result.diagnostics["geometry"]["scale_space_bytes_estimate"] == 10.4 * 20 * 16 * 64 * 64
    assert result.diagnostics["thresholds"] == (0.01, 0.02)
    assert result.diagnostics["effective_settings"]["b"]["num_sigma"] == 20
    single = detect(image[..., 1], replace(override, channel_labels=None))
    rows = result.spots[result.spots.channel == 1].drop(columns=["spot_id", "channel"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(rows, single.spots.drop(columns=["spot_id", "channel"]), check_exact=True)


# --- Config, registry, workflow and pipeline -----------------------------------------------------

def test_the_registry_entry_follows_the_contract():
    spec = SPOT_FINDING_METHODS[StarfishLogConfig]
    assert (spec.name, spec.pipeline, spec.dimensions, spec.min_shape_zyx, spec.output_columns, spec.requires,
            spec.weights) == ("starfish_log", True, frozenset({2, 3}), (1, 1, 1),
                              ("z", "y", "x", "channel", "peak_intensity", "radius"), (), False)
    assert ISS.method == "starfish_log" and (ISS.overlap, ISS.exclude_border) == (0.5, False)


def test_the_four_starfish_settings_are_required():
    with pytest.raises(TypeError):
        StarfishLogConfig()
    with pytest.raises(TypeError):
        StarfishLogConfig(1, 10, 30)


@pytest.mark.parametrize("change", [
    dict(min_sigma=0), dict(min_sigma=-1.0), dict(min_sigma=(1.0, 1.0)), dict(min_sigma=[1.0, 1.0, 1.0]),
    dict(max_sigma=float("nan")), dict(max_sigma=(10.0, 0.5, 10.0)), dict(min_sigma=True),
    dict(num_sigma=0), dict(num_sigma=2.5), dict(num_sigma=True), dict(threshold=-0.01),
    dict(threshold=float("inf")), dict(overlap=1.5), dict(exclude_border=-1), dict(exclude_border=1.5),
    dict(channel_labels=("a", "a"))])
def test_invalid_settings_raise_at_construction(change):
    with pytest.raises(ValueError):
        replace(ISS, **change)


def test_the_workflow_block_selects_starfish_log():
    block = {"method": "starfish_log", "min_sigma": [2, 1, 1], "max_sigma": [6, 3, 3], "num_sigma": 10,
             "threshold": 0.01, "exclude_border": 2, "channel_overrides": {"ch02": {"threshold": 0.02}}}
    config = StarfishLogConfig((2, 1, 1), (6, 3, 3), 10, 0.01, exclude_border=2)
    assert detection(block) == SpotFindingPlan(config, (ChannelOverride("ch02", replace(config, threshold=0.02)),))
    with pytest.raises(ValueError, match="requires max_sigma, num_sigma"):
        detection({"method": "starfish_log", "min_sigma": 1, "threshold": 0.01})
    with pytest.raises(ValueError):
        detection({"method": "starfish_log", "min_sigma": 1, "max_sigma": 10, "num_sigma": 30, "threshold": 0.01,
                   "min_distance": 2})


def test_the_pipeline_detects_and_checkpoints_with_starfish_log(tmp_path):
    config = StarfishLogConfig(1, 10, 10, 0.01)
    dataset = golden_dataset(tmp_path)
    checkpoints = CheckpointConfig(stages=("candidates",), directory=tmp_path / "checkpoints")
    fov = fov_with_fixture(dataset, "3d").run(PipelineConfig(spot_finding=config), checkpoints=checkpoints)
    direct = detect(fixture_image("3d"), replace(config, channel_labels=CHANNELS))
    assert len(direct.spots) > 0
    pd.testing.assert_frame_equal(fov.spot_result.spots, direct.spots, check_exact=True)
    reloaded = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    pd.testing.assert_frame_equal(reloaded.spot_result.spots, fov.spot_result.spots, check_exact=True)
    assert reloaded.spot_result.config == fov.spot_result.config == replace(config, channel_labels=CHANNELS)
