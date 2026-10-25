"""Golden baseline for the current local-maxima spot finding (W-267, §2.7 decision D7).

Pins, as explicit expected values, on one small seeded fixture (12×48×48 voxels, four
uint16 channels, and its plane z=6 as the Z=1 variant):

(a) the ``find_spots`` table and threshold diagnostics (the per-channel thresholds) for
    the four threshold modes, with ``exclude_border`` true and false, in 3D and Z=1;
(b) the ``candidates`` table after ``FOV.run``, written as a CSV checkpoint and reloaded,
    and the written ``candidates.csv`` read back with the dtypes of its header;
(c) the same results when the pipeline is translated from the legacy ``spot_finding``
    YAML keys (``intensity_estimation``, ``intensity_threshold``, ``min_distance`` or
    ``min_distance_voxels``), which cannot express ``exclude_border``.

The expected tables are the files under ``data/expected/test_spot_finding_golden/`` (its
README names the revision and the command that wrote them) and the thresholds are
``EXPECTED_THRESHOLDS``; W-573 replaced the SHA-256 digests pinned before, which these
values reproduce exactly. Every column of a spot table holds integers (voxel coordinates,
the channel and a uint16 peak) and compares exactly; the thresholds are computed float64
values and compare within ``FLOAT64`` (docs/contributing.md, "Expected values and
tolerances"). The fixture inputs stay pinned by digest (integer data).

The fixture holds a saturated multi-voxel amplicon whose plateau gives tied maxima
(W-218), a two-lobe amplicon whose lobes give split maxima, spots on and next to the
volume faces, a channel with more than half its voxels zero (noise-mode MAD 0), and a
dim channel whose maximum is below the image maximum (adaptive differs from
adaptive_round). The first digests were produced with the locked project environment
(NumPy 2.2.6, SciPy 1.17.0, scikit-image 0.26.0, pandas 3.0.0) and are bit-identical
over repeated single-thread runs; see docs/spot-finding-baseline.md.

Every detection configuration is built by ``detection_config``, which also holds the
only imports of detection configuration types and the only construction of
``PipelineConfig(spot_finding=...)``. The §2.7 registry move and the YAML ``method`` key
replace that helper's body only; every pinned value stays. Learned methods and LoG are
not part of this test.
"""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from starfinder.dataset import CheckpointConfig, Dataset, RoundState
from starfinder.dataset.workflow import from_workflow_config
from starfinder.image import ImageMetadata
from starfinder.spot_finding import find_spots

from .coordination_helpers import digest
from .expected_values import FLOAT64, assert_table_matches, read_csv, read_table, table_differences

pytestmark = [pytest.mark.spot_finding, pytest.mark.golden]

SHAPE_ZYX = (12, 48, 48)
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
SEED = 20261001
SATURATION = 3000  # channel 0 is clipped here, so its brightest amplicon is a plateau
Z1_PLANE = 6
NAMESPACE = "golden/sample/FOV_001"
# The pinned threshold_value of each mode: sigma units for noise, fractions otherwise.
MODE_VALUES = {"noise": 5.0, "adaptive": 0.2, "adaptive_round": 0.2, "global": 0.01}


def detection_config(mode, *, form="config", threshold_value=None, min_distance_voxels=1,
                     exclude_border=True, distance_key="min_distance"):
    """The only place that builds detection configuration.

    form "config" returns the typed detection config; "pipeline" a PipelineConfig
    whose only operation is detection; "yaml" the PipelineConfig that
    from_workflow_config translates from the legacy spot_finding keys (the rule
    rsf_single_fov, raw loading off). threshold_value=None uses MODE_VALUES[mode].
    distance_key names the legacy YAML key that carries min_distance_voxels. The
    legacy keys cannot express exclude_border, so form "yaml" requires the
    LocalMaximaConfig default (True).
    """
    from starfinder.dataset import PipelineConfig
    from starfinder.spot_finding import LocalMaximaConfig

    value = MODE_VALUES[mode] if threshold_value is None else threshold_value
    if form == "yaml":
        if exclude_border is not True:
            raise ValueError("the legacy spot_finding keys cannot express exclude_border")
        workflow = {"n_rounds": 1, "ref_round": "round1", "dataset_id": "golden",
                    "sample_id": "sample", "output_id": "out",
                    "root_input_path": "unused-input", "root_output_path": "unused-output",
                    "seq_channel_order": list(CHANNELS),
                    "rules": {"rsf_single_fov": {"parameters": {
                        "load_raw_images": {"run": False},
                        "spot_finding": {"run": True, "ref_round": "round1",
                                         "intensity_estimation": mode, "intensity_threshold": value,
                                         distance_key: min_distance_voxels}}}}}
        return from_workflow_config(workflow, "rsf_single_fov").pipeline
    config = LocalMaximaConfig(threshold_mode=mode, threshold_value=value,
                               min_distance_voxels=min_distance_voxels,
                               exclude_border=exclude_border)
    if form == "config":
        return config
    if form == "pipeline":
        return PipelineConfig(spot_finding=config)
    raise ValueError(f"unknown form {form!r}")


def _gaussians(centers, sigmas, amplitudes):
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in SHAPE_ZYX), indexing="ij")
    out = np.zeros(SHAPE_ZYX, dtype=np.float64)
    for (cz, cy, cx), (sz, sy, sx), amplitude in zip(centers, sigmas, amplitudes):
        out += amplitude * np.exp(-((z - cz) / sz) ** 2 / 2 - ((y - cy) / sy) ** 2 / 2
                                  - ((x - cx) / sx) ** 2 / 2)
    return out


def fixture_volume():
    """uint16 ZYXC golden fixture; every value comes from SEED.

    ch00: offset 100 and noise sd 8; eight random puncta; a saturated amplicon at
          (6, 24.5, 24) clipped to SATURATION (tied maxima); a spot on the z=0 face;
          a dim punctum of amplitude 40 at (8, 36, 12), between the noise thresholds
          for threshold_value 4 and 5.
    ch01: offset 150; eight random puncta; a two-lobe amplicon with lobes 2.4 voxels
          apart in X (split maxima); a spot on the y=0 face at z=6.
    ch02: noise of mean -2 and sd 5 clipped at 0, so about two thirds of the voxels
          are zero and the noise-mode MAD is 0; six random puncta.
    ch03: offset 120; eight dim random puncta; spots on the x=47 face and at x=1, z=6.
    """
    rng = np.random.RandomState(SEED)
    sigma = (1.0, 1.2, 1.2)

    def puncta(n, low, high):
        centers = np.column_stack([rng.uniform(2.0, SHAPE_ZYX[0] - 3.0, n),
                                   rng.uniform(4.0, SHAPE_ZYX[1] - 5.0, n),
                                   rng.uniform(4.0, SHAPE_ZYX[2] - 5.0, n)])
        return _gaussians(centers, [sigma] * n, rng.uniform(low, high, n))

    channels = []
    ch0 = 100.0 + rng.normal(0.0, 8.0, SHAPE_ZYX) + puncta(8, 300.0, 1500.0)
    ch0 += _gaussians([(6.0, 24.5, 24.0), (0.0, 10.0, 20.0), (8.0, 36.0, 12.0)], [sigma] * 3,
                      [5000.0, 1500.0, 40.0])
    channels.append(np.minimum(ch0, SATURATION))
    ch1 = 150.0 + rng.normal(0.0, 8.0, SHAPE_ZYX) + puncta(8, 300.0, 1500.0)
    ch1 += _gaussians([(5.0, 20.0, 30.0), (5.0, 20.0, 32.4), (6.0, 0.0, 15.0)],
                      [(1.0, 0.9, 0.9), (1.0, 0.9, 0.9), sigma], [1200.0, 1000.0, 1500.0])
    channels.append(ch1)
    ch2 = np.maximum(rng.normal(-2.0, 5.0, SHAPE_ZYX), 0.0) + puncta(6, 300.0, 1500.0)
    channels.append(ch2)
    ch3 = 120.0 + rng.normal(0.0, 8.0, SHAPE_ZYX) + puncta(8, 150.0, 500.0)
    ch3 += _gaussians([(6.0, 30.0, 47.0), (6.0, 12.0, 1.0)], [sigma, sigma], [500.0, 500.0])
    channels.append(ch3)
    return np.clip(np.rint(np.stack(channels, axis=-1)), 0, 65535).astype(np.uint16)


def fixture_image(dims):
    volume = fixture_volume()
    return volume if dims == "3d" else volume[Z1_PLANE:Z1_PLANE + 1].copy()


def detect(mode, dims, **options):
    return find_spots(fixture_image(dims), config=detection_config(mode, **options),
                      metadata=ImageMetadata("golden/round1"), spot_namespace=NAMESPACE)


def fov_with_fixture(dataset, dims):
    fov = dataset.fov("FOV_001")
    fov.images["round1"] = fixture_image(dims)
    fov.metadata["round1"] = ImageMetadata("FOV_001/round1")
    return fov


def golden_dataset(root):
    return Dataset(root, root / "out", "golden", "sample", "out",
                   rounds=RoundState(sequencing_rounds=["round1"], reference_round="round1"),
                   channel_order=list(CHANNELS))


def run_and_reload(root, dataset, dims, pipeline):
    """FOV.run with a CSV candidates checkpoint; the reloaded spot table and the written candidates.csv read
    back with the dtypes of its header."""
    checkpoints = CheckpointConfig(stages=("candidates",), directory=root / "checkpoints")
    fov = fov_with_fixture(dataset, dims)
    fov.run(pipeline, checkpoints=checkpoints)
    reloaded = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    directory = root / "checkpoints" / "FOV_001" / "checkpoints"
    assert reloaded.spot_result.config == fov.spot_result.config
    thresholds = fov.spot_result.diagnostics["thresholds"]
    assert reloaded.spot_result.diagnostics["thresholds"] == thresholds
    assert reloaded.intensity_result is None
    pd.testing.assert_frame_equal(reloaded.spot_result.spots, fov.spot_result.spots)
    dtypes = json.loads((directory / "candidates.json").read_text())["dtypes"]
    return {"table": reloaded.spot_result.spots, "csv": read_csv(directory / "candidates.csv", dtypes)}


CASES = [(mode, border, dims)
         for dims in ("3d", "z1") for mode in MODE_VALUES for border in (True, False)]


def case_id(case):
    mode, border, dims = case
    return f"{mode}-{'border' if border else 'noborder'}-{dims}"


PINNED_INPUT = {
    "3d": "9439d1b5f102adf9d5d00c8750f5adee819cf68599fed38b6788af2eac4c5af5",
    "z1": "88cb41775c5d8aa6b528f52076d2b716423bb819f8dc93a6e0452ef957978cd1",
}
# The explicit expected values (data/expected/test_spot_finding_golden/README.md): one find_spots table per
# case, <case_id>.csv, which is also the reloaded candidates table of FOV.run.
EXPECTED = Path(__file__).parent / "data" / "expected" / "test_spot_finding_golden"
# (mode, dims) -> the per-channel thresholds of find_spots, the same with exclude_border true and false
EXPECTED_THRESHOLDS = {
    ("noise", "3d"): (145.478, 195.478, 0.0, 165.478),
    ("adaptive", "3d"): (600.0, 332.20000000000005, 195.0, 124.60000000000001),
    ("adaptive_round", "3d"): (600.0, 600.0, 600.0, 600.0),
    ("global", "3d"): (655.35, 655.35, 655.35, 655.35),
    ("noise", "z1"): (145.478, 196.478, 0.0, 165.478),
    ("adaptive", "z1"): (600.0, 332.20000000000005, 160.60000000000002, 124.60000000000001),
    ("adaptive_round", "z1"): (600.0, 600.0, 600.0, 600.0),
    ("global", "z1"): (655.35, 655.35, 655.35, 655.35),
}
# The spot_namespace column that FOV.run writes first in candidates.csv, before the spot table's columns.
RUN_NAMESPACE = '["golden","sample","FOV_001",null]'


def expected_spots(case):
    return read_table(EXPECTED / f"{case_id(case)}.csv")


def expected_candidates_csv(case):
    table = expected_spots(case)
    table.insert(0, "spot_namespace", pd.array([RUN_NAMESPACE] * len(table), dtype="string"))
    return table


def test_fixture_inputs_are_pinned():
    assert {dims: digest(fixture_image(dims)) for dims in ("3d", "z1")} == PINNED_INPUT


def test_fixture_has_the_d7_features():
    volume = fixture_volume()
    assert (volume[..., 2] == 0).mean() > 0.5
    assert volume[..., 3].max() < volume.max()
    assert (volume[..., 0] == SATURATION).sum() > 1


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_find_spots_tables_and_diagnostics_are_pinned(case):
    mode, border, dims = case
    result = detect(mode, dims, exclude_border=border)
    assert_table_matches(result.spots, expected_spots(case), name=case_id(case))
    np.testing.assert_allclose(result.diagnostics["thresholds"], EXPECTED_THRESHOLDS[(mode, dims)], **FLOAT64)


@pytest.mark.parametrize("dims", ["3d", "z1"])
def test_noise_mode_mad_zero_keeps_threshold_and_warns(dims):
    # The §2.7 diagnostics change (W-270): channel 2 has MAD 0, so its noise threshold still
    # equals its median, 0, and every positive local maximum is kept; one SpotFindingWarning
    # now reports the channel.
    from starfinder.spot_finding import SpotFindingWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = detect("noise", dims)
    assert result.diagnostics["thresholds"][2] == 0.0
    assert [w.category for w in caught] == [SpotFindingWarning]


def test_saturated_plateau_gives_tied_maxima():
    # W-218 legacy behavior: every voxel of a plateau that is a local maximum is a candidate.
    spots = detect("noise", "3d").spots
    tied = spots[(spots.channel == 0) & (spots.peak_intensity == SATURATION)]
    assert len(tied) > 1


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_candidates_after_run_save_and_reload_are_pinned(tmp_path, case):
    mode, border, dims = case
    outputs = run_and_reload(tmp_path, golden_dataset(tmp_path), dims,
                             detection_config(mode, form="pipeline", exclude_border=border))
    assert_table_matches(outputs["table"], expected_spots(case), name="reloaded")
    assert_table_matches(outputs["csv"], expected_candidates_csv(case), name="candidates.csv")


@pytest.mark.parametrize("distance_key", ["min_distance", "min_distance_voxels"])
@pytest.mark.parametrize("case", [c for c in CASES if c[1]], ids=case_id)
def test_legacy_yaml_keys_give_the_same_digests(tmp_path, case, distance_key):
    mode, border, dims = case
    outputs = run_and_reload(tmp_path, golden_dataset(tmp_path), dims,
                             detection_config(mode, form="yaml", distance_key=distance_key))
    assert_table_matches(outputs["table"], expected_spots(case), name="reloaded")
    assert_table_matches(outputs["csv"], expected_candidates_csv(case), name="candidates.csv")


@pytest.mark.parametrize("dims", ["3d", "z1"])
@pytest.mark.parametrize("change", [{"threshold_value": 4.0}, {"min_distance_voxels": 2},
                                    {"exclude_border": False}], ids=lambda c: next(iter(c)))
def test_a_changed_setting_changes_the_table_digest(change, dims):
    # The changed table lies outside the comparison of the expected table (every column exact).
    changed = detect("noise", dims, **change)
    assert table_differences(changed.spots, expected_spots(("noise", True, dims)))
