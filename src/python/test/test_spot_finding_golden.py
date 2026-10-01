"""Golden baseline for the current local-maxima spot finding (W-267, §2.7 decision D7).

Pins, with exact SHA-256 digests, on one small seeded fixture (12×48×48 voxels, four
uint16 channels, and its plane z=6 as the Z=1 variant):

(a) the ``find_spots`` table and threshold diagnostics (the per-channel thresholds) for
    the four threshold modes, with ``exclude_border`` true and false, in 3D and Z=1;
(b) the ``candidates`` table after ``FOV.run``, written as a CSV checkpoint and reloaded;
(c) the same digests when the pipeline is translated from the legacy ``spot_finding``
    YAML keys (``intensity_estimation``, ``intensity_threshold``, ``min_distance`` or
    ``min_distance_voxels``), which cannot express ``exclude_border``.

The fixture holds a saturated multi-voxel amplicon whose plateau gives tied maxima
(W-218), a two-lobe amplicon whose lobes give split maxima, spots on and next to the
volume faces, a channel with more than half its voxels zero (noise-mode MAD 0), and a
dim channel whose maximum is below the image maximum (adaptive differs from
adaptive_round). The digests were produced with the locked project environment
(NumPy 2.2.6, SciPy 1.17.0, scikit-image 0.26.0, pandas 3.0.0) and are bit-identical
over repeated single-thread runs; see docs/spot-finding-baseline.md.

Every detection configuration is built by ``detection_config``, which also holds the
only imports of detection configuration types and the only construction of
``PipelineConfig(detection=...)``. The §2.7 registry move and the YAML ``method`` key
replace that helper's body only; every pinned digest stays. Learned methods and LoG are
not part of this test.
"""
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from starfinder.dataset import CheckpointConfig, Dataset, RoundState
from starfinder.dataset.workflow import from_workflow_config
from starfinder.image import ImageMetadata
from starfinder.spot_finding import find_spots

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
        return PipelineConfig(detection=config)
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


def digest(array):
    """SHA-256 over dtype, shape and C-order bytes."""
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    h.update(array.tobytes())
    return h.hexdigest()


def table_digest(spots):
    """SHA-256 over the column names and dtypes and the CSV text (floats in %.17g)."""
    header = json.dumps([[str(c), str(t)] for c, t in spots.dtypes.items()])
    text = spots.to_csv(index=False, float_format="%.17g", na_rep="<NA>")
    return hashlib.sha256((header + "\n" + text).encode()).hexdigest()


def threshold_digest(diagnostics):
    """SHA-256 of the per-channel thresholds as JSON (floats by repr, which round-trips)."""
    return hashlib.sha256(json.dumps(list(diagnostics["thresholds"])).encode()).hexdigest()


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
    """FOV.run with a CSV candidates checkpoint; digests of the written and reloaded results."""
    checkpoints = CheckpointConfig(stages=("candidates",), directory=root / "checkpoints")
    fov = fov_with_fixture(dataset, dims)
    fov.run(pipeline, checkpoints=checkpoints)
    reloaded = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    directory = root / "checkpoints" / "FOV_001"
    assert reloaded.spot_result.config == fov.spot_result.config
    thresholds = fov.spot_result.diagnostics["thresholds"]
    assert reloaded.spot_result.diagnostics["thresholds"] == thresholds
    assert reloaded.intensity_result is None
    pd.testing.assert_frame_equal(reloaded.spot_result.spots, fov.spot_result.spots)
    return {"table": table_digest(reloaded.spot_result.spots),
            "csv": hashlib.sha256((directory / "candidates.csv").read_bytes()).hexdigest()}


CASES = [(mode, border, dims)
         for dims in ("3d", "z1") for mode in MODE_VALUES for border in (True, False)]


def case_id(case):
    mode, border, dims = case
    return f"{mode}-{'border' if border else 'noborder'}-{dims}"


PINNED_INPUT = {
    "3d": "9439d1b5f102adf9d5d00c8750f5adee819cf68599fed38b6788af2eac4c5af5",
    "z1": "88cb41775c5d8aa6b528f52076d2b716423bb819f8dc93a6e0452ef957978cd1",
}
# (mode, exclude_border, dims) -> (rows, table digest, threshold digest) of find_spots
PINNED_FIND_SPOTS = {
    ("noise", True, "3d"): (
        893, "05b64a92e096de0a449881e52d719bd3822375e326bb55495ae1086ca87d63cf",
        "9643d3a15077196245d8a8fdf7b8b56b42c33370620e6d7a9950b2381a17b564"),
    ("noise", False, "3d"): (
        1342, "a3160dcff7951ffb1c2b931816114a2e489d825371612acaa65c99ecefa37933",
        "9643d3a15077196245d8a8fdf7b8b56b42c33370620e6d7a9950b2381a17b564"),
    ("adaptive", True, "3d"): (
        828, "2132c5c4b598e50210aeee93ad0ab85a4fd150028bd16a673d7882da8c46327b",
        "9bdb7bc340c96fdfd22306a26289d2c1dc9f62f2a7e0eb9568cd882efb5f6f34"),
    ("adaptive", False, "3d"): (
        1197, "35e58decbc4d1d27436eae4ed94454f381b9ef8c0a69076db4def4568ba3c8d2",
        "9bdb7bc340c96fdfd22306a26289d2c1dc9f62f2a7e0eb9568cd882efb5f6f34"),
    ("adaptive_round", True, "3d"): (
        26, "5fa8fa4dad9a265ad81a577cc5d23606f237071a8f87339a6d29ddaa38e47b1f",
        "d0bfc8f7a5742e183c73cd9bb828a0d34e4d6314d207e1554c460165f1852878"),
    ("adaptive_round", False, "3d"): (
        29, "1ba8bf6ea5daa6730157188d9470d9fc0d91d5987645cb7eb997871571779680",
        "d0bfc8f7a5742e183c73cd9bb828a0d34e4d6314d207e1554c460165f1852878"),
    ("global", True, "3d"): (
        23, "534abef36f43cfbfaa04dd2f3506746e8818d6f13d3d3f51f89df4452cd0a733",
        "8b9fab049422f6cbb941a0c67215b60082ce9847fe59327b3ef10cf142ae12d8"),
    ("global", False, "3d"): (
        25, "22532ce6446789f9cd05234e2ecb28f284e13d785ab767e2384e43d539049850",
        "8b9fab049422f6cbb941a0c67215b60082ce9847fe59327b3ef10cf142ae12d8"),
    ("noise", True, "z1"): (
        241, "25f3d85f4e216ec8533cd9eb7ab6104a4035d7adfe420778ed80d3feb47bdec3",
        "443d10894def3e1e7155fa524788d7581a16a1c229b5f82e11f47898c24c8bb6"),
    ("noise", False, "z1"): (
        275, "a62092a122054732ce43b6f3439f49a4ee6b54ac02723c15dc2f8aeb5afee493",
        "443d10894def3e1e7155fa524788d7581a16a1c229b5f82e11f47898c24c8bb6"),
    ("adaptive", True, "z1"): (
        228, "9d19aa2bcc172671fec719a05e625e7af3b39ceb3ee8ba2baa3d9b61fdcc30fa",
        "28cb8707866828523636777402c383110e91dc3549e38bdcab3dc6bdc2a61779"),
    ("adaptive", False, "z1"): (
        254, "41bbe5295b03bcdcbf7d4f6412649c9758cf12703d16a5f8b0e2c8abc092c4c7",
        "28cb8707866828523636777402c383110e91dc3549e38bdcab3dc6bdc2a61779"),
    ("adaptive_round", True, "z1"): (
        13, "1669266806fcdf9494e3490d1fb13422a74b1b992fbb61127e9bb3c7acb36756",
        "d0bfc8f7a5742e183c73cd9bb828a0d34e4d6314d207e1554c460165f1852878"),
    ("adaptive_round", False, "z1"): (
        15, "150eef1f8fc1d20610ab0dab678f83aa180f0a1885697fb13da5dc794935a66a",
        "d0bfc8f7a5742e183c73cd9bb828a0d34e4d6314d207e1554c460165f1852878"),
    ("global", True, "z1"): (
        12, "31949024c2fb0461c8e36fafa77017bf904d551eacc3fcacc5118066eb661ee9",
        "8b9fab049422f6cbb941a0c67215b60082ce9847fe59327b3ef10cf142ae12d8"),
    ("global", False, "z1"): (
        13, "c23f6d23051dfae4c0bf14996c773bdd31917ab023cc88dbf4f47f54f5eabfce",
        "8b9fab049422f6cbb941a0c67215b60082ce9847fe59327b3ef10cf142ae12d8"),
}
# (mode, exclude_border, dims) -> sha256 of candidates.csv written by FOV.run
PINNED_CANDIDATES_CSV = {
    ("noise", True, "3d"):
        "0a583adbb7528a709a9cb3c1267daed4baf510321af2360a1fce69d0976ec861",
    ("noise", False, "3d"):
        "9f81d04f8861b8deef7f2cbdac41df9e4ac4271a577560629a99f6d271d6bf2f",
    ("adaptive", True, "3d"):
        "6d11cac63e06ef2b37c24e97761ab9303292a738f5d44cd5ccb8f8b082dcf2e9",
    ("adaptive", False, "3d"):
        "b6e6184d3dadcdbba259c2b6db04141028492112137595e5d8f0a28f52c5e3ec",
    ("adaptive_round", True, "3d"):
        "7f2f66b17a6441dbfa1b16f57da8467907c17b0d359dc73abd420cc535aac07e",
    ("adaptive_round", False, "3d"):
        "f5415904476040573a21e870a00c764d1f1c09a04f58a6ba8c95bf85ea5cb34e",
    ("global", True, "3d"):
        "5c249e0bbecc15cfc46e18ef0cca98cfaa592674ffce615a9d7b427e430834c4",
    ("global", False, "3d"):
        "be4f27a5c6ad5f5b4321bca2efd76d8f3728ec69dad71065149713446d3cbab1",
    ("noise", True, "z1"):
        "8408f75edec8efe59ece683d3131923cfd25effd4a6edf0339ea883fde9a3ce0",
    ("noise", False, "z1"):
        "b6d664c910df6627edb811a2035b3204e5e1ce27cbf160e78c45493ab8c63dc8",
    ("adaptive", True, "z1"):
        "9f8e4f5bac9f11636e00222fc169fd2f20a45ac690630afd38d7e0a6fc97ea0f",
    ("adaptive", False, "z1"):
        "e4d87b3793c94726fc79944e88b4a127982eefbf9367c310386bb7872bbc2040",
    ("adaptive_round", True, "z1"):
        "04cb11a303b4d288833b016674f05b2d3c6ace6bdc838ef63c186690ce502582",
    ("adaptive_round", False, "z1"):
        "920241874d712e0a4c3f45ef37d933d120d58094e21661058bb2b667918e20d9",
    ("global", True, "z1"):
        "14969a9b408778db125ba962bdc50b39d9ee4e03d00a56ef1ea3e9ecce1adf88",
    ("global", False, "z1"):
        "3b14b680ac7536f801b06a1b01d37d737c67e2901f3c0e32feffdfbd0dc536a4",
}


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
    assert (len(result.spots), table_digest(result.spots),
            threshold_digest(result.diagnostics)) == PINNED_FIND_SPOTS[case]


@pytest.mark.parametrize("dims", ["3d", "z1"])
def test_noise_mode_mad_zero_passes_silently(dims):
    # Legacy behavior that the §2.7 diagnostics change replaces with a warning: channel 2
    # has MAD 0, so its noise threshold equals its median, 0, and every positive local
    # maximum is kept without notice.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = detect("noise", dims)
    assert result.diagnostics["thresholds"][2] == 0.0
    assert caught == []


def test_saturated_plateau_gives_tied_maxima():
    # W-218 legacy behavior: every voxel of a plateau that is a local maximum is a candidate.
    spots = detect("noise", "3d").spots
    tied = spots[(spots.channel == 0) & (spots.peak_intensity == SATURATION)]
    assert len(tied) > 1


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_candidates_after_run_save_and_reload_are_pinned(tmp_path, case):
    mode, border, dims = case
    digests = run_and_reload(tmp_path, golden_dataset(tmp_path), dims,
                             detection_config(mode, form="pipeline", exclude_border=border))
    assert digests["table"] == PINNED_FIND_SPOTS[case][1]
    assert digests["csv"] == PINNED_CANDIDATES_CSV[case]


@pytest.mark.parametrize("distance_key", ["min_distance", "min_distance_voxels"])
@pytest.mark.parametrize("case", [c for c in CASES if c[1]], ids=case_id)
def test_legacy_yaml_keys_give_the_same_digests(tmp_path, case, distance_key):
    mode, border, dims = case
    digests = run_and_reload(tmp_path, golden_dataset(tmp_path), dims,
                             detection_config(mode, form="yaml", distance_key=distance_key))
    assert digests["table"] == PINNED_FIND_SPOTS[case][1]
    assert digests["csv"] == PINNED_CANDIDATES_CSV[case]


@pytest.mark.parametrize("dims", ["3d", "z1"])
@pytest.mark.parametrize("change", [{"threshold_value": 4.0}, {"min_distance_voxels": 2},
                                    {"exclude_border": False}], ids=lambda c: next(iter(c)))
def test_a_changed_setting_changes_the_table_digest(change, dims):
    changed = detect("noise", dims, **change)
    assert table_digest(changed.spots) != PINNED_FIND_SPOTS[("noise", True, dims)][1]
