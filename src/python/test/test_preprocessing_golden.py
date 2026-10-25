"""Golden baseline for the existing preprocessing operations (W-226).

Pins, with exact SHA-256 digests, the outputs of each existing operation and of
the legacy min-max -> histogram matching -> reconstruction sequence through
``FOV.run`` on one small seeded fixture. Any behavior change, including a
changed radius, rounding mode or dtype, changes a digest. The digests were
produced with the locked project environment; a dependency upgrade that changes
scikit-image output must be reviewed, not re-pinned silently.

The per-channel statistics (zero fraction, median, MAD and the local-maxima
``noise`` threshold) are the values reported on the preprocessing baseline page.
Since W-574 they are explicit expected values in full precision
(``data/expected/test_preprocessing_golden/stats.csv``), compared with the
float64 tolerance of docs/contributing.md; before, they were rounded to six
decimals and compared with ``==``.
"""
from pathlib import Path

import numpy as np
import pytest

from starfinder.dataset import Dataset, PipelineConfig, RoundState
from starfinder.image import ImageMetadata
from starfinder.preprocessing import (
    HistogramMatchingConfig,
    MinMaxNormalizationConfig,
    ProjectionConfig,
    ReconstructionConfig,
    TophatConfig,
    filter_tophat,
    match_histogram,
    normalize_min_max,
    project_image,
    correct_background,
)

from .coordination_helpers import digest
from .expected_values import FLOAT64, read_table

pytestmark = [pytest.mark.preprocessing, pytest.mark.golden]

EXPECTED = Path(__file__).resolve().parent / "data" / "expected" / "test_preprocessing_golden"
SHAPE_ZYX = (4, 32, 32)
CHANNELS = 4
SEED = 20260927
NOISE_THRESHOLD = 5.0  # LocalMaximaConfig default threshold_value
MINMAX = MinMaxNormalizationConfig("uint8", (0, 255))


def _round_volume(rng, gains):
    """uint16 ZYXC: per-channel offsets and gradient, Gaussian noise, sparse puncta."""
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in SHAPE_ZYX), indexing="ij")
    volume = np.empty(SHAPE_ZYX + (CHANNELS,), dtype=np.float64)
    for c, offset in enumerate((200.0, 400.0, 800.0, 1600.0)):
        volume[..., c] = offset + 3.0 * x + rng.normal(0.0, 20.0, SHAPE_ZYX)
    for _ in range(12):
        c = rng.randint(CHANNELS)
        cz, cy, cx = rng.uniform(0.5, SHAPE_ZYX[0] - 1.5), rng.uniform(3, 29), rng.uniform(3, 29)
        amplitude = rng.uniform(1000.0, 4000.0)
        volume[..., c] += amplitude * np.exp(
            -((z - cz) ** 2) / (2 * 0.7**2) - ((y - cy) ** 2 + (x - cx) ** 2) / (2 * 1.0**2))
    volume *= np.asarray(gains, dtype=np.float64)
    return np.clip(np.rint(volume), 0, 65535).astype(np.uint16)


def fixture_rounds():
    """Two rounds: round1 is the reference; round2 has unequal channel gains."""
    rng = np.random.RandomState(SEED)
    return {"round1": _round_volume(rng, (1.0, 1.0, 1.0, 1.0)),
            "round2": _round_volume(rng, (0.7, 1.3, 1.0, 1.6))}


def channel_stats(volume):
    """Per-channel zero fraction, median, MAD and local-maxima noise threshold.

    Matches ``spot_finding`` ``noise`` mode: all voxels, float64,
    median + threshold_value * 1.4826 * MAD.
    """
    rows = []
    for c in range(volume.shape[-1]):
        values = volume[..., c].astype(np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        rows.append((float((values == 0).mean()), median, mad, median + NOISE_THRESHOLD * 1.4826 * mad))
    return rows


def operation_outputs(rounds):
    """Each existing operation applied to the moving round (round2)."""
    raw = rounds["round2"]
    minmax = normalize_min_max(raw, config=MINMAX)
    reference = normalize_min_max(rounds["round1"], config=MINMAX)[..., 0].copy()
    matched = match_histogram(minmax, reference, config=HistogramMatchingConfig())
    return {
        "raw": raw,
        "minmax": minmax,
        "histogram_after_minmax": matched,
        "reconstruction_after_histogram": correct_background(matched, config=ReconstructionConfig()),
        "tophat_after_minmax": filter_tophat(minmax, config=TophatConfig()),
        "reconstruction_raw_uint16": correct_background(raw, config=ReconstructionConfig()),
        "tophat_raw_uint16": filter_tophat(raw, config=TophatConfig()),
        "projection_max_raw": project_image(raw, config=ProjectionConfig("max")),
        "projection_sum_raw": project_image(raw, config=ProjectionConfig("sum")),
    }


def run_legacy_sequence(tmp_path, rounds, *, radius=3, rounding="truncate"):
    """FOV.run with the legacy min-max -> histogram -> reconstruction order."""
    dataset = Dataset(tmp_path, tmp_path / "out", "golden", "sample", "out",
                      rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
                      channel_order=["ch00", "ch01", "ch02", "ch03"])
    fov = dataset.fov("FOV_001")
    for name, volume in rounds.items():
        fov.images[name] = volume.copy()
        fov.metadata[name] = ImageMetadata(f"FOV_001/{name}")
    from starfinder.preprocessing import PreprocessingRecipe, PreprocessingStep
    config = PipelineConfig(preprocessing=PreprocessingRecipe((
        PreprocessingStep(MinMaxNormalizationConfig("uint8", (0, 255), rounding=rounding)),
        PreprocessingStep(HistogramMatchingConfig(reference_channel=0)),
        PreprocessingStep(ReconstructionConfig(radius_yx=radius)))))
    fov.run(config)
    return fov.images


PINNED_OPERATIONS = {
    "raw": "178a1db71e2bfbb9b6d2eaf01d9e4f91a901751cbe79f7fcd17cb96c5caf4a95",
    "minmax": "fa79b78487945fab8f5a4b5544e0c33ef23b102038b02110d49a3a88426dce49",
    "histogram_after_minmax": "152f626a4b2c5802121368744caecc5fd4357969294358d986dce9c89d205dcf",
    "reconstruction_after_histogram": "df2ac1debb364bc207129b8cb2f35ea246f9f0d1ad29b9bb6949b596bc6d7ac1",
    "tophat_after_minmax": "49a826ca826bb86f5ee5034de74824063cc97ba62a8e31b8ef88eec6f61c785e",
    "reconstruction_raw_uint16": "97d45dc576c8e30b3dca471fa85045b84154a0b1f603b1367873e5bd7248b090",
    "tophat_raw_uint16": "d656d11e1a5270d30ae4d761cdf21bbe5b72cca6ae2f90dd63f7249b566cb6ea",
    "projection_max_raw": "5d71d4b7aa8c3ac9ed22bb9556a2f4c95d7d6980e619225a4907b7e066c11e80",
    "projection_sum_raw": "73707d7d6b2658f15fd36efde3781a8c4727508a2a0a97d71235dc2424a04d38",
}
PINNED_SEQUENCE = {
    "round1": "16d1ea3dd8f343f4bd6307ec25582eae51c2fd0b3c2615950ad765d7b3b47491",
    "round2": "df2ac1debb364bc207129b8cb2f35ea246f9f0d1ad29b9bb6949b596bc6d7ac1",
}
#: The operations whose per-channel statistics the baseline page reports (expected_stats).
STATS_OPERATIONS = ("raw", "minmax", "histogram_after_minmax", "reconstruction_after_histogram",
                    "tophat_after_minmax", "reconstruction_raw_uint16", "tophat_raw_uint16")


def expected_stats(name):
    """(zero fraction, median, MAD, noise threshold) per channel of one operation, from stats.csv."""
    table = read_table(EXPECTED / "stats.csv")
    rows = table[table.operation == name]
    assert rows.channel.tolist() == list(range(CHANNELS))
    return rows[["zero_fraction", "median", "mad", "noise_threshold"]].to_numpy(np.float64)


def test_fixture_is_bounded():
    rounds = fixture_rounds()
    assert all(v.shape == SHAPE_ZYX + (CHANNELS,) and v.dtype == np.uint16 for v in rounds.values())


@pytest.mark.parametrize("name", sorted(PINNED_OPERATIONS))
def test_existing_operation_digest(name):
    assert digest(operation_outputs(fixture_rounds())[name]) == PINNED_OPERATIONS[name]


def test_legacy_sequence_through_fov_run(tmp_path):
    images = run_legacy_sequence(tmp_path, fixture_rounds())
    assert {name: digest(images[name]) for name in images} == PINNED_SEQUENCE


def test_sequence_matches_composed_operations(tmp_path):
    """FOV.run output for the moving round equals the composed public functions."""
    images = run_legacy_sequence(tmp_path, fixture_rounds())
    composed = operation_outputs(fixture_rounds())["reconstruction_after_histogram"]
    assert np.array_equal(images["round2"], composed)


@pytest.mark.parametrize("change", [{"radius": 4}, {"rounding": "nearest_even"}])
def test_golden_detects_changed_settings(tmp_path, change):
    """Negative control: a changed radius or rounding mode alters the pinned output."""
    images = run_legacy_sequence(tmp_path, fixture_rounds(), **change)
    assert digest(images["round2"]) != PINNED_SEQUENCE["round2"]


@pytest.mark.parametrize("name", sorted(STATS_OPERATIONS))
def test_baseline_page_statistics(name):
    stats = np.array(channel_stats(operation_outputs(fixture_rounds())[name]), dtype=np.float64)
    np.testing.assert_allclose(stats, expected_stats(name), **FLOAT64)
