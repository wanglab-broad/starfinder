"""The W-218 option of local maxima: the opt-in within-channel merge merge_radius_zyx and the S16 check on the
§2.12 formed scene (W-271; docs/spot-finding-algorithms.md, "Local maxima" and check S16).

Every bound here is provisional: merge_radius_zyx is a new option whose only evidence is the W-267
post-hoc merge check and border probe on the `small` scene (2 same-channel ties removed and no
matched candidate lost at radius (2, 2, 2); border misses 5 -> 0), and formed16 is smaller and
denser than `small`, without a W-266 reference. The default (None) is pinned by
test_spot_finding_golden.py.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from starfinder.evaluation.spot_finding import classify_detections, evaluate_spots
from starfinder.image import ImageMetadata
from starfinder.preprocessing import MinMaxNormalizationConfig, normalize_intensity
from starfinder.spot_finding import LocalMaximaConfig, find_spots

from .spot_finding_scenes import FORMED16_SHAPE, SEEDS, formed16
from .test_spot_finding_workflow_key import detection

META = ImageMetadata("local-maxima")
NAMESPACE = "local-maxima/test"
# The hand-built fixture is thresholded at 0.3 of each channel's maximum, above its background.
CONFIG = LocalMaximaConfig(threshold_mode="adaptive", threshold_value=0.3)
RADIUS = (2.0, 2.0, 2.0)
# The W-218 notebook matching (W-267 re-measurement) and the recipe-1 uint8 normalization.
S16_MATCH = dict(policy="greedy", threshold=5.0, boundary="exclusive", units="voxel")
RECIPE1 = MinMaxNormalizationConfig("uint8", (0, 255), rounding="truncate")


def merge_fixture():
    """8x32x32 x 2 channels (uint16): channel 0 holds a 2-voxel plateau, a 3-voxel diagonal tie, a two-lobe
    amplicon (lobes 2.4 voxels apart, the brighter one at the larger X) and a pair of spots 5 voxels apart;
    channel 1 repeats the plateau and the pair at the same positions. The background 100 + (z + y + x) % 3
    has a nonzero MAD and stays below the threshold."""
    z, y, x = np.indices((8, 32, 32))
    background = 100.0 + (z + y + x) % 3
    ch0 = background.copy()
    ch0[2, 5, 5] = ch0[2, 5, 6] = 1000
    ch0[4, 10, 20] = ch0[4, 11, 21] = ch0[5, 10, 21] = 950
    for centre_x, amplitude in ((20.0, 700.0), (22.4, 900.0)):
        ch0 += amplitude * np.exp(-0.5 * ((z - 3) ** 2 + (y - 20) ** 2 + (x - centre_x) ** 2) / 0.7 ** 2)
    ch0[5, 26, 8] = ch0[5, 26, 13] = 800
    ch1 = background.copy()
    ch1[2, 5, 5] = ch1[2, 5, 6] = 1000
    ch1[5, 26, 8] = ch1[5, 26, 13] = 800
    return np.rint(np.stack([ch0, ch1], axis=-1)).astype(np.uint16)


def detect(image, config):
    return find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE)


def positions(result):
    return [(int(r.z), int(r.y), int(r.x), int(r.channel)) for r in result.spots.itertuples()]


# --- Merge semantics on the hand-built fixture --------------------------------------------------------

PLATEAU = [(2, 5, 5), (2, 5, 6)]
TIE = [(4, 10, 20), (4, 11, 21), (5, 10, 21)]
LOBES = [(3, 20, 20), (3, 20, 22)]
PAIR = [(5, 26, 8), (5, 26, 13)]


def test_the_fixture_has_every_tied_and_split_maximum_without_the_merge():
    result = detect(merge_fixture(), CONFIG)
    assert sorted(positions(result)) == sorted([(*p, 0) for p in PLATEAU + TIE + LOBES + PAIR]
                                               + [(*p, 1) for p in PLATEAU + PAIR])
    assert "merged" not in result.diagnostics and result.diagnostics["warnings"] == ()
    peak = {p[:3]: v for p, v in zip(positions(result), result.spots.peak_intensity) if p[3] == 0}
    assert peak[(3, 20, 22)] > peak[(3, 20, 20)]


def test_the_merge_keeps_one_maximum_per_plateau_tie_and_amplicon_and_both_members_of_the_pair():
    image = merge_fixture()
    legacy = detect(image, CONFIG)
    merged = detect(image, replace(CONFIG, merge_radius_zyx=RADIUS))
    # The brightest maximum of each group, ties broken by z, then y, then x; the pair is 5 voxels apart.
    kept = {(2, 5, 5, 0), (4, 10, 20, 0), (3, 20, 22, 0), (5, 26, 8, 0), (5, 26, 13, 0),
            (2, 5, 5, 1), (5, 26, 8, 1), (5, 26, 13, 1)}
    assert set(positions(merged)) == kept
    assert merged.diagnostics["merged"] == {"0": 4, "1": 1}
    # Kept maxima keep their rows (order, coordinates, peak_intensity); identities follow the merge.
    rows = [i for i, p in enumerate(positions(legacy)) if p in kept]
    expected = legacy.spots.drop(columns="spot_id").iloc[rows].reset_index(drop=True)
    pd.testing.assert_frame_equal(merged.spots.drop(columns="spot_id"), expected, check_exact=True)
    assert list(merged.spots.spot_id) == [str(i) for i in range(len(kept))]
    # Nothing is merged across channels: both channels keep their row at the plateau and the pair.
    assert {p[:3] for p in positions(merged) if p[3] == 1} <= {p[:3] for p in positions(merged) if p[3] == 0}


def test_the_merge_without_peak_intensity_keeps_the_same_maxima():
    image = merge_fixture()
    with_peak = detect(image, replace(CONFIG, merge_radius_zyx=RADIUS))
    without = detect(image, replace(CONFIG, merge_radius_zyx=RADIUS, measure_peak_intensity=False))
    pd.testing.assert_frame_equal(without.spots, with_peak.spots.drop(columns="peak_intensity"), check_exact=True)
    assert without.diagnostics["merged"] == with_peak.diagnostics["merged"]


def test_the_ellipsoid_uses_per_axis_radii_and_includes_its_surface():
    image = merge_fixture()
    # Z radius 0.5: the tie member one plane down, (5, 10, 21), lies outside the ellipsoid of (4, 10, 20),
    # while (4, 11, 21), one voxel away in Y and X, lies inside it.
    flat = detect(image, replace(CONFIG, merge_radius_zyx=(0.5, 2.0, 2.0)))
    assert {p[:3] for p in positions(flat) if p[3] == 0} == {
        (2, 5, 5), (4, 10, 20), (5, 10, 21), (3, 20, 22), (5, 26, 8), (5, 26, 13)}
    # An X radius of exactly 5 voxels reaches the pair (the ellipsoid includes its surface); the tie of
    # equal values is broken by X.
    wide = detect(image, replace(CONFIG, merge_radius_zyx=(2.0, 2.0, 5.0)))
    assert (5, 26, 13, 0) not in positions(wide) and (5, 26, 8, 0) in positions(wide)


def test_the_merge_on_a_plane_ignores_the_z_radius():
    plane = merge_fixture()[2:3]   # Z=1: at half the maximum, only the plateau of both channels
    for radius in ((0.1, 2.0, 2.0), (50.0, 2.0, 2.0)):
        result = detect(plane, replace(CONFIG, threshold_value=0.5, merge_radius_zyx=radius))
        assert positions(result) == [(0, 5, 5, 0), (0, 5, 5, 1)]
        assert result.diagnostics["merged"] == {"0": 1, "1": 1}


def test_none_is_the_legacy_result():
    image = merge_fixture()
    assert LocalMaximaConfig().merge_radius_zyx is None
    default, explicit = detect(image, CONFIG), detect(image, replace(CONFIG, merge_radius_zyx=None))
    pd.testing.assert_frame_equal(explicit.spots, default.spots, check_exact=True)
    assert "merged" not in explicit.diagnostics


@pytest.mark.parametrize("radius", [(2.0, 2.0), [2.0, 2.0, 2.0], (0.0, 2.0, 2.0), (2.0, -1.0, 2.0),
                                    (2.0, float("inf"), 2.0), (2.0, True, 2.0), 2.0])
def test_invalid_merge_radii_raise(radius):
    with pytest.raises(ValueError, match="merge_radius_zyx"):
        LocalMaximaConfig(merge_radius_zyx=radius)


def test_the_workflow_block_sets_the_merge_radius_and_exclude_border():
    assert detection({"merge_radius_zyx": [2, 2, 2], "exclude_border": False}) == LocalMaximaConfig(
        merge_radius_zyx=(2, 2, 2), exclude_border=False)


# --- S16: the W-218 resolution on the §2.12 formed scene -------------------------------------------

def s16(image, metadata, truth, eligible, config):
    """Detections, matching and the W-218 classification by channel of one local-maxima run."""
    result = find_spots(image, config=config, metadata=metadata, spot_namespace="s16")
    detected = result.spots[["z", "y", "x"]].to_numpy()
    match = evaluate_spots(detected, truth, reference_metadata=metadata, observed_metadata=metadata,
                           eligible_reference=eligible, **S16_MATCH)
    counts = classify_detections(match, detected, truth, radius=5.0, groups=result.spots.channel.to_numpy()).counts
    matched = {i for i, _, _ in match.details["matched_pairs"]}
    return result, counts, matched


@pytest.mark.parametrize("seed", SEEDS)
def test_s16_the_merge_removes_same_channel_duplicates_without_losing_a_match(seed):
    image, metadata, truth, eligible = formed16(seed)
    image = normalize_intensity(image, config=RECIPE1)
    _, legacy, _ = s16(image, metadata, truth, eligible, LocalMaximaConfig())
    _, merged, _ = s16(image, metadata, truth, eligible, LocalMaximaConfig(merge_radius_zyx=RADIUS))
    assert merged["duplicate_same_group"] == 0
    assert merged["matched"] >= legacy["matched"]
    # Cross-channel duplicates (crosstalk copies, §2.8) are reported in the worker notes, not gated.


@pytest.mark.parametrize("variant", ["uint16", "uint8"])
@pytest.mark.parametrize("seed", SEEDS)
def test_s16_no_eligible_amplicon_near_a_face_is_missed_without_border_exclusion(seed, variant):
    image, metadata, truth, eligible = formed16(seed)
    if variant == "uint8":
        image = normalize_intensity(image, config=RECIPE1)
    shape = np.asarray(FORMED16_SHAPE)
    near_face = np.flatnonzero(eligible & (np.minimum(truth, shape - 1 - truth).min(axis=1) <= 1))
    _, _, matched = s16(image, metadata, truth, eligible, LocalMaximaConfig(exclude_border=False))
    assert len(near_face) and set(near_face) <= matched
