"""§2.8 local background and noise next to the extracted sums (W-294; checks R9 and R10).

docs/readout-contract.md, "Extraction"; docs/readout-algorithms.md, "Background and noise".
R9 uses the hand-built bg_const and bg_neighbor fixtures (12×48×48, four channels, four
rounds) with analytic expectations. R10 (extended tier) reruns the W-278 held-out
measurement of the estimator local_ring at the truth positions of the calibrated scenes
(seeds 103 to 105), with the tolerances of W-278 estimators.csv (heldout, local_ring).
"""
import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import (Codebook, LocalBackgroundConfig, NeighborhoodSumConfig, WtaDecoderConfig,
                                decode_barcodes, extract_intensities, score_reads)
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult

from . import readout_scenes as scenes

pytestmark = [pytest.mark.barcode]

SHAPE = (12, 48, 48)
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
ROUNDS = ("round1", "round2", "round3", "round4")
METADATA = ImageMetadata("bg/FOV_001")
CONSTANT = 37
# The one spot of bg_const sits at the corner candidate, one voxel in channel ch0<k> of round k,
# so the corner read is assigned (color sequence 1234) and its ring holds constant voxels only.
SPOT_AMPLITUDE = 500
# Candidates: the centre, one on each face, and the corner (with the spot).
CANDIDATES = {"centre": (6, 24, 24), "z0": (0, 24, 24), "z11": (11, 24, 24), "y0": (6, 0, 24), "y47": (6, 47, 24),
              "x0": (6, 24, 0), "x47": (6, 24, 47), "corner": (0, 0, 0)}
# Ring voxels (outer box (1, 6, 6) minus inner box (1, 3, 3), clipped to the image): per z-plane
# inside the image, the outer lateral area minus the inner lateral area.
RING_VOXELS = {"centre": 3 * (13 * 13 - 7 * 7), "z0": 2 * (13 * 13 - 7 * 7), "z11": 2 * (13 * 13 - 7 * 7),
               "y0": 3 * (7 * 13 - 4 * 7), "y47": 3 * (7 * 13 - 4 * 7), "x0": 3 * (13 * 7 - 7 * 4),
               "x47": 3 * (13 * 7 - 7 * 4), "corner": 2 * (7 * 7 - 4 * 4)}
# Voxels of the extraction box (1, 2, 2), clipped the same way.
BOX_VOXELS = {"centre": 75, "z0": 50, "z11": 50, "y0": 45, "y47": 45, "x0": 45, "x47": 45, "corner": 18}
# bg_neighbor: candidate "ring" has a flat 13x13 square of amplitude NEIGHBOR over its three
# z-planes centred at lateral distance 5 (dy = +5), which covers 207 of its 360 ring voxels;
# candidate "inner" has a single voxel of amplitude 1000 at lateral distance 3 (dy = +3),
# inside the excluded inner box and outside the extraction box.
NEIGHBOR = 50
NEIGHBOR_CANDIDATES = {"ring": (6, 12, 12), "inner": (6, 36, 36)}


def loaded(image):
    return ImageLoadResult(image, METADATA, CHANNELS, (), {})


def spots(positions):
    points = np.asarray(list(positions.values()), dtype=float)
    frame = pd.DataFrame({"spot_id": pd.array(list(positions), dtype="string"),
                          "z": points[:, 0], "y": points[:, 1], "x": points[:, 2]})
    return SpotFindingResult(frame, METADATA, "bg", LocalMaximaConfig(), {"channel_labels": list(CHANNELS)})


def bg_const():
    rounds = {}
    for r, label in enumerate(ROUNDS):
        image = np.full(SHAPE + (len(CHANNELS),), CONSTANT, dtype=np.uint16)
        image[CANDIDATES["corner"] + (r,)] = SPOT_AMPLITUDE
        rounds[label] = image
    return rounds


def bg_neighbor():
    rounds = bg_const()
    (z, y, x), (zi, yi, xi) = NEIGHBOR_CANDIDATES["ring"], NEIGHBOR_CANDIDATES["inner"]
    for image in rounds.values():
        image[z - 1:z + 2, y + 5 - 6:y + 5 + 7, x - 6:x + 7, :] += NEIGHBOR
        image[zi, yi + 3, xi, :] += 1000
    return rounds


def extract(rounds, positions, config=NeighborhoodSumConfig()):
    return extract_intensities({k: loaded(v) for k, v in rounds.items()}, spots(positions), config=config)


def codebook():
    return Codebook(pd.DataFrame({"gene_id": ["G"], "color_sequence": ["1234"]}), ROUNDS, CHANNELS)


@pytest.mark.validation
def test_r9_constant_background_and_ring_counts():
    result = extract(bg_const(), CANDIDATES)
    index = {name: i for i, name in enumerate(CANDIDATES)}
    assert result.background.shape == result.noise.shape == (len(CANDIDATES), 4, 4)
    # Away from the spot, and at the spot (it lies inside the excluded inner box): exactly 37 and 0.
    assert (result.background == CONSTANT).all() and (result.noise == 0).all()
    for name, i in index.items():
        assert (result.background_voxels[i] == RING_VOXELS[name]).all(), name
        assert (result.box_voxels[i] == BOX_VOXELS[name]).all(), name
    assert RING_VOXELS["centre"] == 360 and RING_VOXELS["corner"] == 66
    # The image median is 37 and its MAD 0 (one spot voxel per round and channel at most).
    assert (result.image_background == CONSTANT).all() and (result.image_noise == 0).all()
    # The measurement changes neither the sums nor valid.
    plain = extract(bg_const(), CANDIDATES, NeighborhoodSumConfig(background=None))
    np.testing.assert_array_equal(result.values, plain.values)
    np.testing.assert_array_equal(result.valid, plain.valid)
    assert plain.background is None and plain.image_noise is None


@pytest.mark.validation
def test_r9_below_min_voxels_is_nan_and_the_score_background_unavailable():
    config = NeighborhoodSumConfig(background=LocalBackgroundConfig(min_voxels=RING_VOXELS["corner"] + 1))
    result = extract(bg_const(), CANDIDATES, config)
    corner = list(CANDIDATES).index("corner")
    assert np.isnan(result.background[corner]).all() and np.isnan(result.noise[corner]).all()
    assert (result.background_voxels[corner] == RING_VOXELS["corner"]).all()
    others = np.arange(len(CANDIDATES)) != corner
    assert (result.background[others] == CONSTANT).all()
    assert np.isfinite(result.values).all() and result.valid.all()
    reads = decode_barcodes(result, codebook(), config=WtaDecoderConfig())
    assert reads.table.call_status.iloc[corner] == "assigned"
    scored = score_reads(reads, result, reference=codebook()).table
    assert scored.qc_reason.iloc[corner] == "background_unavailable"
    assert scored.loc[corner, ["qc_score", "qc_ambiguity_max", "qc_signal_to_background", "qc_rounds"]].isna().all()
    # At min_voxels exactly, the corner is measured.
    exact = extract(bg_const(), CANDIDATES,
                    NeighborhoodSumConfig(background=LocalBackgroundConfig(min_voxels=RING_VOXELS["corner"])))
    assert (exact.background[corner] == CONSTANT).all()
    scored = score_reads(decode_barcodes(exact, codebook(), config=WtaDecoderConfig()), exact,
                         reference=codebook()).table
    assert scored.qc_reason.iloc[corner] == "" and np.isfinite(scored.qc_score.iloc[corner])


@pytest.mark.validation
def test_r9_a_ring_neighbor_raises_the_background_and_an_inner_neighbor_does_not():
    const = extract(bg_const(), NEIGHBOR_CANDIDATES)
    result = extract(bg_neighbor(), NEIGHBOR_CANDIDATES)
    ring, inner = 0, 1
    # 207 of the 360 ring voxels are raised by NEIGHBOR: the median is 37 + NEIGHBOR, and the
    # absolute deviations are 0 at 207 voxels, so the MAD is 0.
    assert (result.background[ring] == CONSTANT + NEIGHBOR).all() and (result.noise[ring] == 0).all()
    assert (const.background[ring] == CONSTANT).all()
    # The inner neighbor is excluded from the ring and lies outside the extraction box.
    assert (result.background[inner] == CONSTANT).all() and (result.noise[inner] == 0).all()
    np.testing.assert_array_equal(result.values[inner], const.values[inner])
    assert (result.background_voxels == 360).all()


@pytest.mark.contract
def test_configs_validate_the_ring_and_its_containment():
    assert NeighborhoodSumConfig().background == LocalBackgroundConfig((1, 3, 3), (1, 6, 6), 16)
    for kwargs in ({"inner_radius_zyx": (1, 3)}, {"outer_radius_zyx": (1, -6, 6)}, {"min_voxels": 0},
                   {"min_voxels": True}, {"inner_radius_zyx": (1, 3, 3), "outer_radius_zyx": (1, 3, 3)},
                   {"inner_radius_zyx": (1, 4, 4), "outer_radius_zyx": (1, 3, 6)}):
        with pytest.raises(ValueError):
            LocalBackgroundConfig(**kwargs)
    with pytest.raises(ValueError, match="must contain the extraction box"):
        NeighborhoodSumConfig((2, 2, 2))
    with pytest.raises(TypeError, match="background"):
        NeighborhoodSumConfig(background={"min_voxels": 16})
    wide = LocalBackgroundConfig((2, 3, 3), (2, 6, 6))
    assert NeighborhoodSumConfig((2, 2, 2), background=wide).background == wide
    assert NeighborhoodSumConfig((2, 2, 2), background=None).background is None


@pytest.mark.contract
def test_direct_mode_measures_the_own_round_only():
    positions = {"a": (6, 24, 24), "b": (6, 12, 36)}
    frame = spots(positions).spots.assign(round=pd.array(["round2", "round4"], dtype="string"),
                                          channel=np.array([1, 3]))
    found = SpotFindingResult(frame, METADATA, "bg", LocalMaximaConfig(), {"channel_labels": list(CHANNELS)})
    result = extract_intensities({k: loaded(v) for k, v in bg_const().items()}, found, readout_mode="direct")
    own = np.array([[False, True, False, False], [False, False, False, True]])
    background = result.background.transpose(0, 2, 1)  # (N, R, C)
    assert (background[own] == CONSTANT).all() and np.isnan(background[~own]).all()
    assert (result.background_voxels[own] == 360).all() and (result.background_voxels[~own] == 0).all()
    # The image statistics are per round and channel, for every round.
    assert (result.image_background == CONSTANT).all()


# --- R10: the estimator on the calibrated scenes (extended tier) ------------------------------

def _box_mean(image, centers, radius=scenes.BOX_RADIUS):
    shape, radius = np.asarray(image.shape[:3]), np.asarray(radius)
    sums, counts = np.empty((len(centers), image.shape[3])), np.empty(len(centers), dtype=np.int64)
    for n, c in enumerate(centers):
        lo, hi = np.maximum(c - radius, 0), np.minimum(c + radius + 1, shape)
        sums[n] = image[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].sum(axis=(0, 1, 2), dtype=np.float64)
        counts[n] = int(np.prod(hi - lo))
    return sums / counts[:, None]


def _estimates(condition, seed):
    """Per (amplicon, channel, round) records at the truth positions, as W-278 estimator_records."""
    book, scene = scenes.scene(condition, seed)
    empty, latent = scenes.twins(condition, seed)
    noise = scene.provenance["effective_config"]["noise"]
    quantization = 1 / 12 if scene.provenance["effective_config"]["dtype"].startswith("uint") else 0.0
    channels = book.channel_labels
    shape = np.asarray(scene.rounds[scene.round_labels[0]].shape[:3])
    records = []
    for j, label in enumerate(scene.round_labels):
        truth = scenes.truth(scene, label)
        points = np.clip(truth[["z", "y", "x"]].to_numpy(float), 0, shape - 1)
        centers = np.floor(points + 0.5).astype(np.int64)
        found = scenes.spots_at(points, scene.metadata, f"r10/{condition}/{seed}/{label}", channels)
        full, spot_free = (extract_intensities({label: scenes.loaded(images.rounds[label], scene.metadata, channels)},
                                               found, config=scenes.EXTRACTION) for images in (scene, empty))
        latent_box = _box_mean(latent.rounds[label], centers)
        realized_box = _box_mean(empty.rounds[label], centers)
        noise_true = np.sqrt(noise["alpha"] * np.maximum(latent_box, 0) + noise["sigma"] ** 2
                             + noise["correlated_sigma"] ** 2 + quantization)
        offsets = np.abs(centers[None, :, :] - centers[:, None, :])
        in_outer = np.all(offsets <= np.array([1, 6, 6]), axis=2)
        in_inner = np.all(offsets <= np.array([1, 3, 3]), axis=2)
        np.fill_diagonal(in_outer, False)
        np.fill_diagonal(in_inner, False)
        for c in range(len(channels)):
            emits = scene.realized[:, c, j] > 0
            records.append(pd.DataFrame({
                "condition": condition, "eligible": truth.center_in_bounds.to_numpy(bool),
                "shell_neighbor": ((in_outer & ~in_inner) & emits[None, :]).any(axis=1),
                "inner_neighbor": (in_inner & emits[None, :]).any(axis=1),
                "box_clipped": full.box_voxels[:, 0] < 75,
                "background": full.background[:, c, 0], "noise": full.noise[:, c, 0],
                "spot_free_background": spot_free.background[:, c, 0],
                "latent": latent_box[:, c], "realized": realized_box[:, c], "noise_true": noise_true[:, c]}))
    return pd.concat(records, ignore_index=True)


@pytest.fixture(scope="module")
def r10_records():
    frame = pd.concat([_estimates(condition, seed) for condition in scenes.CONDITIONS
                       for seed in scenes.HELDOUT_SEEDS], ignore_index=True)
    return frame[frame.eligible]


@pytest.mark.extended
@pytest.mark.slow
@pytest.mark.validation
def test_r10_background_on_calibrated_scenes(r10_records):
    f = r10_records
    report = {}
    for condition, g in f.groupby("condition", sort=False):
        shell = g[g.shell_neighbor]
        contamination = shell.background - shell.spot_free_background
        report[condition] = dict(
            realized=float((g.background - g.realized).abs().median()),
            latent=float((g.background - g.latent).median()),
            contamination=(float(contamination.median()), float(np.quantile(contamination, 0.9))))
    print("\nR10 per condition (realized-error median, latent bias median, shell contamination median and p90):")
    for condition, row in report.items():
        print(f"  {condition}: {row}")
    assert set(report) == set(scenes.CONDITIONS)
    for condition, row in report.items():
        dense = condition == "dense"
        # estimators.csv, heldout, local_ring, stratum=all: 2.16 to 2.48 grey levels.
        assert row["realized"] <= 2.5, condition
        # stratum=all: latent bias +0.87 to +1.0, dense +2.0.
        assert 0 <= row["latent"] <= (2 if dense else 1), condition
        # stratum=neighbor_in_shell: +2 (p90 +5), dense +3 (p90 +6).
        median, p90 = row["contamination"]
        assert median <= (3 if dense else 2) and p90 <= (6 if dense else 5), condition
    # Noise relative error, pooled over the nine conditions (all_calibrated): no neighbor
    # -13.1 %, box clipped at the border -13.8 %.
    relative = f.noise / f.noise_true - 1
    no_neighbor = float(relative[~f.shell_neighbor & ~f.inner_neighbor].median())
    border = float(relative[f.box_clipped].median())
    print(f"R10 noise relative error: no neighbor {no_neighbor:.4f}, at the border {border:.4f}")
    assert -0.15 <= no_neighbor <= 0.10 and -0.15 <= border <= 0.10
