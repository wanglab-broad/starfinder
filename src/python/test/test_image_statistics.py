"""Focused tests of benchmarks/image_statistics.py (W-238) on arrays with analytically known values."""
from dataclasses import replace
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.synthetic import ScalarDistribution, development_scene_preset, generate_formed_scene

ROOT = Path(__file__).resolve().parents[3]


def module():
    spec = importlib.util.spec_from_file_location("image_statistics", ROOT / "benchmarks" / "image_statistics.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


istat = module()


def _volume(counts, shape):
    values = np.concatenate([np.full(n, v, np.uint8) for v, n in counts.items()])
    return np.random.default_rng(0).permutation(values).reshape(shape)


def test_histogram_statistics_match_known_counts():
    # 1000 voxels: 500 zeros, 400 ones, 90 twos, 9 at 200 and one saturated voxel.
    volume = _volume({0: 500, 1: 400, 2: 90, 200: 9, 255: 1}, (4, 10, 25))
    counts = np.bincount(volume.ravel(), minlength=256)
    result = istat.histogram_statistics(counts)
    assert result["voxels"] == 1000
    assert result["zero_fraction"] == 0.5 and result["saturated_fraction"] == 0.001
    assert (result["p50"], result["p90"], result["p99"], result["p99_9"], result["p99_99"]) == (0, 1, 2, 200, 255)
    assert result["max"] == 255 and result["mad"] == 0
    # 1..9 once each: median 5, absolute deviations 0,1,1,2,2,3,3,4,4 -> MAD 2.
    result = istat.histogram_statistics(np.bincount(np.arange(1, 10), minlength=256))
    assert (result["p50"], result["mad"], result["max"], result["zero_fraction"]) == (5, 2, 9, 0.0)
    # Inverted CDF: p90 of 1..10 is the 9th value; p99 and p99.9 the 10th.
    counts = np.bincount(np.arange(1, 11), minlength=256)
    assert [istat.quantile_level(counts, p) for p in (10, 50, 90, 99, 99.9)] == [1, 5, 9, 10, 10]


def test_depth_profile_and_attenuation_are_per_plane():
    # Plane z holds 999 voxels at z and one at 100 + z: its p99.9 is z (rank 999 of 1000).
    volume = np.zeros((8, 20, 50), np.uint8)
    for z in range(8):
        volume[z] = z
        volume[z, 0, 0] = 100 + z
    measured = istat.measure_volume(volume, crop_yx=16)
    assert measured.record["depth_p99_9"] == list(range(8))
    assert measured.record["depth_attenuation"] == pytest.approx(0.5 / 6.5)  # planes 0-1 over planes 6-7
    volume[:] += 1
    measured = istat.measure_volume(volume, crop_yx=16)
    assert measured.record["depth_p99_9"] == list(range(1, 9))
    assert measured.record["depth_attenuation"] == pytest.approx(1.5 / 7.5)
    assert istat.depth_attenuation([4, 4, 2, 2, 1, 1, 1, 1]) == 4.0
    assert istat.depth_attenuation([1, 2, 3]) is None  # fewer than four planes: a quarter is empty
    assert list(istat.depth_quarter(np.arange(30), 30)) == [0] * 8 + [1] * 7 + [2] * 8 + [3] * 7


def test_uint16_saturation_and_rejected_dtypes():
    volume = np.zeros((2, 4, 5), np.uint16)
    volume[0, 0, :2] = 65535
    volume[1, 1, 1] = 300
    result = istat.histogram_statistics(np.bincount(volume.ravel(), minlength=65536))
    assert result["saturated_fraction"] == 2 / 40 and result["max"] == 65535 and result["p99"] == 65535
    with pytest.raises(TypeError):
        istat.measure_volume(volume.astype(np.float32))


def test_detect_maxima_footprint_thresholds_and_plateaus():
    stack = np.zeros((7, 40, 40), np.uint8)
    stack[3, 20, 20] = 100          # adaptive and lenient
    stack[3, 20, 22] = 90           # 2 px away in x: inside the 5-wide footprint of the 100, suppressed
    stack[3, 20, 26] = 50           # 4 px from the 90 and 6 px from the 100: kept
    stack[1, 30, 30] = 15           # lenient only
    stack[2, 30, 30] = 12           # 1 plane from the 15: inside the 3-deep footprint, suppressed
    stack[5, 30, 30] = 11           # 4 planes from the 15: kept by the lenient selection
    stack[3, 10, 30] = stack[3, 10, 31] = 40  # plateau of two equal maxima: counted once, at its first voxel
    stack[3, 5, 5] = 5              # below both thresholds
    stack[6, 2, 20] = 80            # near the Y edge
    adaptive = istat.detect_maxima(stack, 0.2 * stack.max())
    assert adaptive.tolist() == [[3, 10, 30], [3, 20, 20], [3, 20, 26], [6, 2, 20]]
    lenient = istat.detect_maxima(stack, istat.LENIENT_LEVEL)
    assert lenient.tolist() == [[1, 30, 30], [3, 10, 30], [3, 20, 20], [3, 20, 26], [5, 30, 30], [6, 2, 20]]
    # A 10-voxel margin drops maxima with y or x below 10 or above 29 (the plateau at x = 30 and y = 2).
    assert istat.detect_maxima(stack, 20, margin_yx=10).tolist() == [[3, 20, 20], [3, 20, 26]]
    # At exactly 0.2 x max the threshold is inclusive.
    assert [3, 20, 20] in istat.detect_maxima(stack, 100).tolist()
    assert istat.detect_maxima(np.zeros((3, 8, 8), np.uint8), 0).shape == (0, 3)


def test_punctum_statistics_exact_on_constant_and_ramp_annuli():
    stack = np.full((1, 21, 21), 20.0)
    stack[0, 9:12, 9:12] += 5.0  # confined to the central 7x7: no effect on the annulus
    stack[0, 10, 10] = 70.0
    result = istat.punctum_statistics(stack, [[0, 10, 10]])
    assert result["background"][0] == 20 and result["amplitude"][0] == 50
    assert result["clutter_sigma"][0] == 0 and result["pixel_sigma"][0] == 0
    assert np.isnan(result["snr_clutter"][0]) and np.isnan(result["snr_pixel"][0])
    # A ramp 20 + g x: annulus x offsets have mean 0 and mean square 15974 / 392, horizontal
    # differences are all g and vertical ones 0 (364 of each), so the pixel sigma is g / (2 sqrt 2).
    g = 0.5
    ramp = (20 + g * (np.arange(21) - 10))[None, None, :].repeat(21, axis=1)
    ramp[0, 10, 10] = 60.0
    result = istat.punctum_statistics(ramp, [[0, 10, 10]])
    assert result["background"][0] == pytest.approx(20)
    assert result["clutter_sigma"][0] == pytest.approx(g * math.sqrt(15974 / 392))
    assert result["pixel_sigma"][0] == pytest.approx(g / (2 * math.sqrt(2)))
    assert result["snr_pixel"][0] == pytest.approx(40 / (g / (2 * math.sqrt(2))))
    with pytest.raises(ValueError):
        istat.punctum_statistics(ramp, [[0, 9, 10]])


def test_gaussian_punctum_on_constant_background_with_white_noise():
    # 100 Gaussian puncta (sigma 1 voxel, peak 60) on background 20 with white noise sigma 3.
    background, peak, noise = 20.0, 60.0, 3.0
    size, spacing = 250, 25
    grid = np.arange(12, size - 12, spacing)
    yy, xx = np.mgrid[:size, :size]
    image = np.full((size, size), background)
    for y in grid:
        for x in grid:
            image += peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2)
    image += np.random.default_rng(1).normal(0, noise, image.shape)
    centres = np.array([[0, y, x] for y in grid for x in grid])
    result = istat.punctum_statistics(image[None], centres)
    n, ring = len(centres), 392
    # Means over 100 puncta; tolerances are 5 standard errors of each estimator.
    assert result["background"].mean() == pytest.approx(background, abs=5 * noise * 1.26 / math.sqrt(ring * n))
    assert result["clutter_sigma"].mean() == pytest.approx(noise, abs=5 * noise / math.sqrt(2 * ring * n))
    assert result["pixel_sigma"].mean() == pytest.approx(noise, abs=5 * noise / math.sqrt(728 * n / 3))
    assert result["amplitude"].mean() == pytest.approx(peak, abs=5 * noise / math.sqrt(n))
    assert np.median(result["snr_clutter"]) == pytest.approx(peak / noise, rel=0.05)
    assert np.median(result["snr_pixel"]) == pytest.approx(peak / noise, rel=0.05)
    assert np.allclose(result["peak"], image[centres[:, 1], centres[:, 2]])
    # The detector finds every punctum as a maximum at the adaptive threshold.
    found = istat.detect_maxima(image[None], 0.2 * image.max(), margin_yx=10)
    assert {tuple(c) for c in centres} <= {tuple(c) for c in found}


def test_amplitude_breakdown_by_round_channel_and_depth_quarter():
    puncta = pd.DataFrame(dict(
        round=["round1"] * 4 + ["round2"] * 4,
        channel=["ch00", "ch01", "ch00", "ch01"] * 2,
        depth_quarter=[0, 0, 3, 3, 1, 1, 2, 2],
        amplitude=[10.0, 30.0, 20.0, 40.0, 50.0, 70.0, 60.0, 80.0]))
    assert istat.amplitude_breakdown(puncta, "round").amplitude_p50.to_dict() == {"round1": 25.0, "round2": 65.0}
    assert istat.amplitude_breakdown(puncta, "channel").amplitude_p50.to_dict() == {"ch00": 35.0, "ch01": 55.0}
    by_depth = istat.amplitude_breakdown(puncta, "depth_quarter")
    assert by_depth.amplitude_p50.to_dict() == {0: 20.0, 1: 60.0, 2: 70.0, 3: 30.0}
    assert by_depth.n.to_dict() == {0: 2, 1: 2, 2: 2, 3: 2}
    assert istat.max_min_ratio(istat.amplitude_breakdown(puncta, "round").amplitude_p50) == 65 / 25
    # Per volume: the adaptive puncta's median amplitude per depth quarter.
    stack = np.zeros((8, 40, 40), np.uint8)
    for z, value in zip((0, 2, 5, 7), (100, 80, 60, 40)):
        stack[z, 20, 20] = value
    record = istat.measure_volume(stack, crop_yx=40).record
    assert record["adaptive"]["amplitude_by_depth_quarter"] == [100.0, 80.0, 60.0, 40.0]


def test_truncated_lognormal_fit_recovers_sigma():
    rng = np.random.default_rng(3)
    values = np.exp(rng.normal(4.0, 0.5, 20000))
    kept = values[values >= math.exp(4.0)]
    mu, sigma = istat.truncated_lognormal_fit(kept, math.exp(4.0))
    assert sigma == pytest.approx(0.5, abs=0.02) and mu == pytest.approx(4.0, abs=0.03)
    assert np.log(kept).std() < 0.35  # the naive estimate is biased low by the truncation
    assert istat.truncated_lognormal_fit(kept[:5], math.exp(4.0)) == (None, None)
    # Log values exponential above the cut (only a far tail observed): the likelihood keeps
    # improving as mu -> -inf and sigma -> inf, so the fit is reported as unidentified.
    tail = np.exp(math.log(40.0) + rng.exponential(0.5, 2000))
    assert istat.truncated_lognormal_fit(tail, 40.0) == (None, None)


def test_same_function_measures_synthetic_channel_and_tiff(tmp_path):
    book, config = development_scene_preset("combined")
    config = replace(config, shape_zyx=(8, 48, 48), coordinates=None, count=12, amplicon_ids=None, gene_ids=None,
                     elongation=ScalarDistribution("uniform", (1.0, 1.5)),
                     angle=ScalarDistribution("uniform", (0.0, float(np.pi))),
                     brightness=ScalarDistribution(parameters=(96.0,)), dtype="uint8")
    scene = generate_formed_scene(book, config=config)
    round_label, channel_label = scene.round_labels[0], scene.channel_labels[1]
    channel = istat.synthetic_channel(scene, round_label, channel_label)
    assert channel.shape == (8, 48, 48) and channel.dtype == np.uint8
    from_array = istat.measure_volume(channel, crop_yx=48, max_puncta=5, seed=7)
    path = tmp_path / "scene_ch01.tif"
    tifffile.imwrite(path, np.ascontiguousarray(channel), imagej=True)
    with istat.TiffVolume(path) as volume:
        from_tiff = istat.measure_volume(volume, crop_yx=48, max_puncta=5, seed=7)
    assert from_array.record == from_tiff.record
    pd.testing.assert_frame_equal(from_array.puncta, from_tiff.puncta)
    record = from_array.record
    assert record["crop_yx"] == [0, 48, 0, 48] and record["max"] == int(channel.max())
    assert record["adaptive"]["n_maxima"] > 0 and record["adaptive"]["n_measured"] <= 5
    assert set(from_array.puncta.selection) <= set(istat.SELECTIONS)
    source = istat.source_record(path)
    assert source["bytes"] == path.stat().st_size and source["mtime_ns"] == path.stat().st_mtime_ns


def test_fov_rule_and_run_on_a_tiny_root(tmp_path):
    assert istat.fov_positions_indices(64) == [0, 21, 42, 63]
    assert istat.fov_positions_indices(56) == [0, 18, 37, 55]
    assert istat.fov_positions_indices(6) == [0, 2, 3, 5]
    root = tmp_path / "root"
    rng = np.random.default_rng(0)
    for round_label in ("round1", "round2"):
        for fov in ("tile_1", "tile_10", "tile_2", "tile_3", "tile_4"):
            directory = root / "tissue-2D" / round_label / fov
            directory.mkdir(parents=True)
            for channel in (*istat.CHANNELS, "ch04"):
                image = rng.poisson(2, (6, 40, 40)).astype(np.uint8)
                image[3, 20, 20] = 90
                tifffile.imwrite(directory / f"x_{channel}.tif", image, imagej=True)
    plan = istat.fov_plan(root, ["tissue-2D"])["tissue-2D"]
    # String order: tile_1, tile_10, tile_2, tile_3, tile_4 -> indices 0, 1, 3, 4.
    assert [f["fov"] for f in plan["fovs"]] == ["tile_1", "tile_10", "tile_3", "tile_4"]
    assert [f["split"] for f in plan["fovs"]] == ["development", "development", "held_out", "development"]
    manifest = istat.run(root, tmp_path / "out", pilot=False, datasets=["tissue-2D"], crop_yx=32, log=lambda m: None)
    assert len(manifest["inputs"]) == 4 * 2 * 4 and manifest["inputs_unchanged_after_run"]
    assert not any(p["path"].endswith("ch04.tif") for p in manifest["inputs"])
    targets = pd.read_csv(tmp_path / "out" / "tables" / "targets.csv")
    assert set(targets.split) == {"development", "held_out"}
    assert (tmp_path / "out" / "overlays" / "tissue-2D.png").stat().st_size > 0
    with pytest.raises(ValueError):
        istat.run(root, root / "inside", pilot=True, datasets=["tissue-2D"], log=lambda m: None)
