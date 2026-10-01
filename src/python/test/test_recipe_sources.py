"""Recipe snapshots, extraction and registration sources, and snapshot transforms (W-232).

Two synthetic uint16 rounds of 4x32x32 voxels and two channels: the moving
round is the reference round rolled by a known integer YX shift, so both
rounds have the same voxel values and every per-round statistic agrees.
"""
from dataclasses import replace
import json

import numpy as np
import pytest

from starfinder.barcode import NeighborhoodSumConfig, extract_intensities
from starfinder.dataset import (CheckpointConfig, Dataset, ExecutionConfig, PipelineConfig, RegistrationRecipe,
    RegistrationStep, RoundState)
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult
from starfinder.preprocessing import (HistogramMatchingConfig, MinMaxNormalizationConfig, PercentileNormalizationConfig,
    PreprocessingRecipe, PreprocessingStep, ReconstructionConfig, ScalarBackgroundConfig, TophatConfig, normalize_percentile,
    subtract_scalar_background, summary_stage)
from starfinder.registration import DemonsConfig, RegistrationSignalConfig, TranslationConfig, WarpConfig, apply_transform
from starfinder.spot_finding import LocalMaximaConfig

from .test_preprocessing_golden import PINNED_SEQUENCE, digest, fixture_rounds

CHANNELS = ("ch00", "ch01")
SHIFT_YX = (3, -2)  # the moving round's content is displaced by this many voxels
BACKGROUND = ScalarBackgroundConfig(percentile=10.0)
RECIPE_2 = PreprocessingRecipe((PreprocessingStep(BACKGROUND, save_as="bg_corrected"),
                                PreprocessingStep(PercentileNormalizationConfig())), extraction_source="bg_corrected")
TRANSLATION = RegistrationRecipe((RegistrationStep(TranslationConfig()),), signal=RegistrationSignalConfig("sum"))
DETECTION = LocalMaximaConfig(threshold_value=5.0)
EXTRACTION = NeighborhoodSumConfig((0, 1, 1))


def synthetic_rounds():
    """Reference with a smooth channel-dependent baseline, noise and bright puncta; moving is it rolled."""
    rng = np.random.default_rng(232)
    y = np.arange(32)[None, :, None, None]
    volume = 200 + 3 * y * np.array([1, 2]) + rng.normal(0, 4, (4, 32, 32, 2))
    for _ in range(12):
        z, r, c = rng.integers(0, 4), rng.integers(6, 26), rng.integers(6, 26)
        volume[z, r - 1:r + 2, c - 1:c + 2] += rng.uniform(300, 900, 2) * 0.5
        volume[z, r, c] += rng.uniform(600, 1800, 2)
    reference = np.clip(np.rint(volume), 0, 65535).astype(np.uint16)
    return {"round1": reference, "round2": np.roll(reference, SHIFT_YX, axis=(1, 2))}


def dataset(tmp_path, channels=CHANNELS):
    return Dataset(tmp_path, tmp_path / "out", "sources", "sample", "out",
                   rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
                   channel_order=channels)


def resident_fov(tmp_path, rounds=None, channels=CHANNELS):
    fov = dataset(tmp_path, channels).fov("FOV_001")
    for name, volume in (rounds or synthetic_rounds()).items():
        fov.images[name] = volume.copy()
        # One frame for all rounds, so extraction without registration is allowed.
        fov.metadata[name] = ImageMetadata("FOV_001")
    return fov


def manual_intensities(fov, images):
    """extract_intensities called directly on the given per-round images."""
    rounds = {name: ImageLoadResult(image, fov.metadata[name], CHANNELS, ()) for name, image in images.items()}
    return extract_intensities(rounds, fov.spot_result, config=EXTRACTION)


# --- Recipe validation -------------------------------------------------------------

def test_snapshot_names_and_sources_are_validated():
    step = PreprocessingStep(BACKGROUND, save_as="bg_corrected")
    for bad in ("detection", "Bg", "", "bg-corrected", 3):
        with pytest.raises(ValueError, match="save_as"):
            PreprocessingStep(BACKGROUND, save_as=bad)
    with pytest.raises(ValueError, match="more than once"):
        PreprocessingRecipe((step, PreprocessingStep(TophatConfig(), save_as="bg_corrected")))
    for field in ("extraction_source", "registration_source"):
        for name in ("missing", "detection"):
            with pytest.raises(ValueError, match=f"{field} '{name}' is not a snapshot"):
                PreprocessingRecipe((step,), **{field: name})
    with pytest.raises(ValueError, match="keep no snapshots"):
        PreprocessingRecipe(post_registration=(PreprocessingStep(ReconstructionConfig(), save_as="late"),))
    assert RECIPE_2.snapshots == ["bg_corrected"]


def test_summary_stage_drops_the_sources():
    recipe = replace(RECIPE_2, steps=(RECIPE_2.steps[0], PreprocessingStep(PercentileNormalizationConfig(fit="supplied"))),
                     registration_source="bg_corrected", supplied_statistics="s.json")
    prefix, _ = summary_stage(recipe, "percentile_normalization")
    assert prefix.steps == recipe.steps[:1]
    assert prefix.extraction_source is None and prefix.registration_source is None


# --- Default recipe: unchanged results ------------------------------------------------

def test_snapshots_without_sources_leave_recipe_1_results_unchanged(tmp_path):
    """Declared snapshots cost memory only: the detection images keep the pinned digests."""
    recipe = PreprocessingRecipe((PreprocessingStep(MinMaxNormalizationConfig("uint8", (0, 255)), save_as="normalized"),
                                  PreprocessingStep(HistogramMatchingConfig(reference_channel=0), save_as="matched"),
                                  PreprocessingStep(ReconstructionConfig(radius_yx=3))))
    golden = Dataset(tmp_path, tmp_path / "out", "golden", "sample", "out",
                     rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
                     channel_order=("ch00", "ch01", "ch02", "ch03"))
    fov = golden.fov("FOV_001")
    for name, volume in fixture_rounds().items():
        fov.images[name], fov.metadata[name] = volume.copy(), ImageMetadata(f"FOV_001/{name}")
    fov.run(PipelineConfig(preprocessing=recipe))
    assert {name: digest(fov.images[name]) for name in fov.images} == PINNED_SEQUENCE
    assert {name: list(images) for name, images in fov.snapshots.items()} == {
        "round1": ["normalized", "matched"], "round2": ["normalized", "matched"]}
    assert [r["save_as"] for r in fov.preprocessing_record["rounds"]["round2"]] == ["normalized", "matched", None]


def test_default_sources_register_and_extract_the_detection_image(tmp_path):
    """Without sources, run is the detection-image path: same images and intensities with or without snapshots."""
    plain = replace(RECIPE_2, steps=(PreprocessingStep(BACKGROUND), RECIPE_2.steps[1]), extraction_source=None)
    tapped = replace(RECIPE_2, extraction_source=None)
    config = PipelineConfig(preprocessing=plain, registration=TRANSLATION, spot_finding=DETECTION, extraction=EXTRACTION)
    first = resident_fov(tmp_path).run(config)
    second = resident_fov(tmp_path).run(replace(config, preprocessing=tapped))
    for name in ("round1", "round2"):
        np.testing.assert_array_equal(first.images[name], second.images[name])
    np.testing.assert_array_equal(first.intensity_result.values, second.intensity_result.values)
    expected = manual_intensities(first, {name: first.images[name] for name in ("round1", "round2")})
    np.testing.assert_array_equal(first.intensity_result.values, expected.values)


# --- Extraction source ----------------------------------------------------------------

@pytest.mark.parametrize("mode", ["batch", "streaming"])
def test_extraction_source_reads_the_background_snapshot(tmp_path, mode):
    fov = resident_fov(tmp_path)
    raw = {name: image.copy() for name, image in fov.images.items()}
    fov.run(PipelineConfig(preprocessing=RECIPE_2, spot_finding=DETECTION, extraction=EXTRACTION),
            execution=ExecutionConfig(mode, retain_images=True))
    assert len(fov.spot_result.spots) > 0
    snapshots = {name: subtract_scalar_background(image, config=BACKGROUND) for name, image in raw.items()}
    for name in raw:
        np.testing.assert_array_equal(fov.snapshots[name]["bg_corrected"], snapshots[name])
        np.testing.assert_array_equal(fov.images[name], normalize_percentile(snapshots[name]))
    expected = manual_intensities(fov, snapshots)
    np.testing.assert_array_equal(fov.intensity_result.values, expected.values)
    np.testing.assert_array_equal(fov.intensity_result.valid, expected.valid)
    detection = manual_intensities(fov, {name: fov.images[name] for name in raw})
    assert not np.array_equal(fov.intensity_result.values, detection.values)


# --- Registration of snapshots -------------------------------------------------------

def test_known_translation_shifts_detection_and_extraction_snapshot_identically(tmp_path):
    fov = resident_fov(tmp_path)
    raw = {name: image.copy() for name, image in fov.images.items()}
    fov.run(PipelineConfig(preprocessing=RECIPE_2, registration=TRANSLATION, spot_finding=DETECTION, extraction=EXTRACTION))
    (result,) = fov.registration_results["round2"]
    assert result.transform.displacement_zyx == (0, *SHIFT_YX)
    before = {name: subtract_scalar_background(image, config=BACKGROUND) for name, image in raw.items()}
    detection_before = {name: normalize_percentile(image) for name, image in before.items()}
    # The reference round is not transformed.
    np.testing.assert_array_equal(fov.snapshots["round1"]["bg_corrected"], before["round1"])
    np.testing.assert_array_equal(fov.images["round1"], detection_before["round1"])
    # The moving round's detection image and snapshot receive the same single resampling.
    warp = result.application_config
    np.testing.assert_array_equal(fov.snapshots["round2"]["bg_corrected"],
                                  apply_transform(before["round2"], result.transform, config=warp))
    np.testing.assert_array_equal(fov.images["round2"], apply_transform(detection_before["round2"], result.transform, config=warp))
    # Undoing the known roll aligns both with the reference round away from the wrapped border.
    inner = (slice(None), slice(4, 28), slice(4, 28))
    np.testing.assert_array_equal(fov.snapshots["round2"]["bg_corrected"][inner], before["round1"][inner])
    np.testing.assert_array_equal(fov.images["round2"][inner], detection_before["round1"][inner])
    expected = manual_intensities(fov, {name: fov.snapshots[name]["bg_corrected"] for name in raw})
    np.testing.assert_array_equal(fov.intensity_result.values, expected.values)


def test_every_image_of_a_moving_round_is_resampled_once_by_its_chain(tmp_path):
    """A (translation, demons) recipe: the detection image and each snapshot are resampled once, by the round's chain."""
    local = RegistrationStep(DemonsConfig(iterations=(5,)), signal=RegistrationSignalConfig("channel", 0))
    recipe = PreprocessingRecipe((PreprocessingStep(BACKGROUND, save_as="bg_corrected"),
                                  PreprocessingStep(PercentileNormalizationConfig(), save_as="normalized")),
                                 extraction_source="bg_corrected")
    fov = resident_fov(tmp_path)
    raw = {name: image.copy() for name, image in fov.images.items()}
    fov.run(PipelineConfig(preprocessing=recipe, registration=replace(TRANSLATION, steps=TRANSLATION.steps + (local,))))
    methods = [r.diagnostics.method for r in fov.registration_results["round2"]]
    assert methods == ["translation", "demons"]
    chain = fov.registration_chains["round2"]
    assert chain.transforms == tuple(r.transform for r in fov.registration_results["round2"])
    warp = fov.registration_record["application"]["round2"]
    assert warp == WarpConfig(backend="scipy")
    before = subtract_scalar_background(raw["round2"], config=BACKGROUND)
    np.testing.assert_array_equal(fov.snapshots["round2"]["bg_corrected"], apply_transform(before, chain, config=warp))
    np.testing.assert_array_equal(fov.images["round2"], apply_transform(normalize_percentile(before), chain, config=warp))
    # A snapshot of the last step equals the detection image before registration, and stays equal after it.
    np.testing.assert_array_equal(fov.snapshots["round2"]["normalized"], fov.images["round2"])
    applied = fov.preprocessing_record["transforms"]["round2"]
    assert applied == {key: [{"result": 0, "method": "translation", "kind": "translation"},
                             {"result": 1, "method": "demons", "kind": "dense"}]
                       for key in ("detection", "bg_corrected", "normalized")}
    assert fov.preprocessing_record["transforms"]["round1"] == {"detection": [], "bg_corrected": [], "normalized": []}


def test_registration_signals_come_from_the_registration_source(tmp_path, monkeypatch):
    import starfinder.registration as registration
    captured = []
    estimate = registration.estimate_transform

    def spy(reference, moving, **kwargs):
        captured.append((np.array(reference), np.array(moving)))
        return estimate(reference, moving, **kwargs)

    monkeypatch.setattr(registration, "estimate_transform", spy)
    recipe = replace(RECIPE_2, registration_source="bg_corrected")
    fov = resident_fov(tmp_path)
    raw = {name: image.copy() for name, image in fov.images.items()}
    fov.run(PipelineConfig(preprocessing=recipe, registration=TRANSLATION))
    before = {name: subtract_scalar_background(image, config=BACKGROUND) for name, image in raw.items()}
    ((reference, moving),) = captured
    np.testing.assert_array_equal(reference, before["round1"].sum(axis=-1, dtype=np.float64))
    np.testing.assert_array_equal(moving, before["round2"].sum(axis=-1, dtype=np.float64))
    detection = normalize_percentile(before["round1"]).sum(axis=-1, dtype=np.float64)
    assert not np.array_equal(reference, detection)
    # The default source is the detection image.
    captured.clear()
    resident_fov(tmp_path).run(PipelineConfig(preprocessing=RECIPE_2, registration=TRANSLATION))
    np.testing.assert_array_equal(captured[0][0], detection)
    assert fov.preprocessing_record["recipe"]["registration_source"] == "bg_corrected"


def test_registration_source_with_post_registration_reconstruction(tmp_path, monkeypatch):
    """The resident-subtile path: the source snapshot, not the post-processed reference, is registered against."""
    import starfinder.registration as registration
    captured = []
    estimate = registration.estimate_transform
    monkeypatch.setattr(registration, "estimate_transform",
                        lambda reference, moving, **kw: captured.append(np.array(reference)) or estimate(reference, moving, **kw))
    recipe = PreprocessingRecipe((PreprocessingStep(BACKGROUND, save_as="bg_corrected"),),
                                 (PreprocessingStep(ReconstructionConfig()),), registration_source="bg_corrected")
    fov = resident_fov(tmp_path)
    reference = subtract_scalar_background(fov.images["round1"], config=BACKGROUND)
    fov.run(PipelineConfig(preprocessing=recipe, registration=TRANSLATION))
    np.testing.assert_array_equal(captured[0], reference.sum(axis=-1, dtype=np.float64))
    np.testing.assert_array_equal(fov.snapshots["round1"]["bg_corrected"], reference)


def test_streaming_keeps_only_the_reference_registration_source(tmp_path, monkeypatch):
    import starfinder.dataset.fov as fov_module
    seen = []
    register = fov_module.FOV.register

    def spy(self, step, *, rounds=None, source=None):
        seen.append({name: sorted(images) for name, images in self.snapshots.items()})
        return register(self, step, rounds=rounds, source=source)

    monkeypatch.setattr(fov_module.FOV, "register", spy)
    recipe = replace(RECIPE_2, registration_source="bg_corrected",
                     steps=(RECIPE_2.steps[0], replace(RECIPE_2.steps[1], save_as="normalized")))
    fov = resident_fov(tmp_path)
    fov.run(PipelineConfig(preprocessing=recipe, registration=TRANSLATION), execution=ExecutionConfig("streaming"))
    assert seen == [{"round1": ["bg_corrected"], "round2": ["bg_corrected", "normalized"]}]
    assert set(fov.images) == {"round1"} and fov.snapshots == {}


def test_missing_source_snapshot_raises(tmp_path):
    fov = resident_fov(tmp_path)
    with pytest.raises(ValueError, match="registration source 'bg' is not a snapshot of round 'round1'"):
        fov.register(TRANSLATION, source="bg")
    with pytest.raises(ValueError, match="extraction source 'bg' is not a snapshot"):
        fov._extract_round(round_name="round1", source="bg")


# --- Checkpoint and run record ------------------------------------------------------

@pytest.mark.parametrize("mode", ["batch", "streaming"])
def test_registered_checkpoint_round_trips_each_downstream_snapshot(tmp_path, mode):
    fov = resident_fov(tmp_path)
    config = PipelineConfig(preprocessing=RECIPE_2, registration=TRANSLATION, spot_finding=DETECTION, extraction=EXTRACTION)
    fov.run(config, execution=ExecutionConfig(mode, retain_images=True), checkpoints=CheckpointConfig(stages=("registered",)))
    directory = fov.paths.checkpoint_dir
    files = json.loads((directory / "run.json").read_text())["checkpoints"]["registered"]
    for name in ("round1", "round2"):
        assert f"registered/{name}.ome.tif" in files and f"registered/bg_corrected/{name}.ome.tif" in files
    header = json.loads((directory / "registered" / "transforms.json").read_text())
    assert header["snapshots"] == ["bg_corrected"]
    reloaded = dataset(tmp_path).fov("FOV_001").load_checkpoint("registered")
    assert set(reloaded.images) == set(reloaded.snapshots) == {"round1", "round2"}
    for name in ("round1", "round2"):
        for restored, original in ((reloaded.images[name], fov.images[name]),
                                   (reloaded.snapshots[name]["bg_corrected"], fov.snapshots[name]["bg_corrected"])):
            assert restored.dtype == original.dtype and restored.shape == original.shape
            np.testing.assert_array_equal(restored, original)
        assert list(reloaded.snapshots[name]) == ["bg_corrected"]
    # A run resumed from the checkpoint extracts from the restored snapshot.
    reloaded.run(PipelineConfig(spot_finding=DETECTION, extraction=EXTRACTION))
    np.testing.assert_array_equal(reloaded.intensity_result.values, fov.intensity_result.values)
    with pytest.raises(ValueError, match="without images, snapshots"):
        reloaded.load_checkpoint("registered")


def test_recipe_without_extraction_source_stores_one_image_per_round(tmp_path):
    recipe = replace(RECIPE_2, extraction_source=None, registration_source="bg_corrected")
    fov = resident_fov(tmp_path)
    fov.run(PipelineConfig(preprocessing=recipe, registration=TRANSLATION), checkpoints=CheckpointConfig(stages=("registered",)))
    registered = fov.paths.checkpoint_dir / "registered"
    assert sorted(p.name for p in registered.iterdir()) == ["round1.ome.tif", "round2.ome.tif", "transforms.json"]
    reloaded = dataset(tmp_path).fov("FOV_001").load_checkpoint("registered")
    assert reloaded.snapshots == {}
    # overwrite clears snapshot files of an earlier run.
    fov = resident_fov(tmp_path)
    fov.run(PipelineConfig(preprocessing=RECIPE_2), checkpoints=CheckpointConfig(stages=("registered",), overwrite=True))
    assert (registered / "bg_corrected" / "round2.ome.tif").is_file()
    resident_fov(tmp_path).run(PipelineConfig(preprocessing=recipe), checkpoints=CheckpointConfig(stages=("registered",), overwrite=True))
    assert not (registered / "bg_corrected").exists()


def test_run_json_lists_sources_and_transforms_per_snapshot(tmp_path):
    recipe = replace(RECIPE_2, registration_source="bg_corrected")
    fov = resident_fov(tmp_path)
    fov.run(PipelineConfig(preprocessing=recipe, registration=TRANSLATION), checkpoints=CheckpointConfig(stages=("registered",)))
    entry = json.loads((fov.paths.checkpoint_dir / "run.json").read_text())["preprocessing"]
    assert entry["recipe"] == {"steps": ["scalar_background", "percentile_normalization"], "post_registration": [],
                               "extraction_source": "bg_corrected", "registration_source": "bg_corrected"}
    assert [r["save_as"] for r in entry["rounds"]["round2"]] == ["bg_corrected", None]
    translation = [{"result": 0, "method": "translation", "kind": "translation"}]
    assert entry["transforms"] == {"round1": {"detection": [], "bg_corrected": []},
                                   "round2": {"detection": translation, "bg_corrected": translation}}
    header = json.loads((fov.paths.checkpoint_dir / "registered" / "transforms.json").read_text())
    assert header["preprocessing"] == entry
    assert header["transforms"]["round2"][0]["transform"]["displacement_zyx"] == [0, *SHIFT_YX]
