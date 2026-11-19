"""What a FOV's registration attempts and an elastix failure keep (W-861).

A TPS attempt of ``FOV.register`` keeps the diagnostics' ``landmarks`` (W-836) as a B-spline attempt keeps its
``support_mask`` (W-824), also after a save and a reload. An elastix failure without a support mask raises
``RegistrationEstimationError`` with the Description lines of elastix's log, which a temporary folder used to delete
unread. The landmark fixture is uint8 16x32x32 with 27 isolated peaks on a 3x3x3 grid and 1024 noise voxels; the
elastix fixture is float32 8x64x64.
"""
import importlib.util

import numpy as np
import pytest

from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.image import ImageMetadata
from starfinder.registration import (AffineConfig, RegistrationEstimationError, RegistrationRecipe, RegistrationStep,
                                     TpsConfig, TranslationConfig, estimate_transform)
from starfinder.spot_finding import SpotFindingWarning

pytestmark = pytest.mark.registration

SHAPE = (16, 32, 32)
META = ImageMetadata("attempt-records")
GRID = [(z, y, x) for z in (4, 8, 12) for y in (6, 14, 22) for x in (6, 14, 22)]
NOISE = np.repeat(np.arange(33, 49), 64)
TPS = TpsConfig(min_matches=4, max_control_points=20)
needs_elastix = pytest.mark.skipif(importlib.util.find_spec("itk") is None,
                                   reason="requires the registration-elastix extra")
# SWIG wrappers of itk warn when pytest inspects their builtin types.
itk_warning = pytest.mark.filterwarnings("ignore:builtin type .* has no __module__ attribute:DeprecationWarning")


def truncated_image():
    """1024 noise voxels at seeded random voxels off the grid, 0 elsewhere, and a peak of 200 on each grid voxel."""
    flat = np.zeros(int(np.prod(SHAPE)), dtype=np.float64)
    grid = np.ravel_multi_index(np.array(GRID).T, SHAPE)
    free = np.setdiff1d(np.arange(flat.size), grid)
    flat[np.random.default_rng(861).permutation(free)[:len(NOISE)]] = NOISE
    flat[grid] = 200
    return flat.reshape(SHAPE).astype(np.uint8)[..., None]


def landmark_fov(tmp_path):
    dataset = Dataset(tmp_path, tmp_path / "out", "data", "sample", "run",
                      rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
                      channel_order=("ch00",))
    fov = dataset.fov("FOV")
    fov.images = {"round1": truncated_image(), "round2": truncated_image()}
    fov.metadata = {name: META for name in fov.images}
    return dataset, fov


def test_a_tps_attempt_keeps_the_landmarks_also_after_a_save_and_reload(tmp_path):
    dataset, fov = landmark_fov(tmp_path)
    recipe = RegistrationRecipe((RegistrationStep(TPS),))
    with pytest.warns(SpotFindingWarning):
        fov.run(PipelineConfig(registration=recipe), checkpoints=CheckpointConfig())
    diagnostics = fov.registration_results["round2"][0].diagnostics
    (attempt, application) = fov.registration_attempts["round2"]
    assert attempt["actual_method"] == "tps" and attempt["outcome"] == "succeeded"
    landmarks = diagnostics.landmarks
    assert attempt["landmarks"] == landmarks and attempt["landmarks"] is not landmarks
    assert landmarks["reference"]["noise"]["0"]["estimation"] == "above_minimum"
    assert (landmarks["reference"]["candidates"], landmarks["moving"]["candidates"]) == (27, 27)
    assert "landmarks" not in application
    # The saved record reloads with the landmarks, in the form of the other tuples-for-lists attempt fields.
    reloaded = Dataset(tmp_path, tmp_path / "out", "data", "sample", "run", dataset.rounds, ("ch00",)) \
        .fov("FOV").load_checkpoint("registered")
    assert reloaded.registration_attempts == fov.registration_attempts
    assert reloaded.registration_attempts["round2"][0]["landmarks"] == landmarks


def test_a_translation_attempt_has_no_landmarks_key(tmp_path):
    _, fov = landmark_fov(tmp_path)
    fov.register(RegistrationRecipe((RegistrationStep(TranslationConfig()),)))
    assert all("landmarks" not in attempt for attempt in fov.registration_attempts["round2"])


def sparse(shape):
    image = np.zeros(shape, dtype=np.float32)
    image[(0,) * len(shape)] = 1
    return image


@pytest.mark.slow
@needs_elastix
@itk_warning
def test_an_unmasked_elastix_failure_raises_with_the_log_description():
    """Two near-empty 8x64x64 images: NCC finds too few samples inside the moving buffer (270 of 4096 map outside).

    At the base the message only ended "Internal elastix error: See elastix log (use LogToConsoleOn() or
    LogToFileOn())": the log was deleted unread.
    """
    reference, moving = sparse((8, 64, 64)) * 3, sparse((8, 64, 64))
    moving[-1, -1, -1] = 1
    with pytest.raises(RegistrationEstimationError) as caught:
        estimate_transform(reference, moving, config=AffineConfig(), reference_metadata=META, moving_metadata=META)
    message = str(caught.value)
    assert message.startswith("affine estimation failed: ")
    assert "Too many samples map outside moving image buffer" in message
    assert "See elastix log" not in message
    assert type(caught.value.__cause__) is RuntimeError


@needs_elastix
@itk_warning
def test_a_failure_without_a_log_description_keeps_the_original_error(monkeypatch):
    from starfinder.registration import _elastix

    def refuse(*arguments, **keywords):
        raise RuntimeError("Internal elastix error: See elastix log")

    monkeypatch.setattr(_elastix, "_register", refuse)
    rng = np.random.default_rng(861)
    image = rng.random((8, 32, 32)).astype(np.float32)
    with pytest.raises(RegistrationEstimationError) as caught:
        estimate_transform(image, np.roll(image, 1, axis=2), config=AffineConfig(), reference_metadata=META,
                           moving_metadata=META)
    # The message of the base: estimate_transform's own wrapping of the RuntimeError, raised from it.
    assert str(caught.value) == "affine estimation failed: Internal elastix error: See elastix log"
    assert type(caught.value.__cause__) is RuntimeError
