"""The registration recipe through FOV: signals, legacy mapping, records and QC, and persistence (W-256).

Signals and persistence use the seeded golden fixture of test_registration_golden
(two uint16 rounds, 8x32x32 with four channels; its plane z=4 for Z=1). Records
and QC use small hand-built rounds.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from starfinder._registry import _version
from starfinder.dataset import (CheckpointConfig, Dataset, PipelineConfig, RecoveryConfig, RegistrationRecipe,
    RegistrationStep, RoundState, from_workflow_config)
from starfinder.image import ImageMetadata
from starfinder.io import read_checkpoint
from starfinder.io._checkpoint import _atomic, write_json, write_registered_round
from starfinder.registration import (AffineConfig, BSplineConfig, DemonsConfig, InsufficientLandmarksError,
    RegistrationEstimationError, RegistrationQcConfig, RegistrationRejectedError, RegistrationSignalConfig,
    RigidConfig, TpsConfig, TranslationConfig, TranslationTransform, WarpConfig, apply_transform,
    estimate_transform)
from starfinder.registration._chain import transform_kind

from .test_registration_golden import fixture_rounds

pytest.importorskip("SimpleITK")

CHANNELS = ("ch00", "ch01", "ch02", "ch03")
TRANSLATION = RegistrationStep(TranslationConfig())


def golden_fov(tmp_path, z1=False):
    dataset = Dataset(tmp_path, tmp_path / "out", "recipe", "sample", "out",
                      rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
                      channel_order=CHANNELS)
    fov = dataset.fov("FOV_001")
    for name, volume in fixture_rounds().items():
        fov.images[name] = volume[4:5].copy() if z1 else volume
        fov.metadata[name] = ImageMetadata(f"FOV_001/{name}")
    return fov


@pytest.fixture
def estimates(monkeypatch):
    """Every estimate_transform call: (method, reference signal, moving signal); estimation still runs."""
    import starfinder.registration as registration
    captured = []
    estimate = registration.estimate_transform

    def spy(reference, moving, **kwargs):
        captured.append((kwargs["config"].method, np.array(reference), np.array(moving)))
        return estimate(reference, moving, **kwargs)

    monkeypatch.setattr(registration, "estimate_transform", spy)
    return captured


# --- Signal ----------------------------------------------------------------------------------

@pytest.mark.parametrize("signal, reference, moving", [
    (None, lambda r: r.max(axis=-1), lambda m: m.max(axis=-1)),
    (RegistrationSignalConfig("max"), lambda r: r.max(axis=-1), lambda m: m.max(axis=-1)),
    (RegistrationSignalConfig("sum"), lambda r: r.sum(axis=-1, dtype=np.float64), lambda m: m.sum(axis=-1, dtype=np.float64)),
    (RegistrationSignalConfig("channel", 2), lambda r: r[..., 2], lambda m: m[..., 2]),
    (RegistrationSignalConfig("channel", "ch02"), lambda r: r[..., 2], lambda m: m[..., 2]),
    (RegistrationSignalConfig("channel", 2, "ch01"), lambda r: r[..., 2], lambda m: m[..., 1]),
], ids=["default", "max", "sum", "channel-index", "channel-label", "channel-per-round"])
def test_the_signal_follows_the_contract(tmp_path, estimates, signal, reference, moving):
    recipe = RegistrationRecipe((TRANSLATION,)) if signal is None else RegistrationRecipe((TRANSLATION,), signal=signal)
    rounds = fixture_rounds()
    golden_fov(tmp_path).run(PipelineConfig(registration=recipe))
    ((method, captured_reference, captured_moving),) = estimates
    assert method == "translation" and captured_reference.dtype == captured_moving.dtype == np.float64
    np.testing.assert_array_equal(captured_reference, reference(rounds["round1"]).astype(np.float64))
    np.testing.assert_array_equal(captured_moving, moving(rounds["round2"]).astype(np.float64))


def test_later_steps_see_the_moving_signal_resampled_in_float64(tmp_path, estimates):
    """Step 2's moving signal is the round's signal resampled through step 1 without rounding; a step may override the signal."""
    rounds = fixture_rounds()
    local = RegistrationStep(DemonsConfig(iterations=(2,)), signal=RegistrationSignalConfig("channel", "ch00"))
    fov = golden_fov(tmp_path).run(PipelineConfig(registration=RegistrationRecipe((TRANSLATION, local))))
    (_, reference_1, moving_1), (method, reference_2, moving_2) = estimates
    first = fov.registration_results["round2"][0].transform
    np.testing.assert_array_equal(reference_1, rounds["round1"].max(axis=-1).astype(np.float64))
    np.testing.assert_array_equal(reference_2, rounds["round1"][..., 0].astype(np.float64))
    expected = apply_transform(rounds["round2"][..., 0].astype(np.float64), first, config=WarpConfig(output_dtype="float64"))
    assert method == "demons" and moving_2.dtype == np.float64
    np.testing.assert_array_equal(moving_2, expected)


@pytest.mark.parametrize("signal, message", [
    (RegistrationSignalConfig("channel", "ch09"), "'ch09' is not a channel label"),
    (RegistrationSignalConfig("channel", 4), "channel 4 is outside the image"),
    (RegistrationSignalConfig("channel", 0, "dapi"), "'dapi' is not a channel label"),
])
def test_unknown_channels_fail_before_estimation(tmp_path, estimates, signal, message):
    with pytest.raises(ValueError, match=message):
        golden_fov(tmp_path).run(PipelineConfig(registration=RegistrationRecipe((TRANSLATION,), signal=signal)))
    assert estimates == []


@pytest.mark.parametrize("values", [dict(mode="mean"), dict(mode="channel"), dict(mode="max", reference_channel=0),
                                    dict(mode="channel", reference_channel=-1),
                                    dict(mode="channel", reference_channel=True)])
def test_signal_config_validation(values):
    with pytest.raises(ValueError):
        RegistrationSignalConfig(**values)


def workflow_config(tmp_path, **parameters):
    return dict(root_input_path=str(tmp_path), root_output_path=str(tmp_path / "out"), dataset_id="data",
                sample_id="sample", output_id="run", n_rounds=2, ref_round="round1", seq_channel_order=list(CHANNELS),
                rotate_angle=0, img_row=32, img_col=32, rules={"rsf_single_fov": {"parameters": parameters}})


@pytest.mark.parametrize("parameters, signals", [
    (dict(global_registration={"run": True}), [RegistrationSignalConfig("max")]),
    (dict(global_registration={"run": True, "ref_img": "merged-image", "mov_img": "merged-image"}),
     [RegistrationSignalConfig("max")]),
    (dict(global_registration={"run": True, "ref_img": "merged", "mov_img": "merged"}), [RegistrationSignalConfig("max")]),
    (dict(global_registration={"run": True, "ref_img": "single-channel", "mov_img": "single-channel", "ref_channel": 2}),
     [RegistrationSignalConfig("channel", 2)]),
    # The local default is merged-image (the maximum), not single-channel.
    (dict(local_registration={"run": True, "method": "demons", "iterations": [2], "ref_channel": 1}),
     [RegistrationSignalConfig("max")]),
    # Different signals in the two blocks: the first is the recipe's, the second the step's own.
    (dict(global_registration={"run": True},
          local_registration={"run": True, "iterations": [2], "ref_img": "single-channel", "mov_img": "single-channel",
                              "ref_channel": 1}),
     [RegistrationSignalConfig("max"), RegistrationSignalConfig("channel", 1)]),
], ids=["global-default", "merged-image", "merged", "single-channel", "local-default", "two-blocks"])
def test_legacy_signals_map_to_the_recipe_and_reach_the_estimator(tmp_path, estimates, parameters, signals):
    recipe = from_workflow_config(workflow_config(tmp_path, **parameters)).pipeline.registration
    assert recipe.signal == signals[0]
    assert [step.signal or recipe.signal for step in recipe.steps] == signals
    assert [step.signal for step in recipe.steps[1:]] == signals[1:]
    rounds = fixture_rounds()
    golden_fov(tmp_path).run(PipelineConfig(registration=recipe))
    (_, reference, moving), *_ = estimates
    channel = signals[0].reference_channel
    expected = rounds["round1"].max(axis=-1) if channel is None else rounds["round1"][..., channel]
    np.testing.assert_array_equal(reference, expected.astype(np.float64))
    expected = rounds["round2"].max(axis=-1) if channel is None else rounds["round2"][..., channel]
    np.testing.assert_array_equal(moving, expected.astype(np.float64))


@pytest.mark.parametrize("parameters, message", [
    (dict(global_registration={"run": True, "ref_img": "merged-image", "mov_img": "single-channel"}),
     "ref_img 'merged-image' and mov_img 'single-channel' differ"),
    (dict(local_registration={"run": True, "ref_img": "single-channel", "mov_img": "merged"}),
     "ref_img 'single-channel' and mov_img 'merged' differ"),
    (dict(global_registration={"run": True, "boundary_mode": "constant"},
          local_registration={"run": True, "boundary_mode": "nearest"}),
     r"boundary_mode differs between the registration blocks: \['constant', 'nearest'\]"),
    (dict(global_registration={"run": True, "method": "demons"}), "demons is not a global registration method"),
    (dict(local_registration={"run": True, "method": "affine"}), "affine is not a local registration method"),
    (dict(local_registration={"run": True, "method": "tps", "recovery": {
        "allowed_errors": ["InsufficientLandmarksError"], "alternatives": [{"method": "translation"}]}}),
     "translation is not a local registration method"),
])
def test_legacy_mixed_modes_conflicts_and_kinds_are_rejected(tmp_path, parameters, message):
    with pytest.raises(ValueError, match=message):
        from_workflow_config(workflow_config(tmp_path, **parameters))


def test_legacy_boundary_and_recovery_map_to_the_recipe(tmp_path):
    recipe = from_workflow_config(workflow_config(tmp_path, local_registration={
        "run": True, "method": "tps", "boundary_mode": "nearest", "recovery": {
            "allowed_errors": ["InsufficientLandmarksError", "RegistrationRejectedError"],
            "alternatives": [{"method": "fast_symmetric", "iterations": [3]}]}})).pipeline.registration
    assert recipe.warp == WarpConfig(backend="scipy", boundary_mode="nearest")
    (step,) = recipe.steps
    assert step.recovery == RecoveryConfig((InsufficientLandmarksError, RegistrationRejectedError),
                                           (DemonsConfig(variant="fast_symmetric", iterations=(3,)),))
    assert from_workflow_config(workflow_config(tmp_path, global_registration={"run": False})).pipeline.registration is None


# --- Records and QC --------------------------------------------------------------------------

def small_fov(tmp_path, reference, moving):
    dataset = Dataset(tmp_path, tmp_path / "out", "records", "sample", "out",
                      rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
                      channel_order=CHANNELS[:reference.shape[-1]])
    fov = dataset.fov("FOV")
    fov.images = {"round1": reference, "round2": moving}
    fov.metadata = {name: ImageMetadata("common") for name in fov.images}
    return fov


def sparse_rounds():
    """Two bright voxels on zeros: too few landmarks for TPS."""
    image = np.zeros((4, 12, 14, 4), dtype=np.uint16)
    image[1, 4, 5, 0] = 60000
    image[2, 8, 9, 0] = 30000
    return image, image.copy()


def shifted_rounds():
    """Random texture; the moving round is rolled by five voxels along X, so the valid overlap is 27/32."""
    rng = np.random.default_rng(2560)
    reference = rng.integers(100, 4000, size=(4, 32, 32, 2), dtype=np.uint16)
    return reference, np.roll(reference, 5, axis=2)


def test_estimation_and_application_records_of_a_recovered_step(tmp_path):
    fov = small_fov(tmp_path, *sparse_rounds())
    recovery = RecoveryConfig((InsufficientLandmarksError,), (DemonsConfig(iterations=(1,)),))
    fov.register(RegistrationRecipe((RegistrationStep(TpsConfig(), recovery=recovery),)))
    failed, fallback, application = fov.registration_attempts["round2"]
    keys = ("record", "step", "attempt", "requested_method", "actual_method", "fallback", "backend",
            "backend_versions", "reference", "outcome", "qc")
    assert {k: failed[k] for k in keys} == dict(
        record="estimation", step=0, attempt=0, requested_method="tps", actual_method="tps", fallback=False,
        backend=None, backend_versions={}, reference="round1", outcome="failed", qc=None)
    assert failed["config"] == asdict(TpsConfig())
    assert failed["failure"]["type"] == "InsufficientLandmarksError"
    assert {k: fallback[k] for k in keys if k != "qc"} == dict(
        record="estimation", step=0, attempt=1, requested_method="tps", actual_method="demons", fallback=True,
        backend="simpleitk", backend_versions={"SimpleITK": _version("SimpleITK")}, reference="round1",
        outcome="succeeded")
    assert fallback["failure"] is None and fallback["config"] == asdict(DemonsConfig(iterations=(1,)))
    qc = fallback["qc"]
    assert set(qc["values"]) == {"coverage", "ncc_before", "ncc_after", "ncc_gain", "ssim_before", "ssim_after"}
    assert qc["config"]["qc"] == asdict(RegistrationQcConfig())
    assert qc["details"]["transform"]["kind"] == "dense"
    assert qc["details"]["optimizer"]["method"] == "demons"
    assert {k: application[k] for k in ("record", "outcome", "application_config", "failure")} == dict(
        record="application", outcome="succeeded", application_config=asdict(WarpConfig(backend="scipy")), failure=None)
    assert application["qc"]["details"]["transform"]["kind"] == "dense"
    assert fov.registration_results["round2"][0].diagnostics.method == "demons"


def test_the_default_qc_config_rejects_nothing(tmp_path):
    fov = small_fov(tmp_path, *shifted_rounds())
    fov.register(RegistrationRecipe((TRANSLATION,)))
    estimation, application = fov.registration_attempts["round2"]
    assert fov.registration_results["round2"][0].transform.correction_zyx == (0.0, 0.0, -5.0)
    assert estimation["outcome"] == "succeeded" and estimation["qc"]["values"]["coverage"] == 27 / 32
    assert estimation["qc"]["counts"] == {"total": 4 * 32 * 32, "valid": 4 * 32 * 27, "valid_columns": 32 * 27,
                                          "ssim_columns": estimation["qc"]["counts"]["ssim_columns"]}
    assert application["qc"]["values"]["coverage"] == 27 / 32


def test_a_failed_criterion_is_rejected_and_recovered_only_when_allowed(tmp_path):
    qc = RegistrationQcConfig(min_coverage=0.99)
    fov = small_fov(tmp_path, *shifted_rounds())
    with pytest.raises(RegistrationRejectedError, match=r"min_coverage: 0\.84375 is below the bound 0\.99"):
        fov.register(RegistrationRecipe((TRANSLATION,), qc=qc))
    (rejected,) = fov.registration_attempts["round2"]
    assert rejected["outcome"] == "rejected" and rejected["qc"]["values"]["coverage"] == 27 / 32
    assert rejected["failure"] == {"type": "RegistrationRejectedError", "criterion": "min_coverage",
                                   "message": "min_coverage: 0.84375 is below the bound 0.99"}
    assert not fov.registration_results and "round2" not in fov.registration_chains
    alternative = (TranslationConfig(backend="skimage"),)
    # Recovery that does not allow the rejection: one attempt.
    fov = small_fov(tmp_path, *shifted_rounds())
    step = RegistrationStep(TranslationConfig(), recovery=RecoveryConfig((InsufficientLandmarksError,), alternative))
    with pytest.raises(RegistrationRejectedError):
        fov.register(RegistrationRecipe((step,), qc=qc))
    assert [a["outcome"] for a in fov.registration_attempts["round2"]] == ["rejected"]
    # Recovery that allows it (as an estimation error or by name): the alternative runs and is rejected too.
    for allowed in (RegistrationEstimationError, RegistrationRejectedError):
        fov = small_fov(tmp_path, *shifted_rounds())
        step = RegistrationStep(TranslationConfig(), recovery=RecoveryConfig((allowed,), alternative))
        with pytest.raises(RegistrationRejectedError, match="min_coverage"):
            fov.register(RegistrationRecipe((step,), qc=qc))
        attempts = fov.registration_attempts["round2"]
        assert [(a["attempt"], a["fallback"], a["actual_method"], a["backend"], a["outcome"]) for a in attempts] == [
            (0, False, "translation", "scipy_fft", "rejected"), (1, True, "translation", "skimage", "rejected")]
    fov = small_fov(tmp_path, *shifted_rounds())
    with pytest.raises(RegistrationRejectedError, match=r"max_translation_voxels: 5\.0 is above the bound 4"):
        fov.register(RegistrationRecipe((TRANSLATION,), qc=RegistrationQcConfig(max_translation_voxels=4)))


def test_application_failures_are_recorded_and_never_recover(tmp_path, monkeypatch):
    import starfinder.registration._chain as chain_module
    calls = []
    original = chain_module.resample

    def fail(images, chain, config):
        calls.append(config)
        if config.output_dtype == "input":
            raise MemoryError("injected application failure")
        return original(images, chain, config)

    monkeypatch.setattr(chain_module, "resample", fail)
    fov = small_fov(tmp_path, *shifted_rounds())
    step = RegistrationStep(TranslationConfig(), recovery=RecoveryConfig((RegistrationEstimationError,),
                                                                          (TranslationConfig(backend="skimage"),)))
    with pytest.raises(MemoryError):
        fov.register(RegistrationRecipe((step,)))
    estimation, application = fov.registration_attempts["round2"]
    assert estimation["outcome"] == "succeeded"
    assert application["outcome"] == "application_failed"
    assert application["failure"] == {"type": "MemoryError", "message": "injected application failure"}
    assert not fov.registration_results


def test_a_round_is_registered_once(tmp_path):
    fov = small_fov(tmp_path, *shifted_rounds())
    fov.register(RegistrationRecipe((TRANSLATION,)))
    with pytest.raises(ValueError, match="already registered"):
        fov.register(RegistrationRecipe((TRANSLATION,)))
    with pytest.raises(TypeError, match="requires a RegistrationRecipe"):
        fov.register(TRANSLATION)
    with pytest.raises(ValueError, match="differs from the dataset reference"):
        small_fov(tmp_path, *shifted_rounds()).run(PipelineConfig(
            registration=RegistrationRecipe((TRANSLATION,), reference_round="round2")))


# --- Persistence -----------------------------------------------------------------------------

RECIPES = {
    "translation": ((TranslationConfig(),), ["translation"]),
    "affine": ((AffineConfig(),), ["affine"]),
    "bspline": ((BSplineConfig(),), ["bspline"]),
    "dense": ((DemonsConfig(iterations=(20, 10)),), ["dense"]),
    "chain": ((TranslationConfig(), RigidConfig(), BSplineConfig()), ["translation", "affine", "bspline"]),
}


@pytest.mark.parametrize("z1", [False, True], ids=["8x32x32", "1x32x32"])
@pytest.mark.parametrize("kind", list(RECIPES))
def test_registered_checkpoint_reapplies_identically(tmp_path, kind, z1):
    pytest.importorskip("itk")
    configs, kinds = RECIPES[kind]
    fov = golden_fov(tmp_path, z1)
    before = {name: image.copy() for name, image in fov.images.items()}
    fov.run(PipelineConfig(registration=RegistrationRecipe(tuple(map(RegistrationStep, configs)))),
            checkpoints=CheckpointConfig(stages=("registered",)))
    header = json.loads((fov.paths.checkpoint_dir / "registered" / "transforms.json").read_text())
    assert header["format_version"] == 2 and header["registration_semantics"] == "recipe"
    assert header["registration_recipe"]["steps"] == [c.method for c in configs]
    assert [entry["transform"]["kind"] for entry in header["transforms"]["round2"]] == kinds
    assert [entry["step"] for entry in header["transforms"]["round2"]] == list(range(len(configs)))
    reloaded = fov.dataset.fov("FOV_001").load_checkpoint("registered")
    chain, original = reloaded.registration_chains["round2"], fov.registration_chains["round2"]
    assert [transform_kind(t) for t in chain.transforms] == kinds
    np.testing.assert_array_equal(chain.pull_field().displacement_zyx, original.pull_field().displacement_zyx)
    warp = reloaded.registration_record["application"]["round2"]
    assert warp == fov.registration_record["application"]["round2"]
    assert warp == (WarpConfig() if kind == "translation" else WarpConfig(backend="scipy"))
    reapplied = apply_transform(before["round2"], chain, config=warp)
    assert reapplied.dtype == fov.images["round2"].dtype
    np.testing.assert_array_equal(reapplied, fov.images["round2"])
    np.testing.assert_array_equal(reloaded.images["round2"], fov.images["round2"])
    assert reloaded.registration_record["semantics"] == "recipe"
    assert [r.diagnostics for r in reloaded.registration_results["round2"]] == [
        r.diagnostics for r in fov.registration_results["round2"]]
    assert [r.application_config for r in reloaded.registration_results["round2"]] == [warp] * len(configs)
    assert reloaded.registration_attempts == fov.registration_attempts


def write_version_1(directory, header, registration_results):
    """The registered header writer of the start revision (5f828e2), copied: version 1, per-result application_config."""
    directory = Path(directory) / "registered"
    transforms = {}
    for name, results in registration_results.items():
        dense = {f"result_{i}": r.transform.displacement_zyx for i, r in enumerate(results)
                 if not isinstance(r.transform, TranslationTransform)}
        field_name = f"{name}_field.npz"
        if dense:
            with _atomic(directory / field_name) as tmp:
                with open(tmp, "wb") as handle:
                    np.savez(handle, **dense)
        entries = []
        for r in results:
            transform = r.transform
            data = {key: getattr(transform, key) for key in ("reference_shape_zyx", "moving_shape_zyx",
                    "reference_metadata", "moving_metadata", "direction", "units")}
            if isinstance(transform, TranslationTransform):
                data.update(kind="translation", correction_zyx=transform.correction_zyx)
            else:
                data.update(kind="dense", field=field_name)
            entries.append(dict(transform=data, diagnostics=r.diagnostics, application_config=r.application_config))
        transforms[name] = entries
    write_json(dict(header, stage="registered", format_version=1, transforms=transforms), directory / "transforms.json")


def test_version_1_registered_checkpoints_still_load_as_sequential(tmp_path):
    fov = golden_fov(tmp_path)
    reference, moving = (fov.images[name].sum(axis=-1, dtype=np.float64) for name in ("round1", "round2"))
    geometry = dict(reference_metadata=fov.metadata["round1"], moving_metadata=fov.metadata["round2"])
    translation = estimate_transform(reference, moving, config=TranslationConfig(), **geometry)
    shifted = apply_transform(moving, translation.transform, config=WarpConfig(output_dtype="float64"))
    dense = estimate_transform(reference, shifted, config=DemonsConfig(iterations=(3,)), **geometry)
    results = {"round2": [translation, dense]}
    attempts = {"round2": [dict(requested_method=r.diagnostics.method, actual_method=r.diagnostics.method,
                                config=asdict(r.diagnostics.effective_config), failure=None, outcome="succeeded",
                                application_config=asdict(r.application_config)) for r in (translation, dense)]}
    directory = fov.paths.checkpoint_dir
    for name, image in fov.images.items():
        write_registered_round(directory, name, image, fov.metadata[name])
    header = dict(fov._checkpoint_header(), image_rounds=["round1", "round2"], snapshots=[],
                  registration_attempts=attempts, preprocessing=None)
    write_version_1(directory, header, results)
    loaded = read_checkpoint(directory, "registered")
    restored = loaded["registration_results"]["round2"]
    assert restored[0] == translation
    np.testing.assert_array_equal(restored[1].transform.displacement_zyx, dense.transform.displacement_zyx)
    assert restored[1].transform.displacement_zyx.dtype == dense.transform.displacement_zyx.dtype
    assert restored[1].diagnostics == dense.diagnostics
    assert [r.application_config for r in restored] == [WarpConfig(), WarpConfig(backend="simpleitk")]
    assert loaded["registration_record"] == {"semantics": "sequential", "recipe": None, "application": {}}
    assert loaded["registration_chains"] == {}
    assert loaded["registration_attempts"]["round2"][1]["application_config"]["backend"] == "simpleitk"
    reloaded = fov.dataset.fov("FOV_001").load_checkpoint("registered")
    assert reloaded.registration_record["semantics"] == "sequential"
    np.testing.assert_array_equal(reloaded.images["round2"], fov.images["round2"])


def test_version_1_candidates_and_pre_qc_headers_still_load(tmp_path):
    from starfinder.barcode import Codebook, NeighborhoodSumConfig, ReadFilterConfig, WtaDecoderConfig
    from starfinder.spot_finding import LocalMaximaConfig
    import pandas as pd
    fov = golden_fov(tmp_path)
    fov.dataset.codebook = Codebook(pd.DataFrame({"gene_id": ["gene"], "color_sequence": ["11"]}),
                                    ("round1", "round2"), CHANNELS)
    fov.run(PipelineConfig(registration=RegistrationRecipe((TRANSLATION,)),
                           detection=LocalMaximaConfig("adaptive", .1), extraction=NeighborhoodSumConfig((0, 1, 1)),
                           decoding=WtaDecoderConfig(), filtering=ReadFilterConfig()),
            checkpoints=CheckpointConfig())
    directory = fov.paths.checkpoint_dir
    for stage in ("candidates", "pre_qc"):
        path = directory / f"{stage}.json"
        current = read_checkpoint(directory, stage)
        header = json.loads(path.read_text())
        assert header["format_version"] == 2
        path.write_text(json.dumps(dict(header, format_version=1)))
        again = read_checkpoint(directory, stage)
        assert set(again) == set(current)
        path.write_text(json.dumps(dict(header, format_version=3)))
        with pytest.raises(ValueError, match="is not a version 1 or 2"):
            read_checkpoint(directory, stage)
