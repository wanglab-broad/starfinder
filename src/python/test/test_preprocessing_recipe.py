"""Step/recipe contract (W-229): registry, enforcement wrapper, recipe runner and legacy mapping.

Uses the golden fixture of test_preprocessing_golden (two uint16 rounds of
4x32x32 voxels and four channels) and its pinned FOV.run digests.
"""
import copy
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import yaml

from starfinder.dataset import CheckpointConfig, Dataset, ExecutionConfig, PipelineConfig, RoundState, from_workflow_config
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadConfig, save_volume
from starfinder.preprocessing import (STEPS, HistogramMatchingConfig, MinMaxNormalizationConfig,
    PercentileNormalizationConfig, PreprocessingRecipe, RecipeStep, ReconstructionConfig, StepContext, StepResult, StepSpec, TophatConfig, run_step, step_config_type,
    step_spec)

from .test_preprocessing_golden import PINNED_SEQUENCE, digest, fixture_rounds

ROOT = Path(__file__).resolve().parents[3]
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
RECIPE_1 = PreprocessingRecipe((RecipeStep(MinMaxNormalizationConfig("uint8", (0, 255))),
                                RecipeStep(HistogramMatchingConfig(reference_channel=0)),
                                RecipeStep(ReconstructionConfig(radius_yx=3))))


def golden_dataset(tmp_path):
    return Dataset(tmp_path, tmp_path / "out", "golden", "sample", "out",
                   rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
                   channel_order=CHANNELS)


def resident_fov(tmp_path, rounds=None):
    fov = golden_dataset(tmp_path).fov("FOV_001")
    for name, volume in (rounds or fixture_rounds()).items():
        fov.images[name] = volume.copy()
        fov.metadata[name] = ImageMetadata(f"FOV_001/{name}")
    return fov


def workflow_config(tmp_path, rule="rsf_single_fov", **parameters):
    parameters.setdefault("load_raw_images", {"run": False})
    return dict(root_input_path=str(tmp_path), root_output_path=str(tmp_path / "out"), dataset_id="golden",
                sample_id="sample", output_id="out", n_rounds=2, ref_round="round1",
                seq_channel_order=list(CHANNELS), img_row=32, img_col=32,
                rules={rule: {"parameters": parameters}})


def recipe_configs(recipe):
    return ([s.config for s in recipe.steps], [s.config for s in recipe.post_registration])


# --- Registry -------------------------------------------------------------------

def test_registered_steps_match_the_contract_table():
    table = {cls: (spec.name, spec.category, spec.scope, spec.dtype_policy) for cls, spec in STEPS.items()}
    assert table == {
        MinMaxNormalizationConfig: ("min_max_normalization", "intensity", "per_channel", "declared"),
        HistogramMatchingConfig: ("histogram_matching", "intensity", "needs_reference", "preserve"),
        ReconstructionConfig: ("reconstruction", "background", "per_channel", "preserve"),
        TophatConfig: ("white_tophat", "background", "per_channel", "preserve"),
        PercentileNormalizationConfig: ("percentile_normalization", "intensity", "per_channel", "preserve")}


def test_step_names_are_unique_and_name_lookup_is_derived_from_steps(monkeypatch):
    names = [spec.name for spec in STEPS.values()]
    assert len(set(names)) == len(names)
    for config_type, spec in STEPS.items():
        assert step_config_type(spec.name) is config_type
    with pytest.raises(ValueError, match="unknown"):
        step_config_type("no_such_step")

    @dataclass(frozen=True)
    class NewConfig:
        pass

    # A step registered only in STEPS is found by name: there is no second list.
    monkeypatch.setitem(STEPS, NewConfig, StepSpec("new_step", lambda v, c, x: StepResult(v, {}, {}), "contrast", "per_round"))
    assert step_config_type("new_step") is NewConfig
    monkeypatch.setitem(STEPS, NewConfig, StepSpec("white_tophat", lambda v, c, x: StepResult(v, {}, {}), "contrast", "per_round"))
    with pytest.raises(ValueError, match="more than once"):
        step_config_type("white_tophat")


def test_step_spec_validates_its_declarations():
    run = lambda v, c, x: StepResult(v, {}, {})
    for bad in (dict(name="WhiteTophat"), dict(category="colour"), dict(scope="per_fov"), dict(dtype_policy="any")):
        with pytest.raises(ValueError):
            StepSpec(**{**dict(name="ok_step", run=run, category="background", scope="per_channel"), **bad})


def test_lookup_uses_the_exact_config_type():
    @dataclass(frozen=True)
    class SubTophat(TophatConfig):
        pass

    volume = fixture_rounds()["round1"]
    context = StepContext("round1", "round1", ImageMetadata("frame"))
    assert step_spec(TophatConfig()).name == "white_tophat"
    for call in (lambda: step_spec(SubTophat()), lambda: run_step(volume, SubTophat(), context),
                 lambda: RecipeStep(SubTophat()), lambda: RecipeStep("white_tophat")):
        with pytest.raises(TypeError, match="exact config type"):
            call()


# --- Enforcement wrapper ---------------------------------------------------------

@dataclass(frozen=True)
class FaultyConfig:
    fault: str

    def __post_init__(self):
        pass


def _faulty(volume, config, context):
    image, fitted = volume.copy(), {}
    if config.fault == "shape":
        image = image[:-1]
    elif config.fault == "dtype":
        image = image.astype(np.float32)
    elif config.fault == "nonfinite":
        image = image.astype(np.float64)
        image[0, 0, 0] = np.nan
    elif config.fault == "metadata":
        object.__setattr__(context.metadata, "frame_id", "changed")
    elif config.fault == "fitted":
        fitted = {"array": np.zeros(2)}
    elif config.fault == "nan_diagnostic":
        return StepResult(image, {}, {"ratio": float("nan")})
    return StepResult(image, fitted, {})


@pytest.fixture
def faulty_step(monkeypatch):
    monkeypatch.setitem(STEPS, FaultyConfig, StepSpec("faulty_step", _faulty, "contrast", "per_channel"))


@pytest.mark.parametrize("fault, message", [
    ("shape", "changed the shape"), ("dtype", "returned dtype float32, expected uint16 \\(preserve\\)"),
    ("metadata", "modified the image metadata"), ("fitted", "returned fitted values that are not JSON-serializable"),
    ("nan_diagnostic", "returned diagnostics that are not JSON-serializable")])
def test_wrapper_rejects_faulty_steps(faulty_step, fault, message):
    volume = fixture_rounds()["round1"]
    context = StepContext("round1", "round1", ImageMetadata("frame"))
    with pytest.raises(ValueError, match=f"step 'faulty_step' {message}"):
        run_step(volume, FaultyConfig(fault), context)


def test_wrapper_rejects_nonfinite_output(faulty_step):
    volume = fixture_rounds()["round1"].astype(np.float64)
    context = StepContext("round1", "round1", ImageMetadata("frame"))
    with pytest.raises(ValueError, match="step 'faulty_step' returned nonfinite"):
        run_step(volume, FaultyConfig("nonfinite"), context)


def test_wrapper_declared_dtype_and_reference_requirements():
    volume = fixture_rounds()["round1"]
    context = StepContext("round1", "round1", ImageMetadata("frame"))
    result = run_step(volume, MinMaxNormalizationConfig("uint8", (0, 255)), context)
    assert result.image.dtype == np.uint8 and len(result.fitted["groups"]) == 4
    with pytest.raises(ValueError, match="requires a reference"):
        run_step(volume, HistogramMatchingConfig(), context)
    with pytest.raises(ValueError, match="does not take a reference"):
        run_step(volume, TophatConfig(), replace(context, reference=volume[..., 0]))
    # Histogram matching preserves dtype: a declared float output violates the policy.
    with pytest.raises(ValueError, match="expected uint16 \\(preserve\\)"):
        run_step(volume, HistogramMatchingConfig(output_dtype="float32"), replace(context, reference=volume[..., 0]))


@pytest.mark.parametrize("fault", ["shape", "metadata"])
def test_run_record_names_the_failing_step(tmp_path, faulty_step, fault):
    fov = resident_fov(tmp_path)
    recipe = PreprocessingRecipe((RecipeStep(TophatConfig()), RecipeStep(FaultyConfig(fault))))
    with pytest.raises(ValueError, match="faulty_step"):
        fov.run(PipelineConfig(preprocessing=recipe), checkpoints=CheckpointConfig(stages=()))
    data = json.loads((fov.paths.checkpoint_dir / "run.json").read_text())
    assert data["status"] == "failed"
    assert (data["error"]["step"], data["error"]["round"]) == ("preprocess:faulty_step", "round1")
    assert [r["step"] for r in data["preprocessing"]["rounds"]["round1"]] == ["white_tophat"]


# --- Recipe validation ----------------------------------------------------------

def test_post_registration_accepts_only_reconstruction():
    PreprocessingRecipe(post_registration=(RecipeStep(ReconstructionConfig()),))
    for config in (TophatConfig(), MinMaxNormalizationConfig("uint8", (0, 255)), HistogramMatchingConfig()):
        with pytest.raises(ValueError, match="post_registration may contain only ReconstructionConfig"):
            PreprocessingRecipe(post_registration=(RecipeStep(config),))
    with pytest.raises(ValueError, match="post_registration"):
        PipelineConfig(preprocessing=PreprocessingRecipe(post_registration=[RecipeStep(TophatConfig())]))
    with pytest.raises(TypeError):
        PreprocessingRecipe((TophatConfig(),))
    with pytest.raises(ValueError, match="reference_channel"):
        HistogramMatchingConfig(reference_channel=-1)


def test_pipeline_config_has_no_preprocessing_slots_or_projection():
    removed = {"normalization", "histogram", "histogram_reference_channel", "reconstruction",
               "reconstruction_after_registration", "tophat", "projection"}
    names = {f.name for f in fields(PipelineConfig)}
    assert "preprocessing" in names and not removed & names
    for name in removed:
        with pytest.raises(TypeError):
            PipelineConfig(**{name: None})
    migration = (ROOT / "docs/migration.md").read_text()
    section = migration.split("### Preprocessing recipe", 1)[1].split("\n## ", 1)[0]
    for name in removed:
        assert f"`PipelineConfig.{name}`" in section


# --- Runner ----------------------------------------------------------------------

def test_run_record_and_checkpoint_store_per_round_step_records(tmp_path):
    fov = resident_fov(tmp_path)
    fov.run(PipelineConfig(preprocessing=RECIPE_1), checkpoints=CheckpointConfig(stages=("registered",)))
    assert {name: digest(fov.images[name]) for name in fov.images} == PINNED_SEQUENCE
    directory = fov.paths.checkpoint_dir
    entry = json.loads((directory / "run.json").read_text())["preprocessing"]
    assert entry["recipe"] == {"steps": ["min_max_normalization", "histogram_matching", "reconstruction"],
                               "post_registration": []}
    assert set(entry["rounds"]) == {"round1", "round2"}
    for records in entry["rounds"].values():
        assert [(r["index"], r["stage"], r["step"]) for r in records] == [
            (0, "steps", "min_max_normalization"), (1, "steps", "histogram_matching"), (2, "steps", "reconstruction")]
        assert [(r["input_dtype"], r["output_dtype"]) for r in records] == [
            ("uint16", "uint8"), ("uint8", "uint8"), ("uint8", "uint8")]
        assert all(set(r) == {"index", "stage", "step", "config", "fitted", "diagnostics", "input_dtype", "output_dtype"}
                   for r in records)
        minmax, histogram, reconstruction = records
        assert minmax["config"]["output_range"] == [0, 255] and minmax["config"]["rounding"] == "truncate"
        assert [g["channel"] for g in minmax["fitted"]["groups"]] == [0, 1, 2, 3]
        assert histogram["fitted"] == {"reference_round": "round1", "reference_channel": 0}
        assert reconstruction["config"] == {"radius_yx": 3}
    raw = fixture_rounds()["round2"]
    assert entry["rounds"]["round2"][0]["fitted"]["groups"][1] == {
        "channel": 1, "min": int(raw[..., 1].min()), "max": int(raw[..., 1].max()), "mode": "rescale"}
    header = json.loads((directory / "registered" / "transforms.json").read_text())
    assert header["preprocessing"] == entry
    reloaded = golden_dataset(tmp_path).fov("FOV_001").load_checkpoint("registered")
    assert json.loads(json.dumps(reloaded.preprocessing_record)) == entry


def _save_golden_inputs(tmp_path):
    for name, volume in fixture_rounds().items():
        for c, channel in enumerate(CHANNELS):
            save_volume(volume[..., c], tmp_path / name / "FOV_001" / f"{channel}.tif",
                        metadata=ImageMetadata(f"FOV_001/{name}"))


def test_batch_and_streaming_recipe_give_identical_images(tmp_path):
    _save_golden_inputs(tmp_path)
    config = PipelineConfig(load=ImageLoadConfig(channel_labels=CHANNELS), preprocessing=RECIPE_1)
    batch = golden_dataset(tmp_path).fov("FOV_001").run(config)
    stream = golden_dataset(tmp_path).fov("FOV_001").run(config, execution=ExecutionConfig("streaming", retain_images=True))
    for name in ("round1", "round2"):
        np.testing.assert_array_equal(batch.images[name], stream.images[name])
        assert batch.images[name].dtype == stream.images[name].dtype
    assert {name: digest(stream.images[name]) for name in ("round1", "round2")} == PINNED_SEQUENCE
    assert batch.preprocessing_record == stream.preprocessing_record


def test_streaming_retains_the_histogram_reference_after_reference_processing(tmp_path):
    """Streaming keeps the reference round's input to the step for the moving rounds."""
    recipe = PreprocessingRecipe((RecipeStep(HistogramMatchingConfig(reference_channel=2)), RecipeStep(TophatConfig())))
    fov = resident_fov(tmp_path)
    reference = fov.images["round1"][..., 2].copy()
    fov.run(PipelineConfig(preprocessing=recipe), execution=ExecutionConfig("streaming"))
    from starfinder.preprocessing import filter_tophat, match_histogram
    expected = filter_tophat(match_histogram(fixture_rounds()["round2"], reference))
    assert set(fov.images) == {"round1"}
    fov = resident_fov(tmp_path)
    fov.run(PipelineConfig(preprocessing=recipe), execution=ExecutionConfig("streaming", retain_images=True))
    np.testing.assert_array_equal(fov.images["round2"], expected)
    with pytest.raises(ValueError, match="reference_channel 4 is outside"):
        resident_fov(tmp_path).run(PipelineConfig(preprocessing=PreprocessingRecipe(
            (RecipeStep(HistogramMatchingConfig(reference_channel=4)),))))


# --- Workflow adapter -------------------------------------------------------------

def test_adapter_recipe_1_reproduces_golden_digests(tmp_path):
    config = workflow_config(tmp_path, enhance_contrast={"run": True}, hist_equalize={"run": True, "reference_channel": 0},
                             morph_recon={"run": True, "radius": 3})
    adapted = from_workflow_config(config)
    assert adapted.pipeline.load is None and adapted.pipeline.preprocessing == RECIPE_1
    fov = resident_fov(tmp_path)
    fov.run(adapted.pipeline)
    assert {name: digest(fov.images[name]) for name in fov.images} == PINNED_SEQUENCE


@pytest.mark.parametrize("rule", ["rsf_single_fov", "gr_single_fov_subtile", "deep_create_subtile",
                                  "lrsf_single_fov_subtile", "deep_rsf_subtile"])
def test_each_legacy_key_maps_to_its_recipe_step(tmp_path, rule):
    resident = rule in ("lrsf_single_fov_subtile", "deep_rsf_subtile")
    single = {
        "enhance_contrast": ({"run": True, "snr_threshold": 4.0},
                             MinMaxNormalizationConfig("uint8", (0, 255), snr_threshold=4.0, rounding="truncate")),
        "hist_equalize": ({"run": True, "reference_channel": 2}, HistogramMatchingConfig(reference_channel=2)),
        "morph_recon": ({"run": True, "radius": 5}, ReconstructionConfig(radius_yx=5)),
        "tophat": ({"run": True, "radius": 2}, TophatConfig(radius_yx=2)),
    }
    for key, (values, expected) in single.items():
        recipe = from_workflow_config(workflow_config(tmp_path, rule, **{key: values}), rule).pipeline.preprocessing
        post = key == "morph_recon" and resident
        assert recipe_configs(recipe) == (([], [expected]) if post else ([expected], []))
        assert step_spec(expected).name == {"enhance_contrast": "min_max_normalization", "hist_equalize": "histogram_matching",
                                            "morph_recon": "reconstruction", "tophat": "white_tophat"}[key]
        disabled = from_workflow_config(workflow_config(tmp_path, rule, **{key: {**values, "run": False}}), rule)
        assert disabled.pipeline.preprocessing is None
    everything = {key: values for key, (values, _) in single.items()}
    steps, post = recipe_configs(from_workflow_config(workflow_config(tmp_path, rule, **everything), rule).pipeline.preprocessing)
    expected = [single[k][1] for k in ("enhance_contrast", "hist_equalize", "morph_recon", "tophat")]
    assert (steps, post) == ((expected[:2] + expected[3:], expected[2:3]) if resident else (expected, []))
    assert step_spec(steps[0]).dtype_policy == "declared" and step_spec(expected[1]).scope == "needs_reference"


def test_legacy_defaults_and_top_level_snr_threshold(tmp_path):
    config = workflow_config(tmp_path, snr_threshold=3.0, enhance_contrast={"run": True}, hist_equalize={"run": True},
                             morph_recon={"run": True}, tophat={"run": True})
    original = copy.deepcopy(config)
    steps, post = recipe_configs(from_workflow_config(config).pipeline.preprocessing)
    assert config == original
    assert steps == [MinMaxNormalizationConfig("uint8", (0, 255), snr_threshold=3.0), HistogramMatchingConfig(),
                     ReconstructionConfig(), TophatConfig()] and post == []


def test_schema_validates_python_rules_with_tophat(tmp_path):
    schema = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
    config = yaml.safe_load((ROOT / "docs/examples/workflow-full.yaml").read_text())
    for rule in ("rsf_single_fov", "gr_single_fov_subtile", "lrsf_single_fov_subtile", "deep_create_subtile", "deep_rsf_subtile"):
        candidate = copy.deepcopy(config)
        candidate["rules"][rule].setdefault("parameters", {})["tophat"] = {"run": True, "radius": 2}
        jsonschema.validate(candidate, schema)
        recipe = from_workflow_config(candidate, rule).pipeline.preprocessing
        assert TophatConfig(radius_yx=2) in recipe_configs(recipe)[0]
        for bad in ({"run": "yes"}, {"run": True, "radius": 0}, {"run": True, "size": 2}):
            candidate["rules"][rule]["parameters"]["tophat"] = bad
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(candidate, schema)
