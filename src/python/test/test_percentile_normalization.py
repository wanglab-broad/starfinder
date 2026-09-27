"""Percentile normalization, fitting modes and the histogram summary and merge API (W-230).

Exact expectations are independent numpy and scikit-image computations:
numpy.percentile(method="inverted_cdf") on the concatenated volumes and
skimage.exposure.match_histograms against the concatenated reference.
"""
from dataclasses import asdict
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from skimage.exposure import match_histograms

from starfinder.dataset import Dataset, PipelineConfig, RoundState
from starfinder.image import ImageMetadata
from starfinder.preprocessing import (STEPS, HistogramMatchingConfig, HistogramSummary, MinMaxNormalizationConfig,
    PercentileNormalizationConfig, PreprocessingRecipe, RecipeStep, StepContext, TophatConfig, filter_tophat,
    histogram_percentile, match_histogram, merge_histograms, normalize_percentile, read_histograms,
    read_supplied_statistics, run_step, summarize_histograms, summary_stage, supplied_section, supplied_statistics,
    write_histograms, write_supplied_statistics)
from starfinder.preprocessing.normalization import _match_counts

ROOT = Path(__file__).resolve().parents[3]
CHANNELS = ("ch00", "ch01")
ROUNDS = ("round1", "round2")
PERCENTILES = (0, 1, 50, 99.9, 100)
DTYPES = (np.uint8, np.uint16)
CONTEXT = StepContext("round1", "round1", ImageMetadata("frame"))


def expected(x, low, high, dtype_max):
    return np.rint(np.clip((x.astype(np.float64) - low) / (high - low), 0, 1) * dtype_max)


def random_fovs(dtype, seed=0, channels=2):
    """Three FOVs of one dtype with different Z depths and value spreads (same YX)."""
    rng = np.random.default_rng(seed)
    top = np.iinfo(dtype).max
    fovs = {}
    for i, (depth, high) in enumerate(((3, top), (2, top // 3), (4, top // 50))):
        fovs[f"FOV_{i + 1:03d}"] = {name: rng.integers(0, high, (depth, 9, 7, channels), dtype=dtype, endpoint=True)
                                    for name in ROUNDS}
    return fovs


def summaries_of(fovs, after=()):
    return [summarize_histograms(rounds, channel_labels=CHANNELS[:next(iter(rounds.values())).shape[-1]],
                                 fov_id=fov_id, summarized_after=after) for fov_id, rounds in fovs.items()]


def dataset(tmp_path):
    return Dataset(tmp_path, tmp_path / "out", "test", "sample", "out",
                   rounds=RoundState(sequencing_rounds=list(ROUNDS), reference_round="round1"), channel_order=CHANNELS)


def resident(ds, fov_id, rounds):
    fov = ds.fov(fov_id)
    for name, volume in rounds.items():
        fov.images[name] = volume.copy()
        fov.metadata[name] = ImageMetadata(f"{fov_id}/{name}")
    return fov


def document(sections, dtype="uint8", labels=CHANNELS):
    return {"schema": "starfinder.preprocessing.supplied/1", "dtype": dtype, "channel_labels": list(labels),
            "fovs_used": ["FOV_001"], "fovs_excluded": [], "steps": sections}


def percentile_section(fitted, after=(), p_low=1.0, p_high=99.9):
    return {"summarized_after": list(after), "params": {"p_low": p_low, "p_high": p_high}, "fitted": fitted}


# --- Registration and config ------------------------------------------------------

def test_percentile_normalization_is_registered_per_channel_and_preserving():
    spec = STEPS[PercentileNormalizationConfig]
    assert (spec.name, spec.category, spec.scope, spec.dtype_policy) == \
        ("percentile_normalization", "intensity", "per_channel", "preserve")
    assert asdict(PercentileNormalizationConfig()) == {"p_low": 1.0, "p_high": 99.9, "fit": "fov"}
    for bad in (dict(p_low=-1), dict(p_high=100.5), dict(p_low=5, p_high=5), dict(p_low=float("nan")),
                dict(p_low=True), dict(fit="sample")):
        with pytest.raises(ValueError):
            PercentileNormalizationConfig(**bad)
    with pytest.raises(ValueError):
        HistogramMatchingConfig(fit="merged")


# --- Output formula and dtype ------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_output_equals_the_formula_on_hand_constructed_arrays(dtype):
    top = np.iinfo(dtype).max
    # Channel 0 has literal values including exact halves after scaling; channel 1 is a ramp.
    channel0 = np.array([0, 10, 11, 12, 50, 100, 101, 150, 200, 210, 250, top], dtype=dtype).reshape(1, 3, 4)
    channel1 = np.linspace(0, top, 12).astype(dtype).reshape(1, 3, 4)
    x = np.stack([channel0, channel1], axis=-1)
    supplied = {"low": [10, 3], "high": [210, top - 7]}
    y = normalize_percentile(x, config=PercentileNormalizationConfig(fit="supplied"), fitted=supplied)
    assert y.dtype == dtype
    for c in range(2):
        np.testing.assert_array_equal(y[..., c], expected(x[..., c], supplied["low"][c], supplied["high"][c], top))
    # fit="fov": the range is the inverted-CDF percentiles of each channel.
    config = PercentileNormalizationConfig(p_low=10, p_high=90)
    result = run_step(x, config, CONTEXT)
    for c in range(2):
        low, high = (np.percentile(x[..., c], p, method="inverted_cdf") for p in (10, 90))
        assert (result.fitted["low"][c], result.fitted["high"][c]) == (low, high)
        np.testing.assert_array_equal(result.image[..., c], expected(x[..., c], low, high, top))
    assert result.image.dtype == dtype


def test_float_input_maps_to_unit_interval():
    x = np.array([-2.0, 0.0, 1.0, 3.0, 6.0, 10.0]).reshape(1, 2, 3)
    for dtype in (np.float32, np.float64):
        y = normalize_percentile(x.astype(dtype), config=PercentileNormalizationConfig(p_low=0, p_high=100))
        assert y.dtype == dtype
        np.testing.assert_array_equal(y, ((x + 2) / 12).astype(dtype))
        assert y.min() == 0 and y.max() == 1
    y = run_step(x, PercentileNormalizationConfig(p_low=20, p_high=80), CONTEXT).image
    np.testing.assert_array_equal(y, np.clip((x - 0.0) / (6.0 - 0.0), 0, 1))


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32, np.float64))
def test_constant_channel_and_equal_range_give_zeros_and_degenerate_range(dtype):
    x = np.zeros((2, 4, 5, 3), dtype=dtype)
    x[..., 0] = 7
    x[..., 1] = np.arange(40).reshape(2, 4, 5)
    x[..., 2] = 3
    x[0, 0, 0, 2] = 9  # nearly constant: p_low == p_high == 3
    result = run_step(x, PercentileNormalizationConfig(p_low=5, p_high=95), CONTEXT)
    assert result.image.dtype == dtype
    np.testing.assert_array_equal(result.image[..., 0], 0)
    np.testing.assert_array_equal(result.image[..., 2], 0)
    assert result.image[..., 1].max() > 0
    assert result.diagnostics["degenerate_range"] == [True, False, True]
    assert result.diagnostics["constant_channel"] == [True, False, False]
    # A supplied high == low on a nonconstant channel is degenerate too.
    supplied = normalize_percentile(x, config=PercentileNormalizationConfig(fit="supplied"),
                                    fitted={"low": [7, 5, 3], "high": [7, 5, 9]})
    assert supplied.dtype == dtype
    np.testing.assert_array_equal(supplied[..., 1], 0)


def test_diagnostics_follow_the_shared_numerical_policy():
    x = np.zeros((1, 4, 5, 2), dtype=np.uint8)
    x[..., 0] = np.arange(20).reshape(1, 4, 5)
    x[0, 0, :, 1] = 200  # mostly zero: MAD is 0
    result = run_step(x, PercentileNormalizationConfig(p_low=0, p_high=100), CONTEXT)
    diagnostics = result.diagnostics
    for c in range(2):
        values = result.image[..., c].astype(np.float64)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        assert diagnostics["zero_fraction"][c] == np.mean(values == 0)
        assert diagnostics["median"][c] == median and diagnostics["mad"][c] == mad
        assert diagnostics["noise_threshold"][c] == median + 5 * mad * 1.4826
        assert diagnostics["mad_zero"][c] == (mad == 0)
    assert diagnostics["mad_zero"] == [False, True]
    assert json.dumps(diagnostics)


def test_input_is_not_modified_and_invalid_input_raises():
    x = np.arange(24, dtype=np.uint16).reshape(1, 2, 3, 4)
    before = x.copy()
    normalize_percentile(x)
    np.testing.assert_array_equal(x, before)
    for bad in (np.zeros((0, 2, 2), np.uint8), np.array([[[np.nan, 1.0]]])):
        with pytest.raises(ValueError):
            normalize_percentile(bad)
    with pytest.raises(ValueError, match="channels"):
        normalize_percentile(x, config=PercentileNormalizationConfig(fit="supplied"), fitted={"low": [0], "high": [9]})
    with pytest.raises(ValueError, match="requires"):
        normalize_percentile(x, config=PercentileNormalizationConfig(fit="supplied"))
    with pytest.raises(ValueError, match="only with"):
        normalize_percentile(x, fitted={"low": [0] * 4, "high": [9] * 4})


# --- Histogram summary, merge and percentiles ---------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_merged_percentiles_equal_numpy_inverted_cdf_on_concatenated_volumes(dtype):
    fovs = random_fovs(dtype, seed=1)
    merged = merge_histograms(summaries_of(fovs))
    assert merged.counts.shape == (2, 2, np.iinfo(dtype).max + 1)
    assert merged.fovs_used == tuple(fovs)
    for r, name in enumerate(ROUNDS):
        for c in range(2):
            concatenated = np.concatenate([rounds[name][..., c].ravel() for rounds in fovs.values()])
            for p in PERCENTILES:
                value = histogram_percentile(merged.counts[r, c], p)
                assert value == np.percentile(concatenated, p, method="inverted_cdf"), (name, c, p)
    # The supplied section uses the same estimator.
    section = supplied_section(PercentileNormalizationConfig(p_low=1, p_high=99.9), merged)
    for name in ROUNDS:
        concatenated = [np.concatenate([rounds[name][..., c].ravel() for rounds in fovs.values()]) for c in range(2)]
        assert section["fitted"][name] == {
            "low": [np.percentile(v, 1, method="inverted_cdf") for v in concatenated],
            "high": [np.percentile(v, 99.9, method="inverted_cdf") for v in concatenated]}


def test_histogram_percentile_edge_cases():
    counts = np.array([0, 0, 3, 0, 1])  # values 2, 2, 2, 4
    assert histogram_percentile(counts, 0) == 2 and histogram_percentile(counts, 100) == 4
    assert histogram_percentile(counts, 75) == 2 and histogram_percentile(counts, 75.01) == 4
    for n in (1, 7, 1000, 1001):  # numpy's float64 rounding of n * p / 100
        values = np.arange(n) % 5
        for p in PERCENTILES + (33.3, 66.7, 12.5):
            assert histogram_percentile(np.bincount(values), p) == np.percentile(values, p, method="inverted_cdf")
    for bad_counts in (np.zeros(3, int), np.array([1.0, 2.0]), np.array([[1, 2]]), np.array([-1, 2])):
        with pytest.raises(ValueError):
            histogram_percentile(bad_counts, 50)
    for p in (-1, 101, float("nan"), True):
        with pytest.raises(ValueError):
            histogram_percentile(counts, p)


def test_summary_merge_exclusion_and_npz_round_trip(tmp_path):
    fovs = random_fovs(np.uint8, seed=2)
    after = ({"step": "white_tophat", "config": {"radius_yx": 3}},)
    summaries = summaries_of(fovs, after)
    merged = merge_histograms(summaries, exclude=["FOV_003"])
    assert merged.fovs_used == ("FOV_001", "FOV_002") and merged.fovs_excluded == ("FOV_003",)
    np.testing.assert_array_equal(merged.counts, summaries[0].counts + summaries[1].counts)
    assert merged.summarized_after == after
    loaded = read_histograms(write_histograms(merged, tmp_path / "merged.npz"))
    np.testing.assert_array_equal(loaded.counts, merged.counts)
    for field in ("dtype", "round_names", "channel_labels", "summarized_after", "fovs_used", "fovs_excluded"):
        assert getattr(loaded, field) == getattr(merged, field)
    # Incompatible summaries or selections raise.
    other_after = summaries_of({"FOV_009": fovs["FOV_001"]})
    uint16 = summaries_of({"FOV_009": {n: v.astype(np.uint16) for n, v in fovs["FOV_001"].items()}}, after)
    relabeled = HistogramSummary(summaries[0].counts, "uint8", ROUNDS, ("a", "b"), after, ("FOV_009",))
    for bad in (dict(summaries=summaries + other_after), dict(summaries=summaries + uint16),
                dict(summaries=summaries + [relabeled]), dict(summaries=summaries + summaries[:1]),
                dict(summaries=summaries, exclude=["FOV_404"]), dict(summaries=summaries[:1], exclude=["FOV_001"]),
                dict(summaries=[merged, summaries[0]]), dict(summaries=[])):
        with pytest.raises(ValueError):
            merge_histograms(**bad)
    with pytest.raises(ValueError, match="uint8 and uint16"):
        summarize_histograms({"round1": np.ones((1, 2, 2, 2))}, channel_labels=CHANNELS, fov_id="F")
    with pytest.raises(ValueError, match="labels"):
        summarize_histograms({"round1": fovs["FOV_001"]["round1"]}, channel_labels=("a",), fov_id="F")


# --- Histogram matching against a supplied merged reference ------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_supplied_histogram_reference_equals_matching_the_concatenated_reference(dtype):
    fovs = random_fovs(dtype, seed=3, channels=2)
    merged = merge_histograms(summaries_of(fovs))
    config = HistogramMatchingConfig(reference_channel=1, fit="supplied")
    section = supplied_section(config, merged, reference_round="round1")
    reference = np.concatenate([rounds["round1"][..., 1] for rounds in fovs.values()], axis=0)
    source = random_fovs(dtype, seed=4)["FOV_002"]["round2"]
    context = StepContext("round2", "round1", ImageMetadata("frame"), supplied=section)
    # Float output exposes the count-based mapping before the cast to the input dtype.
    fitted = section["fitted"]["round1"]
    float_output = _match_counts(source, fitted["values"], fitted["counts"], HistogramMatchingConfig(output_dtype="float64"))
    for c in range(2):
        np.testing.assert_array_equal(float_output[..., c], match_histograms(source[..., c], reference))
    # The default dtype-preserving step equals the fov-mode matching against the concatenation.
    image = run_step(source, config, context).image
    assert image.dtype == dtype
    np.testing.assert_array_equal(image, match_histogram(source, reference, config=HistogramMatchingConfig()))
    # The supplied mode takes no reference image, and the fov mode still requires one.
    with pytest.raises(ValueError, match="does not take a reference"):
        run_step(source, config, StepContext("round2", "round1", ImageMetadata("frame"), reference, section))
    with pytest.raises(ValueError, match="requires supplied"):
        run_step(source, config, StepContext("round2", "round1", ImageMetadata("frame")))
    with pytest.raises(ValueError, match="does not take supplied"):
        run_step(source, HistogramMatchingConfig(), StepContext("round2", "round1", ImageMetadata("frame"), reference,
                                                                section))


def test_supplied_histogram_reference_through_fov_run_matches_legacy_recipe_1(tmp_path):
    """Recipe 1 with a single-FOV supplied reference reproduces the per-FOV reference."""
    fovs = random_fovs(np.uint16, seed=5)
    minmax = RecipeStep(MinMaxNormalizationConfig("uint8", (0, 255)))
    fov_recipe = PreprocessingRecipe((minmax, RecipeStep(HistogramMatchingConfig())))
    path = tmp_path / "supplied.json"
    recipe = PreprocessingRecipe((minmax, RecipeStep(HistogramMatchingConfig(fit="supplied"))), supplied_statistics=path)
    prefix, after = summary_stage(recipe, "histogram_matching")
    assert prefix.steps == (minmax,) and prefix.supplied_statistics is None
    assert after == ({"step": "min_max_normalization", "config": json.loads(json.dumps(asdict(minmax.config)))},)
    ds = dataset(tmp_path)
    fov = resident(ds, "FOV_001", fovs["FOV_001"]).run(PipelineConfig(preprocessing=prefix))
    merged = merge_histograms([summarize_histograms(fov.images, channel_labels=CHANNELS, fov_id="FOV_001",
                                                    summarized_after=after)])
    section = supplied_section(recipe.steps[1].config, merged, reference_round="round1")
    write_supplied_statistics(supplied_statistics(merged, {"histogram_matching": section}), path)
    supplied = resident(ds, "FOV_001", fovs["FOV_001"]).run(PipelineConfig(preprocessing=recipe))
    legacy = resident(ds, "FOV_001", fovs["FOV_001"]).run(PipelineConfig(preprocessing=fov_recipe))
    for name in ROUNDS:
        np.testing.assert_array_equal(supplied.images[name], legacy.images[name])


# --- Fitting modes through FOV.run --------------------------------------------------

def test_fov_fitted_ranges_written_as_supplied_values_give_identical_output(tmp_path):
    fovs = random_fovs(np.uint16, seed=6)
    ds = dataset(tmp_path)
    tophat = RecipeStep(TophatConfig())
    fitted_run = resident(ds, "FOV_001", fovs["FOV_001"]).run(PipelineConfig(
        preprocessing=PreprocessingRecipe((tophat, RecipeStep(PercentileNormalizationConfig())))))
    records = fitted_run.preprocessing_record["rounds"]
    fitted = {name: records[name][1]["fitted"] for name in ROUNDS}
    path = tmp_path / "supplied.json"
    recipe = PreprocessingRecipe((tophat, RecipeStep(PercentileNormalizationConfig(fit="supplied"))),
                                 supplied_statistics=path)
    after = summary_stage(recipe, "percentile_normalization")[1]
    write_supplied_statistics(document({"percentile_normalization": percentile_section(fitted, after)}, "uint16"), path)
    supplied_run = resident(ds, "FOV_001", fovs["FOV_001"]).run(PipelineConfig(preprocessing=recipe))
    for name in ROUNDS:
        assert supplied_run.images[name].dtype == np.uint16
        np.testing.assert_array_equal(supplied_run.images[name], fitted_run.images[name])
        assert supplied_run.preprocessing_record["rounds"][name][1]["fitted"] == fitted[name]
    record = supplied_run.preprocessing_record["supplied_statistics"]
    assert record["path"] == str(path) and len(record["sha256"]) == 64


def test_supplied_range_is_fitted_after_the_preceding_step(tmp_path):
    fovs = random_fovs(np.uint8, seed=7)
    for rounds in fovs.values():  # a background pedestal the top-hat removes
        for name in ROUNDS:
            rounds[name] = (rounds[name] // 4 + 60).astype(np.uint8)
    path = tmp_path / "supplied.json"
    recipe = PreprocessingRecipe((RecipeStep(TophatConfig()), RecipeStep(PercentileNormalizationConfig(fit="supplied"))),
                                 supplied_statistics=path)
    prefix, after = summary_stage(recipe, "percentile_normalization")
    ds = dataset(tmp_path)
    summaries = []
    for fov_id, rounds in fovs.items():
        fov = resident(ds, fov_id, rounds).run(PipelineConfig(preprocessing=prefix))
        summaries.append(summarize_histograms(fov.images, channel_labels=CHANNELS, fov_id=fov_id,
                                              summarized_after=after))
    merged = merge_histograms(summaries)
    section = supplied_section(recipe.steps[1].config, merged)
    for name in ROUNDS:
        for c in range(2):
            raw = np.concatenate([rounds[name][..., c].ravel() for rounds in fovs.values()])
            tophat = np.concatenate([filter_tophat(rounds[name])[..., c].ravel() for rounds in fovs.values()])
            fitted = (section["fitted"][name]["low"][c], section["fitted"][name]["high"][c])
            assert fitted == tuple(np.percentile(tophat, p, method="inverted_cdf") for p in (1, 99.9))
            assert fitted != tuple(np.percentile(raw, p, method="inverted_cdf") for p in (1, 99.9))
    write_supplied_statistics(supplied_statistics(merged, {"percentile_normalization": section}), path)
    fov = resident(ds, "FOV_002", fovs["FOV_002"]).run(PipelineConfig(preprocessing=recipe))
    low, high = section["fitted"]["round2"]["low"][1], section["fitted"]["round2"]["high"][1]
    np.testing.assert_array_equal(fov.images["round2"][..., 1],
                                  expected(filter_tophat(fovs["FOV_002"]["round2"])[..., 1], low, high, 255))


# --- Supplied-file validation --------------------------------------------------------

def _supplied_setup(tmp_path):
    path = tmp_path / "supplied.json"
    tophat = RecipeStep(TophatConfig())
    recipe = PreprocessingRecipe((tophat, RecipeStep(PercentileNormalizationConfig(fit="supplied"))),
                                 supplied_statistics=path)
    after = summary_stage(recipe, "percentile_normalization")[1]
    fitted = {name: {"low": [0, 1], "high": [200, 150]} for name in ROUNDS}
    return path, recipe, document({"percentile_normalization": percentile_section(fitted, after)})


def _mutations(good):
    def edit(change):
        candidate = json.loads(json.dumps(good))
        change(candidate)
        return candidate
    section = lambda d: d["steps"]["percentile_normalization"]
    return {
        "schema": edit(lambda d: d.update(schema="starfinder.preprocessing.supplied/2")),
        "dtype": edit(lambda d: d.update(dtype="uint16")),
        "channel labels": edit(lambda d: d.update(channel_labels=["ch01", "ch00"])),
        "missing section": edit(lambda d: d["steps"].clear()),
        "missing round": edit(lambda d: section(d)["fitted"].pop("round2")),
        "missing channel": edit(lambda d: section(d)["fitted"]["round1"].update(low=[0], high=[200])),
        "summarized_after": edit(lambda d: section(d).update(summarized_after=[])),
        "summarized_after config": edit(lambda d: section(d)["summarized_after"][0]["config"].update(radius_yx=5)),
        "params": edit(lambda d: section(d)["params"].update(p_high=99.0)),
        "low above high": edit(lambda d: section(d)["fitted"]["round1"].update(low=[201, 1])),
        "unknown step": edit(lambda d: d["steps"].update(no_such_step=section(d))),
        "extra key": edit(lambda d: d.update(comment="x")),
    }


@pytest.mark.parametrize("fault", list(_mutations(_supplied_setup(Path("."))[2])))
def test_supplied_file_faults_raise(tmp_path, fault):
    path, recipe, good = _supplied_setup(tmp_path)
    bad = _mutations(good)[fault]
    path.write_text(json.dumps(bad))
    fovs = random_fovs(np.uint8, seed=8)
    ds = dataset(tmp_path)
    fov = resident(ds, "FOV_001", fovs["FOV_001"])
    with pytest.raises(ValueError):
        fov.run(PipelineConfig(preprocessing=recipe))
    with pytest.raises(ValueError):
        read_supplied_statistics(path, recipe, dtype="uint8", channel_labels=CHANNELS, rounds=ROUNDS)
    if fault == "dtype":
        # The file's dtype is compared with the step's input when the step runs.
        read_supplied_statistics(path, recipe, channel_labels=CHANNELS, rounds=ROUNDS)
    else:
        # Every other fault is found when the file is read, before any step runs.
        np.testing.assert_array_equal(fov.images["round1"], fovs["FOV_001"]["round1"])


def test_valid_supplied_file_is_read_and_written(tmp_path):
    path, recipe, good = _supplied_setup(tmp_path)
    write_supplied_statistics(good, path)
    assert read_supplied_statistics(path, recipe, dtype=np.uint8, channel_labels=CHANNELS, rounds=ROUNDS) == good
    with pytest.raises(ValueError, match="rounds"):
        read_supplied_statistics(path, recipe, rounds=ROUNDS + ("round3",))
    path.write_text("{not json")
    with pytest.raises(ValueError, match="JSON"):
        read_supplied_statistics(path)


def test_step_input_dtype_must_match_the_supplied_file(tmp_path):
    path, recipe, good = _supplied_setup(tmp_path)
    write_supplied_statistics(good, path)
    fovs = random_fovs(np.uint16, seed=9)
    with pytest.raises(ValueError, match="supplied statistics are for uint8"):
        resident(dataset(tmp_path), "FOV_001", fovs["FOV_001"]).run(PipelineConfig(preprocessing=recipe))


def test_histogram_section_faults_raise(tmp_path):
    fovs = random_fovs(np.uint8, seed=10)
    merged = merge_histograms(summaries_of(fovs))
    config = HistogramMatchingConfig(fit="supplied")
    section = supplied_section(config, merged, reference_round="round1")
    recipe = PreprocessingRecipe((RecipeStep(config),), supplied_statistics=tmp_path / "s.json")
    good = supplied_statistics(merged, {"histogram_matching": section})
    write_supplied_statistics(good, tmp_path / "s.json")
    assert read_supplied_statistics(tmp_path / "s.json", recipe)["steps"]["histogram_matching"] == section
    for change in (lambda s: s["params"].update(reference_channel=2), lambda s: s["params"].update(reference_round="round2"),
                   lambda s: s["fitted"]["round1"]["counts"].pop(), lambda s: s["fitted"]["round1"]["values"].reverse(),
                   lambda s: (s["fitted"]["round1"]["values"].append(256), s["fitted"]["round1"]["counts"].append(1)),
                   lambda s: s["fitted"]["round1"]["counts"].__setitem__(0, 0)):
        candidate = json.loads(json.dumps(good))
        change(candidate["steps"]["histogram_matching"])
        with pytest.raises(ValueError):
            write_supplied_statistics(candidate, tmp_path / "bad.json")
    other = PreprocessingRecipe((RecipeStep(HistogramMatchingConfig(reference_channel=1, fit="supplied")),),
                                supplied_statistics=tmp_path / "s.json")
    with pytest.raises(ValueError, match="reference_channel"):
        read_supplied_statistics(tmp_path / "s.json", other)
    for bad in (dict(reference_round="round9"), dict(reference_round=None)):
        with pytest.raises(ValueError, match="reference_round"):
            supplied_section(config, merged, **bad)
    with pytest.raises(ValueError, match="no values for round"):
        run_step(fovs["FOV_001"]["round2"], config, StepContext("round2", "round2", ImageMetadata("f"), supplied=section))


def test_recipe_rejects_repeated_supplied_step_names_and_a_missing_file():
    supplied = RecipeStep(PercentileNormalizationConfig(fit="supplied"))
    with pytest.raises(ValueError, match="more than once"):
        PreprocessingRecipe((supplied, RecipeStep(TophatConfig()), supplied), supplied_statistics="s.json")
    with pytest.raises(ValueError, match="supplied_statistics is not set"):
        PreprocessingRecipe((supplied,))
    # The same name with fit="fov" alongside one supplied step is allowed.
    recipe = PreprocessingRecipe((RecipeStep(PercentileNormalizationConfig()), supplied), supplied_statistics="s.json")
    assert recipe.supplied_statistics == Path("s.json")
    with pytest.raises(ValueError, match="no step"):
        summary_stage(PreprocessingRecipe((RecipeStep(TophatConfig()),)), "percentile_normalization")


# --- Two-pass example -----------------------------------------------------------------

def test_two_pass_example_runs(tmp_path):
    main = runpy.run_path(str(ROOT / "docs/examples/percentile_two_pass.py"))["main"]
    statistics, results = main(tmp_path / "example")
    assert statistics["fovs_used"] == ["FOV_001", "FOV_002", "FOV_003"]
    assert sorted(p.name for p in (tmp_path / "example").iterdir() if p.is_file()) == [
        "FOV_001_histograms.npz", "FOV_002_histograms.npz", "FOV_003_histograms.npz", "supplied.json"]
    for rounds in results.values():
        assert all(image.dtype == np.uint8 for image in rounds.values())
