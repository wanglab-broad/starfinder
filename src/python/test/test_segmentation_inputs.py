"""The §2.9 morphology input functions (W-312).

Row L7 of the engineering validation design in docs/assignment-algorithms.md: the
DAPI–amplicon composite and the Flamingo enhancement against W-307's pinned digests
(``test_segmentation_golden.py``), ``normalize_percentiles`` against its formula computed
here, and ``rescale_input``'s shape and metadata; then input preservation (D4), the shape
rules and records of docs/segmentation-contract.md ("Input preparation and label
functions"), and the reuse of the §2.6 registration outputs: a morphology round
registered by ``FOV.register_rounds`` and the reference merged image, compared with the
same arrays read back from the ``registered`` checkpoint and the round's saved form
(``FOV.load_registered_round``). Every fixture is at most
32×64×64 voxels.
"""
import json
from dataclasses import asdict

import numpy as np
import pytest

from starfinder.dataset import CheckpointConfig, Dataset, RoundState
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import load_volume
from starfinder.preprocessing import PREPROCESSING_METHODS, ProjectionConfig, project_image
from starfinder.segmentation import (CompositeConfig, FlamingoEnhancementConfig, composite_nuclei_amplicon,
    enhance_with_flamingo, normalize_percentiles, rescale_input)

from .test_registration_other_rounds import SEQUENCING, SHIFT, TRANSLATION, grid, texture
from .test_segmentation_golden import (COMPOSITE_DIGESTS, CONSTANT_DIGESTS, FLAMINGO_DIGEST, SHAPE_ZYX,
    SHRUNK_IMAGE_DIGEST, digest, fixture)

pytestmark = [pytest.mark.segmentation, pytest.mark.validation]

UM = ImageMetadata("FOV/round1", spacing_zyx=(0.35, 0.1, 0.1), origin_zyx=(0.0, 0.0, 0.0),
                   direction_zyx=((1, 0, 0), (0, 1, 0), (0, 0, 1)), spatial_unit="micrometer")
RECORD_KEYS = {"function", "config", "inputs", "output"}


@pytest.fixture(scope="module")
def images():
    return fixture()


def ramp(shape=(4, 20, 22)):
    """A uint8 ramp with a stated value at every voxel: (3z + 2y + x) mod 251."""
    z, y, x = np.indices(shape)
    return ((3 * z + 2 * y + x) % 251).astype(np.uint8)


# --- L7: composite and enhancement ----------------------------------------------------------------

def test_l7_composite_and_enhancement_equal_the_pinned_digests(images):
    composite, _ = composite_nuclei_amplicon(images["dapi"], images["amplicon"])
    assert composite.dtype == np.uint8 and composite.shape == SHAPE_ZYX
    assert digest(composite) == COMPOSITE_DIGESTS[False]
    enhanced, _ = enhance_with_flamingo(images["dapi"], images["flamingo"])
    assert enhanced.dtype == np.uint8 and enhanced.shape == SHAPE_ZYX
    assert digest(enhanced) == FLAMINGO_DIGEST


def test_l7_constant_and_zero_inputs_equal_the_pinned_digests(images):
    """The assertions of the golden test_constant_and_zero_inputs, with the package functions."""
    c50, c80, zero = (np.full(SHAPE_ZYX, value, np.uint8) for value in (50, 80, 0))

    def composite(nuclear, amplicon):
        return composite_nuclei_amplicon(nuclear, amplicon)[0]

    def flamingo(nuclear, stain):
        return enhance_with_flamingo(nuclear, stain)[0]

    assert np.unique(composite(c50, c80)).tolist() == [80]
    assert np.unique(composite(zero, zero)).tolist() == [0]
    assert np.unique(flamingo(c50, c80)).tolist() == [34]  # 50 × (1 − 80/255) = 34.3
    assert np.unique(flamingo(zero, images["flamingo"])).tolist() == [0]
    constant = composite(c50, images["amplicon"])
    alone = composite(zero, images["amplicon"])
    assert np.array_equal(constant, np.maximum(alone, 50))
    results = {"composite_constant": constant, "composite_zero": alone,
               "flamingo_constant": flamingo(images["dapi"], c80),
               "flamingo_zero": flamingo(images["dapi"], zero)}
    assert {name: digest(result) for name, result in results.items()} == CONSTANT_DIGESTS


# --- L7: normalization ----------------------------------------------------------------------------

def test_l7_normalization_follows_the_formula(images):
    dapi = images["dapi"]
    low, high = np.percentile(dapi, 1.0), np.percentile(dapi, 99.8)  # NumPy, linear
    expected = (dapi.astype(np.float64) - low) / (high - low + 1e-20)
    output, record = normalize_percentiles(dapi)
    assert output.dtype == np.float32 and output.shape == dapi.shape
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
    assert output.min() < 0 and output.max() > 1  # unclipped
    assert record["percentiles"] == {"low": low, "high": high}
    assert record["config"] == {"p_low": 1.0, "p_high": 99.8, "axes": [0, 1, 2]}


def test_l7_normalization_of_a_constant_image_gives_zeros():
    output, record = normalize_percentiles(np.full(SHAPE_ZYX, 50, np.uint8))
    assert output.dtype == np.float32 and output.shape == SHAPE_ZYX and not output.any()
    assert record["percentiles"] == {"low": 50.0, "high": 50.0}


def test_normalization_is_per_channel_by_default(images):
    """A ZYXC image: the default axes (0, 1, 2) normalize each channel over its own volume."""
    stack = np.stack([images["dapi"], images["flamingo"]], axis=-1)
    output, record = normalize_percentiles(stack, p_low=2.0, p_high=98.0)
    assert output.shape == stack.shape
    for c, name in enumerate(("dapi", "flamingo")):
        low, high = np.percentile(images[name], 2.0), np.percentile(images[name], 98.0)
        np.testing.assert_allclose(output[..., c], (images[name] - low) / (high - low + 1e-20), rtol=1e-6, atol=1e-6)
        assert (record["percentiles"]["low"][c], record["percentiles"]["high"][c]) == (low, high)
    whole, _ = normalize_percentiles(stack, axes=(0, 1, 2, 3))
    low, high = np.percentile(stack, 1.0), np.percentile(stack, 99.8)
    np.testing.assert_allclose(whole, (stack - low) / (high - low + 1e-20), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("options", [dict(p_low=99.8, p_high=1.0), dict(p_low=-1.0), dict(p_high=101.0),
                                     dict(axes=(0, 0)), dict(axes=(3,)), dict(axes=())])
def test_normalization_rejects_invalid_settings(images, options):
    with pytest.raises(ValueError):
        normalize_percentiles(images["dapi"], **options)


# --- L7: rescaling --------------------------------------------------------------------------------

def test_l7_rescaling_divides_the_spacing_and_records_the_rescale():
    image = ramp()
    output, metadata, record = rescale_input(image, UM, scale_zyx=(1, 0.5, 0.5))
    assert output.shape == (4, 10, 11)
    assert metadata.spacing_zyx == (0.35, 0.2, 0.2)
    assert metadata.frame_id == "FOV/round1/rescale:1.0,0.5,0.5"
    assert (metadata.direction_zyx, metadata.spatial_unit) == (UM.direction_zyx, "micrometer")
    assert record == {"function": "rescale_input", "config": {"scale_zyx": [1.0, 0.5, 0.5]},
                      "inputs": [digest(image)], "output": digest(output), "input_shape": [4, 20, 22],
                      "output_shape": [4, 10, 11], "frame_id": metadata.frame_id}
    # Output voxel i covers source voxels 2i and 2i + 1, so its centre is at source index 2i + 0.5.
    index = np.array([[0, 0, 0], [3, 9, 10], [1, 4, 7]], float)
    source = index * [1, 2, 2] + [0, 0.5, 0.5]
    np.testing.assert_allclose(metadata.index_to_world(index), UM.index_to_world(source), rtol=0, atol=1e-12)


def test_l7_rescaling_without_spacing_gives_no_spacing():
    output, metadata, _ = rescale_input(ramp(), ImageMetadata("FOV/round1"), scale_zyx=(1, 0.5, 0.5))
    assert output.shape == (4, 10, 11)
    assert metadata == ImageMetadata("FOV/round1/rescale:1.0,0.5,0.5")


def test_rescaling_is_the_legacy_shrink(images):
    """rescale(dapi, [1, .5, .5]) of the legacy script: W-307's pinned shrunk image."""
    output, _, _ = rescale_input(images["dapi"], ImageMetadata("dapi"), scale_zyx=(1, 0.5, 0.5))
    assert output.dtype == np.float64 and digest(output) == SHRUNK_IMAGE_DIGEST


def test_rescaling_keeps_one_pixel_and_rejects_invalid_factors():
    output, metadata, _ = rescale_input(ramp(), UM, scale_zyx=(0.1, 1, 1))
    assert output.shape == (1, 20, 22) and metadata.spacing_zyx[0] == 0.35 / 0.1
    for factors in [(1, 0, 1), (1, -0.5, 1), (1, float("nan"), 1), (1, 0.5), (True, 1, 1)]:
        with pytest.raises(ValueError):
            rescale_input(ramp(), UM, scale_zyx=factors)
    with pytest.raises(TypeError):
        rescale_input(ramp(), None, scale_zyx=(1, 1, 1))


# --- Inputs are preserved (D4) --------------------------------------------------------------------

def _calls(images):
    """(name, inputs, call) for every input array of the L7 checks above."""
    c50, c80, zero = (np.full(SHAPE_ZYX, value, np.uint8) for value in (50, 80, 0))
    float32 = images["dapi"].astype(np.float32)
    cases = [("composite", (images["dapi"], images["amplicon"]), composite_nuclei_amplicon),
             ("flamingo", (images["dapi"], images["flamingo"]), enhance_with_flamingo)]
    for label, pair in {"constant": (c50, c80), "zero": (zero, zero), "constant_nuclear": (c50, images["amplicon"]),
                        "zero_nuclear": (zero, images["amplicon"]), "constant_stain": (images["dapi"], c80),
                        "zero_stain": (images["dapi"], zero)}.items():
        cases += [(f"composite_{label}", pair, composite_nuclei_amplicon),
                  (f"flamingo_{label}", pair, enhance_with_flamingo)]
    cases += [("normalize_dapi", (images["dapi"],), normalize_percentiles),
              ("normalize_float32", (float32,), normalize_percentiles),
              ("normalize_constant", (c50,), normalize_percentiles),
              ("rescale_spacing", (ramp(),), lambda image: rescale_input(image, UM, scale_zyx=(1, 0.5, 0.5))),
              ("rescale_no_spacing", (ramp(),),
               lambda image: rescale_input(image, ImageMetadata("FOV/round1"), scale_zyx=(1, 0.5, 0.5)))]
    return cases


def test_inputs_are_preserved(images):
    for name, inputs, call in _calls(images):
        inputs = tuple(array.copy() for array in inputs)
        before = [digest(array) for array in inputs]
        output = call(*inputs)[0]
        assert [digest(array) for array in inputs] == before, name
        assert not any(np.shares_memory(output, array) for array in inputs), name


# --- Shapes and records ---------------------------------------------------------------------------

@pytest.mark.parametrize("function", [composite_nuclei_amplicon, enhance_with_flamingo])
def test_different_shapes_and_yx_inputs_raise(function):
    with pytest.raises(IncompatibleGeometryError):
        function(ramp((4, 20, 22)), ramp((4, 20, 21)))
    with pytest.raises(IncompatibleGeometryError):
        function(ramp((4, 20, 22)), ramp((3, 20, 22)))
    with pytest.raises(IncompatibleGeometryError):
        function(ramp((4, 20, 22))[0], ramp((4, 20, 22))[0])


@pytest.mark.parametrize("function", [composite_nuclei_amplicon, enhance_with_flamingo])
def test_a_plane_pair_is_accepted(images, function):
    second = images["amplicon" if function is composite_nuclei_amplicon else "flamingo"]
    nuclear, other = images["dapi"].max(axis=0)[None], second.max(axis=0)[None]
    output, record = function(nuclear, other)
    assert output.dtype == np.uint8 and output.shape == (1,) + SHAPE_ZYX[1:]
    assert record["inputs"] == [digest(nuclear), digest(other)] and record["output"] == digest(output)


def test_composite_record(images):
    config = CompositeConfig(nuclear_quantile=0.01, amplicon_quantile=0.002)
    output, record = composite_nuclei_amplicon(images["dapi"], images["amplicon"], config=config)
    assert set(record) == RECORD_KEYS | {"quantiles"}
    assert record["function"] == "composite_nuclei_amplicon"
    assert record["config"] == asdict(config) == {"nuclear_quantile": 0.01, "amplicon_quantile": 0.002}
    assert record["inputs"] == [digest(images["dapi"]), digest(images["amplicon"])]
    assert record["output"] == digest(output)
    assert record["quantiles"] == {
        "nuclear": [np.quantile(images["dapi"], 0.01), np.quantile(images["dapi"], 1 - 0.01)],
        "amplicon": [np.quantile(images["amplicon"], 0.002), np.quantile(images["amplicon"], 1 - 0.002)]}
    assert json.loads(json.dumps(record)) == record
    assert digest(output) != COMPOSITE_DIGESTS[False]  # the config is used


def test_enhancement_record(images):
    config = FlamingoEnhancementConfig()
    output, record = enhance_with_flamingo(images["dapi"], images["flamingo"], config=config)
    assert set(record) == RECORD_KEYS | {"quantiles"}
    assert record["function"] == "enhance_with_flamingo"
    assert record["config"] == {"flamingo_quantile": 0.005, "nuclear_quantile": 0.001, "median_radius_px": 1}
    assert record["inputs"] == [digest(images["dapi"]), digest(images["flamingo"])]
    assert record["output"] == digest(output) == FLAMINGO_DIGEST
    assert record["quantiles"] == {
        "nuclear": [np.quantile(images["dapi"], 0.001), np.quantile(images["dapi"], 1 - 0.001)],
        "flamingo": [np.quantile(images["flamingo"], 0.005), np.quantile(images["flamingo"], 1 - 0.005)]}
    assert json.loads(json.dumps(record)) == record
    unfiltered, _ = enhance_with_flamingo(images["dapi"], images["flamingo"],
                                          config=FlamingoEnhancementConfig(median_radius_px=0))
    assert digest(unfiltered) != FLAMINGO_DIGEST


def test_normalization_record(images):
    output, record = normalize_percentiles(images["dapi"])
    assert set(record) == RECORD_KEYS | {"percentiles"}
    assert record["function"] == "normalize_percentiles"
    assert record["inputs"] == [digest(images["dapi"])] and record["output"] == digest(output)
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize("make", [lambda: CompositeConfig(nuclear_quantile=0.5),
                                  lambda: CompositeConfig(amplicon_quantile=-0.1),
                                  lambda: FlamingoEnhancementConfig(flamingo_quantile=float("nan")),
                                  lambda: FlamingoEnhancementConfig(median_radius_px=-1),
                                  lambda: FlamingoEnhancementConfig(median_radius_px=1.5)])
def test_invalid_configs_raise(make):
    with pytest.raises(ValueError):
        make()


def test_a_wrong_config_type_raises(images):
    with pytest.raises(TypeError):
        composite_nuclei_amplicon(images["dapi"], images["amplicon"], config=FlamingoEnhancementConfig())
    with pytest.raises(TypeError):
        enhance_with_flamingo(images["dapi"], images["flamingo"], config=CompositeConfig())


def test_no_input_function_is_a_preprocessing_method():
    """D4: the input functions are not preprocessing steps (docs/segmentation-contract.md)."""
    assert not {CompositeConfig, FlamingoEnhancementConfig} & set(PREPROCESSING_METHODS)
    names = {getattr(spec, "name", None) for spec in PREPROCESSING_METHODS.values()}
    assert not names & {"composite_nuclei_amplicon", "enhance_with_flamingo", "normalize_percentiles",
                        "rescale_input"}


# --- Registration outputs are reused --------------------------------------------------------------

# The morphology round has four channels (it was read back from the registered checkpoint,
# with the sequencing rounds' channel labels, before W-337).
MORPH_CHANNELS = ("ch00", "ch01", "ch02", "ch03")


def morphology_fov(root):
    """A FOV with reference round1, sequencing round2 and morphology round morph (8×32×32, uint16).

    The textures of test_registration_other_rounds: texture 0 is ch04 of round1 and ch00 of
    morph, the shared stain (the morphology DAPI); morph is displaced by (0, 3, −2).
    """
    dataset = Dataset(root, root / "output", "dataset", "sample", "output",
                      RoundState(["round1", "round2"], ["morph"], "round1"), SEQUENCING,
                      other_channel_order={"morph": MORPH_CHANNELS})
    return dataset.fov("FOV")


def test_the_composite_reuses_the_registration_outputs(tmp_path):
    pytest.importorskip("SimpleITK").ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    p = grid()
    acquired = np.stack([texture(s, p - SHIFT.reshape(3, 1, 1, 1)) for s in (0, 4, 5, 10)], axis=-1)
    fov = morphology_fov(tmp_path)
    fov.images = {"round1": np.stack([texture(s, p) for s in (1, 2, 3, 0)], axis=-1),
                  "round2": np.stack([texture(s, p) for s in (6, 7, 8, 9)], axis=-1), "morph": acquired}
    fov.metadata = {name: ImageMetadata(f"FOV/{name}") for name in fov.images}
    checkpoints = CheckpointConfig(directory=tmp_path / "checkpoints")
    fov.register_rounds(TRANSLATION, rounds=["morph"], checkpoints=checkpoints)
    assert fov.registration_record["rounds"]["morph"]["reference"] == "round1"
    dapi = fov.images["morph"][..., 0]
    merged = project_image(fov.images["round1"], config=ProjectionConfig(axis="channel"))
    assert not np.array_equal(dapi, acquired[..., 0])  # the registered round, not the acquired one
    composite, record = composite_nuclei_amplicon(dapi, merged)
    assert composite.dtype == np.uint8 and composite.shape == acquired.shape[:3]
    assert record["inputs"] == [digest(dapi), digest(merged)]
    # The merged image is the one save_reference_image writes.
    saved = load_volume(fov.save_reference_image())
    assert np.array_equal(saved.image.reshape(merged.shape), merged)

    fov.save_checkpoint("registered", checkpoints=checkpoints)
    # The registered checkpoint holds the reference and sequencing rounds; morph is reloaded from its saved form.
    loaded = morphology_fov(tmp_path).load_checkpoint("registered", checkpoints=checkpoints)
    loaded.load_registered_round("morph", checkpoints=checkpoints)
    dapi_read = loaded.images["morph"][..., 0]
    merged_read = project_image(loaded.images["round1"], config=ProjectionConfig(axis="channel"))
    read_back, read_record = composite_nuclei_amplicon(dapi_read, merged_read)
    assert np.array_equal(read_back, composite)
    assert read_record["inputs"] == [digest(dapi_read), digest(merged_read)] == record["inputs"]
    assert read_record == record
