"""The segmentation input recipe: its stages, the model input, the route per FOV, the records and the replaced keys.

Rows RC1 to RC9, MI1 to MI6, PF1 and PF2, RE1 and RE2 and MG1 of
design/validation/segmentation-input-recipe.md (W-891). The fixtures are the W-307 golden images
(``seg_golden``, 16 × 64 × 64), the development preset's FOV with a registered morphology round
(``morphology_fov``, 9 × 32 × 32, from ``test_segmentation_segment.py``), hand-built stacks of at most
32 × 4 × 4, and model folders written here whose library calls are replaced by stubs that record what
they receive (no model is loaded, and no backend is needed). The base revision's hashes (RC1, PF2) were
computed at Starfinder 12a8f14 before the change (``test/data/segmentation_record_v1/README.md``).
"""
import json
import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from starfinder.image import ImageMetadata
from starfinder.preprocessing import (ClaheConfig, GaussianConfig, PercentileNormalizationConfig, ProjectionConfig,
                                      TophatConfig)
from starfinder.io.conversion import ImageConversionConfig
from starfinder.segmentation import (SEGMENTATION_METHODS, CellposeConfig, CompositeConfig, FlamingoEnhancementConfig,
                                     InputChannel, InputRecipe, LabelImportConfig, ModelInputConfig, RecipeStep,
                                     ReferenceGrid, SeededWatershedConfig, SegmentationInput, SegmentationPlan,
                                     SegmentationRun, StarDistConfig, apply_input_recipe, composite_nuclei_amplicon,
                                     segment)
from starfinder.segmentation._recipe import check_model_input, model_factor, recipe_sha256

from .coordination_helpers import digest
from .test_segmentation_golden import COMPOSITE_DIGESTS, FLAMINGO_DIGEST, SHAPE_ZYX, fixture
from .test_segmentation_segment import (EXTENSION, ProbeConfig, add_morphology_round, development_root,  # noqa: F401
                                        fov, register, write_mask)

pytestmark = [pytest.mark.segmentation, pytest.mark.validation, pytest.mark.contract]

F32 = {"rtol": 1e-6, "atol": 1e-6}
F64 = {"rtol": 1e-9, "atol": 1e-12}
GOLDEN = ReferenceGrid(SHAPE_ZYX, ImageMetadata("seg_golden", spacing_zyx=(0.35, 0.1, 0.1),
                                                spatial_unit="micrometer"), "declared")
#: The base revision's (12a8f14) array SHA-256 values, computed there before the change (RC1, PF2).
BASE = {"rc1_morphology_input": "2974fc2c374e196026e4a4bcf0390163249d7939ff50d17c44b6021ca35dce98",
        "rc1_golden_input": "be261aecdb8ee47ad7973b9e903fe2a2b6179d238ae1667ec28b092bf2ade8de",
        "pf2_input": "454af56774f18f7a164b33f46fffed9bdc62e01cdbf02503c4238bc6b90a86b2",
        "pf2_labels": "7a79df35c56696e981d2ccb18a21c1dd990765613dfebee631df04606ea079d7"}


@pytest.fixture(scope="module")
def images():
    return fixture()


def stages(record, stage):
    return [entry for entry in record["stages"] if entry["stage"] == stage]


def bare_grid(shape):
    return ReferenceGrid(shape, ImageMetadata("stacks"), "declared")


def project(stack, method, output_dtype=None, **options):
    """The projection stage alone on one source (no µm field, so a grid without spacing serves)."""
    recipe = InputRecipe(projection=ProjectionConfig(method=method, output_dtype=output_dtype))
    return apply_input_recipe([("nuclear", stack, {})], bare_grid(stack.shape), recipe=recipe, run="cell", **options)


# --- RC1: the empty recipe is today's input -------------------------------------------------------------------

def test_rc1_the_empty_recipe_is_the_input_of_the_base_revision(images, fov):
    stack = np.stack([images["dapi"], images["amplicon"]], axis=-1)
    built, unprojected, record = apply_input_recipe([("nuclear", images["dapi"], {}), ("amplicon", images["amplicon"],
                                                                                        {})], GOLDEN, recipe=None)
    assert built.image.dtype == np.uint8 and digest(built.image) == digest(stack) == BASE["rc1_golden_input"]
    assert unprojected is None and built.roles == ("nuclear", "amplicon") and built.model_scale_zyx is None
    assert [entry["stage"] for entry in record["stages"]] == ["combine"] and record["sha256"] == recipe_sha256(None)
    add_morphology_round(fov)
    from starfinder.segmentation._plan import assemble_input
    run = SegmentationRun("cell", "cell", (InputChannel("nuclear", "morph", "ch00"),
                                           InputChannel("amplicon", reference_merged=True)),
                          SeededWatershedConfig(sigma_um=0.1), seeds="nucleus")
    built = assemble_input(fov, run, fov.reference_grid())[0]
    assert built.image.dtype == np.float32 and digest(built.image) == BASE["rc1_morphology_input"]


# --- RC2: the order of the stages and roles -------------------------------------------------------------------------

def test_rc2_the_order_of_the_stages_and_the_roles(images):
    recipe = InputRecipe(channels=(RecipeStep(GaussianConfig(0.2), roles=("nuclear",)),),
                         projection=ProjectionConfig(method="max"),
                         after_projection=(RecipeStep(PercentileNormalizationConfig(1.0, 99.0)),))
    sources = [("nuclear", images["dapi"], {}), ("cytoplasm", images["flamingo"], {})]
    built, unprojected, record = apply_input_recipe(sources, GOLDEN, recipe=recipe)
    order = [(entry["stage"], entry.get("role")) for entry in record["stages"]]
    assert order == [("channels", "nuclear"), ("combine", None), ("projection", None),
                     ("after_projection", "nuclear"), ("after_projection", "cytoplasm")]
    step, combine, projection = record["stages"][:3]
    # The step restricted to nuclear leaves the other channel's array as it is.
    assert step["input"] == digest(images["dapi"]) and combine["inputs"] == [step["output"], digest(images["flamingo"])]
    assert projection["inputs"] == combine["outputs"]
    assert [entry["input"] for entry in record["stages"][3:]] == projection["outputs"]
    assert digest(built.image[..., 0]) == record["stages"][3]["output"]
    # The unprojected model channels are stacked as ZYXC (NumPy's promotion: float32), the flamingo values exact.
    assert unprojected.dtype == np.float32 and np.array_equal(unprojected[..., 1], images["flamingo"])
    assert projection["dtype"] == "float32" and built.image.dtype == np.float32
    assert step["dtype"] == "float32" and step["used"] == {"sigma_px_zyx": [0.2 / 0.35, 2.0, 2.0]}


# --- RC3: combine ----------------------------------------------------------------------------------------------

def test_rc3_combine(images):
    nuclear, amplicon, flamingo = images["dapi"], images["amplicon"], images["flamingo"]
    composite = apply_input_recipe([("nuclear", nuclear, {}), ("amplicon", amplicon, {})], GOLDEN,
                                   recipe=InputRecipe(combine=CompositeConfig()))[0]
    assert composite.roles == ("composite",) and composite.image.dtype == np.uint8
    assert digest(composite.image[..., 0]) == COMPOSITE_DIGESTS[False]
    enhanced, _, record = apply_input_recipe([("nuclear", nuclear, {}), ("cytoplasm", flamingo, {})], GOLDEN,
                                             recipe=InputRecipe(combine=FlamingoEnhancementConfig(median_radius_um=0.1)))
    assert enhanced.roles == ("nuclear",) and digest(enhanced.image[..., 0]) == FLAMINGO_DIGEST
    assert stages(record, "combine")[0]["used"] == {"median_radius_px": 1}
    # After a float32 gaussian on both sources: the composite of the two step outputs, each stretched into [0, 1].
    from skimage.exposure import rescale_intensity
    recipe = InputRecipe(channels=(GaussianConfig(0.2),), combine=CompositeConfig())
    built, _, record = apply_input_recipe([("nuclear", nuclear, {}), ("amplicon", amplicon, {})], GOLDEN, recipe=recipe)
    from starfinder.segmentation._inputs import _stretch
    from starfinder.segmentation._recipe import _working
    from starfinder.preprocessing import StepContext, run_step
    outputs = [run_step(_working(image), GaussianConfig(0.2), StepContext(role, "", GOLDEN.metadata)).image
               for role, image in (("nuclear", nuclear), ("amplicon", amplicon))]
    expected, _ = composite_nuclei_amplicon(*outputs)
    assert built.image.dtype == np.uint8 and np.array_equal(built.image[..., 0], expected)
    for output, q in zip(outputs, (0.005, 0.001)):
        stretched, _ = _stretch(output, q)
        assert stretched.min() >= 0.0 and stretched.max() <= 1.0
    assert rescale_intensity(outputs[0], (0.0, 1.0)).max() <= 1.0
    with pytest.raises(ValueError, match=r"takes the source roles \['amplicon', 'nuclear'\] exactly; the run's "
                                         r"sources have \['nuclear', 'cytoplasm'\]"):
        apply_input_recipe([("nuclear", nuclear, {}), ("cytoplasm", flamingo, {})], GOLDEN,
                           recipe=InputRecipe(combine=CompositeConfig()))


# --- RC4 to RC7: the projection --------------------------------------------------------------------------------------

def test_rc4_max_sum_and_mean_of_integers():
    stack = np.full((32, 4, 4), 65535, np.uint16)
    built, _, record = project(stack, "max")
    assert built.image.dtype == np.uint16 and np.all(built.image == 65535)
    built, _, record = project(stack, "sum")
    entry = stages(record, "projection")[0]
    assert built.image.dtype == np.float32 and np.all(built.image == 2097120.0) and entry["exact"] is True
    built, _, record = project(stack, "mean")
    entry = stages(record, "projection")[0]
    assert built.image.dtype == np.float32 and np.all(built.image == 65535.0) and entry["exact"] is None
    assert entry["rounding"] == "float64 result rounded once to float32"
    assert built.grid.metadata.frame_id.endswith("/projection:mean")
    for method, value in (("sum", 2097120.0), ("mean", 65535.0)):
        built, _, record = project(stack, method, "float64")
        assert built.image.dtype == np.float64 and np.all(built.image == value)
        assert stages(record, "projection")[0]["rounding"] == "float64 result rounded once to float64"
    with pytest.raises(ValueError, match="refuses conversion"):
        InputRecipe(projection=ProjectionConfig(method="sum", conversion=ImageConversionConfig("uint8", "cast")))


@pytest.mark.parametrize("dtype, planes, total, exact32, value32", [
    (np.uint32, (8388608, 8388608), 16777216, True, 16777216.0),
    (np.uint32, (8388608, 8388609), 16777217, False, 16777216.0),
    (np.int32, (-8388608, -8388608), -16777216, True, -16777216.0),
    (np.int32, (-8388608, -8388609), -16777217, False, -16777216.0)])
def test_rc5_the_exact_flag_at_and_above_two_to_the_24(dtype, planes, total, exact32, value32):
    stack = np.stack([np.full((4, 4), v, dtype) for v in planes])
    built, _, record = project(stack, "sum")
    assert stages(record, "projection")[0]["exact"] is exact32 and np.all(built.image == value32)
    built, _, record = project(stack, "sum", "float64")
    assert stages(record, "projection")[0]["exact"] is True and np.all(built.image == float(total))


def test_rc6_a_mean_that_is_not_an_integer():
    stack = np.stack([np.full((4, 4), v, np.uint8) for v in (0, 1, 1)])
    built, _, record = project(stack, "mean")
    entry = stages(record, "projection")[0]
    np.testing.assert_allclose(built.image, np.full((1, 4, 4, 1), 2 / 3), **F32)
    assert built.image.dtype == np.float32 and entry["exact"] is None
    assert entry["rounding"] == "float64 result rounded once to float32"
    built, _, _ = project(stack, "mean", "float64")
    np.testing.assert_allclose(built.image, np.full((1, 4, 4, 1), 2 / 3), **F64)


def test_rc7_the_overflow_check():
    stack = np.full((2, 4, 4), 2e38, np.float32)
    with pytest.raises(ValueError) as raised:
        project(stack, "sum")
    message = str(raised.value)
    expected = float(np.float64(np.float32(2e38)) * 2)
    assert "run 'cell'" in message and "projection sum" in message and repr(expected) in message
    assert "output_dtype: float64" in message
    built, _, record = project(stack, "sum", "float64")
    np.testing.assert_allclose(built.image, np.full((1, 4, 4, 1), expected), **F64)
    assert stages(record, "projection")[0]["max_abs"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    built, _, _ = project(stack, "mean")
    np.testing.assert_allclose(built.image, np.full((1, 4, 4, 1), np.float32(2e38)), **F32)


# --- RC8: the stain of extend_labels_through_z -------------------------------------------------------------------

@pytest.mark.dataset
def test_rc8_the_extension_reads_the_channel_before_the_projection(fov, register, monkeypatch):
    import starfinder.segmentation._plan as plan_module
    seen = []
    real = plan_module.extend_labels_through_z

    def recorded(labels, stain, metadata, *, config):
        seen.append(digest(stain))
        return real(labels, stain, metadata, config=config)
    monkeypatch.setattr(plan_module, "extend_labels_through_z", recorded)
    recipe = InputRecipe(channels=(GaussianConfig(0.1),), projection=ProjectionConfig())
    run = SegmentationRun("nucleus", "nucleus", (InputChannel("nuclear", fov.rounds.reference_round, "ch02"),),
                          ProbeConfig(level=-1.0), input_recipe=recipe, operations=(EXTENSION,))
    fov.segment(SegmentationPlan((run,)))
    record = fov.segmentation_results["nucleus"].record
    (step,) = stages(record["input"]["recipe"], "channels")
    assert seen == [step["output"]] and record["geometry"] == "extended"
    channel = (InputChannel("nuclear", fov.rounds.reference_round, "ch02"),)
    for bad in (InputRecipe(channels=(GaussianConfig(0.1),)), None):
        with pytest.raises(ValueError, match="needs an input recipe with a projection and one model channel"):
            SegmentationRun("nucleus", "nucleus", channel, ProbeConfig(), input_recipe=bad, operations=(EXTENSION,))
    two = (InputChannel("nuclear", fov.rounds.reference_round, "ch02"),
           InputChannel("cytoplasm", fov.rounds.reference_round, "ch01"))
    with pytest.raises(ValueError, match="one model channel"):
        SegmentationRun("nucleus", "nucleus", two, ProbeConfig(), input_recipe=InputRecipe(projection=ProjectionConfig()),
                        operations=(EXTENSION,))


# --- RC9: the recipe's identity -------------------------------------------------------------------------------------

def test_rc9_one_recipe_by_name_and_inline_has_one_sha256():
    from starfinder.dataset.workflow import _segmentation_block
    recipe = {"channels": [{"method": "gaussian", "sigma_um": 0.3}], "projection": {"method": "sum"},
              "after_projection": [{"method": "clahe", "kernel_um": 6.0, "clip_limit": 0.01}],
              "model_input": {"pixel_size_um": 0.37854}}
    runs = [{"name": "named", "target": "nucleus", "method": "stardist", "model": "2D_versatile_fluo",
             "inputs": [{"role": "nuclear", "round": "reference_stain", "channel": "PI"}], "input_recipe": "pi_smooth"},
            {"name": "inline", "target": "nucleus", "method": "stardist", "model": "2D_versatile_fluo",
             "inputs": [{"role": "nuclear", "round": "reference_stain", "channel": "PI"}], "input_recipe": recipe}]
    plan, _ = _segmentation_block({"backend": "python", "segmentation": {"input_recipes": {"pi_smooth": recipe},
                                                                         "runs": runs}})
    named, inline = (run.input_recipe for run in plan.runs)
    assert (named.name, inline.name) == ("pi_smooth", None)
    assert recipe_sha256(named) == recipe_sha256(inline)
    changes = [dict(recipe, channels=[{"method": "gaussian", "sigma_um": 0.31}]),
               dict(recipe, projection={"method": "mean"}),
               dict(recipe, after_projection=[{"method": "clahe", "kernel_um": 6.0, "clip_limit": 0.02}]),
               dict(recipe, model_input={"pixel_size_um": 0.38}),
               dict(recipe, model_input={"factor": 0.25})]
    from starfinder.dataset.workflow import _input_recipe
    shas = {recipe_sha256(_input_recipe(change, "recipe")) for change in changes}
    assert len(shas) == len(changes) and recipe_sha256(named) not in shas


# --- MI1, MI2: the factor --------------------------------------------------------------------------------------------

def test_mi1_the_factor_from_a_pixel_size():
    for spacing, factor in ((0.094635, 0.25), (0.18927, 0.5)):
        record = model_factor(ModelInputConfig(pixel_size_um=0.37854), ImageMetadata("g", (1.0, spacing, spacing)))
        np.testing.assert_allclose(record["factor_zyx"], [1.0, factor, factor], **F64)
        np.testing.assert_allclose(record["factor_zyx_exact"], [1.0, spacing / 0.37854, spacing / 0.37854], **F64)
        assert record["factor_zyx"][1] == factor                   # six significant digits remove the float noise
    zyx = model_factor(ModelInputConfig(pixel_size_um=(2.0, 0.5, 0.5)), ImageMetadata("g", (1.0, 0.25, 0.25)))
    np.testing.assert_allclose(zyx["factor_zyx"], [0.5, 0.5, 0.5], **F64)


def test_mi2_a_factor_per_route():
    for k, expected in ((1, 0.25), (2, 0.5)):
        record = model_factor(ModelInputConfig(factor=0.25), None, k=k)
        np.testing.assert_allclose(record["factor_zyx"], [1.0, expected, expected], **F64)
        assert record["k"] == k and record["spacing_yx"] is None


# --- MI3, MI4: what the library receives -----------------------------------------------------------------------------

CALLS = []


class _Device:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _StarDistStub:
    """StarDist2D and StarDist3D: records the scale ``predict_instances`` receives and returns no object."""

    def __init__(self, config, name, basedir):
        self.thresholds = SimpleNamespace(prob=0.5, nms=0.4)

    def predict_instances(self, x, axes, scale, **options):
        CALLS.append({"scale": scale, "axes": axes, "shape": x.shape})
        return np.zeros(x.shape, np.int32), {"prob": []}


class _CellposeStub:
    """CellposeModel: records the keyword arguments of ``eval`` and returns no object."""

    def __init__(self, pretrained_model, device, use_bfloat16):
        pass

    def eval(self, x, **kwargs):
        CALLS.append(kwargs)
        shape = x.shape[:3] if kwargs.get("do_3D") else (x.shape[:2] if kwargs.get("channel_axis") else x.shape)
        return (np.zeros(shape, np.int32),)


@pytest.fixture
def stubs(monkeypatch):
    """The two learned methods with their libraries replaced by stubs (tensorflow, stardist, torch, cellpose)."""
    tensorflow = types.ModuleType("tensorflow")
    tensorflow.__version__ = "stub"
    tensorflow.config = SimpleNamespace(
        list_physical_devices=lambda kind: [],
        threading=SimpleNamespace(get_intra_op_parallelism_threads=lambda: 1,
                                  get_inter_op_parallelism_threads=lambda: 1))
    tensorflow.sysconfig = SimpleNamespace(get_build_info=lambda: {})
    tensorflow.device = lambda name: _Device()
    stardist, stardist_models = types.ModuleType("stardist"), types.ModuleType("stardist.models")
    stardist_models.StarDist2D = stardist_models.StarDist3D = _StarDistStub
    torch = types.ModuleType("torch")
    torch.__version__ = "stub"
    torch.version = SimpleNamespace(cuda=None)
    torch.get_num_threads = torch.get_num_interop_threads = lambda: 1
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    cellpose, cellpose_models = types.ModuleType("cellpose"), types.ModuleType("cellpose.models")
    cellpose_models.CellposeModel = _CellposeStub
    for name, module in (("tensorflow", tensorflow), ("stardist", stardist), ("stardist.models", stardist_models),
                         ("torch", torch), ("cellpose", cellpose), ("cellpose.models", cellpose_models)):
        monkeypatch.setitem(sys.modules, name, module)
    for config in (StarDistConfig, CellposeConfig):
        monkeypatch.setitem(SEGMENTATION_METHODS, config, replace(SEGMENTATION_METHODS[config], requires=()))
    CALLS.clear()
    return CALLS


def stardist_folder(tmp_path, n_dim):
    folder = tmp_path / f"stardist_{n_dim}d"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"n_dim": n_dim, "grid": [1] * n_dim}))
    (folder / "thresholds.json").write_text(json.dumps({"prob": 0.5, "nms": 0.4}))
    (folder / "weights_best.h5").write_bytes(b"stub")
    return folder


def cellpose_file(tmp_path):
    path = tmp_path / "cellpose_model"
    path.write_bytes(b"stub")
    return path


def model_input(shape, factor, roles=("nuclear",)):
    image = np.zeros((*shape, len(roles)), np.uint8)
    image[..., 0] = 1
    return SegmentationInput(image, ReferenceGrid(shape, ImageMetadata("model", (1.0, 0.25, 0.25)), "declared"),
                             roles, model_scale_zyx=factor)


def test_mi3a_stardist_with_a_2d_model(tmp_path, stubs):
    config = StarDistConfig(model_path=str(stardist_folder(tmp_path, 2)))
    result = segment(model_input((1, 16, 16), (1.0, 0.25, 0.25)), config=config, target="nucleus",
                     label_namespace='["mi3a"]')
    (call,) = stubs
    assert len(call["scale"]) == 2 and call["axes"] == "YX"
    np.testing.assert_allclose(call["scale"], [0.25, 0.25], **F64)
    np.testing.assert_allclose(result.record["methods"][0]["effective"]["scale_zyx"], [1.0, 0.25, 0.25], **F64)
    segment(model_input((1, 16, 16), (1.0, 1.0, 1.0)), config=config, target="nucleus", label_namespace='["mi3a"]')
    assert stubs[-1]["scale"] is None
    with pytest.raises(ValueError, match="a 2D model scales Y and X only"):
        segment(model_input((1, 16, 16), (0.5, 0.25, 0.25)), config=config, target="nucleus",
                label_namespace='["mi3a"]')


def test_mi3b_stardist_with_a_3d_model(tmp_path, stubs):
    config = StarDistConfig(model_path=str(stardist_folder(tmp_path, 3)))
    result = segment(model_input((4, 16, 16), (1.0, 0.25, 0.25)), config=config, target="nucleus",
                     label_namespace='["mi3b"]')
    (call,) = stubs
    assert len(call["scale"]) == 3 and call["axes"] == "ZYX"
    np.testing.assert_allclose(call["scale"], [1.0, 0.25, 0.25], **F64)
    np.testing.assert_allclose(result.record["methods"][0]["effective"]["scale_zyx"], [1.0, 0.25, 0.25], **F64)


def test_mi4_cellpose_receives_the_diameter(tmp_path, stubs):
    config = CellposeConfig(model_path=str(cellpose_file(tmp_path)))
    for factor, diameter in ((0.25, 120.0), (1.0, 30.0)):
        result = segment(model_input((1, 16, 16), (1.0, factor, factor), ("cytoplasm",)), config=config,
                         target="cell", label_namespace='["mi4"]')
        assert stubs[-1]["diameter"] == pytest.approx(diameter, rel=1e-9, abs=1e-12)
        effective = result.record["methods"][0]["effective"]
        # Cellpose 4.2's eval rescales by 30 / diameter: diameter 30 is rescale 1.0, as diameter None was.
        assert effective["rescale"] == pytest.approx(factor, rel=1e-9, abs=1e-12)
    with pytest.raises(ValueError, match="one factor"):
        segment(model_input((1, 16, 16), (1.0, 0.25, 0.5), ("cytoplasm",)), config=config, target="cell",
                label_namespace='["mi4"]')
    with pytest.raises(ValueError, match="one factor"):
        check_model_input(config, InputRecipe(model_input=ModelInputConfig(factor=(1.0, 0.25, 0.5))), run="cell")


def test_mi4b_a_3d_cellpose_run(tmp_path, stubs):
    """Amendment A, on the grid of the validation page: spacing (1.0, 0.25, 0.25) µm and anisotropy 4."""
    config = CellposeConfig(model_path=str(cellpose_file(tmp_path)), do_3d=True, anisotropy=4.0)
    metadata = ImageMetadata("mi4b", (1.0, 0.25, 0.25))
    number = InputRecipe(model_input=ModelInputConfig(pixel_size_um=1.0))
    factor = model_factor(number.model_input, metadata)["factor_zyx"]
    check_model_input(config, number, run="cell", factor_zyx=factor)
    result = segment(model_input((4, 16, 16), tuple(factor), ("cytoplasm",)), config=config, target="cell",
                     label_namespace='["mi4b"]')
    call = stubs[-1]
    assert call["diameter"] == pytest.approx(120.0, rel=1e-9, abs=1e-12) and call["anisotropy"] == 4.0
    np.testing.assert_allclose(result.record["methods"][0]["effective"]["scale_zyx"], [1.0, 0.25, 0.25], **F64)
    accepted = InputRecipe(model_input=ModelInputConfig(pixel_size_um=(1.0, 1.0, 1.0)))
    check_model_input(config, accepted, run="cell",
                      factor_zyx=model_factor(accepted.model_input, metadata)["factor_zyx"])
    refused = InputRecipe(model_input=ModelInputConfig(pixel_size_um=(2.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match=r"run 'cell': input_recipe.model_input sets the Z factor 0.5, but a 3D "
                                         r"Cellpose run scales Z by anisotropy × f_yx = 1.0"):
        check_model_input(config, refused, run="cell",
                          factor_zyx=model_factor(refused.model_input, metadata)["factor_zyx"])
    # A factor given ZYX is checked when the run is built.
    with pytest.raises(ValueError, match="sets the Z factor 0.5"):
        SegmentationRun("cell", "cell", (InputChannel("cytoplasm", "round1", 0),), config,
                        input_recipe=InputRecipe(model_input=ModelInputConfig(factor=(0.5, 0.25, 0.25))))


@pytest.mark.dataset
def test_mi4b_through_fov_segment(fov, tmp_path, stubs):
    """The same rule through FOV.segment on the development grid (0.35, 0.1, 0.1) µm, anisotropy 3.5: a number
    pixel size of 0.4 µm is f_yx 0.25 and Cellpose's Z factor 0.875; a ZYX pixel size whose Z factor is 0.875 is
    accepted, one of 0.5 refused before the method runs."""
    config = CellposeConfig(model_path=str(cellpose_file(tmp_path)), do_3d=True, anisotropy=3.5)
    channel = (InputChannel("cytoplasm", fov.rounds.reference_round, "ch01"),)
    for pixel in (0.4, (0.4, 0.4, 0.4)):
        run = SegmentationRun("cell", "cell", channel, config,
                              input_recipe=InputRecipe(model_input=ModelInputConfig(pixel_size_um=pixel)))
        fov.segment(SegmentationPlan((run,)))
        effective = fov.segmentation_results["cell"].record["methods"][0]["effective"]
        np.testing.assert_allclose(effective["scale_zyx"], [0.875, 0.25, 0.25], **F64)
        assert stubs[-1]["diameter"] == pytest.approx(120.0, rel=1e-9, abs=1e-12)
    calls = len(stubs)
    run = SegmentationRun("cell", "cell", channel, config,
                          input_recipe=InputRecipe(model_input=ModelInputConfig(pixel_size_um=(0.7, 0.4, 0.4))))
    with pytest.raises(ValueError, match=r"run 'cell': .*Z factor 0.5, but a 3D Cellpose run scales Z by "
                                         r"anisotropy × f_yx = 0.875"):
        fov.segment(SegmentationPlan((run,)))
    assert len(stubs) == calls


# --- MI5, MI6: refusals and the length table ----------------------------------------------------------------------

def test_mi5_refusals():
    nuclear = (InputChannel("nuclear", "round1", 0),)
    with pytest.raises(ValueError, match="needs input_recipe.model_input"):
        SegmentationRun("nucleus", "nucleus", nuclear, StarDistConfig(model="2D_versatile_fluo"))
    with pytest.raises(ValueError, match="refuses input_recipe.model_input: cell outlines are never computed on a "
                                         "shrunk grid"):
        SegmentationRun("cell", "cell", (InputChannel("amplicon", reference_merged=True),), SeededWatershedConfig(),
                        seeds="nucleus", input_recipe=InputRecipe(model_input=ModelInputConfig(factor=0.5)))
    for kwargs in ({}, {"pixel_size_um": 0.4, "factor": 0.5}):
        with pytest.raises(ValueError, match="exactly one of pixel_size_um and factor"):
            ModelInputConfig(**kwargs)
    with pytest.raises(ValueError, match="model_input.pixel_size_um needs the grid's spacing") as raised:
        model_factor(ModelInputConfig(pixel_size_um=0.4), ImageMetadata("no spacing"))
    assert "voxel_size_xy and voxel_size_z" in str(raised.value)
    with pytest.raises(ValueError, match="scales its input for its model and needs model_scale_zyx"):
        segment(SegmentationInput(np.zeros((1, 4, 4, 1), np.uint8), bare_grid((1, 4, 4)), ("nuclear",)),
                config=StarDistConfig(model="2D_versatile_fluo"), target="nucleus", label_namespace='["mi5"]')


def test_mi6_the_length_table_has_no_model_scale():
    from starfinder.export._segment import LENGTH_FIELDS
    assert "scale" not in LENGTH_FIELDS["stardist"] and "diameter" not in LENGTH_FIELDS["cellpose"]


# --- PF1, PF2: the route per FOV ------------------------------------------------------------------------------------

def two_runs(path, inputs, recipe):
    return SegmentationPlan((SegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(path), "nucleus")),
                             SegmentationRun("cell", "cell", inputs, SeededWatershedConfig(sigma_um=0.1),
                                             seeds="nucleus", input_recipe=recipe)))


@pytest.mark.dataset
def test_pf1_fov_segment_with_a_recipe(fov, tmp_path, monkeypatch):
    from starfinder.dataset import CheckpointConfig
    from starfinder.io import load_volume
    import starfinder.segmentation._plan as plan_module
    captured = []
    real = plan_module.segment

    def capture(segmentation_input, **kwargs):
        captured.append((segmentation_input, kwargs))
        return real(segmentation_input, **kwargs)
    monkeypatch.setattr(plan_module, "segment", capture)
    path, _ = write_mask(fov, tmp_path)
    recipe = InputRecipe(channels=(GaussianConfig(0.15),),
                         after_projection=(RecipeStep(ClaheConfig(1.6)),), name="smooth")
    fov.segment(two_runs(path, (InputChannel("amplicon", reference_merged=True),), recipe),
                checkpoints=CheckpointConfig(directory=tmp_path / "ck"))
    (received, kwargs), = [c for c in captured if c[1]["target"] == "cell"]
    cell = fov.segmentation_results["cell"]
    folder = tmp_path / "ck" / "FOV_001" / "segmentation" / "cell"
    saved = load_volume(folder / "input.ome.tif").image
    assert saved.dtype == received.image.dtype == np.float32
    np.testing.assert_array_equal(saved.reshape(received.image.shape), received.image)
    assert cell.record["input"]["sha256"] == digest(received.image)
    direct = real(received, config=SeededWatershedConfig(sigma_um=0.1), target="cell", seeds=kwargs["seeds"],
                  label_namespace=kwargs["label_namespace"])
    assert np.array_equal(cell.labels, direct.labels)


@pytest.mark.dataset
def test_pf2_the_composite_run_in_its_new_form(fov, tmp_path):
    add_morphology_round(fov)
    path, _ = write_mask(fov, tmp_path)
    inputs = (InputChannel("nuclear", "morph", "ch00"), InputChannel("amplicon", reference_merged=True))
    fov.segment(two_runs(path, inputs, InputRecipe(combine=CompositeConfig())))
    cell = fov.segmentation_results["cell"]
    assert cell.record["input"]["dtype"] == "uint8"
    assert cell.record["input"]["sha256"] == BASE["pf2_input"] and digest(cell.labels) == BASE["pf2_labels"]


# --- RE1, RE2: the records ----------------------------------------------------------------------------------------

RECIPE_KEYS = {"name", "source", "sha256", "config", "spacing_zyx", "spatial_unit", "spacing_source", "fit_region", "fit_pixels",
               "stages", "model_input"}
STEP_KEYS = {"stage", "index", "role", "method", "config", "used", "fitted", "diagnostics", "input", "output", "dtype"}


@pytest.mark.dataset
def test_re1_the_record_of_format_version_2(fov, tmp_path, stubs):
    from starfinder.dataset import CheckpointConfig
    path, _ = write_mask(fov, tmp_path)
    recipe = InputRecipe(channels=(GaussianConfig(0.2),), projection=ProjectionConfig(method="sum"),
                         after_projection=(ClaheConfig(1.6),), model_input=ModelInputConfig(pixel_size_um=0.4),
                         name="pi_smooth")
    run = SegmentationRun("nucleus", "nucleus", (InputChannel("nuclear", fov.rounds.reference_round, "ch02"),),
                          StarDistConfig(model_path=str(stardist_folder(tmp_path, 2))), input_recipe=recipe)
    fov.segment(SegmentationPlan((run,)), checkpoints=CheckpointConfig(directory=tmp_path / "ck"))
    record = json.loads((tmp_path / "ck" / "FOV_001" / "segmentation" / "nucleus" / "segmentation.json").read_text())
    assert record["format_version"] == 2 and "projection" not in record["input"]
    assert not {"prepare", "prepare_channel"} & {key for c in record["input"]["channels"] for key in c}
    recipe_record = record["input"]["recipe"]
    assert set(recipe_record) == RECIPE_KEYS
    assert (recipe_record["name"], recipe_record["source"], recipe_record["fit_region"]) == ("pi_smooth", "given",
                                                                                             "image")
    assert recipe_record["spacing_source"] == "run grid before projection" and recipe_record["fit_pixels"] == 9 * 32 * 32
    np.testing.assert_allclose(recipe_record["spacing_zyx"], [0.35, 0.1, 0.1], **F64)
    assert recipe_record["spatial_unit"] == "micrometer"
    assert [s["stage"] for s in recipe_record["stages"]] == ["channels", "combine", "projection", "after_projection"]
    for entry in stages(recipe_record, "channels") + stages(recipe_record, "after_projection"):
        assert set(entry) == STEP_KEYS
    combine, projection = stages(recipe_record, "combine")[0], stages(recipe_record, "projection")[0]
    assert {"method", "config", "record", "roles_in", "roles_out"} <= set(combine)
    assert {"method", "config", "applied", "exact", "rounding", "max_abs", "dtype", "grid"} <= set(projection)
    assert projection["applied"] is True and projection["grid"]["shape_zyx"] == [1, 32, 32]
    model = recipe_record["model_input"]
    assert set(model) == {"pixel_size_um", "factor", "k", "spacing_yx", "factor_zyx_exact", "factor_zyx",
                          "method_value"}
    np.testing.assert_allclose(model["factor_zyx"], [1.0, 0.25, 0.25], **F64)
    np.testing.assert_allclose(model["method_value"]["scale_zyx"], [1.0, 0.25, 0.25], **F64)
    np.testing.assert_allclose(model["method_value"]["library_scale"], [0.25, 0.25], **F64)
    assert record["input"]["dtype"] == "float32" and record["input"]["path"] == "input.ome.tif"


@pytest.mark.dataset
def test_re2_a_version_1_record_still_loads(fov, tmp_path):
    """The record written at 12a8f14 (test/data/segmentation_record_v1): its label and input files are written again
    from the same arrays (PF2 shows the new form gives them), and FOV.load_segmentation restores the run."""
    from pathlib import Path

    from starfinder.dataset import CheckpointConfig
    from starfinder.io.tiff import save_volume
    from starfinder.segmentation._persist import write_labels
    from starfinder._records import file_sha256
    add_morphology_round(fov)
    path, _ = write_mask(fov, tmp_path)
    inputs = (InputChannel("nuclear", "morph", "ch00"), InputChannel("amplicon", reference_merged=True))
    computed = {}
    fov.segment(two_runs(path, inputs, InputRecipe(combine=CompositeConfig())))
    from starfinder.segmentation._plan import assemble_input
    cell = fov.segmentation_results["cell"]
    run = two_runs(path, inputs, InputRecipe(combine=CompositeConfig())).runs[1]
    computed["input"] = assemble_input(fov, run, fov.reference_grid())[0]
    record = json.loads((Path(__file__).parent / "data" / "segmentation_record_v1" / "segmentation.json").read_text())
    assert record["format_version"] == 1 and record["input"]["projection"] is None
    assert record["labels"]["sha256"] == digest(cell.labels) and record["input"]["sha256"] == digest(
        computed["input"].image)
    folder = tmp_path / "ck" / "FOV_001" / "segmentation" / "cell"
    folder.mkdir(parents=True)
    write_labels(folder / "labels.tif", cell.labels, cell.grid.metadata)
    save_volume(np.asarray(computed["input"].image), folder / "input.ome.tif", compress=True,
                metadata=computed["input"].grid.metadata)
    # The writers' bytes depend on the library version, not on the record's format: a file hash that differs in
    # this environment from the one the base revision recorded is set to the file written here.
    for entry, name in ((record["labels"], "labels.tif"), (record["input"], "input.ome.tif")):
        entry["file_sha256"] = file_sha256(folder / name)
    (folder / "segmentation.json").write_text(json.dumps(record))
    loaded = type(fov)(fov.dataset, fov.fov_id).load_segmentation("cell",
                                                                   checkpoints=CheckpointConfig(directory=tmp_path / "ck"))
    assert np.array_equal(loaded.labels, cell.labels) and loaded.record["format_version"] == 1
    assert loaded.record["input"].get("recipe") is None             # the recipe is shown as none


# --- MG1: the replaced Python fields ------------------------------------------------------------------------------

@pytest.mark.parametrize("build", [
    lambda: InputChannel("composite", "round1", 0, prepare=CompositeConfig()),
    lambda: InputChannel("nuclear", "round1", 0, prepare_channel=1),
    lambda: SegmentationRun("cell", "cell", (InputChannel("amplicon", reference_merged=True),),
                            SeededWatershedConfig(), seeds="nucleus", projection=ProjectionConfig()),
    lambda: StarDistConfig(model="2D_versatile_fluo", scale=0.25),
    lambda: CellposeConfig(model="cpsam_v2", diameter=30.0)])
def test_mg1_the_replaced_python_fields_are_unknown(build):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build()


def test_the_recipe_refuses_a_tophat_in_pixels_and_admits_one_in_um():
    assert RecipeStep(TophatConfig(radius_um=0.3)).config.radius_um == 0.3
    with pytest.raises(ValueError, match="give radius_um"):
        InputRecipe(after_projection=(TophatConfig(),))
