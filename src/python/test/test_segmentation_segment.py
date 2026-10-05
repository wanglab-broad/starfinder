"""The §2.9 segment entry, the segmentation registry, the seeded watershed and FOV.segment (W-313).

Rows L8 and L9 of the engineering validation design in docs/assignment-algorithms.md, the
registry list of docs/segmentation-contract.md ("Method registry") and the coordination of
``FOV.segment`` ("Coordination per FOV", "Run record", "What the segment entry needs from
FOV.run and FOV.register_rounds"). L8 uses a test-only method registered with monkeypatch on
the W-307 golden DAPI image; L9 uses the hand-built ``seeded`` fixture
(``segmentation_fixtures.py``); the FOV tests use the development preset ``clean``, size
``small``. Every expected value comes from the geometry, a formula in the test or a pinned
golden image.
"""
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pytest
from skimage.measure import label

from starfinder._execution import THREAD_VARIABLES
from starfinder._registry import Dependency, config_type_for, names
from starfinder.dataset import (Dataset, PipelineConfig, RegistrationRecipe, RegistrationStep, RoundState)
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import ImageLoadConfig, save_volume
from starfinder.preprocessing import ProjectionConfig
from starfinder.registration import RegistrationSignalConfig, TranslationConfig
from starfinder.segmentation import (SEGMENTATION_METHODS, CompositeConfig, FlamingoEnhancementConfig, InputChannel,
    LabelImportConfig, MissingModelError, ModelHashMismatchError, ReferenceGrid, SeededWatershedConfig,
    SegmentationBackendUnavailableError, SegmentationInput, SegmentationPlan, SegmentationResult, SegmentationRun,
    SegmentationSpec, ZExtensionConfig, composite_nuclei_amplicon, enhance_with_flamingo, extend_labels_through_z,
    segment)
from starfinder.synthetic import development_scene_preset, generate_formed_scene

from .segmentation_fixtures import (BOXES_METADATA, BOXES_SHAPE, CELL_BOXES, NUCLEUS_BOXES, boxes, paint,
    seeded_stain)
from .test_segmentation_golden import SHAPE_ZYX, digest, fixture

pytestmark = [pytest.mark.segmentation, pytest.mark.contract]

NAMESPACE = '["data","sample","FOV_001",null,"probe"]'
GOLDEN_GRID = ReferenceGrid(SHAPE_ZYX, ImageMetadata("seg_golden", spacing_zyx=(0.35, 0.1, 0.1)), "declared")


def namespace(run):
    return json.dumps(["data", "sample", "FOV_001", None, run], separators=(",", ":"))


# --- The test-only method of L8 ---------------------------------------------------------------------

CALLS = []


@dataclass(frozen=True)
class ProbeConfig:
    """A test-only method: the connected components of channel 0 above level, with an optional fault."""
    level: float = 60.0
    fault: str | None = None   # shape, negative, float, drop_z, untupled, huge
    model_path: str | None = None
    model_sha256: Mapping | None = None
    method: str = field(default="probe", init=False)


class SubProbeConfig(ProbeConfig):
    pass


@dataclass(frozen=True)
class UnregisteredConfig:
    method: str = field(default="unregistered", init=False)


def probe_labels(image, level):
    """The probe's labels by formula: 1-connected components of channel 0 above level, as int32."""
    return label(image[..., 0] > level, connectivity=1).astype(np.int32)


def probe_run(image, config, context):
    CALLS.append(context)
    labels = probe_labels(image, config.level)
    if config.fault == "drop_z":
        labels = labels[0]
    elif config.fault == "shape":
        labels = labels[:, :-1]
    elif config.fault == "negative":
        labels = labels - 1
    elif config.fault == "float":
        labels = labels.astype(np.float64)
    elif config.fault == "huge":
        labels = labels.astype(np.int64) + 2 ** 32
    elif config.fault == "untupled":
        return labels
    return labels, {"effective": {"level": config.level}, "library_dtype": str(labels.dtype)}


PROBE = SegmentationSpec("probe", probe_run, targets=frozenset({"nucleus"}),
                         roles=frozenset({"nuclear", "composite", "cytoplasm"}),
                         required_roles=(frozenset({"nuclear", "composite"}),), seeds="optional",
                         dimensions=frozenset({2, 3}), models=False, devices=frozenset({"cpu"}))
MISSING = Dependency("starfinder_w313_absent_backend", "starfinder-w313-absent-backend", "stardist")


@pytest.fixture
def register(monkeypatch):
    """Register PROBE (or a variant of it) for ProbeConfig; monkeypatch removes it afterwards."""
    def put(**change):
        monkeypatch.setitem(SEGMENTATION_METHODS, ProbeConfig, replace(PROBE, **change))
    put()
    CALLS.clear()
    return put


@pytest.fixture(scope="module")
def dapi():
    return fixture()["dapi"]


def golden_input(dapi, roles=("nuclear",), z=None):
    image = dapi[..., None] if z is None else dapi[z:z + 1, ..., None]
    grid = GOLDEN_GRID if z is None else GOLDEN_GRID.projected()
    return SegmentationInput(image, grid, roles)


def nucleus_seeds(grid, labels=None):
    labels = np.zeros(grid.shape_zyx, np.uint32) if labels is None else labels
    return SegmentationResult(labels, grid, "nucleus", "plane" if grid.shape_zyx[0] == 1 else "volume",
                              namespace("nucleus"), {"run": "nucleus"})


def model_folder(tmp_path, n_dim=3):
    folder = tmp_path / "probe_model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"n_dim": n_dim}))
    (folder / "weights.bin").write_bytes(b"probe weights")
    return folder


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- L8: the stage wrapper --------------------------------------------------------------------------

@pytest.mark.validation
def test_l8_a_registered_method_runs_and_the_record_holds_the_provenance_entry(register, dapi):
    image = dapi[..., None]
    built = segment(golden_input(dapi), config=ProbeConfig(), target="nucleus", label_namespace=NAMESPACE)
    expected = probe_labels(image, 60.0).astype(np.uint32)
    assert built.labels.dtype == np.uint32 and np.array_equal(built.labels, expected)
    assert built.grid == GOLDEN_GRID and built.target == "nucleus" and built.geometry == "volume"
    assert built.label_namespace == NAMESPACE and len(CALLS) == 1
    context = CALLS[0]
    assert context.roles == ("nuclear",) and context.seeds is None and context.grid == GOLDEN_GRID
    assert context.device == "cpu" and context.model is None
    record = built.record
    entry = record["methods"][0]
    assert len(record["methods"]) == 1
    assert set(entry) == {"stage", "method", "config_type", "implementation", "config", "requires", "artifacts",
                          "execution", "effective"}
    assert (entry["stage"], entry["method"], entry["requires"], entry["artifacts"]) == (
        "segmentation", "probe", {}, [])
    assert entry["config_type"].endswith("ProbeConfig") and entry["implementation"].endswith("probe_run")
    assert entry["config"]["level"] == 60.0 and entry["config"]["method"] == "probe"
    assert entry["effective"] == {"level": 60.0}
    assert entry["execution"] == {"device": "cpu", "framework": None,
                                  "threads": {name: entry["execution"]["threads"][name] for name in THREAD_VARIABLES}}
    assert (record["format_version"], record["stage"], record["target"], record["geometry"]) == (
        1, "segmentation", "nucleus", "volume")
    assert (record["dataset_id"], record["sample_id"], record["fov_id"], record["subtile_id"], record["run"]) == (
        "data", "sample", "FOV_001", None, "probe")
    assert record["grid"] == {"shape_zyx": list(SHAPE_ZYX),
                              "metadata": json.loads(json.dumps(asdict(GOLDEN_GRID.metadata))),
                              "source": "declared", "sha256": None}
    assert record["input"] == {"path": None, "sha256": digest(image), "file_sha256": None,
                               "shape_zyxc": [*SHAPE_ZYX, 1], "dtype": "uint8",
                               "metadata": record["grid"]["metadata"], "projection": None,
                               "channels": [{"role": "nuclear"}]}
    n_labels = len(np.unique(expected)) - 1
    assert record["labels"] == {"dtype": "uint32", "sha256": digest(expected.astype(np.uint32)),
                                "n_labels": n_labels, "max_label": int(expected.max())}
    assert n_labels > 0 and built.n_labels == n_labels
    assert (record["seeds"], record["operations"], record["outcome"]) == (None, [], "ok")
    assert set(record["software"]) == {"code", "environment"}
    assert built.diagnostics["details"] == {"library_dtype": "int32"}


@pytest.mark.validation
def test_l8_segment_refuses_a_label_import_config(register, dapi):
    with pytest.raises(TypeError, match="import_labels imports masks; it is not a segmentation method"):
        segment(golden_input(dapi), config=LabelImportConfig("labels.tif", "nucleus"), target="nucleus",
                label_namespace=NAMESPACE)


def _shifted_input(dapi):
    built = golden_input(dapi)
    object.__setattr__(built, "image", dapi[:, :32, :, None])  # the wrapper re-checks the input's own shape
    return built


# Each raising check of the 11 wrapper checks, with its error. A case is (registry change, call
# arguments as a function of the fixtures, error, message).
CHECKS = {
    "1 config not registered": ({}, lambda d, t: dict(config=UnregisteredConfig()), TypeError,
                                "no segmentation method is registered for UnregisteredConfig"),
    "1 config subclass": ({}, lambda d, t: dict(config=SubProbeConfig()), TypeError, "exact config type"),
    "2 target of another method": ({}, lambda d, t: dict(target="cell"), ValueError, "produces nucleus, not 'cell'"),
    "2 unknown target": ({}, lambda d, t: dict(target="tissue"), ValueError, "target must be nucleus or cell"),
    "3 not an input": ({}, lambda d, t: dict(segmentation_input=d[..., None]), TypeError, "SegmentationInput"),
    "3 input shape": ({}, lambda d, t: dict(segmentation_input=_shifted_input(d)), IncompatibleGeometryError,
                      "differs from the grid"),
    "3 role not accepted": ({}, lambda d, t: dict(segmentation_input=golden_input(d, ("membrane",))), ValueError,
                            "does not accept the channel role 'membrane'"),
    "3 required role missing": ({}, lambda d, t: dict(segmentation_input=golden_input(d, ("cytoplasm",))),
                                ValueError, "needs a channel with the role 'composite' or 'nuclear'"),
    "4 unknown device": ({}, lambda d, t: dict(device="gpu"), ValueError, "device must be 'cpu' or 'cuda'"),
    "4 device of another method": ({}, lambda d, t: dict(device="cuda"), ValueError, "runs on cpu, not 'cuda'"),
    "5 dependency": (dict(requires=(MISSING,)), lambda d, t: {}, SegmentationBackendUnavailableError,
                     r"requires starfinder_w313_absent_backend; install the 'stardist' extra"),
    "6 no model named": (dict(models=True), lambda d, t: {}, MissingModelError, "needs a model"),
    "6 missing model path": (dict(models=True), lambda d, t: dict(config=ProbeConfig(model_path=str(t / "absent"))),
                             MissingModelError, "does not exist; segmentation never downloads"),
    "6 changed model file": (dict(models=True), lambda d, t: dict(config=ProbeConfig(
        model_path=str(model_folder(t)), model_sha256={"weights.bin": "0" * 64})), ModelHashMismatchError,
        f"weights.bin: SHA-256 {hashlib.sha256(b'probe weights').hexdigest()} differs from the expected {'0' * 64}"),
    "7 plane for a 3D-only method": (dict(dimensions=frozenset({3})),
                                     lambda d, t: dict(segmentation_input=golden_input(d, z=4)),
                                     IncompatibleGeometryError, "needs Z>1"),
    "7 volume for a 2D-only method": (dict(dimensions=frozenset({2})), lambda d, t: {}, IncompatibleGeometryError,
                                      "runs on one plane"),
    "7 3D model on a plane": (dict(models=True), lambda d, t: dict(
        config=ProbeConfig(model_path=str(model_folder(t, 3))), segmentation_input=golden_input(d, z=4)),
        IncompatibleGeometryError, "a 3D model cannot segment an input with Z=1"),
    "7 2D model on a volume": (dict(models=True), lambda d, t: dict(config=ProbeConfig(
        model_path=str(model_folder(t, 2)))), IncompatibleGeometryError, "a 2D model cannot segment"),
    "7 below the minimum shape": (dict(min_shape_zyx=(17, 1, 1)), lambda d, t: {}, IncompatibleGeometryError,
                                  "at least"),
    "8 seeds refused": (dict(seeds="none"), lambda d, t: dict(seeds=nucleus_seeds(GOLDEN_GRID)), ValueError,
                        "takes no seeds"),
    "8 seeds required": (dict(seeds="required"), lambda d, t: {}, ValueError, "grows from seeds"),
    "8 seeds not a result": ({}, lambda d, t: dict(seeds=np.zeros(SHAPE_ZYX, np.uint32)), TypeError,
                             "SegmentationResult"),
    "8 seeds of another target": ({}, lambda d, t: dict(seeds=replace(nucleus_seeds(GOLDEN_GRID), target="cell")),
                                  ValueError, "target 'nucleus'"),
    "8 seeds on another frame": ({}, lambda d, t: dict(seeds=nucleus_seeds(
        ReferenceGrid(SHAPE_ZYX, ImageMetadata("other"), "declared"))), IncompatibleGeometryError, "seeds are on"),
    "8 seeds of another shape": ({}, lambda d, t: dict(seeds=nucleus_seeds(
        ReferenceGrid((16, 64, 32), GOLDEN_GRID.metadata, "declared"))), IncompatibleGeometryError, "seeds are on"),
}


@pytest.mark.validation
@pytest.mark.parametrize("case", list(CHECKS))
def test_l8_each_check_raises_its_error_before_the_method_runs(register, dapi, tmp_path, case):
    change, arguments, error, message = CHECKS[case]
    register(**change)
    call = dict(segmentation_input=golden_input(dapi), config=ProbeConfig(), target="nucleus",
                label_namespace=NAMESPACE)
    call.update(arguments(dapi, tmp_path))
    with pytest.raises(error, match=message):
        segment(call.pop("segmentation_input"), **call)
    assert CALLS == []


# Two violations in one call: the earlier check raises (case: registry change, arguments, error, message).
PAIRS = {
    "1 before 2": ({}, lambda d, t: dict(config=LabelImportConfig("x.tif", "nucleus"), target="tissue"), TypeError,
                   "import_labels imports masks"),
    "2 before 3": ({}, lambda d, t: dict(target="cell", segmentation_input=golden_input(d, ("membrane",))),
                   ValueError, "produces nucleus"),
    "3 before 4": ({}, lambda d, t: dict(segmentation_input=golden_input(d, ("cytoplasm",)), device="cuda"),
                   ValueError, "needs a channel with the role"),
    "4 before 5": (dict(requires=(MISSING,)), lambda d, t: dict(device="cuda"), ValueError, "runs on cpu"),
    "5 before 6": (dict(requires=(MISSING,), models=True), lambda d, t: {}, SegmentationBackendUnavailableError,
                   "starfinder_w313_absent_backend"),
    "6 before 7": (dict(models=True, dimensions=frozenset({2})),
                   lambda d, t: dict(config=ProbeConfig(model_path=str(t / "absent"))), MissingModelError,
                   "does not exist"),
    "7 before 8": (dict(dimensions=frozenset({3}), seeds="required"),
                   lambda d, t: dict(segmentation_input=golden_input(d, z=4)), IncompatibleGeometryError,
                   "needs Z>1"),
}


@pytest.mark.validation
@pytest.mark.parametrize("case", list(PAIRS))
def test_l8_an_input_violating_two_checks_raises_the_earlier_one(register, dapi, tmp_path, case):
    change, arguments, error, message = PAIRS[case]
    register(**change)
    call = dict(segmentation_input=golden_input(dapi), config=ProbeConfig(), target="nucleus",
                label_namespace=NAMESPACE)
    call.update(arguments(dapi, tmp_path))
    with pytest.raises(error, match=message):
        segment(call.pop("segmentation_input"), **call)
    assert CALLS == []


@pytest.mark.validation
@pytest.mark.parametrize("fault, error, message", [
    ("shape", ValueError, r"returned labels of shape \(16, 63, 64\), not the input's \(16, 64, 64\)"),
    ("negative", ValueError, "must not be negative"),
    ("float", TypeError, "returned float64 labels"),
    ("huge", ValueError, "above 2\\*\\*32 - 1"),
    ("untupled", ValueError, r"must return \(labels, details\)"),
    ("drop_z", ValueError, "returned labels of shape"),   # a dropped Z is restored only for one plane
])
def test_l8_the_output_check_refuses_labels_off_the_contract(register, dapi, fault, error, message):
    with pytest.raises(error, match=message):
        segment(golden_input(dapi), config=ProbeConfig(fault=fault), target="nucleus", label_namespace=NAMESPACE)
    assert len(CALLS) == 1   # the output check runs after the method


@pytest.mark.validation
def test_l8_a_dropped_z_axis_is_restored_for_one_plane(register, dapi):
    built = segment(golden_input(dapi, z=4), config=ProbeConfig(fault="drop_z"), target="nucleus",
                    label_namespace=NAMESPACE)
    expected = probe_labels(dapi[4:5, ..., None], 60.0).astype(np.uint32)
    assert built.labels.shape == (1, 64, 64) and np.array_equal(built.labels, expected)
    assert built.geometry == "plane" and built.grid == GOLDEN_GRID.projected()
    assert built.record["geometry"] == "plane" and built.record["input"]["shape_zyxc"] == [1, 64, 64, 1]


@pytest.mark.validation
def test_l8_a_result_without_objects_has_outcome_empty(register, dapi):
    built = segment(golden_input(dapi), config=ProbeConfig(level=255.0), target="nucleus", label_namespace=NAMESPACE)
    assert not built.labels.any() and built.record["outcome"] == "empty"
    assert built.record["labels"]["n_labels"] == built.record["labels"]["max_label"] == 0


@pytest.mark.validation
def test_l8_a_resolved_model_is_recorded_as_artifacts(register, dapi, tmp_path):
    register(models=True)
    folder = model_folder(tmp_path)
    expected = {"config.json": file_sha256(folder / "config.json"), "weights.bin": file_sha256(folder / "weights.bin")}
    config = ProbeConfig(model_path=str(folder), model_sha256=expected)
    built = segment(golden_input(dapi), config=config, target="nucleus", label_namespace=NAMESPACE)
    assert CALLS[0].model == folder.resolve()
    assert built.record["methods"][0]["artifacts"] == [
        {"name": f"probe/probe_model/{name}", "path": str(folder.resolve() / name), "sha256": digest_,
         "source": "path"} for name, digest_ in sorted(expected.items())]


@pytest.mark.validation
def test_l8_seeds_reach_the_method_and_the_record(register, dapi):
    seeds = nucleus_seeds(GOLDEN_GRID, probe_labels(dapi[..., None], 90.0).astype(np.uint32))
    built = segment(golden_input(dapi), config=ProbeConfig(), target="nucleus", seeds=seeds,
                    label_namespace=NAMESPACE)
    assert CALLS[0].seeds is seeds.labels
    assert built.record["seeds"] == {"run": "nucleus", "label_namespace": namespace("nucleus"),
                                     "sha256": digest(seeds.labels), "label_rule": None}


def test_segmentation_input_validates_its_own_fields(dapi):
    image = dapi[..., None]
    assert golden_input(dapi).roles == ("nuclear",)
    cases = [
        (dict(roles=("tissue",)), ValueError, "unknown channel role 'tissue'"),
        (dict(image=np.concatenate([image, image], axis=-1), roles=("nuclear", "nuclear")), ValueError,
         "appears more than once"),
        (dict(roles=("nuclear", "cytoplasm")), ValueError, "one role per channel"),
        (dict(image=dapi), IncompatibleGeometryError, "ZYXC"),
        (dict(image=np.full(image.shape, np.nan)), ValueError, "finite"),
        (dict(image=image[:, :32]), IncompatibleGeometryError, "differs from the grid"),
        (dict(grid=GOLDEN_GRID.shape_zyx), TypeError, "ReferenceGrid"),
        (dict(roles=["nuclear"]), TypeError, "tuples"),
        (dict(sources=({}, {})), ValueError, "one mapping per channel"),
    ]
    for change, error, message in cases:
        fields = dict(image=image, grid=GOLDEN_GRID, roles=("nuclear",), sources=())
        fields.update(change)
        with pytest.raises(error, match=message):
            SegmentationInput(**fields)


@pytest.mark.parametrize("change, error", [
    (dict(name="Bad-Name"), ValueError), (dict(targets=frozenset({"tissue"})), ValueError),
    (dict(targets=frozenset()), ValueError), (dict(roles=frozenset({"stain"})), ValueError),
    (dict(required_roles=(frozenset({"membrane"}),)), ValueError), (dict(required_roles=[]), ValueError),
    (dict(seeds="maybe"), ValueError), (dict(dimensions=frozenset({1})), ValueError),
    (dict(models="yes"), TypeError), (dict(devices=frozenset({"tpu"})), ValueError),
    (dict(min_shape_zyx=(0, 1, 1)), ValueError), (dict(run="not callable"), TypeError)])
def test_spec_validation(change, error):
    with pytest.raises(error):
        replace(PROBE, **change)


# --- The registry list ------------------------------------------------------------------------------

def test_the_registry_holds_seeded_watershed_with_the_contract_fields():
    # stardist and cellpose (W-316) are checked in test_segmentation_models.py.
    assert set(names(SEGMENTATION_METHODS)) == {"stardist", "cellpose", "seeded_watershed"}
    for config_type, spec in SEGMENTATION_METHODS.items():
        assert config_type.__dataclass_fields__["method"].default == spec.name
        assert config_type_for(SEGMENTATION_METHODS, spec.name, "segmentation method") is config_type
    spec = SEGMENTATION_METHODS[SeededWatershedConfig]
    stains = frozenset({"cytoplasm", "membrane", "amplicon", "composite"})
    assert (spec.targets, spec.roles, spec.required_roles, spec.seeds, spec.dimensions, spec.models, spec.devices,
            spec.requires, spec.min_shape_zyx) == (frozenset({"cell"}), stains, (stains,), "required",
                                                   frozenset({2, 3}), False, frozenset({"cpu"}), (), (1, 1, 1))


def test_seeded_watershed_config_defaults_and_validation():
    assert SeededWatershedConfig() == SeededWatershedConfig(1.5, "otsu", 0.0, 1)
    assert SeededWatershedConfig(threshold=0.2).threshold == 0.2
    for change in (dict(sigma_um=0), dict(sigma_um=float("nan")), dict(threshold="mean"), dict(threshold=True),
                   dict(compactness=-1.0), dict(connectivity=4), dict(connectivity=True)):
        with pytest.raises(ValueError):
            SeededWatershedConfig(**change)


def test_an_inserted_method_is_seen_by_segment_and_a_run(register, dapi):
    assert "probe" in names(SEGMENTATION_METHODS)
    SegmentationRun("nucleus", "nucleus", (InputChannel("nuclear", "round1", 0),), ProbeConfig())
    assert segment(golden_input(dapi), config=ProbeConfig(), target="nucleus", label_namespace=NAMESPACE).n_labels


# --- L9: the seeded watershed -----------------------------------------------------------------------

BOXES_GRID = ReferenceGrid(BOXES_SHAPE, BOXES_METADATA, "declared")


def test_the_boxes_fixture_has_its_stated_geometry():
    cells, nuclei = boxes()
    in_cell = {value: {int(c): int(n) for c, n in zip(*np.unique(cells[nuclei == value], return_counts=True))}
               for value in NUCLEUS_BOXES}
    assert in_cell == {11: {1: 27}, 21: {2: 18}, 22: {2: 18}, 31: {3: 50, 4: 50}, 51: {5: 6, 0: 4},
                       61: {6: 8, 7: 2}, 91: {0: 12}}
    assert all(4 in range(*ranges[0]) for ranges in (*CELL_BOXES.values(), *NUCLEUS_BOXES.values()))
    (_, (y0, y1), (x0, x1)) = NUCLEUS_BOXES[91]
    for (_, (cy0, cy1), (cx0, cx1)) in CELL_BOXES.values():
        assert max(y0 - cy1 + 1, cy0 - y1 + 1) >= 5 or max(x0 - cx1 + 1, cx0 - x1 + 1) >= 5
        assert max(y0 - cy1 + 1, cy0 - y1 + 1, x0 - cx1 + 1, cx0 - x1 + 1) >= 5
    assert not (cells == 8)[nuclei > 0].any()


def seeded(seed_labels=None, grid=BOXES_GRID):
    _, nuclei = boxes()
    seeds = nucleus_seeds(grid, nuclei if seed_labels is None else seed_labels)
    return SegmentationInput(seeded_stain()[..., None], grid, ("cytoplasm",)), seeds


@pytest.mark.validation
def test_l9_every_seed_keeps_its_value_and_no_other_value_appears():
    segmentation_input, seeds = seeded()
    built = segment(segmentation_input, config=SeededWatershedConfig(sigma_um=0.1), target="cell", seeds=seeds,
                    label_namespace=namespace("cell"))
    cells, nuclei = built.labels, seeds.labels
    assert cells.dtype == np.uint32 and cells.shape == BOXES_SHAPE and built.grid == BOXES_GRID
    assert set(np.unique(cells)) - {0} == set(NUCLEUS_BOXES)
    assert np.array_equal(cells[nuclei > 0], nuclei[nuclei > 0])
    # Nucleus 91 lies in the background, 5 voxels from every stained cell: its cell is the seed exactly.
    assert np.array_equal(cells == 91, nuclei == 91)
    # The background part of nucleus 51 keeps its seed's value although it is outside the foreground.
    assert (cells[4, 12, 20:24] == 51).all()
    record = built.record
    assert (record["target"], record["geometry"], record["outcome"]) == ("cell", "volume", "ok")
    entry = record["methods"][0]
    assert (entry["method"], entry["requires"], entry["artifacts"]) == ("seeded_watershed", {}, [])
    assert entry["config"] == {"sigma_um": 0.1, "threshold": "otsu", "compactness": 0.0, "connectivity": 1,
                               "method": "seeded_watershed"}
    assert entry["effective"]["sigma_px_zyx"] == [0.1 / 0.35, 1.0, 1.0]
    assert entry["effective"]["threshold_source"] == "otsu" and 20 / 255 < entry["effective"]["threshold"] < 200 / 255
    assert record["seeds"] == {"run": "nucleus", "label_namespace": namespace("nucleus"), "sha256": digest(nuclei),
                               "label_rule": "seed_values"}
    assert built.diagnostics["details"]["counts"]["seeds_outside_foreground"] == [91]


@pytest.mark.validation
def test_l9_no_seeds_give_an_empty_label_image():
    segmentation_input, seeds = seeded(np.zeros(BOXES_SHAPE, np.uint32))
    built = segment(segmentation_input, config=SeededWatershedConfig(sigma_um=0.1), target="cell", seeds=seeds,
                    label_namespace=namespace("cell"))
    assert not built.labels.any() and built.record["outcome"] == "empty"


@pytest.mark.validation
def test_l9_a_grid_without_spacing_raises():
    grid = ReferenceGrid(BOXES_SHAPE, ImageMetadata("boxes"), "declared")
    segmentation_input, seeds = seeded(grid=grid)
    with pytest.raises(ValueError, match="needs a spacing"):
        segment(segmentation_input, config=SeededWatershedConfig(sigma_um=0.1), target="cell", seeds=seeds,
                label_namespace=namespace("cell"))


@pytest.mark.validation
def test_l9_the_plane_z4_runs_as_a_plane():
    _, nuclei = boxes()
    grid = ReferenceGrid((1, 32, 32), BOXES_METADATA, "declared")
    seeds = nucleus_seeds(grid, np.ascontiguousarray(nuclei[4:5]))
    built = segment(SegmentationInput(seeded_stain()[4:5, ..., None], grid, ("amplicon",)),
                    config=SeededWatershedConfig(sigma_um=0.1), target="cell", seeds=seeds,
                    label_namespace=namespace("cell"))
    assert built.geometry == "plane" and built.labels.shape == (1, 32, 32)
    assert set(np.unique(built.labels)) - {0} == set(NUCLEUS_BOXES)
    assert np.array_equal(built.labels[seeds.labels > 0], seeds.labels[seeds.labels > 0])
    assert built.record["methods"][0]["effective"]["sigma_px_zyx"] == [0.0, 1.0, 1.0]


def test_seeded_watershed_takes_one_stain_channel():
    _, seeds = seeded()
    image = np.repeat(seeded_stain()[..., None], 2, axis=-1)
    with pytest.raises(ValueError, match="one stain channel"):
        segment(SegmentationInput(image, BOXES_GRID, ("cytoplasm", "membrane")),
                config=SeededWatershedConfig(sigma_um=0.1), target="cell", seeds=seeds,
                label_namespace=namespace("cell"))


# --- FOV.segment ------------------------------------------------------------------------------------

SPACING = dict(spacing_zyx=(0.35, 0.1, 0.1), spatial_unit="micrometer")
STAIN = RegistrationSignalConfig("channel", reference_channel="ch02", moving_channel="ch02")
MORPH_RECIPE = RegistrationRecipe((RegistrationStep(TranslationConfig()),), signal=STAIN)
EXTENSION = ZExtensionConfig(median_um=0.1, threshold=0.0, min_area_um2=0.0, dilation_um=0.0, fill_holes="once")


@pytest.fixture(scope="module")
def development_root(tmp_path_factory):
    """The development preset 'clean', size 'small' (9×32×32, four channels, three rounds), as TIFFs.

    The metadata gains a spacing of (0.35, 0.1, 0.1) µm, which the watershed needs.
    """
    book, config = development_scene_preset("clean", size="small")
    scene = generate_formed_scene(book, config=config)
    root = tmp_path_factory.mktemp("development")
    for name, image in scene.rounds.items():
        metadata = replace(scene.round_metadata[name], **SPACING)
        for c, channel in enumerate(scene.channel_labels):
            save_volume(image[..., c], root / name / "FOV_001" / f"{channel}.tif", metadata=metadata)
    return root, scene


@pytest.fixture
def fov(development_root):
    """A FOV after FOV.run (load and translation registration) of the development preset."""
    root, scene = development_root
    rounds = RoundState(list(scene.round_labels), reference_round=scene.round_labels[0])
    dataset = Dataset(root, root / "out", "dev", "sample", "out", rounds, list(scene.channel_labels))
    pipeline = PipelineConfig(load=ImageLoadConfig(channel_labels=tuple(scene.channel_labels)),
                              registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)))
    return dataset.fov("FOV_001").run(pipeline)


def nucleus_mask(shape):
    """Hand-built nuclei on the development grid: 7 and 12, boxes that include every plane."""
    return paint({7: ((0, shape[0]), (4, 9), (4, 10)), 12: ((0, shape[0]), (18, 24), (16, 21))}, shape)


def write_mask(fov, tmp_path, *, plane=False):
    grid = fov.reference_grid()
    if plane:
        grid = grid.projected()
    path = tmp_path / ("nuclei_plane.tif" if plane else "nuclei.tif")
    mask = nucleus_mask(grid.shape_zyx)
    if plane:
        import tifffile
        tifffile.imwrite(path, mask[0])
    else:
        save_volume(mask, path, metadata=grid.metadata)
    return path, mask


def two_run_plan(path, cell_inputs=(InputChannel("amplicon", reference_merged=True),), **cell):
    return SegmentationPlan((
        SegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(path), "nucleus")),
        SegmentationRun("cell", "cell", cell_inputs, SeededWatershedConfig(sigma_um=0.1), seeds="nucleus", **cell)))


def merged(fov):
    return fov.images[fov.rounds.reference_round].max(axis=3)


def add_morphology_round(fov, *, register=True):
    """morph: the reference round displaced by (0, 2, −1), registered back through its ch02 stain."""
    ref = fov.rounds.reference_round
    fov.images["morph"] = np.ascontiguousarray(np.roll(fov.images[ref], (0, 2, -1), axis=(0, 1, 2)))
    fov.metadata["morph"] = replace(fov.metadata[ref], frame_id="morph")
    if register:
        fov.register_rounds(MORPH_RECIPE, rounds=["morph"])
    return fov


@pytest.mark.dataset
def test_a_two_run_plan_imports_the_nuclei_and_grows_the_cells(fov, tmp_path):
    path, mask = write_mask(fov, tmp_path)
    grid = fov.reference_grid()
    assert fov.segment(two_run_plan(path)) is fov
    results = fov.segmentation_results
    assert set(results) == {"nucleus", "cell"}
    nucleus, cell = results["nucleus"], results["cell"]
    grid_record = {"shape_zyx": list(grid.shape_zyx), "metadata": json.loads(json.dumps(asdict(grid.metadata))),
                   "source": grid.source, "sha256": grid.sha256}
    assert nucleus.grid == cell.grid == grid
    assert nucleus.record["grid"] == cell.record["grid"] == grid_record
    # The import run: no method entry and an import entry.
    assert nucleus.record["methods"] == [] and nucleus.record["import"]["path"] == str(path)
    assert nucleus.record["import"]["metadata_source"] == "stored" and np.array_equal(nucleus.labels, mask)
    # The cell run names its seed run and the seed labels' SHA-256; its input is the reference merged image.
    assert cell.record["seeds"] == {"run": "nucleus", "label_namespace": nucleus.label_namespace,
                                    "sha256": digest(mask), "label_rule": "seed_values"}
    assert cell.record["seeds"]["sha256"] == nucleus.record["labels"]["sha256"]
    image = merged(fov)[..., None]
    assert cell.record["input"]["sha256"] == digest(image)
    assert cell.record["input"]["channels"] == [{"role": "amplicon", "round": None, "channel": None,
                                                 "name": None, "wavelength": "unavailable",
                                                 "reference_merged": True, "prepare": None, "prepare_channel": None,
                                                 "registration": None, "sha256": digest(merged(fov))}]
    assert np.array_equal(cell.labels[mask > 0], mask[mask > 0])
    assert set(np.unique(cell.labels)) - {0} == {7, 12}
    for run, result in results.items():
        assert result.label_namespace == json.dumps(["dev", "sample", "FOV_001", None, run], separators=(",", ":"))
        assert result.record["run"] == run and result.record["fov_id"] == "FOV_001"
        assert set(result.record["upstream"]) == {"preprocessing_record_sha256", "registration_record_sha256"}


@pytest.mark.dataset
def test_a_morphology_round_goes_through_its_input_function(fov, tmp_path):
    add_morphology_round(fov)
    path, _ = write_mask(fov, tmp_path)
    channels = fov.dataset.channel_order
    inputs = (InputChannel("composite", "morph", "ch00", prepare=CompositeConfig()),)
    fov.segment(two_run_plan(path, inputs))
    record = fov.segmentation_results["cell"].record
    morph = fov.images["morph"]
    expected, prepared = composite_nuclei_amplicon(morph[..., channels.index("ch00")], merged(fov))
    assert record["input"]["sha256"] == digest(expected[..., None])
    (channel,) = record["input"]["channels"]
    assert channel["prepare"] == json.loads(json.dumps(prepared)) and channel["sha256"] == digest(expected)
    assert channel["registration"] == {"reference": fov.rounds.reference_round, "reference_sha256": None}
    inputs = (InputChannel("cytoplasm", "morph", 1, prepare=FlamingoEnhancementConfig(), prepare_channel="ch03"),)
    fov.segment(two_run_plan(path, inputs))
    expected, _ = enhance_with_flamingo(morph[..., 1], morph[..., channels.index("ch03")])
    assert fov.segmentation_results["cell"].record["input"]["sha256"] == digest(expected[..., None])


@pytest.mark.dataset
def test_a_registered_sequencing_round_is_an_input(fov, tmp_path):
    path, _ = write_mask(fov, tmp_path)
    other = fov.rounds.sequencing_rounds[1]
    fov.segment(two_run_plan(path, (InputChannel("cytoplasm", other, "ch01"),)))
    image = fov.images[other][..., fov.dataset.channel_order.index("ch01")]
    assert fov.segmentation_results["cell"].record["input"]["sha256"] == digest(image[..., None])


@pytest.mark.dataset
def test_coordination_errors(fov, tmp_path):
    path, _ = write_mask(fov, tmp_path)
    morph_input = (InputChannel("cytoplasm", "morph", "ch00"),)
    with pytest.raises(ValueError, match="input round 'morph' is not loaded"):
        fov.segment(two_run_plan(path, morph_input))
    add_morphology_round(fov, register=False)
    with pytest.raises(ValueError, match="morphology round 'morph' has no registration entry"):
        fov.segment(two_run_plan(path, morph_input))
    fov.images.pop("morph")
    add_morphology_round(fov)
    fov.segment(two_run_plan(path, morph_input))   # registered: accepted
    registered = fov.images["morph"]
    fov.images["morph"] = np.ascontiguousarray(registered[:, :16])
    with pytest.raises(IncompatibleGeometryError, match="'morph' has ZYX shape"):
        fov.segment(two_run_plan(path, morph_input))
    fov.images["morph"] = registered
    fov.metadata["morph"] = replace(fov.metadata["morph"], frame_id="elsewhere")
    with pytest.raises(ValueError, match="'morph' has metadata"):
        fov.segment(two_run_plan(path, morph_input))
    # A seed run on another grid: the nuclei imported on the projected grid, the cells on the volume.
    plane_path, _ = write_mask(fov, tmp_path, plane=True)
    plan = SegmentationPlan((
        SegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(plane_path), "nucleus"),
                        projection=ProjectionConfig()),
        SegmentationRun("cell", "cell", (InputChannel("amplicon", reference_merged=True),),
                        SeededWatershedConfig(sigma_um=0.1), seeds="nucleus")))
    before = dict(fov.segmentation_results)
    with pytest.raises(IncompatibleGeometryError, match="seeds are on grid"):
        fov.segment(plan)
    assert fov.segmentation_results == before   # nothing is stored when a run fails


@pytest.mark.dataset
def test_fov_segment_arguments(fov, tmp_path):
    path, _ = write_mask(fov, tmp_path)
    with pytest.raises(TypeError, match="checkpoints must be a CheckpointConfig or None"):
        fov.segment(two_run_plan(path), checkpoints=object())
    with pytest.raises(ValueError, match="device must be"):
        fov.segment(two_run_plan(path), device="gpu")
    with pytest.raises(ValueError, match="runs on cpu, not 'cuda'"):
        fov.segment(two_run_plan(path), device="cuda")
    with pytest.raises(TypeError, match="SegmentationPlan"):
        fov.segment(two_run_plan(path).runs)
    with pytest.raises(ValueError, match="has no channel 'ch09'"):
        fov.segment(two_run_plan(path, (InputChannel("cytoplasm", fov.rounds.reference_round, "ch09"),)))
    assert fov.segmentation_results == {}


@pytest.mark.dataset
def test_a_projected_run_is_extended_through_z(fov, tmp_path, register):
    plane_path, plane_mask = write_mask(fov, tmp_path, plane=True)
    grid = fov.reference_grid()
    plan = SegmentationPlan((
        SegmentationRun("seeds", "nucleus", (), LabelImportConfig(str(plane_path), "nucleus"),
                        projection=ProjectionConfig()),
        SegmentationRun("nucleus", "nucleus", (InputChannel("nuclear", fov.rounds.reference_round, "ch02"),),
                        ProbeConfig(level=-1.0), projection=ProjectionConfig(), operations=(EXTENSION,))))
    fov.segment(plan)
    seeds, nucleus = fov.segmentation_results["seeds"], fov.segmentation_results["nucleus"]
    assert seeds.grid == grid.projected() and seeds.geometry == "plane"
    assert np.array_equal(seeds.labels, plane_mask)
    # Every voxel is above level −1, so the probe labels the whole projected plane 1; the extension
    # keeps that plane where the unprojected channel is foreground.
    stain = fov.images[fov.rounds.reference_round][..., fov.dataset.channel_order.index("ch02")]
    expected, _ = extend_labels_through_z(np.ones((1, *grid.shape_zyx[1:]), np.uint32), stain, grid.metadata,
                                          config=EXTENSION)
    assert nucleus.geometry == "extended" and nucleus.grid == grid
    assert expected.any() and np.array_equal(nucleus.labels, expected)
    record = nucleus.record
    assert record["input"]["projection"] == json.loads(json.dumps(asdict(ProjectionConfig())))
    assert record["input"]["shape_zyxc"] == [1, *grid.shape_zyx[1:], 1]
    assert record["geometry"] == "extended" and record["grid"]["shape_zyx"] == list(grid.shape_zyx)
    (operation,) = record["operations"]
    assert operation["operation"] == "extend_labels_through_z" and operation["record"]["source_run"] == "nucleus"
    assert record["labels"]["sha256"] == digest(nucleus.labels)


def test_plan_validation(register):
    import_run = SegmentationRun("nucleus", "nucleus", (), LabelImportConfig("x.tif", "nucleus"))
    watershed = (InputChannel("amplicon", reference_merged=True),)
    cases = [
        (lambda: InputChannel("tissue", reference_merged=True), ValueError, "unknown channel role"),
        (lambda: InputChannel("amplicon", "round1", 0, reference_merged=True), ValueError, "no round"),
        (lambda: InputChannel("amplicon"), ValueError, "names a round"),
        (lambda: InputChannel("amplicon", "round1"), ValueError, "needs a channel"),
        (lambda: InputChannel("amplicon", "round1", True), TypeError, "channel label"),
        (lambda: InputChannel("amplicon", reference_merged=True, prepare=CompositeConfig()), ValueError,
         "reference merged"),
        (lambda: InputChannel("nuclear", "morph", 0, prepare=FlamingoEnhancementConfig()), ValueError,
         "prepare_channel"),
        (lambda: InputChannel("nuclear", "morph", 0, prepare_channel=1), ValueError, "Flamingo"),
        (lambda: SegmentationRun("Cell", "cell", watershed, SeededWatershedConfig()), ValueError, "snake_case"),
        (lambda: SegmentationRun("cell", "tissue", watershed, SeededWatershedConfig()), ValueError, "target"),
        (lambda: SegmentationRun("cell", "cell", watershed, UnregisteredConfig()), TypeError, "no segmentation"),
        (lambda: SegmentationRun("cell", "cell", (), SeededWatershedConfig()), ValueError, "at least one input"),
        (lambda: SegmentationRun("cell", "cell", watershed * 2, SeededWatershedConfig()), ValueError,
         "repeats a channel role"),
        (lambda: SegmentationRun("nucleus", "nucleus", watershed, LabelImportConfig("x.tif", "nucleus")),
         ValueError, "takes no inputs"),
        (lambda: SegmentationRun("nucleus", "cell", (), LabelImportConfig("x.tif", "nucleus")), ValueError,
         "its import"),
        (lambda: SegmentationRun("nucleus", "nucleus", (), LabelImportConfig("x.tif", "nucleus"), seeds="a"),
         ValueError, "takes no seeds"),
        (lambda: SegmentationRun("cell", "cell", watershed, SeededWatershedConfig(),
                                 projection=ProjectionConfig(axis="channel")), ValueError, "along z"),
        (lambda: SegmentationRun("cell", "cell", watershed, SeededWatershedConfig(), operations=(object(),)),
         TypeError, "operations"),
        (lambda: SegmentationRun("cell", "cell", watershed, SeededWatershedConfig(),
                                 operations=(EXTENSION,)), ValueError, "projected run"),
        (lambda: SegmentationPlan(()), ValueError, "at least one run"),
        (lambda: SegmentationPlan([import_run]), TypeError, "tuple of SegmentationRun"),
        (lambda: SegmentationPlan((import_run, import_run)), ValueError, "more than once"),
        (lambda: SegmentationPlan((SegmentationRun("cell", "cell", watershed, SeededWatershedConfig(),
                                                   seeds="nucleus"), import_run)), ValueError, "not an earlier run"),
    ]
    for build, error, message in cases:
        with pytest.raises(error, match=message):
            build()
