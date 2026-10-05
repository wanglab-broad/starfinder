"""The §2.9 label contract, external-mask import and label functions (W-311).

Rows L1, L2, L3, L5 and L6 of the engineering validation design in
docs/assignment-algorithms.md, and ``FOV.reference_grid`` as docs/segmentation-contract.md
("The reference grid") states it. Every fixture is hand-built in the test (at most
32×64×64 voxels); every expected value comes from the geometry, a formula in the test or
a pinned golden value (``test_segmentation_golden.py``, ``test_assignment_golden.py``).
"""
import hashlib
import math
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest
import tifffile
from skimage.transform import rescale

from starfinder.dataset import (CheckpointConfig, Dataset, ExecutionConfig, PipelineConfig, RegistrationRecipe,
    RegistrationStep, RoundState)
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import ImageLoadConfig, save_volume
from starfinder.registration import TranslationConfig
from starfinder.segmentation import (LabelImportConfig, ReferenceGrid, SegmentationResult, ZExtensionConfig,
    extend_labels_through_z, import_labels, labels_to_grid, reference_grid_from_file, to_label_dtype)
from starfinder.synthetic import development_scene_preset, generate_formed_scene

from .test_assignment_golden import label_fixture as assign_golden_labels
from .test_segmentation_golden import (RESTORED_LABELS, SHAPE_ZYX, SHRUNK_LABELS_DIGEST, digest, fixture,
    stand_in_model)

pytestmark = [pytest.mark.segmentation, pytest.mark.validation]

NAMESPACE = '["data", "sample", "FOV_001", null, "nucleus"]'


def declared(shape, metadata=ImageMetadata("test")):
    return ReferenceGrid(shape, metadata, "declared")


def result(labels, grid=None, *, target="nucleus", geometry="volume"):
    return SegmentationResult(labels, grid or declared(labels.shape), target, geometry, NAMESPACE, {})


# --- L1: label contract ---------------------------------------------------------------------------

def test_l1_a_uint32_zyx_array_on_its_grid_constructs():
    labels = np.zeros((2, 4, 5), np.uint32)
    labels[0, 1, 1], labels[1, 2, 3], labels[1, 3, 4] = 7, 3, 7
    built = result(labels)
    assert (built.n_labels, built.max_label) == (2, 7)
    assert built.grid.shape_zyx == (2, 4, 5) and built.diagnostics == {}
    plane = result(np.zeros((1, 4, 5), np.uint32), geometry="plane", target="cell")
    assert (plane.n_labels, plane.max_label) == (0, 0)
    assert result(np.ones((3, 4, 5), np.uint32), geometry="extended").geometry == "extended"


@pytest.mark.parametrize("case, error", [
    ("int32 with a negative value", TypeError),
    ("uint16", TypeError),
    ("int64", TypeError),
    ("2D array", ValueError),
    ("shape differs from the grid", IncompatibleGeometryError),
    ("target tissue", ValueError),
    ("plane with Z>1", ValueError),
    ("volume with Z=1", ValueError),
    ("extended with Z=1", ValueError),
    ("unknown geometry", ValueError),
    ("not C-contiguous", ValueError),
    ("grid is not a ReferenceGrid", TypeError),
    ("empty namespace", ValueError),
])
def test_l1_each_violation_raises_at_construction(case, error):
    volume = np.zeros((2, 4, 5), np.uint32)
    negative = volume.astype(np.int32)
    negative[0, 0, 0] = -1
    arguments = {
        "int32 with a negative value": dict(labels=negative),
        "uint16": dict(labels=volume.astype(np.uint16)),
        "int64": dict(labels=volume.astype(np.int64)),
        "2D array": dict(labels=volume[0], grid=declared((1, 4, 5))),
        "shape differs from the grid": dict(grid=declared((2, 4, 6))),
        "target tissue": dict(target="tissue"),
        "plane with Z>1": dict(geometry="plane"),
        "volume with Z=1": dict(labels=volume[:1], grid=declared((1, 4, 5))),
        "extended with Z=1": dict(labels=volume[:1], grid=declared((1, 4, 5)), geometry="extended"),
        "unknown geometry": dict(geometry="projected"),
        "not C-contiguous": dict(labels=np.zeros((5, 4, 2), np.uint32).transpose(2, 1, 0)),
        "grid is not a ReferenceGrid": dict(grid=(2, 4, 5)),
        "empty namespace": dict(label_namespace=" "),
    }[case]
    fields = dict(labels=volume, grid=declared((2, 4, 5)), target="nucleus", geometry="volume",
                  label_namespace=NAMESPACE, record={})
    fields.update(arguments)
    with pytest.raises(error):
        SegmentationResult(**fields)


@pytest.mark.parametrize("arguments, error", [
    (dict(shape_zyx=(4, 5)), IncompatibleGeometryError),
    (dict(shape_zyx=(0, 4, 5)), IncompatibleGeometryError),
    (dict(shape_zyx=(1.0, 4, 5)), IncompatibleGeometryError),
    (dict(shape_zyx=(True, 4, 5)), IncompatibleGeometryError),
    (dict(metadata="frame"), TypeError),
    (dict(source="memory"), ValueError),
    (dict(source="fov:"), ValueError),
    (dict(sha256="ABC"), ValueError),
])
def test_l1_reference_grid_validates_its_fields(arguments, error):
    fields = dict(shape_zyx=(2, 4, 5), metadata=ImageMetadata("test"), source="declared", sha256=None)
    fields.update(arguments)
    with pytest.raises(error):
        ReferenceGrid(**fields)


def test_l1_reference_grid_projection():
    metadata = ImageMetadata("frame", spacing_zyx=(0.35, 0.1, 0.1))
    grid = ReferenceGrid([8, np.int64(32), 16], metadata, "fov:round1", "0" * 64)
    assert grid.shape_zyx == (8, 32, 16) and all(type(n) is int for n in grid.shape_zyx)
    projected = grid.projected()
    assert projected.shape_zyx == (1, 32, 16)
    assert projected.metadata == metadata.projected(method="max") == ImageMetadata("frame/projection:max")
    assert (projected.source, projected.sha256) == ("fov:round1", "0" * 64)
    assert grid.projected(method="sum").metadata.frame_id == "frame/projection:sum"


# --- L2: label dtype rule -------------------------------------------------------------------------

def many_labels():
    """32×64×64 int32 labels whose first 70,000 voxels in C order hold 1 to 70,000."""
    labels = np.zeros((32, 64, 64), np.int32)
    labels.reshape(-1)[:70_000] = np.arange(1, 70_001)
    return labels


def test_l2_all_70000_values_survive_the_conversion():
    labels = many_labels()
    converted = to_label_dtype(labels)
    assert converted.dtype == np.uint32 and converted.flags.c_contiguous and converted.shape == labels.shape
    np.testing.assert_array_equal(converted.astype(np.int64), labels.astype(np.int64))
    flat = converted.reshape(-1)
    assert flat[65_534:65_537].tolist() == [65_535, 65_536, 65_537]  # 65,536 does not wrap to 0
    assert np.unique(flat).size - 1 == 70_000 and int(flat.max()) == 70_000
    built = result(converted, target="cell")
    assert (built.n_labels, built.max_label) == (70_000, 70_000)


def test_l2_three_dtypes_give_identical_uint32_arrays():
    golden = assign_golden_labels()
    assert golden.dtype == np.uint16
    converted = [to_label_dtype(golden.astype(dtype)) for dtype in (np.int32, np.uint16, np.uint32)]
    for array in converted:
        assert array.dtype == np.uint32
        np.testing.assert_array_equal(array, converted[0])
    assert len({digest(array) for array in converted}) == 1
    assert sorted(np.unique(converted[0]).tolist()) == [0, 3, 7, 12, 20, 25]


def test_l2_a_negative_value_raises():
    labels = assign_golden_labels().astype(np.int32)
    labels[0, 0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        to_label_dtype(labels)


@pytest.mark.parametrize("labels, error", [
    (np.array([[[2 ** 32]]], np.int64), ValueError),
    (np.array([[[1.0]]], np.float32), TypeError),
    (np.array([[[True]]]), TypeError),
])
def test_l2_values_beyond_uint32_and_non_integer_dtypes_raise(labels, error):
    with pytest.raises(error):
        to_label_dtype(labels)


def test_l2_the_largest_uint32_value_and_big_endian_input_are_kept():
    labels = np.array([[[0, 2 ** 32 - 1]]], np.uint64)
    assert to_label_dtype(labels).tolist() == [[[0, 2 ** 32 - 1]]]
    big = np.array([[[4, 65_536]]], ">u4")
    converted = to_label_dtype(big)
    assert converted.dtype == np.dtype(np.uint32) and converted.tolist() == [[[4, 65_536]]]


# --- L3: import -----------------------------------------------------------------------------------

GRID_SHAPE = (2, 8, 8)
FRAME = ImageMetadata("imports", spacing_zyx=(0.35, 0.1, 0.1), spatial_unit="micrometer")


def values_4_9_30(shape=GRID_SHAPE):
    labels = np.zeros(shape, np.uint16)
    labels[..., 1:3, 1:3], labels[..., 4:6, 1:3], labels[..., 5:7, 5:8] = 4, 9, 30
    return labels


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, array, **options):
    tifffile.imwrite(path, array, **options)
    return path


def imported(path, grid=None, **options):
    return import_labels(path, grid=grid or declared(GRID_SHAPE, FRAME), target=options.pop("target", "nucleus"),
                         label_namespace=NAMESPACE, **options)


def test_l3_big_endian_data_is_read_with_native_values(tmp_path):
    expected = values_4_9_30()
    path = write(tmp_path / "big_endian.tif", expected.astype(">u2"), byteorder=">")
    with tifffile.TiffFile(path) as tif:
        assert tif.byteorder == ">"
    built = imported(path)
    assert built.labels.dtype == np.uint32 and built.labels.dtype.isnative
    np.testing.assert_array_equal(built.labels, expected)
    entry = built.record["import"]
    assert (entry["source_dtype"], entry["source_shape"]) == ("uint16", [2, 8, 8])
    # The file's SHA-256 and the source array's SHA-256 (dtype, shape and native C-order bytes).
    assert entry["file_sha256"] == file_sha256(path)
    assert entry["sha256"] == digest(expected)
    assert built.record["labels"] == {"dtype": "uint32", "sha256": digest(expected.astype(np.uint32)),
                                      "n_labels": 3, "max_label": 30}


def test_l3_the_record(tmp_path):
    path = write(tmp_path / "labels.tif", values_4_9_30().astype(np.int32))
    built = imported(path, target="cell")
    record = built.record
    assert (built.target, built.geometry, built.label_namespace) == ("cell", "volume", NAMESPACE)
    assert {key: record[key] for key in ("format_version", "stage", "dataset_id", "sample_id", "fov_id",
                                         "subtile_id", "run", "target", "geometry", "input", "seeds",
                                         "methods", "operations", "outcome")} == {
        "format_version": 1, "stage": "segmentation", "dataset_id": "data", "sample_id": "sample",
        "fov_id": "FOV_001", "subtile_id": None, "run": "nucleus", "target": "cell", "geometry": "volume",
        "input": None, "seeds": None, "methods": [], "operations": [], "outcome": "ok"}
    assert record["grid"] == {"shape_zyx": [2, 8, 8],
                              "metadata": {"frame_id": "imports", "spacing_zyx": [0.35, 0.1, 0.1],
                                           "origin_zyx": None, "direction_zyx": None,
                                           "spatial_unit": "micrometer"},
                              "source": "declared", "sha256": None}
    entry = record["import"]
    assert {key: entry[key] for key in ("path", "source_dtype", "metadata_source", "relabel", "relabel_map",
                                        "target", "geometry")} == {
        "path": str(path), "source_dtype": "int32", "metadata_source": "declared", "relabel": False,
        "relabel_map": None, "target": "cell", "geometry": "volume"}
    assert set(record["software"]) == {"code", "environment"}


def test_l3_a_yx_file_becomes_1_y_x_and_needs_a_z1_grid(tmp_path):
    plane = values_4_9_30((8, 8))
    path = write(tmp_path / "plane.tif", plane)
    built = imported(path, declared((1, 8, 8), FRAME))
    assert built.labels.shape == (1, 8, 8) and built.geometry == "plane"
    np.testing.assert_array_equal(built.labels[0], plane)
    assert built.record["import"]["source_shape"] == [8, 8]
    volume = declared((4, 8, 8), FRAME)
    with pytest.raises(IncompatibleGeometryError, match="Z=1 grid"):
        imported(path, volume)
    projected = imported(path, volume.projected())
    assert projected.grid.metadata == FRAME.projected(method="max")


@pytest.mark.parametrize("dtype", [np.float32, bool])
def test_l3_float_and_boolean_masks_raise_type_error(tmp_path, dtype):
    path = write(tmp_path / "mask.tif", values_4_9_30().astype(dtype))
    with pytest.raises(TypeError, match="integer dtype"):
        imported(path)


def test_l3_a_file_of_another_shape_raises(tmp_path):
    path = write(tmp_path / "other.tif", values_4_9_30((2, 8, 9)))
    with pytest.raises(IncompatibleGeometryError):
        imported(path)


def test_l3_stored_metadata_must_equal_the_grid(tmp_path):
    differing = tmp_path / "differing.tif"
    save_volume(values_4_9_30(), differing, metadata=ImageMetadata("another frame"))
    with pytest.raises(ValueError, match="differs from the grid"):
        imported(differing)
    equal = tmp_path / "equal.tif"
    save_volume(values_4_9_30(), equal, metadata=FRAME)
    assert imported(equal).record["import"]["metadata_source"] == "stored"


def test_l3_missing_metadata_is_recorded_as_declared(tmp_path):
    path = write(tmp_path / "plain.tif", values_4_9_30())
    built = imported(path)
    assert built.record["import"]["metadata_source"] == "declared"
    assert built.grid.metadata == FRAME


def test_l3_relabel_maps_4_9_30_to_1_2_3(tmp_path):
    source = values_4_9_30()
    path = write(tmp_path / "labels.tif", source)
    built = imported(path, relabel=True)
    expected = np.select([source == 4, source == 9, source == 30], [1, 2, 3], 0)
    np.testing.assert_array_equal(built.labels, expected)
    assert built.record["import"]["relabel"] is True
    assert built.record["import"]["relabel_map"] == [[4, 1], [9, 2], [30, 3]]
    assert built.record["import"]["sha256"] == digest(source)  # the file's values, before relabelling
    kept = imported(path)
    np.testing.assert_array_equal(kept.labels, source)


@pytest.mark.parametrize("case", ["negative", "missing", "plane geometry on Z>1", "unknown target"])
def test_l3_other_rejections(tmp_path, case):
    if case == "negative":
        labels = values_4_9_30().astype(np.int32)
        labels[0, 0, 0] = -2
        path = write(tmp_path / "negative.tif", labels)
        with pytest.raises(ValueError, match="negative"):
            imported(path)
    elif case == "missing":
        with pytest.raises(FileNotFoundError):
            imported(tmp_path / "absent.tif")
    elif case == "plane geometry on Z>1":
        path = write(tmp_path / "labels.tif", values_4_9_30())
        with pytest.raises(ValueError, match="plane needs Z=1"):
            imported(path, geometry="plane")
    else:
        path = write(tmp_path / "labels.tif", values_4_9_30())
        with pytest.raises(ValueError, match="target"):
            imported(path, target="tissue")


def test_l3_an_all_zero_mask_is_empty_and_extended_geometry_is_kept(tmp_path):
    path = write(tmp_path / "zero.tif", np.zeros(GRID_SHAPE, np.uint8))
    built = imported(path)
    assert built.record["outcome"] == "empty" and built.n_labels == 0
    extended = imported(write(tmp_path / "labels.tif", values_4_9_30()), geometry="extended")
    assert extended.geometry == "extended" and extended.record["import"]["geometry"] == "extended"


def test_l3_label_import_config_validates():
    config = LabelImportConfig("masks/cells.tif", "cell")
    assert (config.geometry, config.relabel) == (None, False)
    with pytest.raises(ValueError):
        LabelImportConfig("masks/cells.tif", "tissue")
    with pytest.raises(ValueError):
        LabelImportConfig("masks/cells.tif", "cell", geometry="projected")
    with pytest.raises(TypeError):
        LabelImportConfig("masks/cells.tif", "cell", relabel=1)


def test_reference_grid_from_file(tmp_path):
    image = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(GRID_SHAPE)
    stored = tmp_path / "ref_merged.tif"
    save_volume(image, stored, metadata=FRAME)
    grid = reference_grid_from_file(stored)
    assert (grid.shape_zyx, grid.metadata, grid.source, grid.sha256) == (
        GRID_SHAPE, FRAME, f"file:{stored.resolve()}", digest(image))
    with pytest.raises(ValueError, match="differs"):
        reference_grid_from_file(stored, metadata=ImageMetadata("other"))
    plain = write(tmp_path / "plain.tif", image[0])
    with pytest.raises(ValueError, match="no starfinder_metadata"):
        reference_grid_from_file(plain)
    plane = reference_grid_from_file(plain, metadata=FRAME)
    assert plane.shape_zyx == (1, 8, 8) and plane.metadata == FRAME


# --- L5: extend_labels_through_z ------------------------------------------------------------------

CULTURE_SPACING = (0.35, 0.1, 0.1)
CULTURE = ImageMetadata("culture", spacing_zyx=CULTURE_SPACING, spatial_unit="micrometer")
# Between the two stain levels on the [0, 1] scale of the uint8 range: 10/255 < t < 200/255.
CULTURE_THRESHOLD = 100 / 255


def culture():
    """8×32×32 culture layer: 2D cell and nucleus boxes and a stain of 200 in z 2–4, 10 elsewhere."""
    cells = np.zeros((1, 32, 32), np.uint16)
    cells[0, 2:14, 2:14], cells[0, 16:30, 4:28], cells[0, 3:12, 18:30] = 1, 2, 3
    nuclei = np.zeros_like(cells)
    nuclei[0, 5:10, 5:10], nuclei[0, 20:26, 10:16], nuclei[0, 5:9, 22:26] = 1, 2, 3
    stain = np.full((8, 32, 32), 10, np.uint8)
    stain[2:5] = 200
    return cells, nuclei, stain


def culture_config(threshold=CULTURE_THRESHOLD):
    return ZExtensionConfig(median_um=0.1, threshold=threshold, min_area_um2=0.01, dilation_um=0,
                            fill_holes="once")


@pytest.mark.parametrize("which", ["cells", "nuclei"])
def test_l5_the_labels_extend_through_the_stained_planes(which):
    cells, nuclei, stain = culture()
    plane = cells if which == "cells" else nuclei
    extended, record = extend_labels_through_z(plane, stain, CULTURE, config=culture_config())
    expected = np.zeros((8, 32, 32), np.uint32)
    expected[2:5] = plane[0]
    assert extended.dtype == np.uint32 and extended.shape == (8, 32, 32)
    np.testing.assert_array_equal(extended, expected)
    assert record["geometry"] == "extended" and record["outcome"] == "ok"
    assert (record["threshold"], record["threshold_source"]) == (CULTURE_THRESHOLD, "config")
    assert record["inputs"] == [digest(plane), digest(stain)] and record["output"] == digest(extended)
    assert record["pixels"] == {"median_window_yx": [1, 1], "min_area": 1, "dilation_radius_yx": [0.0, 0.0]}
    assert record["config"] == {"median_um": 0.1, "threshold": CULTURE_THRESHOLD, "min_area_um2": 0.01,
                                "dilation_um": 0.0, "fill_holes": "once"}
    built = SegmentationResult(extended, ReferenceGrid((8, 32, 32), CULTURE, "declared"),
                               "cell" if which == "cells" else "nucleus", "extended", NAMESPACE, {})
    assert built.n_labels == 3


def test_l5_otsu_separates_the_two_levels_too():
    cells, _, stain = culture()
    extended, record = extend_labels_through_z(cells, stain, CULTURE, config=culture_config("otsu"))
    assert record["threshold_source"] == "otsu" and 10 / 255 <= record["threshold"] < 200 / 255
    assert extended[2:5].tolist() == [cells[0].tolist()] * 3 and not extended[[0, 1, 5, 6, 7]].any()


def test_l5_labels_with_z_above_1_raise():
    cells, _, stain = culture()
    with pytest.raises(IncompatibleGeometryError):
        extend_labels_through_z(np.repeat(cells, 2, axis=0), stain, CULTURE, config=culture_config())


def test_l5_a_stain_of_10_everywhere_gives_an_empty_image():
    cells, _, _ = culture()
    stain = np.full((8, 32, 32), 10, np.uint8)
    extended, record = extend_labels_through_z(cells, stain, CULTURE, config=culture_config())
    assert extended.shape == (8, 32, 32) and not extended.any()
    assert record["outcome"] == "empty"
    _, otsu = extend_labels_through_z(cells, stain, CULTURE, config=culture_config("otsu"))
    assert otsu["outcome"] == "empty"  # a constant stack is not strictly above its own threshold


@pytest.mark.parametrize("case, error", [
    ("stain of another Y, X", IncompatibleGeometryError),
    ("YX labels", IncompatibleGeometryError),
    ("no spacing", ValueError),
    ("float labels", TypeError),
    ("config type", TypeError),
])
def test_l5_other_rejections(case, error):
    cells, _, stain = culture()
    arguments = dict(labels_2d=cells, stain=stain, metadata=CULTURE, config=culture_config())
    arguments.update({"stain of another Y, X": dict(stain=stain[:, :, :31]),
                      "YX labels": dict(labels_2d=cells[0]),
                      "no spacing": dict(metadata=ImageMetadata("culture")),
                      "float labels": dict(labels_2d=cells.astype(np.float32)),
                      "config type": dict(config=asdict(culture_config()))}[case])
    with pytest.raises(error):
        extend_labels_through_z(**arguments)


@pytest.mark.parametrize("fields", [
    dict(threshold=100), dict(threshold=-0.1), dict(threshold=True), dict(threshold="li"),
    dict(median_um=0), dict(min_area_um2=-1), dict(dilation_um=float("nan")), dict(fill_holes="never"),
])
def test_l5_z_extension_config_validates(fields):
    """A grey level such as 100 is not on the [0, 1] scale and raises."""
    base = dict(median_um=0.1, threshold=0.5, min_area_um2=0.01, dilation_um=0, fill_holes="once")
    with pytest.raises(ValueError):
        ZExtensionConfig(**{**base, **fields})


def test_l5_hole_filling_area_filter_and_dilation_follow_the_physical_sizes():
    """One plane: a ring with a hole, a 2-pixel speck and a dilation of 0.2 µm (2 pixels)."""
    labels = np.full((1, 32, 32), 5, np.uint16)
    stain = np.zeros((1, 32, 32), np.uint8)
    stain[0, 8:16, 8:16] = 200
    stain[0, 11:13, 11:13] = 0      # a hole, filled before the area filter
    stain[0, 25, 25:27] = 200       # 2 pixels = 0.02 µm², below min_area_um2 0.03
    config = ZExtensionConfig(median_um=0.1, threshold=0.5, min_area_um2=0.03, dilation_um=0.2, fill_holes="once")
    extended, record = extend_labels_through_z(labels, stain, ImageMetadata("plane", spacing_zyx=CULTURE_SPACING),
                                               config=config)
    expected = np.zeros((32, 32), bool)
    expected[8:16, 8:16] = True
    y, x = np.nonzero(expected)
    grown = np.zeros_like(expected)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy * dy + dx * dx <= 4:
                grown[y + dy, x + dx] = True
    np.testing.assert_array_equal(extended[0], np.where(grown, 5, 0))
    assert record["pixels"] == {"median_window_yx": [1, 1], "min_area": 3, "dilation_radius_yx": [2.0, 2.0]}


# --- L6: labels_to_grid ---------------------------------------------------------------------------

def test_l6_the_shrunk_stand_in_labels_return_to_the_restored_values():
    dapi = fixture()["dapi"]
    shrunk = stand_in_model(rescale(dapi, [1, .5, .5]))
    assert shrunk.shape == (16, 32, 32) and digest(shrunk) == SHRUNK_LABELS_DIGEST
    restored, record = labels_to_grid(shrunk, target_shape=SHAPE_ZYX)
    assert restored.dtype == np.uint32 and restored.shape == SHAPE_ZYX and restored.flags.c_contiguous
    assert (restored.astype(np.int32).dtype.str, digest(restored.astype(np.int32))) == RESTORED_LABELS
    assert record == {"function": "labels_to_grid", "config": {"target_shape": [16, 64, 64]},
                      "inputs": [digest(shrunk)], "output": digest(restored),
                      "source_shape": [16, 32, 32], "target_shape": [16, 64, 64]}


def test_l6_an_odd_grid_is_reached_exactly():
    source = np.arange(1, 30 * 32 + 1, dtype=np.int32).reshape(1, 30, 32)
    output, record = labels_to_grid(source, target_shape=(1, 61, 63))
    assert output.shape == (1, 61, 63)

    def index(i, n_source, n_target):
        return math.floor(Fraction(2 * i + 1, 2) * n_source / n_target)

    expected = np.array([[[source[0, index(y, 30, 61), index(x, 32, 63)] for x in range(63)] for y in range(61)]])
    np.testing.assert_array_equal(output, expected)
    assert record["source_shape"] == [1, 30, 32] and record["target_shape"] == [1, 61, 63]


@pytest.mark.parametrize("labels, target_shape, error", [
    (np.zeros((30, 32), np.int32), (1, 61, 63), IncompatibleGeometryError),
    (np.zeros((1, 30, 32), np.int32), (61, 63), ValueError),
    (np.zeros((1, 30, 32), np.int32), (1, 0, 63), ValueError),
    (np.full((1, 30, 32), -1, np.int32), (1, 61, 63), ValueError),
    (np.zeros((1, 30, 32), np.float64), (1, 61, 63), TypeError),
])
def test_l6_rejections(labels, target_shape, error):
    with pytest.raises(error):
        labels_to_grid(labels, target_shape=target_shape)


# --- FOV.reference_grid ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def development_fov_root(tmp_path_factory):
    """The development preset 'clean', size 'small' (9×32×32, four channels, three rounds), as TIFFs."""
    book, config = development_scene_preset("clean", size="small")
    scene = generate_formed_scene(book, config=config)
    root = tmp_path_factory.mktemp("development")
    for name, image in scene.rounds.items():
        for c, label in enumerate(scene.channel_labels):
            save_volume(image[..., c], root / name / "FOV_001" / f"{label}.tif", metadata=scene.round_metadata[name])
    return root, scene


def development_fov(root, scene):
    rounds = RoundState(list(scene.round_labels), reference_round=scene.round_labels[0])
    return Dataset(root, root / "out", "dev", "sample", "out", rounds, list(scene.channel_labels)).fov("FOV_001")


def development_pipeline(scene):
    return PipelineConfig(load=ImageLoadConfig(channel_labels=tuple(scene.channel_labels)),
                          registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)))


@pytest.mark.dataset
@pytest.mark.contract
@pytest.mark.parametrize("mode", ["batch", "streaming"])
def test_fov_reference_grid_after_run(development_fov_root, mode):
    root, scene = development_fov_root
    fov = development_fov(root, scene)
    ref = fov.rounds.reference_round
    with pytest.raises(ValueError, match="not resident"):
        fov.reference_grid()
    fov.run(development_pipeline(scene), execution=ExecutionConfig(mode))
    grid = fov.reference_grid()
    assert isinstance(grid, ReferenceGrid)
    assert grid.shape_zyx == fov.images[ref].shape[:3] == (9, 32, 32)
    assert grid.metadata == fov.metadata[ref]
    assert grid.source == f"fov:{ref}"
    assert grid.sha256 == digest(fov.images[ref])


@pytest.mark.dataset
@pytest.mark.contract
def test_fov_reference_grid_after_loading_the_registered_checkpoint(development_fov_root, tmp_path):
    root, scene = development_fov_root
    fov = development_fov(root, scene).run(development_pipeline(scene),
                                           checkpoints=CheckpointConfig(directory=tmp_path, stages=("registered",)))
    loaded = development_fov(root, scene)
    with pytest.raises(ValueError, match="not resident"):
        loaded.reference_grid()
    loaded.load_checkpoint("registered", checkpoints=CheckpointConfig(directory=tmp_path))
    assert loaded.reference_grid() == fov.reference_grid()
