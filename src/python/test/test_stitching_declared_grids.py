"""W-390: data analysed before §2.10 enters the export, and a one-plane FOV is sampled by y and x.

Criteria 1 to 4 of W-390 (the declared grid, units and direction, tiles saved as pages,
molecules on a single-plane grid; the §2.9 side of criterion 4 is
``test_assignment_single_plane.py``, the workflow's criterion 5
``test_workflow_declared_grids.py``).

Fixtures, all written in the test:

* ``scene_folders``: the synthetic sample scene ``sample-tiny-v1`` (2 × 2 FOVs of 5×36×36,
  spacing (1.0, 0.5, 0.5) µm), saved with ``save_sample_scene`` and written twice as a
  dataset that ``Dataset.export`` reads: per FOV ``images/ref_merged/<fov>.tif`` (the
  saved stain), the §2.9 assignment folder (the truth cell and nucleus labels and the
  saved molecule table, assigned in memory) and the truth placement under
  ``stitching/<sample>/``. The first copy stores the scene's metadata in its reference
  images; the second writes them as plain TIFF pages without any metadata, as the files
  of a dataset analysed before §2.10 are, gives the dataset the voxel sizes, and declares
  its assignment grids and placement with the declared metadata the §2.9 workflow writes
  (``declared:<dataset_id>/<output_id>/<fov_id>``, ``micrometer``, no direction),
  rebuilt here from the rule, not from the package.
* ``own_cases`` of ``assembly_fixtures`` (two FOVs of 4×16×16), with grids spelled in each
  of the three spellings of the micrometre, with and without the identity direction.
* One synthetic 3D tile written by tifffile as pages without axis metadata (``IYX``), as
  shaped pages (``QYX``) and with ``ZYX`` axes.
* Two one-plane FOVs (1×16×16) of ``own_cases``' table with plane labels and molecules whose
  z spans several planes, some outside in y or x.

dask runs with the synchronous scheduler. The fusion tests skip, naming the extra, when
multiview-stitcher is not importable; the export tests need anndata.
"""
import importlib.util
import json
from dataclasses import asdict as dataclasses_asdict
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import tifffile
from pandas.testing import assert_frame_equal

from starfinder.assembly import AssemblyInput, assemble, load_assembly
from starfinder.assignment import assign_molecules, molecule_table_from_csv
from starfinder.assignment._persist import assignment_directory, write_assignment
from starfinder.dataset import Dataset
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import load_volume, save_volume
from starfinder.export import ExportConfig, FusionConfig, FusionInput, LabelConfig, check_export, fuse
from starfinder.segmentation import ReferenceGrid, reference_grid_from_file
from starfinder.stitching import approve_placement, placement_from_table, positions_from_table, save_placement

from .assembly_fixtures import (OWN_CELLS, OWN_GENES, OWN_IDS, OWN_MOLECULES, OWN_NUCLEI, OWN_SHAPE,
                                check_accounting, label_run, molecules, own_table, paint, placement,
                                spot_namespace, with_record)

EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
pytestmark = [pytest.mark.stitching, pytest.mark.validation]
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=EXTRA)
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
DATASET_ID, SAMPLE_ID, OUTPUT_ID = "data", "sample", "output"
SAMPLE = "scene"
SPELLINGS = ("µm", "um", "micrometer")
SAMPLE_COLUMNS = {"sample_z", "sample_y", "sample_x", "sample_centroid_z", "sample_centroid_y", "sample_centroid_x",
                  "sample_expanded_centroid_z", "sample_expanded_centroid_y", "sample_expanded_centroid_x",
                  "sample_ownership_centroid_z", "sample_ownership_centroid_y", "sample_ownership_centroid_x"}


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def declared(fov, spacing_zyx):
    """The metadata the §2.9 workflow declares for a file without stored metadata (its _declared_metadata rule)."""
    return ImageMetadata(f"declared:{DATASET_ID}/{OUTPUT_ID}/{fov}", spacing_zyx=spacing_zyx,
                         spatial_unit="micrometer")


def plain_pages(path, array):
    """A TIFF of the array's pages without any metadata: tifffile reports its axes as IYX (I for one plane: YX)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array, metadata=None, photometric="minisblack")
    return path


# --- Criterion 1: the declared grid ------------------------------------------------------------------------------

def scene_dataset(root, scene, book, saved, *, stored, voxel_sizes=None):
    """The scene written as a dataset; returns the Dataset (voxel sizes given only to the plain copy by default)."""
    spacing = tuple(scene.fov_metadata[scene.positions.fov_id.iloc[0]].spacing_zyx)
    if voxel_sizes is None:
        voxel_sizes = (None, None) if stored else (spacing[1], spacing[0])
    dataset = Dataset(root / "input", root / "output", DATASET_ID, SAMPLE_ID, OUTPUT_ID,
                      voxel_size_xy=voxel_sizes[0], voxel_size_z=voxel_sizes[1])
    out = dataset.output_root
    table = positions_from_table(saved / "positions.csv")
    truth = pd.read_csv(saved / "placement_truth.csv", float_precision="round_trip").set_index("fov_id")
    grids = {}
    for fov in table.fov_ids:
        stain = load_volume(saved / "fovs" / fov / "stain.ome.tif").image
        path = out / "images" / "ref_merged" / f"{fov}.tif"
        if stored:
            metadata = scene.fov_metadata[fov]
            save_volume(stain, path, metadata=metadata)
        else:
            metadata = declared(fov, spacing)
            plain_pages(path, stain)
        grid = grids[fov] = ReferenceGrid(stain.shape, metadata, "declared")
        cells = label_run(scene.fov_labels[fov]["cell"], fov, grid)
        nuclei = label_run(scene.fov_labels[fov]["nucleus"], fov, grid, target="nucleus", run="nucleus")
        mols = molecule_table_from_csv(saved / "fovs" / fov / "molecules.csv", spot_namespace=spot_namespace(fov),
                                       genes=book.genes)
        result = assign_molecules(mols, cells, grid=grid, nuclei=nuclei)
        write_assignment(assignment_directory(out / "checkpoints" / fov, "default"),
                         with_record(result, name="default"), metadata=metadata)
    origins = {fov: tuple(truth.loc[fov, ["true_origin_z", "true_origin_y", "true_origin_x"]]) for fov in grids}
    result = approve_placement(placement_from_table(table, grids, sample=SAMPLE, origins=origins), reviewer="test",
                               note="truth placement")
    save_placement(result, out / "stitching" / SAMPLE)
    return dataset


EXPORT = ExportConfig(fusion=FusionConfig(images=("reference_merged",), dimensionality="both"))


def run_export(dataset):
    return dataset.export(SAMPLE, placement=dataset.output_root / "stitching" / SAMPLE, config=EXPORT,
                          output_dir=dataset.output_root / "export" / SAMPLE)


@pytest.fixture(scope="module")
def scene_folders(tmp_path_factory):
    from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene
    book, config = sample_scene_preset("sample-tiny-v1")
    scene = generate_sample_scene(book, config=config)
    root = tmp_path_factory.mktemp("w390_scene")
    saved = save_sample_scene(scene, root / "scene")
    return {"scene": scene, "book": book, "saved": saved,
            "stored": scene_dataset(root / "stored", scene, book, saved, stored=True),
            "plain": scene_dataset(root / "plain", scene, book, saved, stored=False)}


@pytest.fixture(scope="module")
def scene_exports(scene_folders):
    if importlib.util.find_spec("multiview_stitcher") is None or importlib.util.find_spec("anndata") is None:
        pytest.skip(EXTRA + ", or the anndata extra is not")
    import dask
    with dask.config.set(scheduler="synchronous"):
        return {key: run_export(scene_folders[key]).record for key in ("stored", "plain")}


def test_plain_reference_images_have_no_metadata(scene_folders):
    """The plain copy's reference images are pages without metadata; the stored copy's carry the scene's."""
    for fov in scene_folders["scene"].positions.fov_id:
        plain = scene_folders["plain"].output_root / "images" / "ref_merged" / f"{fov}.tif"
        with tifffile.TiffFile(plain) as tif:
            assert tif.series[0].axes == "IYX" and not tif.is_ome and not tif.is_imagej and not tif.is_shaped
        assert load_volume(plain).diagnostics["metadata_source"] == "unknown"
        stored = scene_folders["stored"].output_root / "images" / "ref_merged" / f"{fov}.tif"
        assert load_volume(stored).metadata == scene_folders["scene"].fov_metadata[fov]


def compare_tables(a, b):
    """Same rows, statuses, reasons and flags; sample coordinates within 1e-9 µm; every other column exactly."""
    for name in ("fovs", "cells", "molecules", "counts"):
        x, y = getattr(a, name), getattr(b, name)
        assert list(x.columns) == list(y.columns) and len(x) == len(y) > 0, name
        near = [c for c in x.columns if c in SAMPLE_COLUMNS]
        assert_frame_equal(x.drop(columns=near), y.drop(columns=near), check_exact=True, obj=name)
        for column in near:
            np.testing.assert_allclose(x[column].to_numpy(float), y[column].to_numpy(float), rtol=0, atol=1e-9,
                                       equal_nan=True, err_msg=f"{name}.{column}")


@needs_library
@needs_anndata
def test_declared_grid_export_equals_the_stored_one(scene_folders, scene_exports):
    """Dataset.export, unchanged, on both copies: equal tables and fused images; the plain copy names its grids
    as declared, from the dataset's voxel sizes."""
    stored, plain = scene_folders["stored"], scene_folders["plain"]
    a, b = load_assembly(stored.output_root / "export" / SAMPLE), load_assembly(plain.output_root / "export" / SAMPLE)
    compare_tables(a, b)
    statuses = b.molecules["assembly_reason"].value_counts(dropna=False).to_dict()
    assert b.molecules["assembly_status"].eq("retained").any() and statuses.get("not_owner", 0) > 0
    for record in scene_exports.values():
        assert record["parts"] == {"assemble": "written", "fuse": "written", "segment": "off", "fuse_labels": "off",
                                   "export_spatialdata": "off"}
    outputs = [o["path"] for o in scene_exports["stored"]["fusion"]["outputs"]]
    assert outputs == [o["path"] for o in scene_exports["plain"]["fusion"]["outputs"]] == [
        "reference_merged_projection.ome.tif", "reference_merged_stack.ome.tif"]
    for name in outputs:
        x = tifffile.imread(stored.output_root / "export" / SAMPLE / name)
        y = tifffile.imread(plain.output_root / "export" / SAMPLE / name)
        assert x.dtype == y.dtype == np.float32 and x.shape == y.shape and np.array_equal(x, y)
    spacing = tuple(scene_folders["scene"].fov_metadata["FOV_001"].spacing_zyx)
    for entry in scene_exports["plain"]["assembly"]["fovs"]:
        grid = entry["grid"]
        assert (grid["source"], grid["calibration_source"], grid["sha256"]) == ("declared", "declared", None)
        assert ImageMetadata(**grid["metadata"]) == declared(entry["fov_id"], spacing)
    for entry in scene_exports["plain"]["fusion"]["fovs"]:
        assert entry["grid"]["source"] == "declared"
        assert ImageMetadata(**entry["grid"]["metadata"]) == declared(entry["fov_id"], spacing)
    for entry in scene_exports["stored"]["assembly"]["fovs"]:
        path = stored.output_root / "images" / "ref_merged" / f"{entry['fov_id']}.tif"
        assert entry["grid"]["source"] == f"file:{path.resolve()}" and entry["grid"]["calibration_source"] == "grid"
    check_export(stored.output_root / "export" / SAMPLE)
    check_export(plain.output_root / "export" / SAMPLE)


def test_a_file_with_stored_metadata_is_read_as_before(scene_folders, tmp_path):
    """The file wins: with voxel sizes configured, a file that stores metadata gives the grid
    reference_grid_from_file gives (shape, metadata, source and hash)."""
    from starfinder.dataset._export import _reference_grid
    stored = scene_folders["stored"]
    other = replace(stored, voxel_size_xy=7.0, voxel_size_z=9.0)
    for fov in scene_folders["scene"].positions.fov_id:
        expected = reference_grid_from_file(stored.output_root / "images" / "ref_merged" / f"{fov}.tif")
        for dataset in (stored, other):
            grid = _reference_grid(dataset, fov)
            assert (grid.shape_zyx, grid.metadata, grid.source, grid.sha256) == (
                expected.shape_zyx, expected.metadata, expected.source, expected.sha256)
            assert grid.source.startswith("file:") and grid.sha256 is not None


@needs_anndata
def test_without_metadata_and_voxel_sizes_the_export_raises(scene_folders):
    """A plain reference image and no configured voxel size: ValueError naming the keys; nothing is written."""
    plain = scene_folders["plain"]
    export_dir = plain.output_root / "export" / "unconfigured"
    for sizes, named in (((None, None), "voxel_size_xy and voxel_size_z"), ((0.5, None), "lacks voxel_size_z"),
                         ((None, 1.0), "lacks voxel_size_xy")):
        dataset = replace(plain, voxel_size_xy=sizes[0], voxel_size_z=sizes[1])
        with pytest.raises(ValueError, match=named) as error:
            dataset.export(SAMPLE, placement=plain.output_root / "stitching" / SAMPLE, config=EXPORT,
                           output_dir=export_dir)
        assert "stores no calibration" in str(error.value) and "Dataset fields" in str(error.value)
        with pytest.raises(ValueError, match="stores no calibration"):
            dataset.export(SAMPLE, placement=plain.output_root / "stitching" / SAMPLE,
                           config=ExportConfig(route="per_fov", assemble=False,
                                               fusion=FusionConfig(images=("reference_merged",))),
                           output_dir=export_dir)
    assert not export_dir.exists()


@pytest.mark.parametrize("value", [0, -0.5, float("nan"), float("inf"), True, "0.5"])
def test_dataset_voxel_sizes_are_validated(tmp_path, value):
    for key in ("voxel_size_xy", "voxel_size_z"):
        with pytest.raises(ValueError, match=key):
            Dataset(tmp_path, tmp_path, "d", "s", "o", **{key: value})
    dataset = Dataset(tmp_path, tmp_path, "d", "s", "o", voxel_size_xy=1, voxel_size_z=0.35)
    assert (dataset.voxel_size_xy, dataset.voxel_size_z) == (1.0, 0.35) and isinstance(dataset.voxel_size_xy, float)
    assert (Dataset(tmp_path, tmp_path, "d", "s", "o").voxel_size_xy, Dataset(tmp_path, tmp_path, "d", "s",
                                                                                "o").voxel_size_z) == (None, None)


# --- Criterion 2: units and direction ----------------------------------------------------------------------------

COMBINATIONS = [(unit, direction) for unit in SPELLINGS for direction in ("identity", None)]


def own_metadata(unit, direction):
    return ImageMetadata("fov", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), IDENTITY if direction == "identity" else direction,
                         unit)


def own_grid(metadata, shape=OWN_SHAPE):
    return ReferenceGrid(tuple(shape), metadata, "declared")


def own_assignments(grid):
    """own_cases' results assigned on the given grid (FOV2 expanded 1 µm, FOV1 with nuclei)."""
    from starfinder.assignment import AssignmentConfig
    from starfinder.segmentation import ExpandLabelsConfig
    results = {}
    for fov in OWN_IDS:
        nuclei = OWN_NUCLEI.get(fov)
        config = AssignmentConfig(expansion=ExpandLabelsConfig(1.0, "um", "planar") if fov == "FOV2" else None)
        results[fov] = assign_molecules(
            molecules(fov, OWN_MOLECULES[fov], OWN_GENES), label_run(paint(OWN_SHAPE, OWN_CELLS[fov]), fov, grid),
            grid=grid, config=config,
            nuclei=None if nuclei is None else label_run(paint(OWN_SHAPE, nuclei), fov, grid, target="nucleus",
                                                         run="nucleus"))
    return results


def merged(fov):
    z, y, x = np.meshgrid(*[np.arange(n) for n in OWN_SHAPE], indexing="ij")
    return ((3 * z + y + 2 * x + (7 if fov == "FOV2" else 0)) % 200 + 1).astype(np.uint8)


def run_combination(root, metadata):
    """placement_from_table, assemble, fuse and fuse_labels on own_cases with every grid given this metadata."""
    from starfinder.assembly import save_assembly
    from starfinder.export import fuse_labels
    grid = own_grid(metadata)
    grids = {fov: grid for fov in OWN_IDS}
    placed = approve_placement(placement_from_table(own_table(), grids, sample="s1"), reviewer="test", note="own")
    results = own_assignments(grid)
    folders = {fov: assignment_directory(root / "checkpoints" / fov, "default") for fov in OWN_IDS}
    for fov in OWN_IDS:
        # Saved with the grid as spelled, so that fuse_labels reads each territory from its folder.
        write_assignment(folders[fov], with_record(results[fov], name="default"), metadata=metadata)
    assembly = assemble(placed, [AssemblyInput(fov, grid, assignment=folders[fov]) for fov in OWN_IDS])
    folder = root / "export"
    assembly = save_assembly(assembly, folder)
    inputs = [FusionInput(fov, grid, {"reference_merged": merged(fov)}) for fov in OWN_IDS]
    fuse(placed, inputs, assembly=assembly, config=FusionConfig(images=("reference_merged",), dimensionality="both",
                                                                 scale_yx=1.0), output_dir=folder)
    fuse_labels(placed, assembly, config=LabelConfig(scale_yx=1.0), output_dir=folder)
    images = {name: tifffile.imread(folder / name) for name in ("reference_merged_projection.ome.tif",
                                                                "reference_merged_stack.ome.tif",
                                                                "cell_labels.ome.tif")}
    return placed, assembly, images


@pytest.fixture(scope="module")
def reference_combination(tmp_path_factory):
    if importlib.util.find_spec("multiview_stitcher") is None or importlib.util.find_spec("anndata") is None:
        pytest.skip(EXTRA + ", or the anndata extra is not")
    import dask
    with dask.config.set(scheduler="synchronous"):
        return run_combination(tmp_path_factory.mktemp("w390_reference"), own_metadata("µm", "identity"))


@needs_library
@needs_anndata
@pytest.mark.parametrize("unit, direction", COMBINATIONS)
def test_every_spelling_and_direction_gives_the_same_result(tmp_path, reference_combination, unit, direction):
    placed, assembly, images = run_combination(tmp_path, own_metadata(unit, direction))
    ref_placed, ref_assembly, ref_images = reference_combination
    assert_frame_equal(placed.fovs, ref_placed.fovs, check_exact=True)
    # The record keeps the spelling it was given; nothing is respelled.
    recorded = placed.record["inputs"]["FOV1"]["grid"]["metadata"]
    assert recorded["spatial_unit"] == unit and recorded["direction_zyx"] == (
        [list(r) for r in IDENTITY] if direction else None)
    for name in ("fovs", "cells", "molecules", "counts"):
        assert_frame_equal(getattr(assembly, name), getattr(ref_assembly, name), check_exact=True, obj=name)
    assert assembly.record["counts"] == ref_assembly.record["counts"]
    for name, image in images.items():
        assert image.dtype == ref_images[name].dtype and np.array_equal(image, ref_images[name]), name
    assert ref_images["cell_labels.ome.tif"].any()


def test_mixed_spellings_are_one_unit():
    """FOVs spelled differently in one sample are one unit; each record keeps its own spelling."""
    grids = {"FOV1": own_grid(own_metadata("µm", "identity")), "FOV2": own_grid(own_metadata("micrometer", None))}
    placed = approve_placement(placement_from_table(own_table(), grids, sample="s1"), reviewer="test", note="own")
    assert placed.record["inputs"]["FOV2"]["grid"]["metadata"]["spatial_unit"] == "micrometer"
    results = own_assignments(grids["FOV1"])
    other = own_assignments(grids["FOV2"])
    result = assemble(placed, [AssemblyInput("FOV1", grids["FOV1"], assignment=results["FOV1"]),
                               AssemblyInput("FOV2", grids["FOV2"], assignment=other["FOV2"])])
    assert len(result.molecules) == sum(len(r.molecules) for r in (results["FOV1"], other["FOV2"]))


@pytest.mark.parametrize("unit, direction", COMBINATIONS)
def test_placed_metadata_gives_the_sample_coordinates_of_assemble(tmp_path, unit, direction):
    """placed_metadata(...).index_to_world(p) equals assemble's sample coordinates within 1e-9 µm for every spelling
    and with or without a direction; the placement, its saved file and the assembly record keep the metadata given."""
    from starfinder.stitching import load_placement, placed_metadata
    metadata = own_metadata(unit, direction)
    grid = own_grid(metadata)
    origins = {"FOV1": (0.3, 0.25, 1.5), "FOV2": (-0.7, -0.125, 12.375)}
    placed = approve_placement(placement_from_table(own_table(), {fov: grid for fov in OWN_IDS}, sample="s1",
                                                    origins=origins), reviewer="test", note="own")
    results = own_assignments(grid)
    assembly = assemble(placed, [AssemblyInput(fov, grid, assignment=results[fov]) for fov in OWN_IDS])
    for fov in OWN_IDS:
        world = placed_metadata(placed, fov)
        assert (world.spatial_unit, world.direction_zyx, world.frame_id, world.origin_zyx) == (
            "µm", IDENTITY, "sample:s1", origins[fov])
        mols = assembly.molecules[assembly.molecules["fov_id"].eq(fov).to_numpy(bool)]
        np.testing.assert_allclose(world.index_to_world(mols[["z", "y", "x"]].to_numpy(float)),
                                   mols[["sample_z", "sample_y", "sample_x"]].to_numpy(float), rtol=0, atol=1e-9)
        cells = assembly.cells[assembly.cells["fov_id"].eq(fov).to_numpy(bool)]
        np.testing.assert_allclose(world.index_to_world(cells[["centroid_z", "centroid_y", "centroid_x"]]
                                                        .to_numpy(float)),
                                   cells[[f"sample_centroid_{a}" for a in "zyx"]].to_numpy(float), rtol=0, atol=1e-9)
    given = json.loads(json.dumps(dataclasses_asdict(metadata)))
    save_placement(placed, tmp_path / "placement")
    saved = json.loads((tmp_path / "placement" / "placement.json").read_text())
    for record in (placed.record, load_placement(tmp_path / "placement").record, saved):
        assert all(record["inputs"][fov]["grid"]["metadata"] == given for fov in OWN_IDS)
    assert all(entry["grid"]["metadata"] == given for entry in assembly.record["fovs"] if entry.get("grid"))


REFUSED = {"nm": (own_metadata("nm", "identity"), "a unit other than µm: FOV1 ('nm'), FOV2 ('nm')"),
           "no unit": (own_metadata(None, "identity"), "a unit other than µm: FOV1 (None), FOV2 (None)"),
           "flipped": (own_metadata("µm", ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
                       "a direction other than the identity: FOV1 (((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, "
                       "1.0))), FOV2 (((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))"),
           "um flipped": (own_metadata("um", ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
                          "a direction other than the identity: FOV1 (((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, "
                          "1.0))), FOV2 (((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))")}


@pytest.mark.parametrize("case", list(REFUSED))
def test_another_unit_no_unit_or_another_direction_is_refused_as_before(tmp_path, case):
    """The four entries refuse these grids with the messages of the base revision."""
    from starfinder.export._labels import _grids as label_grids
    metadata, listed = REFUSED[case]
    grid = own_grid(metadata)
    grids = {fov: grid for fov in OWN_IDS}
    with pytest.raises(IncompatibleGeometryError) as error:
        placement_from_table(own_table(), grids, sample="s1")
    assert str(error.value) == ("the sample frame needs known spacing, unit µm and identity direction on every "
                                f"reference grid; FOVs with {listed}")
    # A placement constructed with these grids (as no entry would make it) reaches the other entries' checks.
    nominal = {"FOV1": (0.0, 0.0, 0.0), "FOV2": (0.0, 0.0, 12.0)}
    constructed = placement("s1", OWN_IDS, grids, nominal, nominal)
    what = listed.split(":")[0]
    rows = [((1.0, 2.0, 2.0), "A")]
    plain_grid = own_grid(own_metadata("µm", "identity"))
    inputs = [AssemblyInput(fov, grid, molecules=molecules(fov, rows, OWN_GENES)) for fov in OWN_IDS]
    with pytest.raises(IncompatibleGeometryError) as error:
        assemble(constructed, inputs)
    assert str(error.value) == ("assemble needs known spacing, unit µm and identity direction on every grid; FOVs "
                                f"with {what}: FOV1, FOV2")
    with pytest.raises(IncompatibleGeometryError) as error:
        fuse(constructed, [FusionInput(fov, grid, {"reference_merged": merged(fov)}) for fov in OWN_IDS],
             config=FusionConfig(images=("reference_merged",)), output_dir=tmp_path / "fused")
    assert str(error.value) == ("fuse needs known spacing, unit µm and identity direction on every grid; FOVs "
                                f"with {what}: FOV1, FOV2")
    assert not (tmp_path / "fused").exists()
    # fuse_labels checks the grids the placement recorded when it fuses (_grids, inside the part).
    good = placement("s1", OWN_IDS, {fov: plain_grid for fov in OWN_IDS}, nominal, nominal)
    assembly = assemble(good, [AssemblyInput(fov, plain_grid, molecules=molecules(fov, rows, OWN_GENES))
                               for fov in OWN_IDS])
    with pytest.raises(IncompatibleGeometryError) as error:
        label_grids(constructed, assembly, OWN_IDS)
    assert str(error.value) == ("fuse_labels needs known spacing, unit µm and identity direction on every grid; "
                                "FOVs without: FOV1, FOV2")
    assert set(label_grids(good, assembly, OWN_IDS)) == set(OWN_IDS)


# --- Criterion 3: tiles saved as pages ---------------------------------------------------------------------------

TILE = (5, 36, 36)


def tile(fov):
    z, y, x = np.meshgrid(*[np.arange(n) for n in TILE], indexing="ij")
    return ((5 * z + 3 * y + x + (11 if fov == "FOV2" else 0)) % 4000 + 1).astype(np.uint16)


def tile_files(root, form):
    """FOV → path of its tile written as pages without metadata (IYX), shaped pages (QYX) or with ZYX axes."""
    paths = {}
    for fov in OWN_IDS:
        path = root / form / f"{fov}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        if form == "IYX":
            plain_pages(path, tile(fov))
        elif form == "QYX":
            tifffile.imwrite(path, tile(fov), photometric="minisblack")
        else:
            tifffile.imwrite(path, tile(fov), photometric="minisblack", metadata={"axes": "ZYX"})
        with tifffile.TiffFile(path) as tif:
            assert tif.series[0].axes == form
        paths[fov] = path
    return paths


def tile_placement():
    grid = own_grid(own_metadata("µm", "identity"), TILE)
    table = positions_from_table(pd.DataFrame({"fov_id": OWN_IDS, "nominal_z": [0.0, 0.0], "nominal_y": [0.0, 0.0],
                                               "nominal_x": [0.0, 30.0]}))
    placed = approve_placement(placement_from_table(table, {fov: grid for fov in OWN_IDS}, sample="tiles"),
                               reviewer="test", note="tiles")
    return placed, grid


@needs_library
def test_paged_tiles_fuse_as_their_zyx_twins(tmp_path, monkeypatch):
    """IYX and QYX pages read as ZYX: the header check and the lazy reader agree, every file is read by its pages,
    neither through tifffile's Zarr store (W-432) nor whole, and the fused images equal the ZYX file's pixel for
    pixel."""
    from starfinder.export._fuse import _source_shape
    placed, grid = tile_placement()
    config = FusionConfig(images=("reference_merged",), dimensionality="both", scale_yx=0.5)
    reads = []
    real_imread, real_asarray = tifffile.imread, tifffile.TiffFile.asarray
    monkeypatch.setattr(tifffile, "imread", lambda *a, **k: reads.append(("imread", k.get("aszarr"))) or
                        real_imread(*a, **k))
    monkeypatch.setattr(tifffile.TiffFile, "asarray", lambda self, *a, **k: reads.append(("asarray", None)) or
                        real_asarray(self, *a, **k))
    fused = {}
    for form in ("ZYX", "IYX", "QYX"):
        paths = tile_files(tmp_path, form)
        assert all(_source_shape(p) == (TILE, np.dtype(np.uint16)) for p in paths.values())
        reads.clear()
        record = fuse(placed, [FusionInput(fov, grid, {"reference_merged": paths[fov]}) for fov in OWN_IDS],
                      config=config, output_dir=tmp_path / f"out_{form}")
        assert reads == [], (form, reads)
        assert record.status == "written"
        assert [e["images"]["reference_merged"]["path"] for e in record.section["fovs"]] == [
            str(paths[f].resolve()) for f in OWN_IDS]
        fused[form] = {o["path"]: real_imread(tmp_path / f"out_{form}" / o["path"])
                       for o in record.section["outputs"]}
    assert list(fused["ZYX"]) == ["reference_merged_projection.ome.tif", "reference_merged_stack.ome.tif"]
    for form in ("IYX", "QYX"):
        for name, image in fused["ZYX"].items():
            assert fused[form][name].dtype == image.dtype and np.array_equal(fused[form][name], image), (form, name)
    assert fused["ZYX"]["reference_merged_stack.ome.tif"].shape[0] == TILE[0]


def test_the_lazy_reader_gives_the_header_shape(tmp_path):
    """The lazy reader of a paged file is a dask array of the header's shape and of the file's pixels."""
    import dask.array as da

    from starfinder.export._fuse import _lazy, _Source, _source_shape
    from contextlib import ExitStack
    for form in ("IYX", "QYX"):
        path = tile_files(tmp_path, form)["FOV1"]
        shape, dtype = _source_shape(path)
        with ExitStack() as stack:
            lazy = _lazy(_Source("FOV1", "reference_merged", path, path, shape, dtype), stack)
            assert isinstance(lazy, da.Array) and lazy.shape == shape == TILE
            assert np.array_equal(lazy.compute(), tile("FOV1"))


CHANNELS = {"CZYX": (np.zeros((2, 3, 36, 36), np.uint16), {"metadata": {"axes": "CZYX"}}),
            "CYX": (np.zeros((3, 36, 36), np.uint16), {"metadata": {"axes": "CYX"}}),
            "ImageJ ZCYX": (np.zeros((3, 2, 36, 36), np.uint16), {"imagej": True, "metadata": {"axes": "ZCYX"}}),
            "ImageJ CYX": (np.zeros((3, 36, 36), np.uint16), {"imagej": True}),
            "pages ISYX": (np.zeros((2, 3, 36, 36), np.uint16), {"metadata": None}),
            "shaped QSYX": (np.zeros((2, 3, 36, 36), np.uint16), {}),
            "RGB YXS": (np.zeros((36, 36, 3), np.uint8), {"metadata": None, "photometric": "rgb"})}


@pytest.mark.parametrize("case", list(CHANNELS))
def test_a_channel_axis_or_more_than_three_axes_is_refused_as_before(tmp_path, case):
    from starfinder.export._fuse import _source_shape
    array, kwargs = CHANNELS[case]
    path = tmp_path / "tile.tif"
    tifffile.imwrite(path, array, **kwargs)
    with tifffile.TiffFile(path) as tif:
        axes, shape = tif.series[0].axes, tuple(tif.series[0].shape)
    with pytest.raises(ValueError) as error:
        _source_shape(path)
    assert str(error.value) == (f"source image {path} has axes {axes} {shape}; a source image is one channel, ZYX "
                                "or YX")


# What the base revision 2303ae6's _source_shape gave for each file (measured on that revision with these files):
# a ZYX shape, or None for the refusal "a source image is one channel, ZYX or YX". Only a series with no axis
# metadata whose reported axes are exactly one page axis over Y and X is read differently now (the first two).
AXIS_CASES = {
    "pages, no metadata (IYX)": ((2, 6, 6), {"metadata": None}, None, (2, 6, 6)),
    "shaped pages without axes (QYX)": ((2, 6, 6), {}, None, (2, 6, 6)),
    "uniform pages, no metadata (IYX)": ((8, 48, 64), {"metadata": None}, None, (8, 48, 64)),
    "declared IYX": ((2, 6, 6), {"metadata": {"axes": "IYX"}}, None, None),
    "declared QYX": ((2, 6, 6), {"metadata": {"axes": "QYX"}}, None, None),
    "declared ICYX, singleton channel after the pages": ((2, 1, 6, 6), {"metadata": {"axes": "ICYX"}}, None, None),
    "declared CIYX, singleton channel before the pages": ((1, 2, 6, 6), {"metadata": {"axes": "CIYX"}}, None, None),
    "declared QIYX, singleton extra axis before": ((1, 2, 6, 6), {"metadata": {"axes": "QIYX"}}, None, None),
    "declared IQYX, singleton extra axis after": ((2, 1, 6, 6), {"metadata": {"axes": "IQYX"}}, None, None),
    "declared TIYX, singleton time axis": ((1, 2, 6, 6), {"metadata": {"axes": "TIYX"}}, None, None),
    "shaped QQYX, singleton first": ((1, 2, 6, 6), {}, None, None),
    "shaped QQYX, singleton second": ((2, 1, 6, 6), {}, None, None),
    "declared ZYX": ((2, 6, 6), {"metadata": {"axes": "ZYX"}}, (2, 6, 6), (2, 6, 6)),
    "declared CZYX, singleton channel": ((1, 2, 6, 6), {"metadata": {"axes": "CZYX"}}, (2, 6, 6), (2, 6, 6)),
    "shaped QYX of one page": ((1, 6, 6), {}, (1, 6, 6), (1, 6, 6)),
    "YX, no metadata": ((6, 6), {"metadata": None}, (1, 6, 6), (1, 6, 6)),
    "ImageJ ZYX": ((2, 6, 6), {"imagej": True, "metadata": {"axes": "ZYX"}}, (2, 6, 6), (2, 6, 6)),
    "ImageJ default (CYX)": ((2, 6, 6), {"imagej": True}, None, None),
    "OME CZYX, singleton channel": ((1, 2, 6, 6), {"ome": True, "metadata": {"axes": "CZYX"}}, (2, 6, 6), (2, 6, 6)),
}

# tifffile's series kind for a file no metadata format matched: small pages read as generic, larger as uniform.
FALLBACK_KINDS = {"pages, no metadata (IYX)": "generic", "uniform pages, no metadata (IYX)": "uniform"}


@pytest.mark.parametrize("case", list(AXIS_CASES))
def test_only_metadata_free_pages_are_read_as_zyx(tmp_path, case):
    """Every file the base revision read or refused is read or refused the same, except the two metadata-free page
    series of the decision; a refusal keeps the base message."""
    from starfinder.export._fuse import _source_shape
    shape, kwargs, base, expected = AXIS_CASES[case]
    assert expected == base or case in (*FALLBACK_KINDS, "shaped pages without axes (QYX)")
    path = tmp_path / "tile.tif"
    tifffile.imwrite(path, np.zeros(shape, np.uint16), photometric="minisblack", **kwargs)
    with tifffile.TiffFile(path) as tif:
        axes, reported = tif.series[0].axes, tuple(tif.series[0].shape)
        if case in FALLBACK_KINDS:
            assert tif.series[0].kind == FALLBACK_KINDS[case]
    if expected is None:
        with pytest.raises(ValueError) as error:
            _source_shape(path)
        assert str(error.value) == (f"source image {path} has axes {axes} {reported}; a source image is one "
                                    "channel, ZYX or YX")
    else:
        assert _source_shape(path) == (expected, np.dtype(np.uint16))


# --- Criterion 4: molecules on a single-plane grid (assemble) ----------------------------------------------------

PLANE = (1, 16, 16)
PLANE_CELLS = {"FOV1": {1: ((0, 0), (2, 6), (2, 6)), 2: ((0, 0), (9, 13), (10, 15))},
               "FOV2": {1: ((0, 0), (9, 13), (0, 3)), 2: ((0, 0), (3, 7), (8, 12))}}
PLANE_NUCLEI = {"FOV1": {1: ((0, 0), (3, 5), (3, 5)), 2: ((0, 0), (10, 12), (11, 13))}}
# z spans several planes (one-based z 1 to 30 on the share; here −3.2 to 29); some lie outside in y or x.
Z_VALUES = (-3.2, -0.5, 0.0, 0.4, 0.6, 1.0, 2.7, 7.0, 15.5, 29.0)
YX_POINTS = ((4.0, 4.0), (3.4, 2.6), (11.0, 12.0), (12.2, 14.4), (0.0, 0.0), (8.0, 8.0), (11.0, 1.0), (5.0, 10.0),
             (15.4, 7.0), (-0.6, 3.0), (16.0, 5.0), (7.0, -0.51), (7.0, 15.5), (-3.0, 20.0))


def plane_rows(fov, z_of=None):
    rows = []
    for k, (y, x) in enumerate(YX_POINTS):
        z = Z_VALUES[(k + (3 if fov == "FOV2" else 0)) % len(Z_VALUES)] if z_of is None else z_of
        rows.append(((z, y, x), OWN_GENES[k % len(OWN_GENES)]))
    return rows


def outside_yx(rows, shape=PLANE):
    return np.array([not (0 <= np.floor(y + 0.5) < shape[1] and 0 <= np.floor(x + 0.5) < shape[2])
                     for (_, y, x), _ in rows])


def plane_inputs(*, kind, z_of=None, shape=PLANE):
    grid = own_grid(own_metadata("µm", "identity"), shape)
    inputs, given = [], {}
    for fov in OWN_IDS:
        mols = molecules(fov, plane_rows(fov, z_of), OWN_GENES)
        if kind == "molecules":
            inputs.append(AssemblyInput(fov, grid, molecules=mols))
            given[fov] = mols
            continue
        nuclei = PLANE_NUCLEI.get(fov)
        result = assign_molecules(mols, label_run(paint((1,) + shape[1:], PLANE_CELLS[fov]), fov, grid), grid=grid,
                                  nuclei=None if nuclei is None else label_run(paint((1,) + shape[1:], nuclei), fov,
                                                                               grid, target="nucleus", run="nucleus"))
        inputs.append(AssemblyInput(fov, grid, assignment=result))
        given[fov] = result
    placed = approve_placement(placement_from_table(own_table(), {fov: grid for fov in OWN_IDS}, sample="s1",
                                                    origins={"FOV1": (0.3, 0.0, 0.0), "FOV2": (-0.7, 0.25, 12.0)}),
                               reviewer="test", note="plane")
    return placed, inputs, given


@pytest.mark.parametrize("kind", ["assignment", "molecules"])
def test_one_plane_grid_molecules_are_assembled_by_y_and_x(kind):
    """Each molecule has the outcome it has with its z on the plane; none is outside_grid by its z, every one
    outside in y or x is; the z column keeps the recorded z; the sample z is the plane's; identities 1 to 5."""
    placed, inputs, given = plane_inputs(kind=kind)
    result = assemble(placed, inputs)
    _, flat_inputs, flat_given = plane_inputs(kind=kind, z_of=0.0)
    flat = assemble(placed, flat_inputs)
    check_accounting(result, given, placed)
    check_accounting(flat, flat_given, placed)
    mols, flat_mols = result.molecules, flat.molecules
    assert mols["z"].nunique() > 3
    assert_frame_equal(mols.drop(columns="z"), flat_mols.drop(columns="z"), check_exact=True)
    assert_frame_equal(result.cells, flat.cells, check_exact=True)
    assert_frame_equal(result.counts, flat.counts, check_exact=True)
    rows = {fov: plane_rows(fov) for fov in OWN_IDS}
    expected_z = np.concatenate([[p[0] for p, _ in rows[fov]] for fov in OWN_IDS])
    np.testing.assert_array_equal(mols["z"].to_numpy(float), expected_z)
    outside = np.concatenate([outside_yx(rows[fov]) for fov in OWN_IDS])
    reason = mols["assembly_reason"].fillna("").to_numpy(object)
    assert outside.sum() >= 5 and (~outside).sum() >= 15
    assert (reason[outside] == "outside_grid").all() and not (reason[~outside] == "outside_grid").any()
    origin_z = {fov: float(placed.fovs.set_index("fov_id").loc[fov, "origin_z"]) for fov in OWN_IDS}
    np.testing.assert_array_equal(mols["sample_z"].to_numpy(float), mols["fov_id"].map(origin_z).to_numpy(float))
    if kind == "assignment":
        status = mols["assignment_status"].astype(object).to_numpy()
        assert set(status[~outside]) >= {"assigned", "unassigned"}
        # voxel_z is the plane, 0, for every molecule, also for one outside in y or x.
        assert mols["voxel_z"].notna().all() and (mols["voxel_z"] == 0).all()
    else:
        assert result.record["mode"] == "molecules_only"


def test_volume_grid_molecule_outside_in_z_is_outside_grid_as_before():
    """On a grid of four planes (plane labels on its projection, and molecules only) z outside [−0.5, 3.5) is
    outside_grid and its sample z is its own."""
    shape = (4, 16, 16)
    for kind in ("assignment", "molecules"):
        placed_rows = {fov: plane_rows(fov) for fov in OWN_IDS}
        grid = own_grid(own_metadata("µm", "identity"), shape)
        placed = approve_placement(placement_from_table(own_table(), {fov: grid for fov in OWN_IDS}, sample="s1"),
                                   reviewer="test", note="volume")
        inputs = []
        for fov in OWN_IDS:
            mols = molecules(fov, placed_rows[fov], OWN_GENES)
            if kind == "molecules":
                inputs.append(AssemblyInput(fov, grid, molecules=mols))
            else:
                cells = label_run(paint((1, 16, 16), PLANE_CELLS[fov]), fov, grid)
                assert cells.geometry == "plane" and cells.grid.shape_zyx == (1, 16, 16)
                inputs.append(AssemblyInput(fov, grid, assignment=assign_molecules(mols, cells, grid=grid)))
        result = assemble(placed, inputs)
        mols = result.molecules
        z = mols["z"].to_numpy(float)
        off = ~((np.floor(z + 0.5) >= 0) & (np.floor(z + 0.5) < shape[0]))
        outside = off | np.concatenate([outside_yx(placed_rows[fov], shape) for fov in OWN_IDS])
        reason = mols["assembly_reason"].fillna("").to_numpy(object)
        assert (off & ~np.concatenate([outside_yx(placed_rows[fov], shape) for fov in OWN_IDS])).sum() >= 5
        assert (reason[outside] == "outside_grid").all() and not (reason[~outside] == "outside_grid").any()
        np.testing.assert_array_equal(mols["sample_z"].to_numpy(float), z)
