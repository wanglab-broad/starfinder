"""W-390, criterion 5 (and the saved-folder case of criterion 2): the workflow on files without stored metadata.

The sample of ``test_workflow_assembly_golden.py`` (three FOVs of 8×48×64, ZYX or YX label
images, zero-based molecules.csv tables) with an ``assembly`` block, as
``test_workflow_assembly_figures.py`` builds it, written twice: once with
``fovs/{fovID}/reference.tif`` storing its metadata (unit spacing, µm, identity) and
the placement of ``assembly_golden`` on those grids, and once with the same pixels as
plain TIFF pages without metadata, the workflow configuration's ``voxel_size_xy`` and
``voxel_size_z`` set to the same spacing (1.0 µm), and the placement of
``assembly_golden`` on the grids the §2.9 workflow declares
(``declared:<dataset_id>/<output_id>/<fov_id>``, ``micrometer``, no direction). The fusion
is on, of ``reference_merged``, in both dimensionalities. The legacy ``reads_assignment`` adapter
(``starfinder._legacy_workflow``) runs through a stub ``snakemake`` object, as
``test_workflow_assembly_golden.py`` runs it, and the export is ``Dataset.export`` of the configuration's
translation (``_export_dataset``), the entry the removed ``export_sample`` adapter called; this case stays for the
MATLAB rule set, whose reads_assignment rule the adapter runs. The same plain sample also runs through the Python
workflow's segment, assign and export commands (W-755: ``FOV.load_reference_image``, which those commands call, now
takes the grid the voxel sizes declare for a file without stored metadata, as ``Dataset.export`` does). Each
comparison runs both copies (about 4 to 8 s on one CPU), so it is ``slow``; the saved-folder case runs
``reads_assignment`` on the plain copy only.
"""
import hashlib
import importlib.util
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from starfinder.assembly import assemble, load_assembly
from starfinder.image import ImageMetadata
from starfinder.io import save_volume
from starfinder.export import check_export
from starfinder.dataset._export import assembly_inputs
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import save_placement

from starfinder._legacy_workflow import _run_reads_assignment

from .assembly_fixtures import GOLDEN_IDS, golden_placement, unit_grid
from .fusion_fixtures import golden_codebook, merged_image
from .test_stitching_declared_grids import compare_tables, plain_pages
from .test_workflow_assembly_golden import TILE_ZYX, _config, build_sample

pytestmark = [pytest.mark.workflow, pytest.mark.stitching]
needs_extras = pytest.mark.skipif(
    importlib.util.find_spec("anndata") is None or importlib.util.find_spec("multiview_stitcher") is None,
    reason="the anndata and multiview-stitcher extras (starfinder[anndata,multiview-stitcher]) are not installed")
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
FUSED = ["reference_merged_projection.ome.tif", "reference_merged_stack.ome.tif"]


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


MULTIVIEW = {"method": "multiview_stitcher", "input": "reference_stain",
             "positions": {"source": "regular_grid",
                           "regular_grid": {"columns": 2, "rows": 2, "order": "column_major_down_right",
                                            "overlap": 0.25}}}


def assembly_config(root):
    """The sample's configuration on the route of the saved assignment ``default`` (test_workflow_assembly_figures.py's
    before W-750)."""
    return {**_config(root, legacy=False), "n_rounds": 3, "ref_round": "round1",
            "seq_channel_order": ["ch00", "ch01", "ch02", "ch03"], "stitching": MULTIVIEW,
            "assembly": {"assignment": "default"}, "export": {"route": "per_fov", "fuse": False}}


def reads_assignment(root, out, fov, config):
    """The legacy reads_assignment adapter with the assembly block for one FOV: it saves the FOV's assignment folder."""
    inputs = [root / "input" / "data" / "sample" / "fovs.csv", out / "images" / "DAPI" / f"{fov}.tif",
              out / "images" / "stardist_segmentation" / f"{fov}.tif", out / "fovs" / fov / "molecules.csv",
              root / "input" / "data" / "sample" / "genes.csv", out / "fovs" / fov / "reference.tif"]
    _run_reads_assignment(SimpleNamespace(input=[str(p) for p in inputs],
                                          output=[str(out / "fovs" / fov / "assignment" / "default" / "assignment.json")],
                                          config=config, wildcards=SimpleNamespace(fovID=fov)))


def export_sample(out, config):
    """Dataset.export of the sample with the configuration's export and assembly blocks (the removed export_sample
    adapter's call)."""
    from starfinder.dataset.workflow import _export_config, _export_dataset
    export, block = _export_config(config)
    _export_dataset(config, block, export).export("sample1", placement=out / "stitching" / "sample1",
                                                  assignment=block.assignment, config=export,
                                                  output_dir=out / "export" / "sample1")


def config_of(root):
    """The figures test's configuration with the declared calibration (1.0 µm) and the fusion of reference_merged."""
    config = assembly_config(root)
    return {**config, "voxel_size_xy": 1.0, "voxel_size_z": 1.0,
            "export": {"images": ["reference_merged"], "dimensionality": "both"}}


def declared(config, fov):
    """The §2.9 workflow's declared metadata for a file without stored metadata, rebuilt from its rule."""
    return ImageMetadata(f"declared:{config['dataset_id']}/{config['output_id']}/{fov}",
                         spacing_zyx=(config["voxel_size_z"], config["voxel_size_xy"], config["voxel_size_xy"]),
                         spatial_unit="micrometer")


def build(root, dimensions, *, stored):
    """build_sample with the reference images (stored metadata or plain pages), the codebook and the placement."""
    _, _, out = build_sample(root, dimensions, legacy=False)
    config = config_of(root)
    codebook = root / "input" / "data" / "sample" / "genes.csv"
    golden_codebook().table[["entry_id", "gene_id", "color_sequence"]].to_csv(codebook, index=False)
    grids = {}
    for fov in GOLDEN_IDS:
        path = out / "fovs" / fov / "reference.tif"
        if stored:
            grids[fov] = unit_grid(TILE_ZYX)
            save_volume(merged_image(TILE_ZYX), path, metadata=grids[fov].metadata)
        else:
            grids[fov] = ReferenceGrid(TILE_ZYX, declared(config, fov), "declared")
            plain_pages(path, merged_image(TILE_ZYX))
    save_placement(golden_placement(grids=grids), out / "stitching" / "sample1")
    return out, config


def run(root, dimensions, *, stored):
    out, config = build(root, dimensions, stored=stored)
    for fov in GOLDEN_IDS:
        reads_assignment(root, out, fov, config)
    export_sample(out, config)
    return out, config


def folder_digests(out, fov):
    folder = out / "fovs" / fov / "assignment" / "default"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


@needs_extras
@pytest.mark.slow
@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_reads_assignment_and_export_sample_on_declared_grids(tmp_path, dimensions):
    stored_out, _ = run(tmp_path / "stored", dimensions, stored=True)
    plain_out, config = run(tmp_path / "plain", dimensions, stored=False)
    with tifffile.TiffFile(plain_out / "fovs" / "Position001" / "reference.tif") as tif:
        assert tif.series[0].axes == "IYX" and not tif.is_shaped
    # reads_assignment: the saved folder declares its grid as the §2.9 workflow does; its molecule tables equal
    # (W-568: the saved folder is the route's one per-FOV output; it holds the rows reads_assignment.csv held).
    for fov in GOLDEN_IDS:
        record = json.loads((plain_out / "fovs" / fov / "assignment" / "default" / "assignment.json")
                            .read_text())
        assert record["grid"]["metadata"] == json.loads(json.dumps(asdict(declared(config, fov))))
        assert record["grid"]["source"] == "declared"
        for name in ("molecules.csv",):
            saved = ("fovs", fov, "assignment", "default", name)
            assert plain_out.joinpath(*saved).read_bytes() == stored_out.joinpath(*saved).read_bytes()
    # export_sample: complete, checked, the same tables and fused images; the plain copy's grids declared.
    folders = {key: out / "export" / "sample1" for key, out in (("stored", stored_out), ("plain", plain_out))}
    records = {key: json.loads((folder / "export.json").read_text()) for key, folder in folders.items()}
    for key, folder in folders.items():
        assert records[key]["complete"] is True, key
        assert records[key]["parts"] == {"assemble": "written", "fuse": "written", "segment": "off",
                                         "fuse_labels": "off", "export_spatialdata": "off"}
        assert [o["path"] for o in records[key]["fusion"]["outputs"]] == FUSED
        check_export(folder)
    compare_tables(load_assembly(folders["stored"]), load_assembly(folders["plain"]))
    for name in FUSED:
        a, b = (tifffile.imread(folders[key] / name) for key in ("stored", "plain"))
        assert a.dtype == b.dtype == np.uint8 and np.array_equal(a, b) and a.any()
    for entry in records["plain"]["assembly"]["fovs"]:
        assert (entry["grid"]["source"], entry["grid"]["calibration_source"]) == ("declared", "declared")
    assert all(e["grid"]["source"] == "declared" for e in records["plain"]["fusion"]["fovs"])
    assert all(e["grid"]["source"].startswith("file:") for e in records["stored"]["assembly"]["fovs"])


def command_run(root, dimensions, *, stored):
    """build() of the sample planned in the Python shape to its export (plan_sample of test_workflow_export_tasks.py:
    the processing block, a segment run cell importing images/stardist_segmentation/{fov_id}.tif, a YX image as a
    plane on the grid's Z projection, the assignment default, the stitching block and config_of's assembly and export
    blocks, voxel_size_xy and voxel_size_z 1.0 µm); the fov tasks (whose reference images and molecule tables build
    wrote) and the stitch task (whose placement it saved) succeeded, then the segment, assign and export tasks run by
    their commands. Returns (output root, manifest, configuration)."""
    from . import workflow_tasks as T
    from .test_workflow_export_tasks import plan_sample
    out, config = build(root, dimensions, stored=stored)
    labels = str(out / "images" / "stardist_segmentation" / "{fov_id}.tif")
    manifest, planned = plan_sample(root, "sample1", list(GOLDEN_IDS), export=dict(config["export"]),
                                    assembly=dict(config["assembly"]), labels=labels, output_id="out", stain=False,
                                    codebook_table=golden_codebook().table, voxel_size_xy=1.0, voxel_size_z=1.0)
    assert T.output_root(planned) == out
    if dimensions == "2d":
        planned["segmentation"]["runs"][0]["projection"] = True
        manifest, _ = T.plan(root, "out-plane", planned)
    assert T.task(manifest, "export", sample="sample1")["outputs"] == ["export/sample1/export.json"]
    T.succeed(manifest, "fov", "stitch")
    T.run_kinds(manifest, "segment", "assign", "export")
    return out, manifest, planned


def saved_folder(out, fov):
    """The saved assignment folder of a FOV: its file names, its tables' bytes, and its record without what the grid
    decides or where the copy lies: the grid (stored metadata, or declared), the unit's spelling it gives size_unit
    (µm^3 or micrometer^3), the saved cell run's record and file (its labels.tif stores the grid) and the molecule
    table's path."""
    folder = out / "fovs" / fov / "assignment" / "default"
    names = sorted(p.name for p in folder.iterdir())
    tables = {name: (folder / name).read_bytes() for name in names if name.endswith(".csv")}
    record = json.loads((folder / "assignment.json").read_text())
    record = {k: v for k, v in record.items() if k not in ("grid", "size_unit")}
    record["inputs"] = dict(record["inputs"], cells={k: v for k, v in record["inputs"]["cells"].items()
                                                     if k not in ("record", "record_sha256", "file")},
                            molecules=dict(record["inputs"]["molecules"], source={
                                k: v for k, v in record["inputs"]["molecules"]["source"].items() if k != "path"}))
    return names, tables, record


@needs_extras
@pytest.mark.slow
@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_the_segment_assign_and_export_commands_on_declared_grids(tmp_path, dimensions):
    """W-755: the plain sample through the Python workflow's commands (W-750 kept it on the legacy adapter above,
    because the segment and assign commands read the reference image with FOV.load_reference_image, which needed
    stored metadata). On the plain pages with voxel_size_xy and voxel_size_z the segment task imports the labels on
    the declared grid, the assign task saves the same tables as on the stored copy (its record's grid declared), and
    the export task writes the same tables and fused images, its grids declared; without the voxel sizes the segment,
    assign and export tasks each fail naming voxel_size_xy."""
    from . import workflow_tasks as T
    stored_out, _, _ = command_run(tmp_path / "stored", dimensions, stored=True)
    plain_out, manifest, config = command_run(tmp_path / "plain", dimensions, stored=False)
    with tifffile.TiffFile(plain_out / "fovs" / "Position001" / "reference.tif") as tif:
        assert tif.series[0].axes == "IYX" and not tif.is_shaped
    for fov in GOLDEN_IDS:
        segmentation = json.loads((plain_out / "fovs" / fov / "segmentation" / "cell" / "segmentation.json")
                                  .read_text())
        assert segmentation["grid"]["source"] == "declared"
        record = json.loads((plain_out / "fovs" / fov / "assignment" / "default" / "assignment.json").read_text())
        assert record["grid"]["metadata"] == json.loads(json.dumps(asdict(declared(config, fov))))
        assert record["grid"]["source"] == "declared"
        assert saved_folder(plain_out, fov) == saved_folder(stored_out, fov), fov
        names, tables, _ = saved_folder(plain_out, fov)
        assert names == ["assignment.json", "cells.csv", "counts.csv", "molecules.csv"] and tables["molecules.csv"]
    folders = {key: out / "export" / "sample1" for key, out in (("stored", stored_out), ("plain", plain_out))}
    records = {key: json.loads((folder / "export.json").read_text()) for key, folder in folders.items()}
    for key, folder in folders.items():
        assert records[key]["complete"] is True, key
        assert records[key]["parts"] == {"assemble": "written", "fuse": "written", "segment": "off",
                                         "fuse_labels": "off", "export_spatialdata": "off"}
        assert [o["path"] for o in records[key]["fusion"]["outputs"]] == FUSED
        check_export(folder)
    compare_tables(load_assembly(folders["stored"]), load_assembly(folders["plain"]))
    for name in FUSED:
        a, b = (tifffile.imread(folders[key] / name) for key in ("stored", "plain"))
        assert a.dtype == b.dtype == np.uint8 and np.array_equal(a, b) and a.any()
    for entry in records["plain"]["assembly"]["fovs"]:
        assert (entry["grid"]["source"], entry["grid"]["calibration_source"]) == ("declared", "declared")
    assert all(e["grid"]["source"].startswith("file:") for e in records["stored"]["assembly"]["fovs"])
    # Without the voxel sizes: the same output root planned again, each command fails naming them (downstream first:
    # a failed task removes its own declared outputs, which the tasks after it read).
    unconfigured = {k: v for k, v in config.items() if k not in ("voxel_size_xy", "voxel_size_z")}
    manifest, _ = T.plan(tmp_path / "plain", "unconfigured", unconfigured)
    for task in ("export/sample1", "assign/sample1/Position001/default", "segment/sample1/Position001/cell"):
        assert T.run(manifest, task) == 1, task
        assert "lacks voxel_size_xy and voxel_size_z: set voxel_size_xy and voxel_size_z" in \
            T.failure(manifest, task)["message"], task


@needs_extras
def test_the_saved_folder_enters_assemble_as_saved(tmp_path):
    """The folder reads_assignment saved (micrometer, no direction) is assembled as saved, against a placement
    that recorded the same grid spelled µm with the identity direction too; no file of it is rewritten."""
    from starfinder.dataset.workflow import _assembly_block, _export_dataset
    out, config = build(tmp_path, "2d", stored=False)
    for fov in GOLDEN_IDS:
        reads_assignment(tmp_path, out, fov, config)
    before = {fov: folder_digests(out, fov) for fov in GOLDEN_IDS}
    dataset = _export_dataset(config, _assembly_block(config))
    assert (dataset.voxel_size_xy, dataset.voxel_size_z) == (1.0, 1.0)
    spelled = {fov: ReferenceGrid(TILE_ZYX, replace(declared(config, fov), spatial_unit="µm", direction_zyx=IDENTITY),
                                  "declared") for fov in GOLDEN_IDS}
    results = []
    for placement in (golden_placement(grids={fov: ReferenceGrid(TILE_ZYX, declared(config, fov), "declared")
                                              for fov in GOLDEN_IDS}),
                      golden_placement(grids=spelled)):
        inputs = assembly_inputs(dataset, placement, "default")
        assert [item.grid.metadata for item in inputs] == [declared(config, fov) for fov in GOLDEN_IDS]
        assert all(item.grid.source == "declared" for item in inputs)
        results.append(assemble(placement, inputs))
        assert len(results[-1].molecules) > 0 and results[-1].record["mode"] == "cells"
    compare_tables(*results)
    assert results[0].record["counts"] == results[1].record["counts"]
    assert {fov: folder_digests(out, fov) for fov in GOLDEN_IDS} == before
