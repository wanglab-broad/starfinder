"""W-390, criterion 5 (and the saved-folder case of criterion 2): the workflow on files without stored metadata.

The sample of ``test_workflow_assembly_golden.py`` (three FOVs of 8×48×64, ZYX or YX label
images, one-based goodSpots tables) with an ``assembly`` block, as
``test_workflow_assembly_figures.py`` builds it, written twice: once with
``images/ref_merged/{fovID}.tif`` storing its metadata (unit spacing, µm, identity) and
the placement of ``assembly_golden`` on those grids, and once with the same pixels as
plain TIFF pages without metadata, the workflow configuration's ``voxel_size_xy`` and
``voxel_size_z`` set to the same spacing (1.0 µm), and the placement of
``assembly_golden`` on the grids the §2.9 workflow declares
(``declared:<dataset_id>/<output_id>/<fov_id>``, ``micrometer``, no direction). The fusion
is on, of ``reference_merged``, in both dimensionalities. ``reads_assignment`` and
``export_sample`` run through stub ``snakemake`` objects, as those tests run them. The
comparison runs the workflow on both copies (about 8 s for 2D on one CPU), so it is
``slow``; the saved-folder case runs ``reads_assignment`` on the plain copy only.
"""
import hashlib
import importlib.util
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import tifffile

from starfinder.assembly import assemble, load_assembly
from starfinder.image import ImageMetadata
from starfinder.io import save_volume
from starfinder.sample_output import check_export
from starfinder.sample_output._export import assembly_inputs
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import save_placement

from .assembly_fixtures import GOLDEN_IDS, golden_placement, unit_grid
from .fusion_fixtures import golden_codebook, merged_image
from .test_stitching_declared_grids import compare_tables, plain_pages
from .test_workflow_assembly_figures import assembly_config, export_sample, reads_assignment
from .test_workflow_assembly_golden import TILE_ZYX, build_sample

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
    _, _, out = build_sample(root, dimensions)
    config = config_of(root)
    codebook = root / "input" / "data" / "sample" / "genes.csv"
    golden_codebook().table[["entry_id", "gene_id", "color_sequence"]].to_csv(codebook, index=False)
    grids = {}
    for fov in GOLDEN_IDS:
        path = out / "images" / "ref_merged" / f"{fov}.tif"
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
    folder = out / "checkpoints" / fov / "assignment" / "default"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


@needs_extras
@pytest.mark.slow
@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_reads_assignment_and_export_sample_on_declared_grids(tmp_path, dimensions):
    stored_out, _ = run(tmp_path / "stored", dimensions, stored=True)
    plain_out, config = run(tmp_path / "plain", dimensions, stored=False)
    with tifffile.TiffFile(plain_out / "images" / "ref_merged" / "Position001.tif") as tif:
        assert tif.series[0].axes == "IYX" and not tif.is_shaped
    # reads_assignment: the saved folder declares its grid as the §2.9 workflow does; its per-FOV files equal.
    for fov in GOLDEN_IDS:
        record = json.loads((plain_out / "checkpoints" / fov / "assignment" / "default" / "assignment.json")
                            .read_text())
        assert record["grid"]["metadata"] == json.loads(json.dumps(asdict(declared(config, fov))))
        assert record["grid"]["source"] == "declared"
        for name in ("reads_assignment.csv",):
            assert (plain_out / "expr" / fov / name).read_bytes() == (stored_out / "expr" / fov / name).read_bytes()
    # export_sample: complete, checked, the same tables and fused images; the plain copy's grids declared.
    folders = {key: out / "export" / "sample1" for key, out in (("stored", stored_out), ("plain", plain_out))}
    records = {key: json.loads((folder / "export.json").read_text()) for key, folder in folders.items()}
    for key, folder in folders.items():
        assert records[key]["complete"] is True, key
        assert records[key]["parts"] == {"assemble": "written", "fuse": "written", "fuse_labels": "off",
                                         "export_spatialdata": "off"}
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
