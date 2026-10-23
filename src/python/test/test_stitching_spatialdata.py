"""SD1 to SD8, the SpatialData parts of EX1 and EX3 and the store clause of FI11 of
docs/stitching-assembly-validation.md: ``export_spatialdata``, its store ``sample.zarr``,
its ``spatialdata`` section and ``lookup_source``.

Fixtures: ``export_cases``, one sample per row of the table "Empty populations" of
docs/sample-output-contract.md, built from ``export_folder`` (``assembly_golden`` written as a
dataset) with the §2.9 assignment folders replaced by the ``empty_cases`` and ``status_cases``
variants; the row "No included FOV" through ``no_fov_assembly``, ``fuse_images`` called alone on its
all-failed placement and ``export_spatialdata`` called alone. ``label_quad`` (label export),
``position_grid`` (seed 354) and ``fuse_band`` (``slow``) with their store placements and
``molecules_only`` assemblies. Expected values are computed here from the sample tables, the
``sample.h5ad`` file and the OME-TIFF files, not from the export's code. The missing extras
are made by making the import fail inside the test. dask runs with the synchronous scheduler.
"""
import importlib.metadata
import importlib.util
import json
import math
import shutil
import sys
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import tifffile

import starfinder.dataset._export as export_module
from starfinder.assembly import (
    AssemblyConfig,
    AssemblyInput,
    assemble,
    load_assembly,
    save_assembly,
)
from starfinder.assignment._persist import read_assignment, write_assignment
from starfinder.export import (
    ExportConfig,
    ExportIncompleteWarning,
    FusionConfig,
    FusionRecord,
    LabelConfig,
    SpatialDataExportConfig,
    check_export,
    export_spatialdata,
    fuse_images,
    fuse_labels,
    lookup_source,
)
from starfinder.export._record import section_sha256
from starfinder._records import file_sha256
from starfinder.stitching import (
    approve_placement,
    placement_from_table,
    positions_from_table,
)

from .assembly_fixtures import (
    GOLDEN_GENES,
    GOLDEN_IDS,
    SAMPLE,
    TILE_ZYX,
    golden_placement,
    golden_results,
    molecules,
    unit_grid,
    with_record,
)
from .fusion_fixtures import (
    LN_XY,
    canonical,
    export_folder,
    fuse_band,
    listing,
    position_grid,
)

MV = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
SD = "the spatialdata extra (starfinder[spatialdata]) is not installed"
pytestmark = [pytest.mark.stitching, pytest.mark.validation,
              pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                 reason="the anndata extra (starfinder[anndata]) is not installed"),
              pytest.mark.skipif(importlib.util.find_spec("spatialdata") is None, reason=SD)]
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=MV)
FIXED = ["molecules", "cell_centroids", "cells", "expression", "cells_unplaced", "molecules_unplaced", "cell_labels"]
TABLE_FILES = ["export.json", "sample.h5ad", "tables/cells.csv", "tables/counts.csv", "tables/fovs.csv",
               "tables/molecules.csv"]
RASTER_VERSION = {3: "0.5-dev-spatialdata", 2: "0.4-dev-spatialdata"}
PLURAL = {"image": "images", "labels": "labels", "points": "points", "shapes": "shapes", "table": "tables"}
SECTION_ITEMS = ("status", "coordinate_system", "frame_id", "assembly_sha256", "fusion_sha256", "labels_sha256",
                 "library", "format", "config", "store", "elements", "counts", "renames", "outputs")


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


# --- Helpers -----------------------------------------------------------------------------------------------------

def export(f, config=None, **kwargs):
    return f.dataset.export(SAMPLE, placement=kwargs.pop("placement", f.placement_dir),
                            config=ExportConfig(route="per_fov") if config is None else config,
                            output_dir=kwargs.pop("output_dir", f.export_dir), **kwargs)


def record_of(folder):
    return json.loads((folder / "export.json").read_text())


def sd_config(zarr_format=3, **fields):
    return ExportConfig(route="per_fov", export_spatialdata=True,
                        spatialdata=SpatialDataExportConfig(zarr_format=zarr_format),
                        **fields)


def read_store(folder):
    import spatialdata
    with pd.option_context("future.infer_string", False):
        return spatialdata.read_zarr(folder / "sample.zarr")


def statuses(record):
    """Element → (status, reason) from the spatialdata section."""
    return {e["name"]: (e["status"], e.get("reason")) for e in record["spatialdata"]["elements"]}


def staging(folder):
    return [p.name for p in folder.iterdir() if p.name.startswith(".sample.zarr")]


def rewrite(f, results):
    """Replace each FOV's saved §2.9 assignment folder of an export_folder by another assignment."""
    for fov in GOLDEN_IDS:
        folder = f.assignment_dir(fov)
        shutil.rmtree(folder)
        write_assignment(folder, with_record(results[fov], name="default"), metadata=f.grids[fov].metadata)
    return f


def values(series):
    """A column as Python values with NaN and every null as None (NaN equals NaN, null equals null)."""
    out = []
    for value in series.astype(object).tolist():
        if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
            out.append(None)
        else:
            out.append(value.item() if isinstance(value, np.generic) else value)
    return out


def join(frame, keys):
    return {tuple(k): i for i, k in enumerate(zip(*(values(frame[c]) for c in keys)))}


def intrinsic(transform, axes, q):
    """The intrinsic coordinate of sample position ``q`` (on ``axes``) under an element's transform."""
    matrix = transform.to_affine_matrix(input_axes=tuple(axes), output_axes=tuple(axes))
    return np.linalg.solve(matrix[:-1, :-1], np.asarray(q, float) - matrix[:-1, -1])


# --- export_cases: one sample per row of "Empty populations" -------------------------------------------------------

def _empty_variants():
    from .test_stitching_assembly_retention import empty_variants
    return empty_variants()


def _no_fov(tmp_path, zarr_format):
    """No included FOV: no_fov_assembly saved, fuse_images alone on the all-failed placement, export_spatialdata alone."""
    from starfinder.assembly import AssemblyResult
    from starfinder.stitching._placement import table_sha256

    from .test_stitching_assembly_retention import no_fov_assembly
    failed = golden_placement(statuses={fov: "failed" for fov in GOLDEN_IDS})
    hand = no_fov_assembly()
    hashes = {"fovs_sha256": table_sha256(failed.fovs), "links_sha256": table_sha256(failed.links)}
    hand = AssemblyResult(hand.sample, hand.fovs, hand.cells, hand.molecules, hand.counts, hand.genes,
                          dict(hand.record, placement=dict(hand.record["placement"], **hashes)))
    folder = tmp_path / "no_included_fov"
    saved = save_assembly(hand, folder)
    record = fuse_images(failed, [], assembly=saved, output_dir=folder)
    assert record.status == "empty" and record.section["reason"] == "no_included_fov"
    export_spatialdata(load_assembly(folder), config=SpatialDataExportConfig(zarr_format=zarr_format),
                       output_dir=folder)
    return folder


def build_case(name, tmp_path, zarr_format, monkeypatch):
    """The export folder of one export case; returns the folder."""
    labels = dict(fuse_labels=True)
    if name == "no_included_fov":
        return _no_fov(tmp_path, zarr_format)
    if name == "fov_not_placed":
        f = export_folder(tmp_path, placement_result=approve_placement(
            golden_placement(statuses={"Position003": "failed"}, review="required"), reviewer="test",
            note="status_cases"))
    else:
        f = export_folder(tmp_path)
    variants = _empty_variants()
    if name in ("no_placed_cell", "no_counted_cell", "no_molecule"):
        rewrite(f, golden_results(**variants[{"no_placed_cell": "no_cell_row"}.get(name, name)]))
    if name == "nothing":
        rewrite(f, golden_results(labels=variants["no_cell_row"]["labels"],
                                  molecules_of=variants["no_molecule"]["molecules_of"]))
        labels = dict(fuse=False)
    if name == "not_requested":
        labels = dict(fuse=False)
    if name == "fov_not_placed":
        labels = {}
    if name == "part_not_run":
        monkeypatch.setitem(sys.modules, "multiview_stitcher", None)
    with pytest.warns(ExportIncompleteWarning) if name == "part_not_run" else nullcontext():
        export(f, sd_config(zarr_format, **labels), assignment=None if name == "molecules_only" else "default")
    return f.export_dir


#: The configurations (case, Zarr format) that SD1 to SD4 each build alike, exported once for the module by ``exports``.
SHARED = (("usual", 3), ("fov_not_placed", 3))


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    """``exports(name, zarr_format)``: the export folder of ``build_case``, built once for the module per complete
    configuration (the case and the Zarr format, one of ``SHARED``). SD1 to SD4 only read it; a test that changes an
    export builds its own."""
    import dask
    made = {}

    def get(name, zarr_format):
        assert (name, zarr_format) in SHARED
        if (name, zarr_format) not in made:
            with dask.config.set(scheduler="synchronous"), pytest.MonkeyPatch.context() as patch:
                made[(name, zarr_format)] = build_case(name, tmp_path_factory.mktemp(f"sd-{name}-{zarr_format}"),
                                                       zarr_format, patch)
        return made[(name, zarr_format)]
    return get


IMAGE = "reference_stain_projection"
W = ("written", None)
E = ("omitted", "empty")
#: The expected element of each case: (status, reason), from the table "Empty populations".
CASES = {
    "usual": {"molecules": W, "cell_centroids": W, "cells": W, "expression": W, "cells_unplaced": E,
              "molecules_unplaced": E, IMAGE: W, "cell_labels": W},
    "fov_not_placed": {"molecules": W, "cell_centroids": W, "cells": W, "expression": W, "cells_unplaced": W,
                       "molecules_unplaced": W, IMAGE: W, "cell_labels": ("omitted", "not_requested")},
    "no_placed_cell": {"molecules": W, "cell_centroids": E, "cells": E, "expression": E, "cells_unplaced": E,
                       "molecules_unplaced": E, IMAGE: W, "cell_labels": E},
    "no_counted_cell": {"molecules": W, "cell_centroids": W, "cells": W, "expression": E, "cells_unplaced": E,
                        "molecules_unplaced": E, IMAGE: W, "cell_labels": W},
    "molecules_only": {"molecules": W, "cell_centroids": E, "cells": E, "expression": E, "cells_unplaced": E,
                       "molecules_unplaced": E, IMAGE: W, "cell_labels": ("omitted", "no_cells")},
    "no_molecule": {"molecules": E, "cell_centroids": W, "cells": W, "expression": W, "cells_unplaced": E,
                    "molecules_unplaced": E, IMAGE: W, "cell_labels": W},
    "no_included_fov": {"molecules": E, "cell_centroids": E, "cells": E, "expression": E, "cells_unplaced": W,
                        "molecules_unplaced": W, IMAGE: ("omitted", "no_included_fov"),
                        "cell_labels": ("omitted", "not_requested")},
    "not_requested": {"molecules": W, "cell_centroids": W, "cells": W, "expression": W, "cells_unplaced": E,
                      "molecules_unplaced": E, IMAGE: ("omitted", "not_requested"),
                      "cell_labels": ("omitted", "not_requested")},
    "part_not_run": {"molecules": W, "cell_centroids": W, "cells": W, "expression": W, "cells_unplaced": E,
                     "molecules_unplaced": E, IMAGE: ("omitted", "part_not_run"),
                     "cell_labels": ("omitted", "part_not_run")},
    "nothing": {name: E for name in FIXED[:-1]} | {IMAGE: ("omitted", "not_requested"),
                                                   "cell_labels": ("omitted", "not_requested")},
}
KIND = {"molecules": "points", "cell_centroids": "shapes", "cells": "table", "expression": "table",
        "cells_unplaced": "table", "molecules_unplaced": "table", IMAGE: "image", "cell_labels": "labels"}


def populations(folder):
    """The rows of each population, recomputed here from the sample tables and sample.h5ad."""
    import anndata
    assembly = load_assembly(folder)
    cells, mols = assembly.cells, assembly.molecules
    placed_c = np.isfinite(cells[["sample_ownership_centroid_y", "sample_ownership_centroid_x"]]
                           .to_numpy(np.float64)).all(axis=1) if len(cells) else np.zeros(0, bool)
    placed_m = np.isfinite(mols[["sample_z", "sample_y", "sample_x"]].to_numpy(np.float64)).all(axis=1) \
        if len(mols) else np.zeros(0, bool)
    n_obs = anndata.read_h5ad(folder / "sample.h5ad").n_obs
    return {"molecules": int(placed_m.sum()), "cell_centroids": int(placed_c.sum()), "cells": int(placed_c.sum()),
            "expression": n_obs, "cells_unplaced": int((~placed_c).sum()),
            "molecules_unplaced": int((~placed_m).sum())}


#: Each case with each Zarr format; fov_not_placed (both unplaced tables written, an extra case beside the rows of
#: the table) takes 5 s or more on one CPU with Zarr format 2, so it is slow.
SD1_CASES = [pytest.param(case, zarr_format, marks=[pytest.mark.slow] if (case, zarr_format) == ("fov_not_placed", 2)
                          else []) for case in CASES for zarr_format in (3, 2)]


@needs_library
@pytest.mark.parametrize("case, zarr_format", SD1_CASES)
def test_sd1_elements_omissions_and_format(case, zarr_format, exports, tmp_path, monkeypatch):
    folder = exports(case, zarr_format) if (case, zarr_format) in SHARED else \
        build_case(case, tmp_path, zarr_format, monkeypatch)
    record = record_of(folder)
    section = record["spatialdata"]
    assert statuses(record) == CASES[case]
    assert {e["name"]: e["kind"] for e in section["elements"]} == {n: KIND[n] for n in CASES[case]}
    # Each omitted reason, recomputed here.
    rows = populations(folder)
    for name, (status, reason) in CASES[case].items():
        if name in rows:
            assert (rows[name] > 0) == (status == "written"), name
    if case == "molecules_only":
        assert record["assembly"]["mode"] == "molecules_only" and record["labels"]["reason"] == "no_cells"
    if case == "no_placed_cell":
        assert record["labels"]["status"] == "written" and record["labels"]["n_ids_output"] == 0
    if case == "no_included_fov":
        assert record["fusion"]["reason"] == "no_included_fov" and record["labels"] == {"status": "off"}
    if case in ("not_requested", "nothing"):
        assert record["fusion"] == {"status": "off"} and record["labels"] == {"status": "off"}
    if case == "part_not_run":
        assert record["parts"]["fuse"] == record["parts"]["fuse_labels"] == "not_run"
    assert not list(folder.glob("*.ome.zarr")) and staging(folder) == []
    written = {n for n, (s, _) in CASES[case].items() if s == "written"}
    if not written:
        assert section["store"] is None and section["status"] == "empty" and not (folder / "sample.zarr").exists()
        assert record["parts"]["export_spatialdata"] == "empty"
        check_export(folder)
        return
    assert section["store"] == "sample.zarr" and section["status"] == "written"
    store = read_store(folder)
    found = {n: k for k, plural in PLURAL.items() for n in getattr(store, plural).keys()}
    assert found == {n: KIND[n] for n in written}
    assert len({n.lower() for n in found}) == len(found)
    import zarr
    group = zarr.open_group(str(folder / "sample.zarr"), mode="r")
    assert group.metadata.zarr_format == zarr_format and section["format"]["zarr_format"] == zarr_format
    assert dict(group.attrs)["spatialdata_attrs"]["version"] == {3: "0.2", 2: "0.1"}[zarr_format]
    assert dict(group.attrs)["frame_id"] == f"sample:{SAMPLE}"
    for name in written:
        if KIND[name] in ("image", "labels"):
            attrs = dict(group[f"{'images' if KIND[name] == 'image' else 'labels'}/{name}"].attrs)
            version = (attrs.get("ome") or {}).get("version") or attrs["multiscales"][0]["version"]
            assert version == RASTER_VERSION[zarr_format]
    check_export(folder)


@pytest.mark.contract
@pytest.mark.parametrize("call", ["entry", "alone"])
def test_sd1_a_sample_name_outside_the_rule_is_refused_before_anything_is_written(call, tmp_path):
    f = export_folder(tmp_path)
    if call == "entry":
        before = listing(f.root)
        with pytest.raises(ValueError, match="sample name 'sample one' breaks spatialdata's element naming rule"):
            f.dataset.export("sample one", placement=f.placement_dir, config=sd_config(), output_dir=f.export_dir)
        assert listing(f.root) == before
        return
    export(f, ExportConfig(route="per_fov", fuse=False))
    saved = load_assembly(f.export_dir)
    renamed = replace(saved, sample="sample one")
    before = listing(f.root)
    with pytest.raises(ValueError, match="sample name 'sample one' breaks"):
        export_spatialdata(renamed, output_dir=f.export_dir)
    assert listing(f.root) == before


# --- SD2 and SD3: points and tables ---------------------------------------------------------------------------------

@needs_library
@pytest.mark.parametrize("case", ["usual", "fov_not_placed"])
def test_sd2_points(case, exports):
    folder = exports(case, 3)
    section = record_of(folder)["spatialdata"]
    source = load_assembly(folder).molecules
    placed = np.isfinite(source[["sample_z", "sample_y", "sample_x"]].to_numpy(np.float64)).all(axis=1)
    store = read_store(folder)
    points = store.points["molecules"].compute().reset_index(drop=True)
    assert not {"sample_x", "sample_y", "sample_z"} & set(points.columns)
    entry = next(e for e in section["elements"] if e["name"] == "molecules")
    assert {str(c): str(d) for c, d in points.dtypes.items()} == entry["dtypes"]
    assert str(points.x.dtype) == str(points.y.dtype) == str(points.z.dtype) == "float64"
    assert str(points.cell_instance_id.dtype) == "UInt32"
    keys = ("spot_namespace", "spot_id")
    at = join(points, keys)
    expected = source[placed].reset_index(drop=True)
    assert sorted(at, key=str) == sorted(join(expected, keys), key=str)
    order = [at[k] for k in join(expected, keys)]
    got = points.iloc[order].reset_index(drop=True)
    for axis in "xyz":
        assert np.array_equal(got[axis].to_numpy(np.float64), expected[f"sample_{axis}"].to_numpy(np.float64))
        assert values(got[f"fov_index_{axis}"]) == values(expected[axis])
    for column in expected.columns:
        if column not in ("x", "y", "z", "sample_x", "sample_y", "sample_z"):
            assert values(got[column]) == values(expected[column]), column
    kept = {c for c in expected.columns if c not in ("x", "y", "z", "sample_x", "sample_y", "sample_z")}
    assert set(got.columns) == kept | {"x", "y", "z", "fov_index_z", "fov_index_y", "fov_index_x", "cell_instance_id"}
    # With molecules_unplaced, every row of the sample molecule table exactly once.
    keys_all = set(join(source, keys))
    unplaced = set(join(store.tables["molecules_unplaced"].obs, keys)) if "molecules_unplaced" in store.tables else set()
    assert len(at) + len(unplaced) == len(source) and set(at) | unplaced == keys_all
    assert (len(unplaced) > 0) == (case == "fov_not_placed")


@needs_library
@pytest.mark.parametrize("case", ["usual", "fov_not_placed"])
def test_sd3_tables_preserve_their_source(case, exports):
    import anndata
    folder = exports(case, 3)
    section = record_of(folder)["spatialdata"]
    entries = {e["name"]: e for e in section["elements"]}
    assembly = load_assembly(folder)
    store = read_store(folder)
    for name, source, keys in (("cells", assembly.cells, ("cell_namespace", "cell_id")),
                               ("cells_unplaced", assembly.cells, ("cell_namespace", "cell_id")),
                               ("molecules_unplaced", assembly.molecules, ("spot_namespace", "spot_id"))):
        if name not in store.tables:
            continue
        table = store.tables[name]
        assert {"obs": {str(c): str(d) for c, d in table.obs.dtypes.items()},
                "var": {str(c): str(d) for c, d in table.var.dtypes.items()}} == entries[name]["dtypes"]
        obs = table.obs.reset_index(drop=True)
        rows = source.iloc[[join(source, keys)[k] for k in join(obs, keys)]].reset_index(drop=True)
        for column in source.columns:
            assert values(obs[column]) == values(rows[column]), (name, column)
        assert table.n_vars == 0
    # cells and cells_unplaced together hold every cell row exactly once
    together = [k for name in ("cells", "cells_unplaced") if name in store.tables
                for k in join(store.tables[name].obs, ("cell_namespace", "cell_id"))]
    assert sorted(together, key=str) == sorted(join(assembly.cells, ("cell_namespace", "cell_id")), key=str)
    # expression against sample.h5ad
    with pd.option_context("future.infer_string", False):
        source = anndata.read_h5ad(folder / "sample.h5ad")
    table = store.tables["expression"]
    assert list(table.obs_names) == list(source.obs_names) and list(table.var_names) == list(source.var_names)
    assert np.array_equal(table.X, source.X) and table.X.dtype == source.X.dtype
    assert sorted(table.layers) == sorted(source.layers) and sorted(table.obsm) == sorted(source.obsm)
    for key in source.layers:
        assert np.array_equal(table.layers[key], source.layers[key], equal_nan=True)
    for key in source.obsm:
        assert np.array_equal(table.obsm[key], source.obsm[key])
    for column in source.obs.columns:
        assert values(table.obs[column]) == values(source.obs[column]), column
    assert table.var.equals(source.var)
    for key in source.uns:
        assert table.uns[key] == source.uns[key], key
    assert {str(c): str(d) for c, d in table.obs.dtypes.items()} == entries["expression"]["dtypes"]["obs"]


def test_sd3_a_table_key_equal_ignoring_case_is_refused(tmp_path):
    f = export_folder(tmp_path)
    export(f, ExportConfig(route="per_fov", fuse=False))
    saved = load_assembly(f.export_dir)
    cells = saved.cells.assign(Size=saved.cells["size_voxels"], size=saved.cells["size_voxels"])
    before = listing(f.root)
    with pytest.raises(ValueError, match="table 'cells': the obs keys 'Size' and 'size' are equal when case is "
                                         "ignored"):
        export_spatialdata(replace(saved, cells=cells), output_dir=f.export_dir)
    assert listing(f.root) == before


# --- SD4: export metadata, separately, and the refusal ---------------------------------------------------------------

ATTRS = {"region": "cell_centroids", "region_key": "region", "instance_key": "instance_id"}


@needs_library
@pytest.mark.parametrize("case", ["usual", "fov_not_placed"])
def test_sd4_export_metadata(case, exports):
    import anndata
    folder = exports(case, 3)
    entries = {e["name"]: e for e in record_of(folder)["spatialdata"]["elements"]}
    assembly = load_assembly(folder)
    store = read_store(folder)
    with pd.option_context("future.infer_string", False):
        source = anndata.read_h5ad(folder / "sample.h5ad")
    for name, columns in (("cells", list(assembly.cells.columns)), ("expression", list(source.obs.columns))):
        table = store.tables[name]
        assert list(table.obs.columns) == columns + ["region", "instance_id"]
        assert isinstance(table.obs.region.dtype, pd.CategoricalDtype) and set(table.obs.region) == {"cell_centroids"}
        assert table.obs.instance_id.dtype == np.uint32
        assert dict(table.uns["spatialdata_attrs"]) == ATTRS and entries[name]["spatialdata_attrs"] == ATTRS
        assert set(table.uns) == set(source.uns if name == "expression" else {}) | {"spatialdata_attrs"}
        assert sorted(table.obsm) == sorted(source.obsm if name == "expression" else [])
    for name, column, frame in (("cells_unplaced", "instance_id", assembly.cells),
                                ("molecules_unplaced", "cell_instance_id", assembly.molecules)):
        if name not in store.tables:
            continue
        table = store.tables[name]
        assert list(table.obs.columns) == list(frame.columns) + [column]
        assert dict(table.uns) == {} and not table.obsm and not table.layers
    for name in ("molecules", "cell_centroids", IMAGE, "cell_labels"):
        element = {"molecules": store.points, "cell_centroids": store.shapes, IMAGE: store.images,
                   "cell_labels": store.labels}[name].get(name)
        assert (element is None) == (name == "cell_labels" and case == "fov_not_placed")
        if element is not None:
            assert set(element.attrs) == {"transform"}, name


def test_sd4_a_table_that_already_has_spatialdata_attrs_is_refused(tmp_path):
    import anndata
    f = export_folder(tmp_path)
    export(f, ExportConfig(route="per_fov", fuse=False))
    path = f.export_dir / "sample.h5ad"
    with pd.option_context("future.infer_string", False):
        adata = anndata.read_h5ad(path)
        adata.uns["spatialdata_attrs"] = dict(ATTRS)
        adata.write_h5ad(path)
    record = record_of(f.export_dir)
    record["assembly"]["files"]["anndata"]["sha256"] = file_sha256(path)
    (f.export_dir / "export.json").write_text(json.dumps(record, indent=2))
    saved = load_assembly(f.export_dir)
    before = listing(f.root)
    with pytest.raises(ValueError, match=r"has already been set") as raised:
        export_spatialdata(saved, output_dir=f.export_dir)
    assert type(raised.value) is ValueError and "spatialdata_attrs" in str(raised.value)
    assert listing(f.root) == before
    assert not (f.export_dir / "sample.zarr").exists() and staging(f.export_dir) == []
    with pd.option_context("future.infer_string", False):
        assert dict(anndata.read_h5ad(path).uns["spatialdata_attrs"]) == ATTRS


# --- SD5: links, accounting and cell_centroids ------------------------------------------------------------------------

def expanded_plane_results():
    """The golden FOVs as one-plane FOVs, each assigned with a 2 µm planar expansion."""
    from starfinder.assignment import AssignmentConfig, assign_molecules
    from starfinder.segmentation import ExpandLabelsConfig

    from .assembly_fixtures import golden_pieces, label_run
    grid = unit_grid((1,) + TILE_ZYX[1:])
    config = AssignmentConfig(expansion=ExpandLabelsConfig(2.0, "um", "planar"))
    results = {}
    for fov, (labels, _, local) in golden_pieces("3d").items():
        rows = [((0.0, p[1], p[2]), gene) for p, gene in local]
        results[fov] = assign_molecules(molecules(fov, rows, GOLDEN_GENES),
                                        label_run(labels.max(axis=0)[None], fov, grid), grid=grid, config=config)
    return results


def expanded_folder(tmp_path, dimensions):
    if dimensions == "3d":
        return rewrite(export_folder(tmp_path), golden_results("3d", expand=tuple(GOLDEN_IDS)))
    return rewrite(export_folder(tmp_path, dimensions="plane"), expanded_plane_results())


@needs_library
@pytest.mark.parametrize("dimensions", ["3d", "plane"])
def test_sd5_links_accounting_and_circles(dimensions, tmp_path):
    f = expanded_folder(tmp_path, dimensions)
    export(f, sd_config(fuse_labels=True))
    folder = f.export_dir
    record = record_of(folder)
    assembly = load_assembly(folder)
    cells = assembly.cells
    assert set(cells.ownership_centroid) == {"expanded"}
    store = read_store(folder)
    instance = {k: i + 1 for k, i in join(cells, ("cell_namespace", "cell_id")).items()}
    ids = []
    for name in ("cells", "cells_unplaced"):
        if name in store.tables:
            obs = store.tables[name].obs
            assert values(obs.instance_id) == [instance[k] for k in join(obs, ("cell_namespace", "cell_id"))]
            ids += values(obs.instance_id)
    assert sorted(ids) == list(range(1, len(cells) + 1))
    shapes = store.shapes["cell_centroids"]
    obs = store.tables["cells"].obs
    assert [int(v) for v in shapes.index] == values(obs.instance_id)
    rows = cells.iloc[[i - 1 for i in values(obs.instance_id)]]
    assert np.array_equal(shapes.geometry.x.to_numpy(), rows.sample_ownership_centroid_x.to_numpy(np.float64))
    assert np.array_equal(shapes.geometry.y.to_numpy(), rows.sample_ownership_centroid_y.to_numpy(np.float64))
    size = rows.size_voxels.to_numpy(np.float64)
    expected = np.cbrt(3 * size * 1.0 * 1.0 * 1.0 / (4 * np.pi)) if dimensions == "3d" else np.sqrt(size / np.pi)
    assert np.allclose(shapes.radius.to_numpy(), expected, rtol=1e-12, atol=0)
    # Molecule links.
    of = {(fov, int(c)): instance[(ns, int(c))] for fov, ns, c in zip(cells.fov_id, cells.cell_namespace, cells.cell_id)}
    for frame in [store.points["molecules"].compute()] + \
            ([store.tables["molecules_unplaced"].obs] if "molecules_unplaced" in store.tables else []):
        expected = [None if c is None else of[(fov, int(c))] for fov, c in zip(values(frame.fov_id),
                                                                              values(frame.cell_id))]
        assert values(frame.cell_instance_id) == expected
    # Expression rows are retained, kept cells rows.
    by_name = dict(zip(obs.obs_name.astype(object), range(len(obs))))
    table = store.tables["expression"]
    picked = obs.iloc[[by_name[n] for n in table.obs_names]]
    assert (picked.assembly_status == "retained").all() and (picked.status == "kept").all()
    assert values(table.obs.instance_id) == values(picked.instance_id)
    # Counts recounted from the store.
    frames = [store.points["molecules"].compute()] + \
        ([store.tables["molecules_unplaced"].obs] if "molecules_unplaced" in store.tables else [])
    joined = pd.concat([fr[["assignment_status", "assembly_status", "assembly_reason"]].astype(object)
                        for fr in frames])
    counts = record["spatialdata"]["counts"]["molecules"]
    for status in ("assigned", "unassigned", "excluded_cell", "outside_grid"):
        assert counts["assignment_status"][status] == int((joined.assignment_status == status).sum())
    assert counts["assembly"]["retained"] == int((joined.assembly_status == "retained").sum()) == \
        record["assembly"]["counts"]["molecules"]["retained"]
    for reason, n in record["assembly"]["counts"]["molecules"]["excluded"].items():
        assert counts["assembly"]["excluded"][reason] == n == int((joined.assembly_reason == reason).sum())
    check_export(folder)


def test_sd5_a_cell_of_zero_voxels_is_refused(tmp_path):
    f = export_folder(tmp_path)
    export(f, ExportConfig(route="per_fov", fuse=False))
    saved = load_assembly(f.export_dir)
    cells = saved.cells.copy()
    i = int(np.flatnonzero(cells.assembly_status.eq("retained").to_numpy(dtype=bool))[0])
    cells.loc[i, "size_voxels"] = 0
    before = listing(f.root)
    with pytest.raises(ValueError, match=f"size_voxels 0.*{cells.obs_name.iloc[i]}"):
        export_spatialdata(replace(saved, cells=cells), output_dir=f.export_dir)
    assert listing(f.root) == before
    assert not (f.export_dir / "sample.zarr").exists() and staging(f.export_dir) == []


# --- SD6 and SD7: calibration, alignment and values -------------------------------------------------------------------

def check_rasters(folder, *, points=True):
    """S6 and S7 against each OME-TIFF: transform, extent, the voxel of each molecule, values, one level."""
    import spatialdata
    from spatialdata.transformations import get_transformation
    record = record_of(folder)
    section = record["spatialdata"]
    store = read_store(folder)
    sample = section["coordinate_system"]
    outputs = {o["path"]: o for o in (record["fusion"].get("outputs") or []) + (record["labels"].get("outputs") or [])}
    mols = load_assembly(folder).molecules
    q_all = mols[["sample_z", "sample_y", "sample_x"]].to_numpy(np.float64)
    checked = 0
    for entry in section["elements"]:
        if entry["status"] != "written" or entry["kind"] not in ("image", "labels"):
            continue
        output = outputs[entry["source"]]
        element = (store.images if entry["kind"] == "image" else store.labels)[entry["name"]]
        S, O0, N = (np.asarray(output[k], float) for k in ("spacing_zyx", "origin_zyx", "shape_zyx"))
        axes = ["y", "x"] if output["axes"] == "YX" else ["z", "y", "x"]
        idx = ["zyx".index(a) for a in axes]
        transform = get_transformation(element, sample)
        matrix = transform.to_affine_matrix(input_axes=tuple(axes), output_axes=tuple(axes))
        assert np.allclose(np.diag(matrix)[:-1], S[idx], rtol=0, atol=1e-9)
        assert np.allclose(matrix[:-1, -1], O0[idx] - S[idx] / 2, rtol=0, atol=1e-9)
        assert np.count_nonzero(matrix[:-1, :-1] - np.diag(np.diag(matrix)[:-1])) == 0
        recorded = entry["transform"]
        assert np.allclose([recorded["scale"][a] for a in "zyx"], S, rtol=0, atol=1e-9)
        assert np.allclose([recorded["translation"][a] for a in "zyx"], O0 - S / 2, rtol=0, atol=1e-9)
        if output["axes"] == "YX":
            z_lo, z_hi = output["slab_z"] if output.get("slab_z") else (O0[0] - S[0] / 2, O0[0] + S[0] / 2)
            assert abs(recorded["scale"]["z"] - (z_hi - z_lo)) <= 1e-9
            assert abs(recorded["translation"]["z"] - z_lo) <= 1e-9
        extent = spatialdata.get_extent(element, coordinate_system=sample)
        for a, k in zip(axes, idx):
            assert abs(extent[a][0] - (O0[k] - S[k] / 2)) <= 1e-9
            assert abs(extent[a][1] - (O0[k] + (N[k] - 0.5) * S[k])) <= 1e-9
        # S7: values and dtype against the file's output level, one level.
        assert type(element).__name__ == "DataArray"
        data = np.asarray(element.data[0] if entry["kind"] == "image" else element.data)
        image = tifffile.imread(folder / output["path"])
        assert data.dtype == image.dtype and np.array_equal(data, image)
        assert sorted(p.name for p in (folder / entry["path"]).iterdir() if p.is_dir()) == ["0"]
        # Condition 5: a point at q falls in intrinsic voxel floor(j + 1/2).
        if points:
            for q in q_all:
                j = (q - O0) / S
                if np.any(np.abs((j + 0.5) - np.round(j + 0.5)) < 1e-6):
                    continue
                v = np.floor(j + 0.5).astype(int)[idx]
                w = np.floor(intrinsic(transform, axes, q[idx])).astype(int)
                assert np.array_equal(w, v)
                checked += 1
    for name in ("molecules", "cell_centroids"):
        element = store.points.get(name) if name == "molecules" else store.shapes.get(name)
        if element is not None:
            assert type(get_transformation(element, sample)).__name__ == "Identity"
    return checked


@needs_library
@pytest.mark.parametrize("scale", [1, 0.5])
def test_sd6_sd7_golden_3d_both(scale, tmp_path):
    f = export_folder(tmp_path)
    export(f, sd_config(fuse_labels=True, fusion=FusionConfig(dimensionality="both", scale_yx=scale)))
    entries = {e["name"]: e for e in record_of(f.export_dir)["spatialdata"]["elements"]}
    assert entries[IMAGE]["model"] == "Image2DModel" and entries["reference_stain_stack"]["model"] == "Image3DModel"
    assert check_rasters(f.export_dir) > 0
    check_export(f.export_dir)


@needs_library
def test_sd6_golden_2d_stack_slab(tmp_path):
    f = export_folder(tmp_path, dimensions="plane")
    export(f, sd_config(fusion=FusionConfig(dimensionality="stack")))
    record = record_of(f.export_dir)
    entry = next(e for e in record["spatialdata"]["elements"] if e["name"] == "reference_stain_stack")
    z_lo, z_hi = record["fusion"]["grid"]["slab_z"]
    assert entry["model"] == "Image2DModel" and entry["transform"]["store_axes"] == ["y", "x"]
    assert abs(entry["transform"]["scale"]["z"] - (z_hi - z_lo)) <= 1e-9
    assert abs(entry["transform"]["translation"]["z"] - z_lo) <= 1e-9
    assert check_rasters(f.export_dir) > 0
    check_export(f.export_dir)


@needs_library
def test_sd6_sd7_label_quad(tmp_path):
    from .label_fixtures import label_quad
    case = label_quad(tmp_path)
    folder = case.export_dir
    saved = save_assembly(case.assembly, folder)
    fuse_labels(case.placement, saved, config=LabelConfig(), output_dir=folder)
    export_spatialdata(load_assembly(folder), output_dir=folder)
    entry = next(e for e in record_of(folder)["spatialdata"]["elements"] if e["name"] == "cell_labels")
    assert entry["status"] == "written" and entry["model"] == "Labels2DModel" and entry["dtype"] == "uint32"
    assert entry["shape"] == [18, 18]
    assert check_rasters(folder) > 0
    check_export(folder)


def _store_placement(case):
    """placement_from_table of the nominal grid of a case, with its origins as ``origins``, approved."""
    ids = case.ids
    nominal = case.extra.get("nominal") or {f: case.origin(f) for f in ids}
    table = positions_from_table(pd.DataFrame({"fov_id": ids, "nominal_z": [0.0] * len(ids),
                                               "nominal_y": [float(nominal[f][1]) for f in ids],
                                               "nominal_x": [float(nominal[f][2]) for f in ids]}))
    origins = {f: tuple(float(v) for v in case.origin(f)) for f in ids}
    result = placement_from_table(table, case.grids, sample=case.placement.sample, origins=origins)
    return approve_placement(result, reviewer="test", note="store placement")


def _molecules_only(case, placement, points):
    inputs = [AssemblyInput(f, case.grids[f], molecules=molecules(f, [(tuple(float(v) for v in points[f]), "A")],
                                                                    ("A",)))
              for f in case.ids]
    return assemble(placement, inputs)


@needs_library
@pytest.mark.slow
def test_sd6_position_grid(tmp_path):
    case = position_grid()
    case.extra["nominal"] = {f"G{3 * r + c}": (0.0, r * 115.2 * LN_XY, c * 115.2 * LN_XY)
                             for r in range(3) for c in range(3)}
    placement = _store_placement(case)
    bright = case.extra["bright"]
    result = _molecules_only(case, placement, {f: bright[f][0] for f in case.ids})
    folder = tmp_path / "position"
    saved = save_assembly(result, folder)
    fuse_images(placement, case.inputs(), assembly=saved, config=FusionConfig(dimensionality="stack", scale_yx=1),
                output_dir=folder)
    export_spatialdata(load_assembly(folder), output_dir=folder)
    assert check_rasters(folder) == 9
    record = record_of(folder)
    from spatialdata.transformations import get_transformation
    store = read_store(folder)
    element = store.images["reference_stain_stack"]
    output = record["fusion"]["outputs"][0]
    S, O0 = np.asarray(output["spacing_zyx"]), np.asarray(output["origin_zyx"])
    image = tifffile.imread(folder / output["path"])
    data = np.asarray(element.data[0])
    transform = get_transformation(element, record["spatialdata"]["coordinate_system"])
    mols = saved.molecules
    for q in mols[["sample_z", "sample_y", "sample_x"]].to_numpy(np.float64):
        w = tuple(np.floor(intrinsic(transform, "zyx", q)).astype(int))
        v = tuple(np.floor((q - O0) / S + 0.5).astype(int))
        assert w == v and data[w] == image[v] > 0
        around = image[tuple(slice(max(i - 1, 0), i + 2) for i in v)]
        assert image[v] == around.max()
    assert np.allclose(mols.sample_z, [b[1][0] for b in bright.values()], rtol=0, atol=1e-9)
    check_export(folder)


@needs_library
@pytest.mark.slow
def test_sd7_fuse_band(tmp_path):
    case = fuse_band()
    placement = _store_placement(case)
    result = _molecules_only(case, placement, {f: (8.0, 128.0, 128.0) for f in case.ids})
    folder = tmp_path / "band"
    saved = save_assembly(result, folder)
    fuse_images(placement, case.inputs(), assembly=saved, config=FusionConfig(dimensionality="both"), output_dir=folder)
    export_spatialdata(load_assembly(folder), output_dir=folder)
    names = {e["name"]: e["status"] for e in record_of(folder)["spatialdata"]["elements"]}
    assert names[IMAGE] == names["reference_stain_stack"] == "written"
    check_rasters(folder)
    check_export(folder)


# --- FI11, the store clause: the stack alone --------------------------------------------------------------------------

@needs_library
@pytest.mark.parametrize("dimensions", ["3d", "plane"])
def test_fi11_the_stack_alone_in_the_store(dimensions, tmp_path):
    f = export_folder(tmp_path, dimensions=dimensions)
    images = ("reference_stain", "reference_merged") if dimensions == "3d" else ("reference_stain",)
    export(f, sd_config(fusion=FusionConfig(dimensionality="stack", images=images)))
    record = record_of(f.export_dir)
    store = read_store(f.export_dir)
    assert sorted(store.images) == sorted(f"{name}_stack" for name in images)
    assert not [n for n in store.images if n.endswith("_projection")]
    from spatialdata.models import get_model
    for name in images:
        output = next(o for o in record["fusion"]["outputs"] if o["image"] == name)
        element = store.images[f"{name}_stack"]
        path = f.export_dir / output["path"]
        n_z = output["shape_zyx"][0]
        if dimensions == "3d":
            assert get_model(element).__name__ == "Image3DModel" and element.dims == ("c", "z", "y", "x")
            i = n_z // 2
            assert np.array_equal(tifffile.imread(path, key=i), np.asarray(element.data[0, i]))
        else:
            assert get_model(element).__name__ == "Image2DModel" and element.dims == ("c", "y", "x") and n_z == 1
            assert np.array_equal(tifffile.imread(path, key=0), np.asarray(element.data[0]))
    check_export(f.export_dir)


# --- SD8: hashes, staging, lookup and the section ----------------------------------------------------------------------

@needs_library
def test_sd8_hashes_and_section(tmp_path):
    f = export_folder(tmp_path)
    export(f, sd_config(fuse_labels=True))
    record = record_of(f.export_dir)
    section = record["spatialdata"]
    assert all(key in section for key in SECTION_ITEMS)
    store = f.export_dir / "sample.zarr"
    files = {p.relative_to(store).as_posix(): file_sha256(p) for p in sorted(store.rglob("*")) if p.is_file()}
    assert section["outputs"][0]["files"] == files and section["outputs"][0]["path"] == "sample.zarr"
    assert section["assembly_sha256"] == section_sha256(record["assembly"])
    assert section["fusion_sha256"] == section_sha256(record["fusion"])
    assert section["labels_sha256"] == section_sha256(record["labels"])
    assert section["library"]["spatialdata"] == importlib.metadata.version("spatialdata")
    assert section["format"]["zarr_format"] == 3 and section["format"]["raster_version"] == "0.5-dev-spatialdata"
    assert section["renames"]["molecules"] == {"z": "fov_index_z", "y": "fov_index_y", "x": "fov_index_x",
                                               "sample_x": "x", "sample_y": "y", "sample_z": "z"}
    assert set(section["counts"]) == {"molecules", "cells"} and "counted" in section["counts"]["cells"]
    for entry in section["elements"]:
        assert {"name", "kind", "status"} <= set(entry)
        if entry["status"] == "omitted":
            assert entry["reason"]
            continue
        assert entry["path"] == f"sample.zarr/{PLURAL[entry['kind']]}/{entry['name']}"
        assert "rows" in entry or "shape" in entry
        if entry["kind"] == "table":
            assert entry["spatialdata_attrs"] in (ATTRS, None) and entry["dtypes"]["obs"]
        else:
            assert entry["transform"]["type"] in ("identity", "sequence")
        if entry["kind"] in ("image", "labels"):
            assert files[entry["array_metadata"][len("sample.zarr/"):]] == entry["array_metadata_sha256"]
    # A store file changed: S8 (R2 first, which lists every store file) fails.
    target = next(p for p in sorted(store.rglob("*")) if p.is_file() and p.name.endswith(".parquet"))
    kept = target.read_bytes()
    target.write_bytes(kept + b"\0")
    with pytest.raises(ValueError, match="R2: the listed file sample.zarr/.* has SHA-256"):
        check_export(f.export_dir)
    target.write_bytes(kept)
    check_export(f.export_dir)


@needs_library
def test_sd8_an_interrupted_export_publishes_nothing(tmp_path, monkeypatch):
    import spatialdata
    f = export_folder(tmp_path)
    export(f, ExportConfig(route="per_fov", fuse_labels=True))
    before = {k: v for k, v in listing(f.export_dir).items() if k != "export.json"}
    real, calls = spatialdata.SpatialData._write_element, []

    def interrupted(self, *args, **kwargs):
        calls.append(kwargs.get("element_name"))
        if len(calls) == 2:
            raise RuntimeError("interrupted by the test")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(spatialdata.SpatialData, "_write_element", interrupted)
    with pytest.raises(RuntimeError, match="interrupted by the test"):
        export_spatialdata(load_assembly(f.export_dir), output_dir=f.export_dir)
    assert len(calls) == 2
    assert not (f.export_dir / "sample.zarr").exists() and staging(f.export_dir) == []
    assert {k: v for k, v in listing(f.export_dir).items() if k != "export.json"} == before
    record = record_of(f.export_dir)
    assert record["parts"]["export_spatialdata"] == "failed" and record["complete"] is False
    assert record["spatialdata"]["error"] == {"type": "RuntimeError", "message": "interrupted by the test"}
    check_export(f.export_dir)


@needs_library
def test_sd8_lookup_source(tmp_path):
    f = export_folder(tmp_path)
    export(f, sd_config(fuse_labels=True))
    fov = "Position002"
    folder = f.assignment_dir(fov)
    loaded = read_assignment(folder, "default")
    for i in (0, len(loaded.molecules) - 1):
        row = loaded.molecules.iloc[i]
        found = lookup_source(f.export_dir, spot=(row.spot_namespace, row.spot_id))
        assert found == {"kind": "spot", "key": [row.spot_namespace, row.spot_id], "fov_id": fov,
                         "assignment_folder": str(folder), "row": i,
                         "assignment_record_sha256": file_sha256(folder / "assignment.json")}
    for j in (0, len(loaded.cells) - 1):
        cell_id = int(loaded.cells.cell_id.iloc[j])
        found = lookup_source(f.export_dir, cell=(loaded.cell_namespace, cell_id))
        assert found["fov_id"] == fov and found["row"] == j and found["assignment_folder"] == str(folder)
        assert found["assignment_record_sha256"] == file_sha256(folder / "assignment.json")
    with pytest.raises(ValueError, match="exactly one"):
        lookup_source(f.export_dir)
    with pytest.raises(ValueError, match="0 rows"):
        lookup_source(f.export_dir, spot=("nowhere", "m0"))
    record = record_of(f.export_dir)
    record["assembly"]["mode"] = "edited"
    (f.export_dir / "export.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="fails its hashes"):
        lookup_source(f.export_dir, spot=(loaded.molecules.spot_namespace.iloc[0], loaded.molecules.spot_id.iloc[0]))


@needs_library
def test_w400_the_readers_accept_the_duplicate_rule(tmp_path):
    """export_folder's one-plane variant holds scene cell 9 twice (FOV3's copy shares 0.90 of its territory with
    FOV1's); the default rule excludes FOV3's. The store, the record's counts, lookup_source and check_export
    accept the new reasons and the column duplicate_of."""
    f = export_folder(tmp_path, dimensions="plane")
    export(f, sd_config(fuse=False))
    assembly = load_assembly(f.export_dir)
    cells = assembly.cells.set_index("obs_name")
    assert (cells.loc["sample1_Position003_5", "assembly_reason"], cells.loc["sample1_Position003_5", "duplicate_of"]) \
        == ("duplicate_observation", "sample1_Position001_6")
    record = record_of(f.export_dir)
    counts = record["spatialdata"]["counts"]
    assert counts["cells"]["assembly"]["excluded"]["duplicate_observation"] == 1
    molecules = assembly.molecules
    assert counts["molecules"]["assembly"]["excluded"]["cell_duplicate_observation"] == int(
        molecules.assembly_reason.eq("cell_duplicate_observation").sum()) > 0
    store = read_store(f.export_dir)
    table = store.tables["cells"].obs
    assert table.loc["sample1_Position003_5", "duplicate_of"] == "sample1_Position001_6"
    assert table.loc["sample1_Position003_5", "assembly_reason"] == "duplicate_observation"
    namespace, cell_id = cells.loc["sample1_Position003_5", ["cell_namespace", "cell_id"]]
    found = lookup_source(f.export_dir, cell=(namespace, int(cell_id)))
    assert found["fov_id"] == "Position003" and found["row"] == 4
    check_export(f.export_dir)


#: Every SHA-256 of a section that another section records when it reads it: (recording section, field, section).
RECORDED = [("fusion", "assembly_sha256", "assembly"), ("labels", "assembly_sha256", "assembly"),
            ("spatialdata", "assembly_sha256", "assembly"), ("spatialdata", "fusion_sha256", "fusion"),
            ("spatialdata", "labels_sha256", "labels")]


@needs_library
def test_sd8_lookup_source_verifies_every_recorded_section_hash(tmp_path):
    """Each recorded section hash alone: the section changed, only that field left stale; spot and cell raise."""
    f = export_folder(tmp_path)
    export(f, sd_config(fuse_labels=True))
    path = f.export_dir / "export.json"
    original = path.read_text()
    record = json.loads(original)
    assert {(r, k, n) for r, fields in record.items() if isinstance(fields, dict)
            for k, n in (("assembly_sha256", "assembly"), ("fusion_sha256", "fusion"), ("labels_sha256", "labels"))
            if fields.get(k) is not None} == set(RECORDED)
    saved = load_assembly(f.export_dir)
    spot = (saved.molecules.spot_namespace.iloc[0], saved.molecules.spot_id.iloc[0])
    cell = (saved.cells.cell_namespace.iloc[0], int(saved.cells.cell_id.iloc[0]))
    expected = {"spot": lookup_source(f.export_dir, spot=spot), "cell": lookup_source(f.export_dir, cell=cell)}
    for recorder, field, name in RECORDED:
        edited = json.loads(original)
        edited[name]["edited_by_the_test"] = True
        for other, other_field, other_name in RECORDED:     # every other record of the changed section matches
            if other_name == name and (other, other_field) != (recorder, field):
                edited[other][other_field] = section_sha256(edited[name])
        assert edited[recorder][field] == record[recorder][field] != section_sha256(edited[name])
        path.write_text(json.dumps(edited))
        for key in ({"spot": spot}, {"cell": cell}):
            with pytest.raises(ValueError, match=f"fails its hashes: the {name} section has SHA-256 [0-9a-f]{{64}}, "
                                                 f"but {recorder}\\.{field} recorded {record[recorder][field]}"):
                lookup_source(f.export_dir, **key)
    path.write_text(original)
    assert lookup_source(f.export_dir, spot=spot) == expected["spot"]
    assert lookup_source(f.export_dir, cell=cell) == expected["cell"]


@pytest.mark.contract
def test_sd8_lookup_source_with_the_spatialdata_part_off(tmp_path):
    """A folder whose spatialdata (and fusion, labels) parts were off: the lookup works from the assembly section."""
    f = export_folder(tmp_path)
    record = export(f, ExportConfig(route="per_fov", fuse=False))
    assert record.parts == {"assemble": "written", "fuse": "off", "segment": "off", "fuse_labels": "off",
                            "export_spatialdata": "off"}
    saved = load_assembly(f.export_dir)
    fov = saved.cells.fov_id.iloc[0]
    folder = f.assignment_dir(fov)
    found = lookup_source(f.export_dir, cell=(saved.cells.cell_namespace.iloc[0], int(saved.cells.cell_id.iloc[0])))
    assert found["fov_id"] == fov and found["row"] == 0 and found["assignment_folder"] == str(folder)
    assert found["assignment_record_sha256"] == file_sha256(folder / "assignment.json")
    found = lookup_source(f.export_dir, spot=(saved.molecules.spot_namespace.iloc[0], saved.molecules.spot_id.iloc[0]))
    assert found["fov_id"] == saved.molecules.fov_id.iloc[0] and found["row"] == 0


# --- EX1 and EX3: the SpatialData part inside the entry -----------------------------------------------------------------

@needs_library
def test_ex1_the_part_runs_fourth(tmp_path, monkeypatch):
    import starfinder.assembly
    f = export_folder(tmp_path)
    calls = []
    real_assemble, real_fuse = starfinder.assembly.assemble, export_module.fuse_images
    real_labels, real_sd = export_module.fuse_labels, export_module._spatialdata_part
    monkeypatch.setattr(starfinder.assembly, "assemble", lambda *a, **k: calls.append("assemble") or
                        real_assemble(*a, **k))
    monkeypatch.setattr(export_module, "fuse_images", lambda *a, **k: calls.append("fuse") or real_fuse(*a, **k))
    monkeypatch.setattr(export_module, "fuse_labels", lambda *a, **k: calls.append("fuse_labels") or
                        real_labels(*a, **k))
    monkeypatch.setattr(export_module, "_spatialdata_part", lambda *a, **k: calls.append("export_spatialdata") or
                        real_sd(*a, **k))
    record = export(f, sd_config(fuse_labels=True))
    assert calls == ["assemble", "fuse", "fuse_labels", "export_spatialdata"]
    assert record.parts == {"assemble": "written", "fuse": "written", "segment": "off", "fuse_labels": "written",
                            "export_spatialdata": "written"} and record.complete
    files = sorted(listing(f.export_dir))
    assert [p for p in files if not p.startswith("sample.zarr/")] == sorted(
        TABLE_FILES + ["cell_labels.ome.tif", "reference_stain_projection.ome.tif"])
    assert any(p.startswith("sample.zarr/") for p in files)
    assert {p.name for p in f.root.iterdir()} >= {"export"} and not (f.root / "images" / "fused").exists()
    check_export(f.export_dir)


@needs_library
def test_ex1_the_part_called_alone(tmp_path):
    """Called alone with the saved files of the parts before it: its section only changes."""
    f = export_folder(tmp_path)
    export(f, ExportConfig(route="per_fov", fuse_labels=True))
    before = record_of(f.export_dir)
    assert before["spatialdata"] == {"status": "off"}
    export_spatialdata(load_assembly(f.export_dir), fused=FusionRecord("written", before["fusion"], ()),
                       config=SpatialDataExportConfig(zarr_format=2), output_dir=f.export_dir)
    after = record_of(f.export_dir)
    for name in ("assembly", "fusion", "labels"):
        assert canonical(after[name]) == canonical(before[name])
    assert after["parts"]["export_spatialdata"] == "written" and after["complete"] is True
    assert after["spatialdata"]["format"]["zarr_format"] == 2
    assert after["spatialdata"]["assembly_sha256"] == section_sha256(after["assembly"])
    assert after["spatialdata"]["labels_sha256"] == section_sha256(after["labels"])
    check_export(f.export_dir)
    other = FusionRecord("written", dict(before["fusion"], status="empty"), ())
    listed = listing(f.export_dir)
    with pytest.raises(ValueError, match="fusion record given is not the fusion section"):
        export_spatialdata(load_assembly(f.export_dir), fused=other, output_dir=f.export_dir)
    assert listing(f.export_dir) == listed


@needs_library
def test_ex3_a_fuse_not_run(tmp_path, monkeypatch):
    f = export_folder(tmp_path)
    monkeypatch.setitem(sys.modules, "multiview_stitcher", None)
    with pytest.warns(ExportIncompleteWarning) as caught:
        record = export(f, sd_config(fuse_labels=True))
    messages = [str(w.message) for w in caught if w.category is ExportIncompleteWarning]
    assert len(messages) == 2 and "fuse could not run: extra_missing" in messages[0]
    assert "fuse_labels could not run: extra_missing" in messages[1]
    assert record.parts == {"assemble": "written", "fuse": "not_run", "segment": "off", "fuse_labels": "not_run",
                            "export_spatialdata": "written"} and record.record["complete"] is False
    assert statuses(record.record)[IMAGE] == ("omitted", "part_not_run")
    assert statuses(record.record)["cell_labels"] == ("omitted", "part_not_run")
    assert [p for p in sorted(listing(f.export_dir)) if not p.startswith("sample.zarr/")] == TABLE_FILES
    check_export(f.export_dir)


def test_ex3_d_spatialdata_not_importable(tmp_path, monkeypatch):
    f = export_folder(tmp_path)
    monkeypatch.setitem(sys.modules, "spatialdata", None)
    config = sd_config(fuse=importlib.util.find_spec("multiview_stitcher") is not None)
    with pytest.warns(ExportIncompleteWarning) as caught:
        record = export(f, config)
    messages = [str(w.message) for w in caught if w.category is ExportIncompleteWarning]
    assert messages == ["export part export_spatialdata could not run: extra_missing: the 'spatialdata' extra is "
                        "not installed"]
    section = record.record["spatialdata"]
    assert record.parts["export_spatialdata"] == "not_run" and section["reason"] == "extra_missing"
    assert section["extra"] == "spatialdata" and section["store"] is None and section["outputs"] == []
    assert record.record["complete"] is False and not (f.export_dir / "sample.zarr").exists()
    expected = TABLE_FILES + (["reference_stain_projection.ome.tif"] if config.fuse else [])
    assert sorted(listing(f.export_dir)) == sorted(expected)
    check_export(f.export_dir)


@needs_library
def test_ex3_g_a_territory_missing_without_conflicts(tmp_path):
    f = export_folder(tmp_path)
    export(f, ExportConfig(route="per_fov", fuse=False, assembly=AssemblyConfig(conflicts=False)))
    territory = f.assignment_dir("Position002") / "cell_labels.tif"
    territory.unlink()
    with pytest.warns(ExportIncompleteWarning, match="fuse_labels could not run: territory_missing") as caught:
        record = export(f, sd_config(assemble=False, fuse=False, fuse_labels=True))
    assert len([w for w in caught if w.category is ExportIncompleteWarning]) == 1
    assert record.parts == {"assemble": "reused", "fuse": "off", "segment": "off", "fuse_labels": "not_run",
                            "export_spatialdata": "written"} and record.record["complete"] is False
    assert statuses(record.record)["cell_labels"] == ("omitted", "part_not_run")
    assert statuses(record.record)[IMAGE] == ("omitted", "not_requested")
    assert [p for p in sorted(listing(f.export_dir)) if not p.startswith("sample.zarr/")] == TABLE_FILES
    check_export(f.export_dir)


@needs_library
def test_check_export_finds_a_store_that_differs_from_its_section(tmp_path):
    """check_export's S conditions fail, naming the condition, on a store changed after the export."""
    f = export_folder(tmp_path)
    export(f, sd_config())
    record = record_of(f.export_dir)
    edited = json.loads(json.dumps(record))
    entry = next(e for e in edited["spatialdata"]["elements"] if e["name"] == "cells_unplaced")
    entry["reason"] = "no_cells"
    (f.export_dir / "export.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="S1: the element cells_unplaced is omitted with reason 'no_cells'"):
        check_export(f.export_dir)
    edited = json.loads(json.dumps(record))
    edited["spatialdata"]["counts"]["molecules"]["assembly"]["retained"] += 1
    (f.export_dir / "export.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="S5: the counts"):
        check_export(f.export_dir)
    edited = json.loads(json.dumps(record))
    next(e for e in edited["spatialdata"]["elements"] if e["name"] == "cells")["dtypes"]["obs"]["cell_id"] = "int64"
    (f.export_dir / "export.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="S3: the column dtypes of cells"):
        check_export(f.export_dir)
    (f.export_dir / "export.json").write_text(json.dumps(record))
    check_export(f.export_dir)
