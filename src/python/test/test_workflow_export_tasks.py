"""The export command, its previews (P5) and the assign command's previews (moved by W-750 from the tests of the
export_sample and reads_assignment adapters: test_stitching_workflow.py, test_sample_segmentation_workflow.py and
test_workflow_assembly_figures.py).

Each test plans a configuration in the Python shape on an input root whose raw images are empty files
(``workflow_tasks``); the dataset the old tests wrote (``assembly_golden`` by ``fusion_fixtures.export_folder``,
``fused_pair`` by ``sample_segmentation_fixtures``, the sample of ``test_workflow_assembly_golden.py``) is written
into the plan's output root as the earlier tasks would have written it, their records say succeeded, and the export
(and assign, segment) tasks run by their commands, or the export's function is called on its ``Run`` with
``Dataset.export`` spied. dask runs with the synchronous scheduler.
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from starfinder.assembly import AssemblyConfig, fov_ownership, load_assembly  # noqa: E402
from starfinder.assignment._persist import read_assignment  # noqa: E402
from starfinder.dataset import Dataset  # noqa: E402
from starfinder.export import ExportConfig, ExportIncompleteWarning, check_export  # noqa: E402
from starfinder.io import save_volume  # noqa: E402
from starfinder.stitching import load_placement, save_placement  # noqa: E402

from . import workflow_examples as X  # noqa: E402
from . import workflow_tasks as T  # noqa: E402
from .assembly_fixtures import GOLDEN_IDS, golden_placement, unit_grid  # noqa: E402
from .assembly_fixtures import SAMPLE as GOLDEN_SAMPLE  # noqa: E402
from .fusion_fixtures import export_folder, golden_codebook, listing  # noqa: E402
from .sample_segmentation_fixtures import IMAGES, SCENES, codebook, per_fov_routes, write_scene  # noqa: E402
from .sample_segmentation_fixtures import SAMPLE as SCENE_SAMPLE  # noqa: E402
from .test_sample_segmentation_route import export_grid, truth_config  # noqa: E402
from .test_stitching_assembly_figure import EXCLUDED, GREY, check_region, points, scatters, stated  # noqa: E402
from .test_workflow_assembly_golden import TILE_ZYX, build_sample  # noqa: E402

pytestmark = [pytest.mark.workflow, pytest.mark.stitching, pytest.mark.contract]

needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")
MERGED = {"method": "multiview_stitcher", "input": "reference_merged",
          "positions": {"source": "regular_grid", "regular_grid": {"columns": 2, "rows": 2,
                                                                   "order": "column_major_down_right",
                                                                   "overlap": 0.25}}}
#: The morphology block and its round, which the export's default fusion image (reference_stain) needs.
STAIN = {"reference_stains": [{"channel": "ch04", "name": "DAPI"}],
         "additional_round": [{"round_name": "morph", "channel_order": ["ch00"]}], "morphology": {}}


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def plan_sample(work, sample, fovs, *, export, assembly, cells=True, labels=None, previews=False, output_id="output",
                stain=True, codebook_table=None, **keys):
    """Plan the route of a sample to its export: the processing block (molecules), with ``cells`` a segment run
    ``cell`` that imports ``labels`` (a ``{fov_id}`` pattern; empty files when None) and the assignment ``default``,
    the stitching block, the given assembly and export blocks, and with ``stain`` the morphology block; the codebook
    ``codebook_table`` (entry_id, gene_id, color_sequence) is written first when given. Returns (manifest,
    configuration)."""
    config = T.dataset_keys(work, output_id=output_id, processing=copy.deepcopy(X.PROCESSING), stitching=MERGED,
                            assembly=assembly, export=export, **(STAIN if stain else {}), **keys)
    if cells:
        if labels is None:
            labels = str(Path(work) / "labels" / "{fov_id}.tif")
            for fov in fovs:
                path = Path(labels.replace("{fov_id}", fov))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
        config.update(segmentation={"runs": [{"name": "cell", "target": "cell", "method": "import", "path": labels}]},
                      assignment={"cells": "cell", "previews": previews})
    if codebook_table is not None:
        write_codebook(config, codebook_table)
    T.write_inputs(config, "sample,fov_id\n" + "".join(f"{sample},{fov}\n" for fov in fovs), codebook=None)
    manifest, _ = T.plan(work, f"{output_id}-{json.dumps(export, sort_keys=True)[:40]}", config)
    return manifest, config


def write_codebook(config, table):
    path = T.input_root(config) / "genes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table[["entry_id", "gene_id", "color_sequence"]].to_csv(path, index=False)
    return path


def golden_export(work, *, export, assembly, dimensions="3d", **keys):
    """assembly_golden written by export_folder into the plan's output root; (folder, manifest, configuration)."""
    folder = export_folder(work / "output" / "data", dimensions=dimensions)
    manifest, config = plan_sample(work, GOLDEN_SAMPLE, list(GOLDEN_IDS), export=export, assembly=assembly,
                                   codebook_table=golden_codebook().table, **keys)
    assert T.output_root(config) == folder.dataset.output_root
    return folder, manifest, config


def spy_export(monkeypatch):
    calls = []
    real = Dataset.export
    monkeypatch.setattr(Dataset, "export", lambda self, *a, **k: calls.append((a, k)) or real(self, *a, **k))
    return calls, real


# --- the export command: the defaults, a missing extra, the duplicate threshold, molecules only ---------------------

@needs_anndata
@needs_library
def test_the_export_command_writes_the_export_folder_with_the_defaults(tmp_path, monkeypatch):
    """From test_wf3_run_export_writes_the_export_folder_with_the_defaults: with an export block that sets only route:
    per_fov the defaults run, on the saved placement of stitching/<sample>/."""
    calls, _ = spy_export(monkeypatch)
    folder, manifest, _ = golden_export(tmp_path, export={"route": "per_fov"}, assembly={"table_format": "csv"})
    T.call(manifest, f"export/{GOLDEN_SAMPLE}")
    (args, kwargs), = calls
    assert args == (GOLDEN_SAMPLE,)
    assert kwargs == {"placement": folder.placement_dir, "assignment": "default",
                      "config": ExportConfig(route="per_fov"), "output_dir": folder.export_dir}
    record = json.loads((folder.export_dir / "export.json").read_text())
    assert record["parts"] == {"assemble": "written", "fuse": "written", "segment": "off", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    assert sorted(p.relative_to(folder.export_dir).as_posix() for p in folder.export_dir.rglob("*") if p.is_file()) \
        == ["export.json", "reference_stain_projection.ome.tif", "sample.h5ad", "tables/cells.csv",
            "tables/counts.csv", "tables/fovs.csv", "tables/molecules.csv"]
    # Nothing is written outside export/<sample>/ and stitching/<sample>/ is the saved placement, unchanged.
    assert load_placement(folder.placement_dir).fovs.equals(folder.placement.fovs)


@needs_anndata
def test_the_export_command_succeeds_without_multiview_stitcher(tmp_path, monkeypatch):
    """From test_wf3_run_export_succeeds_without_multiview_stitcher: the fusion is recorded not_run (extra_missing)
    and the export does not fail."""
    monkeypatch.setitem(sys.modules, "multiview_stitcher", None)
    folder, manifest, _ = golden_export(tmp_path, export={"route": "per_fov", "dimensionality": "stack"},
                                        assembly={"table_format": "csv"})
    with pytest.warns(ExportIncompleteWarning, match="fuse could not run: extra_missing"):
        T.call(manifest, f"export/{GOLDEN_SAMPLE}")
    record = json.loads((folder.export_dir / "export.json").read_text())
    assert record["parts"] == {"assemble": "written", "fuse": "not_run", "segment": "off", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    assert record["fusion"]["reason"] == "extra_missing" and record["complete"] is False


@needs_anndata
@pytest.mark.parametrize("threshold", [None, 0.5, 1.0])
def test_the_export_command_passes_the_duplicate_threshold(tmp_path, monkeypatch, threshold):
    """From test_wf3_run_export_passes_the_duplicate_threshold: assembly.duplicate_threshold reaches assemble, and the
    tables equal those of the same Dataset.export called from Python. The fixture's one-plane variant holds scene
    cell 9 twice (share 0.90 of the smaller)."""
    calls, real = spy_export(monkeypatch)
    folder, manifest, _ = golden_export(tmp_path / "workflow", dimensions="plane", export={"route": "per_fov",
                                                                                           "fuse": False},
                                        assembly={"table_format": "csv", "duplicate_threshold": threshold})
    T.call(manifest, f"export/{GOLDEN_SAMPLE}")
    record = json.loads((folder.export_dir / "export.json").read_text())
    expected = ExportConfig(route="per_fov", fuse=False, assembly=AssemblyConfig(duplicate_threshold=threshold))
    assert [k["config"] for _, k in calls] == [expected]
    assert record["assembly"]["config"]["duplicate_threshold"] == threshold
    python = export_folder(tmp_path / "python" / "data", dimensions="plane")
    real(python.dataset, GOLDEN_SAMPLE, placement=python.placement_dir, assignment="default", config=expected,
         output_dir=python.export_dir)
    again = json.loads((python.export_dir / "export.json").read_text())
    for table in ("fovs", "cells", "molecules", "counts"):
        assert record["assembly"]["files"][table]["sha256"] == again["assembly"]["files"][table]["sha256"]
    cells = pd.read_csv(folder.export_dir / "tables" / "cells.csv")
    excluded = cells[cells.assembly_reason == "duplicate_observation"]
    assert list(excluded.obs_name) == ([] if threshold in (None, 1.0) else ["sample1_Position003_5"])


@needs_anndata
def test_the_export_command_molecules_only_loads_the_codebook(tmp_path):
    """From test_wf3_run_export_molecules_only_loads_the_codebook: assembly.assignment null: the molecule tables are
    assembled, with the genes of the configured codebook."""
    folder, manifest, _ = golden_export(tmp_path, export={"fuse": False}, assembly={"assignment": None}, cells=False,
                                        stain=False)
    T.call(manifest, f"export/{GOLDEN_SAMPLE}")
    record = json.loads((folder.export_dir / "export.json").read_text())
    assert record["parts"] == {"assemble": "written", "fuse": "off", "segment": "off", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    assert record["assembly"]["mode"] == "molecules_only"
    assert record["assembly"]["genes"] == list(golden_codebook().genes)


# --- the export previews on the two routes (P5; from test_sample_segmentation_workflow.py's WF2) ---------------------

def scene_export(work, export, *, export_previews=True):
    """fused_pair written into the plan's output root with the per-FOV assignment folders of per_fov_routes, its
    codebook and the truth labels of the route after fusion; (folder, manifest)."""
    folder = write_scene(work / "output" / "data", SCENES["fused_pair"]())
    per_fov_routes(folder)
    if export.get("route") == "after_fusion":
        segmentation = truth_config(folder, export_grid(folder))
        runs = [{"name": run.name, "target": run.target, "method": "import", "path": run.method.path,
                 "relabel": False} for run in segmentation.runs]
        export = dict(export, segmentation={"cells": "cell", "nuclei": "nucleus", "runs": runs})
    export = dict(export, previews=export_previews)
    manifest, config = plan_sample(work, SCENE_SAMPLE, list(folder.scene.ids), export=export,
                                   assembly={"assignment": "default", "table_format": "csv"}, stain=False,
                                   codebook_table=codebook().table, voxel_size_xy=0.5, voxel_size_z=1.0,
                                   sample_id=SCENE_SAMPLE)
    assert T.output_root(config) == folder.root
    T.succeed(manifest, "fov", "segment", "assign", "stitch")
    return folder, manifest


AFTER_FUSION = {"route": "after_fusion", "images": list(IMAGES), "scale_yx": 1.0}


def figure_paths(folder):
    return {p.relative_to(folder.root).as_posix(): p.read_bytes() for p in sorted(folder.root.rglob("*.png"))}


@needs_anndata
@needs_library
def test_the_export_previews_follow_the_route(tmp_path, monkeypatch):
    """From test_wf2_the_assignment_figures: with export.previews on, on the route after fusion the export draws no
    assignment figure and declares no preview folder (the sample's cells come from no per-FOV assignment); on the
    route per FOV it draws them for every FOV with its ownership into export/<sample>/previews/<fov>/, and leaves the
    assign task's figures in fovs/<fov>/previews/ as they are."""
    import starfinder.workflow._previews as previews
    folder, manifest = scene_export(tmp_path, AFTER_FUSION)
    for fov in folder.scene.ids:  # the figures the assign task wrote first
        (folder.root / "fovs" / fov / "previews").mkdir(parents=True)
        (folder.root / "fovs" / fov / "previews" / "assignment.png").write_bytes(b"written by the assign task")
    before = figure_paths(folder)
    calls = []
    real = previews.assignment_figures
    monkeypatch.setattr(previews, "assignment_figures", lambda result, image, out, **kwargs: calls.append(
        (Path(out).name, kwargs)) or real(result, image, out, **kwargs))
    task = T.task(manifest, "export", sample=SCENE_SAMPLE)
    assert not [o for o in task["outputs"] if "previews" in o]
    assert not [i for i in task["inputs"] if "/assignment/" in i]
    assert T.run(manifest, f"export/{SCENE_SAMPLE}") == 0
    record = json.loads((folder.export_dir / "export.json").read_text())
    assert record["parts"] == {"assemble": "written", "fuse": "written", "segment": "written", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    assert record["assembly"]["route"] == "after_fusion" and record["complete"] is True
    assert set(record["assembly"]["fovs"][i]["kind"] for i in range(2)) == {"molecules"}
    assert calls == [] and figure_paths(folder) == before
    # The route per FOV of the same sample, into the same folder: the figures are drawn again with ownership.
    exported = listing(folder.export_dir)
    folder, manifest = scene_export(tmp_path / "per_fov", {"route": "per_fov", "fuse": False})
    for fov in folder.scene.ids:
        (folder.root / "fovs" / fov / "previews").mkdir(parents=True)
        (folder.root / "fovs" / fov / "previews" / "assignment.png").write_bytes(b"written by the assign task")
    before = figure_paths(folder)
    assert T.run(manifest, f"export/{SCENE_SAMPLE}") == 0
    record = json.loads((folder.export_dir / "export.json").read_text())
    assert record["assembly"]["route"] == "per_fov" and record["parts"]["segment"] == "off"
    assert sorted(fov for fov, _ in calls) == sorted(folder.scene.ids)
    assert all(kwargs["ownership"] is not None for _, kwargs in calls)
    after = figure_paths(folder)
    assert {k: v for k, v in after.items() if k.startswith("fovs/")} == before
    folder_previews = f"export/{SCENE_SAMPLE}/previews"
    assert {k.rsplit("/", 1)[0] for k in after if not k.startswith("fovs/")} == {f"{folder_previews}/{fov}"
                                                                                for fov in folder.scene.ids}
    assert all(after[f"{folder_previews}/{fov}/assignment.png"] != before[f"fovs/{fov}/previews/assignment.png"]
               for fov in folder.scene.ids)
    assert listing(folder.export_dir) != exported
    check_export(folder.export_dir)


@needs_anndata
@needs_library
@pytest.mark.parametrize("assemble", [True, False], ids=["assemble", "no_assemble"])
@pytest.mark.parametrize("route", ["per_fov", "after_fusion"])
def test_every_declared_output_of_the_export_task_exists(tmp_path, route, assemble):
    """From test_wf2_every_declared_output_of_export_sample_exists: with export.previews on, the export task declares
    export/<sample>/previews/ only on the route per FOV with assemble on, and after the command runs every declared
    output exists, on each route with the assemble part on and off; the folder holds each FOV's figures exactly when
    declared."""
    export = (dict(AFTER_FUSION, assemble=assemble, **({} if assemble else {"segment": False}))
              if route == "after_fusion" else {"route": "per_fov", "fuse": False, "assemble": assemble})
    if route == "after_fusion" and not assemble:   # the saved sample segmentation it reuses, from a first export
        folder, first = scene_export(tmp_path, AFTER_FUSION)
        assert T.run(first, f"export/{SCENE_SAMPLE}") == 0
        segmentation = truth_config(folder, export_grid(folder))
        runs = [{"name": run.name, "target": run.target, "method": "import", "path": run.method.path,
                 "relabel": False} for run in segmentation.runs]
        export = dict(export, previews=True, segmentation={"cells": "cell", "nuclei": "nucleus", "runs": runs})
        manifest, _ = plan_sample(tmp_path, SCENE_SAMPLE, list(folder.scene.ids), export=export,
                                  assembly={"assignment": "default", "table_format": "csv"}, stain=False,
                                  codebook_table=codebook().table, voxel_size_xy=0.5, voxel_size_z=1.0,
                                   sample_id=SCENE_SAMPLE)
        T.succeed(manifest, "fov", "segment", "assign", "stitch")
    else:
        folder, manifest = scene_export(tmp_path, export)
    task = T.task(manifest, "export", sample=SCENE_SAMPLE)
    declared = [T.output_root({"root_output_path": str(tmp_path / "output"), "dataset_id": "data",
                               "output_id": "output"}) / o for o in task["outputs"]]
    previews = [o for o in task["outputs"] if o.endswith("previews/")]
    assert previews == ([f"export/{SCENE_SAMPLE}/previews/"] if (route, assemble) == ("per_fov", True) else [])
    assert T.run(manifest, f"export/{SCENE_SAMPLE}") == 0
    assert [p for p in declared if not p.exists()] == []
    drawn = {p.relative_to(folder.export_dir).as_posix() for p in folder.export_dir.rglob("*.png")}
    expected = {f"previews/{fov}/{name}" for fov in folder.scene.ids
                for name in ("assignment.png", "assignment_single_layer.png")}
    assert drawn == (expected if previews else set())
    check_export(folder.export_dir)


@needs_anndata
@needs_library
def test_the_declared_preview_folder_exists_when_the_assemble_part_cannot_run(tmp_path, monkeypatch):
    """From test_wf2_the_declared_preview_folder_exists_when_the_assemble_part_cannot_run: when the folder is declared
    but the assemble part is neither written nor reused (here not_run, as with an extra missing), the export command
    still creates it, empty, and draws no figure."""
    import starfinder.workflow._previews as previews
    folder, manifest = scene_export(tmp_path, {"route": "per_fov", "fuse": False})
    task = T.task(manifest, "export", sample=SCENE_SAMPLE)
    assert f"export/{SCENE_SAMPLE}/previews/" in task["outputs"]

    def not_run(self, sample, *, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "export.json").write_text("{}")
        return SimpleNamespace(parts={"assemble": "not_run"}, folder=Path(output_dir))

    monkeypatch.setattr(Dataset, "export", not_run)
    monkeypatch.setattr(previews, "ownership_figures", lambda *args: pytest.fail("no assembly to draw"))
    T.call(manifest, f"export/{SCENE_SAMPLE}")
    assert (folder.export_dir / "export.json").is_file()
    assert list((folder.export_dir / "previews").iterdir()) == []


# --- the assign and export previews with ownership (from test_workflow_assembly_figures.py) ------------------------

FIGURES = {"3d": ["assignment.png", "assignment_single_layer.png"], "2d": ["assignment.png"]}


def build_route(root, dimensions, *, assignment="default", previews=False, output_id="out"):
    """The sample of test_workflow_assembly_golden.py with the reference grids (unit spacing, zero pixels) and the
    placement of assembly_golden, planned to its export with assignment.previews and export.previews ``previews``;
    the segment tasks import its label images and the assign tasks run by their commands (assignment null: no
    segment or assign task). Returns (output root, manifest, configuration)."""
    _, _, out = build_sample(root, dimensions, legacy=False)
    for fov in GOLDEN_IDS:
        save_volume(np.zeros(TILE_ZYX, np.uint8), out / "fovs" / fov / "reference.tif",
                    metadata=unit_grid(TILE_ZYX).metadata)
    save_placement(golden_placement(), out / "stitching" / "sample1")
    cells = assignment is not None
    labels = str(out / "images" / "stardist_segmentation" / "{fov_id}.tif")
    if cells and dimensions == "2d":
        labels_run = {"projection": True}
    else:
        labels_run = {}
    manifest, config = plan_sample(root, "sample1", list(GOLDEN_IDS),
                                   export={"route": "per_fov", "fuse": False, "previews": previews},
                                   assembly={"assignment": assignment}, cells=cells, labels=labels, previews=previews,
                                   output_id=output_id, stain=False, codebook_table=golden_codebook().table,
                                   voxel_size_xy=1.0, voxel_size_z=1.0)
    assert T.output_root(config) == out
    if labels_run:
        # A YX label image is a plane on the Z projection of the grid (the segment run's projection).
        config["segmentation"]["runs"][0].update(labels_run)
        manifest, _ = T.plan(root, f"{output_id}-plane", config)
    T.succeed(manifest, "fov", "stitch")
    if cells:
        T.run_kinds(manifest, "segment", "assign")
    return out, manifest, config


def snapshot(folder):
    """The export folder's files but the ownership previews (previews/, which export.json does not list), its record
    without its code version and the SHA-256 of its tables."""
    from starfinder._records import file_sha256
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                   if p.is_file() and not p.relative_to(folder).as_posix().startswith("previews/"))

    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k != "software"}
        return [stable(v) for v in value] if isinstance(value, list) else value
    record = stable(json.loads((folder / "export.json").read_text()))
    return files, record, {f: file_sha256(folder / f) for f in files if f.startswith("tables/")}


def check_figure(figure, result, record, single_layer):
    """Panels 1 and 2 carry the FOV's region line, grey points exactly at the excluded observations they draw,
    and the stated counts (test_workflow_assembly_figures.py's check)."""
    cells, mols = result.cells, result.molecules
    territories = result.territories if result.territories is not None else result.cell_labels
    layer = territories.shape[0] // 2 if single_layer else None
    present = set(np.unique(territories[layer]).tolist()) - {0} if single_layer else set(cells.cell_id.astype(int))
    on = (mols.assignment_status != "outside_grid").to_numpy(bool)
    if single_layer:
        on &= mols.voxel_z.eq(layer).fillna(False).to_numpy(bool)
    centre = {int(c): (float(x), float(y)) for c, x, y in zip(cells.cell_id, cells.centroid_x, cells.centroid_y)}
    position = {s: (float(x), float(y)) for s, x, y in zip(mols.spot_id, mols.x, mols.y)}
    for ax, drawn, gone, where, what in ((figure.axes[0], present, record.excluded_cells, centre, "cells"),
                                         (figure.axes[1], set(mols.spot_id[on]), record.excluded_molecules,
                                          position, "molecules")):
        check_region(ax, record.region)
        grey = drawn & gone
        got = scatters(ax)
        assert points(got[EXCLUDED]) == sorted(where[i] for i in grey) if grey else EXCLUDED not in got
        if grey:
            assert all(tuple(c) == GREY for c in got[EXCLUDED].get_facecolors())
        retained, total = stated(ax, what)
        assert (retained, total) == (len(drawn) - len(grey), len(drawn))
    return len(record.excluded_cells) + len(record.excluded_molecules)


def saved_figures(monkeypatch):
    """Record (final path, figure) of every figure saved; the writer saves to a temporary sibling first (OP6)."""
    saved = []
    real = matplotlib.figure.Figure.savefig

    def spy(figure, path, *args, **kwargs):
        path = Path(path)
        name = path.name.split(".tmp.", 1)[1] if ".tmp." in path.name else path.name
        saved.append((str(path.with_name(name)), figure))
        return real(figure, path, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    return saved


@needs_anndata
@pytest.mark.parametrize("dimensions", [pytest.param("3d", marks=pytest.mark.slow), "2d"])
def test_the_export_draws_the_figures_with_ownership(tmp_path, monkeypatch, dimensions):
    """From test_export_sample_draws_the_figures_with_ownership: with assignment.previews and export.previews on, the
    assign task's figures in fovs/{fov}/previews/, the export task's (with the ownership) in
    export/sample1/previews/{fov}/; one writer each, and neither file is overwritten; the export's files and record
    are those of the same export without previews."""
    out, manifest, _ = build_route(tmp_path, dimensions, previews=True)
    before = {fov: {p.name: p.read_bytes() for p in sorted((out / "fovs" / fov / "previews").glob("*.png"))}
              for fov in GOLDEN_IDS}
    assert all(list(figures) == sorted(FIGURES[dimensions]) for figures in before.values())
    assert not (out / "expr").exists()
    # The export without the figures, then with them, into the same folder.
    import starfinder.workflow._previews as previews
    monkeypatch.setattr(previews, "ownership_figures", lambda *args: [])
    T.call(manifest, "export/sample1")
    monkeypatch.undo()
    folder = out / "export" / "sample1"
    plain = snapshot(folder)
    saved = saved_figures(monkeypatch)
    assert T.run(manifest, "export/sample1") == 0
    monkeypatch.undo()
    assert snapshot(folder) == plain
    check_export(folder)
    assert "assignment.png" not in json.dumps(json.loads((folder / "export.json").read_text()))
    assert "export/sample1/previews/" in T.task(manifest, "export", sample="sample1")["outputs"]
    assembly, placement = load_assembly(folder), load_placement(out / "stitching" / "sample1")
    expected = [str(folder / "previews" / fov / name) for fov in GOLDEN_IDS for name in FIGURES[dimensions]]
    assert [path for path, _ in saved] == expected
    for path, figure in saved:
        fov = path.split("/")[-2]
        result = read_assignment(out / "fovs" / fov / "assignment" / "default", "default")
        record = fov_ownership(assembly, fov, placement=placement)
        assert record.region.any() and not record.region.all()
        assert check_figure(figure, result, record, path.endswith("single_layer.png")) > 0
    for fov in GOLDEN_IDS:
        local = {p.name: p.read_bytes() for p in sorted((out / "fovs" / fov / "previews").glob("*.png"))}
        assert local == before[fov]                                            # not overwritten by the export
        after = {p.name: p.read_bytes() for p in sorted((folder / "previews" / fov).glob("*.png"))}
        assert list(after) == list(before[fov])
        assert all(after[name] != before[fov][name] for name in after), fov   # every FOV has an excluded cell
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png")) == sorted(
        [f"fovs/{fov}/previews/{name}" for fov in GOLDEN_IDS for name in FIGURES[dimensions]]
        + [f"export/sample1/previews/{fov}/{name}" for fov in GOLDEN_IDS for name in FIGURES[dimensions]])


@needs_anndata
def test_by_default_the_route_per_fov_writes_no_preview(tmp_path, monkeypatch):
    """From test_by_default_the_saved_assignment_route_writes_no_preview: without assignment.previews and
    export.previews (default false), neither the assign task nor the export task draws a figure; the route writes the
    saved folders and the export, and no expr/ file."""
    out, manifest, config = build_route(tmp_path, "3d")
    assert config["assignment"]["previews"] is False and config["export"]["previews"] is False
    assert not [o for t in T.tasks(manifest, "assign", "export") for o in t["outputs"] if "previews" in o]
    saved = saved_figures(monkeypatch)
    assert T.run(manifest, "export/sample1") == 0
    assert saved == []
    assert json.loads((out / "export" / "sample1" / "export.json").read_text())["parts"]["assemble"] == "written"
    assert list(out.rglob("*.png")) == [] and not (out / "expr").exists()
    assert not (out / "export" / "sample1" / "previews").exists()


@needs_anndata
def test_with_assignment_null_no_figure_is_written(tmp_path, monkeypatch):
    """From test_with_assignment_null_no_figure_is_written_or_changed: molecules only: no assign task, the export
    declares no preview folder, also with export.previews on, and draws nothing."""
    out, manifest, _ = build_route(tmp_path, "3d", assignment=None, previews=True)
    assert T.tasks(manifest, "assign") == []
    assert not [o for o in T.task(manifest, "export", sample="sample1")["outputs"] if "previews" in o]
    saved = saved_figures(monkeypatch)
    assert T.run(manifest, "export/sample1") == 0
    assert json.loads((out / "export" / "sample1" / "export.json").read_text())["assembly"]["mode"] == "molecules_only"
    assert saved == [] and list(out.rglob("*.png")) == []
