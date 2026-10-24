"""The workflow draws the §2.9 assignment figures with the FOV ownership region (W-330): export_sample.

The sample of ``test_workflow_assembly_golden.py`` (three FOVs of 8×48×64 voxels, ZYX and YX label
images) with an assembly block: the reference grids ``fovs/{fovID}/reference.tif`` (unit spacing)
and the placement of ``assembly_golden`` saved in ``stitching/sample1/``. ``reads_assignment`` and
``export_sample`` run through stub ``snakemake`` objects, as that test runs the adapter; the figures
``export_sample`` saves are inspected before saving, by their artists.
"""
import importlib.util
import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import starfinder.dataset.workflow as workflow  # noqa: E402
from starfinder import _snakemake  # noqa: E402
from starfinder.assembly import fov_ownership, load_assembly  # noqa: E402
from starfinder.assignment._persist import read_assignment  # noqa: E402
from starfinder.io import save_volume  # noqa: E402
from starfinder.export import check_export  # noqa: E402
from starfinder.stitching import load_placement, save_placement  # noqa: E402

from .assembly_fixtures import GOLDEN_IDS, golden_placement, unit_grid  # noqa: E402
from .fusion_fixtures import golden_codebook  # noqa: E402
from .test_stitching_assembly_figure import EXCLUDED, GREY, check_region, points, scatters, stated  # noqa: E402
from .test_workflow_assembly_golden import TILE_ZYX, _config, build_sample, run_adapter  # noqa: E402

pytestmark = [pytest.mark.workflow, pytest.mark.stitching]

needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
MULTIVIEW = {"method": "multiview_stitcher", "input": "reference_stain",
             "positions": {"source": "regular_grid",
                           "regular_grid": {"columns": 2, "rows": 2, "order": "column_major_down_right",
                                            "overlap": 0.25}}}
FIGURES = {"3d": ["assignment.png", "assignment_single_layer.png"], "2d": ["assignment.png"]}


def assembly_config(root, assignment="default"):
    return {**_config(root, legacy=False), "n_rounds": 3, "ref_round": "round1", "seq_channel_order": ["ch00", "ch01", "ch02",
                                                                                          "ch03"],
            "stitching": MULTIVIEW, "assembly": {"assignment": assignment},
            "export": {"route": "per_fov", "fuse": False}}


def build(root, dimensions):
    """build_sample with the reference grids, the codebook of the molecule path and the saved placement."""
    _, _, out = build_sample(root, dimensions, legacy=False)
    codebook = root / "input" / "data" / "sample" / "genes.csv"
    golden_codebook().table[["entry_id", "gene_id", "color_sequence"]].to_csv(codebook, index=False)
    for fov in GOLDEN_IDS:
        save_volume(np.zeros(TILE_ZYX, np.uint8), out / "fovs" / fov / "reference.tif",
                    metadata=unit_grid(TILE_ZYX).metadata)
    save_placement(golden_placement(), out / "stitching" / "sample1")
    return out


def reads_assignment(root, out, fov, config):
    """reads_assignment with the assembly block for one FOV; returns the bytes of each figure it wrote."""
    inputs = [root / "input" / "data" / "sample" / "fovs.csv", out / "images" / "DAPI" / f"{fov}.tif",
              out / "images" / "stardist_segmentation" / f"{fov}.tif", out / "fovs" / fov / "molecules.csv",
              root / "input" / "data" / "sample" / "genes.csv", out / "fovs" / fov / "reference.tif"]
    outputs = [out / "expr" / fov / "raw.h5ad", out / "expr" / fov / "reads_assignment.csv"]
    name = config["assembly"]["assignment"]
    if name is not None:
        outputs.append(out / "fovs" / fov / "assignment" / name / "assignment.json")
    _snakemake._run_reads_assignment(SimpleNamespace(input=[str(p) for p in inputs], output=[str(p) for p in outputs],
                                                   config=config, wildcards=SimpleNamespace(fovID=fov)))
    return {p.name: p.read_bytes() for p in sorted((out / "expr" / fov).glob("*.png"))}


def export_sample(out, config, monkeypatch=None):
    """export_sample through a stub; with monkeypatch, returns the (path, figure) of every saved figure."""
    saved = []
    if monkeypatch is not None:
        real = matplotlib.figure.Figure.savefig

        def spy(figure, path, *args, **kwargs):
            saved.append((str(path), figure))
            return real(figure, path, *args, **kwargs)

        monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    # The rule's inputs (export.smk): the placement, then the per-FOV outputs of reads_assignment.
    inputs, output = _snakemake._export_sample_files(config, out, GOLDEN_IDS, "sample1")
    _snakemake._run_export(SimpleNamespace(config=config, input=inputs, output=[output],
                                         wildcards=SimpleNamespace(sample="sample1")))
    if monkeypatch is not None:
        monkeypatch.undo()
    return saved


# The recorded code version (assembly.software); with fusion off this export records no time.
VOLATILE = {"software"}


def stable(value):
    """export.json without its recorded code version (the keys VOLATILE names, at any depth)."""
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def snapshot(folder):
    from starfinder._records import file_sha256
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    record = stable(json.loads((folder / "export.json").read_text()))
    return files, record, {f: file_sha256(folder / f) for f in files if f.startswith("tables/")}


def check_figure(figure, result, record, single_layer):
    """Panels 1 and 2 carry the FOV's region line, grey points exactly at the excluded observations they draw,
    and the stated counts."""
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


@needs_anndata
@pytest.mark.parametrize("dimensions", [pytest.param("3d", marks=pytest.mark.slow), "2d"])
def test_export_sample_draws_the_figures_with_ownership(tmp_path, monkeypatch, dimensions):
    out = build(tmp_path, dimensions)
    config = assembly_config(tmp_path)
    before = {fov: reads_assignment(tmp_path, out, fov, config) for fov in GOLDEN_IDS}
    assert all(list(figures) == sorted(FIGURES[dimensions]) for figures in before.values())
    # The export without the figures (the code before this change), then with them, into the same folder.
    monkeypatch.setattr(_snakemake, "_write_ownership_figures", lambda *args: None)
    export_sample(out, config)
    monkeypatch.undo()
    plain = snapshot(out / "export" / "sample1")
    assert {fov: {p.name: p.read_bytes() for p in (out / "expr" / fov).glob("*.png")} for fov in GOLDEN_IDS} \
        == before
    saved = export_sample(out, config, monkeypatch)
    folder = out / "export" / "sample1"
    assert snapshot(folder) == plain
    check_export(folder)
    assert "assignment.png" not in json.dumps(json.loads((folder / "export.json").read_text()))
    # Every assembled FOV's figures, saved into expr/{fovID}/, each with the FOV's region and excluded observations.
    assembly, placement = load_assembly(folder), load_placement(out / "stitching" / "sample1")
    expected = [str(out / "expr" / fov / name) for fov in GOLDEN_IDS for name in FIGURES[dimensions]]
    assert [path for path, _ in saved] == expected
    for path, figure in saved:
        fov = path.split("/")[-2]
        result = read_assignment(out / "fovs" / fov / "assignment" / "default", "default")
        record = fov_ownership(assembly, fov, placement=placement)
        assert record.region.any() and not record.region.all()
        assert check_figure(figure, result, record, path.endswith("single_layer.png")) > 0
    for fov in GOLDEN_IDS:
        after = {p.name: p.read_bytes() for p in sorted((out / "expr" / fov).glob("*.png"))}
        assert list(after) == list(before[fov])
        assert all(after[name] != before[fov][name] for name in after), fov   # every FOV has an excluded cell


@needs_anndata
def test_with_assignment_null_no_figure_is_written_or_changed(tmp_path, monkeypatch):
    out = build(tmp_path, "3d")
    config = assembly_config(tmp_path, assignment=None)
    before = {fov: reads_assignment(tmp_path, out, fov, config) for fov in GOLDEN_IDS}
    saved = export_sample(out, config, monkeypatch)
    assert json.loads((out / "export" / "sample1" / "export.json").read_text())["assembly"]["mode"] == "molecules_only"
    assert saved == []
    assert {fov: {p.name: p.read_bytes() for p in (out / "expr" / fov).glob("*.png")} for fov in GOLDEN_IDS} \
        == before


@needs_anndata
def test_without_an_assembly_block_reads_assignment_draws_no_ownership(tmp_path, monkeypatch):
    """The legacy configuration: no export_sample rule (the Snakefile includes export.smk with the block only),
    and reads_assignment's figures carry no region line, no grey point and no stated count."""
    snakefile = (workflow.Path(__file__).resolve().parents[3] / "workflow" / "Snakefile").read_text()
    assert 'if ASSEMBLY_BLOCK:\n    include: "rules/export.smk"' in snakefile
    _, _, out = build_sample(tmp_path, "3d")
    saved = []
    real = matplotlib.figure.Figure.savefig
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig",
                        lambda figure, path, *a, **k: saved.append(figure) or real(figure, path, *a, **k))
    run_adapter(tmp_path, out, "Position001")
    assert len(saved) == 2
    for figure in saved:
        for ax in figure.axes[:2]:
            assert all(c.get_gid() != "ownership_region" for c in ax.collections)
            assert EXCLUDED not in scatters(ax) and ax.get_xlabel() == ""
