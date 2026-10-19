"""Generator checks of the appearance scene on the fixtures their rows name and no earlier module ran (W-520).

docs/synthetic-real-appearance-validation.md, "Generator checks". GA13 (κ and the rim background) and GA17 (types and
molecules, GN8 to GN12) on ``app_routine``, the three routine appearance presets as generated; GA20's ``app_streams``
copy with type 2's abundance row changed (the molecules of the other types and the ambient ones unchanged, GN21);
GA21 on the culture copies of ``app_plane`` and of ``app_morph`` with MT1's drawn round; GA22's subprocess that, beside
building ``cf_banks``, generates ``app_routine`` under an ``open`` audit hook; GA15 on ``app_plane`` (GA19 names it);
GA2's fifteen new names together, exported and on the API pages. Each check is the one its row states,
with the row's tolerance; the helpers are those of the stain and contract modules, so the arithmetic is theirs.
"""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from starfinder.report._measurements import mecr
from starfinder import synthetic
from starfinder.synthetic import OutputScaleConfig, generate_appearance_scene
from starfinder.synthetic._appearance import _point_evaluator

from . import cellular_appearance_oracle as oracle
from .cellular_appearance_scenes import ROUTINE_PRESETS, app_routine, appearance_config
from .cellular_scenes import tiny
from .test_synthetic_appearance_contract import test_ga21_layout_versions_nulls_tables_and_hashes as check_ga21
from .test_synthetic_appearance_stains import background_of, frame_of, generated, growth_of, images, placing_cell

pytestmark = [pytest.mark.synthetic]

TOLERANCE = 1e-9
slow = pytest.mark.slow
DOCS = Path(__file__).resolve().parents[3] / "docs"
# GA2's new names ("Opt-in").
NAMES = ("generate_appearance_scene", "AppearanceScene", "AppearanceSceneConfig", "BankReference", "AugmentationConfig",
         "LayoutConfig", "CompositionConfig", "build_appearance_bank", "closed_form_bank", "load_appearance_bank",
         "save_appearance_scene", "appearance_scene_preset", "APPEARANCE_PRESETS", "APPEARANCE_PRESET_VERSION",
         "APPEARANCE_TRUTH_VERSION")


# ---------------------------------------------------------------- GA2: the new names

@pytest.mark.contract
def test_ga2_every_new_name_is_exported_and_on_the_api_pages():
    pages = {page: (DOCS / "api" / page).read_text(encoding="utf-8")
             for page in ("synthetic.rst", "inventory.rst", "python-index.rst")}
    assert len(NAMES) == 15
    for name in NAMES:
        assert name in synthetic.__all__ and hasattr(synthetic, name), name
        assert re.search(rf"^   {name}$", pages["synthetic.rst"], re.M), name
        for page in ("inventory.rst", "python-index.rst"):
            assert f":py:obj:`starfinder.synthetic.{name}`" in pages[page], (page, name)
    assert synthetic.APPEARANCE_PRESET_VERSION == "appearance-presets-v1" and synthetic.APPEARANCE_TRUTH_VERSION == "1"


# ---------------------------------------------------------------- GA15 on app_plane (GA19)

@pytest.mark.validation
@pytest.mark.parametrize("copy", ["tissue2d-plane", "culture-plane"])
def test_ga15_one_plane_8bit_images_are_the_cast(copy):
    """app_plane with the 8-bit cast against its float64 rendering J: each 8-bit image (the morphology round's too)
    equals clip(rint(g · J), 0, 255); the placements are equal; no noise stream is drawn (OB1)."""
    book, float_config, reference, _ = generated(copy)
    stains = tuple(replace(s, output=OutputScaleConfig(dtype="uint8")) for s in float_config.context.stains)
    config = replace(float_config, context=replace(float_config.context, stains=stains))
    scene = generate_appearance_scene(book, config=config)
    g = scene.record["plateau"]
    pd.testing.assert_frame_equal(scene.placements, reference.placements, check_exact=True)
    expected_images = {(st.name, st.round, f): j for st, f, _, j in images(reference, float_config)}
    voxels = 0
    for stain, fov, _, image in images(scene, config):
        assert image.dtype == np.uint8 and image.shape[0] == 1
        expected = np.clip(np.rint(g * expected_images[(stain.name, stain.round, fov)]), 0, 255).astype(np.uint8)
        assert np.array_equal(image, expected), (fov, stain.name, stain.round)
        voxels += image.size
    drawn = {d[4] for d in scene.context.record["stream_scheme"]["streams"]}
    drawn |= {d[4] for d in scene.context.sample.record["stream_scheme"]["streams"]}
    assert not any("noise" in component for component in drawn)
    print(f"GA15 {copy}: {voxels} 8-bit voxels of one plane equal clip(rint(g · J), 0, 255) with g = {g}")


# ---------------------------------------------------------------- GA13 on app_routine
# Each generates its routine preset (5 s or more alone with its closed-form bank), so each is `slow`.

@slow
@pytest.mark.validation
@pytest.mark.parametrize("name", ROUTINE_PRESETS)
def test_ga13_kappa_and_the_rim_background_on_the_routine_presets(name):
    """As GA13 on app_tiny: κ recomputed by the rule of ``proto.py::place_all`` over the rims of the placed masks;
    where κ > 0 the rim mean of B + κH equals b̄ (1e-9 relative), where κ = 0 the rim mean of B is at least b̄
    (reported); each object's blurred copy against scipy's Gaussian filter of its excess (1e-9 of its maximum)."""
    _, config, scene, bank, _ = app_routine(name)
    s, _, origin, shape = frame_of(scene)
    kind = oracle.kind_of(bank)
    points = oracle.grid_points(origin, s, shape)
    objects = oracle.frames(bank, scene.placements)
    N, C, boundary, _ = oracle.labels(points.reshape(-1, 3), objects, kind, growth_of(scene, bank))
    assert not boundary.any()
    rim = oracle.rim(((C if kind == "cell" else N) > 0).reshape(shape), kind == "column")
    evaluator = _point_evaluator(scene, bank)
    reports = []
    for stain_name in dict.fromkeys(stain.name for stain in config.context.stains):
        c = bank.channels.index(stain_name)
        terms = oracle.object_terms(bank, scene.placements, c, s)
        B = background_of(scene, bank, stain_name, points).reshape(shape)
        H = oracle.compose(points, terms, 0.0, B)["haze"].reshape(shape)
        local = bank.entries.set_index("entry_id")[f"local_background_{stain_name}"]
        mean_b = float(np.mean([local[e] for e in scene.placements.entry_id]))
        rim_b, rim_h = float(B[rim].mean()), float(H[rim].mean())
        kappa = max(mean_b - rim_b, 0.0) / rim_h if rim_h > 0 else 0.0
        recorded = scene.record["kappa"][stain_name]
        assert abs(recorded - kappa) <= TOLERANCE * max(1.0, abs(kappa)), (stain_name, recorded, kappa)
        level = float((B + recorded * H)[rim].mean())
        if recorded > 0:
            assert abs(level - mean_b) <= TOLERANCE * mean_b
            case = f"κ {recorded:.6g} > 0: rim mean of B + κH {level:.12g} against b̄ {mean_b:.12g}"
        else:
            assert rim_b >= mean_b
            case = f"κ = 0 (reported): rim mean of B {rim_b:.6g} ≥ b̄ {mean_b:.6g}"
        copy_error = 0.0
        for i, t in enumerate(terms):
            array, pad = evaluator.composer.haze(i, c)
            core = array[tuple(slice(p, p + n) for p, n in zip(pad, t.X.shape))]
            expected = ndi.gaussian_filter(t.X, t.haze_sigma, mode="constant", cval=0.0)
            copy_error = max(copy_error, float(np.abs(core - expected).max()) / max(float(np.abs(expected).max()),
                                                                                   1e-300))
        assert copy_error <= TOLERANCE
        reports.append(f"{stain_name}: {case}; blurred copies {copy_error:.2e} of their maximum")
    print(f"GA13 {name} ({int(rim.sum())} rim voxels): " + "; ".join(reports))


# ---------------------------------------------------------------- GA17 on app_routine

@slow
@pytest.mark.validation
@pytest.mark.parametrize("name", ROUTINE_PRESETS)
def test_ga17_types_and_molecules_on_the_routine_presets(name):
    """GN8 to GN12 with C and N of the appearance on a routine preset (its own cell types and molecules): one
    ordinary cell per placed object; each cell's region the Voronoi region of its anchor (anchors within 1e-9 µm of a
    region boundary reported, expected 0); each abundance row summing to 1 (1e-9); each cell molecule of a gene its
    type's row weights and no marker gene among the ambient molecules (GN8); cell_gene_counts against a recount and
    the marker answers against MECR recomputed (GN9); the realized gene abundance of each type (GN10) and Σ n_k against
    Σ μ_k (GN11) within 4 SE; the nuclear fraction within 4 SE, each cell molecule in the cell that placed it and its
    compartment the oracle's N and C at its position (GN12)."""
    _, config, scene, bank, _ = app_routine(name)
    ctx = scene.context
    sample, context = ctx.sample, ctx.cell_context
    types = config.context.types
    assert context.cell_id.tolist() == sample.cells.cell_id.tolist() == scene.placements.object_id.tolist()
    assert (sample.cells.kind == "ordinary").all() and (sample.cells.n_nuclei == 1).all()
    # GN8: regions and types.
    regions = ctx.regions
    centres = regions[["centre_z", "centre_y", "centre_x"]].to_numpy()
    anchors = sample.cells[["anchor_z", "anchor_y", "anchor_x"]].to_numpy()
    squared = ((anchors[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
    region = np.argmin(squared, axis=1)
    np.testing.assert_array_equal(context.region.astype(int).to_numpy(), regions.region_id.to_numpy()[region])
    order = np.sort(squared, axis=1)
    near_boundary = 0
    if len(centres) > 1:
        gap = np.linalg.norm(centres[region] - centres[np.argsort(squared, axis=1)[:, 1]], axis=1)
        near_boundary = int(((order[:, 1] - order[:, 0]) / (2 * gap) <= 1e-9).sum())
    assert set(context.type) <= set(types.types)
    table = ctx.type_abundances
    for type_name, rows in table.groupby("type"):
        assert abs(rows.weight.sum() - 1) <= 1e-9, type_name
    markers = set(table.gene_id[table.role == "marker"])
    weight = {(r.type, r.gene_id): r.weight for r in table.itertuples()}
    type_of = dict(zip(context.cell_id.astype(int), context.type))
    molecules = sample.molecules
    cell_molecules = molecules[molecules.population == "cell"]
    for row in cell_molecules.itertuples():
        assert weight[type_of[placing_cell(row.amplicon_id)], row.gene_id] > 0, row.amplicon_id
    assert not set(molecules.gene_id[molecules.population == "ambient"]) & markers
    # GN9: counts per cell and gene, and the marker answers over the sample cells.
    owned = molecules[molecules.true_cell_id.notna()]
    expected = {}
    for cell_id, gene, compartment in zip(owned.true_cell_id.astype(int), owned.gene_id, owned.compartment):
        count, nuclear = expected.get((cell_id, gene), (0, 0))
        expected[cell_id, gene] = (count + 1, nuclear + (compartment == "nucleus"))
    counts_table = ctx.cell_gene_counts
    assert {(int(r.cell_id), r.gene_id): (int(r.count), int(r.nuclear_count))
            for r in counts_table.itertuples()} == expected
    sums = counts_table.groupby("cell_id")["count"].sum()
    assert sample.cells.n_molecules.tolist() == [int(sums.get(k, 0)) for k in sample.cells.cell_id]
    counted = sample.cells.cell_id[sample.cells.in_sample].astype(int).tolist()
    counts = {(int(r.cell_id), r.gene_id): int(r.count) for r in counts_table.itertuples()}
    for row in ctx.marker_pairs.itertuples():
        values = mecr([counts.get((c, row.gene_a), 0) for c in counted],
                      [counts.get((c, row.gene_b), 0) for c in counted], 1)
        assert (row.cells_a, row.cells_b, row.both, row.either) == (values["cells_a"], values["cells_b"],
                                                                     values["both"], values["either"])
        if row.either:
            assert abs(row.mecr - values["mecr"]) <= 1e-9  # the same ratio of the same integers; "exact" as 1e-9
        else:
            assert pd.isna(row.mecr)
    # GN10: the realized gene abundance of each type within 4 SE.
    kinds = cell_molecules.amplicon_id.map(lambda a: type_of[placing_cell(a)])
    compared = 0
    for type_name in types.types:
        chosen = cell_molecules.gene_id[kinds == type_name]
        n = len(chosen)
        if not n:
            continue
        realized = chosen.value_counts()
        for row in table[table.type == type_name].itertuples():
            count = int(realized.get(row.gene_id, 0))
            if row.weight == 0:
                assert count == 0
                continue
            assert abs(count / n - row.weight) <= 4 * math.sqrt(row.weight * (1 - row.weight) / n)
            compared += 1
    # GN11: Σ n_k against Σ μ_k, one lognormal mean per cell (M2).
    placed = cell_molecules.amplicon_id.map(placing_cell).value_counts()
    total, means = int(placed.sum()), float(context.molecule_mean.sum())
    assert abs(total - means) <= 4 * math.sqrt(means)
    assert context.molecule_mean.nunique() == len(context)
    # GN12: the nuclear fraction within 4 SE; each cell molecule's cell and compartment against the oracle.
    fraction = config.context.molecules.nuclear_fraction
    nuclear = (cell_molecules.compartment == "nucleus").to_numpy()
    share = float(nuclear.mean())
    se = math.sqrt(fraction * (1 - fraction) / len(nuclear))
    assert abs(share - fraction) <= 4 * se
    np.testing.assert_array_equal(cell_molecules.true_cell_id.astype(int).to_numpy(),
                                  cell_molecules.amplicon_id.map(placing_cell).to_numpy())
    sigma = molecules[["sample_z", "sample_y", "sample_x"]].to_numpy()
    N, C, near, _ = oracle.labels(sigma, oracle.frames(bank, scene.placements), oracle.kind_of(bank),
                                  growth_of(scene, bank))
    np.testing.assert_array_equal(molecules.true_cell_id.fillna(0).astype(int).to_numpy()[~near], C[~near])
    assert ((molecules.compartment == "nucleus").to_numpy() == ((N > 0) & (N == C)))[~near].all()
    print(f"GA17 {name}: {len(context)} cells of types {context.type.value_counts().to_dict()}, {len(cell_molecules)} "
          f"cell molecules, {compared} gene shares within 4 SE; Σ n_k {total} against Σ μ_k {means:.1f}; nuclear "
          f"fraction {share:.4f} against {fraction} (4 SE {4 * se:.4f}); molecule positions at a boundary "
          f"{int(near.sum())} (expected 0); anchors within 1e-9 µm of a region boundary {near_boundary} (expected 0)")
    assert not near.any() and near_boundary == 0


# ---------------------------------------------------------------- GA20: type 2's abundance row (app_streams)

def abundance_copy(copy):
    """The ``<copy>-types-light`` configuration and its copy with type 2's row changed as GN21's ``types-abundance``
    (its markers lighter, the shared genes heavier)."""
    config = appearance_config(f"{copy}-types-light")
    types = config.context.types
    rows = list(types.abundances)
    rows[1] = tuple(0.05 if w == 0.15 else 0.1125 if w == 0.0875 else w for w in rows[1])
    changed = replace(config, context=replace(config.context, types=replace(types, abundances=tuple(rows))))
    return config, changed


@pytest.mark.validation
@pytest.mark.parametrize("copy", ["tissue2d", pytest.param("dense3d", marks=slow)])
def test_ga20_a_type_abundance_changes_only_that_types_genes(copy):
    """GN21 on the appearance scene: with type 2's row changed, no ambient molecule and no molecule of another type
    changes gene, position, compartment or cell, and the placements are equal."""
    book, _ = tiny()
    config, changed = abundance_copy(copy)
    first = generate_appearance_scene(book, config=config)
    second = generate_appearance_scene(book, config=changed)
    pd.testing.assert_frame_equal(first.placements, second.placements, check_exact=True)
    a, b = first.context.sample.molecules, second.context.sample.molecules
    columns = ["amplicon_id", "population", "sample_z", "sample_y", "sample_x", "compartment", "true_cell_id"]
    pd.testing.assert_frame_equal(a[columns], b[columns], check_exact=True)
    context = first.context.cell_context
    type_of = dict(zip(context.cell_id.astype(int), context.type))
    of_type_2 = np.array([population == "cell" and type_of[placing_cell(i)] == "type-2"
                          for i, population in zip(a.amplicon_id, a.population)], dtype=bool)
    differs = (a.gene_id != b.gene_id).to_numpy()
    ambient = (a.population == "ambient").to_numpy()
    assert not differs[ambient].any() and not differs[~of_type_2].any()
    assert of_type_2.any() and differs[of_type_2].any()
    other = lambda t: t[t.type != "type-2"].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(other(first.context.type_abundances), other(second.context.type_abundances),
                                  check_exact=True)
    print(f"GA20 {copy}: type 2's abundance changed: {int(differs.sum())} of {int(of_type_2.sum())} type-2 molecules "
          f"changed gene, none of the other {int((~of_type_2).sum())}, of which {int(ambient.sum())} ambient")


# ---------------------------------------------------------------- GA21 on the culture copies of app_plane and app_morph

@pytest.mark.contract
@pytest.mark.parametrize("name", ["culture-plane", "culture-morph-drawn"])
def test_ga21_layout_on_the_culture_plane_and_drawn_morphology(name, tmp_path):
    """GA21's check of its module (the AT2 layout, versions, nulls, tables and hashes, the labels rebuilt from the
    saved files and the bank) on the two copies it does not run."""
    check_ga21(name, tmp_path)


# ---------------------------------------------------------------- GA22: app_routine read no real data

_AUDIT = r"""
import json, os, sys
import starfinder.synthetic
from starfinder.synthetic import appearance_scene_preset, generate_appearance_scene, load_appearance_bank
import scipy.ndimage, skimage.filters, tifffile, pandas
opened = []
def audit(event, arguments):
    if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
        opened.append(os.fsdecode(arguments[0]))
sys.addaudithook(audit)
book, config = appearance_scene_preset(sys.argv[1])
scene = generate_appearance_scene(book, config=config)
bank = load_appearance_bank(config.bank)
sys.stdout.write(json.dumps(dict(opened=opened, inputs=bank.record["inputs"], held_out=bank.record["held_out"],
                                 preset=bank.record.get("preset"), placed=scene.record["placed"])))
"""


@slow
@pytest.mark.contract
@pytest.mark.parametrize("name", ROUTINE_PRESETS)
def test_ga22_a_routine_preset_reads_no_file_outside_the_package(name, tmp_path):
    """In a subprocess that generates the routine preset (its closed-form bank built in memory), the files opened,
    recorded by an ``open`` audit hook: none outside the test's tmp_path, the package and the Python installation;
    the bank's record names a preset and seeds, no path and no SHA-256 (CF1)."""
    import starfinder
    env = dict(os.environ, TMPDIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", _AUDIT, name], cwd=tmp_path, env=env, capture_output=True,
                            text=True, check=True)
    record = json.loads(result.stdout)
    allowed = [Path(tmp_path).resolve(), Path(starfinder.__file__).resolve().parent,
               *{Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}]
    outside = sorted({p for p in record["opened"]
                      if not any(Path(p).resolve().is_relative_to(root) for root in allowed)})
    print(f"GA22 {name}: {record['placed']} objects on the closed-form bank of {record['preset']}; files opened "
          f"{len(record['opened'])}, outside: {outside}")
    assert outside == []
    assert record["preset"] is not None and record["inputs"] and all(
        "preset" in i and "seed" in i and "path" not in i and "sha256" not in i for i in record["inputs"])
