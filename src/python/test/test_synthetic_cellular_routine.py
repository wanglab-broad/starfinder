"""Generator checks of the cellular scene on the routine presets (docs/synthetic-cellular-scene-validation.md).

The GN rows whose fixture list names ``context_routine`` beside the tiny fixtures: GN4 (labels against the test
oracle and the point evaluator, identities 1 to 5), GN5 (lobes and LB1), GN6 (no overlap; a nucleus inside its own
cell), GN8 (types and genes), GN9 (counts and the marker answers), GN10 (the realized gene abundance), GN11 (molecules
per cell, the count guard and the rendering identity), GN12 (the nuclear fraction and the compartments), GN15 (bright
regions) and GN19 (projection truth, on the 2D tissue and culture presets), each on the three routine presets
``context-dense3d-v1``, ``context-tissue2d-v1`` and ``context-culture-v1`` as generated (seed 42); and GN4 on ctx_stains
(its three territory rules and the copy whose FOV offsets differ by whole voxels). The checks are those of the tiny
fixtures' tests (test_synthetic_cellular_scene.py, test_synthetic_cellular_stains.py), written again here for a scene
whose laws come from a preset; positions whose decision lies within 1e-9 µm of a boundary are counted and reported
(expected 0), and statistical rows compare within four standard errors (4 SE) and print their values.
"""
from dataclasses import replace
from functools import cache
import math

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from starfinder.report._measurements import mecr
from starfinder.synthetic import BackgroundConfig, NoiseConfig, context_scene_preset, generate_context_scene
from starfinder.synthetic._context import _context_labels

from .cellular_checks import check_lobes
from .cellular_oracle import Oracle, check_sample_identities, grids
from .cellular_scenes import RULES, scene
from .formed_oracle import kernel_radius

pytestmark = [pytest.mark.synthetic, pytest.mark.validation]

ROUTINE = ["context-dense3d-v1", "context-tissue2d-v1", "context-culture-v1"]
MONOLAYER = ["context-tissue2d-v1", "context-culture-v1"]
# ctx_stains: the three territory rules (FOVs at fractional offsets) and the copy whose supplied jitter differs between
# FOV_001 and FOV_002 by whole voxels.
STAIN_LABELS = [f"stains-{rule}-whole" for rule in RULES] + ["stains-pair"]


@cache
def preset(name):
    """(codebook, config, ContextScene) of a routine preset as generated, once per session."""
    book, config = context_scene_preset(name)
    return book, config, generate_context_scene(book, config=config)


def placing_cell(amplicon_id):
    """The cell that placed a cell molecule (IDs c<k>-m<j>), None for an ambient one."""
    return int(amplicon_id[1:].split("-")[0]) if amplicon_id.startswith("c") else None


# ---------------------------------------------------------------- GN4: labels against the oracle and the evaluator

def check_labels(name, generated):
    oracle = Oracle(generated)
    sample = generated.sample
    plane = sample.record["sample_grid"]["shape_zyx"][0] == 1
    jitter = sample.placement[["jitter_z_vox", "jitter_y_vox", "jitter_x_vox"]].to_numpy()
    fractional = np.abs(jitter - np.round(jitter)) > 1e-3
    # The FOVs sit at fractional offsets in Z, Y and X.
    assert fractional[:, 1:].any(axis=0).all() and (plane or fractional[:, 0].any())
    assert sample.nuclei.nucleus_id.is_unique
    boundary_total = evaluator_differences = voxels = 0
    pieces = {}
    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connected
    for grid, points, labels in grids(generated):
        cell, nucleus, boundary = oracle.labels(points)
        evaluated = _context_labels(generated, points)
        for index, (target, truth) in enumerate((("cell", cell), ("nucleus", nucleus))):
            produced = labels[target].astype(np.int64)
            assert not np.any((produced != truth) & ~boundary), (name, grid, target)
            assert not np.any((evaluated[index].astype(np.int64) != truth) & ~boundary), (name, grid, target)
            evaluator_differences += int(np.count_nonzero(evaluated[index] != labels[target]))
        boundary_total += int(boundary.sum())
        voxels += boundary.size
        for n in np.unique(labels["nucleus"]):
            if n:
                count = ndimage.label(labels["nucleus"] == n, structure=structure)[1]
                if count > 1:
                    pieces.setdefault(int(n), []).append((grid, int(count)))
    near = check_sample_identities(generated, oracle)
    print(f"GN4 {name}: {voxels} voxel centres over {len(sample.positions) + 1} grids; boundary voxels "
          f"{boundary_total} (expected 0); evaluator against the generator's labels: {evaluator_differences} "
          f"differences; molecules within 1e-9 µm of a boundary {near}; nuclei whose label has several pieces "
          f"(ID: grid, pieces) {pieces or 'none'}")
    assert boundary_total == 0 and near == 0 and evaluator_differences == 0


@pytest.mark.parametrize("name", ROUTINE)
def test_gn4_labels_on_the_routine_presets(name):
    check_labels(name, preset(name)[2])


@pytest.mark.parametrize("name", STAIN_LABELS)
def test_gn4_labels_on_ctx_stains(name):
    _, _, generated = scene(name)
    if name == "stains-pair":
        # The two FOVs' offsets on the sample grid differ by whole voxels (identity 2's worked example, moved by T).
        placement = generated.sample.placement.set_index("fov_id")
        grid = generated.sample.record["sample_grid"]
        origin, spacing = np.array(grid["origin_zyx"]), np.array(grid["spacing_zyx"])
        delta = {fov: (placement.loc[fov, ["true_origin_z", "true_origin_y", "true_origin_x"]].to_numpy(float)
                       - origin) / spacing for fov in ("FOV_001", "FOV_002")}
        assert np.allclose(delta["FOV_002"] - delta["FOV_001"], np.round(delta["FOV_002"] - delta["FOV_001"]),
                           atol=1e-9, rtol=0)
    check_labels(name, generated)


# ---------------------------------------------------------------- GN5: lobes, LB1 and one connected shape

@pytest.mark.parametrize("name", ROUTINE)
def test_gn5_lobes_on_the_routine_presets(name):
    _, config, generated = preset(name)
    check_lobes(name, config, generated)


# ---------------------------------------------------------------- GN6: no overlap; a nucleus inside its own cell

@pytest.mark.parametrize("name", ROUTINE)
def test_gn6_no_overlap_on_the_routine_presets(name):
    _, config, generated = preset(name)
    oracle = Oracle(generated)
    sample = generated.sample
    owner = dict(zip(sample.nuclei.nucleus_id.astype(int), sample.nuclei.cell_id.astype(int)))
    overlap = boundary = 0
    for grid, points, labels in grids(generated):
        hit = labels["nucleus"] > 0
        # Identity 3 on every label voxel, boundary voxels included.
        np.testing.assert_array_equal(labels["cell"][hit], [owner[int(v)] for v in labels["nucleus"][hit]])
        flat = points.reshape(-1, 3)
        rho = np.stack([oracle.rho(n, flat) for n in owner])
        two = np.flatnonzero((rho <= 1).sum(axis=0) >= 2)
        if not len(two):
            continue
        site = oracle.power_nearest(flat[two])
        nucleus = oracle.site_nuclei[site]
        own = np.array([oracle.rho(int(n), flat[i][None])[0] <= 1 if n else False for i, n in zip(two, nucleus)])
        inside = np.all((flat[two] >= oracle.lo) & (flat[two] < oracle.hi), axis=1)
        expected = np.where(own & inside, nucleus, 0)
        near = oracle.labels(flat[two])[2]
        got = labels["nucleus"].reshape(-1)[two].astype(np.int64)
        assert not np.any((got != expected) & ~near), (name, grid)
        overlap += len(two)
        boundary += int(near.sum())
    # Every site centre strictly inside its own power cell in the arithmetic K1 names; each anchor gives its own cell.
    for j in range(len(oracle.centres)):
        power = oracle.label_power(oracle.centres[j])
        assert np.all(power[j] < np.delete(power, j)), j
    anchors = sample.cells[["anchor_z", "anchor_y", "anchor_x"]].to_numpy()
    cell, nucleus = _context_labels(generated, anchors)
    np.testing.assert_array_equal(cell, sample.cells.cell_id.to_numpy())
    first = sample.nuclei.groupby("cell_id").nucleus_id.min()
    for k, n in zip(sample.cells.cell_id, nucleus):
        assert n == first.get(k, 0)
    # The oracle's own arithmetic agrees except at an anchor P1 placed on its partner's power face (W-483 finding 2).
    oracle_cell, _, near = oracle.labels(anchors)
    np.testing.assert_array_equal(oracle_cell[~near], sample.cells.cell_id.to_numpy()[~near])
    print(f"GN6 {name}: {overlap} voxel centres inside two nuclei's shapes, each labelled by the power-nearest site; "
          f"boundary voxels among them {boundary} (expected 0); anchors within 1e-9 µm of a power face by the "
          f"oracle's arithmetic {sample.cells.cell_id[near].tolist() or 'none'}")
    assert boundary == 0


# ---------------------------------------------------------------- GN8 to GN12: types, genes and molecules

@pytest.mark.parametrize("name", ROUTINE)
def test_gn8_types_and_genes_on_the_routine_presets(name):
    book, config, generated = preset(name)
    types = config.types
    sample, context = generated.sample, generated.cell_context
    assert context.cell_id.tolist() == sample.cells.cell_id.tolist()
    regions = generated.regions
    assert regions.region_id.tolist() == list(range(1, types.regions + 1))
    centres = regions[["centre_z", "centre_y", "centre_x"]].to_numpy()
    anchors = sample.cells[["anchor_z", "anchor_y", "anchor_x"]].to_numpy()
    squared = ((anchors[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
    region = np.argmin(squared, axis=1)
    order = np.sort(squared, axis=1)
    gap = np.linalg.norm(centres[region] - centres[np.argsort(squared, axis=1)[:, 1]], axis=1)
    near = int(((order[:, 1] - order[:, 0]) / (2 * gap) <= 1e-9).sum())
    np.testing.assert_array_equal(context.region.astype(int).to_numpy(), region + 1)
    assert set(context.type) <= set(types.types)
    # The presets hold G1's table: two markers per type at 0.15, the eight shared genes at 0.0875; A2 the shared genes.
    table = generated.type_abundances
    genes = list(book.genes)
    for t, type_name in enumerate(types.types):
        row = table[table.type == type_name]
        assert row.gene_id.tolist() == genes and abs(row.weight.sum() - 1) <= 1e-9
        expected = [0.15 if g in (2 * t, 2 * t + 1) else 0.0875 if g >= 8 else 0.0 for g in range(16)]
        assert np.all(np.abs(row.weight.to_numpy() - expected) <= 1e-9)
        assert row.role.tolist() == ["marker"] * 8 + ["shared"] * 8
    ambient = table[table.type == "ambient"]
    assert ambient.gene_id.tolist() == genes and abs(ambient.weight.sum() - 1) <= 1e-9
    assert np.all(np.abs(ambient.weight.to_numpy() - np.array([0.0] * 8 + [0.125] * 8)) <= 1e-9)
    weight = {(r.type, r.gene_id): r.weight for r in table.itertuples()}
    type_of = dict(zip(context.cell_id.astype(int), context.type))
    molecules = sample.molecules
    cell_molecules = molecules[molecules.population == "cell"]
    for row in cell_molecules.itertuples():
        assert weight[type_of[placing_cell(row.amplicon_id)], row.gene_id] > 0, row.amplicon_id
    assert not set(molecules.gene_id[molecules.population == "ambient"]) & set(genes[:8])
    print(f"GN8 {name}: {len(context)} cells of types {context.type.value_counts().to_dict()}; anchors within 1e-9 µm "
          f"of a region boundary {near} (expected 0); {len(cell_molecules)} cell molecules, each of a gene of its "
          f"type; "
          f"{int((molecules.population == 'ambient').sum())} ambient molecules, none of a marker gene")
    assert near == 0


@pytest.mark.parametrize("name", ROUTINE)
def test_gn9_counts_and_marker_answers_on_the_routine_presets(name):
    book, config, generated = preset(name)
    sample = generated.sample
    molecules, cells = sample.molecules, sample.cells
    owned = molecules[molecules.true_cell_id.notna()]
    expected = {}
    for cell_id, gene, compartment in zip(owned.true_cell_id.astype(int), owned.gene_id, owned.compartment):
        count, nuclear = expected.get((cell_id, gene), (0, 0))
        expected[cell_id, gene] = (count + 1, nuclear + (compartment == "nucleus"))
    table = generated.cell_gene_counts
    assert {(int(r.cell_id), r.gene_id): (int(r.count), int(r.nuclear_count)) for r in table.itertuples()} == expected
    assert len(table) == len(expected) and (table["count"] > 0).all()
    order = {g: i for i, g in enumerate(book.genes)}
    keys = list(zip(table.cell_id.astype(int), table.gene_id.map(order)))
    assert keys == sorted(keys)
    sums = table.groupby("cell_id")["count"].sum()
    assert cells.n_molecules.tolist() == [int(sums.get(k, 0)) for k in cells.cell_id]
    counted = cells.cell_id[cells.in_sample].astype(int).tolist()
    counts = {(int(r.cell_id), r.gene_id): int(r.count) for r in table.itertuples()}
    pairs = generated.marker_pairs
    assert [tuple(p) for p in pairs[["gene_a", "gene_b", "relation"]].to_numpy()] == list(config.types.marker_pairs)
    for row in pairs.itertuples():
        values = mecr([counts.get((c, row.gene_a), 0) for c in counted],
                      [counts.get((c, row.gene_b), 0) for c in counted], 1)
        assert (row.cells_a, row.cells_b, row.both, row.either) == (values["cells_a"], values["cells_b"],
                                                                     values["both"], values["either"])
        if row.either:
            assert row.mecr == values["mecr"]
        else:
            assert pd.isna(row.mecr)
        if row.relation == "exclusive":
            assert row.both == 0 and (row.either == 0 or row.mecr == 0)
        print(f"GN9 {name} {row.gene_a}/{row.gene_b} ({row.relation}): cells_a {row.cells_a}, cells_b {row.cells_b}, "
              f"both {row.both}, either {row.either}, mecr {row.mecr} over {len(counted)} sample cells")


@pytest.mark.parametrize("name", ROUTINE)
def test_gn10_realized_gene_abundance_on_the_routine_presets(name):
    book, config, generated = preset(name)
    molecules = generated.sample.molecules
    cell_molecules = molecules[molecules.population == "cell"]
    type_of = dict(zip(generated.cell_context.cell_id.astype(int), generated.cell_context.type))
    kinds = cell_molecules.amplicon_id.map(lambda a: type_of[placing_cell(a)])
    table = generated.type_abundances
    compared = 0
    for type_name in config.types.types:
        genes = cell_molecules.gene_id[kinds == type_name]
        n = len(genes)
        if not n:
            print(f"GN10 {name} {type_name}: no cell molecules; not compared")
            continue
        counts = genes.value_counts()
        for row in table[table.type == type_name].itertuples():
            count = int(counts.get(row.gene_id, 0))
            if row.weight == 0:
                assert count == 0, (type_name, row.gene_id)  # a marker of another type: exactly 0
                continue
            share, se = count / n, math.sqrt(row.weight * (1 - row.weight) / n)
            print(f"GN10 {name} {type_name} {row.gene_id}: realized {share:.4f} against {row.weight:.4f} over {n} "
                  f"molecules, |difference| {abs(share - row.weight):.4f} ≤ 4 SE {4 * se:.4f}")
            assert abs(share - row.weight) <= 4 * se, (type_name, row.gene_id)
            compared += 1
    assert compared > 0


@pytest.mark.parametrize("name", ROUTINE)
def test_gn11_molecules_per_cell_and_the_count_guard_on_the_routine_presets(name):
    book, config, generated = preset(name)
    molecules, context = generated.sample.molecules, generated.cell_context
    placed = molecules.amplicon_id[molecules.population == "cell"].map(placing_cell).value_counts()
    total = int(placed.sum())
    means = float(context.molecule_mean.sum())
    print(f"GN11 {name}: Σ n_k = {total} against Σ μ_k = {means:.2f} over {len(context)} cells; |difference| "
          f"{abs(total - means):.2f} ≤ 4 SE {4 * math.sqrt(means):.2f}")
    assert abs(total - means) <= 4 * math.sqrt(means)
    assert context.molecule_mean.nunique() == len(context)  # one lognormal draw per cell (M2, spread > 0)
    guard = replace(config, sample=replace(config.sample, max_count=len(molecules) - 1))
    with pytest.raises(ValueError, match="max_count"):
        generate_context_scene(book, config=guard)


def quiet(config):
    """The preset with the formed template's noise and background off and float64 accumulation: the conditions of the
    rendering identity of docs/synthetic-sample-scene.md ("Rendering each FOV at its true offset")."""
    fov = replace(config.sample.fov, noise=NoiseConfig(), background=BackgroundConfig(), accumulation="float64")
    return replace(config, sample=replace(config.sample, fov=fov))


# Alone (two generations): 2.4 s dense 3D, 4.3 s 2D tissue (kept clear of the 5-s line), 5.7 s culture.
@pytest.mark.parametrize("name", ["context-dense3d-v1", pytest.param("context-tissue2d-v1", marks=pytest.mark.slow),
                                  pytest.param("context-culture-v1", marks=pytest.mark.slow)])
def test_gn11_rendering_identity_on_the_routine_presets(name):
    """Each FOV's reference-round image equals the independent formed oracle (the kernel of formed_oracle.py at the
    FOV's p, cut at a squared radius of 16) with each molecule's realized amplitudes per channel, which carry the
    preset's readout trend and mixing, within the oracle tolerance of the sample scene's test."""
    book, config, _ = preset(name)
    generated = generate_context_scene(book, config=quiet(config))
    reference = config.sample.fov.geometry.reference_round
    r = book.round_labels.index(reference)
    shape = np.array(config.sample.acquisition.fov_shape_zyx)
    largest = 0.0
    for fov, formed_scene in generated.sample.fovs.items():
        formed = formed_scene.formed.sort_values("formed_index")
        assert formed.formed_index.tolist() == list(range(len(formed_scene.realized)))
        expected = np.zeros((*shape, 4))
        for row, realized in zip(formed.itertuples(), formed_scene.realized[:, :, r]):
            # The support of the kernel (squared radius ≤ 16) lies inside 4 sz in Z and 4 sl max(e, 1) in Y and X.
            centre = np.array([row.z, row.y, row.x])
            half = 4 * np.array([row.sz, row.sl * max(row.e, 1.0), row.sl * max(row.e, 1.0)]) + 1
            lo = np.clip(np.floor(centre - half).astype(int), 0, shape)
            hi = np.clip(np.ceil(centre + half).astype(int) + 1, 0, shape)
            if np.any(hi <= lo):
                continue
            box = tuple(slice(a, b) for a, b in zip(lo, hi))
            points = np.moveaxis(np.indices(tuple(hi - lo), dtype=float), 0, -1) + lo
            radius = kernel_radius(points, centre, row.sz, row.sl, row.e, row.theta)
            expected[box] += np.where(radius <= 16, np.exp(-radius / 2), 0)[..., None] * realized
        image = formed_scene.rounds[reference]
        bound = np.maximum(1e-6, np.abs(expected) * np.finfo(np.float32).eps / 2 + 2e-10)
        assert expected.max() > 50 and np.all(np.abs(image.astype(np.float64) - expected) <= bound), fov
        largest = max(largest, float(np.abs(image.astype(np.float64) - expected).max()))
    print(f"GN11 {name}: rendering identity on {len(generated.sample.fovs)} FOVs, largest difference {largest:.3g}")


@pytest.mark.parametrize("name", ROUTINE)
def test_gn12_nuclear_fraction_and_compartments_on_the_routine_presets(name):
    _, config, generated = preset(name)
    sample = generated.sample
    molecules = sample.molecules
    kind = dict(zip(sample.cells.cell_id.astype(int), sample.cells.kind))
    cell_molecules = molecules[molecules.population == "cell"]
    placers = cell_molecules.amplicon_id.map(placing_cell)
    np.testing.assert_array_equal(cell_molecules.true_cell_id.astype(int).to_numpy(), placers.to_numpy())
    with_nucleus = placers.map(lambda k: kind[k] != "empty").to_numpy()
    nuclear = (cell_molecules.compartment == "nucleus").to_numpy()
    fraction, count = config.molecules.nuclear_fraction, int(with_nucleus.sum())
    share = float(nuclear[with_nucleus].mean())
    se = math.sqrt(fraction * (1 - fraction) / count)
    print(f"GN12 {name}: realized nuclear fraction {share:.4f} against {fraction} over {count} molecules of cells with "
          f"a nucleus; |difference| {abs(share - fraction):.4f} ≤ 4 SE {4 * se:.4f}; molecules of empty cells "
          f"{int((~with_nucleus).sum())}")
    assert abs(share - fraction) <= 4 * se
    assert (cell_molecules.compartment[~with_nucleus] == "cytoplasm").all()
    oracle = Oracle(generated)
    sigma = molecules[["sample_z", "sample_y", "sample_x"]].to_numpy()
    cell, nucleus, near = oracle.labels(sigma)
    owner = dict(zip(sample.nuclei.nucleus_id.astype(int), sample.nuclei.cell_id.astype(int)))
    ok = ~near
    in_own_nucleus = np.array([n > 0 and owner[int(n)] == c for n, c in zip(nucleus, cell)])
    assert ((molecules.compartment == "nucleus").to_numpy() == in_own_nucleus)[ok].all()
    np.testing.assert_array_equal(molecules.true_cell_id.fillna(0).astype(int).to_numpy()[ok], cell[ok])
    print(f"GN12 {name}: molecule positions within 1e-9 µm of a boundary {int(near.sum())} (expected 0)")
    assert not near.any()


# ---------------------------------------------------------------- GN15: bright regions

@pytest.mark.parametrize("name", ROUTINE)
def test_gn15_bright_regions_on_the_routine_presets(name):
    _, config, generated = preset(name)
    regions = generated.bright_regions
    textured = {stain.name for stain in config.stains if stain.source == "nucleus" and stain.texture is not None}
    assert len(regions) and set(regions.stain) <= textured
    assert (regions["size"] > 0).all() and (regions.height > 0).all()
    assert set(regions.nucleus_id.astype(int)) <= set(generated.sample.nuclei.nucleus_id.astype(int))
    centres = regions[["centre_z", "centre_y", "centre_x"]].to_numpy(float)
    _, nucleus = _context_labels(generated, centres)
    assert nucleus.astype(int).tolist() == regions.nucleus_id.astype(int).tolist()
    _, by_oracle, near = Oracle(generated).labels(centres)
    assert np.all((by_oracle == regions.nucleus_id.to_numpy()) | near)
    counts = regions.groupby("nucleus_id").size()
    print(f"GN15 {name}: {len(regions)} regions of {sorted(set(regions.stain))} in {len(counts)} of "
          f"{len(generated.sample.nuclei)} nuclei; centres within 1e-9 µm of a boundary {int(near.sum())} "
          f"(expected 0); sizes {regions['size'].min():.3f} to {regions['size'].max():.3f} µm, heights "
          f"{regions.height.min():.3f} to {regions.height.max():.3f}")
    assert not near.any()


# ---------------------------------------------------------------- GN19: projection truth

@pytest.mark.parametrize("name", MONOLAYER)
def test_gn19_projection_truth_on_the_routine_presets(name):
    _, config, generated = preset(name)
    sample = generated.sample
    z_mid = generated.record["effective_config"]["z_mid"]
    acquisition = config.sample.acquisition
    assert abs(z_mid - (acquisition.fov_shape_zyx[0] - 1) / 2 * acquisition.spacing_zyx[0]) <= 1e-9
    assert (sample.nuclei.centre_z == z_mid).all() and (sample.cells.anchor_z == z_mid).all()
    assert (generated.lobes.offset_z == 0).all()
    oracle = Oracle(generated)
    grid = sample.record["sample_grid"]
    shifted = (np.array(grid["origin_zyx"]) + np.array([0.3, 0.37, 0.41]) * np.array(grid["spacing_zyx"])
               + np.moveaxis(np.indices(tuple(grid["shape_zyx"]), dtype=float), 0, -1) * np.array(grid["spacing_zyx"]))
    cases = list(grids(generated))
    cell, nucleus = _context_labels(generated, shifted)
    cases.append(("fractional", shifted, {"cell": cell, "nucleus": nucleus}))
    faces = 0
    for grid_name, points, labels in cases:
        near = oracle.labels(points)[2].any(axis=0)
        for target in ("nucleus", "cell"):
            volume = labels[target].astype(np.int64)
            top = volume.max(axis=0)
            bottom = np.where(volume > 0, volume, np.iinfo(np.int64).max).min(axis=0)
            mixed = (top > 0) & (bottom != top)
            assert not np.any(mixed & ~near), (name, grid_name, target)
            faces += int((mixed & near).sum())
            # P(y, x) is the Z maximum of the grid's labels: the one ID of the column, or 0.
            assert np.array_equal(np.where(top > 0, bottom, 0), top)
    print(f"GN19 {name}: {len(cases)} grids, no column holds two nucleus or cell IDs; columns within 1e-9 µm of a "
          f"power face {faces} (expected 0)")
    assert faces == 0
