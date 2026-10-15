"""Generator checks of the cellular scene's geometry and molecules (docs/synthetic-cellular-scene-validation.md).

GN1(b), GN4 to GN12, GN19 and GN21 for lobed nuclei, packing with touching nuclei, territories,
cell types and molecules by compartment. Labels are recomputed by the test oracle
(cellular_oracle.py) from the truth tables and by the generator's point evaluator; positions whose
decision lies within 1e-9 µm of a boundary are counted and reported (expected 0). Statistical rows
compare within four standard errors (4 SE) at the realized counts and print their values.
"""
from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from starfinder.report._measurements import mecr
from starfinder.synthetic import (ContextSceneConfig, NoiseConfig, generate_context_scene, generate_sample_scene,
                                  sample_scene_preset, save_context_scene, save_sample_scene)
from starfinder.synthetic._context import _context_labels

from .cellular_oracle import Oracle, check_sample_identities, grids, harmonic_bound, real_harmonic
from .cellular_scenes import RULES, scene
from .formed_oracle import kernel_radius

pytestmark = [pytest.mark.synthetic, pytest.mark.validation]

SHAPE_SCENES = ([f"shapes-{rule}" for rule in RULES] + [f"harmonic-{rule}" for rule in RULES]
                + ["monolayer", "monolayer-harmonic", "plane"])
PLANAR = ("monolayer", "monolayer-harmonic", "plane")


def placing_cell(amplicon_id):
    """The cell that placed a cell molecule (IDs c<k>-m<j>), None for an ambient one."""
    return int(amplicon_id[1:].split("-")[0]) if amplicon_id.startswith("c") else None


def nominal_size(scene_, planar):
    lobes = scene_.lobes
    if planar:
        return float((math.pi * lobes.lateral_radius ** 2 * lobes.elongation).sum())
    return float((4 / 3 * math.pi * lobes.axial_radius * lobes.lateral_radius ** 2 * lobes.elongation).sum())


def domain_size(scene_, planar):
    omega = scene_.sample.record["domain"]["omega"]
    extent = np.array(omega["hi_zyx"]) - np.array(omega["lo_zyx"])
    return float(extent[1] * extent[2]) if planar else float(np.prod(extent))


# ---------------------------------------------------------------- GN1(b): every new part off is the sample scene

def assert_same_sample_scene(expected, got):
    assert got.sample == expected.sample and got.record == expected.record
    for name in ("placement", "positions", "nuclei", "cells", "molecules", "cell_fovs", "molecule_fovs"):
        pd.testing.assert_frame_equal(getattr(got, name), getattr(expected, name), check_exact=True)
    assert got.sample_grid == expected.sample_grid and got.fov_metadata == expected.fov_metadata
    for name in ("stains", "sample_labels"):
        assert getattr(got, name).keys() == getattr(expected, name).keys()
        for key, array in getattr(expected, name).items():
            assert getattr(got, name)[key].dtype == array.dtype and np.array_equal(getattr(got, name)[key], array)
    assert got.fov_labels.keys() == expected.fov_labels.keys()
    for fov, labels in expected.fov_labels.items():
        for key, array in labels.items():
            assert np.array_equal(got.fov_labels[fov][key], array) and got.fov_labels[fov][key].dtype == array.dtype
    assert got.fovs.keys() == expected.fovs.keys()
    for fov, formed in expected.fovs.items():
        other = got.fovs[fov]
        assert other.provenance == formed.provenance and other.amplicon_ids == formed.amplicon_ids
        for label, image in formed.rounds.items():
            assert other.rounds[label].dtype == image.dtype and np.array_equal(other.rounds[label], image)
        for name in ("formed", "round_truth"):
            pd.testing.assert_frame_equal(getattr(other, name), getattr(formed, name), check_exact=True)
        for name in ("intended", "pre_mix", "realized"):
            assert np.array_equal(getattr(other, name), getattr(formed, name))


@pytest.mark.parametrize("preset", ["sample-tiny-v1", pytest.param("sample-small-v1", marks=pytest.mark.slow)])
def test_gn1b_every_part_off_is_the_sample_scene(preset, tmp_path):
    book, config = sample_scene_preset(preset)
    expected = generate_sample_scene(book, config=config)
    result = generate_context_scene(book, config=ContextSceneConfig(sample=config))
    assert_same_sample_scene(expected, result.sample)
    # With every part off no context. stream is drawn (GN21).
    assert result.record["stream_scheme"]["streams"] == []
    sample_root = save_sample_scene(expected, tmp_path / "sample")
    context_root = save_context_scene(result, tmp_path / "context")
    sample_files = sorted(p.relative_to(sample_root).as_posix() for p in sample_root.rglob("*") if p.is_file())
    assert sample_files and "sample.json" in sample_files
    for name in sample_files:
        assert (context_root / name).read_bytes() == (sample_root / name).read_bytes(), name
    extra = sorted(p.relative_to(context_root).as_posix() for p in context_root.rglob("*") if p.is_file())
    assert sorted(set(extra) - set(sample_files)) == sorted(
        ["context.json", "lobes.csv", "harmonics.csv", "cell_context.csv", "regions.csv", "type_abundances.csv",
         "cell_gene_counts.csv", "marker_pairs.csv", "stain_channels.csv", "stain_objects.csv", "bright_regions.csv",
         "unevenness.csv", "morphology_truth.csv"])


# ---------------------------------------------------------------- GN4: labels against the oracle and the evaluator

@pytest.mark.parametrize("name", SHAPE_SCENES + ["types", "density"])
def test_gn4_labels_against_the_oracle_and_the_evaluator(name):
    _, config, generated = scene(name)
    oracle = Oracle(generated)
    sample = generated.sample
    plane = sample.record["sample_grid"]["shape_zyx"][0] == 1
    jitter = sample.placement[["jitter_z_vox", "jitter_y_vox", "jitter_x_vox"]].to_numpy()
    fractional = np.abs(jitter - np.round(jitter)) > 1e-3
    # The FOVs sit at fractional offsets in Z, Y and X (Y and X only at Z=1, where the Z bound is 0).
    assert fractional[:, 1:].any(axis=0).all() and (plane or fractional[:, 0].any())
    assert sample.nuclei.nucleus_id.is_unique
    boundary_total = evaluator_differences = voxels = 0
    pieces = {}
    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connected; on one plane the 8-connected neighbourhood
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


# ---------------------------------------------------------------- GN5: lobes, LB1 and N2

def test_gn5_harmonic_bound_is_the_largest_value():
    """M_Y = sqrt((2L + 1) / 4π) on the sphere (Σ_m Y_lm² = (2l + 1) / 4π, reached by Y_l0 at the poles)."""
    polar, azimuth = np.meshgrid(np.linspace(0, math.pi, 361), np.linspace(0, 2 * math.pi, 721), indexing="ij")
    for degree in (2, 3, 6):
        largest = max(float(np.abs(real_harmonic(l, m, polar, azimuth)).max())
                      for l in range(0, degree + 1) for m in range(-l, l + 1))
        assert abs(largest - harmonic_bound(degree, False)) <= 1e-12


@pytest.mark.parametrize("name", SHAPE_SCENES)
def test_gn5_lobes_harmonics_and_one_connected_shape(name):
    _, config, generated = scene(name)
    oracle, lobes, nuclei = Oracle(generated), generated.lobes, generated.sample.nuclei
    planar = generated.record["effective_config"]["planar"]
    shape = config.nuclei
    first = lobes[lobes.lobe == 1].set_index("nucleus_id").loc[nuclei.nucleus_id]
    assert (first[["offset_z", "offset_y", "offset_x"]].to_numpy() == 0).all()
    for column in ("axial_radius", "lateral_radius", "elongation", "angle"):
        np.testing.assert_array_equal(first[column].to_numpy(), nuclei[column].to_numpy())
    low, high = shape.extra_lobes
    counts = lobes.groupby("nucleus_id").size()
    assert counts.between(1 + low, 1 + high).all()  # lobed_share 1
    assert (lobes.groupby("nucleus_id").lobe.apply(list) == counts.apply(lambda c: list(range(1, c + 1)))).all()
    ratios, binding, margins = [], 0, []
    for row in lobes[lobes.lobe >= 2].itertuples():
        one = first.loc[row.nucleus_id]
        offset = np.array([row.offset_z, row.offset_y, row.offset_x])
        distance = float(np.linalg.norm(offset))
        u = offset / distance
        e1 = oracle.lobe_one_extent(row.nucleus_id, u)
        eps = oracle.lobe_extent(row.nucleus_id, row.lobe - 1, -u)
        # LB1: D_l ≤ 0.9 (e_{n,1} + ε_l), so lobe l meets lobe 1 over at least 0.1 (e + ε).
        assert distance <= 0.9 * (e1 + eps) + 1e-9, row
        assert distance - eps < e1 - 0.09 * (e1 + eps), row
        margins.append(e1 - (distance - eps))
        scale = row.lateral_radius / one.lateral_radius
        assert 0.5 - 1e-9 <= scale <= 0.8 + 1e-9 and abs(row.axial_radius - scale * one.axial_radius) <= 1e-9
        assert row.elongation == one.elongation and 0 <= row.angle < math.pi
        ratio = distance / one.lateral_radius
        ratios.append(ratio)
        binding += ratio < 0.5 - 1e-9
        assert ratio <= 0.9 + 1e-9  # D_l = min(λ a_l, bound) with λ ≤ 0.9
        if planar:
            assert row.offset_z == 0.0
    if planar and shape.harmonics == 0:
        # In Y and X the bound never binds for these laws: D_l / a_l lies in the support of lobe_offset.
        assert binding == 0 and min(ratios) >= 0.5 - 1e-9
    amplitude = shape.harmonic_amplitude if shape.harmonics else 0.0
    bound = harmonic_bound(shape.harmonics, planar) if shape.harmonics else 0.0
    for x in nuclei.itertuples():
        own = lobes[lobes.nucleus_id == x.nucleus_id]
        expected = max(max(a, b * e) * ((1 + amplitude * bound) if l == 1 else 1.0)
                       + math.sqrt(oz ** 2 + oy ** 2 + ox ** 2)
                       for l, oz, oy, ox, a, b, e in zip(own.lobe, own.offset_z, own.offset_y, own.offset_x,
                                                         own.axial_radius, own.lateral_radius, own.elongation))
        assert abs(x.site_radius - expected) <= 1e-9
    harmonics = generated.harmonics
    if shape.harmonics:
        terms = 2 * (shape.harmonics - 1) if planar else sum(2 * l + 1 for l in range(2, shape.harmonics + 1))
        assert (harmonics.groupby("nucleus_id").size() == terms).all() and set(harmonics.nucleus_id) == set(
            nuclei.nucleus_id)
        sums = harmonics.groupby("nucleus_id").coefficient.apply(lambda c: np.abs(c).sum())
        assert np.all(np.abs(sums - amplitude) <= 1e-9)
        orders = set(harmonics.order)
        assert orders == ({"cos", "sin"} if planar else set(range(-shape.harmonics, shape.harmonics + 1)))
    else:
        assert harmonics.empty
    print(f"GN5 {name}: {len(nuclei)} nuclei, {len(ratios)} further lobes; D_l / a_l in [{min(ratios):.4f}, "
          f"{max(ratios):.4f}], LB1 binding for {binding}; smallest overlap of a lobe with lobe 1 "
          f"{min(margins):.4f} µm")


# ---------------------------------------------------------------- GN6: no overlap; a nucleus inside its own cell

@pytest.mark.parametrize("name", SHAPE_SCENES)
def test_gn6_no_overlap_and_a_nucleus_inside_its_own_cell(name):
    _, config, generated = scene(name)
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
    # Every site centre lies strictly inside its own power cell in the arithmetic K1 names; each anchor's own
    # evaluation gives its own cell.
    for j in range(len(oracle.centres)):
        power = oracle.label_power(oracle.centres[j])
        assert np.all(power[j] < np.delete(power, j)), j
    anchors = sample.cells[["anchor_z", "anchor_y", "anchor_x"]].to_numpy()
    cell, nucleus = _context_labels(generated, anchors)
    np.testing.assert_array_equal(cell, sample.cells.cell_id.to_numpy())
    first = sample.nuclei.groupby("cell_id").nucleus_id.min()
    for k, n in zip(sample.cells.cell_id, nucleus):
        assert n == first.get(k, 0)
    # The oracle's own arithmetic agrees except at an anchor that P1 placed on its partner's power face (D at the
    # floor sqrt(abs(R_j² − R_k²)), then nextafter steps): reported, see the worker notes (finding 2).
    oracle_cell, _, near = oracle.labels(anchors)
    np.testing.assert_array_equal(oracle_cell[~near], sample.cells.cell_id.to_numpy()[~near])
    on_face = sample.cells.cell_id[near].tolist()
    print(f"GN6 {name}: {overlap} voxel centres inside two nuclei's shapes, each labelled by the power-nearest site; "
          f"boundary voxels among them {boundary} (expected 0); anchors within 1e-9 µm of a power face by the "
          f"oracle's arithmetic {on_face or 'none'}")
    assert boundary == 0 and overlap > 0


# ---------------------------------------------------------------- GN7: packing

@pytest.mark.parametrize("name", SHAPE_SCENES + ["density"])
def test_gn7_k1_for_every_pair_and_the_record(name):
    _, config, generated = scene(name)
    oracle = Oracle(generated)
    packing = config.packing
    planar = generated.record["effective_config"]["planar"]
    pairs = touching = 0
    centres, radii = oracle.centres, oracle.radii
    for i in range(len(centres)):
        for k in range(i + 1, len(centres)):
            d = float(np.linalg.norm(centres[k] - centres[i]))
            # The two power tests, ‖c_k − c_i‖² > |R_i² − R_k²|: exact in the arithmetic K1 names, within 1e-9 µm in
            # the test's own.
            assert oracle.label_power(centres[k], [k])[0] < oracle.label_power(centres[k], [i])[0], (i, k)
            assert oracle.label_power(centres[i], [i])[0] < oracle.label_power(centres[i], [k])[0], (i, k)
            assert d > math.sqrt(abs(radii[i] ** 2 - radii[k] ** 2)) - 1e-9, (i, k)
            if d >= radii[i] + radii[k] + packing.gap:
                continue
            pairs += 1
            t = oracle.contact_distance(i, k, (centres[k] - centres[i]) / d)
            assert d >= (1 - packing.overlap) * t + packing.gap - 1e-9, (i, k, d, t)
            touching += d <= t
    record = generated.record["effective_config"]["packing"]
    sample = generated.sample
    assert record["placed_seeds"] == len(sample.cells) and record["placed_nuclei"] == len(sample.nuclei)
    assert abs(record["domain_size"] - domain_size(generated, planar)) <= 1e-9
    assert abs(record["nominal_size"] - nominal_size(generated, planar)) <= 1e-9
    assert abs(record["reached_density"] - record["nominal_size"] / record["domain_size"]) <= 1e-12
    if packing.density is None:
        assert record["mode"] == "count" and record["requested_seeds"] == 9 and not record["stopped"]
        assert record["placed_seeds"] == record["requested_seeds"]
        assert sample.cells.kind.tolist() == ["ordinary"] * 7 + ["binucleate", "empty"]
    else:
        # Seeds are added until the summed nominal size reaches φ · ‖Ω‖; the last one reaches it.
        assert record["mode"] == "density" and not record["stopped"]
        last = sample.cells.cell_id.max()
        own = generated.lobes[generated.lobes.nucleus_id.isin(sample.nuclei.nucleus_id[sample.nuclei.cell_id == last])]
        before = record["nominal_size"] - nominal_size(replace(generated, lobes=own), planar)
        target = packing.density * record["domain_size"]
        assert before < target <= record["nominal_size"]
    print(f"GN7 {name}: {len(centres)} sites, {pairs} pairs within reach, all K1 within 1e-9 µm; {touching} pairs at "
          f"or inside their contact distance (touching on their power face); reached density "
          f"{record['reached_density']:.4f} ({record['mode']} mode)")
    assert touching > 0 or packing.contact == 0


def test_gn7_crowded_stops_and_records_above_a_low_floor():
    _, config, generated = scene("crowded-0.1")
    record = generated.record["effective_config"]["packing"]
    assert record["mode"] == "density" and record["density"] == 0.6 and record["stopped"]
    assert record["stopped_at"] == f"cell-{record['placed_seeds'] + 1}"
    assert record["placed_seeds"] == len(generated.sample.cells)
    assert abs(record["nominal_size"] - nominal_size(generated, False)) <= 1e-9
    assert 0.1 * 0.6 <= record["reached_density"] < 0.6
    print(f"GN7 ctx_crowded floor 0.1: stopped at {record['stopped_at']} after {record['placed_seeds']} seeds, "
          f"reached density {record['reached_density']} of 0.6")


def test_gn7_crowded_at_the_page_floor_reaches_it():
    """ctx_crowded with floor 0.9 as the page writes it: the domain holds less than 0.6, but more than 0.9 · 0.6,
    so F2 stops and records without raising (worker notes, finding 1); the refusal below a floor is GN3's."""
    _, config, generated = scene("crowded-0.9")
    record = generated.record["effective_config"]["packing"]
    assert record["stopped"] and record["reached_density"] < 0.6
    assert record["reached_density"] >= 0.9 * 0.6
    print(f"GN7 ctx_crowded floor 0.9: stopped at {record['stopped_at']}, reached density "
          f"{record['reached_density']} ≥ floor · density = 0.54, so no ValueError")


# ---------------------------------------------------------------- GN19: projection truth in a monolayer

@pytest.mark.parametrize("name", ["monolayer", "monolayer-harmonic"])
def test_gn19_projection_truth_in_a_monolayer(name):
    _, config, generated = scene(name)
    sample = generated.sample
    acquisition = config.sample.acquisition
    z_mid = (acquisition.fov_shape_zyx[0] - 1) / 2 * acquisition.spacing_zyx[0]
    assert generated.record["effective_config"]["z_mid"] == z_mid
    assert (sample.nuclei.centre_z == z_mid).all() and (sample.cells.anchor_z == z_mid).all()
    assert (generated.lobes.offset_z == 0).all()
    oracle = Oracle(generated)
    grid = sample.record["sample_grid"]
    shifted = (np.array(grid["origin_zyx"]) + np.array([0.3, 0.37, 0.41]) * np.array(grid["spacing_zyx"])
               + np.moveaxis(np.indices(tuple(grid["shape_zyx"]), dtype=float), 0, -1) * np.array(grid["spacing_zyx"]))
    cases = [(g, p, l) for g, p, l in grids(generated)]
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


# ---------------------------------------------------------------- GN8 to GN12: types, genes and molecules

def test_gn8_types_and_genes():
    book, config, generated = scene("types")
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
    main = dict(zip(regions.region_id, regions.main_type))
    takes_main = int(sum(main[r] == t for r, t in zip(context.region, context.type)))
    assert set(context.type) <= set(types.types)
    table = generated.type_abundances
    genes = list(book.genes)
    for t, name in enumerate(types.types):
        row = table[table.type == name]
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
    markers = set(genes[:8])
    assert not set(molecules.gene_id[molecules.population == "ambient"]) & markers
    print(f"GN8: {len(context)} cells, {takes_main} of them with their region's main type (purity "
          f"{types.purity}); anchors within 1e-9 µm of a region boundary {near} (expected 0); "
          f"{len(cell_molecules)} cell molecules, each of a gene of its type")
    assert near == 0


def test_gn9_counts_and_marker_answers():
    book, config, generated = scene("types")
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
    assert keys == sorted(keys)  # by cell, then in codebook order
    sums = generated.cell_gene_counts.groupby("cell_id")["count"].sum()
    assert cells.n_molecules.tolist() == [int(sums.get(k, 0)) for k in cells.cell_id]
    counted = cells.cell_id[cells.in_sample].astype(int).tolist()
    counts = {(int(r.cell_id), r.gene_id): int(r.count) for r in generated.cell_gene_counts.itertuples()}
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
        print(f"GN9 {row.gene_a}/{row.gene_b} ({row.relation}): cells_a {row.cells_a}, cells_b {row.cells_b}, "
              f"both {row.both}, either {row.either}, mecr {row.mecr} over {len(counted)} sample cells")


def test_gn10_realized_gene_abundance_of_each_type():
    book, config, generated = scene("types")
    molecules = generated.sample.molecules
    cell_molecules = molecules[molecules.population == "cell"]
    type_of = dict(zip(generated.cell_context.cell_id.astype(int), generated.cell_context.type))
    kinds = cell_molecules.amplicon_id.map(lambda a: type_of[placing_cell(a)])
    table = generated.type_abundances
    compared = 0
    for name in config.types.types:
        genes = cell_molecules.gene_id[kinds == name]
        n = len(genes)
        if not n:
            print(f"GN10 {name}: no cell molecules; not compared")
            continue
        counts = genes.value_counts()
        for row in table[table.type == name].itertuples():
            count = int(counts.get(row.gene_id, 0))
            if row.weight == 0:
                assert count == 0, (name, row.gene_id)
                continue
            share, se = count / n, math.sqrt(row.weight * (1 - row.weight) / n)
            print(f"GN10 {name} {row.gene_id}: realized {share:.4f} against {row.weight:.4f} over {n} molecules, "
                  f"|difference| {abs(share - row.weight):.4f} ≤ 4 SE {4 * se:.4f}")
            assert abs(share - row.weight) <= 4 * se, (name, row.gene_id)
            compared += 1
    assert compared > 0


def test_gn11_molecules_per_cell_and_the_count_guard():
    book, config, generated = scene("types")
    molecules, context = generated.sample.molecules, generated.cell_context
    placed = molecules.amplicon_id[molecules.population == "cell"].map(placing_cell).value_counts()
    total = int(placed.sum())
    means = float(context.molecule_mean.sum())
    print(f"GN11 Σ n_k = {total} against Σ μ_k = {means:.2f} over {len(context)} cells; |difference| "
          f"{abs(total - means):.2f} ≤ 4 SE {4 * math.sqrt(means):.2f}")
    assert abs(total - means) <= 4 * math.sqrt(means)
    assert context.molecule_mean.nunique() == len(context)  # one lognormal draw per cell (M2)
    guard = replace(config, sample=replace(config.sample, max_count=len(molecules) - 1))
    with pytest.raises(ValueError, match="max_count"):
        generate_context_scene(book, config=guard)
    _, zero, flat = scene("spread-zero")
    assert (flat.cell_context.molecule_mean == 20.0).all()  # spread 0 is M1: every mean is the median


def test_gn11_molecules_reach_the_formed_generator_unchanged():
    _, config, generated = scene("types")
    sample = generated.sample
    genes = dict(zip(sample.molecules.amplicon_id, sample.molecules.gene_id))
    properties = sample.molecules.set_index("amplicon_id")[["A", "sz", "sl", "e", "theta"]]
    for fov, formed_scene in sample.fovs.items():
        rows = sample.molecule_fovs[sample.molecule_fovs.fov_id == fov].sort_values("csv_row")
        formed = formed_scene.formed.set_index("amplicon_id").loc[rows.amplicon_id]
        np.testing.assert_array_equal(formed[["z", "y", "x"]].to_numpy(), rows[["z", "y", "x"]].to_numpy())
        assert formed.gene_id.tolist() == [genes[a] for a in rows.amplicon_id]
        np.testing.assert_array_equal(formed[["A", "sz", "sl", "e", "theta"]].to_numpy(),
                                      properties.loc[rows.amplicon_id].to_numpy())


@pytest.mark.slow
def test_gn11_rendering_identity_of_the_reference_round():
    book, config, _ = scene("types")
    quiet = replace(config, sample=replace(config.sample, fov=replace(config.sample.fov, noise=NoiseConfig())))
    generated = generate_context_scene(book, config=quiet)
    reference = config.sample.fov.geometry.reference_round
    r = book.round_labels.index(reference)
    shape = np.array(config.sample.acquisition.fov_shape_zyx)
    points = np.moveaxis(np.indices(tuple(shape), dtype=float), 0, -1)
    for fov, formed_scene in generated.sample.fovs.items():
        expected = np.zeros((*shape, 4))
        for row in formed_scene.formed.itertuples():
            radius = kernel_radius(points, np.array([row.z, row.y, row.x]), row.sz, row.sl, row.e, row.theta)
            expected[..., book.color_to_channel[row.color_sequence[r]]] += np.where(radius <= 16, np.exp(-radius / 2),
                                                                                    0) * row.A
        image = formed_scene.rounds[reference]
        bound = np.maximum(1e-6, np.abs(expected) * np.finfo(np.float32).eps / 2 + 2e-10)
        assert expected.max() > 50 and np.all(np.abs(image.astype(np.float64) - expected) <= bound), fov


def test_gn12_nuclear_fraction_and_compartments():
    _, config, generated = scene("types")
    sample = generated.sample
    molecules = sample.molecules
    kind = dict(zip(sample.cells.cell_id.astype(int), sample.cells.kind))
    cell_molecules = molecules[molecules.population == "cell"]
    placers = cell_molecules.amplicon_id.map(placing_cell)
    # Every cell molecule lies in the cell that placed it.
    np.testing.assert_array_equal(cell_molecules.true_cell_id.astype(int).to_numpy(), placers.to_numpy())
    with_nucleus = placers.map(lambda k: kind[k] != "empty").to_numpy()
    nuclear = (cell_molecules.compartment == "nucleus").to_numpy()
    fraction, count = config.molecules.nuclear_fraction, int(with_nucleus.sum())
    share = float(nuclear[with_nucleus].mean())
    se = math.sqrt(fraction * (1 - fraction) / count)
    print(f"GN12 realized nuclear fraction {share:.4f} against {fraction} over {count} molecules of cells with a "
          f"nucleus; |difference| {abs(share - fraction):.4f} ≤ 4 SE {4 * se:.4f}")
    assert abs(share - fraction) <= 4 * se
    # An empty cell places all its molecules in its territory.
    assert (cell_molecules.compartment[~with_nucleus] == "cytoplasm").all()
    oracle = Oracle(generated)
    sigma = molecules[["sample_z", "sample_y", "sample_x"]].to_numpy()
    cell, nucleus, near = oracle.labels(sigma)
    owner = dict(zip(sample.nuclei.nucleus_id.astype(int), sample.nuclei.cell_id.astype(int)))
    ok = ~near
    in_own_nucleus = np.array([n > 0 and owner[int(n)] == c for n, c in zip(nucleus, cell)])
    assert ((molecules.compartment == "nucleus").to_numpy() == in_own_nucleus)[ok].all()
    np.testing.assert_array_equal(molecules.true_cell_id.fillna(0).astype(int).to_numpy()[ok], cell[ok])
    print(f"GN12 molecule positions within 1e-9 µm of a boundary: {int(near.sum())} (expected 0)")
    assert not near.any()


# ---------------------------------------------------------------- molecules over the territory of the selected rule

@pytest.mark.parametrize("rule", ["power", "grown"])
def test_molecules_fill_the_territory_of_the_selected_rule(rule):
    """The review's case: one nucleus of radius 0.5 µm and reach 0.1 µm. Under T3 (the whole power cell, here Ω) and
    T2 (δ = 2 µm) the molecules are placed over that territory, not within the site sphere plus the reach, and the
    same with or without a MoleculeConfig of the same mean."""
    _, _, plain = scene(f"territory-{rule}")
    _, _, configured = scene(f"territory-{rule}-molecules")
    molecules = plain.sample.molecules
    columns = ["amplicon_id", "population", "gene_id", "sample_z", "sample_y", "sample_x", "compartment",
               "true_cell_id"]
    pd.testing.assert_frame_equal(molecules[columns], configured.sample.molecules[columns], check_exact=True)
    assert len(molecules) > 50 and (molecules.population == "cell").all() and (molecules.true_cell_id == 1).all()
    centre = plain.sample.nuclei[["centre_z", "centre_y", "centre_x"]].to_numpy()[0]
    sigma = molecules[["sample_z", "sample_y", "sample_x"]].to_numpy()
    in_reach_box = np.all(np.abs(sigma - centre) <= 0.5 + 0.1, axis=1)
    distance = np.linalg.norm(sigma - centre, axis=1)
    if rule == "grown":
        assert (distance <= 0.5 + 2.0 + 1e-9).all()
    else:
        assert distance.max() > 10  # the power cell of the one site is Ω
    print(f"{rule}: {len(sigma)} molecules, {int(in_reach_box.sum())} inside the site sphere's box grown by the reach, "
          f"largest distance from the centre {distance.max():.2f} µm")
    assert in_reach_box.mean() < 0.1


@pytest.mark.parametrize("part", ["territories", "molecules", "types"])
def test_a_part_at_its_default_gives_the_sample_scene(part):
    """No part falls back to the sample scene's code: with one part set to its page default (T1, M1 with NF3 and the
    sample scene's ambient molecules, or types), the cellular path gives generate_sample_scene's placement, labels and
    molecule positions; with types only the genes are drawn by type."""
    from starfinder.synthetic import CellTypeConfig, MoleculeConfig, TerritoryConfig
    book, config = sample_scene_preset("sample-tiny-v1")
    value = dict(territories=TerritoryConfig("reach"), molecules=MoleculeConfig(), types=CellTypeConfig())[part]
    expected = generate_sample_scene(book, config=config)
    got = generate_context_scene(book, config=ContextSceneConfig(sample=config, **{part: value})).sample
    pd.testing.assert_frame_equal(got.nuclei, expected.nuclei, check_exact=True)
    pd.testing.assert_frame_equal(got.cells, expected.cells, check_exact=True)
    for key, array in expected.sample_labels.items():
        assert np.array_equal(got.sample_labels[key], array)
    for fov, labels in expected.fov_labels.items():
        for key, array in labels.items():
            assert np.array_equal(got.fov_labels[fov][key], array)
    columns = ["amplicon_id", "population", "sample_z", "sample_y", "sample_x", "compartment", "true_cell_id"]
    pd.testing.assert_frame_equal(got.molecules[columns], expected.molecules[columns], check_exact=True)
    if part == "types":
        assert not got.molecules.gene_id.equals(expected.molecules.gene_id)
    else:
        assert got.molecules.gene_id.equals(expected.molecules.gene_id)


def test_the_point_evaluator_on_a_scene_with_every_part_off():
    """With every part off the labels are the sample scene's rule L1; the evaluator rebuilt from the tables agrees."""
    book, config = sample_scene_preset("sample-tiny-v1")
    generated = generate_context_scene(book, config=ContextSceneConfig(sample=config))
    for grid, points, labels in grids(generated):
        cell, nucleus = _context_labels(generated, points)
        assert np.array_equal(cell, labels["cell"]) and np.array_equal(nucleus, labels["nucleus"]), grid


# ---------------------------------------------------------------- GN21: the new random streams

@pytest.mark.parametrize("name", SHAPE_SCENES + ["types", "density"])
def test_gn21_new_components_start_with_context(name):
    _, config, generated = scene(name)
    record = generated.record["stream_scheme"]
    drawn = {descriptor[4] for descriptor in record["streams"]}
    assert drawn and all(c.startswith("context.") for c in drawn)
    assert drawn <= set(record["components"]) and all(c.startswith("context.") for c in record["components"])
    sample_drawn = {descriptor[4] for descriptor in generated.sample.record["stream_scheme"]["streams"]}
    assert not any(c.startswith("context.") for c in sample_drawn)
    if config.packing is not None:
        # With packing given the uniform proposals still come from sample.cell.placement.
        assert "sample.cell.placement" in sample_drawn


@pytest.mark.slow
@pytest.mark.parametrize("copy", ["types-abundance", "types-support"])
def test_gn21_a_type_abundance_changes_only_that_types_genes(copy):
    """Type 2's row changed: its positive weights ("types-abundance"), or its support, a weight set to 0 and given to
    another gene ("types-support", the review's case). No ambient molecule and no molecule of another type changes."""
    _, _, first = scene("types")
    _, _, changed = scene(copy)
    a, b = first.sample.molecules, changed.sample.molecules
    columns = ["amplicon_id", "population", "sample_z", "sample_y", "sample_x", "compartment", "true_cell_id"]
    pd.testing.assert_frame_equal(a[columns], b[columns], check_exact=True)
    type_of = dict(zip(first.cell_context.cell_id.astype(int), first.cell_context.type))
    of_type_2 = a.amplicon_id.map(lambda i: placing_cell(i) is not None and type_of[placing_cell(i)] == "type-2")
    differs = (a.gene_id != b.gene_id).to_numpy()
    ambient = (a.population == "ambient").to_numpy()
    assert ambient.any() and not differs[ambient].any()
    assert not differs[~of_type_2.to_numpy()].any() and differs[of_type_2.to_numpy()].any()
    # The ambient abundance (A2) and the genes' roles do not depend on a type's row.
    other = lambda t: t[t.type != "type-2"].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(other(first.type_abundances), other(changed.type_abundances), check_exact=True)
    print(f"GN21 type 2's abundance changed ({copy}): {int(differs.sum())} of {int(of_type_2.sum())} type-2 molecules "
          f"changed gene, none of the other {int((~of_type_2).sum())}, of which {int(ambient.sum())} ambient")


@pytest.mark.slow
def test_gn21_a_type_count_changes_only_that_types_molecules():
    _, _, first = scene("types")
    _, _, changed = scene("types-count")
    type_of = dict(zip(first.cell_context.cell_id.astype(int), first.cell_context.type))
    columns = ["amplicon_id", "population", "gene_id", "sample_z", "sample_y", "sample_x", "compartment",
               "true_cell_id"]
    of_type_3 = lambda m: m.amplicon_id.map(  # noqa: E731
        lambda i: placing_cell(i) is not None and type_of[placing_cell(i)] == "type-3").to_numpy(dtype=bool)
    keep = lambda m: m[~of_type_3(m)][columns].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(keep(first.sample.molecules), keep(changed.sample.molecules), check_exact=True)
    count = lambda m: int(of_type_3(m).sum())  # noqa: E731
    assert count(changed.sample.molecules) < count(first.sample.molecules)
    print(f"GN21 type 3's molecules per cell changed: type-3 molecules {count(first.sample.molecules)} → "
          f"{count(changed.sample.molecules)}; every other molecule unchanged")
