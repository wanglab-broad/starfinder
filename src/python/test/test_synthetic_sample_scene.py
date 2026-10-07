"""The multi-FOV synthetic sample scene against its page (docs/synthetic-sample-scene.md).

Label functions, the stain signal, nominal positions, bounds and stream draws
are recomputed here from the saved tables and the page's formulas; the
rendering identity uses the independent formed oracle.
"""
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import weakref

import numpy as np
import pandas as pd
import pytest
from scipy.special import erfc

from starfinder.image import ImageMetadata
from starfinder.synthetic import (SAMPLE_PRESET_VERSION, SAMPLE_PRESETS, SAMPLE_TRUTH_VERSION,
                                  BackgroundConfig, CellularSceneConfig, FormedSceneConfig, GeometryConfig,
                                  NoiseConfig, ReadoutEffectsConfig, SampleSceneConfig, ScalarDistribution,
                                  StainConfig, TextureConfig, generate_sample_scene, sample_scene_preset,
                                  save_sample_scene)
from starfinder.synthetic._observation import evaluate_background_translated

from .formed_oracle import kernel_radius

pytestmark = pytest.mark.synthetic

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
PLACEMENT_COLUMNS = [
    "fov_id", "grid_column", "grid_row", "nominal_z", "nominal_y", "nominal_x",
    "true_origin_z", "true_origin_y", "true_origin_x", "true_correction_z", "true_correction_y",
    "true_correction_x", "jitter_z_vox", "jitter_y_vox", "jitter_x_vox", "true_origin_g1_z",
    "true_origin_g1_y", "true_origin_g1_x", "shape_z", "shape_y", "shape_x", "spacing_z", "spacing_y",
    "spacing_x"]
# The page's component tables ("Random streams and seeds").
SAMPLE_COMPONENTS = {
    "sample.jitter", "sample.nucleus.axial", "sample.nucleus.lateral", "sample.nucleus.elongation",
    "sample.nucleus.angle", "sample.cell.placement", "sample.cell.reach", "sample.molecule.count",
    "sample.molecule.placement", "sample.identity", "sample.brightness", "sample.width.axial",
    "sample.width.lateral", "sample.elongation", "sample.angle", "sample.background.count",
    "sample.background.placement", "sample.background.width", "sample.background.brightness",
    "sample.stain.brightness", "sample.stain.background.count", "sample.stain.background.placement",
    "sample.stain.background.width", "sample.stain.background.brightness"}
FOV_COMPONENTS = {"stain.noise.dependent", "stain.noise.independent", "stain.noise.correlated"}


def generator(descriptor):
    encoded = json.dumps(descriptor, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(encoded).digest(), "big")))


def with_(config, *, fov=None, acquisition=None, **changes):
    """dataclasses.replace on the nested template and acquisition in one construction."""
    return replace(config, fov=replace(config.fov, **(fov or {})),
                   acquisition=replace(config.acquisition, **(acquisition or {})), **changes)


def supplied(config, jitter):
    ids = [f"FOV_{k + 1:03d}" for k in range(config.acquisition.columns * config.acquisition.rows)]
    table = jitter if isinstance(jitter, dict) else {fov: jitter for fov in ids}
    return {"jitter": "supplied", "jitter_zyx": table}


@pytest.fixture(scope="module")
def tiny():
    book, config = sample_scene_preset("sample-tiny-v1")
    return book, config, generate_sample_scene(book, config=config)


@pytest.fixture(scope="module")
def saved(tiny, tmp_path_factory):
    return save_sample_scene(tiny[2], tmp_path_factory.mktemp("sample") / "tiny")


# ---------------------------------------------------------------- label functions recomputed from the tables

def sites(scene, config):
    centres, radii, cells, reach = [], [], [], []
    for cell in scene.cells.itertuples():
        if cell.kind == "empty":
            own = [((cell.anchor_z, cell.anchor_y, cell.anchor_x), config.cells.empty_radius)]
        else:
            rows = scene.nuclei[scene.nuclei.cell_id == cell.cell_id]
            own = [((n.centre_z, n.centre_y, n.centre_x), n.site_radius) for n in rows.itertuples()]
        for centre, radius in own:
            centres.append(centre)
            radii.append(radius)
            cells.append(int(cell.cell_id))
            reach.append(cell.reach)
    return np.array(centres, float).reshape(-1, 3), np.array(radii, float), np.array(cells), np.array(reach, float)


def rho(points, centre, axial, lateral, elongation, angle):
    d = points - np.asarray(centre)
    c, s = np.cos(np.float64(angle)), np.sin(np.float64(angle))
    u = c * d[..., 1] + s * d[..., 2]
    v = -s * d[..., 1] + c * d[..., 2]
    return np.sqrt((d[..., 0] / axial) ** 2 + (u / (lateral * elongation)) ** 2 + (v / lateral) ** 2)


def omega(scene):
    domain = scene.record["domain"]["omega"]
    return np.array(domain["lo_zyx"]), np.array(domain["hi_zyx"])


def in_box(points, lo, hi):
    return np.all((points >= lo) & (points < hi), axis=-1)


def labels_at(points, scene, config):
    """C(σ) and N(σ) of the page at (..., 3) points: power diagram with reach, ellipsoidal nuclei."""
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    centres, radii, owner, reach = sites(scene, config)
    lo, hi = omega(scene)
    inside = in_box(flat, lo, hi)
    dz, dy, dx = (flat[:, a, None] - centres[None, :, a] for a in range(3))
    squared = dz * dz + dy * dy + dx * dx
    best = np.argmin(squared - radii * radii, axis=1)
    near = np.sqrt(squared[np.arange(len(flat)), best]) <= radii[best] + reach[best]
    cell = np.where(inside & near, owner[best], 0)
    nucleus = np.zeros(len(flat), dtype=np.int64)
    for n in scene.nuclei.itertuples():
        hit = inside & (rho(flat, (n.centre_z, n.centre_y, n.centre_x), n.axial_radius, n.lateral_radius,
                            n.elongation, n.angle) <= 1)
        assert not nucleus[hit].any()
        nucleus[hit] = n.nucleus_id
    return cell.reshape(shape), nucleus.reshape(shape)


def fov_points(scene, fov):
    row = scene.placement.set_index("fov_id").loc[fov]
    true = np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x])
    spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
    shape = (int(row.shape_z), int(row.shape_y), int(row.shape_x))
    return true + np.moveaxis(np.indices(shape, dtype=float), 0, -1) * spacing, true, spacing, np.array(shape)


def footprint_of(points, true, spacing, shape):
    p = (points - true) / spacing
    return np.all((p >= -.5) & (p < shape - .5), axis=-1)


def check_identities(scene, config):
    """Identities 1 to 5 of "Identities a test can check" and the truth columns."""
    molecules, cells, nuclei = scene.molecules, scene.cells, scene.nuclei
    sigma = molecules[["sample_z", "sample_y", "sample_x"]].to_numpy()
    covered = scene.record["domain"]["covered"]
    k_lo, k_hi = np.array(covered["lo_zyx"]), np.array(covered["hi_zyx"])
    fov_ids = list(scene.positions.fov_id)
    # IDs and truth columns.
    assert molecules.amplicon_id.is_unique
    for row in molecules.itertuples():
        pattern = r"c[1-9]\d*-m\d+" if row.population == "cell" else r"a-m\d+"
        assert re.fullmatch(pattern, row.amplicon_id), row.amplicon_id
    np.testing.assert_array_equal(molecules.in_sample, in_box(sigma, k_lo, k_hi))
    np.testing.assert_array_equal(cells.in_sample, in_box(cells[["anchor_z", "anchor_y", "anchor_x"]].to_numpy(),
                                                          k_lo, k_hi))
    cell_of, nucleus_of = labels_at(sigma, scene, config)
    np.testing.assert_array_equal(molecules.true_cell_id.fillna(0).astype(int), cell_of)
    np.testing.assert_array_equal(molecules.true_nucleus_id.fillna(0).astype(int), nucleus_of)
    expected = np.where(nucleus_of > 0, "nucleus", np.where(cell_of > 0, "cytoplasm", "outside"))
    assert molecules.compartment.tolist() == expected.tolist()
    # 1. one cell or none, nucleus of that cell.
    assert set(molecules.true_cell_id.dropna()) <= set(cells.cell_id)
    owner = dict(zip(nuclei.nucleus_id, nuclei.cell_id))
    for row in molecules.dropna(subset=["true_nucleus_id"]).itertuples():
        assert owner[row.true_nucleus_id] == row.true_cell_id
    membership = dict(zip(cells.cell_id, cells.in_sample))
    assert molecules.true_cell_in_sample.tolist() == [
        pd.NA if pd.isna(c) else membership[c] for c in molecules.true_cell_id]
    counts = molecules.true_cell_id.value_counts()
    assert cells.n_molecules.tolist() == [int(counts.get(k, 0)) for k in cells.cell_id]
    # Site spheres keep the gap; nucleus site radius.
    np.testing.assert_array_equal(nuclei.site_radius,
                                  np.maximum(nuclei.axial_radius, nuclei.lateral_radius * nuclei.elongation))
    centres, radii, _, _ = sites(scene, config)
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            assert np.linalg.norm(centres[i] - centres[j]) >= radii[i] + radii[j] + config.cells.gap, (i, j)
    # 2. per-FOV and sample labels are the label functions at their voxel centres.
    grid = scene.record["sample_grid"]
    q = np.moveaxis(np.indices(tuple(grid["shape_zyx"]), dtype=float), 0, -1)
    sample_cell, sample_nucleus = labels_at(np.array(grid["origin_zyx"]) + q * np.array(grid["spacing_zyx"]),
                                            scene, config)
    np.testing.assert_array_equal(scene.sample_labels["cell"], sample_cell)
    np.testing.assert_array_equal(scene.sample_labels["nucleus"], sample_nucleus)
    volumes = [scene.sample_labels]
    for fov in fov_ids:
        points, true, spacing, shape = fov_points(scene, fov)
        cell, nucleus = labels_at(points, scene, config)
        assert np.count_nonzero(scene.fov_labels[fov]["cell"] != cell) == 0, fov
        assert np.count_nonzero(scene.fov_labels[fov]["nucleus"] != nucleus) == 0, fov
        volumes.append(scene.fov_labels[fov])
    for volume in volumes:
        for target in ("cell", "nucleus"):
            assert volume[target].dtype == np.uint32
        assert set(np.unique(volume["nucleus"])) - {0} <= set(nuclei.nucleus_id)
        assert set(np.unique(volume["cell"])) - {0} <= set(cells.cell_id)
        # Nucleus inside cell on every label voxel.
        hit = volume["nucleus"] > 0
        np.testing.assert_array_equal(volume["cell"][hit], [owner[v] for v in volume["nucleus"][hit]])
    sample_voxels = np.bincount(scene.sample_labels["cell"].ravel(), minlength=len(cells) + 1)
    assert cells.n_voxels.tolist() == [int(sample_voxels[k]) for k in cells.cell_id]
    # 3. coverage and multiplicity, from the placement truth.
    seen = np.zeros((len(molecules), len(fov_ids)), dtype=bool)
    nominal = np.zeros_like(seen)
    bound = np.array(config.acquisition.jitter_max_zyx) * np.array(config.acquisition.spacing_zyx)
    far = np.ones(len(molecules), dtype=bool)
    for f, row in enumerate(scene.placement.itertuples()):
        spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
        shape = np.array([row.shape_z, row.shape_y, row.shape_x])
        seen[:, f] = footprint_of(sigma, np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x]),
                                  spacing, shape)
        start = np.array([row.nominal_z, row.nominal_y, row.nominal_x])
        nominal[:, f] = footprint_of(sigma, start, spacing, shape)
        edges = (start - spacing / 2, start + (shape - .5) * spacing)
        far &= np.all((np.abs(sigma - edges[0]) > bound) & (np.abs(sigma - edges[1]) > bound), axis=1)
    n_fovs = seen.sum(axis=1)
    np.testing.assert_array_equal(molecules.n_fovs, n_fovs)
    assert molecules.fov_ids.fillna("").tolist() == [";".join(np.array(fov_ids)[s]) for s in seen]
    member = molecules.in_sample.to_numpy()
    assert (n_fovs[member] >= 1).all()
    assert not (n_fovs == 0)[member].any()
    rows = scene.molecule_fovs
    visible = rows[rows.in_footprint]
    assert (visible.in_sample.sum(), (~visible.in_sample).sum()) == (n_fovs[member].sum(), n_fovs[~member].sum())
    check = member & far
    assert check.sum() > 0
    np.testing.assert_array_equal(n_fovs[check], nominal[check].sum(axis=1))
    # 4. cells, by their anchors.
    anchors = cells[["anchor_z", "anchor_y", "anchor_x"]].to_numpy()
    anchored = rows_by = scene.cell_fovs
    sample_cells = cells.in_sample.to_numpy()
    assert (cells.n_fovs[sample_cells] >= 1).all()
    assert (anchored.anchor_in_footprint & anchored.in_sample).sum() == cells.n_fovs[sample_cells].sum()
    anchor_far = sample_cells.copy()
    nominal_anchor = np.zeros(len(cells), dtype=np.int64)
    for row in scene.placement.itertuples():
        spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
        shape = np.array([row.shape_z, row.shape_y, row.shape_x])
        start = np.array([row.nominal_z, row.nominal_y, row.nominal_x])
        nominal_anchor += footprint_of(anchors, start, spacing, shape)
        edges = (start - spacing / 2, start + (shape - .5) * spacing)
        anchor_far &= np.all((np.abs(anchors - edges[0]) > bound) & (np.abs(anchors - edges[1]) > bound), axis=1)
    np.testing.assert_array_equal(cells.n_fovs[anchor_far], nominal_anchor[anchor_far])
    for cell in cells.itertuples():
        own = rows_by[rows_by.cell_id == cell.cell_id]
        assert (cell.fov_ids if isinstance(cell.fov_ids, str) else "") == ";".join(own.fov_id[own.anchor_in_footprint])
        assert (cell.label_fov_ids if isinstance(cell.label_fov_ids, str) else "") == ";".join(
            own.fov_id[own.n_voxels > 0])
        assert cell.n_fovs == own.anchor_in_footprint.sum()
    for f, row in enumerate(scene.placement.itertuples()):
        fov = row.fov_id
        points, true, spacing, shape = fov_points(scene, fov)
        labels = scene.fov_labels[fov]["cell"]
        inside = footprint_of(anchors, true, spacing, shape)
        expected_rows = {k for k, a in zip(cells.cell_id, inside) if a} | set(np.unique(labels)) - {0}
        own = scene.cell_fovs[scene.cell_fovs.fov_id == fov]
        assert set(own.cell_id) == expected_rows
        for r in own.itertuples():
            mask = labels == r.cell_id
            assert r.n_voxels == mask.sum() and r.anchor_in_footprint == (r.cell_id in set(cells.cell_id[inside]))
            assert r.fov_edge == bool(mask[:, [0, -1]].any() or mask[:, :, [0, -1]].any())
            assert r.z_edge == bool(shape[0] > 1 and mask[[0, -1]].any())
    # 5. sampled labels against the truth, by the nullable-ID rule.
    for fov in fov_ids:
        points, true, spacing, shape = fov_points(scene, fov)
        own = rows[rows.fov_id == fov]
        p = own[["z", "y", "x"]].to_numpy()
        inside = np.all((p >= -.5) & (p < shape - .5), axis=1)
        np.testing.assert_array_equal(own.in_footprint, inside)
        assert own.fov_cell_label.isna().tolist() == (~inside).tolist()
        assert own.label_agrees.isna().tolist() == (~inside).tolist()
        voxel = np.floor(p[inside] + .5)
        cell_at_voxel, nucleus_at_voxel = labels_at(true + voxel * spacing, scene, config)
        np.testing.assert_array_equal(own.fov_cell_label[inside].astype(int), cell_at_voxel)
        np.testing.assert_array_equal(own.fov_nucleus_label[inside].astype(int), nucleus_at_voxel)
        truth = dict(zip(molecules.amplicon_id, molecules.true_cell_id.fillna(0).astype(int)))
        np.testing.assert_array_equal(own.label_agrees[inside].astype(bool),
                                      [truth[a] == v for a, v in zip(own.amplicon_id[inside], cell_at_voxel)])


def check_margins(scene):
    """molecule_fovs holds exactly the molecules whose p lies within M = 4·w + D + 1."""
    m = scene.molecules
    c, s = np.cos(m.theta), np.sin(m.theta)
    w = np.array([m.sz.max(), (m.sl * np.hypot(m.e * c, s)).max(), (m.sl * np.hypot(m.e * s, c)).max()])
    sigma = m[["sample_z", "sample_y", "sample_x"]].to_numpy()
    for row in scene.placement.itertuples():
        transforms = scene.fovs[row.fov_id].provenance["transforms"].values()
        assert all(t["kind"] in ("identity", "gaussian_rbf_translation") and not np.any(t["vectors_zyx"])
                   for t in transforms)
        d = np.max([np.abs(t["translation_zyx"]) for t in transforms], axis=0)
        margin = 4 * w + d + 1
        np.testing.assert_allclose(scene.record["fovs"][row.fov_id]["margin_zyx"], margin, rtol=0, atol=1e-12)
        true = np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x])
        spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
        shape = np.array([row.shape_z, row.shape_y, row.shape_x])
        p = (sigma - true) / spacing
        within = np.all((p >= -.5 - margin) & (p <= shape - .5 + margin), axis=1)
        own = scene.molecule_fovs[scene.molecule_fovs.fov_id == row.fov_id]
        assert own.amplicon_id.tolist() == m.amplicon_id[within].tolist()
        assert own.csv_row.tolist() == list(range(len(own)))
        assert scene.fovs[row.fov_id].formed.amplicon_id.tolist() == own.amplicon_id.tolist()


# ---------------------------------------------------------------- criterion 2: configuration validation

@pytest.mark.parametrize("change,match", [
    (dict(acquisition=dict(jitter_max_zyx=(0.4, 6.0, 1.5))), r"^y: the overlap rule"),
    (dict(acquisition=dict(jitter_max_zyx=(0.4, 1.5, 6.0))), r"^x: the overlap rule"),
    (dict(acquisition=dict(jitter_max_zyx=(2.5, 1.5, 1.5))), r"^z: jitter_max_z 2.5 must be below n_z / 2"),
    (dict(acquisition=dict(fov_shape_zyx=(1, 36, 36)), fov=dict(shape_zyx=(1, 36, 36))),
     r"^z: jitter_max_zyx must be 0 in z"),
    (dict(fov=dict(count=5)), r"fov\.count must be None"),
    (dict(fov=dict(density=0.1)), r"fov\.density must be None"),
    (dict(fov=dict(coordinates=((1, 1, 1),))), r"fov\.coordinates must be None"),
    (dict(fov=dict(amplicon_ids=("a",))), r"fov\.amplicon_ids must be None"),
    (dict(fov=dict(gene_ids={"a": "balanced-01"})), r"fov\.gene_ids must be None"),
    (dict(fov=dict(placement="weighted")), r"fov\.placement must be 'uniform'"),
    (dict(fov=dict(shape_zyx=(5, 36, 32))), r"fov\.shape_zyx \(5, 36, 32\) must equal acquisition\.fov_shape_zyx"),
    (dict(fov=dict(readout=ReadoutEffectsConfig(loss_enabled=True))), r"fov\.readout\.loss_enabled"),
    (dict(fov=dict(readout=ReadoutEffectsConfig(dropout_enabled=True))), r"fov\.readout\.dropout_enabled"),
    (dict(fov=dict(readout=ReadoutEffectsConfig(weakening_enabled=True))), r"fov\.readout\.weakening_enabled"),
    (dict(acquisition=dict(columns=0)), r"^columns must be an integer"),
    (dict(acquisition=dict(rows=0)), r"^rows must be an integer"),
    (dict(acquisition=dict(fov_shape_zyx=(5, 0, 36)), fov=dict(shape_zyx=(5, 0, 36))), r"^fov_shape_zyx must hold"),
    (dict(acquisition=dict(spacing_zyx=(1.0, 0.0, 0.5))), r"^spacing_zyx must be finite and > 0"),
    (dict(acquisition=dict(spacing_zyx=(1.0, float("nan"), 0.5))), r"^spacing_zyx must be finite"),
    (dict(acquisition=dict(overlap=1.0)), r"^overlap \(y\) must satisfy 0 ≤ overlap < 1"),
    (dict(acquisition=dict(overlap=(1 / 3, 1.0))), r"^overlap \(x\) must satisfy"),
    (dict(acquisition=dict(jitter_max_zyx=(0.4, -1.0, 1.5))), r"^jitter_max_zyx must be finite and ≥ 0"),
    (dict(acquisition=dict(jitter_max_zyx=(0.4, float("inf"), 1.5))), r"^jitter_max_zyx must be finite"),
    (dict(acquisition=dict(jitter="supplied", jitter_zyx={"FOV_001": (0, 0, 0), "FOV_002": (0, 0, 0),
                                                           "FOV_003": (0, 0, 0)})), r"jitter_zyx misses FOV FOV_004"),
    (dict(acquisition=dict(jitter="supplied", jitter_zyx={f"FOV_00{k}": (0, 0, 1.6 if k == 3 else 0)
                                                           for k in range(1, 5)})),
     r"jitter_zyx\['FOV_003'\] exceeds the x bound"),
])
def test_invalid_configs_raise_naming_the_axis_or_field(change, match):
    _, config = sample_scene_preset("sample-tiny-v1")
    with pytest.raises(ValueError, match=match):
        with_(config, **change)


def test_valid_config_constructs_and_generation_errors_name_the_seed_fov_or_field(tiny):
    book, config, _ = tiny
    assert isinstance(SampleSceneConfig(), SampleSceneConfig)
    assert isinstance(with_(config, acquisition=supplied(config, (0.0, 0.0, 0.0))), SampleSceneConfig)
    # Route R1: a negative slope makes the re-expressed FOV intercept negative.
    gradient = BackgroundConfig(gradient_enabled=True, gradient_intercept=0.0, gradient_slopes_zyx=(0, -1, 0))
    with pytest.raises(ValueError, match=r"^FOV_001: the background gradient intercept"):
        generate_sample_scene(book, config=with_(config, fov=dict(background=gradient)))
    crowded = replace(config, cells=CellularSceneConfig(nuclei=40, max_proposals=1))
    with pytest.raises(ValueError, match=r"^cell-\d+: no proposal accepted after max_proposals=1"):
        generate_sample_scene(book, config=crowded)
    lonely = replace(config, cells=CellularSceneConfig(nuclei=1, binucleate=0, max_proposals=1),
                     molecules_per_cell=40.0)
    with pytest.raises(ValueError, match=r"^c1-m\d+: no proposal accepted in cell 1 after max_proposals=1"):
        generate_sample_scene(book, config=lonely)
    with pytest.raises(ValueError, match=r"^max_count: \d+ molecules"):
        generate_sample_scene(book, config=replace(config, max_count=10))
    for name, value in (("nuclei", -1), ("max_proposals", 0), ("gap", 0.0), ("empty_radius", float("nan"))):
        with pytest.raises(ValueError, match=f"^{name}"):
            CellularSceneConfig(**{name: value})
    with pytest.raises(ValueError, match="^reach"):
        CellularSceneConfig(reach=ScalarDistribution("constant", (0.0,)))
    with pytest.raises(ValueError, match="^edge_width"):
        StainConfig(edge_width=0.0)


# ---------------------------------------------------------------- criterion 3: nominal grid and placement truth

def test_nominal_grid_and_placement_truth(tiny):
    book, config, scene = tiny
    acquisition = config.acquisition
    n, s = acquisition.fov_shape_zyx, acquisition.spacing_zyx
    step_y = (1 - acquisition.overlap) * n[1] * s[1]
    step_x = (1 - acquisition.overlap) * n[2] * s[2]
    assert scene.positions.columns.tolist() == ["fov_id", "nominal_z", "nominal_y", "nominal_x"]
    placement = scene.placement
    assert placement.columns.tolist() == PLACEMENT_COLUMNS
    for k, row in enumerate(placement.itertuples()):
        column, grid_row = k // acquisition.rows, k % acquisition.rows
        assert row.fov_id == f"FOV_{k + 1:03d}" == scene.positions.fov_id[k]
        assert (row.grid_column, row.grid_row) == (column, grid_row)
        assert (row.nominal_z, row.nominal_y, row.nominal_x) == (0, grid_row * step_y, column * step_x)
        assert tuple(scene.positions.iloc[k, 1:]) == (0, grid_row * step_y, column * step_x)
        for i, a in enumerate("zyx"):
            nominal, true = getattr(row, f"nominal_{a}"), getattr(row, f"true_origin_{a}")
            correction = getattr(row, f"true_correction_{a}")
            assert correction == true - nominal
            jitter = getattr(row, f"jitter_{a}_vox")
            assert abs(jitter - correction / s[i]) <= 1e-12
            assert abs(jitter) <= acquisition.jitter_max_zyx[i]
            assert jitter != round(jitter)
            assert (getattr(row, f"shape_{a}"), getattr(row, f"spacing_{a}")) == (n[i], s[i])
    g1 = placement[[f"true_origin_g1_{a}" for a in "zyx"]].to_numpy() - placement[
        [f"nominal_{a}" for a in "zyx"]].to_numpy()
    assert np.all(np.abs(g1.mean(axis=0)) <= 1e-9)
    np.testing.assert_allclose(g1, placement[[f"true_correction_{a}" for a in "zyx"]].to_numpy()
                               - placement[[f"true_correction_{a}" for a in "zyx"]].to_numpy().mean(axis=0),
                               rtol=0, atol=1e-12)
    # The placement identity, through the formed tables and placed metadata.
    molecules = scene.molecules.set_index("amplicon_id")
    for row in placement.itertuples():
        true = (row.true_origin_z, row.true_origin_y, row.true_origin_x)
        formed = scene.fovs[row.fov_id].formed
        assert len(formed) > 0
        p = formed[["z", "y", "x"]].to_numpy()
        expected = molecules.loc[formed.amplicon_id.astype(str), ["sample_z", "sample_y", "sample_x"]].to_numpy()
        assert np.max(np.abs(np.asarray(true) + p * s - expected)) <= 1e-9
        placed = ImageMetadata("sample:sample", s, true, IDENTITY, "µm")
        assert np.max(np.abs(placed.index_to_world(p) - expected)) <= 1e-9
        # The FOV's reference metadata never carries the true origin.
        metadata = scene.fov_metadata[row.fov_id]
        assert metadata == scene.fovs[row.fov_id].metadata
        assert (metadata.origin_zyx, metadata.direction_zyx, metadata.spatial_unit, metadata.spacing_zyx) == (
            (0.0, 0.0, 0.0), IDENTITY, "µm", tuple(s))
        assert metadata.frame_id == json.dumps(["sample-tiny-v1", "sample", row.fov_id, "formed"],
                                               separators=(",", ":")) + "/reference"
    # Bounds of Ω and K and the sample grid from the page's formulas.
    top = (0, step_y * (acquisition.rows - 1), step_x * (acquisition.columns - 1))
    j = acquisition.jitter_max_zyx
    lo = [-s[a] / 2 - j[a] * s[a] for a in range(3)]
    hi = [top[a] + (n[a] - .5) * s[a] + j[a] * s[a] for a in range(3)]
    domain = scene.record["domain"]
    assert domain["omega"] == dict(lo_zyx=lo, hi_zyx=hi)
    assert domain["covered"] == dict(lo_zyx=[-s[a] / 2 + j[a] * s[a] for a in range(3)],
                                     hi_zyx=[top[a] + (n[a] - .5) * s[a] - j[a] * s[a] for a in range(3)])
    origin = tuple(lo[a] + s[a] / 2 for a in range(3))
    shape = [math.ceil((hi[a] - lo[a]) / s[a] - 1e-9) for a in range(3)]
    assert shape == [6, 63, 63]
    assert scene.sample_grid == ImageMetadata("sample:sample", s, origin, IDENTITY, "µm")
    assert scene.record["sample_grid"]["shape_zyx"] == shape
    assert scene.sample_labels["cell"].shape == tuple(shape)
    np.testing.assert_allclose([hi[a] - lo[a] for a in range(3)], (5.8, 31.5, 31.5), rtol=0, atol=1e-12)


def test_supplied_jitter_places_each_fov(tiny):
    book, config, reference = tiny
    table = {"FOV_001": (-0.4, 0.5, -1.5), "FOV_002": (0.0, 0.0, 0.0), "FOV_003": (0.25, -1.0, 1.5),
             "FOV_004": (0.4, 1.5, -0.75)}
    scene = generate_sample_scene(book, config=with_(config, acquisition=supplied(config, table)))
    for row in scene.placement.itertuples():
        jitter = table[row.fov_id]
        assert (row.jitter_z_vox, row.jitter_y_vox, row.jitter_x_vox) == jitter
        assert (row.true_origin_z, row.true_origin_y, row.true_origin_x) == tuple(
            nominal + j * s for nominal, j, s in zip((row.nominal_z, row.nominal_y, row.nominal_x), jitter,
                                                     config.acquisition.spacing_zyx))
    pd.testing.assert_frame_equal(scene.positions, reference.positions, check_exact=True)
    delta = scene.record["fovs"]["FOV_001"]["delta_zyx"]
    assert delta == [0.0, 2.0, 0.0]


# ---------------------------------------------------------------- criterion 4: rendering, stain and no interpolation

def test_reference_round_matches_the_independent_oracle(tiny):
    book, config, _ = tiny
    scene = generate_sample_scene(book, config=with_(config, fov=dict(noise=NoiseConfig())))
    reference = config.fov.geometry.reference_round
    r = book.round_labels.index(reference)
    for fov, formed_scene in scene.fovs.items():
        formed = formed_scene.formed
        shape = np.array(config.acquisition.fov_shape_zyx)
        points = np.moveaxis(np.indices(tuple(shape), dtype=float), 0, -1)
        expected = np.zeros((*shape, 4))
        for row in formed.itertuples():
            radius = kernel_radius(points, np.array([row.z, row.y, row.x]), row.sz, row.sl, row.e, row.theta)
            channel = book.color_to_channel[row.color_sequence[r]]
            expected[..., channel] += np.where(radius <= 16, np.exp(-radius / 2), 0) * row.A
        image = formed_scene.rounds[reference]
        assert image.dtype == np.float32 and expected.max() > 50
        bound = np.maximum(1e-6, np.abs(expected) * np.finfo(np.float32).eps / 2 + 2e-10)
        assert np.all(np.abs(image.astype(np.float64) - expected) <= bound), fov


def test_no_interpolation_at_an_integer_offset(tiny):
    book, config, _ = tiny
    base = with_(config, fov=dict(dtype="float64", noise=NoiseConfig()))
    first = generate_sample_scene(book, config=with_(base, acquisition=supplied(base, (0.1, -0.8, -0.6))))
    moved = generate_sample_scene(book, config=with_(base, acquisition=supplied(base, (0.1, 0.2, 0.4))))
    content = ["amplicon_id", "sample_z", "sample_y", "sample_x", "A", "sz", "sl", "e", "theta"]
    pd.testing.assert_frame_equal(first.molecules[content], moved.molecules[content], check_exact=True)
    for fov in first.fovs:
        for label, image in first.fovs[fov].rounds.items():
            other = moved.fovs[fov].rounds[label]
            scale = max(np.abs(image).max(), np.abs(other).max())
            assert scale > 50
            assert np.max(np.abs(other[:, :-1, :-1] - image[:, 1:, 1:])) <= 1e-12 * scale, (fov, label)


def stain_truth(points, nuclei, width=0.5):
    total = np.zeros(points.shape[:-1])
    for n in nuclei.itertuples():
        r = rho(points, (n.centre_z, n.centre_y, n.centre_x), n.axial_radius, n.lateral_radius, n.elongation,
                n.angle)
        semi = (n.axial_radius, n.lateral_radius * n.elongation, n.lateral_radius)
        mean = (semi[0] * semi[1] * semi[2]) ** (1 / 3)
        value = n.stain_brightness * .5 * erfc((r - 1) * mean / (math.sqrt(2) * width))
        total += np.where(r <= 1 + 4 * width / min(semi), value, 0.0)
    return total


def stain_cut(points, nuclei, width=0.5):
    """True where ρ_n > 1 + 4w/min(a) for every nucleus."""
    outside = np.ones(points.shape[:-1], dtype=bool)
    for n in nuclei.itertuples():
        r = rho(points, (n.centre_z, n.centre_y, n.centre_x), n.axial_radius, n.lateral_radius, n.elongation,
                n.angle)
        outside &= r > 1 + 4 * width / min(n.axial_radius, n.lateral_radius)
    return outside


def test_stain_signal_baseline_and_default(tiny):
    book, config, default = tiny
    bare = generate_sample_scene(book, config=replace(config, stain=StainConfig(background=BackgroundConfig(),
                                                                                  noise=NoiseConfig())))
    baseline = generate_sample_scene(book, config=replace(config, stain=StainConfig(noise=NoiseConfig())))
    checked = 0
    for fov in bare.stains:
        points, true, spacing, shape = fov_points(bare, fov)
        image = bare.stains[fov].astype(np.float64)
        top = image.max()
        assert top > 100
        tolerance = 1e-6 * top
        expected = stain_truth(points, bare.nuclei)
        nearest = []
        for n in bare.nuclei.itertuples():
            index = np.floor((np.array([n.centre_z, n.centre_y, n.centre_x]) - true) / spacing + .5).astype(int)
            if np.all((index >= 0) & (index < shape)):
                nearest.append(tuple(index))
        shell = np.zeros(shape, dtype=bool)
        for n in bare.nuclei.itertuples():
            r = rho(points, (n.centre_z, n.centre_y, n.centre_x), n.axial_radius, n.lateral_radius, n.elongation,
                    n.angle)
            shell |= (r >= .9) & (r <= 1.1)
        for index in nearest:
            assert abs(image[index] - expected[index]) <= tolerance
        assert np.all(np.abs(image[shell] - expected[shell]) <= tolerance)
        cut = stain_cut(points, bare.nuclei)
        assert cut.any() and np.all(image[cut] == 0)
        checked += len(nearest) + int(shell.sum())
        with_base = baseline.stains[fov].astype(np.float64)
        assert np.all(np.abs(with_base[shell] - (expected[shell] + 100)) <= tolerance)
        assert np.all(np.abs(with_base[cut] - 100) <= tolerance)
        stain = default.stains[fov]
        assert stain.dtype == np.dtype(config.fov.dtype) and stain.shape == tuple(config.acquisition.fov_shape_zyx)
        assert default.fov_metadata[fov] == default.fovs[fov].metadata
    assert checked > 100
    again = generate_sample_scene(book, config=config)
    for fov, stain in default.stains.items():
        assert stain.tobytes() == again.stains[fov].tobytes()
    descriptors = [d for d in default.record["stream_scheme"]["streams"] if d[4].startswith("stain.noise.")]
    assert sorted({(d[3], d[4], d[7]) for d in descriptors}) == sorted(
        (json.dumps(["sample-tiny-v1", fov], separators=(",", ":")), f"stain.noise.{kind}", "DAPI")
        for fov in default.stains for kind in ("dependent", "independent"))


def check_background(scene, book):
    """Each FOV's rounds against the sample-frame components at δ_f + F_r⁻¹(p)."""
    components = scene.record["effective_config"]["background"]["components"]
    weights = np.asarray(scene.record["effective_config"]["background"]["tissue_weights"])
    baseline = np.asarray(scene.record["effective_config"]["background"]["baseline"])
    assert components
    for fov, formed in scene.fovs.items():
        delta = np.asarray(scene.record["fovs"][fov]["delta_zyx"])
        assert len(formed.formed) == 0
        for transform in formed.provenance["transforms"].values():
            label = transform["round_label"]
            r = book.round_labels.index(label)
            tissue, _ = evaluate_background_translated(components, formed.rounds[label].shape[:3],
                                                       np.asarray(transform["translation_zyx"]) - delta)
            image = formed.rounds[label].astype(np.float64)
            for c in range(image.shape[-1]):
                expected = tissue * weights[r, c] + baseline[r, c]
                assert np.max(np.abs(image[..., c] - expected)) <= 1e-6 * np.abs(image[..., c]).max(), (fov, label, c)


def test_background_in_the_sample_frame(tiny):
    book, config, _ = tiny
    background = BackgroundConfig(
        gradient_enabled=True, gradient_intercept=1.0, gradient_slopes_zyx=(0.5, 2.0, 3.0),
        regions_enabled=True, region_centers=((2.0, 20.0, 40.0), (3.0, 45.0, 15.0)),
        region_sigma_zyx=((2.0, 8.0, 8.0), (1.5, 6.0, 10.0)), region_heights=(4.0, 2.5),
        texture_enabled=True, texture=TextureConfig(count=5), tissue_weights=((1.0, .5, .25, 2.0),) * 4)
    scene = generate_sample_scene(book, config=with_(config, fov=dict(background=background, noise=NoiseConfig()),
                                                     cells=None, ambient_density=0.0))
    assert [c["kind"] for c in scene.record["effective_config"]["background"]["components"]] == [
        "gradient", "region", "region"] + ["texture"] * 5
    check_background(scene, book)


@pytest.mark.slow
def test_small_background_in_the_sample_frame():
    book, config = sample_scene_preset("sample-small-v1")
    scene = generate_sample_scene(book, config=with_(config, fov=dict(noise=NoiseConfig()), cells=None,
                                                     ambient_density=0.0))
    assert len(scene.fovs) == 9 and not scene.molecules.shape[0]
    check_background(scene, book)


# ---------------------------------------------------------------- criterion 5: cellular truth identities 1 to 5

def test_cellular_identities_on_the_preset(tiny):
    book, config, scene = tiny
    assert set(scene.cells.kind) == {"ordinary", "binucleate", "empty"}
    assert scene.molecules.in_sample.any() and (~scene.molecules.in_sample).any()
    assert (scene.molecules.population == "ambient").any()
    assert scene.molecule_fovs.label_agrees.notna().any()
    for volume in (scene.sample_labels, *scene.fov_labels.values()):
        assert volume["cell"].ndim == 3 and volume["cell"].shape[0] > 1
    check_identities(scene, config)
    check_margins(scene)


def test_cellular_identities_with_inward_jitter_and_the_aligned_crop(tiny):
    book, config, reference = tiny
    j = config.acquisition.jitter_max_zyx
    inward = {row.fov_id: (j[0], j[1] if row.grid_row == 0 else -j[1], j[2] if row.grid_column == 0 else -j[2])
              for row in reference.placement.itertuples()}
    scene = generate_sample_scene(book, config=with_(config, acquisition=supplied(config, inward)))
    check_identities(scene, config)
    for column in ("amplicon_id", "in_sample", "sample_z", "sample_y", "sample_x", "true_cell_id"):
        assert scene.molecules[column].equals(reference.molecules[column]), column
    assert scene.cells.in_sample.equals(reference.cells.in_sample)
    aligned = dict(inward, FOV_001=(-0.4, 0.5, -1.5))
    crop = generate_sample_scene(book, config=with_(config, acquisition=supplied(config, aligned)))
    delta = crop.record["fovs"]["FOV_001"]["delta_zyx"]
    assert delta == [0.0, 2.0, 0.0]
    for target in ("cell", "nucleus"):
        expected = crop.sample_labels[target][0:5, 2:38, 0:36]
        assert np.count_nonzero(crop.fov_labels["FOV_001"][target] != expected) == 0
    # The FOV voxel centres are the sample-grid centres of the crop, value for value.
    points, _, _, _ = fov_points(crop, "FOV_001")
    grid = crop.record["sample_grid"]
    q = np.moveaxis(np.indices((5, 36, 36), dtype=float), 0, -1) + delta
    assert np.count_nonzero(points != np.array(grid["origin_zyx"]) + q * np.array(grid["spacing_zyx"])) == 0
    # Voxel centres within 1e-9 µm of a label boundary: the page reports this count and expects 0.
    probed, clearance = boundary_probe(points, crop, config), boundary_clearance(points, crop, config)
    counts = {target: (int(probed[target].sum()), int((clearance[target] <= 1e-9).sum())) for target in probed}
    print(f"aligned crop FOV_001: voxels within 1e-9 µm of a label boundary (axis probe, distance bound) {counts}; "
          f"smallest distance bound (µm) { {t: float(c.min()) for t, c in clearance.items()} }")
    assert counts == {"cell": (0, 0), "nucleus": (0, 0)}, counts
    # Both measures see boundaries at a coarse step, and a probed change always lies within the bound.
    coarse = boundary_probe(points, crop, config, step=0.05)
    for target in coarse:
        assert coarse[target].any() and np.all(clearance[target][coarse[target]] <= 0.05), target
    check_identities(crop, config)


def boundary_probe(points, scene, config, step=1e-9):
    """Per label kind: True where C or N changes between the centre and the centre moved ±step µm on any axis."""
    cell, nucleus = labels_at(points, scene, config)
    near = {"cell": np.zeros(cell.shape, dtype=bool), "nucleus": np.zeros(nucleus.shape, dtype=bool)}
    for axis in range(3):
        for sign in (-1, 1):
            moved = points.copy()
            moved[..., axis] += sign * step
            other_cell, other_nucleus = labels_at(moved, scene, config)
            near["cell"] |= other_cell != cell
            near["nucleus"] |= other_nucleus != nucleus
    return near


def boundary_clearance(points, scene, config):
    """Per label kind, a lower bound (µm) of the distance from each point to any surface where the label can change.

    C changes only on a face of Ω, on a reach sphere ‖σ − c_k‖ = R_k + reach_k of some site, or on a power
    bisector between the nearest site and another site (exact plane distance (pow_k − pow_j*) / 2‖c_k − c_j*‖).
    N changes only on a face of Ω or on a nucleus surface, at least |ρ_n − 1| · min semi-axis away.
    """
    flat = points.reshape(-1, 3)
    lo, hi = omega(scene)
    faces = np.minimum(flat - lo, hi - flat).min(axis=1)
    centres, radii, _, reach = sites(scene, config)
    offset = flat[:, None, :] - centres[None]
    distance = np.sqrt(np.sum(offset * offset, axis=-1))
    power = distance ** 2 - radii ** 2
    nearest = np.argmin(power, axis=1)
    rows = np.arange(len(flat))
    separation = np.linalg.norm(centres[None] - centres[nearest][:, None], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        planes = np.where(separation > 0, (power - power[rows, nearest][:, None]) / (2 * separation), np.inf)
    spheres = np.abs(distance - (radii + reach)).min(axis=1)
    cell = np.minimum.reduce([faces, spheres, planes.min(axis=1)])
    nucleus = faces.copy()
    for n in scene.nuclei.itertuples():
        r = rho(flat, (n.centre_z, n.centre_y, n.centre_x), n.axial_radius, n.lateral_radius, n.elongation, n.angle)
        nucleus = np.minimum(nucleus, np.abs(r - 1) * min(n.axial_radius, n.lateral_radius))
    return {"cell": cell.reshape(points.shape[:-1]), "nucleus": nucleus.reshape(points.shape[:-1])}


@pytest.mark.parametrize("geometry", [
    GeometryConfig(translation_enabled=True, translation_max_zyx=(0, 1, 1), affine_enabled=True,
                   affine_max_zyx=(0.2, 0.8, 0.8), reference_round="round1"),
    GeometryConfig(local_enabled=True, centers_zyx=((2, 18, 18),), scales=(8,), local_magnitude=1.0,
                   reference_round="round1"),
    GeometryConfig(polynomial_enabled=True, polynomial_max_zyx=(0.2, 1.0, 1.0), reference_round="round1")])
def test_margin_covers_deformed_rounds(tiny, geometry):
    from starfinder.synthetic._geometry import _forward
    book, config, _ = tiny
    scene = generate_sample_scene(book, config=with_(config, fov=dict(geometry=geometry)))
    m = scene.molecules
    extent = 4 * np.array([m.sz.max(), m.sl.max(), m.sl.max()])
    for row in scene.placement.itertuples():
        true = np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x])
        spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
        shape = np.array([row.shape_z, row.shape_y, row.shape_x])
        rendered = set(scene.molecule_fovs.amplicon_id[scene.molecule_fovs.fov_id == row.fov_id])
        left = ~m.amplicon_id.isin(rendered).to_numpy()
        p = (m[["sample_z", "sample_y", "sample_x"]].to_numpy()[left] - true) / spacing
        for transform in scene.fovs[row.fov_id].provenance["transforms"].values():
            moved = _forward(p, transform)
            assert not np.all((moved + extent >= 0) & (moved - extent <= shape - 1), axis=1).any()


# ---------------------------------------------------------------- criterion 6: streams and repeatability

def test_redrawing_the_jitter_moves_nothing(tiny):
    book, config, reference = tiny
    scene = generate_sample_scene(book, config=with_(config, acquisition=supplied(config, (0.1, -0.2, 0.3))))
    columns = ["amplicon_id", "population", "in_sample", "gene_id", "sample_z", "sample_y", "sample_x", "A", "sz",
               "sl", "e", "theta", "true_cell_id", "true_nucleus_id", "compartment", "true_cell_in_sample"]
    pd.testing.assert_frame_equal(scene.molecules[columns], reference.molecules[columns], check_exact=True)
    columns = ["cell_id", "kind", "in_sample", "anchor_z", "anchor_y", "anchor_x", "reach", "n_molecules"]
    pd.testing.assert_frame_equal(scene.cells[columns], reference.cells[columns], check_exact=True)
    pd.testing.assert_frame_equal(scene.nuclei, reference.nuclei, check_exact=True)


def fov_draws(scene, fov):
    provenance = scene.fovs[fov].provenance
    streams = sorted(json.dumps(d) for d in provenance["stream_scheme"]["streams"]
                     if d[4].startswith(("geometry.", "noise.")))
    stain = sorted(json.dumps(d) for d in scene.record["stream_scheme"]["streams"]
                   if d[4].startswith("stain.noise.") and json.loads(d[3])[1] == fov)
    translations = {t["round_label"]: t["translation_zyx"] for t in provenance["transforms"].values()}
    row = scene.placement.set_index("fov_id").loc[fov]
    return (row.jitter_z_vox, row.jitter_y_vox, row.jitter_x_vox), streams, stain, translations


def test_adding_a_column_keeps_the_draws_of_existing_fovs(tiny):
    book, config, scene = tiny
    wider = generate_sample_scene(book, config=with_(config, acquisition=dict(columns=3)))
    assert list(wider.positions.fov_id) == [f"FOV_{k:03d}" for k in range(1, 7)]
    for fov in scene.positions.fov_id:
        jitter, streams, stain, translations = fov_draws(scene, fov)
        assert streams and stain and len(translations) == 4
        assert fov_draws(wider, fov) == (jitter, streams, stain, translations)
        assert any(json.loads(d)[3] == json.dumps(["sample-tiny-v1", fov], separators=(",", ":"))
                   for d in streams)


def test_new_draw_descriptors(tiny):
    book, config, scene = tiny
    streams = scene.record["stream_scheme"]["streams"]
    assert scene.record["stream_scheme"]["namespace"] == "starfinder.synthetic/1"
    fov_keys = {json.dumps(["sample-tiny-v1", fov], separators=(",", ":")): fov for fov in scene.positions.fov_id}
    molecules = set(scene.molecules.amplicon_id)
    patterns = {"sample.nucleus.": r"nucleus-[1-9]\d*", "sample.cell.": r"cell-[1-9]\d*",
                "sample.stain.brightness": r"nucleus-[1-9]\d*"}
    components = set()
    for descriptor in streams:
        namespace, split, seed, key, component, entity, round_label, channel = descriptor
        assert (namespace, split, seed) == ("starfinder.synthetic/1", "development", 42)
        components.add(component)
        if component in FOV_COMPONENTS:
            assert key in fov_keys and entity is None and round_label is None and channel == "DAPI"
            continue
        assert component in SAMPLE_COMPONENTS and key == "sample-tiny-v1" and round_label is channel is None
        if component == "sample.jitter":
            assert entity in fov_keys.values()
            u = generator(descriptor).random(3)
            row = scene.placement.set_index("fov_id").loc[entity]
            assert tuple(np.array(config.acquisition.jitter_max_zyx) * (2 * u - 1)) == (
                row.jitter_z_vox, row.jitter_y_vox, row.jitter_x_vox)
        elif component == "sample.molecule.count":
            if entity == "ambient":
                lo, hi = omega(scene)
                expected = (scene.molecules.population == "ambient").sum()
                assert generator(descriptor).poisson(config.ambient_density * float(np.prod(hi - lo))) == expected
            else:
                k = int(re.fullmatch(r"cell-([1-9]\d*)", entity).group(1))
                expected = scene.molecules.amplicon_id.str.startswith(f"c{k}-m").sum()
                assert generator(descriptor).poisson(config.molecules_per_cell) == expected
        elif component in ("sample.molecule.placement", "sample.identity", "sample.brightness",
                           "sample.width.axial", "sample.width.lateral", "sample.elongation", "sample.angle"):
            assert entity in molecules
        else:
            prefix = next(p for p in patterns if component.startswith(p))
            assert re.fullmatch(patterns[prefix], entity), descriptor
    assert {"sample.jitter", "sample.cell.placement", "sample.molecule.count", "sample.molecule.placement",
            "sample.identity", "sample.stain.brightness", "stain.noise.dependent"} <= components
    # Each molecule's gene is its sample.identity draw with the template abundances.
    for row in scene.molecules.head(5).itertuples():
        draw = generator(["starfinder.synthetic/1", "development", 42, "sample-tiny-v1", "sample.identity",
                          row.amplicon_id, None, None]).choice(16, p=np.full(16, 1 / 16))
        assert book.genes[draw] == row.gene_id


CROSS_PROCESS = """
import hashlib, json, re, sys
from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene
book, config = sample_scene_preset("sample-tiny-v1")
root = save_sample_scene(generate_sample_scene(book, config=config), sys.argv[1])
# tifffile writes a fresh urn:uuid into every OME-XML header; it is the only per-write byte.
mask = lambda data: re.sub(rb"urn:uuid:[0-9a-f-]{36}", b"urn:uuid:" + b"0" * 36, data)
files = {p.relative_to(root).as_posix(): hashlib.sha256(mask(p.read_bytes())).hexdigest()
         for p in sorted(root.rglob("*")) if p.is_file() and p.name != "sample.json"}
record = json.loads((root / "sample.json").read_bytes())
uuid = sorted(n for n, h in record["files"].items() if h != files[n])
record["files"] = {n: h for n, h in record["files"].items() if n not in uuid}
print(json.dumps(dict(files=files, uuid=uuid, record=record), sort_keys=True))
"""


def test_cross_process_file_hashes(tmp_path):
    outputs = []
    for seed in ("1", "2"):
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.append(json.loads(subprocess.check_output(
            [sys.executable, "-c", CROSS_PROCESS, str(tmp_path / seed)], env=environment, text=True, timeout=600)))
    assert outputs[0] == outputs[1]
    files, uuid = outputs[0]["files"], outputs[0]["uuid"]
    assert len(files) > 30 and all(name.endswith(".ome.tif") for name in uuid)
    # Every file that is not an OME-TIFF is byte-identical, unmasked, and recorded with its hash.
    assert set(files) - set(uuid) == set(outputs[0]["record"]["files"])
    raw = [json.loads((tmp_path / seed / "sample.json").read_text())["files"] for seed in ("1", "2")]
    assert {n: h for n, h in raw[0].items() if n not in uuid} == {n: h for n, h in raw[1].items() if n not in uuid}


# ---------------------------------------------------------------- criterion 7: the on_fov handoff

def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_on_fov_hands_over_each_fov_and_keeps_none(tiny):
    book, config, reference = tiny
    calls, refs = [], []

    def on_fov(fov, scene, stain, labels):
        gc.collect()
        dead = [ref() is None for ref in refs]
        arrays = dict(scene.rounds, stain=stain, **{f"label:{k}": v for k, v in labels.items()})
        calls.append((fov, {k: digest(v) for k, v in arrays.items()}, all(dead), len(scene.rounds), sorted(labels)))
        refs.extend(weakref.ref(v) for v in arrays.values())

    scene = generate_sample_scene(book, config=config, on_fov=on_fov)
    assert [c[0] for c in calls] == list(reference.positions.fov_id)
    for fov, digests, dead, rounds, labels in calls:
        assert dead and rounds == 4 and labels == ["cell", "nucleus"]
        expected = dict(reference.fovs[fov].rounds, stain=reference.stains[fov],
                        **{f"label:{k}": v for k, v in reference.fov_labels[fov].items()})
        assert digests == {k: digest(v) for k, v in expected.items()}
    assert scene.fovs == {} and scene.stains == {} and scene.fov_labels == {}
    for name in ("placement", "positions", "nuclei", "cells", "molecules", "cell_fovs", "molecule_fovs"):
        pd.testing.assert_frame_equal(getattr(scene, name), getattr(reference, name), check_exact=True)
    assert scene.sample_labels.keys() == reference.sample_labels.keys()
    for key, array in scene.sample_labels.items():
        np.testing.assert_array_equal(array, reference.sample_labels[key], strict=True)
    assert scene.record == reference.record
    assert scene.fov_metadata == reference.fov_metadata and scene.sample_grid == reference.sample_grid
    with pytest.raises(ValueError, match="on_fov"):
        save_sample_scene(scene, Path("unused-never-written"))
    assert not Path("unused-never-written").exists()


# ---------------------------------------------------------------- criterion 8: save, reload and the no-segmenter path

def test_saved_layout_and_hashes(tiny, saved):
    book, config, scene = tiny
    record = json.loads((saved / "sample.json").read_text())
    files = sorted(p.relative_to(saved).as_posix() for p in saved.rglob("*") if p.is_file())
    assert sorted(record["files"]) == [f for f in files if f != "sample.json"]
    for name, value in record["files"].items():
        assert hashlib.sha256((saved / name).read_bytes()).hexdigest() == value
    expected = {"positions.csv", "placement_truth.csv", "nuclei.csv", "cells.csv", "molecules.csv",
                "cell_fovs.csv", "molecule_fovs.csv", "labels/sample_nuclei.tif", "labels/sample_cells.tif"}
    for fov in scene.positions.fov_id:
        expected |= {f"fovs/{fov}/{name}" for name in (
            "formed.csv", "round_truth.csv", "signals.csv", "provenance.json", "stain.ome.tif",
            "labels/nuclei.tif", "labels/cells.tif", "molecules.csv",
            *(f"images/{label}.ome.tif" for label in book.round_labels))}
    assert set(record["files"]) == expected
    assert record["sample_truth_version"] == SAMPLE_TRUTH_VERSION == "1"
    assert record["sample_preset_version"] == SAMPLE_PRESET_VERSION
    assert record["formed_generator_version"] == "6"
    assert {k: v for k, v in record.items() if k != "files"} == json.loads(json.dumps(scene.record))
    for name in ("placement_truth", "positions", "molecules"):
        table = pd.read_csv(saved / f"{name}.csv")
        assert len(table) == len(getattr(scene, "placement" if name == "placement_truth" else name))
    reread = pd.read_csv(saved / "placement_truth.csv", float_precision="round_trip")
    assert reread.columns.tolist() == PLACEMENT_COLUMNS
    np.testing.assert_array_equal(reread[["true_origin_z", "true_origin_y", "true_origin_x"]],
                                  scene.placement[["true_origin_z", "true_origin_y", "true_origin_x"]])
    (saved.parent / "taken").mkdir()
    (saved.parent / "taken" / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        save_sample_scene(scene, saved.parent / "taken")
    assert [p.name for p in (saved.parent / "taken").iterdir()] == ["keep.txt"]
    with pytest.raises(FileExistsError):
        save_sample_scene(scene, saved)


def test_no_segmenter_path_reads_the_saved_files(tiny, saved):
    from starfinder.assignment import molecule_table_from_csv
    from starfinder.io import load_volume
    from starfinder.segmentation import ReferenceGrid, import_labels
    book, config, scene = tiny
    fov = "FOV_002"
    metadata = scene.fov_metadata[fov]
    grid = ReferenceGrid(tuple(config.acquisition.fov_shape_zyx), metadata, source="declared")
    namespace = json.dumps(["sample-tiny-v1", "sample", fov, None, "truth"], separators=(",", ":"))
    for target, name in (("cell", "cells.tif"), ("nucleus", "nuclei.tif")):
        result = import_labels(saved / "fovs" / fov / "labels" / name, grid=grid, target=target, relabel=False,
                               label_namespace=namespace)
        assert result.geometry == "volume" and result.labels.dtype == np.uint32
        assert result.record["import"]["metadata_source"] == "stored"
        np.testing.assert_array_equal(result.labels, scene.fov_labels[fov][target])
    stain = load_volume(saved / "fovs" / fov / "stain.ome.tif")
    assert stain.metadata == metadata
    np.testing.assert_array_equal(stain.image, scene.stains[fov])
    spot_namespace = json.dumps(["sample-tiny-v1", "sample", fov, None], separators=(",", ":"))
    table = molecule_table_from_csv(saved / "fovs" / fov / "molecules.csv", spot_namespace=spot_namespace,
                                    genes=book.genes)
    rows = scene.molecule_fovs[scene.molecule_fovs.fov_id == fov].set_index("csv_row")
    assert len(table.table) == len(rows) > 0
    genes = dict(zip(scene.molecules.amplicon_id, scene.molecules.gene_id))
    for spot in table.table.itertuples():
        row = rows.loc[int(spot.spot_id.removeprefix("csv:"))]
        assert spot.gene_id == genes[row.amplicon_id]
        np.testing.assert_allclose([spot.z, spot.y, spot.x], [row.z, row.y, row.x], rtol=0, atol=1e-12)


# ---------------------------------------------------------------- criterion 9: check SC1, a single-plane scene

def test_single_plane_scene_sc1(tmp_path):
    from starfinder.segmentation import ReferenceGrid, import_labels
    book, config = sample_scene_preset("sample-tiny-v1")
    scene_plane = replace(config, acquisition=replace(config.acquisition, fov_shape_zyx=(1, 36, 36),
                                                      jitter_max_zyx=(0.0, 1.5, 1.5)),
                          fov=replace(config.fov, shape_zyx=(1, 36, 36)))
    scene = generate_sample_scene(book, config=scene_plane)
    assert scene.record["sample_grid"]["shape_zyx"][0] == 1
    assert scene.sample_labels["cell"].shape[0] == 1
    for fov in scene.positions.fov_id:
        assert all(image.shape[:3] == (1, 36, 36) for image in scene.fovs[fov].rounds.values())
        assert scene.stains[fov].shape == (1, 36, 36)
        assert all(v.shape == (1, 36, 36) for v in scene.fov_labels[fov].values())
    assert (scene.nuclei.centre_z == 0).all() and (scene.cells.anchor_z == 0).all()
    assert (scene.molecules.sample_z == 0).all() and len(scene.molecules) > 0
    assert (scene.placement.true_origin_z == 0).all() and (scene.placement.jitter_z_vox == 0).all()
    assert not scene.cell_fovs.z_edge.any()
    check_identities(scene, scene_plane)
    check_margins(scene)
    root = save_sample_scene(scene, tmp_path / "plane")
    fov = "FOV_001"
    grid = ReferenceGrid((1, 36, 36), scene.fov_metadata[fov], source="declared")
    result = import_labels(root / "fovs" / fov / "labels" / "cells.tif", grid=grid, target="cell",
                           label_namespace=json.dumps(["sample-tiny-v1", "sample", fov, None, "truth"]))
    assert result.geometry == "plane"
    assert list(SAMPLE_PRESETS) == ["sample-tiny-v1", "sample-small-v1"]


# ---------------------------------------------------------------- criterion 10: presets and resources

def test_presets_follow_the_page_table():
    assert SAMPLE_PRESET_VERSION == "sample-presets-v1" and SAMPLE_TRUTH_VERSION == "1"
    assert list(SAMPLE_PRESETS) == ["sample-tiny-v1", "sample-small-v1"]
    table = {
        "sample-tiny-v1": ((2, 2), (5, 36, 36), (1.0, 0.5, 0.5), 1 / 3, (0.4, 1.5, 1.5), (0, 1, 1), (6, 63, 63),
                           (7, 1, 1), (1.5, 2.5), 3.0, 8.0, "float64"),
        "sample-small-v1": ((3, 3), (14, 96, 96), (0.346255, 0.194398, 0.194398), 0.25, (0.8, 4.5, 4.5), (1, 2, 2),
                            (16, 249, 249), (14, 2, 2), (2.0, 3.0), 2.5, 30.0, "float32")}
    for name, (grid, shape, spacing, overlap, jitter, translation, sample_grid, seeds, radii, reach, per_cell,
               accumulation) in table.items():
        book, config = sample_scene_preset(name)
        assert book.n_genes == 16 and book.round_labels == ("round1", "round2", "round3", "round4")
        assert len(book.channel_labels) == 4 and book.genes[0] == "balanced-01"
        a, fov, cells = config.acquisition, config.fov, config.cells
        assert (a.columns, a.rows, a.fov_shape_zyx, a.spacing_zyx, a.overlap, a.jitter_max_zyx) == (
            *grid, shape, spacing, overlap, jitter)
        assert a.jitter == "uniform" and a.order == "column_major_down_right"
        assert (fov.seed, fov.split, fov.scene_key, fov.dtype, fov.accumulation) == (
            42, "development", name, "float32", accumulation)
        assert fov.geometry.translation_enabled and tuple(fov.geometry.translation_max_zyx) == translation
        assert not any(getattr(fov.geometry, f"{k}_enabled") for k in ("local", "affine", "polynomial"))
        assert not any(getattr(fov.readout, f"{k}_enabled") for k in ("loss", "dropout", "weakening"))
        assert (cells.nuclei, cells.binucleate, cells.empty) == seeds
        assert (math.exp(cells.nucleus_axial_radius.parameters[0]),
                math.exp(cells.nucleus_lateral_radius.parameters[0])) == pytest.approx(radii, rel=1e-12)
        assert math.exp(cells.reach.parameters[0]) == pytest.approx(reach, rel=1e-12)
        assert (config.molecules_per_cell, config.ambient_density) == (per_cell, 0.002)
        assert config.stain == StainConfig()
        assert SAMPLE_PRESETS[name]["sample_grid_zyx"] == sample_grid
        assert SAMPLE_PRESETS[name]["version"] == SAMPLE_PRESET_VERSION
    _, tiny_config = sample_scene_preset("sample-tiny-v1")
    fov = tiny_config.fov
    assert (fov.brightness, fov.axial_width, fov.lateral_width) == (
        ScalarDistribution(parameters=(100,)), ScalarDistribution(), ScalarDistribution())
    assert fov.noise == NoiseConfig(independent_enabled=True, sigma=1.0) and fov.background == BackgroundConfig()
    _, small_config = sample_scene_preset("sample-small-v1")
    fov = small_config.fov
    assert fov.readout.trend_enabled and fov.readout.trend_base == 0.95 and fov.readout.mixing_enabled
    assert fov.background.gradient_enabled and fov.background.regions_enabled and fov.noise.model == "poisson"
    with pytest.raises(ValueError, match="unknown sample preset"):
        sample_scene_preset("sample-medium-v1")
    assert isinstance(FormedSceneConfig(), FormedSceneConfig) and isinstance(GeometryConfig(), GeometryConfig)


SMALL = """
import sys, time
start = time.perf_counter()
from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene
book, config = sample_scene_preset("sample-small-v1")
scene = generate_sample_scene(book, config=config)
generated = time.perf_counter() - start
save_sample_scene(scene, sys.argv[1])
print(len(scene.molecules), int(scene.molecules.in_sample.sum()), len(scene.cells), round(generated, 2))
"""


@pytest.mark.slow
def test_small_preset_time_and_peak_rss(tmp_path):
    result = subprocess.run(["/usr/bin/time", "-v", sys.executable, "-c", SMALL, str(tmp_path / "small")],
                            capture_output=True, text=True, timeout=1800, check=True)
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", result.stderr).group(1))
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", result.stderr).group(1)
    size = sum(p.stat().st_size for p in (tmp_path / "small").rglob("*") if p.is_file())
    molecules, sample, cells, generated = result.stdout.split()
    print(f"sample-small-v1: wall {elapsed}, generation {generated} s, peak RSS {rss} KiB, saved {size} bytes, "
          f"{molecules} molecules ({sample} in the sample), {cells} cells")
    assert rss < 4 * 1024 * 1024
    assert int(cells) == 18
    record = json.loads((tmp_path / "small" / "sample.json").read_text())
    assert record["sample_grid"]["shape_zyx"] == [16, 249, 249]
    assert len(record["fovs"]) == 9
