"""Checks of the cellular scene's generator that two test files run (docs/synthetic-cellular-scene-validation.md).

``check_lobes`` is GN5 for a scene whose nucleus shapes come from a configuration with ``lobed_share`` below 1: the
routine presets (test_synthetic_cellular_routine.py) and the ``slow`` presets (test_synthetic_cellular_presets.py,
which generates them once per module). The tiny fixtures' GN5 test (test_synthetic_cellular_scene.py) is W-483's.
``check_projection`` is GN19 for a monolayer scene: the tiny monolayers (test_synthetic_cellular_scene.py) and the
2D tissue and culture routine presets (test_synthetic_cellular_routine.py).
"""
import math

import numpy as np

from starfinder.synthetic._context import _context_labels

from .cellular_oracle import EPS, Oracle, grids, harmonic_bound


def check_lobes(name, config, generated):
    oracle, lobes, nuclei = Oracle(generated), generated.lobes, generated.sample.nuclei
    planar = generated.record["effective_config"]["planar"]
    shape = config.nuclei
    first = lobes[lobes.lobe == 1].set_index("nucleus_id").loc[nuclei.nucleus_id]
    assert (first[["offset_z", "offset_y", "offset_x"]].to_numpy() == 0).all()
    for column in ("axial_radius", "lateral_radius", "elongation", "angle"):
        # Lobe 1 repeats nuclei.csv: floats within 1e-9 (µm or radians), the row's tolerance.
        np.testing.assert_allclose(first[column].to_numpy(), nuclei[column].to_numpy(), rtol=0, atol=1e-9)
    low, high = shape.extra_lobes
    counts = lobes.groupby("nucleus_id").size()
    # One lobe, or 1 + an integer of extra_lobes (lobed_share of the preset).
    assert ((counts == 1) | counts.between(1 + low, 1 + high)).all()
    assert (lobes.groupby("nucleus_id").lobe.apply(list) == counts.apply(lambda c: list(range(1, c + 1)))).all()
    scale_low, scale_high = shape.lobe_scale.parameters
    offset_low, offset_high = shape.lobe_offset.parameters
    ratios, binding, margins = [], 0, []
    for row in lobes[lobes.lobe >= 2].itertuples():
        one = first.loc[row.nucleus_id]
        offset = np.array([row.offset_z, row.offset_y, row.offset_x])
        distance = float(np.linalg.norm(offset))
        u = offset / distance
        e1 = oracle.lobe_one_extent(row.nucleus_id, u)
        eps = oracle.lobe_extent(row.nucleus_id, row.lobe - 1, -u)
        # LB1: D_l ≤ 0.9 (e_{n,1} + ε_l), so lobe l meets lobe 1 and the nucleus is one connected shape.
        assert distance <= 0.9 * (e1 + eps) + 1e-9, row
        assert distance - eps < e1 - 0.09 * (e1 + eps), row
        margins.append(e1 - (distance - eps))
        scale = row.lateral_radius / one.lateral_radius
        assert scale_low - 1e-9 <= scale <= scale_high + 1e-9
        assert abs(row.axial_radius - scale * one.axial_radius) <= 1e-9
        assert abs(row.elongation - one.elongation) <= 1e-9 and 0 <= row.angle < math.pi
        ratio = distance / one.lateral_radius
        ratios.append(ratio)
        binding += ratio < offset_low - 1e-9
        assert ratio <= offset_high + 1e-9  # D_l = min(λ a_l, bound) with λ ≤ 0.9
        if planar:
            assert row.offset_z == 0.0  # lobe offsets in the YX plane in a monolayer
    if planar and shape.harmonics == 0:
        # In the monolayer presets the bound does not bind: D_l / a_l lies in the support [0.5, 0.9] of lobe_offset.
        assert binding == 0 and (not ratios or min(ratios) >= offset_low - 1e-9)
    amplitude = shape.harmonic_amplitude if shape.harmonics else 0.0
    bound = harmonic_bound(shape.harmonics, planar) if shape.harmonics else 0.0
    for x in nuclei.itertuples():
        own = lobes[lobes.nucleus_id == x.nucleus_id]
        expected = max(max(a, b * e) * ((1 + amplitude * bound) if lobe == 1 else 1.0)
                       + math.sqrt(oz ** 2 + oy ** 2 + ox ** 2)
                       for lobe, oz, oy, ox, a, b, e in zip(own.lobe, own.offset_z, own.offset_y, own.offset_x,
                                                            own.axial_radius, own.lateral_radius, own.elongation))
        assert abs(x.site_radius - expected) <= 1e-9
    if shape.harmonics:
        sums = generated.harmonics.groupby("nucleus_id").coefficient.apply(lambda c: np.abs(c).sum())
        assert np.all(np.abs(sums - amplitude) <= 1e-9)
    else:
        assert generated.harmonics.empty
    print(f"GN5 {name}: {len(nuclei)} nuclei, {int((counts > 1).sum())} lobed, {len(ratios)} further lobes; "
          + (f"D_l / a_l in [{min(ratios):.4f}, {max(ratios):.4f}], LB1 binding for {binding}; smallest overlap of a "
             f"lobe with lobe 1 {min(margins):.4f} µm" if ratios else "no further lobe"))


def check_projection(name, config, generated):
    """GN19 (identity 11, PT1): every nucleus centre and anchor at z_mid; on the FOV grids, the sample grid and a grid
    at a fractional offset (0.3, 0.37, 0.41) voxels in Z, Y and X, the columns of fixed (y, x) within 1e-9 µm of a
    power face are counted (expected 0, asserted), no column holds two nucleus or two cell IDs, and P(y, x) is the Z
    maximum of the grid's labels. In a monolayer every site centre lies at z_mid, so a column's distance to a power
    face is that of its (y, x); the column counts when any of its voxel centres lies within 1e-9 µm of one."""
    sample = generated.sample
    acquisition = config.sample.acquisition
    z_mid = generated.record["effective_config"]["z_mid"]
    # "Exact" for a non-integer: within the default 1e-9 µm.
    assert abs(z_mid - (acquisition.fov_shape_zyx[0] - 1) / 2 * acquisition.spacing_zyx[0]) <= 1e-9
    assert np.all(np.abs(sample.nuclei.centre_z.to_numpy() - z_mid) <= 1e-9)
    assert np.all(np.abs(sample.cells.anchor_z.to_numpy() - z_mid) <= 1e-9)
    assert (generated.lobes.offset_z == 0).all()  # zeros exact
    oracle = Oracle(generated)
    grid = sample.record["sample_grid"]
    shifted = (np.array(grid["origin_zyx"]) + np.array([0.3, 0.37, 0.41]) * np.array(grid["spacing_zyx"])
               + np.moveaxis(np.indices(tuple(grid["shape_zyx"]), dtype=float), 0, -1) * np.array(grid["spacing_zyx"]))
    cases = list(grids(generated))
    cell, nucleus = _context_labels(generated, shifted)
    cases.append(("fractional", shifted, {"cell": cell, "nucleus": nucleus}))
    faces = mixed_total = columns = 0
    for grid_name, points, labels in cases:
        near_face = oracle.face_distance(points).min(axis=0) <= EPS
        faces += int(near_face.sum())
        columns += near_face.size
        for target in ("nucleus", "cell"):
            volume = labels[target].astype(np.int64)
            top = volume.max(axis=0)
            bottom = np.where(volume > 0, volume, np.iinfo(np.int64).max).min(axis=0)
            mixed = (top > 0) & (bottom != top)
            mixed_total += int(mixed.sum())
            # P(y, x) is the Z maximum of the grid's labels: the one ID of the column, or 0.
            assert np.array_equal(np.where(top > 0, bottom, 0), top), (name, grid_name, target)
    print(f"GN19 {name}: {len(cases)} grids, {columns} columns; columns within 1e-9 µm of a power face {faces} "
          f"(expected 0); columns holding two nucleus or two cell IDs {mixed_total} (expected 0)")
    assert faces == 0 and mixed_total == 0
