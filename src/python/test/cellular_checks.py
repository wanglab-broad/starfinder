"""Checks of the cellular scene's generator that two test files run (docs/synthetic-cellular-scene-validation.md).

``check_lobes`` is GN5 for a scene whose nucleus shapes come from a configuration with ``lobed_share`` below 1: the
routine presets (test_synthetic_cellular_routine.py) and the ``slow`` presets (test_synthetic_cellular_presets.py,
which generates them once per module). The tiny fixtures' GN5 test (test_synthetic_cellular_scene.py) is W-483's.
"""
import math

import numpy as np

from .cellular_oracle import Oracle, harmonic_bound


def check_lobes(name, config, generated):
    oracle, lobes, nuclei = Oracle(generated), generated.lobes, generated.sample.nuclei
    planar = generated.record["effective_config"]["planar"]
    shape = config.nuclei
    first = lobes[lobes.lobe == 1].set_index("nucleus_id").loc[nuclei.nucleus_id]
    assert (first[["offset_z", "offset_y", "offset_x"]].to_numpy() == 0).all()
    for column in ("axial_radius", "lateral_radius", "elongation", "angle"):
        np.testing.assert_array_equal(first[column].to_numpy(), nuclei[column].to_numpy())
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
        assert row.elongation == one.elongation and 0 <= row.angle < math.pi
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
