"""Generator checks of the cellular scene's stains and morphology round (docs/synthetic-cellular-scene-validation.md).

GN13 (the stain identities against the test oracle and the point evaluator, the plateau, the signs, the integer
shift), GN14 (the band at every boundary, on ``band_cases`` and ctx_stains), GN15 (bright regions), GN16 (the field),
GN17 (8-bit output), GN18 (the morphology round's truth translation), GN20 (``Z=1``) and GN21 for the stain components.
The oracle (cellular_oracle.StainOracle) evaluates every face term with every site; positions whose decision lies
within 1e-9 µm of a boundary or of a cut-off surface are counted and reported (expected 0). Tolerances are those of
the validation page: 1e-9 of the image maximum for an identity, 1e-10 for the integer shift (V13), the site-skipping
allowance ½ erfc(2√2) · A_n U (1 + T_n) of choice V9 where the nucleus argument is at most −4w.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from starfinder.io import load_volume_zyxc
from starfinder.segmentation import ReferenceGrid, import_labels
from starfinder.synthetic import generate_sample_scene, save_context_scene
from starfinder.synthetic._context import _context_band, _context_field, _context_labels, _context_stain
from starfinder.synthetic._observation import evaluate_background_translated

from .cellular_oracle import TAIL, Oracle, StainOracle, harmonic_bound
from .cellular_scenes import FIELD, PAIR, RULES, WHOLE, scene, tiny

pytestmark = [pytest.mark.synthetic, pytest.mark.validation]

STAIN_SCENES = ([f"stains-{rule}-{mode}" for rule in RULES for mode in ("whole", "drawn")]
                + ["stains-pair", "stains-empty", "plane-whole", "plane-drawn", "mr1"])
WHOLE_SCENES = [f"stains-{rule}-whole" for rule in RULES] + ["plane-whole"]
BANDS = {"membrane": 0.5, "membrane-thin": 0.01}


# ---------------------------------------------------------------- helpers

def frame(generated, fov):
    """(true origin, spacing, shape) of a FOV from the placement truth."""
    row = generated.sample.placement.set_index("fov_id").loc[fov]
    return (np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x]),
            np.array([row.spacing_z, row.spacing_y, row.spacing_x]),
            (int(row.shape_z), int(row.shape_y), int(row.shape_x)))


def shift(generated, fov):
    """τ_f (µm) of a FOV from morphology_truth."""
    row = generated.morphology_truth.set_index("fov_id").loc[fov]
    return np.array([row.translation_z_um, row.translation_y_um, row.translation_x_um])


def channels(generated, config, fov):
    """(index, stain, image, start, positions) of every stain channel of a FOV: the reference round's at true_f + p ⊙ s,
    the morphology round's at true_f + τ_f + q ⊙ s."""
    true, spacing, shape = frame(generated, fov)
    held = 0
    for index, stain in enumerate(config.stains):
        if stain.round == "reference":
            image, start = generated.stains[fov][stain.name], true
        else:
            image, start = generated.morphology[fov][..., held], true + shift(generated, fov)
            held += 1
        yield index, stain, image, start, start + np.moveaxis(np.indices(shape, dtype=float), 0, -1) * spacing


def background(generated, index, start, shape):
    """The stain's background at a grid starting at ``start``: tissue × weight + baseline, the tissue evaluated by
    ``evaluate_background_translated`` with the grid's offset on the sample grid."""
    grid = generated.sample.record["sample_grid"]
    offset = (start - np.array(grid["origin_zyx"])) / np.array(grid["spacing_zyx"])
    effective = generated.record["effective_config"]["stains"][index]["background"]
    tissue = (evaluate_background_translated(effective["components"], shape, -offset)[0]
              if effective["components"] else 0.0)
    return tissue * effective["tissue_weights"][0][0] + effective["baseline"][0][0]


def lipschitz(generated, k):
    """L_k of β_k (GN14): 1 under T1 and T3 and for an empty cell; under T2 r̄_n / a_min,l over the lobes of the cell's
    nuclei, lobe 1 under N2 times sqrt(1 / r_min² + G² / r_min⁴) (Bernstein's bound G of ‖∇_S r‖)."""
    effective = generated.record["effective_config"]
    if effective["territory_rule"] != "grown":
        return 1.0
    lobes = generated.lobes
    nuclei = generated.sample.nuclei
    harmonics = generated.harmonics
    terms = effective["harmonic_terms"]
    degree = max((t[0] for t in terms), default=0)
    planar = bool(effective["planar"])
    value = 1.0
    for n in nuclei.nucleus_id[nuclei.cell_id == k].astype(int):
        own = lobes[lobes.nucleus_id == n].sort_values("lobe")
        first = own.iloc[0]
        rbar = (first.axial_radius * first.lateral_radius * first.elongation * first.lateral_radius) ** (1 / 3)
        for row in own.itertuples():
            factor = rbar / min(row.axial_radius, row.lateral_radius)
            coefficients = harmonics.coefficient[harmonics.nucleus_id == n].abs()
            if row.lobe == 1 and len(coefficients):
                amplitude = float(coefficients.sum())
                bound = amplitude * harmonic_bound(degree, planar)
                r_min = 1 - bound
                g = degree * (amplitude if planar else bound)
                factor *= math.sqrt(1 / r_min ** 2 + g ** 2 / r_min ** 4)
            value = max(value, factor)
    return value


# ---------------------------------------------------------------- GN13: the stain identities

@pytest.mark.parametrize("name", STAIN_SCENES)
def test_gn13_stains_against_the_oracle_and_the_evaluator(name):
    _, config, generated = scene(name)
    oracle = Oracle(generated)
    counted = {"boundary": 0, "V9": 0, "plateau voxels": 0, "beyond every nucleus cut-off": 0,
               "territory lit by an empty cell only": 0, "band lit by an empty cell only": 0}
    empty_cells = set(generated.sample.cells.cell_id[generated.sample.cells.kind == "empty"].astype(int))
    for fov in generated.sample.positions.fov_id:
        signs_checked = False
        for index, stain, image, start, points in channels(generated, config, fov):
            assert image.dtype == np.float64 and image.shape == tuple(config.sample.acquisition.fov_shape_zyx)
            expected_background = background(generated, index, start, image.shape)
            stain_oracle = StainOracle(oracle, generated, stain.name)
            signal, near, allowance = stain_oracle.signal(points)
            cell, nucleus, boundary = oracle.labels(points)
            skip = near | boundary
            peak = float(np.abs(image).max())
            difference = np.abs(image - expected_background - signal)
            assert not np.any((difference > 1e-9 * peak + allowance) & ~skip), (name, fov, stain.name)
            counted["V9"] += int(((difference > 1e-9 * peak) & ~skip).sum())
            counted["boundary"] += int(skip.sum())
            evaluated = _context_stain(generated, stain.name, points)
            assert not np.any((np.abs(image - expected_background - evaluated) > 1e-9 * peak) & ~skip)
            if not len(oracle.centres):
                # No cell: every stain is its background, within the row's 1e-9 of the image maximum.
                assert np.all(np.abs(image - expected_background) <= 1e-9 * peak), stain.name
                continue
            p = points.reshape(-1, 3)
            power = stain_oracle.powers(p)
            terms = stain_oracle.nuclei(p, power, stain.edge_width)
            dark = np.ones(len(p), dtype=bool)
            for _, edge, _ in terms.values():
                dark &= edge == 0
            dark = dark.reshape(image.shape) & ~skip
            field = stain_oracle.field(p).reshape(image.shape)
            if stain.source == "nucleus":
                counted["beyond every nucleus cut-off"] += int(dark.sum())
                beyond = np.abs(image - np.broadcast_to(expected_background, image.shape))[dark]
                assert np.all(beyond <= 1e-9 * peak), (name, fov, stain.name)
                for n, (argument, _, _) in terms.items():
                    texture = stain_oracle.texture(n, p)[0].reshape(image.shape)
                    plateau = (nucleus == n) & (argument.reshape(image.shape) <= -4 * stain.edge_width) & ~skip
                    full = stain_oracle.plateau[n] * field * (1 + texture)
                    gap = (full - (image - expected_background))[plateau]
                    assert np.all(gap >= -1e-9 * peak) and np.all(gap <= TAIL * full[plateau] + 1e-9 * peak), n
                    counted["plateau voxels"] += int(plateau.sum())
            else:
                lit = image - expected_background
                if stain.source == "territory":
                    only = dark & np.isin(cell, list(empty_cells))
                    counted["territory lit by an empty cell only"] += int(only.sum())
                    assert np.all(lit[only] > 0)
                else:
                    width = stain.band_width
                    banded = {k: stain_oracle.cell(p, power, k)[1].reshape(image.shape) <= 4 * width
                              for k in stain_oracle.cell_sites}
                    only = np.zeros(image.shape, dtype=bool)
                    for k in empty_cells:
                        alone = banded[k].copy()
                        for other, mask in banded.items():
                            if other != k:
                                alone &= ~mask
                        only |= alone
                    counted["band lit by an empty cell only"] += int((only & ~skip).sum())
                    assert np.all(lit[only & ~skip] > 0)
            if not signs_checked:
                # The signs agree with the labels: N = n exactly where the nuclear argument is at most 0, C = k exactly
                # where d_k ≤ 0 (all FOV positions lie in Ω).
                for n, (argument, _, _) in terms.items():
                    agree = (argument.reshape(image.shape) <= 0) == (nucleus == n)
                    assert np.all(agree | boundary), (name, fov, n)
                for k in stain_oracle.cell_sites:
                    d = stain_oracle.cell(p, power, k)[0].reshape(image.shape)
                    assert np.all(((d <= 0) == (cell == k)) | boundary), (name, fov, k)
                signs_checked = True
    print(f"GN13 {name}: boundary voxels {counted['boundary']} (expected 0); voxels counted under the V9 allowance "
          f"{counted['V9']}; " + "; ".join(f"{k} {v}" for k, v in counted.items() if k not in ("boundary", "V9")))
    assert counted["boundary"] == 0
    if name == "stains-empty":
        assert not len(generated.stain_objects) and not len(generated.bright_regions)
    else:
        assert counted["plateau voxels"] > 0 and counted["territory lit by an empty cell only"] >= 0


def test_gn13_integer_shift_between_two_fovs():
    """The whole-voxel pair: FOV_001 and FOV_002 image a common sample position at two roundings; every stain of the
    two rounds agrees there within 1e-10 of the maximum (V13), except at boundary voxels (reported)."""
    _, config, generated = scene("stains-pair")
    oracle = Oracle(generated)
    deltas = {fov: np.array(generated.sample.record["fovs"][fov]["delta_zyx"]) for fov in ("FOV_001", "FOV_002")}
    step = deltas["FOV_002"] - deltas["FOV_001"]
    assert np.allclose(step, np.round(step), rtol=0, atol=1e-9)
    dz, dy, dx = (int(v) for v in np.round(step))
    assert (dz, dy, dx) == (0, 23, 2)
    first = {stain.name + stain.round: (image, points) for _, stain, image, _, points in
             channels(generated, config, "FOV_001")}
    worst, boundary_total, voxels = 0.0, 0, 0
    for _, stain, image, _, points in channels(generated, config, "FOV_002"):
        image_1, points_1 = first[stain.name + stain.round]
        a = image_1[:, dy:, dx:]
        b = image[:, :image.shape[1] - dy, :image.shape[2] - dx]
        pa, pb = points_1[:, dy:, dx:], points[:, :image.shape[1] - dy, :image.shape[2] - dx]
        assert np.abs(pa - pb).max() < 1e-12
        so = StainOracle(oracle, generated, stain.name)
        near = so.signal(pa)[1] | so.signal(pb)[1] | oracle.labels(pa)[2] | oracle.labels(pb)[2]
        peak = max(float(np.abs(image_1).max()), float(np.abs(image).max()))
        error = np.abs(a - b) / peak
        assert np.all((error <= 1e-10) | near), stain.name
        worst = max(worst, float(error[~near].max()))
        boundary_total += int(near.sum())
        voxels += a.size
    print(f"GN13 integer shift (0, 23, 2) between FOV_001 and FOV_002: {voxels} common voxels over every stain; "
          f"largest difference {worst:.2e} of the maximum (≤ 1e-10); boundary voxels {boundary_total} (expected 0)")
    assert boundary_total == 0


# ---------------------------------------------------------------- GN14: the band at every boundary

def _hand_scene(case):
    """A ``band_cases`` scene from truth tables written by hand (audit D of W-480): sites with their lobes, cells with
    reach or growth, Ω, and two boundary stains ``membrane`` (b 0.5 µm) and ``membrane-thin`` (b 0.01 µm)."""
    nuclei, cells, lobes = [], [], []
    for k, cell in enumerate(case["cells"], start=1):
        anchor = None
        for nucleus in cell.get("nuclei", ()):
            n = len(nuclei) + 1
            centre = np.array(nucleus["centre"], dtype=float)
            own = nucleus["lobes"] if "lobes" in nucleus else [((0.0, 0.0, 0.0), *nucleus["semi"])]
            radius = max(math.dist((0, 0, 0), o) + max(a, b * e) for o, a, b, e, _ in own)
            nuclei.append(dict(nucleus_id=n, cell_id=k, centre_z=centre[0], centre_y=centre[1], centre_x=centre[2],
                               axial_radius=own[0][1], lateral_radius=own[0][2], elongation=own[0][3],
                               angle=own[0][4], site_radius=radius))
            lobes.extend(dict(nucleus_id=n, lobe=l, offset_z=o[0], offset_y=o[1], offset_x=o[2], axial_radius=a,
                              lateral_radius=b, elongation=e, angle=t) for l, (o, a, b, e, t) in enumerate(own, 1))
            anchor = centre if anchor is None else anchor
        anchor = np.array(cell.get("anchor", (0, 0, 0)), dtype=float) if anchor is None else anchor
        cells.append(dict(cell_id=k, kind="empty" if not cell.get("nuclei") else
                          "binucleate" if len(cell["nuclei"]) == 2 else "ordinary",
                          anchor_z=anchor[0], anchor_y=anchor[1], anchor_x=anchor[2], reach=cell.get("reach", 1.0)))
    growth = [cell.get("growth", np.nan) for cell in case["cells"]]
    stains = [dict(name=name, source="boundary", edge_width=0.5, band_width=b, nuclear_ratio=0.3, unevenness=None)
              for name, b in BANDS.items()]
    objects = pd.DataFrame([(name, "cell", k, 1000.0) for name in BANDS for k in range(1, len(cells) + 1)],
                           columns=["stain", "object", "object_id", "plateau"])
    lo, hi = case["omega"]
    sample = type("Sample", (), {})()
    sample.cells = pd.DataFrame(cells)
    sample.nuclei = pd.DataFrame(nuclei, columns=["nucleus_id", "cell_id", "centre_z", "centre_y", "centre_x",
                                                  "axial_radius", "lateral_radius", "elongation", "angle",
                                                  "site_radius"])
    sample.record = dict(domain=dict(omega=dict(lo_zyx=list(lo), hi_zyx=list(hi))),
                         requested_config=dict(cells=dict(empty_radius=2.5)))
    hand = type("Scene", (), {})()
    hand.sample = sample
    hand.lobes = pd.DataFrame(lobes, columns=["nucleus_id", "lobe", "offset_z", "offset_y", "offset_x", "axial_radius",
                                              "lateral_radius", "elongation", "angle"])
    hand.harmonics = pd.DataFrame(columns=["nucleus_id", "degree", "order", "coefficient"])
    hand.cell_context = pd.DataFrame(dict(cell_id=range(1, len(cells) + 1), growth=growth))
    hand.record = dict(effective_config=dict(planar=case.get("planar", False), territory_rule=case["rule"],
                                             harmonic_terms=[], empty_radius=2.5),
                       requested_config=dict(stains=stains))
    hand.stain_objects = objects
    hand.bright_regions = pd.DataFrame(columns=["region_id", "nucleus_id", "stain", "centre_z", "centre_y", "centre_x",
                                                "size", "height"])
    hand.unevenness = pd.DataFrame(columns=["stain", "bump_id", "centre_z", "centre_y", "centre_x", "z", "a",
                                            "length"])
    return hand


SPHERE = lambda r: (r, r, 1.0, 0.0)  # noqa: E731 (axial, lateral, elongation, angle)
BAND_CASES = {
    # The reviewer's binucleate cell: spheres of radii 1 and 2 µm at (0, 0, 0) and (0, 0, 2.7), reach 1 (T1).
    "reviewer": dict(rule="reach", omega=((-5.0, -5.0, -5.0), (5.0, 5.0, 8.0)), cells=[
        dict(nuclei=[dict(centre=(0, 0, 0), semi=SPHERE(1.0)), dict(centre=(0, 0, 2.7), semi=SPHERE(2.0))],
             reach=1.0)]),
    # Binucleate cells with sites of unequal radii, touching cells of different owners, an empty cell (T1).
    "unequal": dict(rule="reach", omega=((-4.0, -8.0, -7.0), (4.0, 8.0, 9.0)), cells=[
        dict(nuclei=[dict(centre=(0, 0, 0), semi=SPHERE(1.5)), dict(centre=(0.2, 0.1, 3.4), semi=SPHERE(2.2))],
             reach=0.8),
        dict(nuclei=[dict(centre=(-0.3, 3.0, 1.2), semi=(1.4, 1.6, 1.15, 0.4))], reach=0.9),
        dict(nuclei=[dict(centre=(0.4, -3.4, 2.4), semi=(1.3, 1.5, 1.2, 2.2)),
                     dict(centre=(0.1, -3.1, 5.9), semi=SPHERE(1.1))], reach=0.6),
        dict(anchor=(0.0, 4.6, -4.2), reach=1.0)]),
    # Lobed nuclei under T2: a binucleate cell with a lobed nucleus, a touching lobed neighbour, an empty cell.
    "lobed": dict(rule="grown", omega=((-4.0, -8.0, -6.0), (4.0, 8.0, 9.0)), cells=[
        dict(nuclei=[dict(centre=(0, 0, 0), lobes=[((0, 0, 0), 1.5, 2.0, 1.2, 0.3),
                                                    ((0.3, 1.2, 0.8), 0.9, 1.2, 1.2, 1.4)]),
                     dict(centre=(0.2, 0.5, 4.2), semi=(1.4, 1.8, 1.1, 1.0))], growth=0.8),
        dict(nuclei=[dict(centre=(-0.3, 4.1, 1.0), lobes=[((0, 0, 0), 1.3, 1.7, 1.3, 2.0),
                                                           ((0.0, 0.9, -1.1), 0.8, 1.0, 1.3, 0.2)])], growth=0.5),
        dict(anchor=(0.0, -4.6, 3.0), growth=0.7)]),
    # Z=1: the plane z = 0, a binucleate lobed cell, touching cells, T1.
    "plane": dict(rule="reach", planar=True, omega=((-0.5, -6.0, -6.0), (0.5, 8.0, 10.0)), cells=[
        dict(nuclei=[dict(centre=(0, 0, 0), lobes=[((0, 0, 0), 2.0, 1.8, 1.3, 0.5),
                                                    ((0.0, 1.1, 1.0), 2.0, 1.1, 1.3, 1.9)]),
                     dict(centre=(0.0, -0.6, 3.6), semi=(2.0, 1.5, 1.1, 0.0))], reach=1.0),
        dict(nuclei=[dict(centre=(0.0, 3.6, 1.4), semi=(2.0, 1.6, 1.2, 2.5))], reach=1.2),
        dict(anchor=(0.0, -3.5, -3.0), reach=0.8)]),
}


def cell_label(generated, points):
    return _context_labels(generated, points)[0].astype(np.int64)


def inside(generated, points):
    omega = generated.sample.record["domain"]["omega"]
    return np.all((points >= np.array(omega["lo_zyx"])) & (points < np.array(omega["hi_zyx"])), axis=-1)


def bisect(generated, a, b, tolerance=1e-9):
    """Bisect pairs with C(a) ≠ C(b) until ‖a − b‖ ≤ tolerance (µm); the labels at the two ends stay different."""
    a, b = a.copy(), b.copy()
    la, lb = cell_label(generated, a), cell_label(generated, b)
    assert np.all(la != lb)
    for _ in range(80):
        open_ = np.linalg.norm(a - b, axis=1) > tolerance
        if not open_.any():
            break
        middle = (a[open_] + b[open_]) / 2
        lm = cell_label(generated, middle)
        same = lm == la[open_]
        index = np.flatnonzero(open_)
        a[index[same]] = middle[same]
        b[index[~same]], lb[index[~same]] = middle[~same], lm[~same]
    assert np.all(np.linalg.norm(a - b, axis=1) <= tolerance) and np.all(la != lb)
    return a, b, la, lb


def boundary_pairs(generated, rng, planar, count=600, grid=None):
    """Pairs with different cell labels: neighbouring voxel centres of a grid, random pairs 0.01 to 0.5 µm apart, and
    pairs across the internal face of each binucleate cell; every end inside Ω."""
    A, B = [], []
    if grid is not None:
        labels = cell_label(generated, grid)
        for axis in ((1, 2) if planar else (0, 1, 2)):
            x, p = np.moveaxis(labels, axis, 0), np.moveaxis(grid, axis, 0)
            change = x[:-1] != x[1:]
            A.append(p[:-1][change])
            B.append(p[1:][change])
    omega = generated.sample.record["domain"]["omega"]
    lo, hi = np.array(omega["lo_zyx"]), np.array(omega["hi_zyx"])
    for _ in range(50):
        p = rng.uniform(lo, hi, (20_000, 3))
        v = rng.standard_normal((20_000, 3))
        if planar:
            p[:, 0], v[:, 0] = 0.0, 0.0
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        q = p + rng.uniform(0.01, 0.5, (20_000, 1)) * v
        keep = inside(generated, q) & inside(generated, p)
        p, q = p[keep], q[keep]
        change = cell_label(generated, p) != cell_label(generated, q)
        A.append(p[change][:count])
        B.append(q[change][:count])
        if sum(len(x) for x in A) >= 2 * count:
            break
    nuclei = generated.sample.nuclei
    for k, group in nuclei.groupby("cell_id"):
        if len(group) != 2:
            continue
        c = group[["centre_z", "centre_y", "centre_x"]].to_numpy()
        r = group.site_radius.to_numpy()
        gap = np.linalg.norm(c[1] - c[0])
        normal = (c[1] - c[0]) / gap
        foot = c[0] + (gap ** 2 + r[0] ** 2 - r[1] ** 2) / (2 * gap) * normal
        spread = r.sum() + 3.0
        p = foot + rng.uniform(-spread, spread, (8000, 3))
        if planar:
            p[:, 0] = foot[0]
        p -= ((p - foot) @ normal)[:, None] * normal
        eps = rng.uniform(1e-4, 0.05, (8000, 1))
        a, b = p - eps * normal, p + eps * normal
        keep = inside(generated, a) & inside(generated, b)
        a, b = a[keep], b[keep]
        change = cell_label(generated, a) != cell_label(generated, b)
        A.append(a[change][:count // 2])
        B.append(b[change][:count // 2])
    return np.concatenate(A), np.concatenate(B)


def check_bands(name, generated, rng, planar, grid=None):
    """GN14 on one scene: the band at every bisected boundary (b 0.01 and 0.5 µm), the continuity of β_k off the
    internal face, and the converse. Returns the counts it reports."""
    a, b = boundary_pairs(generated, rng, planar, grid=grid)
    keep = rng.permutation(len(a))[:1500]
    a, b, la, lb = bisect(generated, a[keep], b[keep])
    distance = np.linalg.norm(a - b, axis=1)
    sites = generated.sample.nuclei.groupby("cell_id").size().to_dict()
    report = dict(points=len(a), internal=0, between_cells=int(((la > 0) & (lb > 0)).sum()),
                  cell_and_none=int(((la == 0) | (lb == 0)).sum()))
    lowest = {stain: 1.0 for stain in BANDS}
    for k in sorted(set(la.tolist()) | set(lb.tolist())):
        if not k:
            continue
        own = (la == k) | (lb == k)
        bound = lipschitz(generated, k) * distance[own] + 1e-9
        for stain, width in BANDS.items():
            for end in (a, b):
                _, beta = _context_band(generated, stain, k, end[own])
                assert np.all(np.isfinite(beta) & (beta <= bound)), (name, stain, k)
                lowest[stain] = min(lowest[stain], float(np.exp(-.5 * (beta / width) ** 2).min()))
    for k, count in sites.items():
        if count == 2:
            nuclei = generated.sample.nuclei[generated.sample.nuclei.cell_id == k]
            c = nuclei[["centre_z", "centre_y", "centre_x"]].to_numpy()
            r = nuclei.site_radius.to_numpy()
            power = lambda p: (((p - c[0]) ** 2).sum(1) - r[0] ** 2) - (((p - c[1]) ** 2).sum(1) - r[1] ** 2)  # noqa
            report["internal"] += int(((la == k) | (lb == k))[np.sign(power(a)) != np.sign(power(b))].sum())
    # Continuity of β_k off the internal face: steps of 1e-3 and 1e-6 µm in random directions.
    worst = 0.0
    for k in generated.sample.cells.cell_id.astype(int):
        anchor = generated.sample.cells.set_index("cell_id").loc[k, ["anchor_z", "anchor_y", "anchor_x"]]
        anchor = anchor.to_numpy(float)
        p = anchor + rng.uniform(-6, 6, (2000, 3))
        if planar:
            p[:, 0] = anchor[0]
        p = p[inside(generated, p)]
        for size in (1e-3, 1e-6):
            v = rng.standard_normal(p.shape)
            if planar:
                v[:, 0] = 0.0
            q = p + size * v / np.linalg.norm(v, axis=1, keepdims=True)
            beta_p = _context_band(generated, "membrane", k, p)[1]
            beta_q = _context_band(generated, "membrane", k, q)[1]
            usable = np.isfinite(beta_p) & np.isfinite(beta_q)
            nuclei = generated.sample.nuclei[generated.sample.nuclei.cell_id == k]
            if len(nuclei) == 2:
                c = nuclei[["centre_z", "centre_y", "centre_x"]].to_numpy()
                r = nuclei.site_radius.to_numpy()
                side = lambda x: np.sign((((x - c[0]) ** 2).sum(1) - r[0] ** 2)  # noqa: E731
                                         - (((x - c[1]) ** 2).sum(1) - r[1] ** 2))
                usable &= side(p) == side(q)
            change = np.abs(beta_p - beta_q)[usable]
            limit = lipschitz(generated, k) * np.linalg.norm(p - q, axis=1)[usable] + 1e-9
            assert np.all(change <= limit), (name, k, size)
            if len(change):
                worst = max(worst, float((change / limit).max()))
    # The converse: positions where the internal-face term alone lights the band (g_k < 0.05 µm ≤ abs(d_k)) have a
    # boundary of C = k within 4 · 0.05 µm · F (audit D's search; F the T2 scale of the outer terms).
    lonely, candidates = 0, 0
    directions = rng.standard_normal((200, 3))
    if planar:
        directions[:, 0] = 0.0
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for k, count in sites.items():
        if count != 2:
            continue
        nuclei = generated.sample.nuclei[generated.sample.nuclei.cell_id == k]
        c = nuclei[["centre_z", "centre_y", "centre_x"]].to_numpy()
        r = nuclei.site_radius.to_numpy()
        gap = np.linalg.norm(c[1] - c[0])
        normal = (c[1] - c[0]) / gap
        foot = c[0] + (gap ** 2 + r[0] ** 2 - r[1] ** 2) / (2 * gap) * normal
        p = foot + rng.uniform(-(r.sum() + 2), r.sum() + 2, (20_000, 3))
        if planar:
            p[:, 0] = foot[0]
        p -= ((p - foot) @ normal)[:, None] * normal
        p += rng.uniform(-0.04, 0.04, (len(p), 1)) * normal
        p = p[inside(generated, p)]
        d, beta = _context_band(generated, "membrane", k, p)
        chosen = p[(beta < 0.05) & (np.abs(d) >= 0.05)][:60]
        factor = 1.0
        if generated.record["effective_config"]["territory_rule"] == "grown":
            lobes = generated.lobes[generated.lobes.nucleus_id.isin(nuclei.nucleus_id)]
            for n, own in lobes.groupby("nucleus_id"):
                first = own[own.lobe == 1].iloc[0]
                rbar = (first.axial_radius * first.lateral_radius ** 2 * first.elongation) ** (1 / 3)
                growth = float(generated.cell_context.set_index("cell_id").growth.loc[k])
                semi = np.maximum(own.axial_radius, own.lateral_radius * own.elongation)
                factor = max(factor, float((semi + growth).max()) / rbar)
        radii = np.linspace(0.0, 4 * 0.05 * factor, 41)[1:]
        probes = (radii[:, None, None] * directions[None]).reshape(-1, 3)
        for q in chosen:
            here = cell_label(generated, q[None])[0] == k
            there = cell_label(generated, q + probes) == k
            lonely += int(not np.any(there != here))
        candidates += len(chosen)
    report.update(lowest_band_over_peak=lowest, continuity_worst_ratio=worst, converse_candidates=candidates,
                  converse_lonely=lonely)
    return report


@pytest.mark.parametrize("case", list(BAND_CASES))
def test_gn14_band_cases(case):
    hand = _hand_scene(BAND_CASES[case])
    planar = BAND_CASES[case].get("planar", False)
    # Each anchor lies in its own cell (the hand-written tables are a valid scene).
    anchors = hand.sample.cells[["anchor_z", "anchor_y", "anchor_x"]].to_numpy(float)
    assert cell_label(hand, anchors).tolist() == hand.sample.cells.cell_id.tolist()
    rng = np.random.default_rng(484)
    s = np.array([1.0, 0.194398, 0.194398]) if planar else np.array([0.346255, 0.194398, 0.194398])
    shape = (1, 64, 64) if planar else (16, 40, 40)
    true = np.array([0.0, 0.43, 0.61]) * s + (0.0, -5.0, -4.0) if planar else \
        np.array([0.37, 0.43, 0.61]) * s + (-2.5, -3.0, -1.5)
    grid = true + np.moveaxis(np.indices(shape, dtype=float), 0, -1) * s
    report = check_bands(case, hand, rng, planar, grid=grid)
    if case == "reviewer":
        x = 4.29 / 5.4
        p = np.array([[0.0, 2.0, x - 1e-6], [0.0, 2.0, x + 1e-6]])
        assert cell_label(hand, p).tolist() == [0, 1]
        beta = _context_band(hand, "membrane", 1, p)[1]
        assert np.all(beta <= 1e-6 + 1e-9), beta
        print(f"GN14 the reviewer's points (0, 2, {x:.10f} ∓ 1e-6): C = [0, 1], β_1 = {beta.tolist()}")
    print(f"GN14 band_cases {case}: {report}")
    assert report["points"] >= 300 and report["converse_lonely"] == 0
    if case in ("reviewer", "unequal", "lobed", "plane"):
        assert report["internal"] > 0
    assert min(report["lowest_band_over_peak"].values()) > 1 - 1e-12


@pytest.mark.parametrize("name", WHOLE_SCENES)
def test_gn14_band_on_ctx_stains(name):
    _, config, generated = scene(name)
    planar = config.sample.acquisition.fov_shape_zyx[0] == 1
    true, spacing, shape = frame(generated, "FOV_001")
    grid = true + np.moveaxis(np.indices(shape, dtype=float), 0, -1) * spacing
    report = check_bands(name, generated, np.random.default_rng(42), planar, grid=grid)
    print(f"GN14 {name}: {report}")
    assert report["points"] >= 300 and report["converse_lonely"] == 0
    # Under T3 both sides of a binucleate cell's internal face lie in the cell, so no label boundary lies on it.
    assert (report["internal"] > 0) == (config.territories.rule != "power")
    assert min(report["lowest_band_over_peak"].values()) > 1 - 1e-12


# ---------------------------------------------------------------- GN15: bright regions

@pytest.mark.parametrize("name", [f"stains-{rule}-whole" for rule in RULES] + ["plane-whole", "mr1"])
def test_gn15_bright_regions(name):
    _, config, generated = scene(name)
    regions = generated.bright_regions
    textured = {stain.name for stain in config.stains if stain.source == "nucleus" and stain.texture is not None}
    assert len(regions) and set(regions.stain) <= textured
    assert (regions["size"] > 0).all() and (regions.height > 0).all()
    centres = regions[["centre_z", "centre_y", "centre_x"]].to_numpy(float)
    _, nucleus = _context_labels(generated, centres)
    assert nucleus.astype(int).tolist() == regions.nucleus_id.astype(int).tolist()
    _, by_oracle, near = Oracle(generated).labels(centres)
    assert np.all((by_oracle == regions.nucleus_id.to_numpy()) | near)
    counts = regions.groupby("nucleus_id").size()
    print(f"GN15 {name}: {len(regions)} regions in {len(counts)} of {len(generated.sample.nuclei)} nuclei; centres "
          f"within 1e-9 µm of a boundary {int(near.sum())} (expected 0); sizes {regions['size'].min():.3f} to "
          f"{regions['size'].max():.3f} µm, heights {regions.height.min():.3f} to {regions.height.max():.3f}")
    assert not near.any()


# ---------------------------------------------------------------- GN16: the field

def test_gn16_the_field_between_fovs_and_without_centres():
    _, config, generated = scene("stains-pair")
    oracle = Oracle(generated)
    worst = 0.0
    for stain in {s.name: s for s in config.stains}.values():
        rows = generated.unevenness[generated.unevenness.stain == stain.name]
        assert len(rows) and np.all(np.abs(rows.a.to_numpy() - FIELD.amplitude) <= 1e-9)
        np.testing.assert_allclose(rows.length, 18.0, rtol=0, atol=1e-12)  # one FOV width, 36 × 0.5 µm
        so = StainOracle(oracle, generated, stain.name)
        true_1, spacing, shape = frame(generated, "FOV_001")
        true_2 = frame(generated, "FOV_002")[0]
        q = np.moveaxis(np.indices((shape[0], shape[1] - 23, shape[2] - 2), dtype=float), 0, -1)
        a = (true_1 + (q + (0, 23, 2)) * spacing).reshape(-1, 3)
        b = (true_2 + q * spacing).reshape(-1, 3)
        ratio = so.field(a) / so.field(b)
        worst = max(worst, float(np.abs(ratio - 1).max()))
        assert np.all(np.abs(ratio - 1) <= 1e-12)
        evaluated = _context_field(generated, stain.name, a) / _context_field(generated, stain.name, b)
        assert np.all(np.abs(evaluated - 1) <= 1e-12)
    _, flat_config, flat = scene("stains-flat-field")
    assert flat.unevenness.empty
    points = frame(flat, "FOV_001")[0] + np.random.default_rng(1).uniform(-2, 20, (500, 3))
    for stain in flat_config.stains:
        assert np.array_equal(_context_field(flat, stain.name, points), np.ones(500))
        assert np.array_equal(StainOracle(Oracle(flat), flat, stain.name).field(points), np.ones(500))
    print(f"GN16 the field at the common voxels of FOV_001 and FOV_002: largest |ratio − 1| {worst:.2e} (≤ 1e-12); "
          f"with no centre U = 1 exactly")


# ---------------------------------------------------------------- GN17: 8-bit output

def check_8bit(generated, config, reference, reference_config):
    """Identity 10 on one scene: each stain's k against g / m_A of its configuration (within 1e-9 relative; m_A
    = exp(μ) of the law, compared within 1e-12 relative), and each 8-bit image exactly equal to clip(rint(J · k), 0,
    255) of its float64 rendering J with the same streams; a stain without an output scale is the float32 cast of J.
    Returns (voxels clipped at 255, voxels compared)."""
    table = generated.stain_channels
    assert len(table) == len(config.stains)
    for stain, row in zip(config.stains, table.itertuples()):
        assert (row.stain, row.round) == (stain.name, stain.round)
        median = math.exp(stain.brightness.parameters[0])
        assert row.m_A == pytest.approx(median, rel=1e-12, abs=0)
        if stain.output is None:
            assert row.dtype == "float32" and row.scale == 1.0
        else:
            # k exists and equals g / m_A, also for a FOV or a scene without a nucleus (B1).
            assert row.dtype == "uint8" and row.scale == pytest.approx(stain.output.plateau / median, rel=1e-9, abs=0)
    clipped = voxels = 0
    for fov in generated.sample.positions.fov_id:
        rendered = {(s.name, s.round): j for _, s, j, _, _ in channels(reference, reference_config, fov)}
        for _, stain, image, _, _ in channels(generated, config, fov):
            j = rendered[stain.name, stain.round]
            assert j.dtype == np.float64
            if stain.output is None:
                # Without an output scale the stain is cast to the template dtype (float32) as today.
                assert image.dtype == np.float32 and np.array_equal(image, j.astype(np.float32))
                continue
            scale = float(table[(table.stain == stain.name) & (table["round"] == stain.round)].scale.iloc[0])
            expected = np.clip(np.rint(j * scale), 0, 255).astype(np.uint8)
            assert image.dtype == np.uint8 and np.array_equal(image, expected), (fov, stain.name)
            clipped += int((j * scale > 255).sum())
            voxels += image.size
    return clipped, voxels


def check_round_trip(generated, root):
    """The saved and reloaded stain images (float32 and uint8) and morphology images: dtype and values exact."""
    for fov in generated.sample.positions.fov_id:
        for stain_name, image in generated.stains[fov].items():
            loaded = load_volume_zyxc(root / "fovs" / fov / "stains" / f"{stain_name}.ome.tif")
            assert loaded.image.dtype == image.dtype and np.array_equal(loaded.image[..., 0], image)
            assert loaded.metadata == generated.sample.fov_metadata[fov]
        loaded = load_volume_zyxc(root / "fovs" / fov / "morphology.ome.tif")
        assert loaded.image.dtype == generated.morphology[fov].dtype
        assert np.array_equal(loaded.image, generated.morphology[fov])


@pytest.mark.parametrize("pair", [("uint8", "uint8-float"), ("plane-uint8", "plane-uint8-float")],
                         ids=["tiny", "plane"])
def test_gn17_8bit_output_and_the_dtypes(pair, tmp_path):
    _, config, generated = scene(pair[0])
    _, reference_config, reference = scene(pair[1])
    clipped, voxels = check_8bit(generated, config, reference, reference_config)
    check_round_trip(generated, save_context_scene(generated, tmp_path / "saved"))
    report = f"GN17 {pair[0]}: voxels clipped at 255 {clipped} of {voxels}"
    if pair[0] == "uint8":
        # ctx_uint8's no-cell copy: the same identity against its own float64 rendering, and its round trip.
        _, empty_config, empty = scene("uint8-empty")
        _, empty_reference_config, empty_reference = scene("uint8-empty-float")
        assert empty.sample.nuclei.empty and empty.stain_objects.empty
        empty_clipped, empty_voxels = check_8bit(empty, empty_config, empty_reference, empty_reference_config)
        check_round_trip(empty, save_context_scene(empty, tmp_path / "empty"))
        report += f"; the no-cell copy: {empty_voxels} 8-bit voxels, {empty_clipped} clipped"
    print(report + "; the 8-bit images equal clip(rint(J · k), 0, 255) of the float64 rendering; float32 and uint8 "
          "images round trip unchanged")


# ---------------------------------------------------------------- GN18: the morphology round

@pytest.mark.parametrize("name", [f"stains-{rule}-{mode}" for rule in RULES for mode in ("whole", "drawn")]
                         + ["plane-whole", "plane-drawn", "mr1"])
def test_gn18_truth_translation(name):
    _, config, generated = scene(name)
    truth = generated.morphology_truth.set_index("fov_id")
    spacing = np.array(config.sample.acquisition.spacing_zyx)
    bound = np.array(config.morphology.translation_max_zyx)
    assert list(truth.index) == list(generated.sample.positions.fov_id) and (truth["round"] == "organelle").all()
    vox = truth[["translation_z_vox", "translation_y_vox", "translation_x_vox"]].to_numpy()
    um = truth[["translation_z_um", "translation_y_um", "translation_x_um"]].to_numpy()
    pull = truth[["pull_displacement_z_vox", "pull_displacement_y_vox", "pull_displacement_x_vox"]].to_numpy()
    assert np.all(np.abs(um / spacing - vox) <= 1e-9) and np.all(np.abs(pull + vox) <= 1e-9)
    assert np.all(np.abs(um) <= bound * spacing)
    if config.morphology.translation == "supplied":
        assert np.all(np.abs(vox - np.tile(WHOLE, (len(vox), 1))) <= 1e-9)
    else:
        assert np.any(np.abs(vox - np.round(vox)) > 1e-3)  # MT1 draws a fractional translation
    if config.sample.acquisition.fov_shape_zyx[0] == 1:
        assert np.all(vox[:, 0] == 0) and np.all(um[:, 0] == 0)
    drawn = {d[4] for d in generated.record["stream_scheme"]["streams"]}
    assert ("context.morphology.translation" in drawn) == (config.morphology.translation == "uniform")
    print(f"GN18 {name}: translations (voxels) {vox.round(4).tolist()}")


@pytest.mark.parametrize("name", WHOLE_SCENES)
def test_gn18_whole_voxel_shift_identity(name):
    """With τ ⊘ s = (0, 2, −3) voxels, each morphology stain equals the reference stain of the same name shifted by
    it on the common voxels within 1e-10 of the maximum (identity 8), except at boundary voxels (reported)."""
    _, config, generated = scene(name)
    oracle = Oracle(generated)
    worst, boundary_total, compared = 0.0, 0, 0
    for fov in generated.sample.positions.fov_id:
        found = {(s.name, s.round): (image, points) for _, s, image, _, points in channels(generated, config, fov)}
        for (stain_name, round_), (moved, moved_points) in found.items():
            if round_ != "morphology":
                continue
            image, points = found[stain_name, "reference"]
            ny, nx = image.shape[1:]
            # Morphology index q images true + τ + q ⊙ s = the reference position at q + (0, 2, −3).
            a, pa = image[:, 2:, :nx - 3], points[:, 2:, :nx - 3]
            b, pb = moved[:, :ny - 2, 3:], moved_points[:, :ny - 2, 3:]
            assert np.abs(pa - pb).max() < 1e-12
            so = StainOracle(oracle, generated, stain_name)
            near = so.signal(pa)[1] | so.signal(pb)[1] | oracle.labels(pa)[2] | oracle.labels(pb)[2]
            peak = max(float(np.abs(image).max()), float(np.abs(moved).max()))
            error = np.abs(a - b) / peak
            assert np.all((error <= 1e-10) | near), (fov, stain_name)
            worst = max(worst, float(error[~near].max()))
            boundary_total += int(near.sum())
            compared += a.size
    print(f"GN18 {name}: morphology stains against the reference stains shifted by (0, 2, −3): {compared} voxels; "
          f"largest difference {worst:.2e} of the maximum (≤ 1e-10); boundary voxels {boundary_total} (expected 0)")
    assert boundary_total == 0 and compared > 0


def test_gn18_domain_grown_covered_region_unchanged_and_the_aligned_crop():
    _, config, grown = scene("stains-reach-whole")
    _, _, plain = scene("stains-off")
    spacing = np.array(config.sample.acquisition.spacing_zyx)
    bound = np.array(config.morphology.translation_max_zyx)
    a, b = grown.sample.record["domain"], plain.sample.record["domain"]
    assert np.allclose(np.array(a["omega"]["lo_zyx"]), np.array(b["omega"]["lo_zyx"]) - bound * spacing,
                       rtol=0, atol=1e-9)
    assert np.allclose(np.array(a["omega"]["hi_zyx"]), np.array(b["omega"]["hi_zyx"]) + bound * spacing,
                       rtol=0, atol=1e-9)
    assert sorted(a["covered"]) == sorted(b["covered"])
    for key in a["covered"]:
        assert np.allclose(np.array(a["covered"][key], dtype=float), np.array(b["covered"][key], dtype=float),
                           rtol=0, atol=1e-9), key  # K unchanged
    origin = np.array(grown.sample.record["sample_grid"]["origin_zyx"])
    jitter = np.array(config.sample.acquisition.jitter_max_zyx)
    assert np.allclose(origin, -(jitter + bound) * spacing, rtol=0, atol=1e-12)
    # The aligned crop with T: δ_f = nominal_f ⊘ s + j_f + J + T, whole for FOV_001 and FOV_002 of the pair.
    _, pair_config, pair = scene("stains-pair")
    oracle = Oracle(pair)
    boundary_total = 0
    for fov, expected in (("FOV_001", (1, 8, 6)), ("FOV_002", (1, 31, 8))):
        delta = np.array(pair.sample.record["fovs"][fov]["delta_zyx"])
        row = pair.sample.placement.set_index("fov_id").loc[fov]
        nominal = np.array([row.nominal_z, row.nominal_y, row.nominal_x]) / spacing
        assert np.allclose(delta, nominal + np.array(PAIR[fov]) + jitter + bound, rtol=0, atol=1e-9)
        assert np.allclose(delta, expected, rtol=0, atol=1e-9)
        z, y, x = expected
        points = frame(pair, fov)[0] + np.moveaxis(np.indices((5, 36, 36), dtype=float), 0, -1) * spacing
        near = oracle.labels(points)[2]
        for target in ("cell", "nucleus"):
            crop = pair.sample.sample_labels[target][z:z + 5, y:y + 36, x:x + 36]
            assert np.all((pair.sample.fov_labels[fov][target] == crop) | near), (fov, target)
        boundary_total += int(near.sum())
    print(f"GN18 Ω grown by T ⊙ s = {(bound * spacing).tolist()} µm on each side, K unchanged; aligned crops at δ "
          f"(1, 8, 6) and (1, 31, 8): boundary voxels {boundary_total} (expected 0)")
    assert boundary_total == 0


# ---------------------------------------------------------------- GN20: Z=1

@pytest.mark.parametrize("name", ["plane-whole", "plane-drawn"])
def test_gn20_one_plane(name, tmp_path):
    book, config, generated = scene(name)
    sample = generated.sample
    shape = (1, 36, 36)
    for fov in sample.positions.fov_id:
        assert all(image.shape[:3] == shape for image in sample.fovs[fov].rounds.values())
        assert all(image.shape == shape for image in generated.stains[fov].values())
        assert generated.morphology[fov].shape == shape + (2,)
        assert all(labels.shape == shape for labels in sample.fov_labels[fov].values())
    assert all(labels.shape[0] == 1 for labels in sample.sample_labels.values())
    zero = lambda values: np.all(np.asarray(values, dtype=float) == 0)  # noqa: E731
    assert zero(sample.nuclei.centre_z) and zero(generated.lobes.offset_z) and zero(sample.cells.anchor_z)
    assert zero(sample.molecules.sample_z) and zero(generated.bright_regions.centre_z)
    assert zero(generated.unevenness.centre_z) and len(generated.unevenness)
    assert zero(generated.morphology_truth.translation_z_um) and zero(generated.morphology_truth.translation_z_vox)
    # GN4 and GN6 on this fixture: labels against the oracle, a nucleus inside its own cell.
    oracle = Oracle(generated)
    owner = dict(zip(sample.nuclei.nucleus_id.astype(int), sample.nuclei.cell_id.astype(int)))
    boundary_total = 0
    for fov in sample.positions.fov_id:
        true, spacing, fov_shape = frame(generated, fov)
        points = true + np.moveaxis(np.indices(fov_shape, dtype=float), 0, -1) * spacing
        cell, nucleus, near = oracle.labels(points)
        labels = sample.fov_labels[fov]
        assert np.all((labels["cell"] == cell) | near) and np.all((labels["nucleus"] == nucleus) | near)
        inner = labels["nucleus"] > 0
        assert all(owner[int(n)] == int(c) for n, c in zip(labels["nucleus"][inner], labels["cell"][inner]))
        boundary_total += int(near.sum())
    # The saved labels are read as plane label images.
    root = save_context_scene(generated, tmp_path / "plane")
    for fov in sample.positions.fov_id:
        grid = ReferenceGrid(shape, sample.fov_metadata[fov], "declared")
        for target, file in (("cell", "cells.tif"), ("nucleus", "nuclei.tif")):
            namespace = json.dumps([config.sample.fov.dataset_version, config.sample.fov.sample_id, fov, None,
                                    "truth"])
            imported = import_labels(root / "fovs" / fov / "labels" / file, grid=grid, target=target,
                                     label_namespace=namespace)
            assert imported.record["import"]["geometry"] == "plane"
    print(f"GN20 {name}: every array has one plane, every z is 0, τ_z = 0; labels against the oracle: boundary "
          f"voxels {boundary_total} (expected 0); saved labels imported as plane label images")
    assert boundary_total == 0


# ---------------------------------------------------------------- GN21: the stain components

def test_gn21_stain_components_and_streams():
    _, config, generated = scene("uint8")
    record = generated.record["stream_scheme"]
    drawn = {descriptor[4] for descriptor in record["streams"]}
    noise = {f"stain.noise.{n}" for n in ("dependent", "independent", "correlated")}
    assert drawn - noise <= set(record["components"]) and all(c.startswith("context.") for c in drawn - noise)
    assert {"context.stain.brightness", "context.texture.count", "context.texture.placement", "context.texture.size",
            "context.texture.height", "context.unevenness.count", "context.unevenness.placement",
            "context.unevenness.amplitude"} <= drawn
    # A reference-round DAPI draws its noise from the sample scene's stain streams; the morphology round's DAPI
    # differs by its round label.
    book, sample_config = tiny()
    sample = generate_sample_scene(book, config=sample_config)
    sample_noise = {tuple(map(str, d)) for d in sample.record["stream_scheme"]["streams"] if d[4] in noise}
    context_noise = [d for d in record["streams"] if d[4] in noise]
    reference = {tuple(map(str, d)) for d in context_noise if d[7] == "DAPI" and d[6] is None}
    assert reference and reference == sample_noise
    rounds = {(d[6], d[7]) for d in context_noise}
    assert rounds == {(None, "DAPI"), (None, "Flamingo"), (None, "membrane"), (None, "membrane-thin"),
                      ("organelle", "DAPI"), ("organelle", "Flamingo")}
    # Adding a stain (first in the order) changes no other stain.
    _, _, base = scene("stains-reach-whole")
    _, _, extra = scene("stains-extra")
    for fov in base.sample.positions.fov_id:
        for stain_name, image in base.stains[fov].items():
            assert np.array_equal(extra.stains[fov][stain_name], image), (fov, stain_name)
        assert np.array_equal(extra.morphology[fov], base.morphology[fov])
    for table in ("stain_objects", "bright_regions", "unevenness"):
        own = getattr(extra, table)
        pd.testing.assert_frame_equal(own[own.stain != "PI"].reset_index(drop=True), getattr(base, table),
                                      check_exact=True)
    pd.testing.assert_frame_equal(extra.morphology_truth, base.morphology_truth, check_exact=True)
    print(f"GN21 stain components drawn: {sorted(drawn)}; the reference-round DAPI's noise descriptors equal the "
          f"sample scene's ({len(reference)}); one more stain changes no other stain")


def test_stain_configuration_record():
    """The requested stains and morphology round are recorded in context.json, with the effective background, noise
    and B1 scale of each channel and the morphology round's stains."""
    _, config, generated = scene("stains-reach-drawn")
    requested = generated.record["requested_config"]
    assert [s["name"] for s in requested["stains"]] == [s.name for s in config.stains]
    assert requested["morphology"]["type"] == "MorphologyRoundConfig"
    effective = generated.record["effective_config"]
    assert [s["channel"] for s in effective["stains"]] == [s.channel for s in config.stains]
    assert all(set(s) >= {"background", "noise", "scale", "m_A", "dtype"} for s in effective["stains"])
    assert effective["morphology"]["stains"] == ["DAPI", "Flamingo"]
