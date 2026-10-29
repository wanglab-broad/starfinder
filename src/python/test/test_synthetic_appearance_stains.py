"""Generator checks of the appearance scene's composition, output, morphology round and molecules
(design/validation/synthetic-real-appearance-scene.md, "Generator checks").

GA12 (composition), GA13 (out-of-focus light and the rim background), GA14 (brightness within a scene: the scene's
part), GA15 (8-bit output), GA16 (the morphology round), GA17 (molecules and types), the stain parts of GA19 (``Z=1``;
the saved labels read as plane label images) and GA20 (the streams; a dropped stain), and GA10's DN1 part, on the
generated fixtures of
``cellular_appearance_scenes.py`` (``app_scene``), recomputed by the test oracle's composition part
(``cellular_appearance_oracle.py``). GA12's and GA16's tolerance, 1e-9 of the image maximum, was evaluated on these
fixtures before these tests assert it (W-518's ``tolerance_check.py`` in its run directory: worst case 3.3e-12).
Boundary voxels (an interpolated ``D_e`` within 1e-9 µm of 0) are counted, printed and expected to be 0. Every bank is
a closed-form bank or is built in the session from hand-made arrays; nothing is read from a run directory or a real
image. A test whose run alone measures 5 s or more on one CPU (the dense 3D closed-form bank alone takes 3.5 s to
build) is ``slow``; each check keeps a routine copy, except the dropped stain of GA20, which needs two culture scenes.
"""
from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from starfinder.report._measurements import mecr
from starfinder.segmentation import ReferenceGrid, import_labels
from starfinder.synthetic import (BankReference, build_appearance_bank, generate_appearance_scene,
                                  load_appearance_bank, save_appearance_scene)
from starfinder.synthetic._appearance import _Canvas, _Composer, _point_evaluator
from starfinder.synthetic._appearance_layout import _labels_from_table
from starfinder.synthetic._context import _CONTEXT_COMPONENTS
from starfinder.synthetic._formed import _COMPONENTS as _FORMED_COMPONENTS
from starfinder.synthetic._sample import _SAMPLE_COMPONENTS

from . import cellular_appearance_oracle as oracle
from .cellular_appearance_scenes import (APP_BANKS, MORPH_WHOLE, app_scene, bg2_settings, case_placements,
                                         cf_sources, spheres_case, write_case)
from .formed_oracle import kernel_radius

pytestmark = [pytest.mark.synthetic, pytest.mark.validation]

TOLERANCE = 1e-9
SHIFT = np.array([1, 2, 3])
slow = pytest.mark.slow


def copy_of(name):
    return name.split("-")[0]


def frame_of(scene):
    """(spacing, Ω, sample grid origin and shape) of a scene."""
    sample = scene.context.sample.record
    grid = sample["sample_grid"]
    domain = sample["domain"]["omega"]
    return (np.asarray(grid["spacing_zyx"], dtype=np.float64),
            (np.asarray(domain["lo_zyx"], dtype=np.float64), np.asarray(domain["hi_zyx"], dtype=np.float64)),
            np.asarray(grid["origin_zyx"], dtype=np.float64), tuple(grid["shape_zyx"]))


def images(scene, config):
    """(stain, FOV, the grid's origin (true_f, or true_f + τ_f in the morphology round), image) of every stain image."""
    ctx = scene.context
    placement = ctx.sample.placement.set_index("fov_id")
    truth = ctx.morphology_truth.set_index("fov_id")
    held = [s for s in config.context.stains if s.round == "morphology"]
    for fov in ctx.sample.positions.fov_id:
        row = placement.loc[fov]
        true = np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x])
        for stain in config.context.stains:
            if stain.round == "morphology":
                tr = truth.loc[fov]
                origin = true + np.array([tr.translation_z_um, tr.translation_y_um, tr.translation_x_um])
                yield stain, fov, origin, ctx.morphology[fov][..., held.index(stain)]
            else:
                yield stain, fov, true, ctx.stains[fov][stain.name]


def background_of(scene, bank, stain_name, points):
    """B at (P, 3) points by the oracle: BG1 from background_patches.csv on the lattice the bank and Ω give, or BG2
    from the closed-form preset's own BackgroundConfig."""
    c = bank.channels.index(stain_name)
    if scene.record["background"]["mode"] == "patches":
        _, omega, _, _ = frame_of(scene)
        table = scene.background_patches[scene.background_patches.stain == stain_name]
        return oracle.patch_background(points, bank, table, oracle.patch_lattice(bank, omega), c)[0]
    return oracle.closed_form_background(points, bg2_settings(bank.record["preset"], stain_name),
                                         scene.record["plateau"])


def growth_of(scene, bank):
    if oracle.kind_of(bank) == "cell":
        return None
    order = scene.context.cell_context.set_index("cell_id").growth
    return [float(order[int(k)]) for k in scene.placements.sort_values("object_id").object_id]


def generated(name):
    book, config, scene = app_scene(name)
    assert not isinstance(scene, Exception), scene
    return book, config, scene, load_appearance_bank(config.bank)


# ---------------------------------------------------------------- GA12: composition (identity 4)

GA12 = ["tissue2d", pytest.param("dense3d", marks=slow), "culture",
        "tissue2d-plane", pytest.param("dense3d-plane", marks=slow), "culture-plane",
        "culture-morph-drawn", "culture-morph-supplied", "tissue2d-morph-supplied",
        pytest.param("dense3d-coarse", marks=slow), "tissue2d-bg1", pytest.param("dense3d-bg1", marks=slow),
        "culture-bg1"]


@pytest.mark.parametrize("name", GA12)
def test_ga12_stains_equal_the_composition(name):
    """Each float64 stain image against I = B + κH + S recomputed by the oracle from placements.csv, the bank, the
    background and the recorded κ at the voxel centres (every FOV, both rounds), and against the generator's point
    evaluator; inside each mask I − B − κH against the owner's excess alone; the image rendered again at
    true_f + k ⊙ s against the first shifted by k."""
    _, config, scene, bank = generated(name)
    s, _, _, _ = frame_of(scene)
    evaluator = _point_evaluator(scene, bank)
    shift = SHIFT.copy()
    if config.context.sample.acquisition.fov_shape_zyx[0] == 1:
        shift[0] = 0
    terms, worst, owner_worst, point_worst, shift_worst, boundary, compared, inside = {}, 0.0, 0.0, 0.0, 0.0, 0, 0, 0
    for stain, fov, origin, image in images(scene, config):
        c = bank.channels.index(stain.name)
        if stain.name not in terms:
            terms[stain.name] = oracle.object_terms(bank, scene.placements, c, s)
        kappa = scene.record["kappa"][stain.name]
        assert image.dtype == np.float64
        points = oracle.grid_points(origin, s, image.shape)
        B = background_of(scene, bank, stain.name, points)
        out = oracle.compose(points, terms[stain.name], kappa, B)
        peak = float(np.abs(image).max())
        ok = ~out["boundary"].reshape(image.shape)
        boundary += int((~ok).sum())
        error = np.abs(out["image"].reshape(image.shape) - image) / peak
        assert np.all(error[ok] <= TOLERANCE), (fov, stain.name, float(error[ok].max()))
        worst = max(worst, float(error[ok].max()))
        # Inside each mask only the owner's signal counts: I − B − κH is the owner's excess alone.
        held = out["held"].reshape(image.shape) & ok
        residual = image - B.reshape(image.shape) - kappa * out["haze"].reshape(image.shape)
        owner_error = np.abs(residual - out["owner"].reshape(image.shape))[held] / peak
        assert np.all(owner_error <= TOLERANCE)
        owner_worst = max(owner_worst, float(owner_error.max(initial=0.0)))
        inside += int(held.sum())
        # The generator's point evaluator at the same positions.
        point = evaluator.stain(stain.name, points).reshape(image.shape)
        point_error = np.abs(point - image) / peak
        assert np.all(point_error[ok] <= TOLERANCE)
        point_worst = max(point_worst, float(point_error[ok].max()))
        # The no-resampling identity: rendered again at origin + k ⊙ s, equal to the first shifted by k.
        moved = evaluator.grid(stain.name, origin + shift * s, image.shape)
        n = image.shape
        a = image[shift[0]:, shift[1]:, shift[2]:]
        b = moved[:n[0] - shift[0], :n[1] - shift[1], :n[2] - shift[2]]
        near = ok[shift[0]:, shift[1]:, shift[2]:]
        shift_error = np.abs(a - b)[near] / peak
        assert np.all(shift_error <= TOLERANCE)
        shift_worst = max(shift_worst, float(shift_error.max()))
        compared += image.size
    if name == "dense3d-coarse":
        # app_coarse: IN2's prefilter is active on every object, in Y and X.
        sigmas = [t.prefilter for t in next(iter(terms.values()))]
        assert all(sigma[1] > 0 and sigma[2] > 0 for sigma in sigmas)
        print(f"GA12 {name}: IN2's prefilter (source voxels) from {min(x[1] for x in sigmas):.4f} to "
              f"{max(x[1] for x in sigmas):.4f} in Y and X on all {len(sigmas)} objects")
    print(f"GA12 {name}: {compared} voxels; largest |image − oracle| {worst:.2e}, |image − point evaluator| "
          f"{point_worst:.2e}, the owner's excess inside the masks ({inside} voxels) {owner_worst:.2e}, the integer "
          f"shift {tuple(int(v) for v in shift)} {shift_worst:.2e}, each over the image maximum (≤ 1e-9); boundary "
          f"voxels {boundary} (expected 0)")
    assert boundary == 0


@pytest.mark.parametrize("name", ["tissue2d-bg1", "culture-bg1", pytest.param("dense3d-bg1", marks=slow)])
def test_ga12_background_patches_are_the_redrawn_lattice(name):
    """app_bg1: each lattice cell's patch in background_patches.csv against the index redrawn from
    ``context.appearance.background`` (entity ``patch-<z>-<y>-<x>``), shared by the stains; the lattice (Ω's lower
    corner, the patch size less its overlap) against the record."""
    _, config, scene, bank = generated(name)
    _, omega, _, _ = frame_of(scene)
    origin, period, shape, overlap, cells, spacing = oracle.patch_lattice(bank, omega)
    info = scene.record["background"]
    assert info["mode"] == "patches" and info["cells_zyx"] == cells
    assert np.allclose(info["lattice_origin_zyx"], origin, rtol=0, atol=1e-12)
    assert np.allclose(info["period_zyx"], period, rtol=0, atol=1e-12)
    assert info["patch_shape_zyx"] == shape.tolist() and info["overlap_zyx"] == overlap.tolist()
    scheme = scene.record["stream_scheme"]
    table = scene.background_patches
    names = list(dict.fromkeys(s.name for s in config.context.stains))
    assert list(dict.fromkeys(table.stain)) == names
    count = len(bank.arrays["patch"])
    images_ = bank.patches.source_image.tolist()
    for stain_name in names:
        own = table[table.stain == stain_name]
        assert len(own) == int(np.prod(cells))
        for row in own.itertuples():
            entity = f"patch-{row.lattice_z}-{row.lattice_y}-{row.lattice_x}"
            assert row.patch_id == int(oracle.stream(scheme, "context.appearance.background", entity).integers(count))
            assert row.source_image == images_[row.patch_id]
    print(f"GA12 {name}: {int(np.prod(cells))} lattice cells {cells} of period {np.round(period, 4).tolist()} µm, "
          f"each patch the redrawn index, shared by {names}")


def test_ga12_an_isolated_object_with_the_identity_transform_is_its_entry(tmp_path):
    """app_cases: an isolated sphere with the identity transform, placed so that c_k − x̄_e is a whole number of source
    voxels, gives S equal to its entry's excess shifted by (c_k − x̄_e) ⊘ s_e times its gain (no prefilter: the grid
    is the entry's own)."""
    config = write_case(spheres_case(), tmp_path / "source")
    path = tmp_path / "bank" / f"{config['bank']}.npz"
    build_appearance_bank(config, path)
    bank = load_appearance_bank(BankReference(name=config["bank"], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                              directory=str(path.parent)))
    entry = bank.entries.entry_id.iloc[0]
    row = bank.entries.iloc[0]
    shape = np.array([row.shape_z, row.shape_y, row.shape_x], dtype=np.int64)
    s_e = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
    xbar = (shape - 1) / 2 * s_e
    offset = np.array([3, -2, 5])
    centre = xbar + offset * s_e
    placements = case_placements([(1, entry, centre)])
    placements["gain"] = 1.05
    labels = _labels_from_table(bank, placements, growth=[0.5])
    composer = _Composer(bank, s_e, (2.0, 1.0, 1.0), "objects")
    composer.objects = labels.objects
    coords = [offset[a] * s_e[a] + np.arange(shape[a]) * s_e[a] for a in range(3)]
    signal = _Canvas(composer, coords, [0]).add_until(1).signal(0)
    expected = 1.05 * oracle.stored(bank, entry)[1][0]
    peak = float(np.abs(expected).max())
    error = float(np.abs(signal - expected).max()) / peak
    print(f"GA12 app_cases: an isolated sphere shifted by {offset.tolist()} source voxels, gain 1.05: S against the "
          f"entry's excess times the gain, largest difference {error:.2e} of its maximum (≤ 1e-9)")
    assert error <= TOLERANCE


# ---------------------------------------------------------------- GA13: out-of-focus light and the rim background

GA13 = ["tissue2d", pytest.param("dense3d", marks=slow), "culture", "tissue2d-bg1",
        "culture-bg1", "tissue2d-plane", "culture-plane"]


@pytest.mark.parametrize("name", GA13)
def test_ga13_kappa_and_the_rim_background(name):
    """Per stain: κ recomputed on the sample grid as max(b̄ − mean_rim(B), 0) / mean_rim(H) over the rims of the placed
    masks (their union less its face-connected erosion), with b̄ the mean local source background of the placed
    objects; where κ > 0 the rim mean of B + κH equals b̄ (1e-9 relative), where κ = 0 the rim mean of B is at least
    b̄ (reported); each object's blurred copy against scipy's Gaussian filter of its excess with H2's deviations; H2
    against the grid blur H1 of the composed signal, reported."""
    _, config, scene, bank = generated(name)
    s, _, origin, shape = frame_of(scene)
    kind = oracle.kind_of(bank)
    column = kind == "column"
    points = oracle.grid_points(origin, s, shape)
    objects = oracle.frames(bank, scene.placements)
    N, C, boundary, _ = oracle.labels(points.reshape(-1, 3), objects, kind, growth_of(scene, bank))
    assert not boundary.any()
    mask = ((C if kind == "cell" else N) > 0).reshape(shape)
    rim = oracle.rim(mask, column)
    reports = []
    evaluator = _point_evaluator(scene, bank)
    for stain_name in dict.fromkeys(s_.name for s_ in config.context.stains):
        c = bank.channels.index(stain_name)
        terms = oracle.object_terms(bank, scene.placements, c, s)
        B = background_of(scene, bank, stain_name, points).reshape(shape)
        out = oracle.compose(points, terms, 0.0, B)
        H = out["haze"].reshape(shape)
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
        # Each object's blurred copy against the library call on its own excess (the box without the grown margin).
        copy_error = 0.0
        for i, t in enumerate(terms):
            array, pad = evaluator.composer.haze(i, c)
            core = array[tuple(slice(p, p + n) for p, n in zip(pad, t.X.shape))]
            expected = ndi.gaussian_filter(t.X, t.haze_sigma, mode="constant", cval=0.0)
            copy_error = max(copy_error, float(np.abs(core - expected).max()) / max(float(np.abs(expected).max()),
                                                                                   1e-300))
        assert copy_error <= TOLERANCE
        # H1: the prototype's blur of the composed signal on the sample grid, against H2 (reported, no bound).
        sigma = [h / x if n > 1 else 0.0 for h, x, n in zip(oracle.HAZE_UM, s, shape)]
        H1 = ndi.gaussian_filter(out["signal"].reshape(shape), sigma)
        difference = float(np.abs(H - H1).max()) / max(float(np.abs(H1).max()), 1e-300)
        reports.append(f"{stain_name}: {case}; blurred copies against the library call {copy_error:.2e} of their "
                       f"maximum; H2 against H1 largest difference {difference:.3g} of H1's maximum (reported)")
    print(f"GA13 {name} ({int(rim.sum())} rim voxels): " + "; ".join(reports))


# ---------------------------------------------------------------- GA14: brightness within a scene (the scene's part)

@pytest.mark.parametrize("name", ["tissue2d", pytest.param("dense3d", marks=slow), "culture"])
def test_ga14_g_is_the_median_plateau_and_each_gain_scales_its_object(name):
    """``g`` is the median plateau of the bank's source images (O8, exact); each placed object's gain lies in AU1's
    range and scales its excess: inside the voxels it owns, S over its interpolated excess is its gain."""
    _, config, scene, bank = generated(name)
    plateaus = [row["plateau"] for row in bank.record["images"]]
    assert scene.record["plateau"] == float(np.median(plateaus)) == bank.record["plateau_median"]
    gains = scene.placements.gain.to_numpy()
    assert np.all((gains >= 0.9) & (gains <= 1.1))
    s, _, origin, shape = frame_of(scene)
    points = oracle.grid_points(origin, s, shape).reshape(-1, 3)
    stain_name = config.context.stains[0].name
    terms = oracle.object_terms(bank, scene.placements, bank.channels.index(stain_name), s)
    evaluator = _point_evaluator(scene, bank)
    B = background_of(scene, bank, stain_name, points)
    out = oracle.compose(points, terms, 0.0, B)
    signal = evaluator.stain(stain_name, points) - B - scene.record["kappa"][stain_name] * out["haze"]
    first = np.full(len(points), -1)
    for i, t in enumerate(terms):
        held = (t.frame.d(points) <= 0) & (first < 0)
        first[held] = i
    worst, owners = 0.0, 0
    for i, t in enumerate(terms):
        own = first == i
        if not own.any():
            continue
        excess = t.value(t.E, t.frame.entry_frame(points[own]))
        big = excess > 1e-3 * float(np.abs(excess).max())
        ratio = signal[own][big] / excess[big]
        worst = max(worst, float(np.abs(ratio - t.frame.gain).max()))
        owners += 1
    print(f"GA14 {name}: g = {scene.record['plateau']} = median of the source plateaus {plateaus}; gains "
          f"{gains.min():.4f} to {gains.max():.4f}; on {owners} objects owning sample-grid voxels S / ε equals the "
          f"gain within {worst:.2e}")
    assert worst <= 1e-9


# ---------------------------------------------------------------- GA15: 8-bit output (identity 5)

@pytest.mark.parametrize("copy", ["tissue2d", pytest.param("dense3d", marks=slow), pytest.param("culture",
                                                                                                 marks=slow)])
def test_ga15_8bit_images_are_the_cast_and_no_noise_is_drawn(copy):
    """app_tiny_uint8 against its float64 rendering J: each 8-bit image equals clip(rint(g · J), 0, 255); no noise
    stream is drawn (OB1)."""
    _, config, scene, _ = generated(f"{copy}-uint8")
    _, float_config, reference, _ = generated(copy)
    g = scene.record["plateau"]
    pd.testing.assert_frame_equal(scene.placements, reference.placements, check_exact=True)
    voxels, clipped = 0, 0
    expected_images = {(st.name, st.round, f): j for st, f, _, j in images(reference, float_config)}
    for stain, fov, _, image in images(scene, config):
        J = expected_images[(stain.name, stain.round, fov)]
        assert image.dtype == np.uint8
        expected = np.clip(np.rint(g * J), 0, 255).astype(np.uint8)
        assert np.array_equal(image, expected), (fov, stain.name)
        voxels += image.size
        clipped += int((g * J > 255).sum())
    channels = scene.context.stain_channels
    assert (channels.dtype == "uint8").all() and (channels.scale == g).all()
    drawn = {d[4] for d in scene.context.record["stream_scheme"]["streams"]}
    drawn |= {d[4] for d in scene.context.sample.record["stream_scheme"]["streams"]}
    assert not any("noise" in component for component in drawn)
    print(f"GA15 {copy}: {voxels} 8-bit voxels equal clip(rint(g · J), 0, 255) with g = {g}, {clipped} clipped at "
          f"255; no noise stream among {len(drawn)} components drawn")



# ---------------------------------------------------------------- GA16: the morphology round (identity 6)

@pytest.mark.parametrize("name", ["culture-morph-supplied", "culture-morph-drawn", "culture-plane",
                                  "tissue2d-morph-supplied"])
def test_ga16_morphology_round(name):
    """morphology_truth against the configuration as GN18 (µm over s equals the voxels within 1e-9, the pull
    displacement is minus the translation, within the bound; supplied (1, 0, 0) voxels, drawn fractional, 0 in Z at
    one plane); with the whole-voxel translation, each morphology stain against the reference stain of its name shifted
    by τ ⊘ s, within 1e-9 of the maximum; both rounds against the entries' channels at their points (GA12)."""
    _, config, scene, bank = generated(name)
    ctx = scene.context
    truth = ctx.morphology_truth.set_index("fov_id")
    s, _, _, _ = frame_of(scene)
    morphology = config.context.morphology
    bound = np.asarray(morphology.translation_max_zyx)
    assert list(truth.index) == list(ctx.sample.positions.fov_id) and (truth["round"] == morphology.name).all()
    vox = truth[["translation_z_vox", "translation_y_vox", "translation_x_vox"]].to_numpy()
    um = truth[["translation_z_um", "translation_y_um", "translation_x_um"]].to_numpy()
    pull = truth[["pull_displacement_z_vox", "pull_displacement_y_vox", "pull_displacement_x_vox"]].to_numpy()
    assert np.all(np.abs(um / s - vox) <= 1e-9) and np.all(np.abs(pull + vox) <= 1e-9)
    assert np.all(np.abs(um) <= bound * s + 1e-12)
    flat = config.context.sample.acquisition.fov_shape_zyx[0] == 1
    if flat:
        assert np.all(vox[:, 0] == 0) and np.all(um[:, 0] == 0)
    report = f"GA16 {name}: translations (voxels) {np.round(vox, 4).tolist()}"
    if morphology.translation == "uniform":
        assert np.any(np.abs(vox - np.round(vox)) > 1e-3)
    else:
        assert np.all(np.abs(vox - np.tile(MORPH_WHOLE, (len(vox), 1))) <= 1e-9)
        whole = np.rint(vox[0]).astype(int)
        terms = {}
        worst, boundary, compared = 0.0, 0, 0
        found = {(st.name, st.round, fov): (origin, image) for st, fov, origin, image in images(scene, config)}
        for (stain_name, round_, fov), (origin, moved) in found.items():
            if round_ != "morphology" or (stain_name, "reference", fov) not in found:
                continue  # a morphology stain without a reference stain of its name is checked by GA12 alone
            reference_origin, image = found[(stain_name, "reference", fov)]
            n = image.shape
            a = image[whole[0]:, whole[1]:, whole[2]:]
            b = moved[:n[0] - whole[0], :n[1] - whole[1], :n[2] - whole[2]]
            if stain_name not in terms:
                terms[stain_name] = oracle.object_terms(bank, scene.placements, bank.channels.index(stain_name), s)
            pa = oracle.grid_points(reference_origin, s, n)[whole[0]:, whole[1]:, whole[2]:]
            pb = oracle.grid_points(origin, s, n)[:n[0] - whole[0], :n[1] - whole[1], :n[2] - whole[2]]
            near = (oracle.compose(pa, terms[stain_name], 0.0, 0.0)["boundary"]
                    | oracle.compose(pb, terms[stain_name], 0.0, 0.0)["boundary"]).reshape(a.shape)
            peak = max(float(np.abs(image).max()), float(np.abs(moved).max()))
            error = np.abs(a - b) / peak
            assert np.all(error[~near] <= TOLERANCE), (fov, stain_name)
            worst = max(worst, float(error[~near].max()))
            boundary += int(near.sum())
            compared += a.size
        report += (f"; morphology stains against the reference stains shifted by {whole.tolist()}: {compared} voxels, "
                   f"largest difference {worst:.2e} of the maximum (≤ 1e-9); boundary voxels {boundary} (expected 0)")
        assert boundary == 0 and compared > 0
    print(report)


# ---------------------------------------------------------------- GA17: molecules and types (identity 8)

def placing_cell(amplicon_id):
    """The cell that placed a cell molecule (IDs c<k>-m<j>)."""
    return int(amplicon_id[1:].split("-")[0])


GA17 = ["tissue2d-types-light", "culture-types-light", pytest.param("dense3d-types-light", marks=slow),
        pytest.param("tissue2d-types", marks=slow), pytest.param("culture-types", marks=slow),
        pytest.param("dense3d-types", marks=slow)]


@pytest.mark.parametrize("name", GA17)
def test_ga17_types_genes_and_molecules(name):
    """GN8 to GN12 with C and N of the appearance: one cell_context row per cell, the regions, each cell's region the
    Voronoi region of its anchor (anchors within 1e-9 µm of a region boundary reported, expected 0), each abundance row
    summing to 1 with G1's configured values and roles, each cell molecule of a gene of its cell's type and no marker
    among the ambient ones (GN8); cell_gene_counts against a recount, its sums against cells.csv n_molecules, and the
    marker answers against MECR recomputed (values within 1e-9, null when either is 0, 0 for the exclusive pair) (GN9);
    the realized gene abundance per type (GN10); Σ n_k against Σ μ_k, one mean per cell, and the molecules handed to
    the formed generator unchanged (GN11); the nuclear fraction within 4 SE, every cell molecule in the cell that placed
    it and its compartment the oracle's N and C at its position (GN12)."""
    book, config, scene, bank = generated(name)
    ctx = scene.context
    sample, context = ctx.sample, ctx.cell_context
    types = config.context.types
    assert context.cell_id.tolist() == sample.cells.cell_id.tolist() == scene.placements.object_id.tolist()
    assert (sample.cells.kind == "ordinary").all() and (sample.cells.n_nuclei == 1).all()
    regions = ctx.regions
    assert regions.region_id.tolist() == list(range(1, types.regions + 1))
    centres = regions[["centre_z", "centre_y", "centre_x"]].to_numpy()
    anchors = sample.cells[["anchor_z", "anchor_y", "anchor_x"]].to_numpy()
    squared = ((anchors[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
    region = np.argmin(squared, axis=1)
    np.testing.assert_array_equal(context.region.astype(int).to_numpy(), region + 1)
    # Anchors within 1e-9 µm of a region boundary (the Voronoi decision's rounding), as GN8 reports them.
    order = np.sort(squared, axis=1)
    gap = np.linalg.norm(centres[region] - centres[np.argsort(squared, axis=1)[:, 1]], axis=1)
    near_boundary = int(((order[:, 1] - order[:, 0]) / (2 * gap) <= 1e-9).sum())
    assert set(context.type) <= set(types.types)
    table = ctx.type_abundances
    genes = list(book.genes)
    markers = 2 * len(types.types)
    for t, type_name in enumerate(types.types):
        row = table[table.type == type_name]
        expected_row = [0.15 if g in (2 * t, 2 * t + 1) else 0.0875 if g >= markers else 0.0 for g in range(len(genes))]
        assert row.gene_id.tolist() == genes and abs(row.weight.sum() - 1) <= 1e-9
        assert np.all(np.abs(row.weight.to_numpy() - expected_row) <= 1e-9)
        assert row.role.tolist() == ["marker"] * markers + ["shared"] * (len(genes) - markers)
    ambient = table[table.type == "ambient"]
    shared = 1.0 / (len(genes) - markers)
    assert ambient.gene_id.tolist() == genes and abs(ambient.weight.sum() - 1) <= 1e-9
    assert np.all(np.abs(ambient.weight.to_numpy() - np.array([0.0] * markers + [shared] * (len(genes) - markers)))
                  <= 1e-9)
    weight = {(r.type, r.gene_id): r.weight for r in table.itertuples()}
    type_of = dict(zip(context.cell_id.astype(int), context.type))
    molecules = sample.molecules
    cell_molecules = molecules[molecules.population == "cell"]
    for row in cell_molecules.itertuples():
        assert weight[type_of[placing_cell(row.amplicon_id)], row.gene_id] > 0, row.amplicon_id
    assert not set(molecules.gene_id[molecules.population == "ambient"]) & set(genes[:2 * len(types.types)])
    # GN9: counts per cell and gene, and the marker answers over the sample cells.
    owned = molecules[molecules.true_cell_id.notna()]
    expected = {}
    for cell_id, gene, compartment in zip(owned.true_cell_id.astype(int), owned.gene_id, owned.compartment):
        count, nuclear = expected.get((cell_id, gene), (0, 0))
        expected[cell_id, gene] = (count + 1, nuclear + (compartment == "nucleus"))
    counts_table = ctx.cell_gene_counts
    assert {(int(r.cell_id), r.gene_id): (int(r.count), int(r.nuclear_count))
            for r in counts_table.itertuples()} == expected
    assert len(counts_table) == len(expected) and (counts_table["count"] > 0).all()
    sums = counts_table.groupby("cell_id")["count"].sum()
    assert sample.cells.n_molecules.tolist() == [int(sums.get(k, 0)) for k in sample.cells.cell_id]
    counted = sample.cells.cell_id[sample.cells.in_sample].astype(int).tolist()
    counts = {(int(r.cell_id), r.gene_id): int(r.count) for r in counts_table.itertuples()}
    pairs = ctx.marker_pairs
    assert [tuple(p) for p in pairs[["gene_a", "gene_b", "relation"]].to_numpy()] == list(types.marker_pairs)
    answers = []
    for row in pairs.itertuples():
        values = mecr([counts.get((c, row.gene_a), 0) for c in counted], [counts.get((c, row.gene_b), 0) for c in
                                                                          counted], 1)
        assert (row.cells_a, row.cells_b, row.both, row.either) == (values["cells_a"], values["cells_b"],
                                                                     values["both"], values["either"])
        if row.either:
            assert abs(row.mecr - values["mecr"]) <= 1e-9  # the same ratio of the same integers; "exact" as 1e-9
        else:
            assert pd.isna(row.mecr)
        if row.relation == "exclusive":
            assert row.both == 0 and (row.either == 0 or row.mecr == 0)
        answers.append(f"{row.gene_a}/{row.gene_b} {row.relation} either {row.either} mecr {row.mecr}")
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
    # GN11: the molecules reach the formed generator unchanged (positions, genes, properties per FOV).
    properties = molecules.set_index("amplicon_id")[["A", "sz", "sl", "e", "theta"]]
    gene_of = dict(zip(molecules.amplicon_id, molecules.gene_id))
    for fov, formed_scene in sample.fovs.items():
        rows = sample.molecule_fovs[sample.molecule_fovs.fov_id == fov].sort_values("csv_row")
        formed = formed_scene.formed.set_index("amplicon_id").loc[rows.amplicon_id]
        np.testing.assert_array_equal(formed[["z", "y", "x"]].to_numpy(), rows[["z", "y", "x"]].to_numpy())
        assert formed.gene_id.tolist() == [gene_of[a] for a in rows.amplicon_id]
        np.testing.assert_array_equal(formed[["A", "sz", "sl", "e", "theta"]].to_numpy(),
                                      properties.loc[rows.amplicon_id].to_numpy())
    # GN12: the nuclear fraction within 4 SE, and each cell molecule's cell and compartment against the oracle.
    fraction = config.context.molecules.nuclear_fraction
    nuclear = (cell_molecules.compartment == "nucleus").to_numpy()
    share = float(nuclear.mean())
    se = math.sqrt(fraction * (1 - fraction) / len(nuclear))
    assert abs(share - fraction) <= 4 * se
    placers = cell_molecules.amplicon_id.map(placing_cell).to_numpy()
    np.testing.assert_array_equal(cell_molecules.true_cell_id.astype(int).to_numpy(), placers)
    sigma = molecules[["sample_z", "sample_y", "sample_x"]].to_numpy()
    N, C, near, edge = oracle.labels(sigma, oracle.frames(bank, scene.placements), oracle.kind_of(bank),
                                     growth_of(scene, bank))
    np.testing.assert_array_equal(molecules.true_cell_id.fillna(0).astype(int).to_numpy()[~near], C[~near])
    in_own_nucleus = (N > 0) & (N == C)
    assert ((molecules.compartment == "nucleus").to_numpy() == in_own_nucleus)[~near].all()
    print(f"GA17 {name}: {len(context)} cells of types {context.type.value_counts().to_dict()}, {len(cell_molecules)} "
          f"cell molecules, {compared} gene shares within 4 SE; Σ n_k {total} against Σ μ_k {means:.1f}; nuclear "
          f"fraction {share:.4f} against {fraction} (4 SE {4 * se:.4f}); molecule positions at a boundary "
          f"{int(near.sum())} (expected 0), at a growth edge {int(edge.sum())}; anchors within 1e-9 µm of a region "
          f"boundary {near_boundary} (expected 0); marker answers: " + "; ".join(answers))
    assert not near.any() and near_boundary == 0


@pytest.mark.parametrize("copy", ["tissue2d", "culture", pytest.param("dense3d", marks=slow)])
def test_ga17_the_count_guard(copy):
    """GN11: max_count guards the total; the appearance scene with max_count one below its molecule count raises."""
    book, config, scene, _ = generated(f"{copy}-types-light")
    count = len(scene.context.sample.molecules)
    guard = replace(config, context=replace(config.context, sample=replace(config.context.sample,
                                                                            max_count=count - 1)))
    with pytest.raises(ValueError, match="max_count"):
        generate_appearance_scene(book, config=guard)
    print(f"GA17 {copy}: max_count {count - 1} refuses the scene's {count} molecules")


@pytest.mark.parametrize("copy", ["tissue2d", "culture", pytest.param("dense3d", marks=slow)])
def test_ga17_zero_spread_gives_the_median_mean(copy):
    """GN11: with per_cell_spread 0 (M1) every cell's molecule mean is the median ("exact" for a non-integer: within
    1e-9)."""
    _, config, scene, _ = generated(f"{copy}-types-light-spread0")
    means = scene.context.cell_context.molecule_mean.to_numpy()
    median = config.context.molecules.per_cell_median
    assert len(means) and np.all(np.abs(means - median) <= 1e-9)
    print(f"GA17 {copy}: spread 0 gives every one of {len(means)} cells the mean {median}")


@pytest.mark.parametrize("copy", ["tissue2d", "culture", pytest.param("dense3d", marks=slow)])
def test_ga17_rendering_identity_of_the_reference_round(copy):
    """GN11: the amplicon images of the reference round, without FOV noise, equal the rendering identity of
    design/specs/synthetic/sample-scene.md: each molecule's Gaussian kernel (formed_oracle.kernel_radius, cut at a normalized
    radius² of 16) times its brightness, summed into its colour's channel; the formed generator renders the appearance
    scene's molecules unchanged."""
    book, config, scene, _ = generated(f"{copy}-types-light-quiet")
    reference = config.context.sample.fov.geometry.reference_round
    r = book.round_labels.index(reference)
    shape = np.array(config.context.sample.acquisition.fov_shape_zyx)
    points = np.moveaxis(np.indices(tuple(shape), dtype=float), 0, -1)
    rendered = 0
    for fov, formed_scene in scene.context.sample.fovs.items():
        expected = np.zeros((*shape, len(book.channel_labels)))
        for row in formed_scene.formed.itertuples():
            radius = kernel_radius(points, np.array([row.z, row.y, row.x]), row.sz, row.sl, row.e, row.theta)
            expected[..., book.color_to_channel[row.color_sequence[r]]] += np.where(radius <= 16, np.exp(-radius / 2),
                                                                                    0) * row.A
            rendered += 1
        image = formed_scene.rounds[reference]
        bound = np.maximum(1e-6, np.abs(expected) * np.finfo(np.float32).eps / 2 + 2e-10)
        assert expected.max() > 0 and np.all(np.abs(image.astype(np.float64) - expected) <= bound), fov
    print(f"GA17 {copy}: the reference round of {len(scene.context.sample.fovs)} FOVs equals the kernel sum of its "
          f"{rendered} rendered molecules (float32 rounding)")


@pytest.mark.parametrize("copy", ["tissue2d", pytest.param("culture", marks=slow), pytest.param("dense3d",
                                                                                                 marks=slow)])
def test_ga17_a_type_size_scale_changes_no_placement_and_no_label(copy):
    """app_streams: ctx_types with type 2's size_scale 1.5 against the scene without it: placements.csv and every label
    image equal (sizes come from the entries and AU1's scale)."""
    _, _, scaled, _ = generated(f"{copy}-size-scale-light")
    _, _, plain, _ = generated(f"{copy}-types-light")
    pd.testing.assert_frame_equal(scaled.placements, plain.placements, check_exact=True)
    for target in ("nucleus", "cell"):
        assert np.array_equal(scaled.context.sample.sample_labels[target], plain.context.sample.sample_labels[target])
        for fov, labels in plain.context.sample.fov_labels.items():
            assert np.array_equal(scaled.context.sample.fov_labels[fov][target], labels[target])
    assert scaled.context.cell_context.type.tolist() == plain.context.cell_context.type.tolist()
    print(f"GA17 {copy}: type 2's size_scale 1.5 changes no placement ({len(plain.placements)} objects) and no label")


# ---------------------------------------------------------------- GA19: Z=1, the stain part

@pytest.mark.parametrize("name", ["tissue2d-plane", "culture-plane", "dense3d-plane"])
def test_ga19_one_plane_stains(name):
    """app_plane: every stain image and the morphology round have one plane, every centre and molecule lies at z = 0
    and τ_z = 0 (GA12, GA15 and GA16 on this fixture are their parametrizations above)."""
    _, config, scene, _ = generated(name)
    ctx = scene.context
    shape = (1, 36, 36)
    for fov in ctx.sample.positions.fov_id:
        assert all(image.shape == shape for image in ctx.stains[fov].values())
        if fov in ctx.morphology:
            assert ctx.morphology[fov].shape[:3] == shape
        assert all(labels.shape == shape for labels in ctx.sample.fov_labels[fov].values())
    assert all(labels.shape[0] == 1 for labels in ctx.sample.sample_labels.values())
    assert (scene.placements.centre_z == 0).all() and (ctx.sample.molecules.sample_z == 0).all()
    assert (ctx.sample.cells.anchor_z == 0).all() and (ctx.sample.nuclei.centre_z == 0).all()
    assert (ctx.morphology_truth.translation_z_um == 0).all()
    off = float(np.abs(scene.placements.anchor_z).max())
    print(f"GA19 {name}: every stain and label image of one plane, centres, cell anchors and molecules at z = 0, "
          f"τ_z = 0; placed anchors (the entries' deepest points) up to {off:.3f} µm off the plane, the cells' "
          "anchors in the plane")


def test_ga19_saved_labels_import_as_plane_label_images(tmp_path):
    """app_plane: the saved FOV labels are read as ``plane`` label images (GA19)."""
    _, config, scene, _ = generated("tissue2d-plane")
    root = save_appearance_scene(scene, tmp_path / "plane")
    sample = scene.context.sample
    for fov in sample.positions.fov_id:
        grid = ReferenceGrid((1, 36, 36), sample.fov_metadata[fov], "declared")
        for target, file in (("cell", "cells.tif"), ("nucleus", "nuclei.tif")):
            namespace = json.dumps([config.context.sample.fov.dataset_version, config.context.sample.fov.sample_id,
                                    fov, None, "truth"])
            imported = import_labels(root / "fovs" / fov / "labels" / file, grid=grid, target=target,
                                     label_namespace=namespace)
            assert imported.record["import"]["geometry"] == "plane"
    print("GA19 tissue2d-plane: the saved labels are imported as plane label images")


# ---------------------------------------------------------------- GA20: the streams, the stain part

ACCEPTED = set(_SAMPLE_COMPONENTS) | set(_CONTEXT_COMPONENTS) | {"sample." + c for c in _FORMED_COMPONENTS}
FORBIDDEN = ("context.nucleus.", "context.cell.kind", "context.cell.contact", "context.stain.", "context.texture.")


@pytest.mark.parametrize("name", ["tissue2d", "culture-bg1", "culture-morph-drawn", pytest.param("dense3d",
                                                                                                 marks=slow)])
def test_ga20_components_of_the_appearance_scene(name):
    """Every component the appearance generator draws from is an accepted one it reuses or starts with
    ``context.appearance.``; no ``context.nucleus.*``, ``context.cell.kind``, ``context.cell.contact``,
    ``context.stain.*``, ``context.texture.*`` or noise stream is drawn."""
    _, _, scene, _ = generated(name)
    context = {d[4] for d in scene.context.record["stream_scheme"]["streams"]}
    sample = {d[4] for d in scene.context.sample.record["stream_scheme"]["streams"]}
    new = {c for c in context | sample if c not in ACCEPTED}
    assert new and all(c.startswith("context.appearance.") for c in new), new
    assert not any(c.startswith(FORBIDDEN) for c in context | sample)
    assert not any("noise" in c for c in context | sample)
    appearance = {d[4] for d in scene.record["stream_scheme"]["streams"]}
    assert appearance == {c for c in context if c.startswith("context.appearance.")}
    print(f"GA20 {name}: new components {sorted(new)}; reused {sorted((context | sample) - new)}")


@slow
def test_ga20_dropping_a_stain_changes_no_other_stain():
    """app_streams: the culture copy without its Flamingo stain against the copy with it: the DAPI images, κ of DAPI,
    the placements and the labels are equal."""
    _, _, dropped, _ = generated("culture-no-flamingo")
    _, _, full, _ = generated("culture")
    pd.testing.assert_frame_equal(dropped.placements, full.placements, check_exact=True)
    assert dropped.record["kappa"]["DAPI"] == full.record["kappa"]["DAPI"] and "Flamingo" not in dropped.record["kappa"]
    for fov, stains in full.context.stains.items():
        assert list(dropped.context.stains[fov]) == ["DAPI"]
        assert np.array_equal(dropped.context.stains[fov]["DAPI"], stains["DAPI"])
        for target in ("nucleus", "cell"):
            assert np.array_equal(dropped.context.sample.fov_labels[fov][target],
                                  full.context.sample.fov_labels[fov][target])
    print("GA20: dropping the Flamingo stain changes no DAPI image, κ, placement or label")


# ---------------------------------------------------------------- GA10: DN1's stop (app_dn1)

def dn1_source_medians(bank, copy, extent_um):
    """DN1's target and threshold recomputed from the closed-form bank's source images: per image the middle planes
    matching the scene's Z extent (as a Z maximum for culture cells, the cells' stain), the stain smoothed by one
    voxel, its Otsu threshold and visible share, the threshold over the image's plateau; the medians over images."""
    from skimage.filters import threshold_otsu
    shares, thresholds = [], []
    for source in cf_sources(APP_BANKS[copy]):
        channel = len(source["stains"]) - 1 if copy == "culture" else 0
        stain = source["stains"][channel]
        planes = min(max(int(round(extent_um / float(source["spacing"][0]))), 1), stain.shape[0])
        start = (stain.shape[0] - planes) // 2
        region = stain[start:start + planes]
        plane = region.max(axis=0) if copy == "culture" else region
        smooth = ndi.gaussian_filter(plane, [1.0 if n > 1 else 0.0 for n in plane.shape])
        threshold = float(threshold_otsu(smooth))
        labels = source["nucleus_labels"] if source["nucleus_labels"] is not None else source["labels"]
        shares.append(float(np.mean(smooth > threshold)))
        thresholds.append(threshold / oracle.plateau(labels, source["stains"][0]))
    return float(np.median(shares)), float(np.median(thresholds))


def visible_after(scene, bank, config, count, threshold, form, channel_name):
    """The visible share of the composed stain on the sample grid after the first ``count`` placements, κ recomputed
    for them, by the oracle; with the voxels within 1e-9 of the threshold."""
    s, _, origin, shape = frame_of(scene)
    kind = oracle.kind_of(bank)
    placements = scene.placements.sort_values("object_id").iloc[:count]
    points = oracle.grid_points(origin, s, shape)
    c = bank.channels.index(channel_name)
    terms = oracle.object_terms(bank, placements, c, s)
    B = background_of(scene, bank, channel_name, points).reshape(shape)
    out = oracle.compose(points, terms, 0.0, B)
    mask = out["held"].reshape(shape)
    rim = oracle.rim(mask, kind == "column")
    local = bank.entries.set_index("entry_id")[f"local_background_{channel_name}"]
    mean_b = float(np.mean([local[e] for e in placements.entry_id]))
    H = out["haze"].reshape(shape)
    kappa = max(mean_b - float(B[rim].mean()), 0.0) / float(H[rim].mean()) if rim.any() and H[rim].mean() > 0 else 0.0
    image = B + kappa * H + out["signal"].reshape(shape)
    return oracle.visible_share(image, threshold, form)


@pytest.mark.parametrize("copy", ["culture", pytest.param("dense3d", marks=slow)])
def test_ga10_dn1_target_threshold_and_stop(copy):
    """app_dn1: the recorded target and threshold against the medians over the bank's source images; placement stops at
    the first check whose visible share reaches the target: the share after the last placement is at least the target
    and after the one before it below, each recomputed by the oracle (κ recomputed for the placed objects) as a voxel
    count."""
    _, config, scene, bank = generated(f"{copy}-dn1")
    density = scene.record["density"]
    extent = config.context.sample.acquisition.fov_shape_zyx[0] * config.context.sample.acquisition.spacing_zyx[0]
    target, threshold = dn1_source_medians(bank, copy, extent)
    assert abs(density["target"] - target) <= 1e-9 and abs(density["threshold"] - threshold) <= 1e-9
    assert density["stopped_by"] == "target" and density["check_every"] == 1
    trace = density["trace"]
    assert [t["placed"] for t in trace] == list(range(1, len(trace) + 1)) == list(range(1, scene.record["placed"] + 1))
    assert trace[-1]["visible_share"] >= target > trace[-2]["visible_share"]
    assert all(t["visible_share"] < target for t in trace[:-1])
    _, _, _, shape = frame_of(scene)
    voxels = int(np.prod(shape if density["form"] == "volume" else shape[1:]))
    reports = []
    for count in (len(trace), len(trace) - 1):
        share, near = visible_after(scene, bank, config, count, threshold, density["form"], density["stain"])
        assert round(share * voxels) == round(trace[count - 1]["visible_share"] * voxels), count
        reports.append(f"after {count} placements {share:.4f} ({near} voxels within 1e-9 of the threshold)")
    print(f"GA10 {copy} (DN1): target {density['target']:.6f} and threshold {density['threshold']:.6f} (relative to "
          f"the plateau) equal the medians over the bank's source images; stopped at the target after "
          f"{scene.record['placed']} placements and {density['proposals']} proposals; oracle " + ", ".join(reports))


@pytest.mark.parametrize("copy", ["culture", "dense3d"])
def test_ga10_dn1_short_copy_records_or_raises_at_its_floor(copy):
    """app_dn1's short copy (max_proposals too small to reach the target): at floor 0.1 placement stops at
    max_proposals and records the reached share; at floor 0.99 it raises ValueError naming both values."""
    _, config, recorded, _ = generated(f"{copy}-dn1-short-0.1")
    density = recorded.record["density"]
    assert density["stopped_by"] == "max_proposals" and density["proposals"] == config.layout.max_proposals
    assert 0.1 * density["target"] <= density["reached"] < density["target"]
    assert density["reached"] == density["trace"][-1]["visible_share"]
    _, _, error = app_scene(f"{copy}-dn1-short-0.99")
    assert isinstance(error, ValueError)
    message = str(error)
    assert "max_proposals" in message and str(density["reached"]) in message
    assert f"0.99 · {density['target']}" in message
    print(f"GA10 {copy} (DN1, short): floor 0.1 stops at max_proposals={config.layout.max_proposals} with a reached "
          f"share {density['reached']:.4f} of the target {density['target']:.4f}; floor 0.99 raises: {message}")
