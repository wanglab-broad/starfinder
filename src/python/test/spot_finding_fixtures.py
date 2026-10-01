"""The hand-built fixtures of the §2.7 engineering validation (docs/spot-finding-algorithms.md,
"Engineering validation design"; W-274).

The W-266 isolated-spot and seam scenes are the ports in spot_finding_scenes
(``isolated_scene``, ``w266_common.make_case``), test_spot_finding_metrics
(``isolated_positions``) and learned_detectors (``seam_layout``, ``seam_scene``);
``render_w266`` below is ``make_case``'s isolated-spot branch for any list of
centres, so ``iso_z1_sparse`` and ``pairs`` share the W-266 appearance and
rendering exactly. The multi-channel fixtures (``channels``, ``coincident``) and
``borders`` are rendered here with NumPy (``render``) because their checks need
control over which channels share a noise realization. Every fixture is
generated in session from one seed (100, 101 or 102) and is at most 32x64x64
voxels and four channels; nothing is written to disk.
"""
import numpy as np
import pandas as pd

from starfinder.barcode import Codebook
from starfinder.synthetic import (BackgroundConfig, FormedSceneConfig, NoiseConfig, ScalarDistribution,
    generate_formed_scene)

# The W-266 isolated-spot appearance: brightness (peak) 1500, sigma Z 1.5 and YX 1.3, baseline 100,
# Poisson (alpha 1) plus read noise 3, uint16.
BRIGHTNESS, SIGMA_ZYX, BASELINE, READ_NOISE = 1500.0, (1.5, 1.3, 1.3), 100.0, 3.0
SHAPES = {"iso_z1_sparse": (1, 64, 64), "pairs": (32, 64, 64), "pairs_z1": (1, 64, 64),
          "channels": (16, 64, 64, 4), "coincident": (16, 64, 64, 2), "borders": (16, 64, 64),
          "borders_z1": (1, 64, 64)}
# Random streams of the fixtures, separate from W-266's [266, seed, k] streams.
_STREAMS = {"iso_z1_sparse": 1, "pairs": 2, "channels": 3, "coincident": 4, "borders": 5}
CHANNEL_OFFSETS = (0, 300, 1500)   # channels 0, 1 and 2 of ``channels``
BORDER_DISTANCES = (0, 1, 2, 3)


def _rng(name, seed):
    return np.random.default_rng([274, seed, _STREAMS[name]])


def render_w266(coords, shape, seed, key):
    """uint16 ZYX image of the given centres with the W-266 isolated-spot appearance, rendered by the §2.12
    generator exactly as ``w266_common.make_case`` renders iso3d and iso_z1; truth (N, 3) ZYX."""
    book = Codebook(pd.DataFrame(dict(gene_id=["iso"], color_sequence=["1"])), ("round1",),
                    ("ch00", "ch01", "ch02", "ch03"))
    config = FormedSceneConfig(
        dataset_version="w266-isolated-v1", sample_id="w266", scene_key=f"w274/{key}", seed=seed,
        shape_zyx=tuple(shape), dtype="uint16", coordinates=tuple(map(tuple, coords)),
        brightness=ScalarDistribution(parameters=(BRIGHTNESS,)),
        axial_width=ScalarDistribution(parameters=(SIGMA_ZYX[0],)),
        lateral_width=ScalarDistribution(parameters=(SIGMA_ZYX[1],)),
        background=BackgroundConfig(baseline_enabled=True, baseline=((BASELINE,) * 4,)),
        noise=NoiseConfig(dependent_enabled=True, alpha=1.0, model="poisson", independent_enabled=True,
                          sigma=READ_NOISE))
    scene = generate_formed_scene(book, config=config)
    return scene.rounds["round1"][..., 0], scene.round_truth[["z", "y", "x"]].to_numpy(float)


def signal(shape, centres):
    """Noiseless float64 ZYX image: the baseline plus one Gaussian of peak BRIGHTNESS per centre."""
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape), indexing="ij")
    image = np.full(shape, BASELINE)
    for cz, cy, cx in centres:
        image += BRIGHTNESS * np.exp(-0.5 * (((z - cz) / SIGMA_ZYX[0]) ** 2 + ((y - cy) / SIGMA_ZYX[1]) ** 2
                                             + ((x - cx) / SIGMA_ZYX[2]) ** 2))
    return image


def render(rng, shape, centres):
    """uint16 ZYX image of the centres with the W-266 appearance: Poisson then read noise from rng."""
    noisy = rng.poisson(signal(shape, centres)) + rng.normal(0.0, READ_NOISE, shape)
    return np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)


def _jitter(rng, grid, axes=3):
    jitter = rng.uniform(-0.5, 0.5, size=(len(grid), 3))
    jitter[:, :3 - axes] = 0.0
    return np.asarray(grid, dtype=float) + jitter


def iso_z1_sparse(seed):
    """(uint16 1x64x64, truth (25, 3)): 25 spots on a 5x5 YX grid of step 12 (8 to 56), Y and X jittered in
    [-0.5, 0.5) from the seed, z=0; otherwise as iso_z1. Density 0.0061 spots per voxel."""
    grid = [(0.0, y, x) for y in range(8, 57, 12) for x in range(8, 57, 12)]
    coords = _jitter(_rng("iso_z1_sparse", seed), grid, axes=2)
    return render_w266(coords, SHAPES["iso_z1_sparse"], seed, "iso_z1_sparse")


def pair_layout(seed):
    """ZYX centres of the 12 lateral and 8 axial pairs of ``pairs`` and the kind of each member.

    Lateral pairs lie in the plane z=26, members 6 px apart in X, on rows y = 8, 22, 36, 50 and
    columns x = 6, 26, 46 (the first member); axial pairs lie in the columns y = 12, 40 and
    x = 8, 22, 36, 50, members at z = 4 and 12 (8 planes apart). Each pair moves as a whole by a
    jitter in [-0.5, 0.5) per axis from the seed (Z unjittered for the lateral pairs, which share one
    plane), so the separations are exact and pairs stay at least 12 voxels apart.
    """
    rng = _rng("pairs", seed)
    centres, kinds = [], []
    for y in (8, 22, 36, 50):
        for x in (6, 26, 46):
            dy, dx = rng.uniform(-0.5, 0.5, size=2)
            centres += [(26.0, y + dy, x + dx), (26.0, y + dy, x + 6 + dx)]
            kinds += ["lateral", "lateral"]
    for y in (12, 40):
        for x in (8, 22, 36, 50):
            dz, dy, dx = rng.uniform(-0.5, 0.5, size=3)
            centres += [(4 + dz, y + dy, x + dx), (12 + dz, y + dy, x + dx)]
            kinds += ["axial", "axial"]
    return np.array(centres), np.array(kinds)


def pairs(seed):
    """(uint16 32x64x64, truth (40, 3), kinds (40,)): the 12 lateral and 8 axial pairs. Density 3.1e-4."""
    coords, kinds = pair_layout(seed)
    image, truth = render_w266(coords, SHAPES["pairs"], seed, "pairs")
    return image, truth, kinds


def pairs_z1(seed):
    """(uint16 1x64x64, truth (24, 3)): the lateral pairs of ``pairs`` on a plane (z=0), as iso_z1 is to iso3d."""
    coords, kinds = pair_layout(seed)
    plane = coords[kinds == "lateral"].copy()
    plane[:, 0] = 0.0
    return render_w266(plane, SHAPES["pairs_z1"], seed, "pairs_z1")


def _layered_grid(rows, columns, layers):
    return [(layers[(i + j) % len(layers)], y, x) for i, y in enumerate(rows) for j, x in enumerate(columns)]


def channels(seed):
    """(uint16 16x64x64x4, truth (25, 3)): 25 spots in channel 0 (5x5 YX grid of step 12, Z layers 4, 8 and 12
    alternating, every axis jittered); channels 1 and 2 are channel 0 plus 300 and 1500 (the same noise
    realization); channel 3 holds the same spots with an independent noise draw. Density 3.8e-4 per channel."""
    rng = _rng("channels", seed)
    shape = SHAPES["channels"][:3]
    truth = _jitter(rng, _layered_grid(range(8, 57, 12), range(8, 57, 12), (4.0, 8.0, 12.0)))
    first = render(rng, shape, truth)
    copies = [first + np.uint16(offset) for offset in CHANNEL_OFFSETS]
    return np.stack(copies + [render(rng, shape, truth)], axis=-1), truth


def coincident(seed):
    """(uint16 16x64x64x2, truth (20, 3)): 20 spots at the same positions and amplitude in both channels
    (4x5 YX grid, rows 10 to 52 step 14 and columns 8 to 56 step 12, Z layers 4, 8 and 12), with an
    independent noise draw per channel. Density 3.1e-4 per channel."""
    rng = _rng("coincident", seed)
    shape = SHAPES["coincident"][:3]
    truth = _jitter(rng, _layered_grid(range(10, 53, 14), range(8, 57, 12), (4.0, 8.0, 12.0)))
    return np.stack([render(rng, shape, truth) for _ in range(2)], axis=-1), truth


def border_layout(shape):
    """(integer ZYX centres, distance to the nearest face) of ``borders``: one spot at 0, 1, 2 and 3 voxels
    from each face (Y and X faces only for Z=1), at least 8 voxels from every other spot."""
    nz, ny, nx = shape
    mid = 0 if nz == 1 else nz // 2
    along = (12, 24, 38, 52)
    centres, distances = [], []
    for d, a in zip(BORDER_DISTANCES, along):
        centres += [(mid, d, a), (mid, ny - 1 - d, a), (mid, a, d), (mid, a, nx - 1 - d)]
        distances += [d] * 4
    if nz > 1:
        low, high = ((20, 20), (20, 44), (44, 20), (44, 44)), ((32, 14), (14, 32), (32, 50), (50, 32))
        for d, (y, x), (v, u) in zip(BORDER_DISTANCES, low, high):
            centres += [(d, y, x), (nz - 1 - d, v, u)]
            distances += [d] * 2
    return np.array(centres, dtype=float), np.array(distances)


def borders(seed, z1=False):
    """(uint16 image, truth, distance to the nearest face) of ``borders``: 16x64x64 with 24 spots, or 1x64x64
    with the 16 spots of the Y and X faces; integer centres, brightness 1500."""
    shape = SHAPES["borders_z1" if z1 else "borders"]
    truth, distances = border_layout(shape)
    return render(_rng("borders", seed), shape, truth), truth, distances


def zero_channels(seed):
    """uint16 10x32x32x2 with 10 single-voxel spots (1000) per channel: channel 0 exactly 60 % zeros (MAD 0),
    channel 1 exactly 40 % zeros (MAD > 0); the other voxels are 20 to 59. 10240 voxels make both fractions
    whole numbers of voxels (8x32x32 does not)."""
    rng = np.random.default_rng([274, seed, 6])
    shape = (10, 32, 32)
    planes = []
    for zeros in (6144, 4096):
        values = rng.integers(20, 60, size=int(np.prod(shape))).astype(np.uint16)
        order = rng.permutation(values.size)
        values[order[:zeros]] = 0
        values[order[zeros:zeros + 10]] = 1000
        planes.append(values.reshape(shape))
    return np.stack(planes, axis=-1)


def probe(shape):
    """W-266's minimum-shape probe as uint16: baseline 100 and one Gaussian spot (sigma 1.3, amplitude 1500)
    at the centre (check S15)."""
    grids = np.meshgrid(*[np.arange(n, dtype=float) for n in shape], indexing="ij")
    r2 = sum((g - (n - 1) / 2) ** 2 / 1.3 ** 2 for g, n in zip(grids, shape))
    return np.rint(100 + 1500 * np.exp(-0.5 * r2)).astype(np.uint16)


def min_separation(points):
    points = np.asarray(points, dtype=float)
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return float(distances[~np.eye(len(points), dtype=bool)].min())
