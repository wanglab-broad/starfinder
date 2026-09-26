"""Benchmark tier of the scene preset registry; documented, uncalibrated appearance.

Sizes, amplicon counts, seeds and shift ranges keep the historical benchmark
values. Appearance defaults are engineering choices recorded under
PRESET_VERSION; they are not calibrated against acquired images.
"""
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from starfinder.barcode import Codebook, EncodingConfig

from ._common import ScalarDistribution, _json
from ._formed import FormedSceneConfig, ReadoutEffectsConfig
from ._geometry import GeometryConfig
from ._observation import BackgroundConfig, NoiseConfig

#: Version of the benchmark appearance defaults; changing any value bumps it.
PRESET_VERSION = "benchmark-presets-v1"
_CHANNELS = ("ch00", "ch01", "ch02", "ch03")
# Historical GeneA-GeneH never use color 1 in rounds 1 and 3; GeneI-GeneL add it so
# every channel holds amplicons in every round (each color 2-4 times per round).
_TEST_CODEBOOK = (("GeneA", "CACGC"), ("GeneB", "CATGC"), ("GeneC", "CGAAC"), ("GeneD", "CGTAC"),
                  ("GeneE", "CTGAC"), ("GeneF", "CTAGC"), ("GeneG", "CCATC"), ("GeneH", "CGCTC"),
                  ("GeneI", "CAAAC"), ("GeneJ", "CAACC"), ("GeneK", "CCCGC"), ("GeneL", "CCTCC"))

#: Benchmark presets: historical ZYX shape, amplicons per FOV, seed, FOV count,
#: e2e and registration translation half-ranges (z, yx) in voxels, and genes.
BENCHMARK_PRESETS = {
    "tiny": dict(shape_zyx=(8, 128, 128), count=10, seed=42, fovs=2,
                 e2e_shift=(2, 5), registration_shift=(2, 10), genes=12),
    "small": dict(shape_zyx=(16, 256, 256), count=50, seed=42, fovs=2,
                  e2e_shift=(2, 5), registration_shift=(4, 25), genes=12),
    "medium": dict(shape_zyx=(32, 512, 512), count=400, seed=42, fovs=2,
                   e2e_shift=(8, 50), registration_shift=(8, 50), genes=12),
    "large": dict(shape_zyx=(30, 1024, 1024), count=1500, seed=123, fovs=2,
                  e2e_shift=(7, 100), registration_shift=(7, 100), genes=64),
    "tissue": dict(shape_zyx=(30, 3072, 3072), count=14000, seed=456, fovs=2,
                   e2e_shift=(7, 300), registration_shift=(7, 300), genes=64),
    "thick_medium": dict(shape_zyx=(100, 1024, 1024), count=5200, seed=789, fovs=2,
                         e2e_shift=(25, 100), registration_shift=(25, 100), genes=64),
}

#: Registration deformations: percent of min(Y, X) with a pixel cap for YX
#: (percent of Z for Z); RBF radius as percent of min(Y, X).
DEFORMATION_PRESETS = {
    "polynomial_small": dict(kind="polynomial", percent=3.0, cap_px=15.0),
    "polynomial_large": dict(kind="polynomial", percent=6.0, cap_px=30.0),
    "gaussian_small": dict(kind="rbf", percent=3.0, radius_percent=6.0, cap_px=15.0, points=1),
    "gaussian_large": dict(kind="rbf", percent=6.0, radius_percent=10.0, cap_px=30.0, points=1),
    "multi_point": dict(kind="rbf", percent=4.0, radius_percent=5.0, cap_px=20.0, points=4),
    "linear_small": dict(kind="affine", percent=1.0, cap_px=10.0),
}

# Documented appearance (intensity units of the uint16 output). See docs/api/synthetic.rst.
_APPEARANCE = dict(
    brightness_median=1500.0, brightness_log_sd=0.4,     # lognormal peak amplitude
    axial_width=1.5, lateral_width=1.3, width_log_sd=0.1,  # lognormal Gaussian sigmas, voxels
    elongation_log_sd=0.1,                               # folded lognormal, uniform angle
    trend_base=0.95,                                     # 5% signal loss per round
    crosstalk=0.05,                                      # bleed-through into the next channel
    baseline=100.0,                                      # camera offset per channel
    tissue_weights=(40.0, 30.0, 30.0, 20.0),             # per-channel tissue background
    gradient=(0.5, (0.0, 0.5, 0.5)),                     # intercept, ZYX slopes (normalized)
    regions=(((.5, .3, .35), 1.0), ((.5, .7, .6), .8), ((.5, .45, .8), .6)),
    alpha=1.0, read_sigma=3.0,                           # Poisson gain and read noise
)
# Intensity scale per output dtype: uint8 keeps spots below 255 at typical brightness.
_INTENSITY_SCALE = {"uint16": 1.0, "uint8": 1 / 16, "float32": 1.0, "float64": 1.0}


def generate_codebook(n_genes: int, *, rounds: int | None = None) -> Codebook:
    """Return a Codebook of n_genes CNNNC barcodes with distinct color sequences.

    Barcodes enumerate C-{A,C,G,T}^3-C (at most 64 genes), encoded with
    reverse_bases=True as four-color sequences. Round labels are round1..round4
    (the color sequence length; ``rounds`` may state it explicitly) and the
    channels are ch00..ch03 with color k in channel k-1.
    """
    encoding = EncodingConfig(reverse_bases=True)
    seen, entries = set(), []
    for middle in product("ACGT", repeat=3):
        barcode = f"C{''.join(middle)}C"
        colors = encoding.encode(barcode)
        if colors not in seen:
            seen.add(colors)
            entries.append((f"Gene{len(entries) + 1:03d}", barcode))
    if type(n_genes) is not int or not 1 <= n_genes <= len(entries):
        raise ValueError(f"n_genes must be in [1, {len(entries)}] unique color sequences")
    return _codebook(entries[:n_genes], rounds)


def _codebook(entries, rounds=None):
    encoding = EncodingConfig(reverse_bases=True)
    colors = [encoding.encode(barcode) for _, barcode in entries]
    length = len(colors[0])
    if rounds is not None and rounds != length:
        raise ValueError(f"codebook sequences have {length} rounds, not {rounds}")
    table = pd.DataFrame(dict(gene_id=[g for g, _ in entries], color_sequence=colors,
                              base_sequence=[b for _, b in entries]))
    return Codebook(table, tuple(f"round{r + 1}" for r in range(length)), _CHANNELS, encoding=encoding)


def _appearance(shape, rounds, dtype):
    """Formed-scene fields shared by every benchmark preset (see PRESET_VERSION)."""
    if dtype not in _INTENSITY_SCALE:
        raise ValueError("dtype must be uint8, uint16, float32 or float64")
    a, scale = _APPEARANCE, _INTENSITY_SCALE[dtype]
    shape = np.asarray(shape, dtype=float)
    lognormal = lambda median, sd: ScalarDistribution("lognormal", (float(np.log(median)), sd))  # noqa: E731
    mixing = np.eye(4) + a["crosstalk"] * np.eye(4, k=-1)   # destination row s+1 from source s
    regions = [(tuple(float(f * (n - 1)) for f, n in zip(fraction, shape)), height)
               for fraction, height in a["regions"]]
    sigma = tuple(float(v) for v in np.maximum(1, [shape[0] / 2, shape[1] / 5, shape[2] / 5]))
    if shape[0] == 1:
        regions = [((0.0, y, x), h) for (_, y, x), h in regions]
    return dict(
        dtype=dtype,
        brightness=lognormal(a["brightness_median"] * scale, a["brightness_log_sd"]),
        axial_width=lognormal(a["axial_width"], a["width_log_sd"]),
        lateral_width=lognormal(a["lateral_width"], a["width_log_sd"]),
        elongation=ScalarDistribution("folded_lognormal", (0.0, a["elongation_log_sd"])),
        angle=ScalarDistribution("uniform", (0.0, float(np.pi))),
        readout=ReadoutEffectsConfig(trend_enabled=True, trend_base=a["trend_base"],
                                     mixing_enabled=True, mixing=np.repeat(mixing[None], rounds, axis=0)),
        background=BackgroundConfig(
            baseline_enabled=True, baseline=np.full((rounds, 4), a["baseline"] * scale),
            gradient_enabled=True, gradient_intercept=a["gradient"][0], gradient_slopes_zyx=a["gradient"][1],
            regions_enabled=True, region_centers=tuple(c for c, _ in regions),
            region_sigma_zyx=(sigma,) * len(regions), region_heights=tuple(h for _, h in regions),
            tissue_weights=np.tile(np.asarray(a["tissue_weights"]) * scale, (rounds, 1))),
        noise=NoiseConfig(dependent_enabled=True, alpha=a["alpha"] * scale, model="poisson",
                          independent_enabled=True, sigma=a["read_sigma"] * scale))


def _preset(name):
    if name not in BENCHMARK_PRESETS:
        raise ValueError(f"unknown benchmark preset: {name}; choose from {list(BENCHMARK_PRESETS)}")
    return BENCHMARK_PRESETS[name]


def benchmark_scene_preset(name: str, *, dtype: str = "uint16") -> tuple[Codebook, FormedSceneConfig]:
    """Return (codebook, config) for one FOV of an e2e benchmark preset.

    Rounds follow the codebook (four), round1 is the reference and every other
    round receives a uniform translation within the preset's e2e half-range.
    Appearance follows PRESET_VERSION; uint8 scales every intensity by 1/16.
    Pass the config to generate_dataset with the FOV IDs to generate.
    """
    spec = _preset(name)
    codebook = _codebook(_TEST_CODEBOOK) if spec["genes"] == 12 else generate_codebook(spec["genes"])
    rounds = len(codebook.round_labels)
    z, yx = spec["e2e_shift"]
    shape = spec["shape_zyx"]
    geometry = GeometryConfig(translation_enabled=True, reference_round=codebook.round_labels[0],
                              translation_max_zyx=(z if shape[0] > 1 else 0, yx, yx))
    config = FormedSceneConfig(
        dataset_version=f"{PRESET_VERSION}-{name}", sample_id="synthetic", scene_key=f"{PRESET_VERSION}/{name}",
        seed=spec["seed"], split="development", shape_zyx=shape, count=spec["count"],
        max_count=max(1024, spec["count"]), geometry=geometry, **_appearance(shape, rounds, dtype))
    return codebook, config


def deformation_geometry(name: str, shape_zyx, *, reference_round: str | None = None,
                         translation_max_zyx=None) -> GeometryConfig:
    """Map a historical deformation name onto GeometryConfig for one grid shape.

    YX magnitudes are percent of min(Y, X) capped at cap_px, Z magnitudes are
    percent of Z. polynomial_* and linear_small draw bounded polynomial or
    affine coefficients; gaussian_* and multi_point use Gaussian RBF controls
    with fixed centres, random directions and the percent radius. RBF
    magnitudes are reduced when needed to meet the conservative invertibility
    bound (sum of norm(v)/(scale*sqrt(e)) <= 0.5); polynomial and affine draws
    are reduced in the same way when their Jacobian bound would exceed it.
    ``shift`` gives translation only.
    """
    shape = tuple(int(n) for n in shape_zyx)
    translation = dict(translation_enabled=translation_max_zyx is not None,
                       translation_max_zyx=translation_max_zyx, reference_round=reference_round)
    if name == "shift":
        return GeometryConfig(**translation)
    if name not in DEFORMATION_PRESETS:
        raise ValueError(f"unknown deformation: {name}; choose shift or {list(DEFORMATION_PRESETS)}")
    spec = DEFORMATION_PRESETS[name]
    lateral = min(spec["percent"] * min(shape[1:]) / 100, spec["cap_px"])
    axial = spec["percent"] * shape[0] / 100 if shape[0] > 1 else 0.0
    bound = (axial, lateral, lateral)
    if spec["kind"] == "polynomial":
        return GeometryConfig(**translation, polynomial_enabled=True, polynomial_max_zyx=bound)
    if spec["kind"] == "affine":
        return GeometryConfig(**translation, affine_enabled=True, affine_max_zyx=bound)
    radius = spec["radius_percent"] * min(shape[1:]) / 100
    fractions = ([(.5, .4, .6)] if spec["points"] == 1 else
                 [(.5, .3, .3), (.5, .3, .7), (.5, .7, .3), (.5, .7, .7)])
    centers = tuple(tuple(float(f * (n - 1)) for f, n in zip(fraction, shape)) for fraction in fractions)
    # 0.999 keeps the float bound strictly inside the limit after summation.
    magnitude = min(lateral, .999 * .5 * np.sqrt(np.e) * radius / len(centers))
    return GeometryConfig(**translation, local_enabled=True, centers_zyx=centers,
                          scales=(radius,) * len(centers), local_magnitude=float(magnitude))


def registration_scene_preset(name: str, deformation: str = "shift", *,
                              dtype: str = "uint16") -> tuple[Codebook, FormedSceneConfig]:
    """Return (codebook, config) for a reference/moving registration pair.

    The codebook has rounds (reference, <deformation>) and one gene whose
    signal is in ch00; only the moving round is moved. Every pair of a preset
    shares the scene key, so reference images and amplicons are identical.
    """
    spec = _preset(name)
    shape = spec["shape_zyx"]
    rounds = ("reference", deformation)
    codebook = Codebook(pd.DataFrame(dict(gene_id=["registration"], color_sequence=["11"])),
                        rounds, _CHANNELS)
    z, yx = spec["registration_shift"]
    shift = (z if shape[0] > 1 else 0, yx, yx) if deformation == "shift" else None
    geometry = deformation_geometry(deformation, shape, reference_round="reference",
                                    translation_max_zyx=shift)
    config = FormedSceneConfig(
        dataset_version=f"{PRESET_VERSION}-{name}-registration", sample_id="synthetic", FOV_id=name,
        scene_key=_json([PRESET_VERSION, name, "registration"]), seed=spec["seed"],
        shape_zyx=shape, count=spec["count"], max_count=max(1024, spec["count"]), geometry=geometry,
        **_appearance(shape, len(rounds), dtype))
    return codebook, config


def _estimate_peak_bytes(shape_zyx, count=0, *, dtype="uint16", accumulation="float32", channels=4):
    """Upper estimate of generation working memory for one round, in bytes.

    Output round (ZYXC), one accumulation plane, the sampled background plane,
    noise draws for one chunk, one inverse-geometry block with temporaries and
    the round's cached kernels (float64 boxes of up to 16^3 voxels).
    """
    from ._geometry import _BLOCK_VOXELS
    from ._observation import _NOISE_CHUNK
    voxels = int(np.prod(shape_zyx))
    item = np.dtype(accumulation).itemsize
    return (voxels * channels * np.dtype(dtype).itemsize + 2 * voxels * item
            + 4 * _NOISE_CHUNK * 8 + 12 * min(voxels, _BLOCK_VOXELS) * 8 + count * 16**3 * 8)


def _with_seed(config, seed):
    return config if seed is None else replace(config, seed=seed)


def _registry():
    from ._development import DEVELOPMENT_FIXTURES, DEVELOPMENT_SIZES
    fixtures = {name: dict(tier="fixture", shape_zyx=(1 if name == "formed-z1-v1" else 8, 32, 32),
                           count=8, seed=42, accumulation="float64", oracle="none; literal tests")
                for name in ("formed-small-v1", "formed-z1-v1")}
    development = {name: dict(tier="development_fixture", shape_zyx=DEVELOPMENT_SIZES[size], count=2,
                              seed=42, accumulation="float64", condition=condition,
                              oracle="independent full-grid oracle")
                   for name, (condition, size) in DEVELOPMENT_FIXTURES.items()}
    benchmark = {name: dict(tier="benchmark", version=PRESET_VERSION, accumulation="float32",
                            dtype="uint16", peak_bytes_estimate=_estimate_peak_bytes(spec["shape_zyx"], spec["count"]),
                            **spec)
                 for name, spec in BENCHMARK_PRESETS.items()}
    return {**fixtures, **development, **benchmark}
