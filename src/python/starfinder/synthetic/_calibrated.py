"""Calibrated development scenes: appearance derived from measured image-statistics targets.

Development calibration against the processed uint8 image statistics of
docs/image-statistics.md; not D04 and not a benchmark calibration. Every value
below is in uint8 grey levels; uint16 multiplies every intensity by 16.
"""
from itertools import product

import numpy as np
import pandas as pd

from starfinder.barcode import Codebook

from ._common import ScalarDistribution
from ._formed import FormedSceneConfig, ReadoutEffectsConfig
from ._geometry import GeometryConfig
from ._observation import BackgroundConfig, NoiseConfig, TextureConfig

#: Version of the calibrated appearance; changing any value bumps it.
CALIBRATED_VERSION = "calibrated-development-v1"
_CHANNELS = ("ch00", "ch01", "ch02", "ch03")
_ROUNDS = ("round1", "round2", "round3", "round4")
_INTENSITY_SCALE = {"uint8": 1.0, "uint16": 16.0}
_READOUT = ("weakening", "trend", "gain", "mixing")
_BACKGROUND = ("baseline", "gradient", "regions", "texture")

#: The §2.5 evaluation conditions mapped to the factors they add to the calibrated clean scene.
CALIBRATED_CONDITIONS = {
    "clean": (),
    "background_only": _BACKGROUND,
    "round_effect_only": _READOUT,
    "combined": _READOUT + _BACKGROUND,
    "baseline": ("baseline",),
    "gradient": ("gradient",),
    "regions": ("regions",),
    "texture": ("texture",),
    "gain": ("gain",),
    "trend": ("trend",),
    "gain_baseline": ("gain", "baseline"),
    "gain_texture": ("gain", "texture"),
    "combined_geometry": _READOUT + _BACKGROUND + ("translation", "local"),
}

# Every target -> parameter step is in docs/image-statistics.md, section "Calibrated preset".
_CALIBRATION = dict(
    shape_zyx=(8, 64, 64), count=80,                     # crowding: about one same-channel neighbour per annulus
    brightness_median=88.0, brightness_log_sd=0.46,      # lognormal peak amplitude A
    axial_width=1.5, lateral_width=1.3, width_log_sd=0.1,  # engineering choices (benchmark presets)
    elongation_log_sd=0.1,
    pedestal=5.0,                                        # uniform baseline in every condition
    alpha=0.25, read_sigma=1.5,                          # Poisson gain and white read noise
    correlated_sigma=6.5, correlation_length_zyx=(2.0, 4.0, 4.0),
    channel_gains=(1.0, 0.94, 0.88, 0.83),               # clean channel gains (spread 1.2)
    trend_base=0.963,                                    # clean round trend 0.963**-3 = 1.12
    # Factors added on top of clean; gain and trend keep the geometric-mean brightness.
    gain_factor=(1.08, 1.02, 0.98, 0.93),                # total spread 1.2 * 1.16 = 1.39
    trend_factor=0.96,                                   # total trend (0.963 * 0.96)**-3 = 1.27
    weakening_probability=(0.0, 0.05, 0.05, 0.05), weak_factor=0.6,
    crosstalk=0.05,                                      # bleed-through into the next channel
    baseline_offsets=(0.0, 0.5, 1.0, 1.5),               # per channel, rotated by one per round
    tissue_weights=(1.0, 0.75, 0.5, 0.25),
    gradient=(0.0, (1.0, 0.0, 2.0)),                     # intercept, ZYX slopes (normalized)
    regions=(((.5, .3, .35), 3.0), ((.5, .7, .6), 2.5), ((.5, .45, .8), 2.0)),
    region_sigma_zyx=(4.0, 10.0, 10.0),
    texture_count=8, texture_brightness=5.0, texture_width=(1.5, 3.0),
    translation_max_zyx=(0.5, 3.0, 3.0),
    local_scale=16.0, local_magnitude=1.0,
)

# GF(4) multiplication by the primitive element w: 0, 1, w, w^2 -> 0, w, w^2, 1.
_TIMES_W = (0, 2, 3, 1)
# 16 sequences whose per-round color counts are 7, 5, 3 and 1, the heavy color rotating by round
# (found once by a seeded swap search for a minimum pairwise distance of two).
_UNBALANCED = ("1211", "1232", "1244", "1334", "1424", "1431", "1442", "2134",
               "2233", "2241", "2331", "2344", "3212", "3314", "3341", "4234")


def development_codebook(variant: str = "balanced") -> Codebook:
    """Return a four-round, four-channel development codebook of 16 genes.

    ``balanced``: the codewords (a, b, a+b, a+w*b) over GF(4) (colors 1-4 for
    0, 1, w, w^2), so every color appears exactly four times in every round and
    any two codewords differ in at least three rounds. ``unbalanced`` is a
    deliberately unbalanced variant for the histogram-matching harm test: every
    round uses the four colors 7, 5, 3 and 1 times, and the most frequent color
    moves by one channel per round (any two codewords differ in at least two
    rounds). Gene IDs carry the variant name
    (``balanced-01`` ...). Rounds are round1..round4, channels ch00..ch03 with
    color k in channel k-1.
    """
    if variant == "balanced":
        sequences = ["".join(str(v + 1) for v in (a, b, a ^ b, a ^ _TIMES_W[b]))
                     for a, b in product(range(4), repeat=2)]
    elif variant == "unbalanced":
        sequences = list(_UNBALANCED)
    else:
        raise ValueError("variant must be balanced or unbalanced")
    table = pd.DataFrame(dict(gene_id=[f"{variant}-{i + 1:02d}" for i in range(len(sequences))],
                              color_sequence=sequences))
    return Codebook(table, _ROUNDS, _CHANNELS)


def calibrated_scene_preset(condition: str = "clean", dtype: str = "uint8", *, seed: int = 0,
                            codebook: str = "balanced") -> tuple[Codebook, FormedSceneConfig]:
    """Return (codebook, config) for one FOV of calibrated-development-v1.

    A development calibration against the development target table of
    docs/image-statistics.md, which records every target -> parameter step;
    not D04 and not a benchmark calibration. One 8x64x64 ZYX FOV holds 80
    uniformly placed amplicons of development_codebook(codebook) with
    lognormal brightness. Every condition, including clean, carries the
    calibrated baseline: a uniform pedestal, Poisson, white and spatially
    correlated noise, non-uniform channel gains and a mild round trend.
    ``condition`` names a CALIBRATED_CONDITIONS entry, whose factors are
    added on top (stronger gains or trend, weakening, crosstalk, per-channel
    offsets, gradient, regions, texture, translation and a local deformation).
    ``dtype`` is uint8 (the measured scale) or uint16, which multiplies every
    intensity by 16 (unverified against real data). Conditions share the
    scene key, so the amplicons and noise draws of one seed are identical
    across conditions and dtypes except where a factor changes them.
    """
    if condition not in CALIBRATED_CONDITIONS:
        raise ValueError(f"unknown calibrated condition: {condition}; choose from {list(CALIBRATED_CONDITIONS)}")
    if dtype not in _INTENSITY_SCALE:
        raise ValueError("dtype must be uint8 or uint16")
    book = development_codebook(codebook)
    enabled = set(CALIBRATED_CONDITIONS[condition])
    a, scale = _CALIBRATION, _INTENSITY_SCALE[dtype]
    rounds, shape = len(book.round_labels), np.asarray(a["shape_zyx"], dtype=float)
    lognormal = lambda median, sd: ScalarDistribution("lognormal", (float(np.log(median)), sd))  # noqa: E731

    gains = np.tile(a["channel_gains"], (rounds, 1))
    if "gain" in enabled:
        factor = np.asarray(a["gain_factor"])
        gains = gains * factor / np.exp(np.mean(np.log(factor)))
    trend = a["trend_base"]
    if "trend" in enabled:
        # Rescale every round so the geometric mean over rounds matches the clean trend.
        trend = trend * a["trend_factor"]
        gains = gains * a["trend_factor"] ** (-(rounds - 1) / 2)
    mixing = np.eye(4) + a["crosstalk"] * np.eye(4, k=-1)   # destination row s+1 from source s
    readout = ReadoutEffectsConfig(
        weakening_enabled="weakening" in enabled, weakening_probability=a["weakening_probability"],
        weak_factor=(a["weak_factor"],) * rounds,
        trend_enabled=True, trend_base=trend,
        gain_enabled=True, gains=gains.tolist(),
        mixing_enabled="mixing" in enabled, mixing=np.repeat(mixing[None], rounds, axis=0).tolist())

    baseline = np.full((rounds, 4), a["pedestal"])
    if "baseline" in enabled:
        baseline = baseline + np.array([np.roll(a["baseline_offsets"], r) for r in range(rounds)])
    regions = [(tuple(float(f * (n - 1)) for f, n in zip(fraction, shape)), height)
               for fraction, height in a["regions"]]
    axial, lateral = a["texture_width"]
    background = BackgroundConfig(
        baseline_enabled=True, baseline=(baseline * scale).tolist(),
        gradient_enabled="gradient" in enabled, gradient_intercept=a["gradient"][0] * scale,
        gradient_slopes_zyx=tuple(v * scale for v in a["gradient"][1]),
        regions_enabled="regions" in enabled, region_centers=tuple(c for c, _ in regions),
        region_sigma_zyx=(a["region_sigma_zyx"],) * len(regions),
        region_heights=tuple(h * scale for _, h in regions),
        texture_enabled="texture" in enabled,
        texture=TextureConfig(count=a["texture_count"], axial_width=lognormal(axial, a["width_log_sd"]),
                              lateral_width=lognormal(lateral, a["width_log_sd"]),
                              brightness=lognormal(a["texture_brightness"] * scale, 0.3)),
        tissue_weights=np.tile(a["tissue_weights"], (rounds, 1)).tolist())
    noise = NoiseConfig(dependent_enabled=True, alpha=a["alpha"] * scale, model="poisson",
                        independent_enabled=True, sigma=a["read_sigma"] * scale,
                        correlated_enabled=True, correlated_sigma=a["correlated_sigma"] * scale,
                        correlation_length_zyx=a["correlation_length_zyx"])
    center = tuple(float((n - 1) / 2) for n in shape)
    geometry = GeometryConfig(
        translation_enabled="translation" in enabled, translation_max_zyx=a["translation_max_zyx"],
        local_enabled="local" in enabled, centers_zyx=(center,), scales=(a["local_scale"],),
        local_magnitude=a["local_magnitude"], reference_round=book.round_labels[0])
    config = FormedSceneConfig(
        dataset_version=f"{CALIBRATED_VERSION}-{condition}-{codebook}", sample_id="synthetic",
        scene_key=CALIBRATED_VERSION, seed=seed, split="development", shape_zyx=tuple(a["shape_zyx"]), dtype=dtype,
        count=a["count"], brightness=lognormal(a["brightness_median"] * scale, a["brightness_log_sd"]),
        axial_width=lognormal(a["axial_width"], a["width_log_sd"]),
        lateral_width=lognormal(a["lateral_width"], a["width_log_sd"]),
        elongation=ScalarDistribution("folded_lognormal", (0.0, a["elongation_log_sd"])),
        angle=ScalarDistribution("uniform", (0.0, float(np.pi))),
        readout=readout, background=background, noise=noise, geometry=geometry)
    return book, config
