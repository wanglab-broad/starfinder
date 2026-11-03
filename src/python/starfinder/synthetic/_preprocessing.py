"""Conditions extracted from the preprocessing development evaluation.

The public preset registries remain unchanged; these controls build on them.
"""
from dataclasses import replace
import math
import numpy as np
from ._common import ScalarDistribution
from ._observation import TextureConfig
from ._development import development_scene_preset
from ._calibrated import CALIBRATED_CONDITIONS, calibrated_scene_preset
from ._formed import AmpliconSceneConfig
from starfinder.readout import Codebook

__all__ = ["preprocessing_scene_config", "calibrated_preprocessing_scene_config",
           "uncalibrated_preprocessing_scene_config"]

INTENSITY_SCALE = {"uint8": 12.0, "uint16": 192.0}

CONDITIONS = {
    "clean": {},
    "background_only": {"background": "combined"},
    "round_effect_only": {"readout": "combined"},
    "combined": {"readout": "combined", "background": "combined", "noise": "combined"},
    "baseline": {"background": "baseline"},
    "gradient": {"background": "gradient"},
    "regions": {"background": "regions"},
    "texture": {"background": "texture"},
    "gain": {"readout": "gain"},
    "trend": {"readout": "trend"},
    "gain_baseline": {"readout": "gain", "background": "baseline"},
    "gain_texture": {"readout": "gain", "background": "texture"},
    "combined_geometry": {"readout": "combined", "background": "combined", "noise": "combined",
                          "geometry": "combined"},
}

GRADIENT_SLOPES_ZYX = (2.0, 0.0, 2.0)

CALIBRATED_VERSION = "calibrated-development-v1"

ADDED_CONDITIONS = ("bright_outliers", "saturation", "gain_strong", "clean_unbalanced")

CALIBRATED_ALL = tuple(CALIBRATED_CONDITIONS) + ADDED_CONDITIONS

#: A pilot identity, explicitly not a calibrated appearance version.
UNCALIBRATED_PREPROCESSING_VERSION = "uncalibrated-preprocessing-pilot-v1"
UNCALIBRATED_PREPROCESSING_CONDITIONS = (
    "clean", "combined", "baseline", "gradient", "regions", "texture", "gain", "trend",
    "gain_strong", "bright_outliers", "saturation", "clean_unbalanced", "dense",
)

BRIGHT_OUTLIERS = dict(count=4, axial_width=1.5, lateral_width=2.0, brightness_factor=4.0, log_sd=0.1)

GAIN_STRONG_SPREAD = 3.15


def _scaled(config, scale):
    bg, noise = config.background, config.noise
    mult = lambda v: None if v is None else (np.asarray(v, dtype=float) * scale).tolist()  # noqa: E731
    return replace(config,
        brightness=ScalarDistribution(parameters=(float(config.brightness.parameters[0]) * scale,)),
        background=replace(bg, baseline=mult(bg.baseline), tissue_weights=mult(bg.tissue_weights)),
        noise=replace(noise, alpha=noise.alpha * scale, sigma=noise.sigma * scale))



def preprocessing_scene_config(condition, *, dtype, seed, shape, count, parts=None, fov_id="FOV_001", gain=1.0,
                               scene_key=None) -> tuple[Codebook, AmpliconSceneConfig]:
    """Configure one preprocessing development condition without generating it.

    Parameters
    ----------
    condition : str
        Name in PREPROCESSING_CONDITIONS.
    dtype : str
        Output dtype, uint8 or uint16, with the original intensity scaling.
    seed : int
        Scene random seed.
    shape : tuple of int
        ZYX image shape.
    count : int
        Number of amplicons.
    parts : mapping or None
        Field-group to development-condition mapping, overriding the registry.
    fov_id : str
        FOV identifier.
    gain : float
        Uniform multiplier of all readout gains.
    scene_key : str or None
        Random stream scene key; None keeps the clean preset key.

    Returns
    -------
    tuple of Codebook and AmpliconSceneConfig
        Codebook and configured scene, with the historical W-233 values.
    """
    parts = CONDITIONS[condition] if parts is None else parts
    book, config = development_scene_preset("clean")
    sources = {group: development_scene_preset(name)[1] for group, name in parts.items()}
    config = replace(config, **{group: getattr(source, group) for group, source in sources.items()})
    if parts.get("background") == "gradient":
        config = replace(config, background=replace(config.background, gradient_slopes_zyx=GRADIENT_SLOPES_ZYX))
    if gain != 1.0:
        readout = config.readout
        gains = np.ones((3, 4)) if not readout.gain_enabled else np.asarray(readout.gains, dtype=float)
        config = replace(config, readout=replace(readout, gain_enabled=True, gains=(gains * gain).tolist()))
    config = replace(config, dataset_version=f"controlled-development-v1-w233-{condition}", FOV_id=fov_id,
        scene_key=scene_key or config.scene_key, shape_zyx=tuple(shape), coordinates=None, count=count,
        amplicon_ids=None, gene_ids=None, elongation=ScalarDistribution("uniform", (1.0, 1.5)),
        angle=ScalarDistribution("uniform", (0.0, float(np.pi))), seed=seed, dtype=dtype)
    return book, _scaled(config, INTENSITY_SCALE[dtype])



def _strong_gain_factors(clean_gains, spread=GAIN_STRONG_SPREAD):
    """Channel factors for gain_strong: log-linear across channels with geometric mean 1, chosen so the
    clean channel gains times the factors span max/min = spread (the brightest clean channel is first)."""
    gains = np.asarray(clean_gains, dtype=np.float64)
    if not (np.all(np.diff(gains) <= 0) and gains[-1] > 0):
        raise ValueError("clean channel gains must be positive and non-increasing")
    own = spread * gains[-1] / gains[0]
    position = ((len(gains) - 1) / 2 - np.arange(len(gains))) / (len(gains) - 1)
    return own ** position



def _saturation_config(config, k):
    """Item 5: multiply every intensity parameter the uint16 variant multiplies by 16 and that the clean
    scene uses (brightness median, pedestal, Poisson alpha, white sigma and correlated sigma) by k."""
    background, noise = config.background, config.noise
    if background.gradient_enabled or background.regions_enabled or background.texture_enabled:
        raise ValueError("saturation is defined on the calibrated clean scene")
    location, log_sd = config.brightness.parameters
    return replace(config,
        brightness=ScalarDistribution("lognormal", (float(location + math.log(k)), float(log_sd))),
        background=replace(background, baseline=(np.asarray(background.baseline, dtype=float) * k).tolist()),
        noise=replace(noise, alpha=noise.alpha * k, sigma=noise.sigma * k, correlated_sigma=noise.correlated_sigma * k))



def calibrated_preprocessing_scene_config(condition, *, dtype, seed, k=None, fov_id=None, gain=1.0, count=None,
                                          scene_key=None, shape=None) -> tuple[Codebook, AmpliconSceneConfig]:
    """Configure a calibrated preprocessing condition without generating it.

    Parameters
    ----------
    condition : str
        Calibrated base name or an EXTRA_CALIBRATED_CONDITIONS name. Extras
        build on clean; only clean_unbalanced uses the unbalanced codebook.
    dtype : str
        Calibrated output dtype, uint8 or uint16.
    seed : int
        Scene random seed.
    k : float or None
        Required searched intensity multiplier for saturation.
    fov_id : str or None
        Replacement FOV identifier; None keeps the preset value.
    gain : float
        Uniform multiplier of readout gains.
    count : int or None
        Replacement amplicon count; clears fixed coordinates and identities.
    scene_key : str or None
        Replacement random stream scene key.
    shape : tuple of int or None
        Replacement ZYX shape; clears fixed coordinates and identities.

    Returns
    -------
    tuple of Codebook and AmpliconSceneConfig
        Codebook and configured scene, preserving the W-239 field changes.
    """
    if condition not in CALIBRATED_ALL:
        raise ValueError(f"unknown calibrated condition: {condition}")
    codebook = "unbalanced" if condition == "clean_unbalanced" else "balanced"
    base = condition if condition in CALIBRATED_CONDITIONS else "clean"
    book, config = calibrated_scene_preset(base, dtype, seed=seed, codebook=codebook)
    if config.scene_key != CALIBRATED_VERSION:
        raise RuntimeError(f"expected the {CALIBRATED_VERSION} preset, got scene key {config.scene_key}")
    if condition == "bright_outliers":
        spec, (location, _) = BRIGHT_OUTLIERS, config.brightness.parameters
        rounds, channels = len(book.round_labels), len(book.channel_labels)
        texture = TextureConfig(count=spec["count"], axial_width=ScalarDistribution("constant", (spec["axial_width"],)),
                                lateral_width=ScalarDistribution("constant", (spec["lateral_width"],)),
                                brightness=ScalarDistribution("lognormal", (float(location + math.log(
                                    spec["brightness_factor"])), spec["log_sd"])))
        config = replace(config, background=replace(config.background, texture_enabled=True, texture=texture,
                                                     tissue_weights=np.ones((rounds, channels)).tolist()))
    elif condition == "saturation":
        if k is None:
            raise ValueError("saturation needs the searched k")
        config = _saturation_config(config, k)
    elif condition == "gain_strong":
        gains = np.asarray(config.readout.gains, dtype=float)
        config = replace(config, readout=replace(config.readout, gains=(gains * _strong_gain_factors(gains[0])).tolist()))
    if condition in ADDED_CONDITIONS:
        config = replace(config, dataset_version=f"{CALIBRATED_VERSION}-{condition}-{codebook}")
    if gain != 1.0:
        config = replace(config, readout=replace(config.readout,
                                                 gains=(np.asarray(config.readout.gains, dtype=float) * gain).tolist()))
    if count is not None or shape is not None:
        config = replace(config, count=config.count if count is None else count, coordinates=None, amplicon_ids=None,
                         gene_ids=None, shape_zyx=config.shape_zyx if shape is None else tuple(shape))
    if fov_id is not None:
        config = replace(config, FOV_id=fov_id)
    if scene_key is not None:
        config = replace(config, scene_key=scene_key)
    return book, config


def uncalibrated_preprocessing_scene_config(
        condition: str = "clean", *, seed: int = 0, k: float | None = None
) -> tuple[Codebook, AmpliconSceneConfig]:
    """Configure an explicitly uncalibrated uint8 preprocessing pilot scene.

    Parameters
    ----------
    condition : str
        Name in UNCALIBRATED_PREPROCESSING_CONDITIONS. ``dense`` adds only molecule
        density to clean; other names retain their calibrated controls.
    seed : int
        Root seed for the scene random streams.
    k : float or None
        Required intensity multiplier for the saturation condition, passed
        through to calibrated_preprocessing_scene_config.

    Returns
    -------
    tuple of Codebook and AmpliconSceneConfig
        Four-round, four-channel codebook and 30 x 512 x 512 uint8 scene with
        5,000 uniformly placed molecules, or 12,000 for ``dense``.

    Notes
    -----
    Clean at seed 0 misses clutter SNR (14.16 versus 6.05--9.96), pixel SNR
    (37.00 versus 14.5--32) and channel gain spread (1.152 versus 1.28--2.64).
    Combined misses clutter SNR (12.65) and pixel SNR (32.39) against the same
    ranges. The complete measured comparisons and target sources are in
    ``runs/W-779/20261103T040857Z-5f9d78e8/results/statistics/clean-against-targets.csv``
    and ``combined-against-targets.csv`` beside it, outside the repository.
    This preset has no calibrated version; its cleaner appearance makes
    absolute performance optimistic. Calibration belongs to W-93.

    Only molecule count, texture count and region extent change from the small preset.
    Texture density is preserved with the volume ratio 240; region centres
    keep their fractional positions and region sigma is (15, 80, 80).
    Allocation guards grow to fit the explicit counts. Spot widths, noise
    and correlation lengths are unchanged. Background opening radii belong
    to preprocessing configurations, not this scene; no Z-radius cap is
    applied here. The original candidate's random-stream key is retained so
    the measured images and corrected local pilot remain unchanged.
    """
    if condition not in UNCALIBRATED_PREPROCESSING_CONDITIONS:
        raise ValueError(f"unknown uncalibrated preprocessing condition: {condition}")
    base = "clean" if condition == "dense" else condition
    book, config = calibrated_preprocessing_scene_config(base, dtype="uint8", seed=seed, k=k)
    shape = (30, 512, 512)
    count = 12000 if condition == "dense" else 5000
    background = config.background
    ratio = math.prod(shape) // math.prod(config.shape_zyx)
    texture_count = background.texture.count * ratio
    centres = tuple(tuple(float(c * (new - 1) / (old - 1))
                          for c, old, new in zip(centre, config.shape_zyx, shape))
                    for centre in background.region_centers)
    background = replace(background, region_centers=centres,
                         region_sigma_zyx=((15.0, 80.0, 80.0),) * len(centres),
                         texture=replace(background.texture, count=texture_count,
                                         max_count=max(background.texture.max_count, texture_count)))
    return book, replace(config, dataset_version=f"{UNCALIBRATED_PREPROCESSING_VERSION}-{condition}",
                         scene_key="scaled-calibrated-preprocessing-candidate", shape_zyx=shape,
                         count=count, max_count=count, background=background)


