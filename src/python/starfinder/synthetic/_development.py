"""Controlled development scenes: four packaged fixtures and on-demand factors."""
from dataclasses import replace

import numpy as np

from ._common import ScalarDistribution
from ._formed import ReadoutEffectsConfig, formed_scene_preset
from ._geometry import GeometryConfig
from ._observation import BackgroundConfig, NoiseConfig, TextureConfig

DEVELOPMENT_SIZES = {"z1": (1, 32, 32), "small": (9, 32, 32)}
DEVELOPMENT_FACTORS = (
    "brightness", "axial_width", "lateral_width", "elongation", "placement",
    "dropout", "weakening", "trend", "loss", "gain", "mixing", "baseline",
    "gradient", "regions", "texture", "dependent_noise", "independent_noise",
    "translation", "local",
)
DEVELOPMENT_FIXTURES = {f"{size}-{condition}": (condition, size)
                        for size in DEVELOPMENT_SIZES for condition in ("clean", "combined")}


def development_scene_preset(condition="clean", *, size="small"):
    """Return a codebook/config for controlled-development-v1.

    ``condition`` is clean, combined or one of DEVELOPMENT_FACTORS. ``size``
    is z1 or small. Each size shares one reference scene across conditions,
    with seed 42 and a fixed scene key; variant names never enter random keys.
    The scene holds gt-A (e=1, theta=0) and gt-B (e=1.5, theta=pi/6).
    Combined enables all readout/background/noise/geometry controls except loss
    and dropout (so weakening/recovery and geometry remain visible).
    Only DEVELOPMENT_FIXTURES (clean/combined in each size) are packaged and
    checked against an independent oracle; single-factor conditions are
    on-demand comparisons against clean. None is calibrated.
    """
    if size not in DEVELOPMENT_SIZES:
        raise ValueError(f"unknown development size: {size}")
    if condition not in ("clean", "combined", *DEVELOPMENT_FACTORS):
        raise ValueError(f"unknown development condition: {condition}")
    book, base = formed_scene_preset()
    shape = DEVELOPMENT_SIZES[size]
    z = shape[0] // 2
    center = (z, 10, 10)
    base = replace(base, dataset_version=f"controlled-development-v1-{size}-{condition}",
        scene_key="controlled-development-v1", shape_zyx=shape,
        coordinates=(center, (z, 22, 22)), amplicon_ids=("gt-A", "gt-B"),
        gene_ids={"gt-A": "gene-A", "gt-B": "gene-B"},
        brightness=ScalarDistribution(parameters=(8,)),
        lateral_width=ScalarDistribution(parameters=(1.25,)),
        elongation=ScalarDistribution("supplied", (), {"gt-A": 1, "gt-B": 1.5}),
        angle=ScalarDistribution("supplied", (), {"gt-A": 0, "gt-B": np.pi / 6}))
    enabled = set(development_preset_factors(condition))
    changes = {}
    for name, value in dict(brightness=16, axial_width=1.5, lateral_width=2,
                            elongation=2).items():
        if name in enabled:
            changes[name] = ScalarDistribution(parameters=(value,))
    if "placement" in enabled:
        changes.update(coordinates=None, count=2, placement="clustered",
            cluster_centers=(center,), cluster_weights=(1,), spread_zyx=(1, 2, 2))
    mixing = ((1, .25, 0, 0), (0, 1, 0, 0), (0, 0, 1, .25), (0, 0, 0, 1))
    readout = ReadoutEffectsConfig(
        dropout_enabled="dropout" in enabled, dropout_probability=(0, 1, 0),
        weakening_enabled="weakening" in enabled, weakening_probability=(0, 1, 0), weak_factor=(1, .25, 1),
        trend_enabled="trend" in enabled, trend_base=.5,
        loss_enabled="loss" in enabled, loss_probability=1, loss_start=1,
        gain_enabled="gain" in enabled, gains=((.5, .5, .5, .5),)*3,
        mixing_enabled="mixing" in enabled, mixing=(mixing,)*3)
    background = BackgroundConfig(
        baseline_enabled="baseline" in enabled, baseline=((1, 2, 3, 4),)*3,
        gradient_enabled="gradient" in enabled, gradient_intercept=1, gradient_slopes_zyx=(0, 0, 2),
        regions_enabled="regions" in enabled, region_centers=(center,),
        region_sigma_zyx=((2, 5, 5),), region_heights=(3,),
        texture_enabled="texture" in enabled, texture=TextureConfig(count=2),
        tissue_weights=((1, .5, .25, 0),)*3)
    geometry = GeometryConfig(translation_enabled="translation" in enabled,
        translations_zyx=((0, 0, 0), (0, .5, -.5), (0, -1, 1)),
        local_enabled="local" in enabled, centers_zyx=(center,), scales=(8,),
        vectors_zyx=(((0, 0, 0),), ((0, 0, .5),), ((0, .25, 0),)))
    return book, replace(base, **changes, readout=readout, background=background,
        noise=NoiseConfig(dependent_enabled="dependent_noise" in enabled, alpha=.25,
                          independent_enabled="independent_noise" in enabled, sigma=.5),
        geometry=geometry)


def development_preset_factors(condition):
    """Return the exact named enabled factors; reject unknown conditions."""
    if condition == "clean":
        return ()
    if condition == "combined":
        return tuple(f for f in DEVELOPMENT_FACTORS[5:] if f not in ("dropout", "loss"))
    if condition in DEVELOPMENT_FACTORS:
        return (condition,)
    raise ValueError(f"unknown development condition: {condition}")
