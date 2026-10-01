"""In-session §2.7 spot-finding scenes (docs/spot-finding-algorithms.md, "Engineering validation design").

isolated_scene ports W-266 ``w266_common.make_case`` for the isolated-spot cases (iso3d,
iso_z1); formed16 is the §2.12 ``formed_scene_preset("small")`` appearance and codebook
at 16x64x64 with 24 amplicons (check S16). Nothing is written to disk.
"""
from dataclasses import replace

import pandas as pd

from starfinder.barcode import Codebook
from starfinder.synthetic import (BackgroundConfig, FormedSceneConfig, NoiseConfig, ScalarDistribution,
    formed_scene_preset, generate_formed_scene)
from starfinder.synthetic._presets import _appearance

from .test_spot_finding_metrics import isolated_positions

SEEDS = (100, 101, 102)
ISOLATED_SHAPES = {"iso3d": (32, 64, 64), "iso_z1": (1, 64, 64)}
FORMED16_SHAPE = (16, 64, 64)


def isolated_scene(case, seed):
    """(uint16 ZYX image, truth (100, 3) ZYX): W-266's isolated-spot scene, rendered by the §2.12 generator.

    100 spots on a 10x10 YX grid of step 6 (Z layers 8, 16, 24 for iso3d; Z=0 for
    iso_z1), brightness 1500, sigma Z 1.5 and YX 1.3, baseline 100, Poisson (alpha 1)
    plus read noise 3; truth is the rendered centres.
    """
    coords = isolated_positions(case, seed)
    book = Codebook(pd.DataFrame(dict(gene_id=["iso"], color_sequence=["1"])), ("round1",),
                    ("ch00", "ch01", "ch02", "ch03"))
    config = FormedSceneConfig(
        dataset_version="w266-isolated-v1", sample_id="w266", scene_key=f"w266/{case}", seed=seed,
        shape_zyx=ISOLATED_SHAPES[case], dtype="uint16", coordinates=tuple(map(tuple, coords)),
        brightness=ScalarDistribution(parameters=(1500.0,)),
        axial_width=ScalarDistribution(parameters=(1.5,)), lateral_width=ScalarDistribution(parameters=(1.3,)),
        background=BackgroundConfig(baseline_enabled=True, baseline=((100.0, 100.0, 100.0, 100.0),)),
        noise=NoiseConfig(dependent_enabled=True, alpha=1.0, model="poisson", independent_enabled=True, sigma=3.0))
    scene = generate_formed_scene(book, config=config)
    return scene.rounds["round1"][..., 0], scene.round_truth[["z", "y", "x"]].to_numpy(float)


def formed16(seed):
    """(reference-round uint16 ZYXC image, its metadata, truth (24, 3) ZYX, eligible mask) of the S16 scene.

    The ``small`` benchmark preset (codebook, four rounds, translations, appearance)
    with the shape 16x64x64, 24 amplicons and the seed replaced; the appearance is
    re-derived for the shape, as W-266 did for its 1x512x512 case. Truth is the
    formed amplicon centres; eligible is ``center_in_bounds`` in the reference
    round (the W-218 notebook policy).
    """
    book, config = formed_scene_preset("small")
    config = replace(config, shape_zyx=FORMED16_SHAPE, count=24, seed=seed,
                     **_appearance(FORMED16_SHAPE, len(book.round_labels), "uint16"))
    scene = generate_formed_scene(book, config=config)
    reference = scene.round_labels[0]
    in_frame = scene.round_truth.query("round_label == @reference").set_index("amplicon_id")
    eligible = in_frame.loc[scene.formed.amplicon_id, "center_in_bounds"].to_numpy(dtype=bool)
    truth = scene.formed[["z", "y", "x"]].to_numpy(float)
    return scene.rounds[reference], scene.round_metadata[reference], truth, eligible
