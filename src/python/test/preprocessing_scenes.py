"""Explicit bounded package scenes and preprocessing recipes for the preprocessing evaluation tests (W-781).

The retired benchmarks/preprocessing_synthetic.py (W-233, W-239; archived at 8094b30) built its scenes and recipe
arms from public package names. These are those definitions, without its orchestration (threshold sweeps,
development-seed selection, comparison tables, flags or budgets): a scene from ``starfinder.synthetic``'s
preprocessing configurations, its background and signal truth, the recipe of each arm from
``starfinder.preprocessing``, and FOVs processed by ``FOV.run``. design/baselines/benchmark.md, "Test disposition
rules", names the tests that use them.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from starfinder.dataset import Dataset, PipelineConfig, RoundState
from starfinder.evaluation.preprocessing import NORMALIZED, SNAPSHOT
from starfinder.preprocessing import (Background3DConfig, HistogramMatchingConfig, MinMaxNormalizationConfig,
                                      PercentileNormalizationConfig, PreprocessingRecipe, PreprocessingStep,
                                      ReconstructionConfig, ScalarBackgroundConfig, TophatConfig)
from starfinder.synthetic import BackgroundConfig, NoiseConfig, ScalarDistribution, generate_amplicon_scene

#: The cap of the calibrated rerun's Z radius (design/specs/preprocessing-algorithms.md, amendment C4).
RZ_CAP = 3


def background_radius(config):
    """3D background radius r = ceil(3 sigma) + 1 per axis from a development preset's constant puncta widths."""
    sz, sl = float(config.axial_width.parameters[0]), float(config.lateral_width.parameters[0])
    return tuple(int(math.ceil(3 * s)) + 1 for s in (sz, sl, sl))


def calibrated_background_radius(config):
    """(radius, uncapped) from a calibrated preset's median widths (the exponentials of the lognormal locations),
    r_z capped at RZ_CAP; a capped Z footprint longer than the scene is refused."""
    sz, sl = (float(np.exp(d.parameters[0])) for d in (config.axial_width, config.lateral_width))
    uncapped = tuple(int(math.ceil(3 * s)) + 1 for s in (sz, sl, sl))
    radius = (min(uncapped[0], RZ_CAP), *uncapped[1:])
    if 2 * radius[0] + 1 > config.shape_zyx[0]:
        raise ValueError(f"the capped Z footprint {2 * radius[0] + 1} exceeds {config.shape_zyx[0]} planes")
    return radius, uncapped


def generate(book, config):
    """(observed scene, float64 background truth rounds, float64 signal truth rounds, reference-round truth)."""
    scene = generate_amplicon_scene(book, config=config)
    background = generate_amplicon_scene(book, config=replace(
        config, brightness=ScalarDistribution(parameters=(0.0,)), noise=NoiseConfig(), dtype="float64",
        accumulation="float64"))
    signal = generate_amplicon_scene(book, config=replace(
        config, background=BackgroundConfig(), noise=NoiseConfig(), dtype="float64", accumulation="float64"))
    ref = book.round_labels[0]
    truth = scene.round_truth[(scene.round_truth.round_label == ref) & scene.round_truth.emitting
                              & scene.round_truth.center_in_bounds]
    truth = truth.merge(scene.formed[["amplicon_id", "gene_id", "color_sequence"]], on="amplicon_id")
    return scene, background.rounds, signal.rounds, truth.reset_index(drop=True)


def arms(radius_zyx):
    """Arm name -> PreprocessingRecipe (None: no preprocessing), each step with its page defaults."""
    bg3d = Background3DConfig(radius_voxels_zyx=tuple(radius_zyx))
    minmax = MinMaxNormalizationConfig("uint8", (0, 255))

    def recipe(*steps, source=None):
        return PreprocessingRecipe(tuple(steps), extraction_source=source)

    def r2(background, source=None):
        return recipe(PreprocessingStep(background, save_as=SNAPSHOT),
                      PreprocessingStep(PercentileNormalizationConfig()), source=source)
    return {
        "none": None,
        "r1": recipe(PreprocessingStep(minmax), PreprocessingStep(HistogramMatchingConfig())),
        "r1_recon": recipe(PreprocessingStep(minmax), PreprocessingStep(HistogramMatchingConfig(), save_as=NORMALIZED),
                           PreprocessingStep(ReconstructionConfig())),
        "r2_scalar": r2(ScalarBackgroundConfig()),
        "r2_scalar_xsrc": r2(ScalarBackgroundConfig(), SNAPSHOT),
        "r2_3d": r2(bg3d),
        "r2_3d_xsrc": r2(bg3d, SNAPSHOT),
        "minmax": recipe(PreprocessingStep(minmax)),
        "hist": recipe(PreprocessingStep(HistogramMatchingConfig())),
        "recon": recipe(PreprocessingStep(ReconstructionConfig(), save_as=SNAPSHOT)),
        "tophat": recipe(PreprocessingStep(TophatConfig(), save_as=SNAPSHOT)),
        "scalar": recipe(PreprocessingStep(ScalarBackgroundConfig(), save_as=SNAPSHOT)),
        "bg3d": recipe(PreprocessingStep(bg3d, save_as=SNAPSHOT)),
        "pct": recipe(PreprocessingStep(PercentileNormalizationConfig())),
    }


def make_fov(scene, book, fov_id, workdir, images=None, metadata=None):
    """A resident FOV holding the scene's rounds, or the given images and metadata."""
    labels = list(book.round_labels)
    dataset = Dataset(Path(workdir), Path(workdir) / "out", "synthetic", "w781", "out",
                      rounds=RoundState(sequencing_rounds=labels, reference_round=labels[0]),
                      channel_order=tuple(book.channel_labels), codebook=book)
    fov = dataset.fov(fov_id)
    images = scene.rounds if images is None else images
    metadata = scene.round_metadata if metadata is None else metadata
    for name in labels:
        fov.images[name] = images[name].copy()
        fov.metadata[name] = metadata[name]
    return fov


def preprocess(fov, recipe):
    """The FOV after FOV.run with the recipe alone (unchanged for the ``none`` arm)."""
    if recipe is not None:
        fov.run(PipelineConfig(preprocessing=recipe))
    return fov


#: Arms whose images equal another arm's FOV.run output: (source arm, image taken). r2_*'s detection image is
#: r2_*_xsrc's (only the extraction source differs), and bg3d's is r2_3d_xsrc's bg_corrected snapshot.
SHARED = {"r2_scalar": ("r2_scalar_xsrc", "detection"), "r2_3d": ("r2_3d_xsrc", "detection"),
          "bg3d": ("r2_3d_xsrc", SNAPSHOT)}


def processed_arms(scene, book, fov_id, workdir, radius):
    """Yield (arm, recipe, processed FOV) for every arm; a shared arm is rebuilt from its source arm's images."""
    done = {}
    recipes = arms(radius)
    for arm in sorted(recipes, key=lambda a: a in SHARED):
        recipe = recipes[arm]
        if arm in SHARED:
            source, image = SHARED[arm]
            processed = done[source]
            images = {r: processed.images[r] if image == "detection" else processed.snapshots[r][image]
                      for r in book.round_labels}
            fov = make_fov(scene, book, fov_id, workdir, images, processed.metadata)
            fov.snapshots = {r: {SNAPSHOT: s[SNAPSHOT].copy()} for r, s in processed.snapshots.items()}
        else:
            fov = preprocess(make_fov(scene, book, fov_id, workdir), recipe)
        done[arm] = fov
        yield arm, recipe, fov
