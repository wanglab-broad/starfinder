"""Evaluate the §2.5 preprocessing recipes on §2.12 synthetic development presets.

Development evidence for Chapter II §2.5 task group 5 (W-233); not scientific
acceptance and no recommended defaults. Runs the bounded comparison matrix of
docs/preprocessing-algorithms.md (*Evaluation design for task groups 5 and 6*)
and writes tables, curves and a manifest to an output directory outside Git.

Scenes are development_scene_preset conditions with only existing
FormedSceneConfig fields changed (shape, count, seed, dtype, per-FOV gain and a
uniform intensity scale); the generator is not changed. Every arm is processed
by FOV.run (recipe, snapshots, registration), then detected with local maxima
in noise mode over the threshold grid, extracted, decoded and filtered with
their defaults, and matched to the truth with starfinder.evaluation.match_points.

``--design calibrated`` runs the calibrated rerun instead (W-239; *Evaluation
design amendment for the calibrated rerun (Accepted, W-243)* of the same page):
calibrated_scene_preset scenes with the added conditions, both threshold modes,
the precondition check, the revised low-benefit rule, histogram matching's harm
test and per-condition image statistics. The default design is W-233's, unchanged.

    uv run python ../../benchmarks/preprocessing_synthetic.py run --output <run dir>/evaluation
    uv run python ../../benchmarks/preprocessing_synthetic.py run --design calibrated --scope pilot --output <dir>
    uv run python ../../benchmarks/preprocessing_synthetic.py attach-time --output <dir> --time-log <file>
"""
import argparse
from dataclasses import asdict, replace
from functools import cached_property
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

from starfinder.barcode import Codebook, NeighborhoodSumConfig, ReadFilterConfig, WtaDecoderConfig
from starfinder.dataset import Dataset, PipelineConfig, RegistrationStep, RoundState
from starfinder.evaluation.matching import match_points
from starfinder.preprocessing import (Background3DConfig, HistogramMatchingConfig, MinMaxNormalizationConfig,
    PercentileNormalizationConfig, PreprocessingRecipe, RecipeStep, ReconstructionConfig, ScalarBackgroundConfig,
    TophatConfig, merge_histograms, step_spec, summarize_histograms, summary_stage, supplied_section,
    supplied_statistics, write_supplied_statistics)
from starfinder.preprocessing._diagnostics import channel_diagnostics
from starfinder.registration import TranslationConfig
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.synthetic import (CALIBRATED_CONDITIONS, BackgroundConfig, NoiseConfig, ScalarDistribution,
    TextureConfig, calibrated_scene_preset, development_scene_preset, generate_formed_scene)

SCHEMA = "starfinder.benchmark.preprocessing_synthetic/1"
THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)
DEFAULT_THRESHOLD = 5.0
DEV_SEEDS = (0, 1, 2)
EVAL_SEEDS = (100, 101, 102)
#: Seeds whose every arm also reruns FOV.run at threshold_value=5 to verify the threshold subset.
VERIFY_SEEDS = (DEV_SEEDS[0], EVAL_SEEDS[0])
DTYPES = ("uint8", "uint16")
#: Uniform intensity scale per dtype (signal, baseline, tissue weights and noise
#: strengths): the development presets are float scenes with peak 8, and uint16
#: is 16 times uint8, as in the benchmark presets (PRESET_VERSION).
INTENSITY_SCALE = {"uint8": 12.0, "uint16": 192.0}
MATCHING = dict(policy="greedy", threshold=2.0, units="voxel", boundary="inclusive")
EXTRACTION = NeighborhoodSumConfig()
DECODING = WtaDecoderConfig()
FILTERING = ReadFilterConfig()
REGISTRATION = (RegistrationStep(TranslationConfig()),)
#: Provisional low-benefit threshold (algorithm page): 2 percentage points.
LOW_BENEFIT_POINTS = 0.02
SNAPSHOT = "bg_corrected"
#: Snapshot of recipe 1's normalized image, the input of its reconstruction step.
NORMALIZED = "normalized"

#: Each condition names, per group of FormedSceneConfig fields, the development
#: condition whose value it takes; unnamed groups come from "clean".
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
PRIMARY = ("clean", "background_only", "round_effect_only", "combined")
#: The gradient condition's slopes (normalized ZYX): the preset's X slope of 2 plus a Z slope of 2,
#: set through the existing gradient_slopes_zyx field (the page's targeted condition needs a Z slope).
GRADIENT_SLOPES_ZYX = (2.0, 0.0, 2.0)

#: Multi-FOV sets: base appearance, per-FOV amplicon counts and per-FOV gain multipliers.
MULTI_FOV = {
    "mf_density": dict(base="combined", counts=(80, 20, 2), gains=(1.0, 1.0, 1.0)),
    "mf_gain_drift": dict(base="combined", counts=(80, 80, 80), gains=(1.0, 0.75, 0.5)),
    "mf_density_clean": dict(base="clean", counts=(80, 20, 2), gains=(1.0, 1.0, 1.0)),
}
FOV_ROLES = {"mf_density": ("dense", "sparse", "near_empty"), "mf_gain_drift": ("gain_1.00", "gain_0.75", "gain_0.50"),
             "mf_density_clean": ("dense", "sparse", "near_empty")}


def _minmax():
    return MinMaxNormalizationConfig("uint8", (0, 255))


def arms(radius_zyx, fit="fov", supplied=None):
    """Arm name -> PreprocessingRecipe (None: no preprocessing). Page defaults throughout."""
    s = "supplied" if fit == "supplied" else "fov"
    bg3d = Background3DConfig(radius_voxels_zyx=tuple(radius_zyx))

    def recipe(*steps, source=None):
        return PreprocessingRecipe(tuple(steps), extraction_source=source,
                                   supplied_statistics=supplied if fit == "supplied" else None)
    r2 = lambda bg, source=None: recipe(RecipeStep(bg, save_as=SNAPSHOT),  # noqa: E731
                                        RecipeStep(PercentileNormalizationConfig(fit=s)), source=source)
    return {
        "none": None,
        "r1": recipe(RecipeStep(_minmax()), RecipeStep(HistogramMatchingConfig(fit=s))),
        "r1_recon": recipe(RecipeStep(_minmax()), RecipeStep(HistogramMatchingConfig(), save_as=NORMALIZED),
                           RecipeStep(ReconstructionConfig())),
        "r2_scalar": r2(ScalarBackgroundConfig(fit=s)),
        "r2_scalar_xsrc": r2(ScalarBackgroundConfig(), SNAPSHOT),
        "r2_3d": r2(bg3d),
        "r2_3d_xsrc": r2(bg3d, SNAPSHOT),
        "minmax": recipe(RecipeStep(_minmax())),
        "hist": recipe(RecipeStep(HistogramMatchingConfig())),
        "recon": recipe(RecipeStep(ReconstructionConfig(), save_as=SNAPSHOT)),
        "tophat": recipe(RecipeStep(TophatConfig(), save_as=SNAPSHOT)),
        "scalar": recipe(RecipeStep(ScalarBackgroundConfig(), save_as=SNAPSHOT)),
        "bg3d": recipe(RecipeStep(bg3d, save_as=SNAPSHOT)),
        "pct": recipe(RecipeStep(PercentileNormalizationConfig())),
    }


PRIMARY_ARMS = ("none", "r1", "r1_recon", "r2_scalar", "r2_scalar_xsrc", "r2_3d", "r2_3d_xsrc",
                "minmax", "hist", "recon", "tophat", "scalar", "bg3d", "pct")
#: Multi-FOV arms: per-FOV (fov) and sample-level (supplied) fitting of each recipe.
MULTI_FOV_ARMS = {"none": ("none", "fov"), "r1": ("r1", "fov"), "r1_sample": ("r1", "supplied"),
                  "r2_scalar": ("r2_scalar", "fov"), "r2_scalar_sample": ("r2_scalar", "supplied"),
                  "r2_3d": ("r2_3d", "fov"), "r2_3d_sample": ("r2_3d", "supplied")}
BACKGROUND_ARMS = ("recon", "tophat", "scalar", "bg3d", "r2_scalar", "r2_scalar_xsrc", "r2_3d", "r2_3d_xsrc")


def recipe_record(recipe):
    if recipe is None:
        return None
    return {"steps": [{"step": step_spec(s.config).name, "config": asdict(s.config), "save_as": s.save_as}
                      for s in recipe.steps],
            "extraction_source": recipe.extraction_source,
            "supplied_statistics": None if recipe.supplied_statistics is None else "per-set file (sample-level)"}


# --- Scenes ---------------------------------------------------------------------

def _scaled(config, scale):
    bg, noise = config.background, config.noise
    mult = lambda v: None if v is None else (np.asarray(v, dtype=float) * scale).tolist()  # noqa: E731
    return replace(config,
        brightness=ScalarDistribution(parameters=(float(config.brightness.parameters[0]) * scale,)),
        background=replace(bg, baseline=mult(bg.baseline), tissue_weights=mult(bg.tissue_weights)),
        noise=replace(noise, alpha=noise.alpha * scale, sigma=noise.sigma * scale))


def scene_config(condition, *, dtype, seed, shape, count, parts=None, fov_id="FOV_001", gain=1.0, scene_key=None):
    """(codebook, config) for one FOV; parts maps field groups to development conditions."""
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


def background_radius(config):
    """3D background radius r = ceil(3 sigma) + 1 per axis from the preset's puncta widths."""
    sz, sl = float(config.axial_width.parameters[0]), float(config.lateral_width.parameters[0])
    return tuple(int(math.ceil(3 * s)) + 1 for s in (sz, sl, sl))


def generate(book, config):
    """Observed scene plus float64 background truth and signal truth rounds."""
    scene = generate_formed_scene(book, config=config)
    background = generate_formed_scene(book, config=replace(config, brightness=ScalarDistribution(parameters=(0.0,)),
        noise=NoiseConfig(), dtype="float64", accumulation="float64"))
    signal = generate_formed_scene(book, config=replace(config, background=BackgroundConfig(), noise=NoiseConfig(),
        dtype="float64", accumulation="float64"))
    ref = book.round_labels[0]
    truth = scene.round_truth[(scene.round_truth.round_label == ref) & scene.round_truth.emitting
                              & scene.round_truth.center_in_bounds]
    truth = truth.merge(scene.formed[["amplicon_id", "gene_id", "color_sequence"]], on="amplicon_id")
    return scene, background.rounds, signal.rounds, truth.reset_index(drop=True)


# --- Pipeline -------------------------------------------------------------------

class _LookupCodebook(Codebook):
    """Codebook whose sequence-to-gene lookup is built once instead of per decoded spot.

    The table is not modified during a run, so decoding results are identical;
    this only removes repeated dictionary construction in WTA decoding.
    """

    @cached_property
    def seq_to_gene(self):
        return dict(zip(self.table.color_sequence, self.table.gene_id))


def make_fov(scene, book, fov_id, workdir, images=None, metadata=None):
    """Resident FOV holding the scene's rounds, or the given images and metadata."""
    labels = list(book.round_labels)
    book = _LookupCodebook(book.table, book.round_labels, book.channel_labels, book.color_to_channel, book.encoding)
    dataset = Dataset(Path(workdir), Path(workdir) / "out", "synthetic", "w233", "out",
                      rounds=RoundState(sequencing_rounds=labels, reference_round=labels[0]),
                      channel_order=tuple(book.channel_labels), codebook=book)
    fov = dataset.fov(fov_id)
    images = scene.rounds if images is None else images
    metadata = scene.round_metadata if metadata is None else metadata
    for name in labels:
        fov.images[name] = images[name].copy()
        fov.metadata[name] = metadata[name]
    return fov


#: Without registration these arms reuse another arm's FOV.run output: (source arm, image taken).
#: The detection image of r2_*_xsrc equals that of r2_*, and its bg_corrected snapshot is the
#: output of the background step alone. Extraction in the reusing arm reads its detection image.
SHARED = {"r2_scalar": ("r2_scalar_xsrc", "detection"), "r2_3d": ("r2_3d_xsrc", "detection"),
          "bg3d": ("r2_3d_xsrc", SNAPSHOT)}


def processed_arms(scene, book, fov_id, workdir, radius, register):
    """Yield (arm, recipe, processed FOV); shared arms are rebuilt from resident images."""
    done = {}
    recipes = arms(radius)
    order = sorted(recipes, key=lambda a: a in SHARED)
    for arm in order:
        recipe = recipes[arm]
        if arm in SHARED and not register:
            source, image = SHARED[arm]
            processed = done[source]
            images = {r: processed.images[r] if image == "detection" else processed.snapshots[r][image]
                      for r in book.round_labels}
            fov = make_fov(scene, book, fov_id, workdir, images, processed.metadata)
            fov.snapshots = {r: {SNAPSHOT: s[SNAPSHOT].copy()} for r, s in processed.snapshots.items()}
        else:
            fov = preprocess(make_fov(scene, book, fov_id, workdir), recipe, register)
        done[arm] = fov
        yield arm, recipe, fov


def preprocess(fov, recipe, register):
    if recipe is not None or register:
        fov.run(PipelineConfig(preprocessing=recipe, registration=REGISTRATION if register else ()))
    return fov


def _pipeline(fov, threshold, mode="noise"):
    fov.run(PipelineConfig(detection=LocalMaximaConfig(threshold_mode=mode, threshold_value=threshold),
                           extraction=EXTRACTION, decoding=DECODING, filtering=FILTERING))
    spots = fov.spot_result.spots.reset_index(drop=True)
    reads = fov.filtering_result.table.set_index("spot_id").loc[spots.spot_id]
    return spots, reads.reset_index(), list(fov.spot_result.diagnostics["thresholds"])


def _noise_thresholds(image, value):
    """Per-channel noise-mode cutoffs, computed exactly as find_spots computes them."""
    result = []
    for c in range(image.shape[-1]):
        values = image[..., c].astype(np.float64)
        median = np.median(values)
        result.append(float(median + value * np.median(np.abs(values - median)) * 1.4826))
    return result


def _cutoffs(image, value, mode="noise"):
    """Per-channel cutoffs of a threshold mode, computed exactly as find_spots computes them."""
    if mode == "noise":
        return _noise_thresholds(image, value)
    if mode == "adaptive":
        return [float(image[..., c].max()) * value for c in range(image.shape[-1])]
    raise ValueError(f"unsupported threshold mode: {mode}")


def _key(spots, reads):
    return sorted(zip(spots.z, spots.y, spots.x, spots.channel, reads.gene_id.astype(str), reads.accepted,
                      reads.observed_color_sequence.astype(str)))


def sweep(fov, truth, verify=True, mode="noise", grid=THRESHOLDS, verify_value=DEFAULT_THRESHOLD):
    """Metrics at every threshold value of one mode's grid; one row per value.

    Detection, extraction, decoding and filtering run once through FOV.run at
    the lowest value. With min_distance_voxels=1, local maxima at a higher
    threshold_value are exactly those whose peak intensity exceeds that
    channel's higher cutoff (noise: median + value x 1.4826 x MAD; adaptive:
    value x channel maximum), and extraction, WTA decoding and read filtering
    act on each spot independently, so every other value is the corresponding
    subset. With verify, FOV.run is repeated at verify_value (noise 5, the
    default) and must give the same spots and reads as the subset (else
    RuntimeError).
    """
    ref = fov.rounds.reference_round
    metadata = fov.metadata[ref]
    points = truth[["z", "y", "x"]].to_numpy(float)
    all_spots, all_reads, lowest = _pipeline(fov, grid[0], mode)
    image = fov.images[ref]
    assert np.allclose(lowest, _cutoffs(image, grid[0], mode), rtol=0, atol=0)
    rows, verified = [], None
    for threshold in grid:
        cutoff = np.asarray(_cutoffs(image, threshold, mode))
        keep = all_spots.peak_intensity.to_numpy() > cutoff[all_spots.channel.to_numpy()]
        spots, reads = all_spots[keep].reset_index(drop=True), all_reads[keep].reset_index(drop=True)
        if verify and threshold == verify_value:
            direct_spots, direct_reads, direct_cutoff = _pipeline(fov, threshold, mode)
            if _key(direct_spots, direct_reads) != _key(spots, reads) or direct_cutoff != cutoff.tolist():
                raise RuntimeError(f"threshold subset differs from a direct run at threshold_value={threshold:g} "
                                   f"({mode} mode)")
            verified = len(direct_spots)
        result = match_points(points, spots[["z", "y", "x"]].to_numpy(float), reference_metadata=metadata,
                              observed_metadata=metadata, **MATCHING)
        matched = {j: i for i, j, _ in result.details["matched_pairs"]}
        correct = wrong = false = agree = 0
        for j, (gene, accepted, sequence) in enumerate(zip(reads.gene_id, reads.accepted,
                                                           reads.observed_color_sequence)):
            i = matched.get(j)
            if i is not None and str(sequence) == str(truth.color_sequence.iat[i]):
                agree += 1
            if not accepted:
                continue
            if i is None:
                false += 1
            elif pd.notna(gene) and str(gene) == str(truth.gene_id.iat[i]):
                correct += 1
            else:
                wrong += 1
        n_truth, n_spots, n = len(points), len(spots), len(matched)
        precision = n / n_spots if n_spots else None
        recall = n / n_truth if n_truth else None
        f1 = (2 * precision * recall / (precision + recall) if precision is not None and recall is not None
              and precision + recall > 0 else 0.0 if n_truth else None)
        rows.append(dict(threshold=threshold, n_truth=n_truth, n_detected=n_spots, n_matched=n,
            precision=precision, recall=recall, f1=f1, localization_error=result.values["mean_distance"],
            reads_accepted=correct + wrong + false, reads_correct=correct, reads_wrong_gene=wrong,
            reads_false_detection=false, correct_fraction=correct / n_truth if n_truth else None,
            color_call_agreement=agree / n if n else None, noise_cutoffs=json.dumps(cutoff.tolist()),
            verified_direct_run=threshold == verify_value and verified is not None))
    return rows


# --- Direct metrics ---------------------------------------------------------------

def _box(shape, center, radius):
    return tuple(slice(max(0, c - r), min(n, c + r + 1)) for c, r, n in zip(center, radius, shape))


#: Stage at which arms without a background step are measured (estimate 0).
_BG_STAGE = {"none": "input (no background step)",
             "pct": "before normalization, in input units (no background step); the stage of r2's bg_corrected"}


def _normalized_truth(raw, normalized, background):
    """Background truth mapped into recipe 1's normalized units, per channel.

    Min-max and histogram matching map each channel's voxel values through one
    monotone function fitted on the observed image. That function is read from
    the (raw value, normalized value) pairs and applied to the float truth by
    linear interpolation between observed raw values (constant beyond them).
    """
    mapped = np.empty(background.shape, dtype=np.float64)
    for c in range(raw.shape[-1]):
        values, first = np.unique(raw[..., c].ravel(), return_index=True)
        mapped[..., c] = np.interp(background[..., c], values.astype(np.float64),
                                   normalized[..., c].ravel()[first].astype(np.float64))
    return mapped


def direct_metrics(fov, raw, background, signal, truth, book, arm, recipe):
    """Background error, puncta contrast, true-spot intensity spread, clipped/saturated fractions."""
    ref = book.round_labels[0]
    detection = {name: fov.images[name] for name in book.round_labels}
    out = {}
    # Background estimation error on the reference round: estimate = input - output of the background step.
    truth_background, units = background[ref], "input intensity"
    if arm in ("none", "pct"):
        # No background step: estimate 0. For pct this is measured before normalization, in input
        # units, the stage of the r2 arms' bg_corrected snapshot, so the pct -> r2 ablations compare.
        estimate = np.zeros_like(background[ref])
    elif arm in BACKGROUND_ARMS:
        corrected = fov.snapshots.get(ref, {}).get(SNAPSHOT, detection[ref])
        estimate = raw[ref].astype(np.float64) - corrected.astype(np.float64)
    elif arm in ("r1", "r1_recon"):
        # Recipe 1 reconstructs after min-max and histogram matching, so its estimate is in the
        # normalized image's units; the truth is mapped into them (see _normalized_truth).
        normalized = detection[ref] if arm == "r1" else fov.snapshots[ref][NORMALIZED]
        truth_background, units = _normalized_truth(raw[ref], normalized, background[ref]), "recipe 1 normalized"
        estimate = (np.zeros_like(truth_background) if arm == "r1"
                    else normalized.astype(np.float64) - detection[ref].astype(np.float64))
    else:
        estimate = None
    if estimate is not None:
        error = estimate - truth_background
        out.update(bg_rmse=float(np.sqrt(np.mean(error ** 2))), bg_bias=float(np.mean(error)),
                   bg_truth_mean=float(np.mean(truth_background)), bg_units=units, bg_stage=_BG_STAGE.get(arm, (
                       "recipe 1 normalized image (reconstruction input)" if arm in ("r1", "r1_recon")
                       else "background step output (bg_corrected), before normalization")))
    # Puncta contrast (peak - local background) / noise on the reference detection image.
    image, sig = detection[ref].astype(np.float64), signal[ref]
    peak_amplitude = float(sig.max()) if sig.size else 0.0
    quiet = sig < 0.01 * peak_amplitude if peak_amplitude > 0 else np.ones(sig.shape, bool)
    contrasts, zero_noise = [], 0
    noise = [float(np.std(image[..., c][quiet[..., c]])) for c in range(image.shape[-1])]
    for row in truth.itertuples():
        center = tuple(int(round(v)) for v in (row.z, row.y, row.x))
        channel = book.color_to_channel[row.color_sequence[0]]
        volume = image[..., channel]
        peak = volume[_box(volume.shape, center, (1, 1, 1))].max()
        shell = volume[_box(volume.shape, center, (2, 6, 6))].copy()
        inner = np.zeros(volume.shape, bool)
        inner[_box(volume.shape, center, (1, 3, 3))] = True
        local = np.median(shell[~inner[_box(volume.shape, center, (2, 6, 6))]])
        if noise[channel] == 0:
            zero_noise += 1
            continue
        contrasts.append((peak - local) / noise[channel])
    out.update(contrast_median=float(np.median(contrasts)) if contrasts else None,
               contrast_zero_noise_fraction=zero_noise / len(truth) if len(truth) else None)
    # Spread of true-spot on-channel intensities across (round, channel) groups.
    groups = {}
    for row in truth.itertuples():
        center = tuple(int(round(v)) for v in (row.z, row.y, row.x))
        for r, name in enumerate(book.round_labels):
            channel = book.color_to_channel[row.color_sequence[r]]
            volume = detection[name][..., channel]
            value = float(volume[_box(volume.shape, center, EXTRACTION.neighborhood_radius_zyx)].astype(np.float64).sum())
            groups.setdefault((name, channel), []).append(value)
    medians = np.array([np.median(v) for v in groups.values()])
    out["intensity_cv"] = float(np.std(medians) / np.mean(medians)) if len(medians) and np.mean(medians) > 0 else None
    # Clipped (positive input -> 0) and saturated (dtype maximum) fractions over all rounds.
    clipped = saturated = total = 0
    for name in book.round_labels:
        output = detection[name]
        total += output.size
        clipped += int(np.count_nonzero((output == 0) & (raw[name] > 0)))
        if output.dtype.kind == "u":
            saturated += int(np.count_nonzero(output == np.iinfo(output.dtype).max))
    out.update(clipped_fraction=clipped / total, saturated_fraction=saturated / total,
               output_dtype=str(detection[ref].dtype))
    return out


def diagnostics_rows(fov, book):
    rows = []
    for name in book.round_labels:
        values = channel_diagnostics(fov.images[name])
        for c, channel in enumerate(book.channel_labels):
            rows.append(dict(round=name, channel=channel, **{k: v[c] for k, v in values.items()}))
    return rows


def output_digest(fov, book):
    digest = hashlib.sha256()
    for name in book.round_labels:
        digest.update(np.ascontiguousarray(fov.images[name]).tobytes())
    return digest.hexdigest()


# --- Sample-level statistics --------------------------------------------------------

def fit_supplied(recipe, scenes, book, workdir):
    """Summary passes over the FOVs for each fit="supplied" step, in order; writes the file."""
    path, sections, merged = recipe.supplied_statistics, {}, None
    ref = book.round_labels[0]
    for entry in recipe.steps:
        if getattr(entry.config, "fit", None) != "supplied":
            continue
        name = step_spec(entry.config).name
        stage, after = summary_stage(recipe, name)
        summaries = []
        for fov_id, (scene, *_rest) in scenes.items():
            fov = preprocess(make_fov(scene, book, fov_id, workdir), stage if stage.steps else None, False)
            summaries.append(summarize_histograms({r: fov.images[r] for r in book.round_labels},
                                                  channel_labels=book.channel_labels, fov_id=fov_id,
                                                  summarized_after=after))
        merged = merge_histograms(summaries)
        kwargs = {"reference_round": ref} if type(entry.config) is HistogramMatchingConfig else {}
        sections[name] = supplied_section(entry.config, merged, **kwargs)
        write_supplied_statistics(supplied_statistics(merged, sections), path)
    return sections


# --- Summaries, comparisons and flags ---------------------------------------------

def _defined(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _auprc(rows):
    """Average precision over the threshold grid, integrated in threshold order.

    Points are taken from the highest threshold to the lowest, so recall is
    non-decreasing (detections at a higher threshold are a subset); AUPRC is
    the sum of (R_i - R_(i-1)) * P_i with R_0 = 0. A threshold without
    detections has undefined precision and recall 0, so it adds no area. The
    result is None only when recall itself is undefined (no truth).
    """
    area, previous = 0.0, 0.0
    for row in sorted(rows, key=lambda r: -r["threshold"]):
        recall, precision = row["recall"], row["precision"]
        if not _defined(recall):
            return None
        if recall > previous:
            if not _defined(precision):
                raise ValueError("recall increased at a threshold with undefined precision")
            area += (recall - previous) * precision
            previous = recall
    return area


def summarize(curves, keys):
    """Per seed: max-F1 and AUPRC; per group: dev-selected threshold and eval-seed operating points."""
    per_seed = []
    for key, group in curves.groupby(keys + ["seed"], sort=False):
        rows = group.to_dict("records")
        best = max(rows, key=lambda r: (r["f1"] or 0.0))
        per_seed.append(dict(zip(keys + ["seed"], key), split="development" if key[-1] in DEV_SEEDS else "evaluation",
                             max_f1=best["f1"], max_f1_threshold=best["threshold"], auprc=_auprc(rows)))
    per_seed = pd.DataFrame(per_seed)
    selected = {}
    for key, group in curves[curves.seed.isin(DEV_SEEDS)].groupby(keys, sort=False):
        mean = group.groupby("threshold", sort=True).f1.mean()
        selected[key] = float(mean.idxmax())  # the first (smallest) threshold among ties
    points = []
    for key, group in curves[curves.seed.isin(EVAL_SEEDS)].groupby(keys, sort=False):
        for label, threshold in (("dev_max_f1", selected.get(key)), ("default", DEFAULT_THRESHOLD)):
            if threshold is None:
                continue
            rows = group[group.threshold == threshold]
            points.append(dict(zip(keys, key), operating_point=label, threshold=threshold, seeds=len(rows),
                **{f"{m}_{s}": getattr(rows[m].astype(float), s)() for m in
                   ("precision", "recall", "f1", "localization_error", "correct_fraction", "reads_correct",
                    "reads_wrong_gene", "reads_false_detection", "color_call_agreement") for s in ("mean", "min", "max")}))
    return per_seed, pd.DataFrame(points), selected


def endpoints(curves, per_seed, selected, keys):
    """Per evaluation seed: max-F1, AUPRC, and reads at the dev-selected threshold and at threshold_value=5."""
    rows = []
    ev = per_seed[per_seed.split == "evaluation"]
    for record in ev.to_dict("records"):
        key = tuple(record[k] for k in keys)
        match = curves
        for k in keys:
            match = match[match[k] == record[k]]
        match = match[(match.seed == record["seed"])]
        at = match[match.threshold == selected[key]].iloc[0]
        at5 = match[match.threshold == DEFAULT_THRESHOLD].iloc[0]
        rows.append(dict(record, correct_fraction=at.correct_fraction, color_call_agreement=at.color_call_agreement,
                         reads_wrong_gene=at.reads_wrong_gene, reads_false_detection=at.reads_false_detection,
                         correct_fraction_t5=at5.correct_fraction, f1_t5=at5.f1,
                         reads_wrong_gene_t5=at5.reads_wrong_gene, reads_false_detection_t5=at5.reads_false_detection))
    return pd.DataFrame(rows)


# Before/after comparisons: (method or mode, comparison, before arm, after arm, targeted conditions).
_BG = ("regions", "texture", "gradient")
_INT = ("gain", "trend", "texture")
COMPARISONS = [
    ("scalar_background", "isolated", "none", "scalar", ("baseline",)),
    ("scalar_background", "ablation", "pct", "r2_scalar", ("baseline",)),
    ("background_3d", "isolated", "none", "bg3d", _BG),
    ("background_3d", "ablation", "pct", "r2_3d", _BG),
    ("percentile_normalization", "isolated", "none", "pct", _INT),
    ("percentile_normalization", "ablation_scalar", "scalar", "r2_scalar", _INT),
    ("percentile_normalization", "ablation_3d", "bg3d", "r2_3d", _INT),
    ("extraction_source", "ablation_scalar", "r2_scalar", "r2_scalar_xsrc", ("gain_baseline", "gain_texture")),
    ("extraction_source", "ablation_3d", "r2_3d", "r2_3d_xsrc", ("gain_baseline", "gain_texture")),
    ("min_max_normalization", "isolated", "none", "minmax", ("gain", "trend")),
    ("min_max_normalization", "ablation", "hist", "r1", ("gain", "trend")),
    ("histogram_matching", "isolated", "none", "hist", ("gain", "trend")),
    ("histogram_matching", "ablation", "minmax", "r1", ("gain", "trend")),
    ("reconstruction", "isolated", "none", "recon", _BG),
    ("reconstruction", "ablation", "r1", "r1_recon", _BG),
    ("white_tophat", "isolated", "none", "tophat", _BG),
]
SAMPLE_COMPARISONS = [("sample_level_fitting", f"ablation_{recipe}", recipe, recipe + "_sample",
                       ("mf_density", "mf_gain_drift")) for recipe in ("r1", "r2_scalar", "r2_3d")]
DIRECT = ("bg_rmse", "bg_bias", "contrast_median", "intensity_cv", "clipped_fraction", "saturated_fraction")
DIAGNOSTIC = ("zero_fraction", "median", "mad", "noise_threshold", "mad_zero")
ENDPOINTS = ("max_f1", "auprc", "correct_fraction", "color_call_agreement", "reads_wrong_gene", "reads_false_detection",
             "f1_t5", "correct_fraction_t5", "reads_wrong_gene_t5", "reads_false_detection_t5")


def _stats(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if not len(values):
        return None, None, None
    return float(values.mean()), float(values.min()), float(values.max())


def comparisons_table(ends, direct, diagnostics, specs, harm, combined, extra_roles=None, endpoints=ENDPOINTS):
    """One row per comparison, condition role and dtype, with before/after/delta mean and range.

    Means and ranges are across held-out (evaluation) seeds. Direct metrics and
    MAD diagnostics are first averaged within each seed (over channels, and
    over FOVs for multi-FOV sets); mad_zero becomes the fraction of channels
    with MAD 0. extra_roles(method, comparison) may add (condition, role) pairs.
    """
    rows = []
    for method, mode, before, after, targeted in specs:
        roles = [(c, "targeted") for c in targeted] + [(harm, "harm_check"), (combined, "combined")]
        roles += list(extra_roles(method, mode)) if extra_roles else []
        for condition, role in roles:
            for dtype in DTYPES:
                row = dict(method=method, comparison=mode, before=before, after=after, condition=condition,
                           role=role, dtype=dtype, seeds="evaluation " + ",".join(map(str, EVAL_SEEDS)))
                sel = lambda t, arm: t[(t.condition == condition) & (t.dtype == dtype) & (t.arm == arm)  # noqa: E731
                                       & t.seed.isin(EVAL_SEEDS)].sort_values("seed")
                b, a = sel(ends, before), sel(ends, after)
                if not len(b) or not len(a):
                    continue
                for metric in endpoints:
                    for label, values in (("before", b[metric].to_numpy(float)), ("after", a[metric].to_numpy(float)),
                                          ("delta", a[metric].to_numpy(float) - b[metric].to_numpy(float))):
                        mean, low, high = _stats(values)
                        row.update({f"{metric}_{label}_mean": mean, f"{metric}_{label}_min": low,
                                    f"{metric}_{label}_max": high})
                for table, metrics in ((direct, DIRECT), (diagnostics, DIAGNOSTIC)):
                    if table is None:
                        continue
                    for label, arm in (("before", before), ("after", after)):
                        part = sel(table, arm)
                        for metric in metrics:
                            if metric in part:
                                # Aggregate channels (and FOVs) within each seed first, so the range
                                # is across held-out seeds only.
                                mean, low, high = _stats(part.assign(value=pd.to_numeric(part[metric].astype(float)))
                                                         .groupby("seed").value.mean())
                                row.update({f"{metric}_{label}_mean": mean, f"{metric}_{label}_min": low,
                                            f"{metric}_{label}_max": high})
                rows.append(row)
    return pd.DataFrame(rows)


def _num(value):
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _span(record, metric):
    """Seed range: max of the before and after ranges across evaluation seeds."""
    return float(np.nanmax([_num(record.get(f"{metric}_{s}_max")) - _num(record.get(f"{metric}_{s}_min"))
                            for s in ("before", "after")] + [float("-inf")]))


def flags_table(comparisons):
    """Provisional low-benefit flag per method, comparison, dtype and targeted condition, plus an aggregate.

    Improvement is the mean paired delta over evaluation seeds of the primary
    endpoint (max-F1 and the correct-decode fraction at the dev-selected
    threshold). The seed range is max(before range, after range) across
    evaluation seeds. Benefit on a targeted condition requires one endpoint to
    improve by at least max(seed range, 2 points); harm is a clean-condition
    worsening of either endpoint by more than its seed range.
    """
    rows = []
    if not len(comparisons):
        return pd.DataFrame(rows)
    for (method, mode, dtype), group in comparisons.groupby(["method", "comparison", "dtype"], sort=False):
        clean = group[group.role == "harm_check"]
        harm, harm_detail = False, []
        for metric in ("max_f1", "correct_fraction"):
            if len(clean) and clean[f"{metric}_delta_mean"].notna().all():
                c = clean.iloc[0]
                span = _span(c, metric)
                worse = c[f"{metric}_delta_mean"] < -span
                harm |= bool(worse)
                harm_detail.append(f"{metric} delta {c[f'{metric}_delta_mean']:+.4f} vs range {span:.4f}")
        benefits = []
        for record in group[group.role == "targeted"].to_dict("records"):
            gains = {}
            for metric in ("max_f1", "correct_fraction"):
                delta, span = _num(record.get(f"{metric}_delta_mean")), _span(record, metric)
                needed = max(0.0 if np.isnan(span) else span, LOW_BENEFIT_POINTS)
                gains[metric] = (delta, span, bool(not np.isnan(delta) and delta >= needed))
            benefit = any(g[2] for g in gains.values())
            benefits.append(benefit)
            rows.append(dict(method=method, comparison=mode, dtype=dtype, condition=record["condition"],
                level="targeted_condition", max_f1_delta=gains["max_f1"][0], max_f1_seed_range=gains["max_f1"][1],
                correct_fraction_delta=gains["correct_fraction"][0],
                correct_fraction_seed_range=gains["correct_fraction"][1], clean_harm=harm,
                low_benefit=(not benefit) or harm, threshold_points=LOW_BENEFIT_POINTS, provisional=True,
                reason=("no endpoint improved by max(seed range, 2 points)" if not benefit else "")
                       + ("; clean harm beyond seed range" if harm else "")))
        rows.append(dict(method=method, comparison=mode, dtype=dtype, condition="all targeted", level="method",
            clean_harm=harm, low_benefit=(not any(benefits)) or harm, threshold_points=LOW_BENEFIT_POINTS,
            provisional=True, reason=("no targeted condition shows benefit" if not any(benefits) else
                                      "benefit on at least one targeted condition")
                                     + ("; clean harm: " if harm else "; clean: ") + ", ".join(harm_detail)))
    return pd.DataFrame(rows)


# --- Runner ---------------------------------------------------------------------------

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = frame.to_csv(index=False).encode()
    path.write_bytes(data)


def _revision():
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
        diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD", "--binary"], capture_output=True).stdout
        untracked = subprocess.run(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
                                   capture_output=True, text=True).stdout.split()
    except OSError:
        return {"revision": None, "dirty": None}
    digest = hashlib.sha256(diff)
    for name in sorted(untracked):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return {"revision": head, "dirty": dirty, "uncommitted_diff_sha256": digest.hexdigest() if dirty else None,
            "uncommitted_diff": "sha256 of `git diff HEAD --binary` followed by each untracked file's name and bytes",
            "untracked_files": sorted(untracked)}


def plan(scope):
    """Conditions, dtypes, seeds and multi-FOV sets for a scope (full, pilot or smoke)."""
    if scope == "smoke":
        return dict(conditions=("clean",), dtypes=("uint8",), seeds=(0, 100), multi_fov=(), mf_dtypes=())
    if scope == "mini":
        return dict(conditions=("clean", "combined", "baseline", "gain"), dtypes=("uint8",), seeds=(0, 100),
                    multi_fov=tuple(MULTI_FOV),
                    mf_dtypes=("uint8",))
    if scope == "pilot":
        return dict(conditions=("clean", "combined", "combined_geometry"), dtypes=("uint8",), seeds=(0,),
                    multi_fov=("mf_density",),
                    mf_dtypes=("uint8",))
    return dict(conditions=tuple(CONDITIONS), dtypes=DTYPES, seeds=DEV_SEEDS + EVAL_SEEDS,
                multi_fov=tuple(MULTI_FOV), mf_dtypes=DTYPES)


def run(output, *, scope="full", shape=(32, 64, 64), count=80, multi_fov_dtypes=None, reductions=(), projection=(),
        log=print):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    selection = plan(scope)
    if multi_fov_dtypes is not None:
        selection["mf_dtypes"] = tuple(multi_fov_dtypes)
    curves, direct, diagnostics, scenes_record, runs_record, timing = [], [], [], [], [], {}
    mf_curves, mf_scenes = [], []
    with tempfile.TemporaryDirectory(prefix="w233-") as workdir:
        for condition in selection["conditions"]:
            t0 = time.perf_counter()
            register = CONDITIONS[condition].get("geometry") is not None
            for dtype in selection["dtypes"]:
                for seed in selection["seeds"]:
                    book, config = scene_config(condition, dtype=dtype, seed=seed, shape=shape, count=count)
                    scene, background, signal, truth = generate(book, config)
                    radius = background_radius(config)
                    base = dict(condition=condition, dtype=dtype, seed=seed,
                                split="development" if seed in DEV_SEEDS else "evaluation")
                    scenes_record.append(dict(base, preset_parts={g: CONDITIONS[condition].get(g, "clean") for g in
                        ("readout", "background", "noise", "geometry")}, fov_id=config.FOV_id,
                        requested_config_sha256=hashlib.sha256(json.dumps(scene.provenance["requested_config"],
                            sort_keys=True).encode()).hexdigest(), image_sha256=scene.provenance["image_sha256"],
                        clipping_counts=scene.provenance["clipping_counts"], n_truth=len(truth),
                        background_radius_voxels_zyx=list(radius)))
                    for arm, recipe, fov in processed_arms(scene, book, config.FOV_id, workdir, radius, register):
                        direct.append(dict(base, arm=arm, **direct_metrics(fov, scene.rounds, background, signal,
                                                                           truth, book, arm, recipe)))
                        diagnostics.extend(dict(base, arm=arm, **r) for r in diagnostics_rows(fov, book))
                        runs_record.append(dict(base, arm=arm, output_sha256=output_digest(fov, book)))
                        curves.extend(dict(base, arm=arm, **r) for r in sweep(fov, truth, seed in VERIFY_SEEDS))
            timing[condition] = time.perf_counter() - t0
            log(f"{condition}: {timing[condition]:.1f} s")
        for name in selection["multi_fov"]:
            t0 = time.perf_counter()
            spec = MULTI_FOV[name]
            for dtype in selection["mf_dtypes"]:
                for seed in selection["seeds"]:
                    fovs = {}
                    for k, (role, n, gain) in enumerate(zip(FOV_ROLES[name], spec["counts"], spec["gains"])):
                        fov_id = f"Position{k + 1:03d}"
                        book, config = scene_config(spec["base"], dtype=dtype, seed=seed, shape=shape,
                            count=min(n, count), fov_id=fov_id, gain=gain,
                            scene_key=f"controlled-development-v1/w233/{name}/{fov_id}")
                        fovs[fov_id] = (*generate(book, config), role, config)
                        mf_scenes.append(dict(multi_fov=name, dtype=dtype, seed=seed, fov_id=fov_id, role=role,
                            count=min(n, count), gain_multiplier=gain, base_condition=spec["base"],
                            image_sha256=fovs[fov_id][0].provenance["image_sha256"], n_truth=len(fovs[fov_id][3])))
                    radius = background_radius(config)
                    for arm, (recipe_name, fit) in MULTI_FOV_ARMS.items():
                        path = Path(workdir) / f"supplied-{name}-{dtype}-{seed}-{arm}.json"
                        recipe = arms(radius, fit, path)[recipe_name]
                        supplied = fit_supplied(recipe, fovs, book, workdir) if fit == "supplied" else None
                        for fov_id, (scene, background, signal, truth, role, _config) in fovs.items():
                            fov = preprocess(make_fov(scene, book, fov_id, workdir), recipe, False)
                            base = dict(multi_fov=name, dtype=dtype, seed=seed, fov_id=fov_id, role=role, arm=arm,
                                        fit=fit, split="development" if seed in DEV_SEEDS else "evaluation")
                            runs_record.append(dict(base, output_sha256=output_digest(fov, book),
                                supplied_sections=None if supplied is None else sorted(supplied)))
                            diagnostics.extend(dict(base, condition=name, **r) for r in diagnostics_rows(fov, book))
                            mf_curves.extend(dict(base, **r) for r in sweep(fov, truth, seed in VERIFY_SEEDS))
            timing[name] = time.perf_counter() - t0
            log(f"{name}: {timing[name]:.1f} s")
    compute_seconds = time.perf_counter() - started
    curves, direct, diagnostics = pd.DataFrame(curves), pd.DataFrame(direct), pd.DataFrame(diagnostics)
    files = {}

    def save(frame, relative):
        _write_csv(frame, output / relative)
        files[relative] = output / relative

    save(curves, "curves/pr_curves.csv")
    save(direct, "tables/direct_metrics.csv")
    save(diagnostics, "tables/diagnostics.csv")
    keys = ["condition", "dtype", "arm"]
    per_seed, points, selected = summarize(curves, keys)
    save(per_seed, "tables/recipe_per_seed.csv")
    save(points, "tables/operating_points.csv")
    save(curves[curves.threshold.isin([DEFAULT_THRESHOLD] + sorted(set(selected.values())))], "tables/detection_reads.csv")
    comparisons = flags = None
    if set(EVAL_SEEDS) & set(selection["seeds"]) and set(DEV_SEEDS) & set(selection["seeds"]):
        ends = endpoints(curves, per_seed, selected, keys)
        diag_ref = diagnostics[diagnostics["round"] == diagnostics["round"].iloc[0]] if len(diagnostics) else diagnostics
        specs = [c for c in COMPARISONS if all(x in selection["conditions"] for x in ("clean", "combined"))]
        comparisons = comparisons_table(ends, direct, diag_ref[diag_ref.condition.isin(selection["conditions"])]
                                        if "condition" in diag_ref else None, specs, "clean", "combined")
        if selection["multi_fov"]:
            mf = pd.DataFrame(mf_curves)
            save(mf, "curves/multi_fov_curves.csv")
            pooled = (mf.groupby(["multi_fov", "dtype", "arm", "seed", "threshold"], sort=False)
                      [["n_truth", "n_detected", "n_matched", "reads_accepted", "reads_correct", "reads_wrong_gene",
                        "reads_false_detection"]].sum().reset_index())
            pooled["precision"] = pooled.n_matched / pooled.n_detected.where(pooled.n_detected > 0)
            pooled["recall"] = pooled.n_matched / pooled.n_truth.where(pooled.n_truth > 0)
            pooled["f1"] = (2 * pooled.precision * pooled.recall / (pooled.precision + pooled.recall)).fillna(0.0)
            pooled["correct_fraction"] = pooled.reads_correct / pooled.n_truth.where(pooled.n_truth > 0)
            pooled["color_call_agreement"] = np.nan
            pooled = pooled.rename(columns={"multi_fov": "condition"})
            mf_per_seed, mf_points, mf_selected = summarize(pooled.assign(localization_error=np.nan), keys)
            save(mf_per_seed, "tables/multi_fov_pooled_per_seed.csv")
            per_fov = multi_fov_tables(mf, mf_selected)
            save(per_fov, "tables/multi_fov_per_fov.csv")
            spread = multi_fov_spread(per_fov)
            save(spread, "tables/multi_fov_spread.csv")
            mf_ends = endpoints(pooled, mf_per_seed, mf_selected, keys)
            spread_direct = per_fov[per_fov.operating_point == "dev_max_f1"].rename(columns={"multi_fov": "condition"})
            spread_direct = (spread_direct.groupby(["condition", "dtype", "arm", "seed"]).correct_fraction
                             .agg(lambda v: v.max() - v.min()).rename("per_fov_correct_fraction_range").reset_index())
            mf_diag = diag_ref[diag_ref.condition.isin(selection["multi_fov"])] if "condition" in diag_ref else None
            mf_cmp = comparisons_table(mf_ends, None, mf_diag, SAMPLE_COMPARISONS, "mf_density_clean", "mf_density")
            _add_per_fov_spread(mf_cmp, spread_direct)
            comparisons = pd.concat([comparisons, mf_cmp], ignore_index=True)
        if comparisons is not None and len(comparisons):
            save(comparisons, "tables/comparisons.csv")
            flags = flags_table(comparisons)
            save(flags, "tables/low_benefit_flags.csv")
    figures = plot_curves(curves, output / "curves") if scope != "smoke" else []
    for path in figures:
        files[str(path.relative_to(output))] = path
    mad_zero = sorted({(r["condition"], r["dtype"], r["arm"]) for r in diagnostics.to_dict("records")
                       if r["mad_zero"] and r.get("round") == diagnostics["round"].iloc[0]})
    manifest = dict(schema=SCHEMA, issue="W-233", scope=scope,
        qualification="development evidence on uncalibrated synthetic presets; not scientific acceptance; "
                      "no recommended defaults; the low-benefit threshold is provisional",
        software=dict(**_revision(), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      command=sys.argv),
        parameters=dict(thresholds=list(THRESHOLDS), default_threshold=DEFAULT_THRESHOLD,
            development_seeds=list(DEV_SEEDS), evaluation_seeds=list(EVAL_SEEDS), seeds_run=list(selection["seeds"]),
            dtypes=list(selection["dtypes"]), multi_fov_dtypes=list(selection["mf_dtypes"]), shape_zyx=list(shape),
            amplicons_per_fov=count, intensity_scale=INTENSITY_SCALE,
            detection=dict(asdict(LocalMaximaConfig()), threshold_value="swept"),
            extraction=asdict(EXTRACTION), decoding=asdict(DECODING), filtering=asdict(FILTERING),
            registration_coupled=[dict(method=s.config.method, config=asdict(s.config)) for s in REGISTRATION],
            matching=dict(function="starfinder.evaluation.match_points", reference="truth", observed="detections",
                truth="reference-round amplicons that emit and whose centre is in bounds", **MATCHING),
            reads="accepted reads (default filter) split into correct (matched, same gene), wrong-gene (matched, "
                  "other or missing gene) and false-detection (unmatched detection)",
            auprc="average precision over the threshold grid: sum of recall increments times precision",
            operating_points="max mean F1 over development seeds (first threshold among ties); reported on evaluation "
                             "seeds, and threshold_value=5.0",
            background_error=dict(truth="float64 background truth (same config, brightness 0, no noise), reference round",
                estimate="input minus the background step's output; 0 for arms without a background step",
                stages={**_BG_STAGE, "background arms and r2": "background step output (bg_corrected), before "
                        "normalization, input units", "r1, r1_recon": "recipe 1 normalized image (reconstruction "
                        "input), truth mapped through the fitted value map"}),
            low_benefit=dict(rule="targeted: no endpoint (max-F1, correct-decode fraction) improves by max(seed range, "
                                  "2 points); or clean: an endpoint worsens by more than its seed range",
                             threshold_points=LOW_BENEFIT_POINTS, provisional=True,
                             seed_range="max of before and after ranges across evaluation seeds")),
        conditions={name: dict(parts={g: parts.get(g, "clean") for g in ("readout", "background", "noise", "geometry")},
                               preset="starfinder.synthetic.development_scene_preset (controlled-development-v1, "
                                      "size small, scene_key controlled-development-v1)",
                               primary=name in PRIMARY, run=name in selection["conditions"])
                    for name, parts in CONDITIONS.items()},
        multi_fov={name: dict(spec, roles=FOV_ROLES[name], run=name in selection["multi_fov"])
                   for name, spec in MULTI_FOV.items()},
        scene_changes="shape_zyx, count (coordinates, amplicon_ids and gene_ids cleared), elongation uniform(1, 1.5) "
                      "and angle uniform(0, pi) replacing the two ID-keyed values, seed, dtype, FOV_id/scene_key per "
                      "multi-FOV FOV, readout gains per FOV (gain drift), and a uniform intensity scale of brightness, "
                      "baseline, tissue weights and noise strengths",
        scene_field_changes=dict(
            shape_zyx=list(shape), count=count, cleared=["coordinates", "amplicon_ids", "gene_ids"],
            elongation="uniform(1, 1.5)", angle="uniform(0, pi)", seed="per scene", dtype="per scene",
            multi_fov=["FOV_id", "scene_key", "readout gains (gain drift)"],
            gradient_slopes_zyx=dict(conditions=["gradient"], value=list(GRADIENT_SLOPES_ZYX),
                                     preset_value=[0, 0, 2], reason="the page's gradient target needs a Z slope"),
            intensity_scale=dict(factors=INTENSITY_SCALE, fields=["brightness", "background.baseline",
                                 "background.tissue_weights", "noise.alpha", "noise.sigma"],
                                 decision="accepted as within the batch's scene rule by the owner's delegate "
                                          "(recovery note R-20260928T011104Z-7543430d)")),
        recipes={arm: recipe_record(recipe) for arm, recipe in arms((4, 5, 5)).items()},
        multi_fov_recipes={arm: dict(recipe=r, fit=f, record=recipe_record(arms((4, 5, 5), f, Path("supplied.json"))[r]))
                           for arm, (r, f) in MULTI_FOV_ARMS.items()},
        comparisons=[dict(method=m, comparison=c, before=b, after=a, targeted=list(t))
                     for m, c, b, a, t in COMPARISONS + SAMPLE_COMPARISONS],
        scenes=scenes_record, multi_fov_scenes=mf_scenes, runs=runs_record,
        mad_zero_reference_round=[dict(condition=c, dtype=d, arm=a) for c, d, a in mad_zero],
        fixture_gaps=[
            "Bright outliers: the development texture blobs (count 2, peak 5 times the tissue weight) are dimmer than "
            "puncta (peak 8), so they are not bright outliers; the texture condition is reported with the "
            "percentile-normalization comparisons but no bright-outlier fixture exists. The generator is not extended.",
            "White top-hat: no agreed recipe contains it, so it has only the isolated comparison."],
        threshold_subset_verification=dict(seeds=list(VERIFY_SEEDS), threshold_value=DEFAULT_THRESHOLD,
            runs_verified=int(curves.verified_direct_run.sum()) + sum(bool(r["verified_direct_run"]) for r in mf_curves)),
        reductions=list(reductions), projection=list(projection), timing_seconds=timing, compute_seconds=compute_seconds)
    manifest["files"] = [dict(path=k, bytes=v.stat().st_size, sha256=_digest(v)) for k, v in sorted(files.items())]
    manifest["artifact_bytes"] = sum(f["bytes"] for f in manifest["files"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1, default=_json_default) + "\n")
    return manifest


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def multi_fov_tables(mf, selected, fixed=(("default", DEFAULT_THRESHOLD),)):
    """Per-FOV decoding accuracy and false-positive rate at the pooled dev-selected threshold and at 5."""
    rows = []
    for key, group in mf[mf.seed.isin(EVAL_SEEDS)].groupby(["multi_fov", "dtype", "arm", "seed", "fov_id", "role"],
                                                           sort=False):
        name, dtype, arm, seed, fov_id, role = key
        for label, threshold in (("dev_max_f1", selected.get((name, dtype, arm))), *fixed):
            if threshold is None:
                continue
            r = group[group.threshold == threshold].iloc[0]
            accepted = r.reads_accepted
            rows.append(dict(multi_fov=name, dtype=dtype, arm=arm, seed=seed, fov_id=fov_id, role=role,
                operating_point=label, threshold=threshold, n_truth=r.n_truth, reads_accepted=accepted,
                decoding_accuracy=r.reads_correct / (r.reads_correct + r.reads_wrong_gene)
                if r.reads_correct + r.reads_wrong_gene else None,
                correct_fraction=r.correct_fraction,
                false_positive_rate=r.reads_false_detection / accepted if accepted else None,
                precision=r.precision, recall=r.recall, f1=r.f1))
    return pd.DataFrame(rows)


def _add_per_fov_spread(mf_cmp, spread_direct):
    """Add the per-FOV correct-fraction range (sample-level fitting's direct metric) to multi-FOV comparison rows."""
    for record_index, record in mf_cmp.iterrows():
        for label, arm in (("before", record["before"]), ("after", record["after"])):
            part = spread_direct[(spread_direct["condition"] == record["condition"])
                                 & (spread_direct["dtype"] == record["dtype"])
                                 & (spread_direct["arm"] == arm) & spread_direct["seed"].isin(EVAL_SEEDS)]
            mean, low, high = _stats(part.per_fov_correct_fraction_range)
            mf_cmp.loc[record_index, f"per_fov_correct_fraction_range_{label}_mean"] = mean
            mf_cmp.loc[record_index, f"per_fov_correct_fraction_range_{label}_min"] = low
            mf_cmp.loc[record_index, f"per_fov_correct_fraction_range_{label}_max"] = high


def multi_fov_spread(per_fov, keys=("multi_fov", "dtype", "arm", "operating_point")):
    """Spread across FOVs (range and standard deviation), then mean and range across evaluation seeds."""
    rows = []
    for key, group in per_fov.groupby(list(keys), sort=False):
        record = dict(zip(keys, key))
        for metric in ("decoding_accuracy", "correct_fraction", "false_positive_rate", "f1"):
            per_seed = group.groupby("seed")[metric].agg(lambda v: pd.to_numeric(v, errors="coerce").max()
                                                         - pd.to_numeric(v, errors="coerce").min())
            sd = group.groupby("seed")[metric].agg(lambda v: pd.to_numeric(v, errors="coerce").std(ddof=0))
            for label, values in (("fov_range", per_seed), ("fov_sd", sd)):
                mean, low, high = _stats(values)
                record.update({f"{metric}_{label}_mean": mean, f"{metric}_{label}_min": low,
                               f"{metric}_{label}_max": high})
            for role, part in group.groupby("role", sort=False):
                record[f"{metric}_{role}_mean"] = _stats(part[metric])[0]
        rows.append(record)
    return pd.DataFrame(rows)


def plot_curves(curves, directory):
    """One precision-recall figure per condition and dtype (mean over evaluation seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for (condition, dtype), group in curves[curves.seed.isin(EVAL_SEEDS)].groupby(["condition", "dtype"], sort=False):
        figure, axis = plt.subplots(figsize=(6, 5))
        for arm, part in group.groupby("arm", sort=False):
            mean = part.groupby("threshold")[["precision", "recall"]].mean()
            axis.plot(mean.recall, mean.precision, marker="o", markersize=3, label=arm)
        axis.set(xlabel="recall", ylabel="precision", xlim=(0, 1.02), ylim=(0, 1.02),
                 title=f"{condition}, {dtype}: mean over evaluation seeds")
        axis.legend(fontsize=6, ncol=2)
        path = directory / f"pr_{condition}_{dtype}.png"
        buffer = io.BytesIO()
        figure.savefig(buffer, dpi=90, format="png")
        path.write_bytes(buffer.getvalue())  # one full-file write (network mounts)
        plt.close(figure)
        paths.append(path)
    return paths


def attach_time(output, time_log):
    """Add a /usr/bin/time -v record (wall time, maximum RSS) and artifact bytes to the manifest."""
    output = Path(output)
    text = Path(time_log).read_text()
    record = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, value = line.strip().rpartition(": ")
            record[key.strip()] = value.strip()
    wall = record.get("Elapsed (wall clock) time (h:mm:ss or m:ss)")
    seconds = None
    if wall:
        parts = [float(p) for p in wall.split(":")]
        seconds = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["time_v"] = dict(wall_seconds=seconds, max_rss_kib=int(record["Maximum resident set size (kbytes)"]),
                              exit_status=int(record.get("Exit status", -1)), raw=text.splitlines())
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest["artifact_bytes_total"] = total
    if isinstance(manifest.get("projection"), dict) and seconds is not None:
        # Calibrated design: add the measured start-up and shutdown time outside the timed compute.
        manifest["projection"] = project_wall(manifest["projection"], seconds - manifest["compute_seconds"])
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest["time_v"]


# --- Calibrated rerun (W-239): the accepted amendment of W-243 --------------------------------

CALIBRATED_SCHEMA = "starfinder.benchmark.preprocessing_synthetic.calibrated/1"
CALIBRATED_VERSION = "calibrated-development-v1"
#: Item 2: grid and fixed operating points per threshold mode. The threshold subset is verified by a
#: direct run at each mode's first fixed point.
THRESHOLD_MODES = {"noise": dict(grid=THRESHOLDS, fixed=(5.0,)),
                   "adaptive": dict(grid=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4), fixed=(0.2, 0.4))}
#: Item 5 and C6: conditions defined on top of the calibrated clean scene of the same dtype and seed.
ADDED_CONDITIONS = ("bright_outliers", "saturation", "gain_strong", "clean_unbalanced")
CALIBRATED_ALL = tuple(CALIBRATED_CONDITIONS) + ADDED_CONDITIONS
#: C2: uint16 only for these conditions; multi-FOV sets in uint8 only (C2 equals reductions R1 and R2).
UINT16_CONDITIONS = ("clean", "combined", "bright_outliers")
#: Item 5 and C8: four texture blobs at 4 x the preset's brightness median, log SD 0.1, fixed widths.
BRIGHT_OUTLIERS = dict(count=4, axial_width=1.5, lateral_width=2.0, brightness_factor=4.0, log_sd=0.1)
#: Item 5 and C1: the clipped fraction f reached by the saturation condition; k = 2^(j/4), j = 0..32.
SATURATION_FRACTION = 1e-3
SATURATION_STEPS = tuple(range(33))
#: C6: the LN channel gain spread of W-238.
GAIN_STRONG_SPREAD = 3.15
#: C4: the capped Z radius of the 3D background on the 8-plane scenes.
RZ_CAP = 3
#: C3: the wall-time budget of the full run, in seconds.
BUDGET_SECONDS = 2700.0
#: Items 3 and 4: the endpoints, and the decimals every mean, range and delta is rounded to.
RULE_ENDPOINTS = ("max_f1", "correct_fraction")
RULE_DECIMALS = 10
#: Item 3, scope: the precondition is not applied to these conditions.
NO_PRECONDITION = ("clean", "combined", "clean_unbalanced")
#: C5: the reference and degraded FOV roles of each multi-FOV set's set-specific check.
SET_SPECIFIC_FOVS = {"mf_density": ("dense", "near_empty"), "mf_gain_drift": ("gain_1.00", "gain_0.50")}
#: Item 3 and C5 (amended): the multi-FOV checks that gate a recipe's sample-level-fitting flag. The
#: general check (none on the set against mf_density_clean) follows item 3's rule, and C5 adds the
#: set-specific check of the recipe's per-FOV-fitted arm; a set is eligible only when both pass.
MULTI_FOV_GATES = ("fixture", "set_specific")
#: Multi-FOV spread rows: one per operating point and threshold value (adaptive has two fixed points).
SPREAD_KEYS = ("multi_fov", "dtype", "arm", "operating_point", "threshold")
#: Held-out endpoints at each fixed point of a mode (fixed1 = noise 5 or adaptive 0.2; fixed2 = adaptive 0.4).
FIXED_ENDPOINTS = ("f1", "correct_fraction", "reads_wrong_gene", "reads_false_detection")
CALIBRATED_ENDPOINTS = (("max_f1", "auprc", "correct_fraction", "color_call_agreement", "reads_wrong_gene",
                         "reads_false_detection") + tuple(f"{m}_fixed{i}" for i in (1, 2) for m in FIXED_ENDPOINTS))
#: Pilot subset: every added condition in uint8, uint8 and uint16 of clean and bright_outliers, and one
#: multi-FOV set with its clean reference (both precondition checks), on seeds 0 and 100.
PILOT_CONDITIONS = ("clean", "combined", "gain_strong", "bright_outliers", "saturation", "clean_unbalanced")
PILOT_UINT16 = ("clean", "bright_outliers")
PILOT_MULTI_FOV = ("mf_density", "mf_density_clean")
#: W-238 development target ranges, adaptive selection: min and max over the four datasets
#: (docs/image-statistics.md, *Target ranges*; tables/target_ranges.csv of the W-238 run
#: runs/W-238/20260928T194004Z-1da19f6c/measurement, outside the repository).
TARGET_RANGES = {
    "peak_p50": (81.0, 89.0), "amplitude_p10": (33.0, 49.0), "amplitude_p50": (72.0, 82.0),
    "amplitude_p90": (130.0, 174.0), "snr_clutter_p10": (2.2633262980499285, 3.8453983794557383),
    "snr_clutter_p50": (5.969868438448263, 11.149339815194574),
    "snr_clutter_p90": (15.018030997114124, 45.004945662174066),
    "snr_pixel_p10": (6.587029069583931, 14.512916497295265), "snr_pixel_p50": (12.907496598694987, 37.73092464817424),
    "snr_pixel_p90": (30.55415444157595, 112.22874597856142),
    "background_fraction_p50": (0.014084507042253521, 0.09523809523809523),
    "clutter_sigma_p50": (7.4131371395019094, 13.775640116324883),
    "pixel_sigma_p50": (2.2072476889837485, 6.41861747860156),
    "clutter_pixel_ratio_p50": (2.1227445404526852, 3.256174483833088),
    "zero_fraction_p50": (0.5827006134721968, 0.9189336745879684),
    "channel_gain_spread": (1.2638888888888888, 3.1463414634146343),
    "round_trend": (1.0389610389610389, 2.0377358490566038),
    "depth_attenuation_p50": (0.8063949631438823, 6.681724154519651),
    "sigma_log_amplitude_within_volume": (0.36419681604716564, 0.499913613563966),
    "sigma_log_peak_truncated_fit_p50": (0.4197027802304654, 0.49530414412823753),
}
#: Statistics in grey levels: their uint16 targets are the uint8 ranges x 16, a scale unverified against real data.
INTENSITY_STATISTICS = ("peak_p50", "amplitude_p10", "amplitude_p50", "amplitude_p90", "clutter_sigma_p50",
                        "pixel_sigma_p50")

_INT_CALIBRATED = ("gain", "trend", "gain_strong")
_PCT_CALIBRATED = _INT_CALIBRATED + ("bright_outliers",)
_XSRC_CALIBRATED = ("gain_baseline", "gain_texture", "saturation")
#: Before/after comparisons of the rerun. Targets follow item 5: bright_outliers replaces texture as
#: percentile normalization's outlier target, saturation joins the extraction-source targets, and
#: gain_strong (C6) joins every target list that holds gain.
CALIBRATED_COMPARISONS = [
    ("scalar_background", "isolated", "none", "scalar", ("baseline",)),
    ("scalar_background", "ablation", "pct", "r2_scalar", ("baseline",)),
    ("background_3d", "isolated", "none", "bg3d", _BG),
    ("background_3d", "ablation", "pct", "r2_3d", _BG),
    ("percentile_normalization", "isolated", "none", "pct", _PCT_CALIBRATED),
    ("percentile_normalization", "ablation_scalar", "scalar", "r2_scalar", _PCT_CALIBRATED),
    ("percentile_normalization", "ablation_3d", "bg3d", "r2_3d", _PCT_CALIBRATED),
    ("extraction_source", "ablation_scalar", "r2_scalar", "r2_scalar_xsrc", _XSRC_CALIBRATED),
    ("extraction_source", "ablation_3d", "r2_3d", "r2_3d_xsrc", _XSRC_CALIBRATED),
    ("min_max_normalization", "isolated", "none", "minmax", _INT_CALIBRATED),
    ("min_max_normalization", "ablation", "hist", "r1", _INT_CALIBRATED),
    ("histogram_matching", "isolated", "none", "hist", _INT_CALIBRATED),
    ("histogram_matching", "ablation", "minmax", "r1", _INT_CALIBRATED),
    ("reconstruction", "isolated", "none", "recon", _BG),
    ("reconstruction", "ablation", "r1", "r1_recon", _BG),
    ("white_tophat", "isolated", "none", "tophat", _BG),
]


def calibrated_extra_roles(method, comparison):
    """Rows reported beside a comparison's flag: bright_outliers for min-max and histogram matching
    (reported, outside their flags) and clean_unbalanced for histogram matching's harm test (C7)."""
    roles = []
    if method in ("min_max_normalization", "histogram_matching"):
        roles.append(("bright_outliers", "reported"))
    if method == "histogram_matching":
        roles.append(("clean_unbalanced", "harm_test"))
    return roles


def calibrated_targets():
    """Condition -> sorted names of the methods that target it (single-FOV and multi-FOV comparisons)."""
    targets = {}
    for method, _mode, _before, _after, conditions in CALIBRATED_COMPARISONS + SAMPLE_COMPARISONS:
        for condition in conditions:
            targets.setdefault(condition, set()).add(method)
    return {condition: sorted(methods) for condition, methods in targets.items()}


# Scenes -------------------------------------------------------------------------------------------

def strong_gain_factors(clean_gains, spread=GAIN_STRONG_SPREAD):
    """Channel factors for gain_strong: log-linear across channels with geometric mean 1, chosen so the
    clean channel gains times the factors span max/min = spread (the brightest clean channel is first)."""
    gains = np.asarray(clean_gains, dtype=np.float64)
    if not (np.all(np.diff(gains) <= 0) and gains[-1] > 0):
        raise ValueError("clean channel gains must be positive and non-increasing")
    own = spread * gains[-1] / gains[0]
    position = ((len(gains) - 1) / 2 - np.arange(len(gains))) / (len(gains) - 1)
    return own ** position


def saturation_config(config, k):
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


def calibrated_scene_config(condition, *, dtype, seed, k=None, fov_id=None, gain=1.0, count=None, scene_key=None,
                            shape=None):
    """(codebook, config) for one FOV of the calibrated rerun (amendment items 1 and 5).

    Every condition starts from calibrated_scene_preset(base, dtype, seed=seed, codebook=...), used as
    returned: the 13 CALIBRATED_CONDITIONS are their own base; the added conditions are built on the
    calibrated clean scene. Only clean_unbalanced uses the unbalanced codebook. Multi-FOV FOVs replace
    count, FOV_id, scene_key and the readout gains (x gain) and clear coordinates and IDs. shape and a
    smaller count are for the default-tier smoke test only.
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
        config = saturation_config(config, k)
    elif condition == "gain_strong":
        gains = np.asarray(config.readout.gains, dtype=float)
        config = replace(config, readout=replace(config.readout, gains=(gains * strong_gain_factors(gains[0])).tolist()))
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


def calibrated_background_radius(config):
    """(radius, uncapped): r = ceil(3 sigma) + 1 per axis from the preset's median widths (the exponentials
    of the lognormal locations), with r_z capped at 3 (C4)."""
    sz, sl = (float(np.exp(d.parameters[0])) for d in (config.axial_width, config.lateral_width))
    uncapped = tuple(int(math.ceil(3 * s)) + 1 for s in (sz, sl, sl))
    radius = (min(uncapped[0], RZ_CAP), *uncapped[1:])
    if 2 * radius[0] + 1 > config.shape_zyx[0]:
        raise ValueError(f"the capped Z footprint {2 * radius[0] + 1} exceeds {config.shape_zyx[0]} planes")
    return radius, uncapped


def clipped_fraction(scene):
    """Item 5: the generator's `above` clipping count summed over rounds and channels over rounds x channels x voxels."""
    counts = scene.provenance["clipping_counts"]
    return sum(int(c["above"]) for c in counts.values()) / sum(int(scene.rounds[r].size) for r in counts)


def _saturation_fraction(dtype, seed, k):
    book, config = calibrated_scene_config("saturation", dtype=dtype, seed=seed, k=k)
    return clipped_fraction(generate_formed_scene(book, config=config))


def saturation_k(dtype, *, fraction=_saturation_fraction, target=SATURATION_FRACTION, seeds=DEV_SEEDS,
                 steps=SATURATION_STEPS):
    """Item 5 and C1: k is the smallest 2^(j/4), j in steps, whose mean clipped fraction over the
    development seeds reaches target. fraction(dtype, seed, k) generates only development scenes.
    Returns k, j, each seed's fraction and the search trace; raises when no step reaches the target."""
    trace = []
    for j in steps:
        k = 2.0 ** (j / 4)
        values = [float(fraction(dtype, seed, k)) for seed in seeds]
        mean = float(np.mean(values))
        trace.append(dict(j=j, k=k, mean_clipped_fraction=mean))
        if mean >= target:
            return dict(dtype=dtype, j=j, k=k, target_fraction=target, seeds=list(seeds),
                        per_seed_clipped_fraction=dict(zip(map(str, seeds), values)), mean_clipped_fraction=mean,
                        trace=trace)
    raise RuntimeError(f"no k = 2^(j/4) with j <= {steps[-1]} reaches a mean clipped fraction of {target} ({dtype})")


def condition_definitions(k_values):
    """Manifest record of every calibrated condition: preset base, codebook and the item 5 field changes."""
    definitions = {}
    for condition in CALIBRATED_ALL:
        base = condition if condition in CALIBRATED_CONDITIONS else "clean"
        record = dict(preset=f"calibrated_scene_preset ({CALIBRATED_VERSION})", base=base,
                      factors=list(CALIBRATED_CONDITIONS[base]),
                      codebook="unbalanced" if condition == "clean_unbalanced" else "balanced",
                      dtypes=["uint8", "uint16"] if condition in UINT16_CONDITIONS else ["uint8"])
        if condition == "bright_outliers":
            record["texture"] = dict(BRIGHT_OUTLIERS, brightness_median="4 x the preset brightness median "
                                     "(352 grey levels in uint8, 5632 in uint16)", tissue_weights=1.0,
                                     widths="constant")
        elif condition == "saturation":
            record.update(scaled=["brightness median", "pedestal", "noise.alpha", "noise.sigma",
                                  "noise.correlated_sigma"], target_fraction=SATURATION_FRACTION,
                          k={dtype: v["k"] for dtype, v in k_values.items()})
        elif condition == "gain_strong":
            _, clean = calibrated_scene_preset("clean", "uint8")
            gains = np.asarray(clean.readout.gains, dtype=float)[0]
            factors = strong_gain_factors(gains)
            record.update(channel_spread=GAIN_STRONG_SPREAD, clean_channel_gains=gains.tolist(),
                          channel_factors=factors.tolist(), channel_gains=(gains * factors).tolist())
        definitions[condition] = record
    return definitions


def calibrated_plan(scope, reductions=()):
    """Single-FOV (condition, dtype) units, multi-FOV (set, dtype) units and seeds of a scope.

    full is the approved matrix: every condition in uint8, uint16 only for UINT16_CONDITIONS (C2) and
    multi-FOV sets in uint8. R3 drops uint16. pilot and smoke are the bounded subsets.
    """
    if scope == "full":
        single = [(c, "uint8") for c in CALIBRATED_ALL] + [(c, "uint16") for c in UINT16_CONDITIONS]
        multi, seeds = [(name, "uint8") for name in MULTI_FOV], DEV_SEEDS + EVAL_SEEDS
    elif scope == "pilot":
        single = [(c, "uint8") for c in PILOT_CONDITIONS] + [(c, "uint16") for c in PILOT_UINT16]
        multi, seeds = [(name, "uint8") for name in PILOT_MULTI_FOV], VERIFY_SEEDS
    elif scope == "smoke":
        single, multi, seeds = [("clean", "uint8")], [], VERIFY_SEEDS
    else:
        raise ValueError(f"unknown calibrated scope: {scope}")
    unknown = set(reductions) - {"R3"}
    if unknown:
        raise ValueError(f"only reduction R3 remains (C2 equals R1 and R2): {sorted(unknown)}")
    if "R3" in reductions:
        single = [(c, d) for c, d in single if d != "uint16"]
    return dict(scope=scope, single_fov=single, multi_fov=multi, seeds=tuple(seeds), reductions=list(reductions))


def _plan_record(plan):
    return dict(scope=plan["scope"], seeds=list(plan["seeds"]), reductions=plan["reductions"],
                single_fov=[dict(condition=c, dtype=d) for c, d in plan["single_fov"]],
                multi_fov=[dict(multi_fov=s, dtype=d) for s, d in plan["multi_fov"]],
                uint16_conditions=sorted({c for c, d in plan["single_fov"] if d == "uint16"}),
                multi_fov_dtypes=sorted({d for _, d in plan["multi_fov"]}))


# Image statistics (W-238 tool) ----------------------------------------------------------------------

def _image_statistics():
    module = sys.modules.get("image_statistics")
    if module is None:
        spec = importlib.util.spec_from_file_location("image_statistics", Path(__file__).resolve().parent
                                                      / "image_statistics.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["image_statistics"] = module
        spec.loader.exec_module(module)
    return module


def measure_scene(scene, dataset, fov):
    """Measure every round and channel volume of a scene with the W-238 tool, labelled as one FOV of `dataset`."""
    ist = _image_statistics()
    records, puncta = [], []
    for round_label in scene.round_labels:
        for channel in scene.channel_labels:
            label = dict(dataset=dataset, fov=fov, split="development", round=round_label, channel=channel)
            measured = ist.measure_volume(ist.synthetic_channel(scene, round_label, channel))
            records.append(dict(label=label, source=None, record=ist._nan_to_none(measured.record)))
            table = measured.puncta
            for key in reversed(ist.LABELS):
                table.insert(0, key, label[key])
            puncta.append(table)
    return records, puncta


def statistics_table(measured):
    """Per condition and dtype: the W-238 adaptive-selection statistics next to the W-238 development ranges.

    measured maps dtype -> (volume records, puncta tables) of development-seed scenes; each condition
    (or multi-FOV set) is pooled like one W-238 dataset whose FOVs are its scenes.
    """
    ist = _image_statistics()
    rows = []
    for dtype, (records, puncta) in measured.items():
        targets = ist.target_table(ist.volume_table(records), pd.concat(puncta, ignore_index=True))
        for record in targets[targets.selection == "adaptive"].to_dict("records"):
            for statistic, (low, high) in TARGET_RANGES.items():
                scale = 16.0 if dtype == "uint16" and statistic in INTENSITY_STATISTICS else 1.0
                value = record.get(statistic)
                value = None if value is None or not np.isfinite(float(value)) else float(value)
                rows.append(dict(condition=record["dataset"], dtype=dtype, selection="adaptive", statistic=statistic,
                                 value=value, target_min=low * scale, target_max=high * scale, target_scale=scale,
                                 within_target_range=None if value is None else bool(low * scale <= value
                                                                                     <= high * scale),
                                 n_volumes=record["n_volumes"], n_puncta=record["n_puncta"]))
    return pd.DataFrame(rows)


# Operating points and endpoints ---------------------------------------------------------------------

def select_operating_point(dev_curves, grid):
    """Item 2: the development-selected value of one mode and group.

    The mean F1 over the development seeds at each grid value, in float64 with seeds in ascending
    order; the highest mean wins and exact ties go to the smallest value. Returns (value, mean F1,
    at_grid_edge); value is None when no grid value has a defined F1 on every seed.
    """
    best, best_mean = None, None
    for value in sorted(grid):
        f1 = dev_curves[dev_curves.threshold == value].sort_values("seed").f1.to_numpy(np.float64)
        if not len(f1) or not np.isfinite(f1).all():
            continue
        mean = float(np.mean(f1))
        if best is None or mean > best_mean:
            best, best_mean = value, mean
    return best, best_mean, best is not None and best in (min(grid), max(grid))


_POINT_METRICS = ("precision", "recall", "f1", "localization_error", "correct_fraction", "reads_correct",
                  "reads_wrong_gene", "reads_false_detection", "color_call_agreement")


def summarize_modes(curves, keys):
    """Per threshold mode: per-seed max-F1 and AUPRC, the development-selected values, the operating
    points on the held-out seeds (selected and fixed) and the held-out endpoints of item 2."""
    per_seed, points, ends, selected = [], [], [], {}
    for mode, spec in THRESHOLD_MODES.items():
        part = curves[curves.threshold_mode == mode]
        if not len(part):
            continue
        grid, fixed = spec["grid"], spec["fixed"]
        seed_rows = []
        for key, group in part.groupby(keys + ["seed"], sort=False):
            rows = sorted(group.to_dict("records"), key=lambda r: r["threshold"])
            defined = [r for r in rows if _defined(r["f1"])]
            best = max(defined, key=lambda r: r["f1"]) if defined else None  # first maximum: the smallest value
            record = dict(zip(keys + ["seed"], key), threshold_mode=mode,
                          split="development" if key[-1] in DEV_SEEDS else "evaluation",
                          max_f1=best["f1"] if best else None, max_f1_threshold=best["threshold"] if best else None,
                          auprc=_auprc(rows))
            per_seed.append(record)
            seed_rows.append((key[:-1], record, {r["threshold"]: r for r in rows}))
        for key, group in part[part.seed.isin(DEV_SEEDS)].groupby(keys, sort=False):
            value, mean, edge = select_operating_point(group, grid)
            selected[(mode, *key)] = dict(value=value, dev_mean_f1=mean, at_grid_edge=edge)
        for key, group in part[part.seed.isin(EVAL_SEEDS)].groupby(keys, sort=False):
            choice = selected.get((mode, *key), dict(value=None, dev_mean_f1=None, at_grid_edge=None))
            for label, value in (("dev_max_f1", choice["value"]), *(("fixed", v) for v in fixed)):
                if value is None:
                    continue
                rows = group[group.threshold == value]
                dev = label == "dev_max_f1"
                points.append(dict(zip(keys, key), threshold_mode=mode, operating_point=label, threshold=value,
                    at_grid_edge=choice["at_grid_edge"] if dev else None,
                    dev_mean_f1=choice["dev_mean_f1"] if dev else None, seeds=len(rows),
                    **{f"{m}_{s}": getattr(rows[m].astype(float), s)() for m in _POINT_METRICS
                       for s in ("mean", "min", "max")}))
        for key, record, by_value in seed_rows:
            if record["split"] != "evaluation":
                continue
            choice = selected.get((mode, *key), {})
            at = by_value.get(choice.get("value"))
            end = dict(record, selected_threshold=choice.get("value"), at_grid_edge=choice.get("at_grid_edge"))
            for metric in ("correct_fraction", "color_call_agreement", "reads_wrong_gene", "reads_false_detection"):
                end[metric] = at[metric] if at is not None else None
            for i, value in enumerate(fixed, 1):
                end[f"fixed{i}_threshold"] = value
                end.update({f"{m}_fixed{i}": by_value[value][m] for m in FIXED_ENDPOINTS})
            ends.append(end)
    return pd.DataFrame(per_seed), pd.DataFrame(points), selected, pd.DataFrame(ends)


def pool_multi_fov(mf):
    """Pool the multi-FOV curves over each set's FOVs per seed, mode and value (as W-233)."""
    pooled = (mf.groupby(["multi_fov", "dtype", "arm", "threshold_mode", "seed", "threshold"], sort=False)
              [["n_truth", "n_detected", "n_matched", "reads_accepted", "reads_correct", "reads_wrong_gene",
                "reads_false_detection"]].sum().reset_index())
    pooled["precision"] = pooled.n_matched / pooled.n_detected.where(pooled.n_detected > 0)
    pooled["recall"] = pooled.n_matched / pooled.n_truth.where(pooled.n_truth > 0)
    pooled["f1"] = (2 * pooled.precision * pooled.recall / (pooled.precision + pooled.recall)).fillna(0.0)
    pooled["correct_fraction"] = pooled.reads_correct / pooled.n_truth.where(pooled.n_truth > 0)
    pooled["color_call_agreement"] = np.nan
    pooled["localization_error"] = np.nan
    return pooled.rename(columns={"multi_fov": "condition"})


def mode_multi_fov_tables(mf, selected, mode):
    """Per-FOV rows of one mode at the set's pooled development-selected value and at each fixed point."""
    chosen = {key[1:]: v["value"] for key, v in selected.items() if key[0] == mode}
    return multi_fov_tables(mf[mf.threshold_mode == mode], chosen,
                            tuple(("fixed", v) for v in THRESHOLD_MODES[mode]["fixed"]))


def per_fov_endpoints(mf, selected):
    """Per held-out seed, mode, set, arm and FOV: max-F1 over the mode's grid, and the correct-decode
    fraction at the set's pooled development-selected value (the endpoints of the set-specific check)."""
    rows = []
    keys = ["multi_fov", "dtype", "arm", "threshold_mode", "seed", "fov_id", "role"]
    for key, group in mf[mf.seed.isin(EVAL_SEEDS)].groupby(keys, sort=False):
        name, dtype, arm, mode, *_ = key
        f1 = group.f1.to_numpy(np.float64)
        value = selected.get((mode, name, dtype, arm), {}).get("value")
        at = group[group.threshold == value].correct_fraction.to_numpy(np.float64)
        rows.append(dict(zip(keys, key), max_f1=float(np.nanmax(f1)) if np.isfinite(f1).any() else None,
                         correct_fraction=float(at[0]) if len(at) and np.isfinite(at[0]) else None,
                         selected_threshold=value))
    return pd.DataFrame(rows)


# Precondition, revised low-benefit rule and harm test ---------------------------------------------------

def _rounded(value):
    return round(float(value), RULE_DECIMALS)


def _series(values):
    return np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)


def precondition_check(reference, degraded):
    """Item 3 for one condition (or FOV pair), dtype and mode.

    reference and degraded map each endpoint to its values on the held-out seeds, paired and in
    ascending seed order. g_E = mean_s[reference - degraded] and R_E = the larger of the two seed ranges,
    each rounded to 10 decimals; E degrades when g_E > R_E, strictly. An endpoint that is undefined on
    some seed, or unpaired, does not degrade. The fixture is valid when some endpoint degrades.
    """
    out, notes = {}, []
    for endpoint in RULE_ENDPOINTS:
        ref, deg = _series(reference.get(endpoint, ())), _series(degraded.get(endpoint, ()))
        if len(ref) and len(ref) == len(deg) and np.isfinite(ref).all() and np.isfinite(deg).all():
            g, r = _rounded(np.mean(ref - deg)), _rounded(max(np.ptp(ref), np.ptp(deg)))
            out.update({f"g_{endpoint}": g, f"range_{endpoint}": r, f"degrades_{endpoint}": g > r})
        else:
            out.update({f"g_{endpoint}": None, f"range_{endpoint}": None, f"degrades_{endpoint}": False})
            notes.append(f"{endpoint} undefined (does not degrade)")
    out["valid"] = any(out[f"degrades_{e}"] for e in RULE_ENDPOINTS)
    out["note"] = "; ".join(notes)
    return out


def paired_rule(before, after):
    """Item 4 for one condition: per endpoint, the paired delta mean_s[after - before] and the seed range
    max(range before, range after), rounded to 10 decimals; None when undefined or unpaired."""
    out = {}
    for endpoint in RULE_ENDPOINTS:
        b, a = _series(before.get(endpoint, ())), _series(after.get(endpoint, ()))
        ok = bool(len(b)) and len(b) == len(a) and np.isfinite(a).all() and np.isfinite(b).all()
        out[f"rule_delta_{endpoint}"] = _rounded(np.mean(a - b)) if ok else None
        out[f"rule_range_{endpoint}"] = _rounded(max(np.ptp(a), np.ptp(b))) if ok else None
    return out


def _endpoint_lookup(ends, keys=("condition", "dtype", "arm", "threshold_mode")):
    """key -> (held-out seeds, endpoint -> values in ascending seed order)."""
    lookup = {}
    if not len(ends):
        return lookup
    for key, group in ends[ends.seed.isin(EVAL_SEEDS)].groupby(list(keys), sort=False):
        group = group.sort_values("seed")
        lookup[key] = (tuple(group.seed), {e: list(group[e]) for e in RULE_ENDPOINTS})
    return lookup


def _paired(first, second):
    """Endpoint values of two lookups, or empty ones when the held-out seeds do not pair."""
    if first is None or second is None or first[0] != second[0]:
        return {}, {}
    return first[1], second[1]


def preconditions_table(ends, mf_ends=None, per_fov=None):
    """Item 3 per targeted condition, dtype and mode; for multi-FOV sets both checks (C5).

    check "fixture": arm none on the condition against none on clean (multi-FOV sets: pooled, against
    mf_density_clean). check "set_specific": each recipe's per-FOV-fitted arm on the set's degraded FOV
    against its reference FOV. gates_flag says whether a failure excludes the condition from a flag.
    """
    targets = calibrated_targets()
    rows = []

    def add(check, condition, dtype, mode, arm, reference, degraded, pair, gates):
        seeds = pair[0][0] if pair[0] is not None and pair[1] is not None else ()
        ref, deg = _paired(*pair)
        rows.append(dict(check=check, condition=condition, dtype=dtype, threshold_mode=mode, arm=arm,
                         reference=reference, degraded=degraded, targeted_by=",".join(targets.get(condition, [])),
                         seeds=",".join(map(str, seeds)), gates_flag=gates, **precondition_check(ref, deg)))

    single = _endpoint_lookup(ends)
    for (condition, dtype, arm, mode), values in single.items():
        if arm == "none" and condition in targets and condition not in NO_PRECONDITION:
            add("fixture", condition, dtype, mode, "none", "clean", condition,
                (single.get(("clean", dtype, "none", mode)), values), True)
    if mf_ends is not None and len(mf_ends):
        pooled = _endpoint_lookup(mf_ends)
        for (condition, dtype, arm, mode), values in pooled.items():
            if arm == "none" and condition in SET_SPECIFIC_FOVS:
                add("fixture", condition, dtype, mode, "none", "mf_density_clean", condition,
                    (pooled.get(("mf_density_clean", dtype, "none", mode)), values), "fixture" in MULTI_FOV_GATES)
    if per_fov is not None and len(per_fov):
        fovs = _endpoint_lookup(per_fov.rename(columns={"multi_fov": "condition"}),
                                ("condition", "dtype", "arm", "threshold_mode", "role"))
        recipes = sorted({before for _m, _c, before, _a, _t in SAMPLE_COMPARISONS})
        for name, (reference, degraded) in SET_SPECIFIC_FOVS.items():
            for dtype, mode in sorted({(d, m) for c, d, _a, m, _r in fovs if c == name}):
                for recipe in recipes:
                    pair = (fovs.get((name, dtype, recipe, mode, reference)), fovs.get((name, dtype, recipe, mode,
                                                                                          degraded)))
                    if pair[0] is not None or pair[1] is not None:
                        add("set_specific", name, dtype, mode, recipe, reference, degraded, pair,
                            "set_specific" in MULTI_FOV_GATES)
    return pd.DataFrame(rows)


def fixture_gaps(preconditions):
    """Item 3, failure: one fixture-gap row per failed check (both degradations and both ranges)."""
    if not len(preconditions):
        return pd.DataFrame()
    columns = ["check", "condition", "dtype", "threshold_mode", "arm", "reference", "degraded", "targeted_by",
               "g_max_f1", "range_max_f1", "g_correct_fraction", "range_correct_fraction", "gates_flag", "note"]
    return preconditions.loc[~preconditions.valid.astype(bool), columns].reset_index(drop=True)


def attach_rule(comparisons, lookups, preconditions):
    """Add item 4's rounded paired delta and range per endpoint to every comparison row, and to each
    targeted row whether the condition is eligible (in T*) with its precondition status."""
    checks = {}
    for record in preconditions.to_dict("records") if len(preconditions) else []:
        checks[(record["check"], record["condition"], record["dtype"], record["threshold_mode"], record["arm"])] = record
    rows = []
    for record in comparisons.to_dict("records"):
        lookup = lookups["multi_fov" if record["condition"] in MULTI_FOV else "single_fov"]
        key = (record["dtype"], record["threshold_mode"])
        before, after = _paired(lookup.get((record["condition"], record["dtype"], record["before"], key[1])),
                                lookup.get((record["condition"], record["dtype"], record["after"], key[1])))
        record.update(paired_rule(before, after))
        if record["role"] == "targeted":
            if record["condition"] in SET_SPECIFIC_FOVS:
                found = {check: checks.get((check, record["condition"], *key, record["before"] if check ==
                                            "set_specific" else "none")) for check in ("fixture", "set_specific")}
            else:
                found = {"fixture": checks.get(("fixture", record["condition"], *key, "none"))}
            gates = [c for c in found if c in MULTI_FOV_GATES or record["condition"] not in SET_SPECIFIC_FOVS]
            eligible = all(found[c] is not None and bool(found[c]["valid"]) for c in gates)
            status = [f"{c} {'missing' if r is None else 'passed' if r['valid'] else 'failed'}"
                      + ("" if c in gates else " (reported only, C5)") for c, r in found.items()]
            record.update(eligible=eligible, fixture_gap=not eligible, precondition="; ".join(status))
        rows.append(record)
    return pd.DataFrame(rows)


def _rule_values(record):
    delta = {e: _num(record.get(f"rule_delta_{e}")) for e in RULE_ENDPOINTS}
    span = {e: _num(record.get(f"rule_range_{e}")) for e in RULE_ENDPOINTS}
    undefined = [e for e in RULE_ENDPOINTS if np.isnan(delta[e]) or np.isnan(span[e])]
    return delta, span, undefined


def _benefit(record):
    """Item 4 on one eligible targeted condition: (benefit, failing clause, anomaly)."""
    delta, span, undefined = _rule_values(record)
    improved = [e for e in RULE_ENDPOINTS if e not in undefined and delta[e] >= max(span[e], LOW_BENEFIT_POINTS)]
    anomaly = f"undefined {', '.join(undefined)}" if undefined else ""
    for endpoint in improved:
        other = next(e for e in RULE_ENDPOINTS if e != endpoint)
        if other not in undefined and delta[other] >= -span[other]:
            return True, "", anomaly
    if not improved:
        return False, "no endpoint improved by at least max(seed range, 0.02)", anomaly
    endpoint = improved[0]
    other = next(e for e in RULE_ENDPOINTS if e != endpoint)
    if other in undefined:
        return False, f"{endpoint} improved but {other} is undefined", anomaly
    return (False, f"{endpoint} improved but {other} worsened beyond its seed range "
                   f"({delta[other]:+.4f} < -{span[other]:.4f})", anomaly)


def _clean_holds(record):
    """Item 4's clean clause: (holds, failing clause, anomaly)."""
    if record is None:
        return False, "clean not run", "clean not run"
    delta, span, undefined = _rule_values(record)
    worse = [e for e in RULE_ENDPOINTS if e not in undefined and delta[e] < -span[e]]
    clauses = ([f"{e} worsened on clean beyond its seed range ({delta[e]:+.4f} < -{span[e]:.4f})" for e in worse]
               + [f"{e} undefined on clean" for e in undefined])
    return not clauses, "; ".join(clauses), (f"undefined {', '.join(undefined)} on clean" if undefined else "")


def revised_flags(comparisons):
    """Item 4 (provisional) per comparison, dtype and mode.

    One row per targeted condition naming the failing clause (or its fixture gap), one clean row, and
    one method row whose result is not_flagged, low_benefit or not_assessable (T* empty). Rows with
    role combined, reported or harm_test do not enter the flag.
    """
    rows = []
    if not len(comparisons):
        return pd.DataFrame(rows)
    keys = ["method", "comparison", "before", "after", "dtype", "threshold_mode"]
    for key, group in comparisons.groupby(keys, sort=False):
        common = dict(zip(keys, key), threshold_points=LOW_BENEFIT_POINTS, provisional=True)
        values = lambda r: {f"{p}_{e}": r.get(f"rule_{p}_{e}") for p in ("delta", "range")  # noqa: E731
                            for e in RULE_ENDPOINTS}
        benefits, gaps, anomalies, shown = [], [], [], []
        for record in group[group.role == "targeted"].to_dict("records"):
            if not record.get("eligible"):
                gaps.append(f"{record['condition']} ({record.get('precondition')})")
                benefit, clause = None, f"fixture gap, not in T*: {record.get('precondition')}"
            else:
                benefit, clause, anomaly = _benefit(record)
                benefits.append(benefit)
                if benefit:
                    shown.append(record["condition"])
                if anomaly:
                    anomalies.append(f"{record['condition']}: {anomaly}")
            rows.append(dict(common, level="targeted_condition", condition=record["condition"],
                             eligible=bool(record.get("eligible")), benefit=benefit, clause=clause, **values(record)))
        clean = group[group.role == "harm_check"].to_dict("records")
        holds, clean_clause, anomaly = _clean_holds(clean[0] if clean else None)
        if anomaly:
            anomalies.append(anomaly)
        rows.append(dict(common, level="clean", condition=clean[0]["condition"] if clean else None, clean_holds=holds,
                         clause=clean_clause, **(values(clean[0]) if clean else {})))
        if not benefits:
            result = "not_assessable"
            reason = "T* is empty: " + ("; ".join(f"fixture gap {g}" for g in gaps) if gaps else
                                        "no targeted condition was run in this dtype")
        elif any(benefits) and holds:
            result, reason = "not_flagged", f"benefit on {', '.join(shown)}; clean holds"
        else:
            result = "low_benefit"
            reason = "; ".join(([] if any(benefits) else ["no eligible targeted condition shows a benefit"])
                               + ([] if holds else [f"clean does not hold: {clean_clause}"]))
        if result == "not_assessable" and not holds:
            reason += f"; clean does not hold: {clean_clause}"
        rows.append(dict(common, level="method", condition="all targeted", result=result,
                         low_benefit=result == "low_benefit", clean_holds=holds, eligible_targets=len(benefits),
                         reason=reason, anomalies="; ".join(anomalies)))
    return pd.DataFrame(rows)


def harm_test_table(comparisons, flags):
    """C7: histogram matching's unbalanced-codebook harm test per comparison, dtype and mode, reported
    beside the comparison's flag result and its balanced clean deltas; it does not enter the flag."""
    rows = []
    if not len(comparisons):
        return pd.DataFrame(rows)
    keys = ["method", "comparison", "before", "after", "dtype", "threshold_mode"]
    tests = comparisons[(comparisons.method == "histogram_matching") & (comparisons.role == "harm_test")]
    for record in tests.to_dict("records"):
        match = lambda t: np.logical_and.reduce([t[k] == record[k] for k in keys])  # noqa: E731
        delta, span, undefined = _rule_values(record)
        harmed = [e for e in RULE_ENDPOINTS if e not in undefined and delta[e] < -span[e]]
        clean = comparisons[match(comparisons) & (comparisons.role == "harm_check")].to_dict("records")
        flag = flags[match(flags) & (flags.level == "method")].to_dict("records") if len(flags) else []
        rows.append(dict({k: record[k] for k in keys}, condition=record["condition"], harm=bool(harmed),
            harm_endpoints=",".join(harmed), anomalies=f"undefined {', '.join(undefined)}" if undefined else "",
            **{f"unbalanced_{p}_{e}": record.get(f"rule_{p}_{e}") for p in ("delta", "range") for e in RULE_ENDPOINTS},
            **{f"clean_{p}_{e}": clean[0].get(f"rule_{p}_{e}") if clean else None for p in ("delta", "range")
               for e in RULE_ENDPOINTS},
            flag_result=flag[0]["result"] if flag else None, enters_flag=False))
    return pd.DataFrame(rows)


# Projection -------------------------------------------------------------------------------------------

def project(units, fixed_seconds, pilot_plan, full_plans):
    """Project the compute time of full plans from the pilot's measured unit costs.

    A unit is one condition (or multi-FOV set), dtype and seed with every arm in both threshold modes.
    Each full unit costs the pilot's mean for the same condition and dtype, or else the largest mean
    measured for any condition of that kind and dtype (conservative). The pilot's seeds 0 and 100 also
    run the verification rerun and seed 0 the image statistics, so the mean over them is an upper
    estimate for the other seeds. The k search is measured once per dtype. Post-processing (tables and
    manifest) scales with the number of units. Wall overhead outside the timed compute is added by
    attach-time from the /usr/bin/time -v record.
    """
    measured = {}
    for unit in units:
        measured.setdefault((unit["kind"], unit["name"], unit["dtype"]), []).append(unit["seconds"])
    means = {key: float(np.mean(v)) for key, v in measured.items()}
    pilot_units = sum(len(pilot_plan["seeds"]) for _ in pilot_plan["single_fov"] + pilot_plan["multi_fov"])
    result = dict(basis=project.__doc__.split("\n\n")[1].replace("\n    ", " ").strip(),
                  unit_seconds={f"{k}:{n}:{d}": v for (k, n, d), v in means.items()}, budget_seconds=BUDGET_SECONDS,
                  fixed_seconds=fixed_seconds, plans={})
    for label, plan in full_plans.items():
        total, missing = 0.0, []
        for kind, pairs in (("single_fov", plan["single_fov"]), ("multi_fov", plan["multi_fov"])):
            for name, dtype in pairs:
                cost = means.get((kind, name, dtype))
                if cost is None:
                    same = [v for (k, _n, d), v in means.items() if k == kind and d == dtype]
                    same = same or [v for (k, _n, _d), v in means.items() if k == kind]
                    cost = max(same)
                    missing.append(f"{name}:{dtype}")
                total += cost * len(plan["seeds"])
        search = sum(fixed_seconds["saturation_k"].get(d, 0.0) for c, d in plan["single_fov"] if c == "saturation")
        n_units = sum(len(plan["seeds"]) for _ in plan["single_fov"] + plan["multi_fov"])
        post = fixed_seconds["post_processing"] * n_units / max(pilot_units, 1)
        seconds = total + search + post + fixed_seconds["overhead"]
        result["plans"][label] = dict(units=n_units, unit_seconds=total, saturation_k_seconds=search,
                                      post_processing_seconds=post, overhead_seconds=fixed_seconds["overhead"],
                                      compute_seconds=seconds, projected_seconds=seconds,
                                      within_budget=seconds <= BUDGET_SECONDS,
                                      costed_at_largest_measured=sorted(missing))
    return result


def project_wall(projection, wall_overhead):
    """Add the measured wall time outside the timed compute (start-up, imports, shutdown) to each plan."""
    projection = dict(projection, wall_overhead_seconds=wall_overhead)
    for plan in projection["plans"].values():
        plan["projected_seconds"] = plan["compute_seconds"] + wall_overhead
        plan["within_budget"] = plan["projected_seconds"] <= BUDGET_SECONDS
    return projection


# Runner -------------------------------------------------------------------------------------------------

def _mode_sweeps(fov, truth, verify):
    rows = []
    for mode, spec in THRESHOLD_MODES.items():
        for r in sweep(fov, truth, verify, mode, spec["grid"], spec["fixed"][0]):
            cutoffs = r.pop("noise_cutoffs")  # the cutoffs of this mode, not only noise-mode ones
            rows.append(dict(threshold_mode=mode, **r, cutoffs=cutoffs))
    return rows


def run_calibrated(output, *, scope="full", reductions=(), issue="W-239", shape=None, count=None, log=print):
    """Run the calibrated rerun matrix (or its pilot or smoke subset) and write tables and a manifest."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} exists and is not empty")
    if scope != "smoke" and (shape is not None or count is not None):
        raise ValueError("shape and count overrides are for the smoke scope only")
    output.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # Calibrated scenes clip their negative noise to 0 in every round (zero fraction 0.14-0.23, W-241);
        # the generator warns above 1 %, and each scene's clipping counts are recorded in the manifest.
        warnings.filterwarnings("ignore", r"round .* voxel values .* were clipped", RuntimeWarning)
        return _run_calibrated(output, scope, reductions, issue, shape, count, log)


def _run_calibrated(output, scope, reductions, issue, shape, count, log):
    started = time.perf_counter()
    plan = calibrated_plan(scope, reductions)
    seeds = plan["seeds"]
    timing, k_values = dict(saturation_k={}), {}
    # Item 5: k is set per dtype from development seeds before any held-out scene is generated.
    for dtype in sorted({d for c, d in plan["single_fov"] if c == "saturation"}):
        t0 = time.perf_counter()
        k_values[dtype] = saturation_k(dtype)
        timing["saturation_k"][dtype] = time.perf_counter() - t0
        log(f"saturation k ({dtype}): {k_values[dtype]['k']:.4f} (j={k_values[dtype]['j']}), "
            f"{timing['saturation_k'][dtype]:.1f} s")
    curves, direct, diagnostics, scenes_record, runs_record, units = [], [], [], [], [], []
    mf_curves, mf_scenes, measured, stats_seconds = [], [], {}, 0.0
    with tempfile.TemporaryDirectory(prefix="w239-") as workdir:
        for condition, dtype in plan["single_fov"]:
            register = condition == "combined_geometry"
            for seed in seeds:
                t0 = time.perf_counter()
                book, config = calibrated_scene_config(condition, dtype=dtype, seed=seed, k=k_values.get(dtype, {})
                                                       .get("k") if condition == "saturation" else None,
                                                       shape=shape, count=count)
                scene, background, signal, truth = generate(book, config)
                radius, uncapped = calibrated_background_radius(config)
                base = dict(condition=condition, dtype=dtype, seed=seed,
                            split="development" if seed in DEV_SEEDS else "evaluation")
                scenes_record.append(dict(base, dataset_version=config.dataset_version, scene_key=config.scene_key,
                    codebook="unbalanced" if condition == "clean_unbalanced" else "balanced",
                    shape_zyx=list(config.shape_zyx), count=config.count,
                    requested_config_sha256=hashlib.sha256(json.dumps(scene.provenance["requested_config"],
                        sort_keys=True).encode()).hexdigest(), image_sha256=scene.provenance["image_sha256"],
                    clipping_counts=scene.provenance["clipping_counts"], clipped_fraction=clipped_fraction(scene),
                    n_truth=len(truth), background_radius_voxels_zyx=list(radius),
                    background_radius_uncapped_zyx=list(uncapped), saturation_k=k_values[dtype]["k"]
                    if condition == "saturation" else None))
                if seed in DEV_SEEDS:
                    t1 = time.perf_counter()
                    records, puncta = measure_scene(scene, condition, f"seed{seed}")
                    measured.setdefault(dtype, ([], []))[0].extend(records)
                    measured[dtype][1].extend(puncta)
                    stats_seconds += time.perf_counter() - t1
                for arm, recipe, fov in processed_arms(scene, book, config.FOV_id, workdir, radius, register):
                    direct.append(dict(base, arm=arm, **direct_metrics(fov, scene.rounds, background, signal,
                                                                       truth, book, arm, recipe)))
                    diagnostics.extend(dict(base, arm=arm, **r) for r in diagnostics_rows(fov, book))
                    runs_record.append(dict(base, arm=arm, output_sha256=output_digest(fov, book)))
                    curves.extend(dict(base, arm=arm, **r) for r in _mode_sweeps(fov, truth, seed in VERIFY_SEEDS))
                units.append(dict(kind="single_fov", name=condition, dtype=dtype, seed=seed,
                                  seconds=time.perf_counter() - t0))
                log(f"{condition} {dtype} seed {seed}: {units[-1]['seconds']:.1f} s")
        for name, dtype in plan["multi_fov"]:
            spec = MULTI_FOV[name]
            for seed in seeds:
                t0 = time.perf_counter()
                fovs = {}
                for k, (role, n, gain) in enumerate(zip(FOV_ROLES[name], spec["counts"], spec["gains"])):
                    fov_id = f"Position{k + 1:03d}"
                    # The scene key names the FOV position only, so a seed pairs across sets as across conditions.
                    book, config = calibrated_scene_config(spec["base"], dtype=dtype, seed=seed, fov_id=fov_id,
                        gain=gain, count=n if count is None else min(n, count), shape=shape,
                        scene_key=f"{CALIBRATED_VERSION}/multi-fov/{fov_id}")
                    fovs[fov_id] = (*generate(book, config), role, config)
                    scene = fovs[fov_id][0]
                    radius, uncapped = calibrated_background_radius(config)
                    mf_scenes.append(dict(multi_fov=name, dtype=dtype, seed=seed, fov_id=fov_id, role=role,
                        count=config.count, gain_multiplier=gain, base_condition=spec["base"],
                        scene_key=config.scene_key, image_sha256=scene.provenance["image_sha256"],
                        clipped_fraction=clipped_fraction(scene), n_truth=len(fovs[fov_id][3]),
                        background_radius_voxels_zyx=list(radius), background_radius_uncapped_zyx=list(uncapped)))
                    if seed in DEV_SEEDS:
                        t1 = time.perf_counter()
                        records, puncta = measure_scene(scene, name, f"seed{seed}-{fov_id}")
                        measured.setdefault(dtype, ([], []))[0].extend(records)
                        measured[dtype][1].extend(puncta)
                        stats_seconds += time.perf_counter() - t1
                for arm, (recipe_name, fit) in MULTI_FOV_ARMS.items():
                    path = Path(workdir) / f"supplied-{name}-{dtype}-{seed}-{arm}.json"
                    recipe = arms(radius, fit, path)[recipe_name]
                    supplied = fit_supplied(recipe, fovs, book, workdir) if fit == "supplied" else None
                    for fov_id, (scene, _background, _signal, truth, role, _config) in fovs.items():
                        fov = preprocess(make_fov(scene, book, fov_id, workdir), recipe, False)
                        base = dict(multi_fov=name, dtype=dtype, seed=seed, fov_id=fov_id, role=role, arm=arm,
                                    fit=fit, split="development" if seed in DEV_SEEDS else "evaluation")
                        runs_record.append(dict(base, output_sha256=output_digest(fov, book),
                            supplied_sections=None if supplied is None else sorted(supplied)))
                        diagnostics.extend(dict(base, condition=name, **r) for r in diagnostics_rows(fov, book))
                        mf_curves.extend(dict(base, **r) for r in _mode_sweeps(fov, truth, seed in VERIFY_SEEDS))
                units.append(dict(kind="multi_fov", name=name, dtype=dtype, seed=seed,
                                  seconds=time.perf_counter() - t0))
                log(f"{name} {dtype} seed {seed}: {units[-1]['seconds']:.1f} s")
    post_started = time.perf_counter()
    curves, direct, diagnostics = pd.DataFrame(curves), pd.DataFrame(direct), pd.DataFrame(diagnostics)
    mf = pd.DataFrame(mf_curves)
    files = {}

    def save(frame, relative):
        _write_csv(frame, output / relative)
        files[relative] = output / relative

    keys = ["condition", "dtype", "arm"]
    per_seed, points, selected, ends = summarize_modes(curves, keys)
    lookups = dict(single_fov=_endpoint_lookup(ends), multi_fov={})
    diag_ref = diagnostics[diagnostics["round"] == diagnostics["round"].iloc[0]]
    single_diag = diag_ref[diag_ref.condition.isin([c for c, _ in plan["single_fov"]])]
    comparisons = [comparisons_table(ends[ends.threshold_mode == mode], direct, single_diag, CALIBRATED_COMPARISONS,
                                     "clean", "combined", calibrated_extra_roles, CALIBRATED_ENDPOINTS)
                   .assign(threshold_mode=mode) for mode in THRESHOLD_MODES if len(ends)]
    mf_ends = per_fov = None
    save(curves, "curves/pr_curves.csv")
    if len(mf):
        save(mf, "curves/multi_fov_curves.csv")
        mf_per_seed, mf_points, mf_selected, mf_ends = summarize_modes(pool_multi_fov(mf), keys)
        per_seed = pd.concat([per_seed.assign(fov_scope="single_fov"), mf_per_seed.assign(fov_scope="multi_fov_pooled")],
                             ignore_index=True)
        points = pd.concat([points.assign(fov_scope="single_fov"), mf_points.assign(fov_scope="multi_fov_pooled")],
                           ignore_index=True)
        per_fov = per_fov_endpoints(mf, mf_selected)
        save(per_fov, "tables/multi_fov_per_fov_endpoints.csv")
        lookups["multi_fov"] = _endpoint_lookup(mf_ends)
        mf_diag = diag_ref[diag_ref.condition.isin([s for s, _ in plan["multi_fov"]])]
        spreads = []
        for mode in THRESHOLD_MODES:
            tables = mode_multi_fov_tables(mf, mf_selected, mode)
            if not len(tables):
                continue
            spreads.append(multi_fov_spread(tables, SPREAD_KEYS).assign(threshold_mode=mode))
            spread_direct = tables[tables.operating_point == "dev_max_f1"].rename(columns={"multi_fov": "condition"})
            spread_direct = (spread_direct.groupby(["condition", "dtype", "arm", "seed"]).correct_fraction
                             .agg(lambda v: v.max() - v.min()).rename("per_fov_correct_fraction_range").reset_index())
            mode_ends = mf_ends[mf_ends.threshold_mode == mode] if len(mf_ends) else mf_ends
            mf_cmp = comparisons_table(mode_ends, None, mf_diag, SAMPLE_COMPARISONS, "mf_density_clean", None,
                                       endpoints=CALIBRATED_ENDPOINTS).assign(threshold_mode=mode)
            _add_per_fov_spread(mf_cmp, spread_direct)
            comparisons.append(mf_cmp)
        if spreads:
            save(pd.concat(spreads, ignore_index=True), "tables/multi_fov_spread.csv")
    comparisons = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    preconditions = preconditions_table(ends, mf_ends, per_fov)
    gaps = fixture_gaps(preconditions)
    comparisons = attach_rule(comparisons, lookups, preconditions) if len(comparisons) else comparisons
    flags = revised_flags(comparisons)
    harm = harm_test_table(comparisons, flags)
    statistics = statistics_table(measured) if measured else pd.DataFrame()
    for frame, relative in ((direct, "tables/direct_metrics.csv"), (diagnostics, "tables/diagnostics.csv"),
                            (per_seed, "tables/recipe_per_seed.csv"), (points, "tables/operating_points.csv"),
                            (ends, "tables/endpoints.csv"), (preconditions, "tables/preconditions.csv"),
                            (gaps, "tables/fixture_gaps.csv"), (comparisons, "tables/comparisons.csv"),
                            (flags, "tables/low_benefit_flags.csv"), (harm, "tables/harm_test.csv"),
                            (statistics, "tables/image_statistics.csv")):
        save(frame, relative)
    if mf_ends is not None:
        save(mf_ends, "tables/multi_fov_endpoints.csv")
    verification = {}
    for mode, spec in THRESHOLD_MODES.items():
        verified = [r for r in curves.to_dict("records") + mf.to_dict("records")
                    if r["threshold_mode"] == mode and r["verified_direct_run"]]
        expected = (sum(len(PRIMARY_ARMS) for _c, _d in plan["single_fov"] for s in seeds if s in VERIFY_SEEDS)
                    + sum(len(MULTI_FOV_ARMS) * len(FOV_ROLES[n]) for n, _d in plan["multi_fov"]
                          for s in seeds if s in VERIFY_SEEDS))
        verification[mode] = dict(threshold_value=spec["fixed"][0], seeds=list(VERIFY_SEEDS), runs_verified=len(verified),
                                  runs_expected=expected, every_arm_verified=len(verified) == expected,
                                  arms=sorted({r["arm"] for r in verified}),
                                  rule="subset of the lowest-value run equals a direct FOV.run at this value (spots "
                                       "and reads); a difference stops the run")
    timing["post_processing"] = time.perf_counter() - post_started
    compute_seconds = time.perf_counter() - started
    timing["image_statistics"] = stats_seconds
    timing["units"] = sum(u["seconds"] for u in units)
    timing["overhead"] = compute_seconds - timing["units"] - sum(timing["saturation_k"].values()) - timing["post_processing"]
    projection = None
    if scope == "pilot":
        projection = project(units, timing, plan, {"C2 (R1 and R2), before R3": calibrated_plan("full"),
                                                   "after R3 (no uint16)": calibrated_plan("full", ("R3",))})
    mad_zero = sorted({(r["condition"], r["dtype"], r["arm"]) for r in diag_ref.to_dict("records") if r["mad_zero"]})
    manifest = dict(schema=CALIBRATED_SCHEMA, issue=issue, design="calibrated", scope=scope,
        specification="docs/preprocessing-algorithms.md, Evaluation design amendment for the calibrated rerun "
                      "(Accepted, W-243), with the approved resolutions C1-C8",
        qualification="development evidence on calibrated synthetic presets; not scientific acceptance; no "
                      "recommended defaults; the low-benefit rule and its 0.02 threshold are provisional",
        software=dict(**_revision(), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      command=sys.argv),
        parameters=dict(threshold_modes={m: dict(grid=list(s["grid"]), fixed=list(s["fixed"]),
                                                 verify_value=s["fixed"][0]) for m, s in THRESHOLD_MODES.items()},
            development_seeds=list(DEV_SEEDS), evaluation_seeds=list(EVAL_SEEDS), seeds_run=list(seeds),
            detection=dict(asdict(LocalMaximaConfig()), threshold_mode="per mode", threshold_value="swept"),
            extraction=asdict(EXTRACTION), decoding=asdict(DECODING), filtering=asdict(FILTERING),
            registration_coupled=[dict(method=s.config.method, config=asdict(s.config)) for s in REGISTRATION],
            matching=dict(function="starfinder.evaluation.match_points", reference="truth", observed="detections",
                          truth="reference-round amplicons that emit and whose centre is in bounds", **MATCHING),
            operating_point="per mode, condition, dtype and arm (multi-FOV: set, F1 pooled over FOVs): highest mean "
                            "F1 over development seeds in float64, seeds ascending; exact ties to the smallest value; "
                            "a selection at either grid end is marked at_grid_edge; the grid is not extended",
            endpoints="per held-out seed and mode: max-F1 over the grid; correct-decode fraction at the "
                      "development-selected value; both also at the fixed points (fixed1, fixed2)",
            precondition=dict(rule="g_E = mean_s[E(none, clean, s) - E(none, c, s)] > R_E = max of both seed ranges, "
                                   "strictly; valid when some endpoint degrades; undefined endpoints do not degrade",
                              rounding_decimals=RULE_DECIMALS, not_applied_to=list(NO_PRECONDITION),
                              multi_fov_checks=["fixture", "set_specific"], multi_fov_gates=list(MULTI_FOV_GATES),
                              set_specific_fovs={k: list(v) for k, v in SET_SPECIFIC_FOVS.items()}),
            low_benefit=dict(rule="benefit on c in T*: some E with delta >= max(R, 0.02) and the other E' with delta' "
                                  ">= -R'; clean holds: both deltas >= -R; not_flagged when some c shows a benefit "
                                  "and clean holds; not_assessable when T* is empty; low_benefit otherwise",
                             threshold_points=LOW_BENEFIT_POINTS, provisional=True, rounding_decimals=RULE_DECIMALS),
            harm_test="histogram matching on clean_unbalanced: harm when some endpoint has delta < -R; reported "
                      "beside the flag, not in it (C7)",
            background_radius=dict(rule="ceil(3 sigma) + 1 from the median widths, r_z capped at 3 (C4)",
                                   r_z_cap=RZ_CAP),
            image_statistics=dict(tool="benchmarks/image_statistics.py (W-238)", selection="adaptive",
                                  seeds="development seeds run, pooled per condition (multi-FOV: per set)",
                                  targets="W-238 development min-max ranges, docs/image-statistics.md (Target "
                                          "ranges); uint16 intensity targets x 16 (unverified)")),
        plan=dict(approved_matrix=_plan_record(calibrated_plan("full")),
                  approved_after_R3=_plan_record(calibrated_plan("full", ("R3",))), run=_plan_record(plan),
                  uint16_rule="uint16 only for clean, combined and bright_outliers (C2)",
                  multi_fov_rule="multi-FOV sets in uint8 only (C2 equals R1 and R2)"),
        conditions=condition_definitions(k_values), saturation_k=k_values,
        multi_fov={name: dict(spec, roles=FOV_ROLES[name], scene_key=f"{CALIBRATED_VERSION}/multi-fov/<FOV_id>")
                   for name, spec in MULTI_FOV.items()},
        recipes={arm: recipe_record(recipe) for arm, recipe in arms((RZ_CAP, 5, 5)).items()},
        multi_fov_recipes={arm: dict(recipe=r, fit=f, record=recipe_record(arms((RZ_CAP, 5, 5), f,
                                                                               Path("supplied.json"))[r]))
                           for arm, (r, f) in MULTI_FOV_ARMS.items()},
        comparisons=[dict(method=m, comparison=c, before=b, after=a, targeted=list(t),
                          reported=[x for x, _ in calibrated_extra_roles(m, c)])
                     for m, c, b, a, t in CALIBRATED_COMPARISONS + SAMPLE_COMPARISONS],
        scenes=scenes_record, multi_fov_scenes=mf_scenes, runs=runs_record,
        mad_zero_reference_round=[dict(condition=c, dtype=d, arm=a) for c, d, a in mad_zero],
        threshold_subset_verification=verification,
        preconditions_summary=dict(checks=len(preconditions), failed=len(gaps)),
        flags_summary=(flags[flags.level == "method"].result.value_counts().to_dict() if len(flags) else {}),
        reductions=list(reductions), projection=projection, timing_seconds=timing, units=units,
        compute_seconds=compute_seconds)
    manifest["files"] = [dict(path=k, bytes=v.stat().st_size, sha256=_digest(v)) for k, v in sorted(files.items())]
    manifest["artifact_bytes"] = sum(f["bytes"] for f in manifest["files"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1, default=_json_default) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="run the comparison matrix")
    runner.add_argument("--output", type=Path, required=True, help="new or empty directory outside Git")
    runner.add_argument("--design", choices=("w233", "calibrated"), default="w233",
                        help="w233: the W-233 evaluation (default); calibrated: the accepted rerun amendment (W-239)")
    runner.add_argument("--scope", choices=("full", "pilot", "mini", "smoke"), default="full")
    runner.add_argument("--shape", type=int, nargs=3, default=(32, 64, 64), metavar=("Z", "Y", "X"))
    runner.add_argument("--count", type=int, default=80, help="amplicons per FOV (at most 80)")
    runner.add_argument("--multi-fov-dtypes", nargs="+", choices=DTYPES)
    runner.add_argument("--reduction", action="append", default=[],
                        help="w233: record an applied reduction; calibrated: apply R3 (no uint16) (repeatable)")
    runner.add_argument("--projection", action="append", default=[], help="record the pilot projection (repeatable)")
    runner.add_argument("--issue", default="W-239", help="issue recorded in a calibrated manifest")
    timer = sub.add_parser("attach-time", help="add a /usr/bin/time -v record to the manifest")
    timer.add_argument("--output", type=Path, required=True)
    timer.add_argument("--time-log", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run" and args.design == "calibrated":
        if args.scope == "mini" or args.multi_fov_dtypes or args.projection:
            parser.error("the calibrated design has the scopes full, pilot and smoke, its own dtype plan (C2) and "
                         "computes its projection")
        # Calibrated scenes are the preset's 8x64x64 with 80 amplicons; --shape and --count are not used.
        run_calibrated(args.output, scope=args.scope, reductions=args.reduction, issue=args.issue,
                       log=lambda message: print(message, flush=True))
    elif args.command == "run":
        if args.count > 80 or any(n > m for n, m in zip(args.shape, (32, 64, 64))):
            parser.error("scenes are bounded to 32x64x64 voxels and 80 amplicons per FOV")
        run(args.output, scope=args.scope, shape=tuple(args.shape), count=args.count,
            multi_fov_dtypes=args.multi_fov_dtypes, reductions=args.reduction, projection=args.projection,
            log=lambda message: print(message, flush=True))
    else:
        print(json.dumps(attach_time(args.output, args.time_log)["max_rss_kib"]))


if __name__ == "__main__":
    main()
