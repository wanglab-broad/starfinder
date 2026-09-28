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

    uv run python ../../benchmarks/preprocessing_synthetic.py run --output <run dir>/evaluation
    uv run python ../../benchmarks/preprocessing_synthetic.py attach-time --output <dir> --time-log <file>
"""
import argparse
from dataclasses import asdict, replace
from functools import cached_property
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

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
from starfinder.synthetic import (BackgroundConfig, NoiseConfig, ScalarDistribution, development_scene_preset,
    generate_formed_scene)

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


def _pipeline(fov, threshold):
    fov.run(PipelineConfig(detection=LocalMaximaConfig(threshold_value=threshold), extraction=EXTRACTION,
                           decoding=DECODING, filtering=FILTERING))
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


def _key(spots, reads):
    return sorted(zip(spots.z, spots.y, spots.x, spots.channel, reads.gene_id.astype(str), reads.accepted,
                      reads.observed_color_sequence.astype(str)))


def sweep(fov, truth, verify=True):
    """Metrics at every threshold of the grid; one row per threshold.

    Detection, extraction, decoding and filtering run once through FOV.run at
    the lowest threshold. With min_distance_voxels=1, local maxima at a higher
    threshold_value are exactly those whose peak intensity exceeds that
    channel's higher noise cutoff, and extraction, WTA decoding and read
    filtering act on each spot independently, so every other threshold is the
    corresponding subset. With verify, FOV.run is repeated at threshold_value=5
    and must give the same spots and reads as the subset (else RuntimeError).
    """
    ref = fov.rounds.reference_round
    metadata = fov.metadata[ref]
    points = truth[["z", "y", "x"]].to_numpy(float)
    all_spots, all_reads, lowest = _pipeline(fov, THRESHOLDS[0])
    image = fov.images[ref]
    assert np.allclose(lowest, _noise_thresholds(image, THRESHOLDS[0]), rtol=0, atol=0)
    rows, verified = [], None
    for threshold in THRESHOLDS:
        cutoff = np.asarray(_noise_thresholds(image, threshold))
        keep = all_spots.peak_intensity.to_numpy() > cutoff[all_spots.channel.to_numpy()]
        spots, reads = all_spots[keep].reset_index(drop=True), all_reads[keep].reset_index(drop=True)
        if verify and threshold == DEFAULT_THRESHOLD:
            direct_spots, direct_reads, direct_cutoff = _pipeline(fov, threshold)
            if _key(direct_spots, direct_reads) != _key(spots, reads) or direct_cutoff != cutoff.tolist():
                raise RuntimeError("threshold subset differs from a direct run at threshold_value=5")
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
            verified_direct_run=threshold == DEFAULT_THRESHOLD and verified is not None))
    return rows


# --- Direct metrics ---------------------------------------------------------------

def _box(shape, center, radius):
    return tuple(slice(max(0, c - r), min(n, c + r + 1)) for c, r, n in zip(center, radius, shape))


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
    if arm == "none":
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
                   bg_truth_mean=float(np.mean(truth_background)), bg_units=units)
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


def comparisons_table(ends, direct, diagnostics, specs, harm, combined):
    """One row per comparison, condition role and dtype, with before/after/delta mean and range.

    Means and ranges are across held-out (evaluation) seeds. Direct metrics and
    MAD diagnostics are first averaged within each seed (over channels, and
    over FOVs for multi-FOV sets); mad_zero becomes the fraction of channels
    with MAD 0.
    """
    rows = []
    for method, mode, before, after, targeted in specs:
        roles = [(c, "targeted") for c in targeted] + [(harm, "harm_check"), (combined, "combined")]
        for condition, role in roles:
            for dtype in DTYPES:
                row = dict(method=method, comparison=mode, before=before, after=after, condition=condition,
                           role=role, dtype=dtype, seeds="evaluation " + ",".join(map(str, EVAL_SEEDS)))
                sel = lambda t, arm: t[(t.condition == condition) & (t.dtype == dtype) & (t.arm == arm)  # noqa: E731
                                       & t.seed.isin(EVAL_SEEDS)].sort_values("seed")
                b, a = sel(ends, before), sel(ends, after)
                if not len(b) or not len(a):
                    continue
                for metric in ENDPOINTS:
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
            for record_index, record in mf_cmp.iterrows():
                for label, arm in (("before", record["before"]), ("after", record["after"])):
                    part = spread_direct[(spread_direct["condition"] == record["condition"])
                                         & (spread_direct["dtype"] == record["dtype"])
                                         & (spread_direct["arm"] == arm) & spread_direct["seed"].isin(EVAL_SEEDS)]
                    mean, low, high = _stats(part.per_fov_correct_fraction_range)
                    mf_cmp.loc[record_index, f"per_fov_correct_fraction_range_{label}_mean"] = mean
                    mf_cmp.loc[record_index, f"per_fov_correct_fraction_range_{label}_min"] = low
                    mf_cmp.loc[record_index, f"per_fov_correct_fraction_range_{label}_max"] = high
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


def multi_fov_tables(mf, selected):
    """Per-FOV decoding accuracy and false-positive rate at the pooled dev-selected threshold and at 5."""
    rows = []
    for key, group in mf[mf.seed.isin(EVAL_SEEDS)].groupby(["multi_fov", "dtype", "arm", "seed", "fov_id", "role"],
                                                           sort=False):
        name, dtype, arm, seed, fov_id, role = key
        for label, threshold in (("dev_max_f1", selected[(name, dtype, arm)]), ("default", DEFAULT_THRESHOLD)):
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


def multi_fov_spread(per_fov):
    """Spread across FOVs (range and standard deviation), then mean and range across evaluation seeds."""
    rows = []
    for key, group in per_fov.groupby(["multi_fov", "dtype", "arm", "operating_point"], sort=False):
        record = dict(zip(["multi_fov", "dtype", "arm", "operating_point"], key))
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
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest["time_v"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="run the comparison matrix")
    runner.add_argument("--output", type=Path, required=True, help="new or empty directory outside Git")
    runner.add_argument("--scope", choices=("full", "pilot", "mini", "smoke"), default="full")
    runner.add_argument("--shape", type=int, nargs=3, default=(32, 64, 64), metavar=("Z", "Y", "X"))
    runner.add_argument("--count", type=int, default=80, help="amplicons per FOV (at most 80)")
    runner.add_argument("--multi-fov-dtypes", nargs="+", choices=DTYPES)
    runner.add_argument("--reduction", action="append", default=[], help="record an applied reduction (repeatable)")
    runner.add_argument("--projection", action="append", default=[], help="record the pilot projection (repeatable)")
    timer = sub.add_parser("attach-time", help="add a /usr/bin/time -v record to the manifest")
    timer.add_argument("--output", type=Path, required=True)
    timer.add_argument("--time-log", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.count > 80 or any(n > m for n, m in zip(args.shape, (32, 64, 64))):
            parser.error("scenes are bounded to 32x64x64 voxels and 80 amplicons per FOV")
        run(args.output, scope=args.scope, shape=tuple(args.shape), count=args.count,
            multi_fov_dtypes=args.multi_fov_dtypes, reductions=args.reduction, projection=args.projection,
            log=lambda message: print(message, flush=True))
    else:
        print(json.dumps(attach_time(args.output, args.time_log)["max_rss_kib"]))


if __name__ == "__main__":
    main()
