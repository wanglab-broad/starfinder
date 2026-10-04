"""Calibrated readout scenes of W-278 for the §2.8 background, score and duplicate checks (R10 to R12, R14).

The scenes, conditions, pipeline and matching follow W-278 (run directory
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-278/20261002T011718Z-0cfc7de7,
scripts/w278_lib.py): calibrated_scene_preset (8x64x64, four rounds and channels, uint8,
80 amplicons; dense: 160) under nine conditions, LocalMaximaConfig() on round 1,
NeighborhoodSumConfig((1, 2, 2)), both decoders at their defaults, and match_points
(greedy, 3 voxels, inclusive) against the amplicon centres with center_in_bounds in the
detection round. The duplicate population (R14) follows W-278 attribute and
duplicate_pairs: each round-1 candidate's source is the amplicon with the largest
noise-free contribution in its channel at the candidate voxel, if at least 2 analytic
noise sd. Generated scenes are cached per process.
"""
from dataclasses import replace
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd

from starfinder.barcode import CodebookAwareDecoderConfig, NeighborhoodSumConfig, WtaDecoderConfig, decode_barcodes
from starfinder.barcode import extract_intensities, score_reads
from starfinder.evaluation.matching import match_points
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult, find_spots
from starfinder.synthetic import NoiseConfig, ScalarDistribution, calibrated_scene_preset, generate_formed_scene

HELDOUT_SEEDS = (103, 104, 105)
DENSE_COUNT = 160
# The W-278 conditions: "cal" holds the first eight, "dense" is cal with 160 amplicons.
CAL_CONDITIONS = ("noise", "mixing", "weakening", "gain", "trend", "round_effect_only", "background_only",
                  "combined")
CONDITIONS = CAL_CONDITIONS + ("dense",)
BOX_RADIUS = (1, 2, 2)
EXTRACTION = NeighborhoodSumConfig(BOX_RADIUS)
DECODERS = {"wta": WtaDecoderConfig(), "codebook_aware": CodebookAwareDecoderConfig()}
DECODER_SCORE = {"wta": "wta_l2_nll", "codebook_aware": "probability_nll"}
MATCH = dict(policy="greedy", threshold=3.0, units="voxel", boundary="inclusive")
MAD_SCALE = 1.4826


def calibrated_config(condition, seed):
    """(codebook, FormedSceneConfig) of one W-278 condition (w278_lib.calibrated_config)."""
    named = {"noise": "clean", "mixing": "clean", "weakening": "clean", "dense": "clean"}.get(condition, condition)
    book, cfg = calibrated_scene_preset(named, seed=seed)
    if condition == "noise":
        cfg = replace(cfg, readout=replace(cfg.readout, gain_enabled=False, trend_enabled=False))
    elif condition == "mixing":
        cfg = replace(cfg, readout=replace(cfg.readout, mixing_enabled=True))
    elif condition == "weakening":
        cfg = replace(cfg, readout=replace(cfg.readout, weakening_enabled=True))
    elif condition == "dense":
        cfg = replace(cfg, count=DENSE_COUNT)
    return book, cfg


def _generate(book, cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_formed_scene(book, config=cfg)


@lru_cache(maxsize=None)
def scene(condition, seed):
    """(codebook, scene) of one condition and seed."""
    book, cfg = calibrated_config(condition, seed)
    return book, _generate(book, cfg)


@lru_cache(maxsize=None)
def twins(condition, seed):
    """The spot-free twin (same noise streams, zero amplitude) and the noise-free latent twin (float32)."""
    book, cfg = calibrated_config(condition, seed)
    empty = replace(cfg, brightness=ScalarDistribution("constant", (0.0,)))
    latent = replace(empty, noise=NoiseConfig(), dtype="float32", accumulation="float32")
    return _generate(book, empty), _generate(book, latent)


def loaded(image, metadata, channels):
    return ImageLoadResult(image, metadata, tuple(channels), ())


def spots_at(points, metadata, namespace, channels):
    """A SpotFindingResult of given ZYX points (no detection)."""
    frame = pd.DataFrame({"spot_id": pd.array([str(i) for i in range(len(points))], dtype="string"),
                          "z": points[:, 0], "y": points[:, 1], "x": points[:, 2]})
    return SpotFindingResult(frame, metadata, namespace, LocalMaximaConfig(), {"channel_labels": list(channels)})


def truth(the_scene, label):
    """Per-round truth table of the amplicons, in amplicon order."""
    rounds = the_scene.round_truth
    return rounds[rounds.round_label == label].set_index("amplicon_id").loc[list(the_scene.amplicon_ids)]


@lru_cache(maxsize=None)
def pipeline(condition, seed):
    """Detection on round 1, extraction with background, both decoders, scores and truth (W-278 run_pipeline).

    Returns (spots, intensities, {decoder: scored table}, truth gene per candidate or None).
    """
    book, the_scene = scene(condition, seed)
    labels = list(the_scene.round_labels)
    channels = book.channel_labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spots = find_spots(the_scene.rounds[labels[0]], config=LocalMaximaConfig(), metadata=the_scene.metadata,
                           spot_namespace=f"w294/{condition}/{seed}")
    rounds = {label: loaded(the_scene.rounds[label], the_scene.metadata, channels) for label in labels}
    intensities = extract_intensities(rounds, spots, config=EXTRACTION)
    first = truth(the_scene, labels[0])
    points = spots.spots[["z", "y", "x"]].to_numpy(float)
    matches = match_points(first[["z", "y", "x"]].to_numpy(float), points, reference_metadata=the_scene.metadata,
                           observed_metadata=the_scene.metadata,
                           eligible_reference=first.center_in_bounds.to_numpy(bool), **MATCH)
    genes = the_scene.formed.gene_id.astype(str).to_numpy()
    truth_gene = np.full(len(points), None, dtype=object)
    for i, j, _ in matches.details["matched_pairs"]:
        truth_gene[j] = genes[i]
    tables = {}
    for name, config in DECODERS.items():
        decoded = decode_barcodes(intensities, book, config=config)
        tables[name] = score_reads(decoded, intensities, reference=book).table
    return spots, intensities, tables, truth_gene


def ring_estimate(image, centers, outer=(1, 6, 6), inner=(1, 3, 3)):
    """W-278 local_estimate: ring median, 1.4826 x MAD (N, C) and ring voxels (N,), written out here."""
    shape = np.asarray(image.shape[:3])
    outer, inner = np.asarray(outer), np.asarray(inner)
    grids = np.meshgrid(*[np.arange(-o, o + 1) for o in outer], indexing="ij")
    full = ~np.logical_and.reduce([np.abs(g) <= i for g, i in zip(grids, inner)])
    n, c = len(centers), image.shape[3]
    med, mad = np.full((n, c), np.nan), np.full((n, c), np.nan)
    voxels = np.zeros(n, dtype=np.int64)
    for k, p in enumerate(centers):
        lo, hi = np.maximum(p - outer, 0), np.minimum(p + outer + 1, shape)
        block = image[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        o = lo - (p - outer)
        mask = full[o[0]:o[0] + hi[0] - lo[0], o[1]:o[1] + hi[1] - lo[1], o[2]:o[2] + hi[2] - lo[2]]
        values = block[mask].astype(np.float64)
        voxels[k] = len(values)
        if len(values):
            m = np.median(values, axis=0)
            med[k] = m
            mad[k] = MAD_SCALE * np.median(np.abs(values - m), axis=0)
    return med, mad, voxels


def w278_components(values, box_voxels, background, assigned):
    """The W-278 D1 score and components (scripts/w278_lib.py, components), for (N, R) assigned channels.

    Rows with an assigned channel of -1 are NaN. Returns (bgcorr_probability_nll__local_ring,
    ambiguity_max__local_ring, sbr_mean__local_ring).
    """
    n, c, r = values.shape
    ok = (assigned >= 0).all(axis=1)
    a = np.where(assigned >= 0, assigned, 0)
    idx_n, idx_r = np.arange(n)[:, None], np.arange(r)[None, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vt = values - box_voxels[:, None, :] * background
        va = vt[idx_n, a, idx_r]
        ba = (box_voxels[:, None, :] * background)[idx_n, a, idx_r]
        sbr = (va / ba).mean(axis=1)
        other = vt.copy()
        other[idx_n, a, idx_r] = -np.inf
        ambiguity = (np.maximum(other.max(axis=1), 0) / np.maximum(va, 0)).max(axis=1)
        p = np.maximum(vt, 0) + 1e-6
        p = p / p.sum(axis=1, keepdims=True)
        nll = -np.log(np.maximum(p[idx_n, a, idx_r], 1e-12)).sum(axis=1)
    return tuple(np.where(ok, x, np.nan) for x in (nll, ambiguity, sbr))


def assigned_channels(table, codebook, rounds):
    """(N, R) assigned channel indices of the decoded color sequences; -1 for reads that are not assigned."""
    a = np.full((len(table), rounds), -1, dtype=np.int64)
    for i, (status, seq) in enumerate(zip(table.call_status, table.decoded_color_sequence)):
        if status == "assigned":
            a[i] = [codebook.color_to_channel[c] for c in seq]
    return a


# --- Duplicate population (W-278 scripts/w278_lib.py: attribute, duplicate_pairs) --------------

DUP_POOL_DISTANCE = 5.0
DUP_ATTRIBUTION_SIGMA = 2.0


def noise_parameters(the_scene):
    """W-278 noise_parameters: the analytic noise model of a scene."""
    effective = the_scene.provenance["effective_config"]
    noise = effective["noise"]
    return dict(alpha=float(noise["alpha"]), sigma=float(noise["sigma"]),
                correlated_sigma=float(noise["correlated_sigma"]),
                quantization_variance=1.0 / 12.0 if effective["dtype"].startswith("uint") else 0.0)


def _kernel_value(delta, sz, sl, e, theta):
    c, s = np.cos(theta), np.sin(theta)
    dz, dy, dx = delta[..., 0], delta[..., 1], delta[..., 2]
    radius = (dz / sz) ** 2 + ((c * dy + s * dx) / (sl * e)) ** 2 + ((-s * dy + c * dx) / sl) ** 2
    return np.where(radius <= 16, np.exp(-.5 * radius), 0.0)


def attribute(the_scene, latent_round, points, channels):
    """W-278 attribute: the source amplicon of each round-1 candidate, or None when unattributed."""
    label = the_scene.round_labels[0]
    position = truth(the_scene, label)[["z", "y", "x"]].to_numpy(float)
    formed = the_scene.formed.set_index("amplicon_id").loc[list(the_scene.amplicon_ids)]
    if len(points) == 0 or len(position) == 0:
        return np.full(len(points), None, dtype=object)
    kernel = _kernel_value(points[:, None, :] - position[None, :, :], formed.sz.to_numpy()[None],
                           formed.sl.to_numpy()[None], formed.e.to_numpy()[None], formed.theta.to_numpy()[None])
    contribution = the_scene.realized[:, :, 0].T[channels] * kernel
    best = contribution.argmax(axis=1)
    value = contribution[np.arange(len(points)), best]
    centers = np.clip(np.floor(points + 0.5).astype(np.int64), 0, np.asarray(latent_round.shape[:3]) - 1)
    background = latent_round[centers[:, 0], centers[:, 1], centers[:, 2], channels].astype(float)
    params = noise_parameters(the_scene)
    floor = DUP_ATTRIBUTION_SIGMA * np.sqrt(params["alpha"] * np.maximum(background, 0) + params["sigma"] ** 2
                                            + params["correlated_sigma"] ** 2 + params["quantization_variance"])
    return np.array([the_scene.amplicon_ids[b] if v >= f and v > 0 else None
                     for b, v, f in zip(best, value, floor)], dtype=object)


@lru_cache(maxsize=None)
def duplicate_population(condition, seed):
    """(sources, pairs) of the round-1 candidates of one scene (W-278 duplicate_pairs).

    pairs holds every cross-channel candidate pair (i, j) within DUP_POOL_DISTANCE voxels and
    every cross-channel pair of one source at any distance, with its distance and pair_class
    (true_duplicate, distinct or involves_unattributed).
    """
    _, the_scene = scene(condition, seed)
    spots, _, _, _ = pipeline(condition, seed)
    points = spots.spots[["z", "y", "x"]].to_numpy(float)
    channels = spots.spots.channel.to_numpy()
    source = attribute(the_scene, twins(condition, seed)[1].rounds[the_scene.round_labels[0]], points, channels)
    rows = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if channels[i] == channels[j]:
                continue
            distance = float(np.linalg.norm(points[i] - points[j]))
            if source[i] is None or source[j] is None:
                kind = "involves_unattributed"
            else:
                kind = "true_duplicate" if source[i] == source[j] else "distinct"
            if distance > DUP_POOL_DISTANCE and kind != "true_duplicate":
                continue
            rows.append(dict(i=i, j=j, distance=distance, pair_class=kind))
    return source, pd.DataFrame(rows, columns=["i", "j", "distance", "pair_class"])

