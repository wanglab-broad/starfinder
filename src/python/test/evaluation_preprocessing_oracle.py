"""Pre-W-770 algorithms from d845939, retained only for same-environment parity.

The sweep's orchestration helpers are supplied by the test; its read loop,
direct metrics and AUPRC below are verbatim pre-move implementations.
"""
import math
import json
import numpy as np
import pandas as pd
from starfinder.readout import NeighborhoodSumConfig
from starfinder.evaluation.matching import match_points

THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)

DEFAULT_THRESHOLD = 5.0

MATCHING = dict(policy="greedy", threshold=2.0, units="voxel", boundary="inclusive")

SNAPSHOT = "bg_corrected"

NORMALIZED = "normalized"

BACKGROUND_ARMS = ("recon", "tophat", "scalar", "bg3d", "r2_scalar", "r2_scalar_xsrc", "r2_3d", "r2_3d_xsrc")

_BG_STAGE = {"none": "input (no background step)",
             "pct": "before normalization, in input units (no background step); the stage of r2's bg_corrected"}

EXTRACTION = NeighborhoodSumConfig()

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


