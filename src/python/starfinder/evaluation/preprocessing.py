"""Preprocessing image metrics on supplied truth and detected spots.

This domain has no image-metric home in the existing matching, registration,
spot or readout modules. No benchmark orchestration is imported here.
"""
import numpy as np
from starfinder.readout import NeighborhoodSumConfig

__all__ = ["direct_metrics", "detected_spot_intensity_spread", "fov_variation"]

EXTRACTION = NeighborhoodSumConfig()
SNAPSHOT = "bg_corrected"

NORMALIZED = "normalized"

BACKGROUND_ARMS = ("recon", "tophat", "scalar", "bg3d", "r2_scalar", "r2_scalar_xsrc", "r2_3d", "r2_3d_xsrc")

_BG_STAGE = {"none": "input (no background step)",
             "pct": "before normalization, in input units (no background step); the stage of r2's bg_corrected"}


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



def direct_metrics(fov, raw, background, signal, truth, book, arm, recipe) -> dict[str, object]:
    """Background error, puncta contrast, intensity spread and clipping (M07–M13).

    Parameters
    ----------
    fov : FOV
        Processed images and named background/normalized snapshots.
    raw : mapping of str to numpy.ndarray
        Input ZYXC images, keyed by round label.
    background : mapping of str to numpy.ndarray
        Float background truth on each input grid.
    signal : mapping of str to numpy.ndarray
        Float signal truth on each input grid.
    truth : pandas.DataFrame
        Reference-round molecules with z, y, x and color_sequence columns.
    book : Codebook
        Ordered rounds and color-to-channel mapping for the images and truth.
    arm : str
        Preprocessing arm name selecting the background-estimation stage.
    recipe : PreprocessingRecipe or None
        Recipe associated with the arm; retained as part of the supplied context.

    Returns
    -------
    dict
        Legacy metric keys, stage/unit labels and output dtype. Background error
        is reference-round only; clipping is positive input becoming zero and
        saturation is unsigned output at its dtype maximum across all rounds.
        Intensity spread is population std/mean of round/channel medians of
        true-spot neighborhood sums. Undefined contrast or spread is None.
    """
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
    out["intensity_cv"] = _spot_intensity_spread(detection, truth, book, "color_sequence",
                                                  EXTRACTION.neighborhood_radius_zyx)
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


def _spot_intensity_spread(images, spots, book, sequence_column, radius):
    groups = {}
    for row in spots.itertuples():
        center = tuple(int(round(v)) for v in (row.z, row.y, row.x))
        for r, name in enumerate(book.round_labels):
            channel = book.color_to_channel[getattr(row, sequence_column)[r]]
            volume = images[name][..., channel]
            value = float(volume[_box(volume.shape, center, radius)].astype(np.float64).sum())
            groups.setdefault((name, channel), []).append(value)
    medians = np.array([np.median(v) for v in groups.values()])
    return float(np.std(medians) / np.mean(medians)) if len(medians) and np.mean(medians) > 0 else None


def detected_spot_intensity_spread(images, spots, book, *, sequence_column="observed_color_sequence",
                                   neighborhood_radius_zyx=EXTRACTION.neighborhood_radius_zyx) -> float | None:
    """Measure M15 using detected coordinates and observed on-channel sequences.

    Parameters
    ----------
    images : mapping of str to numpy.ndarray
        Processed ZYXC round images in book channel order.
    spots : pandas.DataFrame
        Detected spots with z, y, x and the chosen sequence column, one color
        per round. Supply the detection population, including rejected reads.
    book : Codebook
        Round labels and color-to-channel mapping for the supplied sequences.
    sequence_column : str
        Column holding the observed sequence, rather than a truth barcode.
    neighborhood_radius_zyx : tuple of int
        Inclusive ZYX half-widths of the neighborhood sum, clipped at edges.

    Returns
    -------
    float or None
        Population std/mean of the nonempty round/channel group medians,
        exactly the M09 formula on detected spots. None for no groups or a
        nonpositive mean. Coordinates use Python rounding as in M09.
    """
    return _spot_intensity_spread(images, spots, book, sequence_column, neighborhood_radius_zyx)


def fov_variation(values) -> float | None:
    """Measure between-FOV population std/mean (M16).

    Parameters
    ----------
    values : iterable of float or None
        One nonnegative metric value per FOV, such as M15, unmatched rate or
        assigned-read count. Call separately for edge and dense populations.

    Returns
    -------
    float or None
        Population std/mean, or None for an empty population, any undefined
        value or a zero mean. FOVs with undefined values are not dropped.
    """
    values = np.asarray(list(values), dtype=float)
    if values.ndim != 1 or np.isinf(values).any() or (values < 0).any():
        raise ValueError("values must be a one-dimensional nonnegative finite metric population or missing")
    if not len(values) or np.isnan(values).any() or np.mean(values) <= 0:
        return None
    return float(np.std(values) / np.mean(values))


