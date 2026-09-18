"""Pure registration metrics on supplied images, masks, landmarks and shifts."""
import numpy as np
from skimage.metrics import structural_similarity as _ssim
from ._result import _result, _geometry, _threshold
from .matching import match_points

__all__ = ["evaluate_translation", "structural_similarity", "normalized_cross_correlation",
           "evaluate_mask_overlap", "evaluate_landmark_alignment", "evaluate_registration"]


def evaluate_translation(detected, truth, *, reference_metadata, observed_metadata,
                         units, tolerance, eligible_rounds=None, failed_rounds=()):
    """Compare labeled ZYX displacements in the same convention and units.

    No sign inversion or integer rounding is performed. Explicitly exclude the
    reference round using eligible_rounds; absent estimates are missing, not
    silently treated as reference rounds. Passing uses strict per-axis tolerance.
    Partial comparisons retain observed errors but never produce a passing gate.
    tolerance=None explicitly requests errors without a pass/fail gate.
    """
    _geometry(reference_metadata, observed_metadata, units)
    if tolerance is not None:
        _threshold(tolerance)
    rounds = list(truth if eligible_rounds is None else eligible_rounds)
    if len(set(rounds)) != len(rounds) or not set(rounds) <= set(truth):
        raise ValueError("eligible rounds must be unique truth labels")
    failed = set(failed_rounds)
    if not failed <= set(rounds):
        raise ValueError("failed rounds must be eligible")
    details, reasons, errors, norms = {}, {}, [], []
    for label in rounds:
        gt = np.asarray(truth[label], dtype=float)
        if gt.shape != (3,) or not np.isfinite(gt).all():
            raise ValueError("truth displacements must be finite ZYX triples")
        if label in failed:
            reasons[label] = "upstream registration failed"
            continue
        if label not in detected or detected[label] is None:
            reasons[label] = "missing estimated displacement"
            continue
        det = np.asarray(detected[label], dtype=float)
        if det.shape != (3,) or not np.isfinite(det).all():
            raise ValueError("estimated displacements must be finite ZYX triples")
        error = np.abs(det - gt)
        errors.append(float(error.max()))
        norms.append(float(np.linalg.norm(error)))
        details[label] = {"gt": gt.tolist(), "detected": det.tolist(), "error": error.tolist(),
                          "max_axis_error": errors[-1], "error_l2": norms[-1]}
    maximum = max(errors) if errors else None
    if tolerance is None:
        reasons["passed"] = "tolerance gate not requested"
    return _result({"max_error": maximum, "mean_error_l2": float(np.mean(norms)) if norms else None,
                    "passed": maximum < tolerance if maximum is not None and not reasons and tolerance is not None else None},
                   {"max_error": units, "mean_error_l2": units, "passed": "boolean"},
                   {"total": len(truth), "eligible": len(rounds), "matched": len(details),
                    "failed": len(failed), "missing": len(rounds) - len(details) - len(failed)},
                   {"tolerance": tolerance, "units": units, "boundary": "exclusive",
                    "eligible_rounds": rounds, "frame_id": reference_metadata.frame_id,
                    "convention": "supplied displacements; no sign conversion"}, reasons=reasons,
                   details={"per_round": details},
                   status="failed" if failed else "missing" if any(k != "passed" for k in reasons) else None)


def _images(ref, observed):
    arrays = [np.asarray(x) for x in (ref, observed)]
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("images must have the same shape")
    if any(a.dtype.kind not in "uif" or not np.isfinite(a).all() for a in arrays):
        raise ValueError("images must contain finite real values")
    return [a.astype(np.float64) for a in arrays]


def normalized_cross_correlation(ref, observed):
    """Centered NCC on supplied equal-shaped arrays; constants are undefined."""
    ref, observed = _images(ref, observed)
    value = None
    if ref.size:
        a, b = ref - ref.mean(), observed - observed.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator > 0:
            value = float(np.sum(a * b) / denominator)
    return _result({"ncc": value}, {"ncc": "dimensionless"}, {"total": ref.size},
                   {"policy": "all supplied elements"})


def structural_similarity(ref, observed, *, data_range, policy, win_size=None, slice_index=None):
    """SSIM with required positive data range and explicit spatial reduction.

    Policies: volume (ZYX), mip (max over Z), slice (explicit Z index), or
    plane (already supplied YX). Small/empty domains are undefined. Default
    window is the largest odd size <=7 fitting the selected domain, at least 3.
    """
    ref, observed = _images(ref, observed)
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    if policy not in ("volume", "mip", "slice", "plane"):
        raise ValueError("unknown SSIM policy")
    if ref.ndim != (2 if policy == "plane" else 3):
        raise ValueError("SSIM policy does not match image dimensions")
    if policy == "slice":
        if not isinstance(slice_index, (int, np.integer)) or not 0 <= slice_index < ref.shape[0]:
            raise ValueError("slice requires an in-bounds integer slice_index")
        ref, observed = ref[slice_index], observed[slice_index]
    elif slice_index is not None:
        raise ValueError("slice_index requires slice policy")
    if policy == "mip" and ref.size:
        ref, observed = ref.max(axis=0), observed.max(axis=0)
    size = min(7, min(ref.shape)) if win_size is None else win_size
    if win_size is None and size % 2 == 0:
        size -= 1
    if win_size is not None and (not isinstance(size, (int, np.integer)) or size < 3 or size % 2 == 0):
        raise ValueError("win_size must be odd and >=3")
    value = float(_ssim(ref, observed, data_range=data_range, win_size=size)) if 3 <= size <= min(ref.shape) else None
    return _result({"ssim": value}, {"ssim": "dimensionless"}, {"total": ref.size},
                   {"policy": policy, "data_range": float(data_range), "win_size": size,
                    "slice_index": slice_index})


def evaluate_mask_overlap(ref, observed):
    """IoU and Dice from supplied Boolean masks; empty unions are undefined."""
    ref, observed = np.asarray(ref), np.asarray(observed)
    if ref.dtype != bool or observed.dtype != bool or ref.shape != observed.shape:
        raise ValueError("mask overlap requires equal-shaped Boolean masks")
    intersection = int((ref & observed).sum())
    union = int((ref | observed).sum())
    nr, no = int(ref.sum()), int(observed.sum())
    return _result({"iou": intersection / union if union else None,
                    "dice": 2 * intersection / (nr + no) if nr + no else None},
                   {"iou": "fraction", "dice": "fraction"},
                   {"total": ref.size, "reference": nr, "observed": no,
                    "matched": intersection, "union": union}, {"policy": "supplied Boolean masks"})


def evaluate_landmark_alignment(reference, observed, **matching_config):
    """Evaluate supplied landmarks using an explicit match_points policy."""
    return match_points(reference, observed, **matching_config)


def evaluate_registration(ref, before, after, *, reference_spots, before_spots, after_spots,
                          reference_mask, before_mask, after_mask, reference_metadata,
                          before_metadata, after_metadata, data_range, ssim_policy,
                          matching_policy, match_threshold, units, boundary="inclusive",
                          win_size=None, slice_index=None):
    """Combine before/after metrics; never detect, threshold, register or write.

    Images share a grid. Supplied masks/points describe the caller's explicitly
    chosen measurement domain (e.g. projected detections); config records SSIM
    and matching policies. NCC always measures the full supplied arrays.
    """
    values, metric_units, counts, reasons, configs = {}, {}, {}, {}, {}
    for label, image, points, mask, metadata in (
        ("before", before, before_spots, before_mask, before_metadata),
        ("after", after, after_spots, after_mask, after_metadata)):
        _geometry(reference_metadata, metadata, units)
        components = {
            "image": normalized_cross_correlation(ref, image),
            "structure": structural_similarity(ref, image, data_range=data_range, policy=ssim_policy,
                                                 win_size=win_size, slice_index=slice_index),
            "spot": evaluate_mask_overlap(reference_mask, mask),
            "landmark": match_points(reference_spots, points, policy=matching_policy,
                                      threshold=match_threshold, units=units, boundary=boundary,
                                      reference_metadata=reference_metadata, observed_metadata=metadata)}
        for component, result in components.items():
            for key, value in result.values.items():
                name = {"iou": "spot_iou", "dice": "spot_dice", "recall": "match_rate",
                        "mean_distance": "match_distance"}.get(key, key) + "_" + label
                values[name] = value
                metric_units[name] = result.units[key]
                if key in result.reasons:
                    reasons[name] = result.reasons[key]
            counts.update({f"{component}_{k}_{label}": v for k, v in result.counts.items()})
            configs[f"{component}_{label}"] = result.config
    counts.update(n_spots_ref=len(reference_spots), n_spots_before=len(before_spots), n_spots_after=len(after_spots))
    return _result(values, metric_units, counts, configs, reasons=reasons)
