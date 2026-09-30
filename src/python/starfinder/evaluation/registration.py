"""Pure registration metrics on supplied images, masks, landmarks, shifts and transforms."""
from dataclasses import asdict, is_dataclass
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity as _ssim
from ._result import _result, _geometry, _threshold
from .matching import match_points

__all__ = ["evaluate_translation", "structural_similarity", "normalized_cross_correlation",
           "evaluate_mask_overlap", "evaluate_landmark_alignment", "evaluate_registration",
           "registration_qc", "evaluate_displacement_field"]

# Routine QC: SSIM on Z maximum projections with a uniform 7x7 window.
_QC_WINDOW = 7


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


def _mask(mask, shape):
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != tuple(shape):
        raise ValueError(f"mask must be a Boolean array of shape {tuple(shape)}")
    return mask


def _masked_ncc(ref, observed, mask):
    a, b = ref[mask], observed[mask]
    if a.size < 2:
        return None, "fewer than two voxels in the mask"
    if a.min() == a.max() or b.min() == b.max():
        return None, "constant signal within the mask"
    a, b = a - a.mean(), b - b.mean()
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))), None


def normalized_cross_correlation(ref, observed, *, mask=None):
    """Centered NCC on supplied equal-shaped arrays; constants are undefined.

    With a Boolean mask of the arrays' shape, the Pearson correlation in
    float64 over the masked elements; undefined when fewer than two are
    masked or either side is constant there. Without it, all elements.
    """
    ref, observed = _images(ref, observed)
    if mask is not None:
        mask = _mask(mask, ref.shape)
        value, reason = _masked_ncc(ref, observed, mask)
        return _result({"ncc": value}, {"ncc": "dimensionless"}, {"total": ref.size, "masked": int(mask.sum())},
                       {"policy": "masked elements"}, reasons={"ncc": reason} if reason else None)
    value = None
    if ref.size:
        a, b = ref - ref.mean(), observed - observed.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator > 0:
            value = float(np.sum(a * b) / denominator)
    return _result({"ncc": value}, {"ncc": "dimensionless"}, {"total": ref.size},
                   {"policy": "all supplied elements"})


def structural_similarity(ref, observed, *, data_range, policy, win_size=None, slice_index=None, mask=None):
    """SSIM with required positive data range and explicit spatial reduction.

    Policies: volume (ZYX), mip (max over Z), slice (explicit Z index), or
    plane (already supplied YX). Small/empty domains are undefined. Default
    window is the largest odd size <=7 fitting the selected domain, at least 3.
    With a Boolean mask of the selected domain's shape (ZYX for volume, YX
    otherwise), the SSIM map is averaged over the masked positions instead
    of the domain cropped by half the window; an empty mask is undefined.
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
    if mask is not None:
        mask = _mask(mask, ref.shape)
        value, reason = None, "empty mask"
        if not 3 <= size <= min(ref.shape):
            reason = "window larger than the domain"
        elif mask.any():
            ssim_map = _ssim(ref, observed, data_range=data_range, win_size=size, full=True)[1]
            value, reason = float(ssim_map[mask].mean()), None
        return _result({"ssim": value}, {"ssim": "dimensionless"}, {"total": ref.size, "masked": int(mask.sum())},
                       {"policy": policy, "data_range": float(data_range), "win_size": size,
                        "slice_index": slice_index, "mask": "supplied"},
                       reasons={"ssim": reason} if reason else None)
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


def _pull_displacement(transform):
    """Pull displacement u (ZYX3, float64) of a translation, affine, B-spline, dense or chain-like transform."""
    if hasattr(transform, "displacement_zyx"):
        u = np.asarray(transform.displacement_zyx, dtype=np.float64)
        if u.shape == (3,):  # a translation: one displacement for every voxel
            return np.broadcast_to(u, (*tuple(transform.reference_shape_zyx), 3))
        return u
    if hasattr(transform, "dense"):
        return np.asarray(transform.dense().displacement_zyx, dtype=np.float64)
    if hasattr(transform, "pull_field"):
        return np.asarray(transform.pull_field().displacement_zyx, dtype=np.float64)
    raise TypeError(f"registration QC does not support {type(transform).__qualname__}")


def _valid_overlap(u, shape):
    """Reference voxels whose pull point lies inside the closed box [0, n-1] of the moving grid on every axis."""
    valid = np.ones(shape, dtype=bool)
    for axis, n in enumerate(shape):
        index = np.arange(n, dtype=np.float64).reshape([-1 if a == axis else 1 for a in range(3)])
        point = index + u[..., axis]
        valid &= (point >= 0) & (point <= n - 1)
    return valid


def _statistics(magnitude):
    return {"median": float(np.median(magnitude)), "p95": float(np.percentile(magnitude, 95)),
            "max": float(magnitude.max())}


def _fold_fraction(u):
    """Fraction of voxels with det(I + grad u) <= 0; central differences, zero along singleton axes."""
    jacobian = np.zeros(u.shape + (3,))
    for j, n in enumerate(u.shape[:3]):
        if n > 1:
            jacobian[..., j] = np.gradient(u, axis=j)
    jacobian += np.eye(3)
    return float(np.mean(np.linalg.det(jacobian) <= 0))


def _rotation_angle(physical):
    """Rotation angle in radians of a physical rigid (Euler) matrix, else None."""
    if not physical or physical.get("transform") != "EulerTransform":
        return None
    m = np.asarray(physical["matrix_xyz"], dtype=float)
    if len(m) == 2:
        return float(np.arctan2(m[1, 0], m[0, 0]))
    return float(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)))


def _transform_summary(transform, u):
    if isinstance(getattr(transform, "displacement_zyx", None), tuple):
        return {"kind": "translation", "displacement_zyx": [float(v) for v in transform.displacement_zyx]}
    if hasattr(transform, "matrix_zyx"):
        a, b = transform.matrix_zyx[:3, :3], transform.matrix_zyx[:3, 3]
        det = float(np.linalg.det(a))
        return {"kind": "affine", "matrix": a.tolist(), "offset": b.tolist(), "det": det,
                "reflection_or_collapse": det <= 0,
                "max_singular_value_a_minus_i": float(np.linalg.svd(a - np.eye(3), compute_uv=False)[0]),
                "rotation_angle": _rotation_angle(getattr(transform, "physical", None))}
    summary = {"kind": "bspline" if hasattr(transform, "coefficients") else "dense", "displacement_voxels": _statistics(np.linalg.norm(u, axis=-1)),
               "displacement_physical": None, "fold_fraction": _fold_fraction(u)}
    spacing = getattr(getattr(transform, "reference_metadata", None), "spacing_zyx", None)
    if spacing is not None:
        summary["displacement_physical"] = _statistics(np.linalg.norm(u * np.asarray(spacing, dtype=float), axis=-1))
    return summary


def _optimizer(diagnostics):
    if diagnostics is None:
        return None
    return {name: getattr(diagnostics, name, None) for name in (
        "method", "backend", "converged", "iterations_completed", "final_metric_value", "stop_condition",
        "elapsed_iterations", "final_rms_change")}


def registration_qc(reference, before, after, transform, *, config=None, diagnostics=None):
    """Routine QC of one registration step or chain on supplied ZYX signals.

    reference is the reference signal, before the moving signal at the start
    of the step and after that signal resampled by transform, all on the
    transform's reference grid. The valid overlap is the set of reference
    voxels whose pull point under transform lies inside the closed box
    [0, n-1] of the moving grid on every axis; coverage is its fraction.
    ncc_before and ncc_after are Pearson correlations in float64 over the
    valid overlap (matched domains). ssim_before and ssim_after use a uniform
    7x7 window on the Z maximum projections, averaged over the valid columns
    (YX positions valid in every plane) eroded by 3 pixels, with data_range
    the maximum minus the minimum of the reference projection there. Every
    undefined value is None with a reason. details holds the transform
    summary (translation: the displacement; affine: A, b, det A with a
    reflection_or_collapse flag for det A <= 0, the largest singular value of
    A - I and, for rigid, the rotation angle in radians; B-spline and dense:
    displacement statistics and the fold fraction), the optimizer
    diagnostics (from a RegistrationDiagnostics, None when unknown) and, when config.projections is True, the three maximum
    projections. config is a starfinder.registration.RegistrationQcConfig
    (None: no projections); it is recorded, and nothing is rejected here.
    """
    if config is not None and (not is_dataclass(config) or isinstance(config, type)
                               or not isinstance(getattr(config, "projections", None), bool)):
        raise TypeError("config must be a RegistrationQcConfig")
    reference, before = _images(reference, before)
    after = _images(reference, after)[1]
    shape = tuple(getattr(transform, "reference_shape_zyx", ()))
    if reference.ndim != 3 or reference.shape != shape:
        raise ValueError(f"signals must be ZYX arrays on the transform's reference grid {shape}")
    u = _pull_displacement(transform)
    valid = _valid_overlap(u, shape)
    n_valid = int(valid.sum())
    values, reasons = {"coverage": n_valid / valid.size}, {}
    for label, image in (("before", before), ("after", after)):
        values["ncc_" + label], reason = _masked_ncc(reference, image, valid)
        if reason:
            reasons["ncc_" + label] = ("no valid overlap" if n_valid == 0 else
                                       "fewer than two valid voxels" if n_valid < 2 else
                                       "constant signal over the valid overlap")
    gain = None if values["ncc_before"] is None or values["ncc_after"] is None else \
        values["ncc_after"] - values["ncc_before"]
    values["ncc_gain"] = gain
    if gain is None:
        reasons["ncc_gain"] = "ncc_before or ncc_after is undefined"
    projections = {label: image.max(axis=0) for label, image in
                   (("reference", reference), ("before", before), ("after", after))}
    columns = valid.all(axis=0)
    data_range, eroded = None, np.zeros_like(columns)
    if min(shape[1:]) < _QC_WINDOW:
        reason = f"Y or X is smaller than the {_QC_WINDOW}x{_QC_WINDOW} window"
    else:
        eroded = binary_erosion(columns, structure=np.ones((_QC_WINDOW, _QC_WINDOW), dtype=bool), border_value=0)
        if not eroded.any():
            reason = "the eroded valid columns are empty"
        else:
            data_range = float(projections["reference"][eroded].max() - projections["reference"][eroded].min())
            reason = None if data_range > 0 else "the reference projection is constant on the valid columns"
    for label in ("before", "after"):
        values["ssim_" + label] = None
        if reason:
            reasons["ssim_" + label] = reason
        else:
            values["ssim_" + label] = structural_similarity(
                projections["reference"], projections[label], data_range=data_range, policy="plane",
                win_size=_QC_WINDOW, mask=eroded).values["ssim"]
    details = {"transform": _transform_summary(transform, u), "optimizer": _optimizer(diagnostics)}
    if config is not None and config.projections:
        details["projections"] = projections
    return _result(values, {"coverage": "fraction", "ncc_before": "dimensionless", "ncc_after": "dimensionless",
                            "ncc_gain": "dimensionless", "ssim_before": "dimensionless",
                            "ssim_after": "dimensionless"},
                   {"total": valid.size, "valid": n_valid, "valid_columns": int(columns.sum()),
                    "ssim_columns": int(eroded.sum())},
                   {"overlap": "pull point inside the closed box [0, n-1] on every axis",
                    "ssim_policy": "mip", "ssim_window": _QC_WINDOW, "ssim_erosion": (_QC_WINDOW - 1) // 2,
                    "data_range": data_range, "qc": None if config is None else asdict(config)},
                   reasons=reasons, details=details)


def _field(value, name):
    field = np.asarray(getattr(value, "displacement_zyx", value))
    if field.ndim != 4 or field.shape[-1] != 3 or field.dtype.kind not in "uif" or not np.isfinite(field).all():
        raise ValueError(f"{name} must be a finite ZYX3 pull displacement field")
    return field.astype(np.float64)


def evaluate_displacement_field(estimated, truth, *, mask, spacing_zyx=None):
    """Error of an estimated pull field against a supplied truth pull field.

    estimated and truth are (Z, Y, X, 3) ZYX displacement arrays in voxels
    (or transforms with displacement_zyx) on the same grid, for example
    forward_displacement of a synthetic pair. The median, 95th percentile
    (linear interpolation) and maximum of the error norm ``|u - u*|`` are taken over mask (the
    valid overlap; None uses every voxel), in voxels and, when spacing_zyx is
    given, in physical units of that spacing (component errors scaled per
    axis). An empty mask is undefined. Never computed from images alone.
    """
    estimated, truth = _field(estimated, "estimated"), _field(truth, "truth")
    if estimated.shape != truth.shape:
        raise ValueError("estimated and truth fields must have the same shape")
    mask = np.ones(truth.shape[:3], dtype=bool) if mask is None else _mask(mask, truth.shape[:3])
    error = (estimated - truth)[mask]
    scales = {"": np.ones(3)}
    if spacing_zyx is not None:
        spacing = np.asarray(spacing_zyx, dtype=float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("spacing_zyx must be three finite positive values")
        scales["_physical"] = spacing
    values, units, reasons = {}, {}, {}
    for suffix, scale in scales.items():
        statistics = _statistics(np.linalg.norm(error * scale, axis=-1)) if len(error) else None
        for key in ("median", "p95", "max"):
            name = f"{key}_error{suffix}"
            values[name] = None if statistics is None else statistics[key]
            units[name] = "physical" if suffix else "voxel"
            if statistics is None:
                reasons[name] = "empty mask"
    return _result(values, units, {"total": int(mask.size), "masked": int(mask.sum())},
                   {"percentile_method": "linear",
                    "spacing_zyx": None if spacing_zyx is None else [float(v) for v in spacing_zyx],
                    "error": "Euclidean norm of estimated minus truth pull displacement"}, reasons=reasons)
