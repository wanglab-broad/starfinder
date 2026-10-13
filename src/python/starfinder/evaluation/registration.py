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


def _finite(a):
    """Whether every value of a is finite, checked one Z plane at a time for a volume."""
    return all(np.isfinite(p).all() for p in a) if a.ndim >= 3 else bool(np.isfinite(a).all())


def _images(ref, observed):
    return [a.astype(np.float64) for a in _signals(ref, observed)]


def _signals(ref, observed):
    """The two arrays, checked for one shape and finite real values; registration_qc reads them by plane."""
    arrays = [np.asarray(x) for x in (ref, observed)]
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("images must have the same shape")
    if any(a.dtype.kind not in "uif" or not _finite(a) for a in arrays):
        raise ValueError("images must contain finite real values")
    return arrays


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


def _pull_planes(transform):
    """z -> pull displacement u (YX3, float64) of reference plane z of a translation, affine, B-spline, dense or
    chain-like transform.

    A translation is one displacement broadcast over the plane, and a chain is
    evaluated one plane at a time with the arithmetic of its pull_field. An
    affine or B-spline transform is evaluated whole by dense(), as before.
    """
    if hasattr(transform, "displacement_zyx"):
        u = np.asarray(transform.displacement_zyx)
        if u.shape == (3,):  # a translation: one displacement for every voxel
            plane = np.broadcast_to(u.astype(np.float64), (*tuple(transform.reference_shape_zyx)[1:], 3))
            return lambda z: plane
        return lambda z: np.asarray(u[z], dtype=np.float64)
    if hasattr(transform, "dense"):
        u = np.asarray(transform.dense().displacement_zyx, dtype=np.float64)
        return lambda z: u[z]
    if hasattr(transform, "pull_points"):
        shape = tuple(transform.reference_shape_zyx)
        grid = np.stack(np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape[1:]), indexing="ij"))

        def plane(z):
            # TransformChain.pull_field, row z.
            points = transform.pull_points(z)
            u = np.empty((*shape[1:], 3), dtype=np.float64)
            u[..., 0] = points[0] - z
            u[..., 1:] = np.moveaxis(points[1:] - grid, 0, -1)
            return u
        return plane
    if hasattr(transform, "pull_field"):
        u = np.asarray(transform.pull_field().displacement_zyx, dtype=np.float64)
        return lambda z: u[z]
    raise TypeError(f"registration QC does not support {type(transform).__qualname__}")


def _valid_overlap(u, shape):
    """Reference voxels whose pull point lies inside the closed box [0, n-1] of the moving grid on every axis.

    u is the pull field (ZYX3) or a function that returns its plane z (YX3);
    the mask is filled one plane at a time.
    """
    planes = u if callable(u) else lambda z: u[z]
    index = [np.arange(n, dtype=np.float64).reshape([-1 if a == axis else 1 for a in range(2)])
             for axis, n in enumerate(shape[1:])]
    valid = np.empty(shape, dtype=bool)
    for z in range(shape[0]):
        plane = planes(z)
        point = np.float64(z) + plane[..., 0]
        valid[z] = (point >= 0) & (point <= shape[0] - 1)
        for axis, n in enumerate(shape[1:], start=1):
            point = index[axis - 1] + plane[..., axis]
            valid[z] &= (point >= 0) & (point <= n - 1)
    return valid


def _overlap_ncc(reference, images, valid):
    """The NCC of reference with each image over valid, as _masked_ncc computes it, one plane at a time.

    The sums of each plane's masked float64 voxels are added in Z order, so a
    volume of one plane gets _masked_ncc's value exactly and a larger one
    differs from it only by the order of the additions. None when fewer than
    two voxels are valid or either side is constant there.
    """
    n = int(valid.sum())
    if n < 2:
        return [None] * len(images)
    planes = [z for z in range(valid.shape[0]) if valid[z].any()]

    def masked(image, z):
        return np.asarray(image[z], dtype=np.float64)[valid[z]]

    signals = (reference, *images)
    sums, low, high = [0.0] * len(signals), [np.inf] * len(signals), [-np.inf] * len(signals)
    for z in planes:
        for k, image in enumerate(signals):
            a = masked(image, z)
            sums[k] += a.sum()
            low[k], high[k] = min(low[k], a.min()), max(high[k], a.max())
    means = [s / n for s in sums]
    products, squares = [0.0] * len(images), [0.0] * len(signals)
    for z in planes:
        a = masked(reference, z) - means[0]
        squares[0] += a.dot(a)
        for k, image in enumerate(images, start=1):
            b = masked(image, z) - means[k]
            products[k - 1] += np.sum(a * b)
            squares[k] += b.dot(b)
    return [None if low[0] == high[0] or low[k] == high[k] else
            float(products[k - 1] / (np.sqrt(squares[0]) * np.sqrt(squares[k]))) for k in range(1, len(signals))]


def _statistics(magnitude, overwrite=False):
    """Median, 95th percentile and maximum; overwrite lets the median and percentile partition magnitude in place."""
    return {"median": float(np.median(magnitude, overwrite_input=overwrite)),
            "p95": float(np.percentile(magnitude, 95, overwrite_input=overwrite)), "max": float(magnitude.max())}


def _field_summary(planes, shape, spacing):
    """Displacement statistics and fold fraction of the pull field, read one plane at a time.

    The fold fraction is the fraction of voxels with det(I + grad u) <= 0,
    by np.gradient's arithmetic: central differences, one-sided at the ends,
    zero along singleton axes; along Z from the neighbouring planes. Only
    the |u| of every voxel is held whole, for the median and percentile.
    """
    nz = shape[0]
    magnitude = np.empty(shape)
    folded = 0
    previous, u, following = None, planes(0), planes(1) if nz > 1 else None
    for z in range(nz):
        magnitude[z] = np.linalg.norm(u, axis=-1)
        jacobian = np.zeros(u.shape + (3,))
        if nz > 1:
            jacobian[..., 0] = ((following - previous) / 2.0 if 0 < z < nz - 1 else
                                following - u if z == 0 else u - previous)
        for j, n in enumerate(shape[1:], start=1):
            if n > 1:
                jacobian[..., j] = np.gradient(u, axis=j - 1)
        jacobian += np.eye(3)
        folded += int(np.count_nonzero(np.linalg.det(jacobian) <= 0))
        previous, u, following = u, following, planes(z + 2) if z + 2 < nz else None
    summary = {"displacement_voxels": _statistics(magnitude, overwrite=True), "displacement_physical": None,
               "fold_fraction": folded / magnitude.size}
    if spacing is not None:
        for z in range(nz):
            magnitude[z] = np.linalg.norm(planes(z) * np.asarray(spacing, dtype=float), axis=-1)
        summary["displacement_physical"] = _statistics(magnitude, overwrite=True)
    return summary


def _rotation_angle(physical):
    """Rotation angle in radians of a physical rigid (Euler) matrix, else None."""
    if not physical or physical.get("transform") != "EulerTransform":
        return None
    m = np.asarray(physical["matrix_xyz"], dtype=float)
    if len(m) == 2:
        return float(np.arctan2(m[1, 0], m[0, 0]))
    return float(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)))


def _transform_summary(transform, planes, shape):
    if isinstance(getattr(transform, "displacement_zyx", None), tuple):
        return {"kind": "translation", "displacement_zyx": [float(v) for v in transform.displacement_zyx]}
    if hasattr(transform, "matrix_zyx"):
        a, b = transform.matrix_zyx[:3, :3], transform.matrix_zyx[:3, 3]
        det = float(np.linalg.det(a))
        return {"kind": "affine", "matrix": a.tolist(), "offset": b.tolist(), "det": det,
                "reflection_or_collapse": det <= 0,
                "max_singular_value_a_minus_i": float(np.linalg.svd(a - np.eye(3), compute_uv=False)[0]),
                "rotation_angle": _rotation_angle(getattr(transform, "physical", None))}
    spacing = getattr(getattr(transform, "reference_metadata", None), "spacing_zyx", None)
    return {"kind": "bspline" if hasattr(transform, "coefficients") else "dense",
            **_field_summary(planes, shape, spacing)}


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
    # The signals are read one Z plane at a time and never copied whole (W-461).
    reference, before = _signals(reference, before)
    after = _signals(reference, after)[1]
    shape = tuple(getattr(transform, "reference_shape_zyx", ()))
    if reference.ndim != 3 or reference.shape != shape:
        raise ValueError(f"signals must be ZYX arrays on the transform's reference grid {shape}")
    planes = _pull_planes(transform)
    valid = _valid_overlap(planes, shape)
    n_valid = int(valid.sum())
    values, reasons = {"coverage": n_valid / valid.size}, {}
    for label, value in zip(("before", "after"), _overlap_ncc(reference, (before, after), valid)):
        values["ncc_" + label] = value
        if value is None:
            reasons["ncc_" + label] = ("no valid overlap" if n_valid == 0 else
                                       "fewer than two valid voxels" if n_valid < 2 else
                                       "constant signal over the valid overlap")
    gain = None if values["ncc_before"] is None or values["ncc_after"] is None else \
        values["ncc_after"] - values["ncc_before"]
    values["ncc_gain"] = gain
    if gain is None:
        reasons["ncc_gain"] = "ncc_before or ncc_after is undefined"
    # The maximum commutes with the exact, monotonic cast to float64.
    projections = {label: image.max(axis=0).astype(np.float64) for label, image in
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
    details = {"transform": _transform_summary(transform, planes, shape), "optimizer": _optimizer(diagnostics)}
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
