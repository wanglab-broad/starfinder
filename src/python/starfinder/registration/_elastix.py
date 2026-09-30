"""Rigid, affine and B-spline estimation with elastix (itk-elastix) in physical space.

itk-elastix is imported only when one of these methods runs. Z=1 inputs are
estimated as 2D images. The shared numerical rules (pyramid, sampling, seed,
diagnostics and failures) are those of the registration algorithm page.
"""

import re
import tempfile
from pathlib import Path

import numpy as np

from starfinder._registry import _version

from ._errors import RegistrationBackendUnavailableError, RegistrationEstimationError

_EXTRA = "registration-elastix"
_METRICS = {"mattes": "AdvancedMattesMutualInformation", "ncc": "AdvancedNormalizedCorrelation"}
_UNKNOWN_SPACING = "spacing_zyx is unknown; estimated with unit spacing (spacing_source='unknown_unit')"


def _import_itk():
    """itk with the elastix wrappers loaded (itk loads its modules on first attribute access)."""
    try:
        import itk

        itk.ElastixRegistrationMethod
    except (ImportError, AttributeError) as exc:
        raise RegistrationBackendUnavailableError(
            f"itk-elastix is required for rigid, affine and bspline registration; "
            f"install the '{_EXTRA}' extra (starfinder[{_EXTRA}])") from exc
    return itk


def _levels(shape_zyx):
    """Largest L <= 4 with min(Y, X) / 2^(L-1) >= 16, at least 1."""
    side, levels = min(shape_zyx[1:]), 1
    while levels < 4 and side / 2 ** levels >= 16:
        levels += 1
    return levels


def _schedule(levels, shape_zyx, dim):
    """Shrink factors per level in XYZ order: 2^(L-1-l) on every axis, Z capped at max(1, Z // 4)."""
    schedule = []
    for level in range(levels):
        factor = 2 ** (levels - 1 - level)
        schedule += [factor] * 2 + ([min(factor, max(1, shape_zyx[0] // 4))] if dim == 3 else [])
    return schedule


def _parameter_object(itk, config, shape_zyx, spacing_zyx, dim):
    """elastix's default parameter map of the method with the matched settings of the algorithm page."""
    levels = config.levels or _levels(shape_zyx)
    schedule = [str(f) for f in _schedule(levels, shape_zyx, dim)]
    settings = dict(
        Registration=["MultiResolutionRegistration"], Metric=[_METRICS[config.metric]],
        FixedImagePyramid=["FixedRecursiveImagePyramid"], MovingImagePyramid=["MovingRecursiveImagePyramid"],
        NumberOfResolutions=[str(levels)], FixedImagePyramidSchedule=schedule, MovingImagePyramidSchedule=schedule,
        MaximumNumberOfIterations=[str(config.iterations)], NumberOfSpatialSamples=[str(config.samples)],
        RandomSeed=[str(config.random_seed)], WriteResultImage=["false"])
    if config.metric == "mattes":
        settings["NumberOfHistogramBins"] = [str(getattr(config, "histogram_bins", 32))]
    if config.method == "bspline":
        grid = config.grid_spacing_physical
        if grid is None:
            grid = shape_zyx[2] * spacing_zyx[2] / 8
        settings.update(FinalGridSpacingInPhysicalUnits=[repr(float(grid))], GridSpacingSchedule=["1"] * levels)
    else:
        settings.update(AutomaticTransformInitialization=["true"],
                        AutomaticTransformInitializationMethod=["GeometricalCenter"])
    parameter_object = itk.ParameterObject.New()
    parameter_map = parameter_object.GetDefaultParameterMap(config.method)
    for key in ("Metric0Weight", "Metric1Weight"):
        if key in parameter_map:
            del parameter_map[key]
    for key, value in settings.items():
        parameter_map[key] = value
    parameter_object.AddParameterMap(parameter_map)
    return parameter_object, {key: " ".join(value) for key, value in sorted(parameter_map.items())}


def _register(itk, fixed, moving, parameter_object, directory):
    """Run elastix once, logging to directory; returns the finished ElastixRegistrationMethod."""
    registration = itk.ElastixRegistrationMethod.New(fixed, moving)
    registration.SetParameterObject(parameter_object)
    registration.SetLogToConsole(False)
    registration.SetLogToFile(True)
    registration.SetOutputDirectory(str(directory))
    registration.UpdateLargestPossibleRegion()
    return registration


def _parse_log(text):
    """Iterations, final metric value and stop condition per level from an elastix log."""
    iterations, metrics, stops, rows = [], [], [], None
    for line in text.splitlines():
        if line.startswith("1:ItNr"):
            rows = []
        elif rows is not None and re.match(r"\d+\t", line):
            rows.append(line)
        elif rows is not None and line.startswith("Time spent in resolution"):
            iterations.append(len(rows))
            metrics.append(float(rows[-1].split("\t")[1]) if rows else None)
            rows = None
        match = re.match(r"Stopping condition:\s*(.+)", line)
        if match:
            stops.append(match.group(1).strip())
    return tuple(iterations), tuple(metrics), tuple(stops)


def _values(parameters):
    return np.array([parameters.GetElement(i) for i in range(parameters.Size())], dtype=np.float64)


def _index_matrix(matrix_xyz, center_xyz, translation_xyz, spacing_zyx):
    """4x4 ZYX index matrix of the physical pull map x -> M (x - c) + c + t (origin 0, identity direction).

    A = S^-1 P M P S and b = S^-1 P (t + c - M c), with S = diag(spacing_zyx)
    and P the ZYX/XYZ reversal. A 2D (YX) map is embedded with Z row and
    column (1, 0, 0) and b_z = 0.
    """
    m = np.asarray(matrix_xyz, dtype=np.float64)
    c, t = (np.asarray(v, dtype=np.float64) for v in (center_xyz, translation_xyz))
    dim = len(c)
    s = np.asarray(spacing_zyx, dtype=np.float64)[3 - dim:]
    out = np.eye(4)
    out[3 - dim:3, 3 - dim:3] = m[::-1, ::-1] * s[None, :] / s[:, None]
    out[3 - dim:3, 3] = (t + c - m @ c)[::-1] / s
    return out


def _parameter_names(method, dim):
    axes = "xyz"[:dim]
    translations = tuple(f"translation_{a}" for a in axes)
    if method == "rigid":
        return (("angle",) if dim == 2 else tuple(f"angle_{a}" for a in axes)) + translations
    return tuple(f"matrix_{r}{c}" for r in axes for c in axes) + translations


def _estimate(reference, moving, config, geometry):
    """Registered estimator of RigidConfig, AffineConfig and BSplineConfig."""
    from ._config import WarpConfig
    from ._types import AffineTransform, BSplineTransform

    if np.ptp(reference) == 0 or np.ptp(moving) == 0:
        raise RegistrationEstimationError("constant registration signal")
    itk = _import_itk()
    spacing = geometry["reference_metadata"].spacing_zyx
    source, warnings = "metadata", ()
    if spacing is None:
        spacing, source, warnings = (1.0, 1.0, 1.0), "unknown_unit", (_UNKNOWN_SPACING,)
    shape = reference.shape
    dim = 2 if shape[0] == 1 else 3

    def image(array):
        result = itk.GetImageFromArray(np.ascontiguousarray(array[0] if dim == 2 else array, dtype=np.float32))
        result.SetSpacing([float(v) for v in spacing[::-1][:dim]])
        return result

    parameter_object, parameters = _parameter_object(itk, config, shape, spacing, dim)
    with tempfile.TemporaryDirectory(prefix="starfinder-elastix-") as directory:
        registration = _register(itk, image(reference), image(moving), parameter_object, directory)
        text = "".join(path.read_text(errors="replace") for path in sorted(Path(directory).glob("*.log")))
    iterations, metrics, stops = _parse_log(text)
    transform = itk.down_cast(registration.ConvertToItkTransform(registration.GetNthTransform(0)))
    values = _values(transform.GetParameters())
    if not np.isfinite(values).all():
        raise RegistrationEstimationError(f"{config.method} estimation returned non-finite parameters")
    if config.method == "bspline":
        fixed = _values(transform.GetFixedParameters())
        size = tuple(int(n) for n in fixed[:dim])
        result = BSplineTransform(size, fixed[dim:2 * dim], fixed[2 * dim:3 * dim], fixed[3 * dim:],
                                  values.reshape((dim, *size[::-1])), spacing, **geometry)
    else:
        matrix = np.asarray(itk.array_from_matrix(transform.GetMatrix()), dtype=np.float64)
        center = np.array([transform.GetCenter()[i] for i in range(dim)], dtype=np.float64)
        translation = np.array([transform.GetTranslation()[i] for i in range(dim)], dtype=np.float64)
        physical = dict(transform=parameters["Transform"], matrix_xyz=matrix.tolist(), center_xyz=center.tolist(),
                        translation_xyz=translation.tolist(), parameters=values.tolist(),
                        parameter_names=list(_parameter_names(config.method, dim)),
                        spacing_zyx=[float(v) for v in spacing])
        result = AffineTransform(_index_matrix(matrix, center, translation, spacing), physical=physical, **geometry)
    details = dict(converged=None, iterations_completed=iterations, final_metric_value=metrics,
                   stop_condition=stops, warnings=warnings, spacing_source=source, backend_parameters=parameters,
                   backend_versions={name: _version(name) for name in ("itk-elastix", "itk")})
    return result, "elastix", WarpConfig(backend="scipy"), details


def estimate_rigid(reference, moving, config, geometry):
    """Registered estimator of RigidConfig: an Euler transform stored as AffineTransform."""
    return _estimate(reference, moving, config, geometry)


def estimate_affine(reference, moving, config, geometry):
    """Registered estimator of AffineConfig: an affine transform stored as AffineTransform."""
    return _estimate(reference, moving, config, geometry)


def estimate_bspline(reference, moving, config, geometry):
    """Registered estimator of BSplineConfig: a cubic B-spline stored as BSplineTransform."""
    return _estimate(reference, moving, config, geometry)
