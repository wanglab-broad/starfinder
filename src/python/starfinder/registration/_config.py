"""Frozen algorithm configurations; class identity selects the estimator."""

import math
from dataclasses import dataclass, field

from ._errors import InvalidRegistrationConfigError


def _number(value, name, *, minimum=0, strict=False, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= minimum if strict else value < minimum)
        or (integer and not isinstance(value, int))
    ):
        raise InvalidRegistrationConfigError(f"invalid {name}: {value!r}")


def _choice(value, name, choices):
    if value not in choices:
        raise InvalidRegistrationConfigError(f"{name} must be one of {choices}")


@dataclass(frozen=True)
class TranslationConfig:
    """Integer-peak translation estimation (ZYX, including singleton Z).

    Parameters
    ----------
    backend : str
        Phase-correlation backend: ``scipy_fft`` or ``skimage``.
    fft_workers : int
        Positive FFT worker count for ``scipy_fft``.
    method : str
        Method identifier, ``translation``.
    """

    backend: str = "scipy_fft"
    fft_workers: int = 1
    method: str = field(default="translation", init=False)

    def __post_init__(self):
        _choice(self.backend, "backend", ("scipy_fft", "skimage"))
        _number(self.fft_workers, "fft_workers", minimum=0, strict=True, integer=True)
        if self.backend == "skimage" and self.fft_workers != 1:
            raise InvalidRegistrationConfigError("skimage requires fft_workers=1")


def _elastix_config(config, metrics):
    _choice(config.metric, "metric", metrics)
    for name in ("iterations", "samples", "random_seed"):
        _number(getattr(config, name), name, minimum=0 if name == "random_seed" else 1, integer=True)
    if config.levels is not None:
        _number(config.levels, "levels", minimum=1, integer=True)


@dataclass(frozen=True)
class RigidConfig:
    """elastix rigid (Euler) registration in physical space; Z=1 is estimated in 2D.

    Parameters
    ----------
    metric : str
        Similarity metric: ``mattes`` mutual information or ``ncc`` normalized correlation.
    histogram_bins : int
        Histogram bin count for the ``mattes`` metric.
    iterations : int
        Positive optimization iteration budget per pyramid level.
    samples : int
        Positive spatial sample count for optimization.
    levels : int or None
        Pyramid level count; ``None`` requests an image-size-derived count.
    random_seed : int
        Nonnegative random sampling seed.
    method : str
        Method identifier, ``rigid``.
    """

    metric: str = "mattes"
    histogram_bins: int = 32
    iterations: int = 200
    samples: int = 4096
    levels: int | None = None
    random_seed: int = 1
    method: str = field(default="rigid", init=False)

    def __post_init__(self):
        _elastix_config(self, ("mattes", "ncc"))
        _number(self.histogram_bins, "histogram_bins", minimum=2, integer=True)


@dataclass(frozen=True)
class AffineConfig:
    """elastix affine registration in physical space; Z=1 is estimated in 2D.

    Parameters
    ----------
    metric : str
        Similarity metric: ``mattes`` mutual information or ``ncc`` normalized correlation.
    iterations : int
        Positive optimization iteration budget per pyramid level.
    samples : int
        Positive spatial sample count for optimization.
    levels : int or None
        Pyramid level count; ``None`` requests an image-size-derived count.
    random_seed : int
        Nonnegative random sampling seed.
    method : str
        Method identifier, ``affine``.
    """

    metric: str = "ncc"
    iterations: int = 200
    samples: int = 4096
    levels: int | None = None
    random_seed: int = 1
    method: str = field(default="affine", init=False)

    def __post_init__(self):
        _elastix_config(self, ("ncc", "mattes"))


@dataclass(frozen=True)
class BSplineConfig:
    """elastix cubic B-spline registration on a physical control grid; Z=1 is estimated in 2D.

    Parameters
    ----------
    metric : str
        Similarity metric: ``mattes`` mutual information or ``ncc`` normalized correlation.
    grid_spacing_physical : float or None
        Final control-grid spacing in the image's spatial unit; ``None`` requests an image-size-derived spacing.
    iterations : int
        Positive optimization iteration budget per pyramid level.
    samples : int
        Positive spatial sample count for optimization.
    levels : int or None
        Pyramid level count; ``None`` requests an image-size-derived count.
    random_seed : int
        Nonnegative random sampling seed.
    method : str
        Method identifier, ``bspline``.
    """

    metric: str = "ncc"
    grid_spacing_physical: float | None = None
    iterations: int = 200
    samples: int = 4096
    levels: int | None = None
    random_seed: int = 1
    method: str = field(default="bspline", init=False)

    def __post_init__(self):
        _elastix_config(self, ("ncc", "mattes"))
        if self.grid_spacing_physical is not None:
            _number(self.grid_spacing_physical, "grid_spacing_physical", strict=True)


@dataclass(frozen=True)
class DemonsConfig:
    """SimpleITK demons; Z=1 is estimated in 2D (Y and X at least four), otherwise 3D with every axis at least four.

    Parameters
    ----------
    variant : str
        Demons variant: ``demons``, ``diffeomorphic``, ``symmetric`` or ``fast_symmetric``.
    iterations : tuple[int, ...]
        Positive iteration budgets in coarse-to-fine pyramid order.
    smoothing_sigma : float
        Nonnegative displacement-field smoothing scale in voxel units.
    pyramid_mode : str
        Pyramid construction mode: ``antialias`` or ``sitk``.
    method : str
        Method identifier, ``demons``.
    """

    variant: str = "demons"
    iterations: tuple[int, ...] = (100, 50, 25)
    smoothing_sigma: float = 1
    pyramid_mode: str = "antialias"
    method: str = field(default="demons", init=False)

    def __post_init__(self):
        _choice(self.variant, "variant", ("demons", "diffeomorphic", "symmetric", "fast_symmetric"))
        _choice(self.pyramid_mode, "pyramid_mode", ("antialias", "sitk"))
        if not isinstance(self.iterations, (tuple, list)) or not self.iterations:
            raise InvalidRegistrationConfigError("iterations must be a nonempty sequence")
        for n in self.iterations:
            _number(n, "iterations", integer=True, strict=True)
        object.__setattr__(self, "iterations", tuple(self.iterations))
        _number(self.smoothing_sigma, "smoothing_sigma")


@dataclass(frozen=True)
class TpsConfig:
    """Thin-plate-spline field from spots matched between the two rounds; 3D only.

    Parameters
    ----------
    detection_noise_sigma : float
        Nonnegative noise-scale multiplier for landmark detection.
    match_distance_voxels : float
        Positive landmark matching distance in voxels.
    min_matches : int
        Required landmark match count.
    max_control_points : int
        Control-point count budget for TPS fitting.
    smoothing : float
        Nonnegative thin-plate-spline regularization strength.
    grid_spacing_voxels : int
        Positive coarse-field sampling stride setting in voxels.
    interpolation_order : int
        Spline order for coarse-field expansion.
    field_smoothing_sigma : float or None
        Gaussian field-smoothing standard deviation in voxels; ``None`` disables smoothing.
    clamp_sampling_coordinates : bool
        Whether to clamp the estimated field's source coordinates to the image grid.
    method : str
        Method identifier, ``tps``.
    """

    detection_noise_sigma: float = 3
    match_distance_voxels: float = 10
    min_matches: int = 50
    max_control_points: int = 1000
    smoothing: float = 1
    grid_spacing_voxels: int = 32
    interpolation_order: int = 3
    field_smoothing_sigma: float | None = None
    clamp_sampling_coordinates: bool = True
    method: str = field(default="tps", init=False)

    def __post_init__(self):
        _landmark_config(self)
        _number(self.match_distance_voxels, "match_distance_voxels", strict=True)
        _number(self.min_matches, "min_matches", minimum=4, integer=True)
        _number(self.smoothing, "smoothing")


@dataclass(frozen=True)
class CpdConfig:
    """Coherent point drift between the spots of the two rounds; 3D only.

    Parameters
    ----------
    detection_noise_sigma : float
        Nonnegative noise-scale multiplier for landmark detection.
    max_control_points : int
        Reference control-point count budget for CPD anchor selection.
    kernel_width_voxels : float or None
        Gaussian deformation-kernel width setting in voxels; ``None`` requests a point-density-derived width.
    regularization_weight : float
        Positive nonrigid CPD regularization weight.
    outlier_fraction : float
        Nonnegative CPD outlier probability weight.
    affine_first : bool
        Whether to fit an affine alignment before nonrigid CPD.
    grid_spacing_voxels : int
        Positive coarse-field sampling stride setting in voxels.
    candidate_radius_voxels : float
        Positive moving-landmark candidate search radius in voxels.
    neighbors_per_anchor : int
        Nonnegative reference-neighbor count setting per anchor.
    interpolation_order : int
        Spline order for coarse-field expansion.
    field_smoothing_sigma : float or None
        Gaussian field-smoothing standard deviation in voxels; ``None`` disables smoothing.
    clamp_sampling_coordinates : bool
        Whether to clamp the estimated field's source coordinates to the image grid.
    method : str
        Method identifier, ``cpd``.
    """

    detection_noise_sigma: float = 5
    max_control_points: int = 1000
    kernel_width_voxels: float | None = None
    regularization_weight: float = 2
    outlier_fraction: float = 0.15
    affine_first: bool = True
    grid_spacing_voxels: int = 16
    candidate_radius_voxels: float = 15
    neighbors_per_anchor: int = 3
    interpolation_order: int = 3
    field_smoothing_sigma: float | None = None
    clamp_sampling_coordinates: bool = True
    method: str = field(default="cpd", init=False)

    def __post_init__(self):
        _landmark_config(self)
        if self.kernel_width_voxels is not None:
            _number(self.kernel_width_voxels, "kernel_width_voxels", strict=True)
        _number(self.regularization_weight, "regularization_weight", strict=True)
        _number(self.outlier_fraction, "outlier_fraction")
        if self.outlier_fraction >= 1:
            raise InvalidRegistrationConfigError("outlier_fraction must be < 1")
        _number(self.candidate_radius_voxels, "candidate_radius_voxels", strict=True)
        _number(self.neighbors_per_anchor, "neighbors_per_anchor", integer=True)
        if type(self.affine_first) is not bool:
            raise InvalidRegistrationConfigError("affine_first must be bool")


def _landmark_config(config):
    _number(config.detection_noise_sigma, "detection_noise_sigma")
    _number(config.max_control_points, "max_control_points", minimum=4, integer=True)
    _number(config.grid_spacing_voxels, "grid_spacing_voxels", strict=True, integer=True)
    _number(config.interpolation_order, "interpolation_order", integer=True)
    if config.interpolation_order > 5:
        raise InvalidRegistrationConfigError("interpolation_order must be 0..5")
    if config.field_smoothing_sigma is not None:
        _number(config.field_smoothing_sigma, "field_smoothing_sigma")
    if type(config.clamp_sampling_coordinates) is not bool:
        raise InvalidRegistrationConfigError("clamp_sampling_coordinates must be bool")


@dataclass(frozen=True)
class RegistrationQcConfig:
    """Rejection criteria of routine registration QC; every criterion None rejects nothing.

    Parameters
    ----------
    min_coverage : float or None
        Minimum valid-overlap fraction criterion; ``None`` disables this criterion.
    min_ncc_gain : float or None
        Minimum normalized-correlation gain criterion; ``None`` disables this criterion.
    max_fold_fraction : float or None
        Maximum nonpositive-Jacobian voxel fraction criterion; ``None`` disables this criterion.
    max_translation_voxels : float or None
        Maximum translation displacement norm criterion in voxels; ``None`` disables this criterion.
    projections : bool
        Whether routine QC includes reference, before-registration and after-registration Z maximum projections.

    Notes
    -----
    Coverage and fold-fraction criteria are fractions in [0, 1].
    NCC gain is the correlation after registration minus the correlation before.
    """

    min_coverage: float | None = None
    min_ncc_gain: float | None = None
    max_fold_fraction: float | None = None
    max_translation_voxels: float | None = None
    projections: bool = False

    def __post_init__(self):
        for name in ("min_coverage", "max_fold_fraction"):
            value = getattr(self, name)
            if value is not None:
                _number(value, name)
                if value > 1:
                    raise InvalidRegistrationConfigError(f"{name} must be in [0, 1]")
        if self.min_ncc_gain is not None:
            _number(self.min_ncc_gain, "min_ncc_gain", minimum=-math.inf)
        if self.max_translation_voxels is not None:
            _number(self.max_translation_voxels, "max_translation_voxels")
        if type(self.projections) is not bool:
            raise InvalidRegistrationConfigError("projections must be bool")


def _channel(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, str))
                              or (isinstance(value, int) and value < 0) or value == ""):
        raise InvalidRegistrationConfigError(f"{name} must be a nonnegative index, a channel label or None")


@dataclass(frozen=True)
class RegistrationSignalConfig:
    """How one float64 ZYX registration signal is built from a round's ZYXC source image.

    Parameters
    ----------
    mode : str
        Signal construction mode: channel maximum (``max``), channel sum (``sum``) or selected channel (``channel``).
    reference_channel : int or str or None
        Reference-round channel selector for ``channel`` mode, as a zero-based index or channel label.
    moving_channel : int or str or None
        Moving-round channel selector for ``channel`` mode, as a zero-based index or channel label; ``None`` uses the reference selector.

    Notes
    -----
    Channels are set only for mode ``channel``.
    """

    mode: str = "max"
    reference_channel: int | str | None = None
    moving_channel: int | str | None = None

    def __post_init__(self):
        _choice(self.mode, "mode", ("max", "sum", "channel"))
        _channel(self.reference_channel, "reference_channel")
        _channel(self.moving_channel, "moving_channel")
        if self.mode == "channel" and self.reference_channel is None:
            raise InvalidRegistrationConfigError('mode="channel" requires reference_channel')
        if self.mode != "channel" and (self.reference_channel is not None or self.moving_channel is not None):
            raise InvalidRegistrationConfigError('channels apply only to mode="channel"')


@dataclass(frozen=True)
class WarpConfig:
    """Resampling policy, with one final nearest-even integer cast.

    Parameters
    ----------
    backend : str
        Resampling backend: ``translation``, ``scipy`` or ``simpleitk``.
    boundary_mode : str
        Boundary sampling mode: ``constant`` or ``nearest``.
    fill_value : float
        Constant-boundary fill intensity.
    output_dtype : str
        Output dtype choice: ``input``, ``float32`` or ``float64``.
    integer_rounding : str
        Integer-output rounding policy, ``nearest_even``.
    clip_to_dtype : bool
        Integer-output clipping policy, ``True``.
    fft_workers : int
        Positive FFT worker count for translation resampling.

    Notes
    -----
    Estimator ``interpolation_order`` controls coarse-field expansion only.
    """

    backend: str = "translation"
    boundary_mode: str = "constant"
    fill_value: float = 0
    output_dtype: str = "input"
    integer_rounding: str = "nearest_even"
    clip_to_dtype: bool = True
    fft_workers: int = 1

    def __post_init__(self):
        _choice(self.backend, "backend", ("translation", "scipy", "simpleitk"))
        _choice(self.boundary_mode, "boundary_mode", ("constant", "nearest"))
        _choice(self.output_dtype, "output_dtype", ("input", "float32", "float64"))
        _choice(self.integer_rounding, "integer_rounding", ("nearest_even",))
        if self.clip_to_dtype is not True:
            raise InvalidRegistrationConfigError("clip_to_dtype=True is required")
        if not isinstance(self.fill_value, (int, float)) or not math.isfinite(self.fill_value):
            raise InvalidRegistrationConfigError("fill_value must be finite")
        _number(self.fft_workers, "fft_workers", strict=True, integer=True)
        if self.backend == "translation" and (
            self.boundary_mode != "constant" or self.fill_value != 0
        ):
            raise InvalidRegistrationConfigError("translation requires constant zero fill")
        if self.backend != "translation" and self.fft_workers != 1:
            raise InvalidRegistrationConfigError("dense backends require fft_workers=1")
        if self.boundary_mode == "nearest" and self.fill_value != 0:
            raise InvalidRegistrationConfigError("nearest boundary does not use fill_value")
