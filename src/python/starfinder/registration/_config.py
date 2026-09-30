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
    """Integer-peak translation estimation (ZYX, including singleton Z)."""

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

    metric is "mattes" (Mattes mutual information with histogram_bins bins) or
    "ncc"; iterations are per pyramid level, samples are the random spatial
    samples per iteration and random_seed seeds them. levels None derives the
    shared pyramid rule of the registration algorithm page.
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

    Fields as for RigidConfig; the default metric is normalized correlation.
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

    grid_spacing_physical is the final control-point spacing, in the spatial
    unit of spacing_zyx and equal on every axis; None uses the physical X
    extent divided by 8. Other fields as for AffineConfig.
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
    """SimpleITK demons; Z=1 is estimated in 2D (Y and X at least four), otherwise 3D with every axis at least four."""

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
    """Noise landmarks and 3D thin-plate-spline interpolation."""

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
    """3D coherent point drift; direct defaults differ from FOV defaults."""

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

    min_coverage is the smallest valid-overlap fraction in [0, 1];
    min_ncc_gain the smallest NCC gain (after minus before);
    max_fold_fraction the largest fraction in [0, 1] of voxels with
    det(I + grad u) <= 0; max_translation_voxels the largest translation
    correction norm in voxels. projections returns the reference, before and
    after Z maximum projections from registration_qc for overlays.
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

    mode "max" takes the channel maximum (MATLAB merged-image), "sum" the
    float64 channel sum (the earlier Python merged) and "channel" one channel
    per round: reference_channel for the reference round and moving_channel
    (None: reference_channel) for the moving round, each a zero-based index or
    a channel label. Channels are set only for mode "channel".
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

    Translation supports constant zero fill only and applies translations and
    chains of translations. Dense backends support constant fill or nearest
    boundary extrapolation. SciPy uses linear samples; estimator
    interpolation_order controls coarse field expansion only.
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
