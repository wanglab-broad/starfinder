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


@dataclass(frozen=True)
class DemonsConfig:
    """SimpleITK demons; 3D only, with each axis at least four voxels."""

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
class WarpConfig:
    """Method-specific resampling, with one final nearest-even integer cast.

    Translation supports constant zero fill only. Dense backends support
    constant fill or nearest boundary extrapolation. SciPy uses linear samples;
    estimator interpolation_order controls coarse field expansion only.
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
