"""Scalar and 3D background subtraction (preprocessing algorithms page, task group 2)."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import ndimage

from starfinder.image import ImageMetadata, _validate_image
from starfinder.preprocessing._diagnostics import channel_diagnostics
from starfinder.preprocessing.histograms import HistogramSummary, _steps_record, histogram_percentile
from starfinder.preprocessing.normalization import _FITS, _channel_percentile


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _cast_subtracted(values, dtype):
    """max(values, 0) in dtype: integers round half to even and clip to the dtype range."""
    values = np.maximum(values, 0)
    if dtype.kind in "ui":
        limits = np.iinfo(dtype)
        values = np.clip(np.rint(values), limits.min, limits.max)
    return values.astype(dtype)


@dataclass(frozen=True)
class ScalarBackgroundConfig:
    """One background level per channel and round, subtracted uniformly.

    percentile is an intensity percent in [0, 100) (provisional default 10);
    the level is that percentile of all voxels of the channel by the
    inverted-CDF definition. fit="fov" estimates it on each round of each
    FOV; fit="supplied" reads it per round from the scalar_background section
    of the recipe's supplied-statistics file.
    """

    percentile: float = 10.0
    fit: str = "fov"

    def __post_init__(self):
        if not _number(self.percentile) or not 0 <= self.percentile < 100:
            raise ValueError("percentile must be a finite number in [0, 100)")
        if self.fit not in _FITS:
            raise ValueError('fit must be "fov" or "supplied"')


def _supplied_background(fitted, n_channels):
    if not isinstance(fitted, Mapping) or "background" not in fitted:
        raise ValueError('supplied scalar background requires "background"')
    background = list(fitted["background"])
    if len(background) != n_channels:
        raise ValueError(f"supplied scalar background has {len(background)} channels, the image has {n_channels}")
    if not all(map(_number, background)):
        raise ValueError("supplied scalar background values must be finite numbers")
    return background


def _fit_scalar(config, merged, reference_round):
    """params {"percentile"} and, per round, {"background": [...]} from merged counts."""
    fitted = {name: {"background": [histogram_percentile(counts, config.percentile) for counts in merged.counts[r]]}
              for r, name in enumerate(merged.round_names)}
    return {"params": {"percentile": float(config.percentile)}, "fitted": fitted}


def _validate_scalar(section, dtype, n_channels):
    params = section["params"]
    if not isinstance(params, Mapping) or set(params) != {"percentile"}:
        raise ValueError('scalar_background params must be {"percentile"}')
    ScalarBackgroundConfig(params["percentile"])
    for name, entry in section["fitted"].items():
        try:
            _supplied_background(entry, n_channels)
        except ValueError as error:
            raise ValueError(f"scalar_background round {name!r}: {error}") from None


def _check_scalar_params(config, params, name):
    if params["percentile"] != config.percentile:
        raise ValueError(f"section {name!r} was fitted with percentile={params['percentile']}, "
                         f"not the recipe's {config.percentile}")


def _scalar_background(volume, config, fitted=None):
    """Subtracted image, fitted {"background"} and per-channel diagnostics."""
    volume = _validate_image(volume)
    channels = volume[..., None] if volume.ndim == 3 else volume
    n = channels.shape[-1]
    if config.fit == "supplied":
        if fitted is None:
            raise ValueError('fit="supplied" requires the supplied background values')
        background = _supplied_background(fitted, n)
    else:
        if fitted is not None:
            raise ValueError('fitted values are accepted only with fit="supplied"')
        background = [_channel_percentile(channels[..., c], config.percentile) for c in range(n)]
    result = np.empty(channels.shape, dtype=volume.dtype)
    constant = []
    for c in range(n):
        x = channels[..., c].astype(np.float64)
        constant.append(bool(x.min() == x.max()))
        result[..., c] = _cast_subtracted(x - background[c], volume.dtype)
    result = result[..., 0] if volume.ndim == 3 else result
    return result, {"background": background}, {**channel_diagnostics(result), "constant_channel": constant}


def subtract_scalar_background(volume: np.ndarray, *, config: ScalarBackgroundConfig = ScalarBackgroundConfig(),
                               fitted: Mapping | None = None) -> np.ndarray:
    """Subtract one background level b per channel: max(x - b, 0) in the input dtype.

    With fit="fov", b is the config's percentile of all voxels of the channel
    by the inverted-CDF definition, from the integer histogram for uint8 and
    uint16 and numpy.percentile(method="inverted_cdf") otherwise; a constant
    channel therefore gives zeros. With fit="supplied", fitted must give
    {"background": [...]} with one value per channel. Integer output is
    rounded half to even and clipped to the dtype range; float output is not
    rounded. Works in float64 one channel at a time; the input is not modified.

    Raises
    ------
    ValueError
        Empty or nonfinite input, missing or mismatched supplied values, or
        fitted values given with fit="fov".
    """
    return _scalar_background(volume, config, fitted)[0]


def _subtract_counts(counts, background):
    """Histogram of max(x - b, 0) from the histogram of x, for an integer b >= 0."""
    background = min(background, counts.size - 1)
    shifted = np.zeros_like(counts)
    shifted[0] = counts[:background + 1].sum()
    shifted[1:counts.size - background] = counts[background + 1:]
    return shifted


def scalar_background_histograms(summary: HistogramSummary, config: ScalarBackgroundConfig, *,
                                 supplied: Mapping | None = None) -> HistogramSummary:
    """Histograms after a scalar background step, derived exactly from those at its input.

    The histogram of max(x - b, 0) follows from the histogram of x: bins
    above b shift down by b and bins at or below b accumulate in bin 0. With
    fit="fov", b is fitted per round and channel from the summary's own
    counts, so the summary must hold one unmerged FOV (apply this before
    merging). With fit="supplied", b is read per round from supplied, the
    scalar_background section of the supplied-statistics file, and the
    summary may be merged. The result records the step after the summary's
    summarized_after, as summary_stage records it for the next fitted step.

    Raises
    ------
    ValueError
        fit="fov" on a merged summary; a supplied section with fit="fov";
        a missing section, round or channel, or one fitted with another
        percentile or at another recipe stage; or a background that is not a
        nonnegative integer.
    """
    if not isinstance(summary, HistogramSummary):
        raise TypeError("scalar_background_histograms requires a HistogramSummary")
    if type(config) is not ScalarBackgroundConfig:
        raise TypeError("scalar_background_histograms requires a ScalarBackgroundConfig")
    config.__post_init__()
    n = len(summary.channel_labels)
    if config.fit == "fov":
        if supplied is not None:
            raise ValueError('a supplied section is accepted only with fit="supplied"')
        if len(summary.fovs_used) != 1 or summary.fovs_excluded:
            raise ValueError('fit="fov" backgrounds differ per FOV; derive each FOV\'s histograms before merging')
        backgrounds = [[histogram_percentile(counts, config.percentile) for counts in round_counts]
                       for round_counts in summary.counts]
    else:
        if not isinstance(supplied, Mapping) or not isinstance(supplied.get("fitted"), Mapping):
            raise ValueError('fit="supplied" requires the scalar_background section of the supplied statistics')
        if supplied.get("params") != {"percentile": config.percentile}:
            raise ValueError(f"the supplied section was fitted with {supplied.get('params')}, "
                             f"not percentile={config.percentile}")
        if supplied.get("summarized_after") != list(summary.summarized_after):
            raise ValueError(f"the supplied section was summarized after {supplied.get('summarized_after')}, "
                             f"not after the summary's {list(summary.summarized_after)}")
        backgrounds = []
        for name in summary.round_names:
            if name not in supplied["fitted"]:
                raise ValueError(f"the supplied statistics have no values for round {name!r}")
            backgrounds.append(_supplied_background(supplied["fitted"][name], n))
    counts = np.empty_like(summary.counts)
    for r, round_backgrounds in enumerate(backgrounds):
        for c, b in enumerate(round_backgrounds):
            if not float(b).is_integer() or b < 0:
                raise ValueError(f"the histogram shortcut requires integer backgrounds >= 0, not {b!r}")
            counts[r, c] = _subtract_counts(summary.counts[r, c], int(b))
    from starfinder.preprocessing.steps import step_spec
    after = _steps_record(list(summary.summarized_after) + [{"step": step_spec(config).name, "config": asdict(config)}])
    return HistogramSummary(counts, summary.dtype, summary.round_names, summary.channel_labels, tuple(after),
                            summary.fovs_used, summary.fovs_excluded)


def _radii(value, name, integer):
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 3:
        raise ValueError(f"{name} must be three values (z, y, x)")
    value = tuple(value)
    valid = (lambda r: isinstance(r, int) and not isinstance(r, bool)) if integer else _number
    if not all(map(valid, value)):
        raise ValueError(f"{name} must hold {'integers' if integer else 'finite numbers'}")
    if any(r < 0 for r in value):
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class Background3DConfig:
    """Anisotropic ellipsoidal radii (z, y, x) of a volumetric white top-hat.

    Exactly one of radius_um_zyx (micrometres, converted with
    ImageMetadata.spacing_zyx) and radius_voxels_zyx (voxels) must be set;
    there is no default radius. A radius of 0 leaves that axis unfiltered.
    Choose radii larger than the puncta.
    """

    radius_um_zyx: tuple[float, float, float] | None = None
    radius_voxels_zyx: tuple[int, int, int] | None = None

    def __post_init__(self):
        if (self.radius_um_zyx is None) == (self.radius_voxels_zyx is None):
            raise ValueError("exactly one of radius_um_zyx and radius_voxels_zyx must be set")
        object.__setattr__(self, "radius_um_zyx", _radii(self.radius_um_zyx, "radius_um_zyx", False))
        object.__setattr__(self, "radius_voxels_zyx", _radii(self.radius_voxels_zyx, "radius_voxels_zyx", True))


def _voxel_radii(config, metadata):
    if config.radius_voxels_zyx is not None:
        return config.radius_voxels_zyx
    spacing = None if metadata is None else metadata.spacing_zyx
    if spacing is None:
        raise ValueError("radius_um_zyx requires ImageMetadata.spacing_zyx; give radius_voxels_zyx when spacing is unknown")
    return tuple(int(np.rint(r / s)) for r, s in zip(config.radius_um_zyx, spacing))


def _ellipsoid(radii):
    """Boolean footprint of voxels with sum((d / r)^2) <= 1, length 1 along axes with r = 0.

    Evaluated in integers: sum(d_i^2 * prod_{j != i} r_j^2) <= prod r_i^2 over the nonzero radii.
    """
    grids = np.meshgrid(*(np.arange(-r, r + 1) for r in radii), indexing="ij")
    nonzero = [i for i, r in enumerate(radii) if r > 0]
    total = math.prod(radii[i] ** 2 for i in nonzero)
    lhs = np.zeros(grids[0].shape, dtype=np.int64)
    for i in nonzero:
        lhs += grids[i].astype(np.int64) ** 2 * (total // radii[i] ** 2)
    return lhs <= total


def _background_3d(volume, config, metadata=None):
    """Subtracted image, fitted {"radius_voxels_zyx"} and per-channel diagnostics."""
    volume = _validate_image(volume)
    radii = _voxel_radii(config, metadata)
    for axis, r, size in zip("zyx", radii, volume.shape[:3]):
        if 2 * r + 1 > size:
            raise ValueError(f"the footprint ({2 * r + 1} voxels along {axis}, radius {r}) is larger than the volume ({size})")
    footprint = _ellipsoid(radii)
    channels = volume[..., None] if volume.ndim == 3 else volume
    result = np.empty(channels.shape, dtype=volume.dtype)
    constant = []
    for c in range(channels.shape[-1]):
        x = channels[..., c]
        constant.append(bool(x.min() == x.max()))
        # Opening takes minima and maxima of input values, so it is exact in the input dtype.
        background = ndimage.grey_opening(x, footprint=footprint, mode="reflect")
        result[..., c] = _cast_subtracted(x.astype(np.float64) - background, volume.dtype)
    result = result[..., 0] if volume.ndim == 3 else result
    diagnostics = {**channel_diagnostics(result), "constant_channel": constant,
                   "footprint_voxels": int(footprint.sum())}
    return result, {"radius_voxels_zyx": list(radii)}, diagnostics


def subtract_background_3d(volume: np.ndarray, *, config: Background3DConfig,
                           metadata: ImageMetadata | None = None) -> np.ndarray:
    """Volumetric white top-hat: max(x - grey_opening(x), 0) per channel, in the input dtype.

    The background is scipy.ndimage.grey_opening with an ellipsoidal
    footprint of semi-axes (r_z, r_y, r_x) voxels, reflect boundaries. Radii
    in micrometres are converted with metadata.spacing_zyx and rounded half
    to even. Cost is about voxels x footprint voxels comparisons for each of
    the erosion and dilation, single-threaded; there is no fast path and no
    downsampling. A constant channel gives zeros. The input is not modified.

    Raises
    ------
    ValueError
        Empty or nonfinite input, radius_um_zyx without spacing, or a
        footprint longer than the volume along an axis.
    """
    return _background_3d(volume, config, metadata)[0]
