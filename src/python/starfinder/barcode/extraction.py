"""Labeled neighborhood sums, independent of decoding and filtering."""

from dataclasses import dataclass
from numbers import Integral
from collections.abc import Mapping
import numpy as np

from starfinder.image import ImageMetadata, _validate_image
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import SpotFindingResult
from .codebook import _labels

# Scale of the median absolute deviation to the standard deviation of a normal distribution.
MAD_SCALE = 1.4826


def _radius(value, name):
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in value)
    ):
        raise ValueError(f"{name} must be a nonnegative integer triple")


@dataclass(frozen=True)
class LocalBackgroundConfig:
    """Local background and noise of each candidate, per channel and round (estimator local_ring).

    The ring is the voxels of the outer box (half-widths outer_radius_zyx) that
    lie outside the inner box (inner_radius_zyx) around the rounded centre,
    clipped to the image: by default the extraction box's own three z-planes at
    lateral Chebyshev distance 4 to 6, 360 voxels unclipped. background is the
    ring's median and noise 1.4826 x its median absolute deviation, in grey
    levels per voxel. With fewer than min_voxels ring voxels inside the image
    both are NaN (min_voxels 16 is provisional). The ring is not masked for
    other spots (docs/readout-algorithms.md, "Background and noise").
    """

    inner_radius_zyx: tuple[int, int, int] = (1, 3, 3)
    outer_radius_zyx: tuple[int, int, int] = (1, 6, 6)
    min_voxels: int = 16

    def __post_init__(self):
        _radius(self.inner_radius_zyx, "inner_radius_zyx")
        _radius(self.outer_radius_zyx, "outer_radius_zyx")
        if any(o < i for o, i in zip(self.outer_radius_zyx, self.inner_radius_zyx)) or (
                self.outer_radius_zyx == self.inner_radius_zyx):
            raise ValueError("the outer box must contain the inner box and be larger along at least one axis")
        if isinstance(self.min_voxels, bool) or not isinstance(self.min_voxels, Integral) or self.min_voxels < 1:
            raise ValueError("min_voxels must be a positive integer")


@dataclass(frozen=True)
class NeighborhoodSumConfig:
    """Float64 sums with nearest floor(coord+.5) sampling and zero boundaries.

    Half-widths are voxel counts, never physical spacing. Inputs are not mutated.
    A clipped slice is equivalent to zero padding without allocating padded images.
    background measures the local background and noise next to the sums
    (LocalBackgroundConfig, on by default); None turns it off. Its inner box must
    contain the extraction box.
    """

    neighborhood_radius_zyx: tuple[int, int, int] = (1, 2, 2)
    sampling: str = "nearest"
    boundary: str = "zero"
    background: LocalBackgroundConfig | None = LocalBackgroundConfig()

    def __post_init__(self):
        _radius(self.neighborhood_radius_zyx, "neighborhood_radius_zyx")
        if self.sampling != "nearest" or self.boundary != "zero":
            raise ValueError("only nearest sampling and zero boundary are supported")
        if self.background is not None:
            if not isinstance(self.background, LocalBackgroundConfig):
                raise TypeError("background must be LocalBackgroundConfig or None")
            self.background.__post_init__()
            if any(i < r for i, r in zip(self.background.inner_radius_zyx, self.neighborhood_radius_zyx)):
                raise ValueError(
                    f"the background inner box {self.background.inner_radius_zyx} must contain the extraction box "
                    f"{self.neighborhood_radius_zyx}; widen LocalBackgroundConfig.inner_radius_zyx (and "
                    "outer_radius_zyx) or pass background=None")


@dataclass(frozen=True)
class IntensityExtractionResult:
    """Finite float64 N×C×R sums, stable identities and explicit acquisition axes.

    valid is Boolean N×R; false marks an unavailable measurement, not zero signal
    (in readout mode direct, every round but a candidate's own). box_voxels is
    int64 N×R, the number of voxels each box summed (fewer where the box is
    clipped by a face, 0 where a round was not extracted), or None when not
    recorded (a candidates checkpoint without background measurements does not
    store it).
    Signed extraction is allowed; decoding chooses reject/clip_negative explicitly.

    The local background measurements (config.background) are None when not
    measured, otherwise all present: background and noise are float64 N×C×R in
    grey levels per voxel of the extraction image (NaN where the ring has fewer
    than min_voxels voxels inside the image, or the round was not extracted);
    background_voxels is int64 N×R, the ring voxels inside the image (0 where the
    round was not extracted); image_background and image_noise are float64 C×R,
    the median and 1.4826 x MAD of the whole round and channel. A box's
    background in sum units is background × box_voxels.
    The repr is a one-line N×C×R summary without values.
    """

    values: np.ndarray
    spot_ids: tuple[str, ...]
    spot_namespace: str
    channel_labels: tuple[str, ...]
    round_labels: tuple[str, ...]
    metadata: ImageMetadata
    config: NeighborhoodSumConfig
    valid: np.ndarray
    diagnostics: dict
    box_voxels: np.ndarray | None = None
    background: np.ndarray | None = None
    noise: np.ndarray | None = None
    background_voxels: np.ndarray | None = None
    image_background: np.ndarray | None = None
    image_noise: np.ndarray | None = None

    def __post_init__(self):
        _labels(self.channel_labels, "channel_labels")
        _labels(self.round_labels, "round_labels")
        if not isinstance(self.spot_namespace, str) or not self.spot_namespace.strip():
            raise ValueError("spot_namespace must be nonempty")
        if (
            not isinstance(self.spot_ids, tuple)
            or any(not isinstance(s, str) or not s for s in self.spot_ids)
            or len(set(self.spot_ids)) != len(self.spot_ids)
        ):
            raise ValueError("spot_ids must be unique nonempty strings")
        shape = (len(self.spot_ids), len(self.channel_labels), len(self.round_labels))
        if (
            not isinstance(self.values, np.ndarray)
            or self.values.dtype != np.float64
            or self.values.shape != shape
            or not np.isfinite(self.values).all()
        ):
            raise ValueError("values must be finite float64 (N,C,R) matching identities/labels")
        if (
            not isinstance(self.valid, np.ndarray)
            or self.valid.dtype != bool
            or self.valid.shape != (shape[0], shape[2])
        ):
            raise ValueError("valid must be Boolean (N,R)")
        if self.box_voxels is not None and (
            not isinstance(self.box_voxels, np.ndarray)
            or self.box_voxels.dtype != np.int64
            or self.box_voxels.shape != (shape[0], shape[2])
            or (self.box_voxels < 0).any()
        ):
            raise ValueError("box_voxels must be nonnegative int64 (N,R) or None")
        measured = [getattr(self, name) is not None for name in _BACKGROUND_FIELDS]
        if any(measured) and not all(measured):
            raise ValueError(f"the background measurements {', '.join(_BACKGROUND_FIELDS)} are all None or all given")
        if all(measured):
            for name, expected in (("background", shape), ("noise", shape),
                                   ("image_background", shape[1:]), ("image_noise", shape[1:])):
                value = getattr(self, name)
                if (not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.shape != expected
                        or np.isinf(value).any()):
                    raise ValueError(f"{name} must be float64 {expected} without infinities (NaN when unavailable)")
            if (
                not isinstance(self.background_voxels, np.ndarray)
                or self.background_voxels.dtype != np.int64
                or self.background_voxels.shape != (shape[0], shape[2])
                or (self.background_voxels < 0).any()
            ):
                raise ValueError("background_voxels must be nonnegative int64 (N,R)")
        if not isinstance(self.metadata, ImageMetadata) or not isinstance(
            self.config, NeighborhoodSumConfig
        ):
            raise TypeError("metadata/config type mismatch")

    def _summary(self):
        return (
            f"{len(self.spot_ids)} spots × {len(self.channel_labels)} channels × "
            f"{len(self.round_labels)} rounds"
        )

    def __repr__(self):
        return f"IntensityExtractionResult: {self._summary()}"


_BACKGROUND_FIELDS = ("background", "noise", "background_voxels", "image_background", "image_noise")


def _ring_mask(config):
    """Boolean outer-box mask, true on the ring (outside the inner box), indexed by offset + outer."""
    outer = np.asarray(config.outer_radius_zyx)
    grids = np.meshgrid(*[np.arange(-o, o + 1) for o in outer], indexing="ij")
    return ~np.logical_and.reduce([np.abs(g) <= i for g, i in zip(grids, config.inner_radius_zyx)])


def _local_background(image, centers, rows, config, mask):
    """Ring median, 1.4826 x MAD (N, C) and ring voxel counts (N,) of one ZYXC round at rows."""
    shape = np.asarray(image.shape[:3])
    outer = np.asarray(config.outer_radius_zyx)
    n, c = len(centers), image.shape[3]
    median, noise = np.full((n, c), np.nan), np.full((n, c), np.nan)
    voxels = np.zeros(n, dtype=np.int64)
    for k in rows:
        p = centers[k]
        lo, hi = np.maximum(p - outer, 0), np.minimum(p + outer + 1, shape)
        offset = lo - (p - outer)
        ring = mask[tuple(slice(o, o + h - l) for o, l, h in zip(offset, lo, hi))]
        values = image[tuple(slice(a, b) for a, b in zip(lo, hi))][ring].astype(np.float64)
        voxels[k] = len(values)
        if len(values) >= config.min_voxels:
            m = np.median(values, axis=0)
            median[k] = m
            noise[k] = MAD_SCALE * np.median(np.abs(values - m), axis=0)
    return median, noise, voxels


def _image_background(image):
    """Median and 1.4826 x MAD of every voxel of each channel of one ZYXC round, (C,) each."""
    median, noise = np.empty(image.shape[3]), np.empty(image.shape[3])
    for c in range(image.shape[3]):
        values = image[..., c].astype(np.float64).ravel()
        median[c] = np.median(values)
        noise[c] = MAD_SCALE * np.median(np.abs(values - median[c]))
    return median, noise


def extract_intensities(
    rounds: Mapping[str, ImageLoadResult],
    spots: SpotFindingResult,
    *,
    config: NeighborhoodSumConfig = NeighborhoodSumConfig(),
    readout_mode: str = "multiplexed",
) -> IntensityExtractionResult:
    """Sum ordered, labeled ZYXC rounds at spot coordinates on their common grid.

    Mapping insertion order defines R. Every round must have exactly the same
    shape, metadata and channel order. Coordinates must lie within voxel-center
    bounds [0, size-1] before rounding. Empty spots retain shape (0,C,R).

    readout_mode ``multiplexed`` (default) sums every round at every spot, and
    valid is all true. ``direct`` needs a ``round`` column in the spot table and
    sums each spot in its own round only; its other rounds (and every round of
    a spot whose round is not in the mapping) are 0.0 with valid false and
    box_voxels 0 (docs/readout-contract.md, "Direct readout").

    With config.background (on by default) each extracted (spot, round) also
    gets the local ring background and noise per channel, and each round and
    channel the image median and noise (IntensityExtractionResult). The
    measurements never change values or valid. Rounds that are not extracted
    have NaN background and noise and 0 ring voxels.
    """
    if not isinstance(config, NeighborhoodSumConfig) or not isinstance(spots, SpotFindingResult):
        raise TypeError("expected NeighborhoodSumConfig and SpotFindingResult")
    if readout_mode not in ("multiplexed", "direct"):
        raise ValueError(f"readout_mode must be multiplexed or direct; got {readout_mode!r}")
    spots.__post_init__()
    if readout_mode == "direct" and "round" not in spots.spots:
        raise ValueError("readout mode direct extracts each candidate in its own round and needs a round "
                         "column: detect with a SpotFindingPlan with explicit rounds")
    labels = tuple(rounds)
    _labels(labels, "round_labels")
    first = rounds[labels[0]]
    if not isinstance(first, ImageLoadResult):
        raise TypeError("rounds must contain ImageLoadResult")
    channel_labels = tuple(first.channel_labels)
    _labels(channel_labels, "channel_labels")
    coords = spots.spots[["z", "y", "x"]].to_numpy()
    shape = first.image.shape
    images = []
    for label, loaded in rounds.items():
        if not isinstance(loaded, ImageLoadResult):
            raise TypeError(f"{label}: expected ImageLoadResult")
        image = _validate_image(loaded.image, ndim=(4,))
        if (
            image.shape != shape
            or loaded.metadata != spots.metadata
            or tuple(loaded.channel_labels) != channel_labels
            or image.shape[3] != len(channel_labels)
        ):
            raise ValueError(f"{label}: frame/grid/channel label mismatch")
        images.append(image)
    detector_labels = spots.diagnostics.get("channel_labels")
    if detector_labels is not None and tuple(detector_labels) != channel_labels:
        raise ValueError("spot channel labels disagree with extraction labels")
    if "channel" in spots.spots and (spots.spots.channel >= len(channel_labels)).any():
        raise ValueError("spot channel outside labeled image")
    if (coords < 0).any() or (coords > np.asarray(shape[:3]) - 1).any():
        raise ValueError("coordinates outside voxel-center bounds")
    centers = np.floor(coords + 0.5).astype(np.int64)
    radius = np.asarray(config.neighborhood_radius_zyx)
    lows = np.maximum(centers - radius, 0)
    highs = np.minimum(centers + radius + 1, np.asarray(shape[:3]))
    if readout_mode == "direct":
        own = spots.spots["round"]
        read = np.stack([own.eq(label).fillna(False).to_numpy(dtype=bool) for label in labels],
                        axis=1).reshape(len(coords), len(labels))
        values = np.zeros((len(coords), len(channel_labels), len(labels)), dtype=np.float64)
    else:
        read = np.ones((len(coords), len(labels)), dtype=bool)
        values = np.empty((len(coords), len(channel_labels), len(labels)), dtype=np.float64)
    for r, image in enumerate(images):
        for n in np.flatnonzero(read[:, r]):
            values[n, :, r] = image[tuple(slice(a, b) for a, b in zip(lows[n], highs[n]))].sum(
                axis=(0, 1, 2), dtype=np.float64
            )
    box_voxels = np.where(read, np.prod(highs - lows, axis=1)[:, None], 0).astype(np.int64)
    measurements = {}
    if config.background is not None:
        n, c, r = values.shape
        background, noise = np.full((n, c, r), np.nan), np.full((n, c, r), np.nan)
        background_voxels = np.zeros((n, r), dtype=np.int64)
        image_background, image_noise = np.empty((c, r)), np.empty((c, r))
        mask = _ring_mask(config.background)
        for j, image in enumerate(images):
            background[:, :, j], noise[:, :, j], background_voxels[:, j] = _local_background(
                image, centers, np.flatnonzero(read[:, j]), config.background, mask)
            image_background[:, j], image_noise[:, j] = _image_background(image)
        measurements = dict(background=background, noise=noise, background_voxels=background_voxels,
                            image_background=image_background, image_noise=image_noise)
    return IntensityExtractionResult(
        values,
        tuple(spots.spots.spot_id),
        spots.spot_namespace,
        channel_labels,
        labels,
        spots.metadata,
        config,
        read,
        {
            "source_shape_zyx": shape[:3],
            "calculation_dtype": "float64",
            "sampling": "floor(coord+.5)",
        },
        box_voxels,
        **measurements,
    )
