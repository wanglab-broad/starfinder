"""Labeled neighborhood sums, independent of decoding and filtering."""

from dataclasses import dataclass
from numbers import Integral
from collections.abc import Mapping
import numpy as np

from starfinder.image import ImageMetadata, _validate_image
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import SpotFindingResult
from .codebook import _labels


@dataclass(frozen=True)
class NeighborhoodSumConfig:
    """Float64 sums with nearest floor(coord+.5) sampling and zero boundaries.

    Half-widths are voxel counts, never physical spacing. Inputs are not mutated.
    A clipped slice is equivalent to zero padding without allocating padded images.
    """

    neighborhood_radius_zyx: tuple[int, int, int] = (1, 2, 2)
    sampling: str = "nearest"
    boundary: str = "zero"

    def __post_init__(self):
        if (
            not isinstance(self.neighborhood_radius_zyx, tuple)
            or len(self.neighborhood_radius_zyx) != 3
            or any(
                isinstance(x, bool) or not isinstance(x, Integral) or x < 0
                for x in self.neighborhood_radius_zyx
            )
        ):
            raise ValueError("neighborhood_radius_zyx must be a nonnegative integer triple")
        if self.sampling != "nearest" or self.boundary != "zero":
            raise ValueError("only nearest sampling and zero boundary are supported")


@dataclass(frozen=True)
class IntensityExtractionResult:
    """Finite float64 N×C×R sums, stable identities and explicit acquisition axes.

    valid is Boolean N×R; false marks an unavailable measurement, not zero signal.
    Signed extraction is allowed; decoding chooses reject/clip_negative explicitly.
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
        if not isinstance(self.metadata, ImageMetadata) or not isinstance(
            self.config, NeighborhoodSumConfig
        ):
            raise TypeError("metadata/config type mismatch")


def extract_intensities(
    rounds: Mapping[str, ImageLoadResult],
    spots: SpotFindingResult,
    *,
    config: NeighborhoodSumConfig = NeighborhoodSumConfig(),
) -> IntensityExtractionResult:
    """Sum ordered, labeled ZYXC rounds at spot coordinates on their common grid.

    Mapping insertion order defines R. Every round must have exactly the same
    shape, metadata and channel order. Coordinates must lie within voxel-center
    bounds [0, size-1] before rounding. Empty spots retain shape (0,C,R).
    """
    if not isinstance(config, NeighborhoodSumConfig) or not isinstance(spots, SpotFindingResult):
        raise TypeError("expected NeighborhoodSumConfig and SpotFindingResult")
    spots.__post_init__()
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
    values = np.empty((len(coords), len(channel_labels), len(labels)), dtype=np.float64)
    radius = np.asarray(config.neighborhood_radius_zyx)
    for r, image in enumerate(images):
        for n, center in enumerate(centers):
            lo = np.maximum(center - radius, 0)
            hi = np.minimum(center + radius + 1, shape[:3])
            values[n, :, r] = image[tuple(slice(a, b) for a, b in zip(lo, hi))].sum(
                axis=(0, 1, 2), dtype=np.float64
            )
    return IntensityExtractionResult(
        values,
        tuple(spots.spots.spot_id),
        spots.spot_namespace,
        channel_labels,
        labels,
        spots.metadata,
        config,
        np.ones((len(coords), len(labels)), dtype=bool),
        {
            "source_shape_zyx": shape[:3],
            "calculation_dtype": "float64",
            "sampling": "floor(coord+.5)",
        },
    )
