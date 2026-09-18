"""Image geometry and finite ZYX/ZYXC array contracts."""
from dataclasses import dataclass, replace

import numpy as np

__all__ = ["ImageMetadata"]


def _validate_image(image, *, ndim=(3, 4)):
    image = np.asarray(image)
    if image.ndim not in ndim or any(n == 0 for n in image.shape):
        raise ValueError("image must be nonempty ZYX or ZYXC")
    if image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError("image must contain finite real numeric values")
    return image


def _triple(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite triple")
    return tuple(float(x) for x in a)


@dataclass(frozen=True)
class ImageMetadata:
    """Spatial geometry independent of image shape and intensity.

    Unknown physical fields remain None. World coordinates are
    ``origin + direction @ (index * spacing)`` in ZYX order. All physical
    fields, including unit, are required for physical conversion.
    """

    frame_id: str
    spacing_zyx: tuple[float, float, float] | None = None
    origin_zyx: tuple[float, float, float] | None = None
    direction_zyx: tuple[tuple[float, float, float], ...] | None = None
    spatial_unit: str | None = None

    def __post_init__(self):
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a nonempty string")
        for name in ("spacing_zyx", "origin_zyx"):
            value = getattr(self, name)
            if value is not None:
                value = _triple(value, name)
                if name == "spacing_zyx" and any(x <= 0 for x in value):
                    raise ValueError("spacing_zyx must be strictly positive")
                object.__setattr__(self, name, value)
        if self.direction_zyx is not None:
            d = np.asarray(self.direction_zyx, dtype=float)
            if d.shape != (3, 3) or not np.isfinite(d).all() or not np.allclose(d.T @ d, np.eye(3), atol=1e-8, rtol=0):
                raise ValueError("direction_zyx must be finite and orthonormal")
            object.__setattr__(self, "direction_zyx", tuple(tuple(float(x) for x in row) for row in d))
        if self.spatial_unit is not None and (not isinstance(self.spatial_unit, str) or not self.spatial_unit):
            raise ValueError("spatial_unit must be a nonempty string or None")

    def index_to_world(self, index_zyx):
        """Convert finite (..., 3) voxel indices to physical ZYX coordinates."""
        self._require_physical()
        points = self._points(index_zyx)
        return np.asarray(self.origin_zyx) + (points * self.spacing_zyx) @ np.asarray(self.direction_zyx).T

    def world_to_index(self, world_zyx):
        """Convert finite (..., 3) physical coordinates to voxel indices."""
        self._require_physical()
        return ((self._points(world_zyx) - self.origin_zyx) @ np.asarray(self.direction_zyx)) / self.spacing_zyx

    def _require_physical(self):
        if any(getattr(self, n) is None for n in ("spacing_zyx", "origin_zyx", "direction_zyx", "spatial_unit")):
            raise ValueError("physical conversion requires complete geometry and spatial_unit")

    @staticmethod
    def _points(value):
        points = np.asarray(value, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("coordinates must be finite (..., 3) ZYX points")
        return points

    def cropped(self, start_zyx, *, frame_id):
        """Map output indices to source indices by adding the crop start.

        Origin is updated only if spacing, origin and direction are known;
        otherwise it stays unknown. No unit calibration is invented.
        """
        start = _triple(start_zyx, "start_zyx")
        origin = None
        if all(x is not None for x in (self.origin_zyx, self.direction_zyx, self.spacing_zyx)):
            origin = tuple(np.asarray(self.origin_zyx) + np.asarray(self.direction_zyx) @ (np.array(start) * self.spacing_zyx))
        return replace(self, frame_id=frame_id, origin_zyx=origin)

    def rotated(self, shape_zyx, angle, *, frame_id):
        """Geometry for XY rotation: rot90 expands; other angles keep shape.

        Pull map is source = R @ (output - output_center) + source_center,
        R in YX is [[cos, sin], [-sin, cos]]. Known anisotropic XY spacing
        at non-right angles would require shear and is rejected.
        """
        shape = np.asarray(_triple(shape_zyx, "shape_zyx"))
        if not np.isfinite(angle):
            raise ValueError("angle must be finite")
        k = round(angle / 90)
        right = abs(angle - k * 90) < 1e-6
        theta = np.deg2rad(k * 90 if right else angle)
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
        out_shape = shape.copy()
        if right and k % 2:
            out_shape[1:] = shape[:0:-1]
        offset = (shape - 1) / 2 - r @ ((out_shape - 1) / 2)
        spacing = self.spacing_zyx
        direction = None
        origin = None
        if spacing is not None:
            if not right and not np.isclose(spacing[1], spacing[2]):
                raise ValueError("non-right-angle rotation of anisotropic XY geometry requires unsupported shear")
            new_spacing = tuple(np.asarray(spacing)[[0, 2, 1]]) if right and k % 2 else spacing
            if self.direction_zyx is not None:
                direction = np.asarray(self.direction_zyx) @ np.diag(spacing) @ r @ np.diag(1 / np.array(new_spacing))
                if self.origin_zyx is not None:
                    origin = tuple(np.array(self.origin_zyx) + np.asarray(self.direction_zyx) @ (offset * spacing))
            spacing = new_spacing
        # Unknown spacing prevents an orientation claim for an index rotation.
        return ImageMetadata(frame_id, spacing, origin, direction, self.spatial_unit)

    def projected(self, *, method):
        """Derived singleton-Z frame: (0,y,x) represents all source (z,y,x).

        A collapsed column has no unique 3D physical position. Physical fields
        are unknown in the derived frame; source frame and reduction appear in
        its identifier. The source metadata must be retained by the caller.
        """
        if method not in ("max", "sum"):
            raise ValueError("projection method must be max or sum")
        return ImageMetadata(f"{self.frame_id}/projection:{method}")
