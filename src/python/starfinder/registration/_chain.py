"""Composition of the step transforms of one moving round and its one final resampling."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from starfinder.image import IncompatibleGeometryError

from ._errors import UnsupportedTransformOperationError
from ._resampling import _cast_warp_output, _output_dtype
from ._types import AffineTransform, BSplineTransform, DenseDisplacementTransform, TranslationTransform

_TRANSFORMS = (TranslationTransform, AffineTransform, BSplineTransform, DenseDisplacementTransform)


def transform_kind(transform):
    """The saved kind of a step transform: translation, affine, bspline or dense."""
    for kind, cls in zip(("translation", "affine", "bspline", "dense"), _TRANSFORMS):
        if isinstance(transform, cls):
            return kind
    raise UnsupportedTransformOperationError("unsupported transform type")


def _affine(matrix, q):
    """A q + b on stacked ZYX points q of shape (3, ...), in a fixed summation order."""
    a, b = matrix[:3, :3], matrix[:3, 3]
    return np.stack([a[i, 0] * q[0] + a[i, 1] * q[1] + a[i, 2] * q[2] + b[i] for i in range(3)])


@dataclass(frozen=True, eq=False)
class TransformChain:
    """The ordered step transforms of one moving round, composed into one pull map.

    Step k is estimated on the moving signal resampled through steps 1 to
    k-1, so the composite pull map is Phi(p) = T1(T2(...Tn(p))): the last
    transform is evaluated at reference grid points and every earlier one
    analytically at the resulting points. Only the last transform may be a
    B-spline or dense field (a local step is always last), so no dense field
    is interpolated. All transforms share one equal-shaped grid. The chain's
    reference geometry is that of the last transform and its moving geometry
    that of the first.
    """

    transforms: tuple
    direction: str = "reference_to_moving"
    units: str = "voxel_index"

    def __post_init__(self):
        transforms = tuple(self.transforms) if isinstance(self.transforms, (tuple, list)) else None
        if not transforms:
            raise UnsupportedTransformOperationError("a transform chain requires a nonempty sequence of transforms")
        if any(not isinstance(t, _TRANSFORMS) for t in transforms):
            raise UnsupportedTransformOperationError("unsupported transform type in chain")
        if any(isinstance(t, (BSplineTransform, DenseDisplacementTransform)) for t in transforms[:-1]):
            raise UnsupportedTransformOperationError("only the last transform of a chain may be a B-spline or dense field")
        shapes = {t.reference_shape_zyx for t in transforms} | {t.moving_shape_zyx for t in transforms}
        if len(shapes) != 1:
            raise IncompatibleGeometryError("chained transforms must share one grid shape")
        if self.direction != "reference_to_moving" or self.units != "voxel_index":
            raise UnsupportedTransformOperationError("unsupported chain direction/units")
        object.__setattr__(self, "transforms", transforms)

    @property
    def reference_shape_zyx(self):
        return self.transforms[-1].reference_shape_zyx

    @property
    def moving_shape_zyx(self):
        return self.transforms[0].moving_shape_zyx

    @property
    def reference_metadata(self):
        return self.transforms[-1].reference_metadata

    @property
    def moving_metadata(self):
        return self.transforms[0].moving_metadata

    def translation(self):
        """The one TranslationTransform a chain of translations reduces to, else None.

        Its correction is the sum of the corrections, in step order.
        """
        if not all(isinstance(t, TranslationTransform) for t in self.transforms):
            return None
        correction = np.zeros(3)
        for t in self.transforms:
            correction = correction + np.asarray(t.correction_zyx, dtype=np.float64)
        return TranslationTransform(tuple(correction), self.reference_shape_zyx, self.moving_shape_zyx,
                                    self.reference_metadata, self.moving_metadata)

    def _last_field(self):
        """The last transform's float64 displacement on the reference grid, when it is a field."""
        last = self.transforms[-1]
        if isinstance(last, DenseDisplacementTransform):
            return last.displacement_zyx
        if isinstance(last, BSplineTransform):
            cached = self.__dict__.get("_bspline_field")
            if cached is None:
                cached = last.dense().displacement_zyx
                object.__setattr__(self, "_bspline_field", cached)
            return cached
        return None

    def pull_points(self, z):
        """Float64 pull points Phi(p) of reference plane z, shape (3, Y, X) in ZYX order."""
        _, ny, nx = self.reference_shape_zyx
        q = np.empty((3, ny, nx), dtype=np.float64)
        q[0] = z
        q[1] = np.arange(ny, dtype=np.float64)[:, None]
        q[2] = np.arange(nx, dtype=np.float64)[None, :]
        field = self._last_field()
        if field is not None:
            q = q + np.moveaxis(np.asarray(field[z], dtype=np.float64), -1, 0)
            earlier = self.transforms[:-1]
        else:
            earlier = self.transforms
        for transform in reversed(earlier):
            if isinstance(transform, TranslationTransform):
                q = q - np.asarray(transform.correction_zyx, dtype=np.float64)[:, None, None]
            else:
                q = _affine(transform.matrix_zyx, q)
        return q

    def pull_field(self):
        """The composite pull displacement u(p) = Phi(p) - p as a float64 DenseDisplacementTransform."""
        shape = self.reference_shape_zyx
        field = np.empty((*shape, 3), dtype=np.float64)
        grid = np.stack(np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape[1:]), indexing="ij"))
        for z in range(shape[0]):
            points = self.pull_points(z)
            field[z, ..., 0] = points[0] - z
            field[z, ..., 1:] = np.moveaxis(points[1:] - grid, 0, -1)
        return DenseDisplacementTransform(field, self.reference_shape_zyx, self.moving_shape_zyx,
                                          self.reference_metadata, self.moving_metadata)


def resample(images, chain, config):
    """Resample each ZYX(C) image once through chain with WarpConfig config; one output per image.

    A chain of translations requires the translation backend (exact integer
    shift or Fourier shift of the summed correction); every other chain needs
    scipy (linear map_coordinates in float64, plane by plane: the pull points
    of one Z plane are computed once and every channel of every image is
    sampled at them) or simpleitk (linear, on the composite pull field).
    Integer outputs are rounded nearest-even, clipped and cast once.
    """
    translation = chain.translation()
    if (translation is not None) != (config.backend == "translation"):
        raise UnsupportedTransformOperationError(
            "the translation backend applies exactly the chains of translations")
    stacks = []
    for image in images:
        if image.shape[:3] != chain.moving_shape_zyx:
            raise IncompatibleGeometryError("moving image does not match transform grid")
        channels = image[..., None] if image.ndim == 3 else image
        stacks.append((channels, np.empty(channels.shape, dtype=_output_dtype(image.dtype, config.output_dtype))))
    if config.backend == "translation":
        from ._translation import apply_shift

        for channels, output in stacks:
            for c in range(channels.shape[-1]):
                output[..., c] = apply_shift(channels[..., c], translation.correction_zyx,
                                             workers=config.fft_workers, output_dtype=config.output_dtype)
    elif config.backend == "scipy":
        for z in range(chain.reference_shape_zyx[0]):
            points = chain.pull_points(z)
            for channels, output in stacks:
                for c in range(channels.shape[-1]):
                    sampled = map_coordinates(channels[..., c], points, order=1, mode=config.boundary_mode,
                                              cval=config.fill_value, output=np.float64)
                    output[z, ..., c] = _cast_warp_output(sampled, output.dtype)
    else:
        from ._api import apply_transform

        field = chain.pull_field()
        for channels, output in stacks:
            output[...] = apply_transform(channels, field, config=config)
    return [output[..., 0] if image.ndim == 3 else output for image, (_, output) in zip(images, stacks)]
