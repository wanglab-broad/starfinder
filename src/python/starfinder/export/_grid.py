"""The fusion grid and the geometry of each output level.

design/specs/sample-output.md ("The fusion grid", "The scale factor").
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from starfinder.image import ImageMetadata
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import PlacementResult
from starfinder.stitching._placement import OWNING

from ._config import FusionConfig

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
AXES = ("z", "y", "x")


def included_fovs(placement: PlacementResult) -> list[str]:
    """The included FOVs of a placement, in placement table order."""
    return [fov for fov, status in zip(placement.fovs["fov_id"], placement.fovs["status"]) if status in OWNING]


def _names(values):
    from starfinder.stitching._positions import _names as names
    return names(values)


@dataclass(frozen=True)
class FusedGrid:
    """The library's union grid of the included FOVs, and the output levels of a factor ``k`` in Y and X.

    Parameters
    ----------
    sample : str
        Identifier of the fused sample.
    fov_ids : tuple of str
        Included FOV identifiers in placement table order.
    spacing_zyx : tuple of float
        Shared input voxel spacing in micrometres, in ZYX order.
    origin_zyx : tuple of float
        Union-grid origin in the sample frame, in micrometres and ZYX order.
    shape_zyx : tuple of int
        Union-grid voxel counts in ZYX order before output scaling.
    one_plane : bool
        Whether the included FOVs have singleton Z.
    slab_z : tuple of float
        Lower and upper Z edges of the projection slab in sample-frame micrometres.
    origins_z : tuple of float
        Included FOV origins along Z, in micrometres and ``fov_ids`` order.
    factor_yx : int
        Output downsampling factor along Y and X; 1 leaves their spacing unchanged.

    Notes
    -----
    ``spacing_zyx`` is the FOVs' shared spacing ``s``, ``origin_zyx`` the
    smallest included origin per axis ``o`` and ``shape_zyx`` the number of grid
    voxel centres ``n0`` up to the last voxel centre of any included FOV;
    index ``v`` lies at ``o + v ⊙ s`` (:attr:`metadata`). ``one_plane`` is true
    when every included FOV has one plane: they are fused as 2D, ``n0_z`` is 1
    and ``slab_z`` ``[z_lo, z_hi)`` spans every FOV's plane. For multi-plane
    FOVs ``slab_z`` is the grid's Z extent, the projection's one block.
    ``origins_z`` lists each included FOV's ``origin_z``. :meth:`level` gives
    the spacing, origin and shape of the stack or the projection at
    ``factor_yx``.
    """

    sample: str
    fov_ids: tuple[str, ...]
    spacing_zyx: tuple[float, float, float]
    origin_zyx: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    one_plane: bool
    slab_z: tuple[float, float]
    origins_z: tuple[float, ...]
    factor_yx: int = 1

    @property
    def metadata(self) -> ImageMetadata:
        """The fusion grid's ``ImageMetadata`` in the sample frame (index ``v`` at ``o + v ⊙ s``)."""
        return ImageMetadata(f"sample:{self.sample}", self.spacing_zyx, self.origin_zyx, IDENTITY, "µm")

    def level(self, dimensionality: str) -> tuple[tuple[float, float, float], tuple[float, float, float],
                                                  tuple[int, int, int]]:
        """``(S, O, N)`` per axis Z, Y, X of the ``stack`` or ``projection`` at ``factor_yx``.

        Parameters
        ----------
        dimensionality : str
            Output form: ``stack`` or ``projection``.

        Returns
        -------
        spacing_zyx : tuple of float
            Output voxel spacing in micrometres, in ZYX order.
        origin_zyx : tuple of float
            Output origin in sample-frame micrometres, in ZYX order.
        shape_zyx : tuple of int
            Output voxel counts in ZYX order.

        Notes
        -----
        Y and X: ``S = k·s``, ``O = o + s·(k − 1)/2``, ``N = ceil(n0/k)``. Z of
        the stack: ``s_z``, ``o_z``, ``n0_z``; of the projection: the slab,
        ``n0_z·s_z`` and ``o_z + s_z·(n0_z − 1)/2``, one plane. For one-plane
        FOVs Z is the slab of both files: ``z_hi − z_lo`` and
        ``(z_lo + z_hi)/2``.
        """
        if dimensionality not in ("stack", "projection"):
            raise ValueError(f"dimensionality must be 'stack' or 'projection'; got {dimensionality!r}")
        k = self.factor_yx
        (sz, sy, sx), (oz, oy, ox), (nz, ny, nx) = self.spacing_zyx, self.origin_zyx, self.shape_zyx
        if self.one_plane:
            z_lo, z_hi = self.slab_z
            z = (z_hi - z_lo, (z_lo + z_hi) / 2, 1)
        elif dimensionality == "stack":
            z = (sz, oz, nz)
        else:
            z = (nz * sz, oz + sz * (nz - 1) / 2, 1)
        yx = [(k * s, o + s * (k - 1) / 2, math.ceil(n / k)) for s, o, n in ((sy, oy, ny), (sx, ox, nx))]
        return ((z[0], yx[0][0], yx[1][0]), (z[1], yx[0][1], yx[1][1]), (z[2], yx[0][2], yx[1][2]))

    def level_metadata(self, dimensionality: str) -> ImageMetadata:
        """The ``ImageMetadata`` of the stack or projection at ``factor_yx`` in the sample frame.

        Parameters
        ----------
        dimensionality : str
            Output form: ``stack`` or ``projection``.

        Returns
        -------
        ImageMetadata
            Output grid metadata in the sample frame, with micrometre units.
        """
        spacing, origin, _ = self.level(dimensionality)
        return ImageMetadata(f"sample:{self.sample}", spacing, origin, IDENTITY, "µm")


def fused_grid(placement: PlacementResult, grids: Mapping[str, ReferenceGrid], *,
               config: FusionConfig = FusionConfig()) -> FusedGrid:
    """The fusion grid of a placement's included FOVs, from their origins and grids only.

    Spacing ``s`` is the included FOVs' shared spacing; origin
    ``o_a = min_f origin_f,a``; shape ``n0_a = floor((max_f (origin_f,a +
    (n_f,a − 1)·s_a) − o_a) / s_a + 1e-9) + 1``. When every included FOV has
    one plane, ``n0_z = 1`` and the slab is ``[min_f origin_f,z − s_z/2,
    max_f origin_f,z + s_z/2)``. Excluded FOVs (``placeholder``, ``missing``,
    ``failed``) do not extend it. ``config.scale_yx`` sets the factor of
    :meth:`FusedGrid.level`. No library is imported.

    Parameters
    ----------
    placement : PlacementResult
        Any placement.
    grids : Mapping[str, ReferenceGrid]
        At least the grid of every included FOV; other keys are ignored.
    config : FusionConfig
        Its ``scale_yx`` gives ``factor_yx``.

    Returns
    -------
    FusedGrid

    Raises
    ------
    ValueError
        No included FOV, an included FOV without a grid or without known
        spacing, included FOVs of different spacings, or one-plane FOVs
        beside multi-plane ones.
    """
    if not isinstance(placement, PlacementResult):
        raise TypeError("placement must be a PlacementResult")
    if not isinstance(config, FusionConfig):
        raise TypeError("config must be a FusionConfig")
    ids = included_fovs(placement)
    if not ids:
        raise ValueError(f"the placement of sample {placement.sample!r} has no included FOV (no_included_fov)")
    without = [fov for fov in ids if fov not in grids]
    if without:
        raise ValueError(f"every included FOV needs a grid; FOVs without one: {_names(without)}")
    unknown = [fov for fov in ids if grids[fov].metadata.spacing_zyx is None]
    if unknown:
        raise ValueError(f"the included FOVs need known spacing; FOVs without: {_names(unknown)}")
    spacings = {}
    for fov in ids:
        spacings.setdefault(tuple(float(v) for v in grids[fov].metadata.spacing_zyx), []).append(fov)
    if len(spacings) > 1:
        raise ValueError("the included FOVs must share one spacing; " +
                         "; ".join(f"{_names(fovs)}: {key!r}" for key, fovs in spacings.items()))
    planes = {}
    for fov in ids:
        planes.setdefault(grids[fov].shape_zyx[0] == 1, []).append(fov)
    if len(planes) > 1:
        raise ValueError("one-plane FOVs cannot be fused beside multi-plane ones (the library fuses one "
                         f"dimensionality at a time); one plane: {_names(planes[True])}, more: {_names(planes[False])}")
    one_plane = next(iter(planes))
    spacing = next(iter(spacings))
    rows = placement.fovs.set_index("fov_id").loc[ids]
    origins = rows[[f"origin_{a}" for a in AXES]].to_numpy(dtype=float)
    shapes = [grids[fov].shape_zyx for fov in ids]
    origin, shape = [], []
    for i, s in enumerate(spacing):
        o = float(origins[:, i].min())
        last = max(float(origins[f, i]) + (shapes[f][i] - 1) * s for f in range(len(ids)))
        origin.append(o)
        shape.append(int(math.floor((last - o) / s + 1e-9)) + 1)
    sz = spacing[0]
    if one_plane:
        shape[0] = 1
        slab = (float(origins[:, 0].min()) - sz / 2, float(origins[:, 0].max()) + sz / 2)
    else:
        slab = (origin[0] - sz / 2, origin[0] + (shape[0] - 0.5) * sz)
    return FusedGrid(placement.sample, tuple(ids), spacing, tuple(origin), tuple(shape), one_plane, slab,
                     tuple(float(v) for v in origins[:, 0]), config.factor_yx)
