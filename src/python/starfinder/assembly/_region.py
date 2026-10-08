"""A FOV's ownership region and excluded observations, for the §2.9 assignment figure (W-330).

The region is the set of pixels of the FOV's Y, X grid whose centres the FOV owns, found with
the ``owner`` test of the sample's ownership map, rebuilt from the placement and the grids the
assembly records (docs/assembly-contract.md, "Ownership regions"). The excluded observations
are read from the assembled tables; nothing here applies the retention rule again.
"""
from __future__ import annotations

import numpy as np

from starfinder.assignment import FOVOwnership
from starfinder.image import ImageMetadata
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import PlacementResult
from starfinder.stitching._placement import PLACED, table_sha256

from ._assemble import AssemblyResult
from ._config import AssemblyConfig
from ._ownership import ownership_map

# Rows of pixel centres tested per owner call, so that a large grid is tested in parts.
_ROWS = 256


def fov_ownership(result: AssemblyResult, fov_id: str, *, placement: PlacementResult) -> FOVOwnership:
    """The ownership record of one FOV of an assembled sample, for :func:`starfinder.assignment.plot_assignment`.

    ``region`` holds, for each pixel ``(y, x)`` of the FOV's grid, whether the
    FOV owns its centre ``origin + (y, x) · spacing`` in the sample frame (the
    rule :func:`assemble` uses for every observation); the owner is the
    ``owner`` test of :func:`ownership_map` on ``placement``, the grids and the
    config the assembly records. A FOV without an origin has an empty region.
    ``excluded_cells`` and ``excluded_molecules`` are the ``cell_id`` and
    ``spot_id`` of the FOV's rows of ``result.cells`` and ``result.molecules``
    whose assembly status is ``excluded``.

    Parameters
    ----------
    result : AssemblyResult
        The assembled sample, in memory or from :func:`load_assembly`.
    fov_id : str
        A FOV with an input in the assembly.
    placement : PlacementResult
        The placement the sample was assembled from: the region needs its
        tile position table, which the assembly does not hold.

    Returns
    -------
    FOVOwnership

    Raises
    ------
    TypeError
        result is not an AssemblyResult or placement not a PlacementResult.
    ValueError
        The assembly holds no input of ``fov_id`` (naming it and the FOVs it
        holds), or the placement is not the one the sample was assembled from.
    """
    if not isinstance(result, AssemblyResult):
        raise TypeError("result must be an AssemblyResult")
    if not isinstance(placement, PlacementResult):
        raise TypeError("placement must be a PlacementResult")
    entries = {entry["fov_id"]: entry for entry in result.record.get("fovs") or []}
    held = [fov for fov, entry in entries.items() if entry.get("grid") is not None]
    if fov_id not in held:
        raise ValueError(f"the assembly of sample {result.sample!r} holds no FOV {fov_id!r}; its FOVs with an input "
                         f"are {held}")
    recorded = result.record.get("placement") or {}
    current = {"fovs_sha256": table_sha256(placement.fovs), "links_sha256": table_sha256(placement.links)}
    differ = [k for k, v in current.items() if recorded.get(k) != v]
    if differ:
        raise ValueError(f"the assembly of sample {result.sample!r} was made from another placement: "
                         + "; ".join(f"{k} recorded {recorded.get(k)}, the placement {current[k]}" for k in differ))
    grids = {fov: ReferenceGrid(tuple(entries[fov]["grid"]["shape_zyx"]),
                                ImageMetadata(**entries[fov]["grid"]["metadata"]), "declared") for fov in held}
    shape = grids[fov_id].shape_zyx[1:]
    row = placement.fovs[placement.fovs["fov_id"].eq(fov_id).to_numpy(dtype=bool)].iloc[0]
    region = np.zeros(shape, bool)
    if row["status"] in PLACED:
        owners = ownership_map(placement, grids, config=AssemblyConfig(**result.record["config"]))
        origin = np.array([row["origin_z"], row["origin_y"], row["origin_x"]], np.float64)
        spacing = np.asarray(grids[fov_id].metadata.spacing_zyx, np.float64)
        x = np.arange(shape[1], dtype=np.float64)
        for y0 in range(0, shape[0], _ROWS):
            y = np.arange(y0, min(y0 + _ROWS, shape[0]), dtype=np.float64)
            index = np.zeros((len(y) * shape[1], 3))
            index[:, 1], index[:, 2] = np.repeat(y, shape[1]), np.tile(x, len(y))
            # The index-to-sample rule of assemble (to_sample in _assemble.assemble): origin + p · spacing.
            region[y0:y0 + len(y)] = (owners.owner(origin + index * spacing) == fov_id).reshape(len(y), shape[1])
    cells, molecules = result.cells, result.molecules
    mine = cells["fov_id"].eq(fov_id).to_numpy(dtype=bool) & cells["assembly_status"].eq("excluded").to_numpy(
        dtype=bool)
    gone = molecules["fov_id"].eq(fov_id).to_numpy(dtype=bool) & molecules["assembly_status"].eq(
        "excluded").to_numpy(dtype=bool)
    return FOVOwnership(region, frozenset(int(c) for c in cells["cell_id"][mine]),
                        frozenset(str(s) for s in molecules["spot_id"][gone]), fov_id)
