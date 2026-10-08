"""The reference grid of a saved FOV: the file's stored metadata, else the grid declared by the configuration.

docs/sample-output-contract.md ("The export entry") and docs/assembly-contract.md ("Workflow
configuration"). One rule for ``Dataset.export`` (the assembly and fusion inputs) and the
workflow's ``reads_assignment`` with an ``assembly`` block, so that a dataset analysed before
§2.10, whose files store no calibration, is exported as saved.
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

#: The configuration keys of the declared calibration (workflow YAML; ``Dataset`` fields of the same names).
VOXEL_KEYS = ("voxel_size_xy", "voxel_size_z")


def check_voxel_size(name, value):
    """None, or a finite positive real number (as float); ValueError naming ``name`` otherwise."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number of micrometres or None; got {value!r}")
    return float(value)


def declared_metadata(dataset_id, output_id, fov_id, *, voxel_size_xy=None, voxel_size_z=None):
    """The ``ImageMetadata`` of a FOV's file without stored metadata, declared from the configuration.

    Frame ``declared:<dataset_id>/<output_id>/<fov_id>``; with both voxel sizes
    the spacing ``(voxel_size_z, voxel_size_xy, voxel_size_xy)`` in
    ``micrometer``, no origin and no direction; with either missing, the frame
    alone. The §2.9 workflow declares the files it reads with this metadata.
    """
    from starfinder.image import ImageMetadata
    frame = f"declared:{dataset_id}/{output_id}/{fov_id}"
    if voxel_size_z is None or voxel_size_xy is None:
        return ImageMetadata(frame)
    return ImageMetadata(frame, spacing_zyx=(float(voxel_size_z), float(voxel_size_xy), float(voxel_size_xy)),
                         spatial_unit="micrometer")


def reference_grid(output_root, fov_id, *, dataset_id, output_id, voxel_size_xy=None, voxel_size_z=None):
    """The reference grid of ``images/ref_merged/<fov_id>.tif``.

    A file with a ``starfinder_metadata`` description is read as
    :func:`starfinder.segmentation.reference_grid_from_file` reads it, and its
    metadata wins. A file without one takes the grid declared by the
    configuration: the file's shape, :func:`declared_metadata`, the source
    ``declared`` and no hash. Without stored metadata and without both voxel
    sizes it raises ``ValueError`` naming what to configure.
    """
    from starfinder.io import load_volume
    from starfinder.segmentation import ReferenceGrid
    from starfinder.segmentation._labels import grid_sha256

    from ._paths import _FovPaths
    path = _FovPaths(Path(output_root), fov_id).ref_merged_tif
    loaded = load_volume(path)
    if loaded.diagnostics["metadata_source"] == "stored":
        return ReferenceGrid(loaded.image.shape, loaded.metadata, f"file:{Path(path).resolve()}",
                             grid_sha256(loaded.image))
    missing = [key for key, value in zip(VOXEL_KEYS, (voxel_size_xy, voxel_size_z)) if value is None]
    if missing:
        raise ValueError(f"{path} stores no calibration (no starfinder_metadata description), so its grid is declared "
                         f"by the dataset configuration, which lacks {' and '.join(missing)}: set "
                         f"{' and '.join(missing)} (µm) in the workflow configuration, or the Dataset fields of the "
                         "same names")
    metadata = declared_metadata(dataset_id, output_id, fov_id, voxel_size_xy=voxel_size_xy,
                                 voxel_size_z=voxel_size_z)
    return ReferenceGrid(loaded.image.shape, metadata, "declared")
