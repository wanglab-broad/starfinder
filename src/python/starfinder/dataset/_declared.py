"""The reference grid of a saved FOV: the file's stored metadata, else the grid declared by the configuration.

docs/sample-output-contract.md ("The export entry") and design/specs/assembly.md ("Workflow
configuration"). One rule for ``Dataset.export`` (the assembly and fusion inputs),
``Dataset.stitch``, the workflow's stitching rules and its ``reads_assignment`` with an
``assembly`` block, so that a dataset analysed before the stitching and sample-assembly stages,
whose files store no calibration or keep the reference image elsewhere, is placed and exported
as saved. The file is the one ``_layout.reference_image_path`` resolves: the setting
``reference_image_import`` when it is given, else ``reference_image``.
"""
# Before §2.10
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

from starfinder._layout import (REFERENCE_IMAGE, REFERENCE_IMAGE_IMPORT_KEY, REFERENCE_IMAGE_KEY,
                                check_reference_import, reference_image_path)

#: The configuration keys of the declared calibration (workflow YAML; ``Dataset`` fields of the same names).
VOXEL_KEYS = ("voxel_size_xy", "voxel_size_z")


def reference_settings(source) -> dict:
    """The keyword arguments of :func:`reference_grid` from a ``Dataset`` or a workflow configuration mapping.

    A mapping that gives both ``reference_image`` and ``reference_image_import``
    (neither null) raises ``ValueError``: the two exclude each other.
    """
    if isinstance(source, Mapping):
        pattern, imported = source.get(REFERENCE_IMAGE_KEY), source.get(REFERENCE_IMAGE_IMPORT_KEY)
        if pattern is not None and imported is not None:
            raise ValueError(f"{REFERENCE_IMAGE_IMPORT_KEY} ({imported!r}) excludes {REFERENCE_IMAGE_KEY} "
                             f"({pattern!r}): give one of the two")
        pattern = REFERENCE_IMAGE if pattern is None else pattern
        return {"dataset_id": source["dataset_id"], "output_id": source["output_id"],
                "voxel_size_xy": source.get("voxel_size_xy"), "voxel_size_z": source.get("voxel_size_z"),
                "reference_image": pattern, "reference_image_import": check_reference_import(pattern, imported)}
    return {"dataset_id": source.dataset_id, "output_id": source.output_id, "voxel_size_xy": source.voxel_size_xy,
            "voxel_size_z": source.voxel_size_z, "reference_image": source.reference_image,
            "reference_image_import": source.reference_image_import}


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
    alone. The workflow's segmentation and assignment rules declare the files
    they read with this metadata.
    """
    # The §2.9 workflow
    from starfinder.image import ImageMetadata
    frame = f"declared:{dataset_id}/{output_id}/{fov_id}"
    if voxel_size_z is None or voxel_size_xy is None:
        return ImageMetadata(frame)
    return ImageMetadata(frame, spacing_zyx=(float(voxel_size_z), float(voxel_size_xy), float(voxel_size_xy)),
                         spatial_unit="micrometer")


def reference_grid(output_root, fov_id, *, dataset_id, output_id, voxel_size_xy=None, voxel_size_z=None,
                   reference_image=REFERENCE_IMAGE, reference_image_import=None):
    """The reference grid of the FOV's reference image, the file ``reference_image_import`` or ``reference_image`` names.

    The default is ``fovs/<fov_id>/reference.tif``. A file with a
    ``starfinder_metadata`` description is read as
    :func:`starfinder.segmentation.reference_grid_from_file` reads it, and its
    metadata wins. A file without one takes the grid declared by the
    configuration: the file's shape, :func:`declared_metadata`, the source
    ``declared`` and no hash. Without stored metadata and without both voxel
    sizes it raises ``ValueError`` naming what to configure; a missing file
    raises ``FileNotFoundError`` naming the setting and the path.

    The shape of a file without stored metadata is taken from its TIFF header,
    and no pixel is read; a file with stored metadata (whose grid holds the
    SHA-256 of its pixels) and a file whose header does not settle the shape
    ``load_volume`` gives (a floating-point file, or one it refuses) are read
    whole, as :func:`reference_volume` reads them.
    """
    from starfinder.io.tiff import _volume_header
    from starfinder.segmentation import ReferenceGrid

    settings = {"dataset_id": dataset_id, "output_id": output_id, "voxel_size_xy": voxel_size_xy,
                "voxel_size_z": voxel_size_z}
    path = _existing_reference(output_root, fov_id, reference_image, reference_image_import)
    header = _volume_header(path)
    if header is None or header[1] or header[0] is None:
        return reference_volume(output_root, fov_id, reference_image=reference_image,
                                reference_image_import=reference_image_import, **settings)[0]
    return ReferenceGrid(header[0], _declared_grid_metadata(path, fov_id, **settings), "declared")


def _existing_reference(output_root, fov_id, reference_image, reference_image_import=None):
    """The path of the FOV's reference image; FileNotFoundError naming the setting and the path if it is missing."""
    path = reference_image_path(output_root, fov_id, reference_image, reference_image_import)
    if not path.is_file():
        key, pattern = ((REFERENCE_IMAGE_KEY, reference_image) if reference_image_import is None
                        else (REFERENCE_IMAGE_IMPORT_KEY, reference_image_import))
        raise FileNotFoundError(f"the reference image of FOV {fov_id!r} is missing: {path} (the file the setting "
                                f"{key} {pattern!r} names)")
    return path


def _declared_grid_metadata(path, fov_id, *, dataset_id, output_id, voxel_size_xy, voxel_size_z):
    """The declared metadata of a file without stored metadata; ValueError naming what to configure without both
    voxel sizes."""
    missing = [key for key, value in zip(VOXEL_KEYS, (voxel_size_xy, voxel_size_z)) if value is None]
    if missing:
        raise ValueError(f"{path} stores no calibration (no starfinder_metadata description), so its grid is declared "
                         f"by the dataset configuration, which lacks {' and '.join(missing)}: set "
                         f"{' and '.join(missing)} (µm) in the workflow configuration, or the Dataset fields of the "
                         "same names")
    return declared_metadata(dataset_id, output_id, fov_id, voxel_size_xy=voxel_size_xy, voxel_size_z=voxel_size_z)


def reference_volume(output_root, fov_id, *, dataset_id, output_id, voxel_size_xy=None, voxel_size_z=None,
                     reference_image=REFERENCE_IMAGE, reference_image_import=None):
    """(grid, the loaded file, its path) of :func:`reference_grid`, for a caller that also needs the pixels.

    The file is loaded whole with ``load_volume``; the grid is the one
    :func:`reference_grid` returns.
    """
    from starfinder.io import load_volume
    from starfinder.segmentation import ReferenceGrid
    from starfinder.segmentation._labels import grid_sha256

    path = _existing_reference(output_root, fov_id, reference_image, reference_image_import)
    loaded = load_volume(path)
    if loaded.diagnostics["metadata_source"] == "stored":
        return (ReferenceGrid(loaded.image.shape, loaded.metadata, f"file:{Path(path).resolve()}",
                              grid_sha256(loaded.image)), loaded, path)
    metadata = _declared_grid_metadata(path, fov_id, dataset_id=dataset_id, output_id=output_id,
                                       voxel_size_xy=voxel_size_xy, voxel_size_z=voxel_size_z)
    return ReferenceGrid(loaded.image.shape, metadata, "declared"), loaded, path
