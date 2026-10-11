"""The reference grid of a saved FOV: the file's stored metadata, else the grid declared by the configuration.

docs/sample-output-contract.md ("The export entry") and docs/assembly-contract.md ("Workflow
configuration"). One rule for ``Dataset.export`` (the assembly and fusion inputs),
``Dataset.stitch``, the workflow's stitching rules and its ``reads_assignment`` with an
``assembly`` block, so that a dataset analysed before §2.10, whose files store no calibration
or keep the reference image elsewhere, is placed and exported as saved. The file is the one
the setting ``reference_image`` names.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

#: The configuration keys of the declared calibration (workflow YAML; ``Dataset`` fields of the same names).
VOXEL_KEYS = ("voxel_size_xy", "voxel_size_z")
#: The setting naming a FOV's reference image (workflow YAML key and ``Dataset`` field) and its default: a file
#: pattern with ``{fov_id}``, absolute or relative to the output root.
REFERENCE_IMAGE_KEY = "reference_image"
REFERENCE_IMAGE = "images/ref_merged/{fov_id}.tif"


def check_reference_image(value) -> str:
    """The ``reference_image`` pattern; ValueError naming the setting for anything but a string with ``{fov_id}``."""
    if not isinstance(value, str) or "{fov_id}" not in value:
        raise ValueError(f"{REFERENCE_IMAGE_KEY} must be a file pattern with {{fov_id}}, absolute or relative to the "
                         f"output root (default {REFERENCE_IMAGE!r}); got {value!r}")
    return value


def reference_image_path(output_root, fov_id, pattern=REFERENCE_IMAGE) -> Path:
    """The file of a FOV's reference image: ``pattern`` with ``{fov_id}`` replaced, relative to ``output_root``."""
    path = Path(check_reference_image(pattern).replace("{fov_id}", fov_id))
    return path if path.is_absolute() else Path(output_root) / path


def reference_settings(source) -> dict:
    """The keyword arguments of :func:`reference_grid` from a ``Dataset`` or a workflow configuration mapping."""
    if isinstance(source, Mapping):
        pattern = source.get(REFERENCE_IMAGE_KEY)
        return {"dataset_id": source["dataset_id"], "output_id": source["output_id"],
                "voxel_size_xy": source.get("voxel_size_xy"), "voxel_size_z": source.get("voxel_size_z"),
                "reference_image": REFERENCE_IMAGE if pattern is None else pattern}
    return {"dataset_id": source.dataset_id, "output_id": source.output_id, "voxel_size_xy": source.voxel_size_xy,
            "voxel_size_z": source.voxel_size_z, "reference_image": source.reference_image}


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


def reference_grid(output_root, fov_id, *, dataset_id, output_id, voxel_size_xy=None, voxel_size_z=None,
                   reference_image=REFERENCE_IMAGE):
    """The reference grid of the FOV's reference image, the file ``reference_image`` names.

    The default is ``images/ref_merged/<fov_id>.tif``. A file with a
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
    path = _existing_reference(output_root, fov_id, reference_image)
    header = _volume_header(path)
    if header is None or header[1] or header[0] is None:
        return reference_volume(output_root, fov_id, reference_image=reference_image, **settings)[0]
    return ReferenceGrid(header[0], _declared_grid_metadata(path, fov_id, **settings), "declared")


def _existing_reference(output_root, fov_id, reference_image):
    """The path of the FOV's reference image; FileNotFoundError naming the setting and the path if it is missing."""
    path = reference_image_path(output_root, fov_id, reference_image)
    if not path.is_file():
        raise FileNotFoundError(f"the reference image of FOV {fov_id!r} is missing: {path} (the file the setting "
                                f"{REFERENCE_IMAGE_KEY} {reference_image!r} names)")
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
                     reference_image=REFERENCE_IMAGE):
    """(grid, the loaded file, its path) of :func:`reference_grid`, for a caller that also needs the pixels.

    The file is loaded whole with ``load_volume``; the grid is the one
    :func:`reference_grid` returns.
    """
    from starfinder.io import load_volume
    from starfinder.segmentation import ReferenceGrid
    from starfinder.segmentation._labels import grid_sha256

    path = _existing_reference(output_root, fov_id, reference_image)
    loaded = load_volume(path)
    if loaded.diagnostics["metadata_source"] == "stored":
        return (ReferenceGrid(loaded.image.shape, loaded.metadata, f"file:{Path(path).resolve()}",
                              grid_sha256(loaded.image)), loaded, path)
    metadata = _declared_grid_metadata(path, fov_id, dataset_id=dataset_id, output_id=output_id,
                                       voxel_size_xy=voxel_size_xy, voxel_size_z=voxel_size_z)
    return ReferenceGrid(loaded.image.shape, metadata, "declared"), loaded, path
