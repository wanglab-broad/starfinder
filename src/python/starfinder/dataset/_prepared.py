"""The saved form of a prepared morphology image (docs/checkpoints.md, "Prepared morphology images").

Each image prepared by ``FOV.prepare_morphology`` or registered by
``FOV.register_rounds`` with checkpoints is a folder
``<checkpoint dir>/<fov_id>/other_rounds/<name>/`` holding ``image.ome.tif``
(ZYXC, every channel, with its ``ImageMetadata``) and ``registration.json``
(the record); a chain with a dense or B-spline transform also has
``field.npz``. ``FOV.load_registered_round`` reads it back.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
STAGE = "prepared_image"
IMAGE_FILE = "image.ome.tif"
RECORD_FILE = "registration.json"
FIELD_FILE = "field.npz"
FILES = (RECORD_FILE, IMAGE_FILE, FIELD_FILE)   # removal order: the record first
# The registration_record["rounds"] relation of the reference stain: no transform, no attempt.
SAME_ACQUISITION = "same acquisition as the reference round"
# Keys of the record's registration section besides the registration_record["rounds"] entry.
CHAIN_KEYS = ("transforms", "application_config", "attempts", "field")


def image_directory(fov_directory, name) -> Path:
    """``<fov checkpoint dir>/other_rounds/<name>``."""
    return Path(fov_directory) / "other_rounds" / name


def image_sha256(image) -> str:
    """SHA-256 of an image's C-order bytes, the convention of reference_sha256 and the reference grid."""
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def check_writable(directory, overwrite) -> None:
    """FileExistsError when the image folder exists and overwrite is False."""
    if Path(directory).exists() and not overwrite:
        raise FileExistsError(f"prepared image folder {directory} exists; pass overwrite=True to replace it")


def write_image(directory, *, identity, name, image, metadata, channels, rotation, entry, chain_results=None,
                application=None, attempts=None) -> dict:
    """Write one prepared image's folder; return the written record.

    identity holds dataset_id, sample_id, fov_id and subtile_id; channels the
    ChannelInfo records of the image's C order; rotation the rotation
    diagnostics (None without rotation); entry the image's
    registration_record["rounds"] entry. A registered round also passes its
    RegistrationResult list, its application WarpConfig and its attempts.
    """
    from starfinder.io._checkpoint import _atomic, _transform_json, write_json
    from starfinder.io.tiff import save_volume
    from starfinder.registration import BSplineTransform, DenseDisplacementTransform
    from starfinder.segmentation._import import _software
    from starfinder.segmentation._persist import file_sha256
    directory = Path(directory)
    for file_name in FILES:
        (directory / file_name).unlink(missing_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    path = directory / IMAGE_FILE
    with _atomic(path) as tmp:
        save_volume(image, tmp, metadata=metadata)
    registration = dict(entry, **dict.fromkeys(CHAIN_KEYS))
    field = None
    if chain_results is not None:
        arrays = {}
        for i, result in enumerate(chain_results):
            if isinstance(result.transform, (DenseDisplacementTransform, BSplineTransform)):
                arrays[f"result_{i}"] = (result.transform.displacement_zyx
                                         if isinstance(result.transform, DenseDisplacementTransform)
                                         else result.transform.coefficients)
        if arrays:
            with _atomic(directory / FIELD_FILE) as tmp:
                with open(tmp, "wb") as handle:
                    np.savez(handle, **arrays)
            field = {"path": FIELD_FILE, "file_sha256": file_sha256(directory / FIELD_FILE)}
        registration.update(transforms=[_transform_json(r, i, FIELD_FILE) for i, r in enumerate(chain_results)],
                            application_config=asdict(application), attempts=attempts, field=field)
    record = dict(format_version=FORMAT_VERSION, stage=STAGE, name=name, **identity,
                  channels=list(channels), rotation=rotation, registration=registration,
                  image={"path": IMAGE_FILE, "shape": list(image.shape), "dtype": str(image.dtype),
                         "metadata": asdict(metadata), "sha256": image_sha256(image),
                         "file_sha256": file_sha256(path)},
                  software=_software())
    write_json(record, directory / RECORD_FILE)
    return json.loads((directory / RECORD_FILE).read_text())


def read_image(directory, name, identity, channel_labels) -> dict:
    """Read one prepared image's folder, checking the recorded SHA-256 values.

    Returns the record, the image, its metadata, its registration_record["rounds"]
    entry and, for a registered round, the RegistrationResult list and the
    application WarpConfig (else None).
    ValueError for a record of another version or image, a missing file or
    one whose SHA-256 differs from the record (naming both hashes), stored
    metadata other than the recorded, an identity (dataset_id, sample_id,
    fov_id, subtile_id) other than identity, or channels other than
    channel_labels.
    """
    from starfinder.image import ImageMetadata
    from starfinder.io._checkpoint import _config, _field_loader, _registration_methods, _restore_result
    from starfinder.io.tiff import load_volume_zyxc
    from starfinder.registration import WarpConfig
    from starfinder.segmentation._persist import check_file
    directory = Path(directory)
    path = directory / RECORD_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no prepared image {name!r} at {path}")
    record = json.loads(path.read_text())
    if record.get("format_version") != FORMAT_VERSION or record.get("stage") != STAGE:
        raise ValueError(f"{path} is not a version {FORMAT_VERSION} prepared image record")
    if record.get("name") != name:
        raise ValueError(f"{path} records image {record.get('name')!r}, not {name!r}")
    for key, value in identity.items():
        if record.get(key) != value:
            raise ValueError(f"prepared image {name!r} at {path} has {key} {record.get(key)!r}, this FOV {value!r}")
    recorded = tuple(c["channel"] for c in record["channels"])
    if recorded != tuple(channel_labels):
        raise ValueError(f"prepared image {name!r} has the channels {list(recorded)}, the dataset "
                         f"{list(channel_labels)}")
    stored = record["image"]
    image_path = directory / stored["path"]
    check_file(image_path, stored["file_sha256"], "prepared image file")
    loaded = load_volume_zyxc(image_path, channel_labels=recorded)
    if image_sha256(loaded.image) != stored["sha256"]:
        raise ValueError(f"prepared image file {image_path} holds an image with SHA-256 "
                         f"{image_sha256(loaded.image)}, the record {stored['sha256']}")
    metadata = ImageMetadata(**stored["metadata"])
    if loaded.metadata != metadata:
        raise ValueError(f"prepared image file {image_path} stores metadata {loaded.metadata!r}, the record "
                         f"{metadata!r}")
    registration = record["registration"]
    results = application = None
    if registration.get("transforms") is not None:
        if registration.get("field") is not None:
            check_file(directory / registration["field"]["path"], registration["field"]["file_sha256"],
                       "prepared image field file")
        application = _config(registration["application_config"], WarpConfig)
        fields, methods = _field_loader(directory), _registration_methods()
        results = [_restore_result(e, i, fields, 2, application, methods)
                   for i, e in enumerate(registration["transforms"])]
    entry = {key: value for key, value in registration.items() if key not in CHAIN_KEYS}
    return dict(record=record, image=loaded.image, metadata=metadata, entry=entry, results=results,
                application=application)
