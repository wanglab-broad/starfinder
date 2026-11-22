"""Writers and readers of the compressed image outputs, run at this revision and at the base revision (W-909, W-922).

design/validation/compressed-image-outputs.md compares the files the package writes now with those of v0.3.0
(``70edc72``, ``BASE``) for the same arrays and metadata, and checks that v0.3.0 reads a stage written now. This module
holds both sides' code. It imports only the package's public API and the private writers that both revisions have, so
that a test runs it in this process and, in a subprocess whose ``PYTHONPATH`` starts with the base revision's
``src/python``, as a script (``python image_compression_cases.py <command> <JSON arguments>``, which prints one JSON
line). ``image_compression`` is passed only when it is not None: the base revision has no such argument and writes
every one of these files uncompressed.

The fixtures are the page's ``volumes`` (ZYXC 8 × 64 × 64 × 2 and ZYX 8 × 64 × 64 of uint8, uint16, uint32, float32
and float64, zeros with Gaussian spots, spacing (1.0, 0.1, 0.1) µm) and a FOV of two sequencing rounds and one other
round built from the uint16 volume, held resident (no input file is read).
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

#: The base revision, v0.3.0, whose files the opt-out reproduces (CW3) and whose reader loads a new stage (CR3).
BASE = "70edc72f5e8ecf16f48250159a8a22096383c846"
DTYPES = ("uint8", "uint16", "uint32", "float32", "float64")
SHAPE_ZYXC = (8, 64, 64, 2)
ROUNDS = ("round1", "round2")
CHANNELS = ("a", "b")
OTHER = "morph"
SNAPSHOT = "smoothed"


def metadata():
    from starfinder.image import ImageMetadata
    return ImageMetadata("compression-cases", spacing_zyx=(1.0, 0.1, 0.1), origin_zyx=(0.0, 0.0, 0.0),
                         direction_zyx=((1, 0, 0), (0, 1, 0), (0, 0, 1)), spatial_unit="um")


def _spots():
    """A float64 ZYXC image of zeros with 12 Gaussian spots (sigma 1.5 voxels in YX, 1 in Z), peaks in (0.5, 1]."""
    rng = np.random.default_rng(909)
    z, y, x = np.meshgrid(*(np.arange(n) for n in SHAPE_ZYXC[:3]), indexing="ij")
    image = np.zeros(SHAPE_ZYXC)
    for _ in range(12):
        cz, cy, cx = rng.uniform(1, SHAPE_ZYXC[0] - 1), rng.uniform(4, 60), rng.uniform(4, 60)
        c, peak = rng.integers(0, SHAPE_ZYXC[3]), rng.uniform(0.5, 1.0)
        image[..., c] += peak * np.exp(-((z - cz) ** 2 / 2 + ((y - cy) ** 2 + (x - cx) ** 2) / (2 * 1.5 ** 2)))
    # Below 1e-3 of a peak is zero, so that most voxels are zero and zlib shortens the file.
    image[image < 1e-3] = 0.0
    return image


def volumes():
    """{"<dtype>-zyxc" or "<dtype>-zyx": array}: the sparse image in each dtype (integers scaled to most of their
    range), and its first channel as ZYX."""
    spots = _spots()
    out = {}
    for dtype in DTYPES:
        kind = np.dtype(dtype)
        array = (spots * (min(np.iinfo(kind).max, 60000))).astype(kind) if kind.kind == "u" else spots.astype(kind)
        out[f"{dtype}-zyxc"] = array
        out[f"{dtype}-zyx"] = np.ascontiguousarray(array[..., 0])
    return out


def identity(array):
    """The decoded identity of an array: dtype, shape and the SHA-256 of its C-order bytes."""
    array = np.asarray(array)
    return [str(array.dtype), list(array.shape), hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _codec(image_compression):
    return {} if image_compression is None else {"image_compression": image_compression}


def _dataset(root, *, other=False):
    from starfinder.dataset import Dataset, RoundState
    rounds = RoundState(list(ROUNDS), [OTHER] if other else [], "round1")
    return Dataset(Path(root) / "in", Path(root) / "out", "data", "sample", "run", rounds, CHANNELS)


def write_files(root, image_compression=None):
    """I1, I2, I3 (ZYX and the YX maximum projection) and I4 of every ZYXC volume under ``root``; {path relative to
    root: SHA-256 of the file}."""
    from starfinder import _checkpoint
    from starfinder.dataset import _prepared
    from starfinder.preprocessing import ProjectionConfig
    root = Path(root)
    meta = metadata()
    written = []
    for name, image in volumes().items():
        if image.ndim != 4:
            continue
        folder = root / name
        written.append(_checkpoint.write_registered_round(folder / "stage", "round1", image, meta,
                                                          **_codec(image_compression)))
        written.append(_checkpoint.write_registered_round(folder / "stage", "round1", image, meta, snapshot=SNAPSHOT,
                                                          **_codec(image_compression)))
        _prepared.write_image(folder / "prepared", identity=dict(dataset_id="data", sample_id="sample", fov_id="FOV",
                                                                 subtile_id=None),
                              name=OTHER, image=image, metadata=meta, channels=[], rotation=None, entry={},
                              **_codec(image_compression))
        written.append(folder / "prepared" / _prepared.IMAGE_FILE)
        for view, projection in (("zyx", None), ("yx", ProjectionConfig())):
            fov = _dataset(folder / f"reference-{view}").fov("FOV")
            fov.images, fov.metadata = {"round1": image}, {"round1": meta}
            written.append(fov.save_reference_image(projection=projection, **_codec(image_compression)))
    return {Path(path).relative_to(root).as_posix(): file_sha256(path) for path in written}


def write_compressed(root):
    """The writers that compress at both revisions (CW4): ``save_volume(compress=True)`` of every volume, ZYX and ZYXC
    (the segmentation input I7 is this call on a ZYXC image), ZYXC without metadata (the preview's model inputs, I9),
    and ``segmentation._persist.write_labels`` (I6, and I8 through ``assignment._persist``); {path: SHA-256}."""
    from starfinder.io import save_volume
    from starfinder.segmentation._persist import write_labels
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    meta = metadata()
    written = []
    for name, image in volumes().items():
        path = root / (f"{name}.ome.tif" if image.ndim == 4 else f"{name}.tif")
        save_volume(image, path, compress=True, metadata=meta)
        written.append(path)
        if image.ndim == 4:
            path = root / f"{name}-without-metadata.ome.tif"
            save_volume(image, path, compress=True)
            written.append(path)
    labels = (volumes()["uint16-zyx"] > 1000).astype(np.uint32) * np.arange(1, 9, dtype=np.uint32)[:, None, None]
    write_labels(root / "labels.tif", labels, meta)
    written.append(root / "labels.tif")
    return {Path(path).relative_to(root).as_posix(): file_sha256(path) for path in written}


def _resident(root, *, other=True):
    """The FOV of two sequencing rounds and one other round, resident: round1 the uint16 volume, round2 it shifted by
    two voxels in X, the other round the volume itself."""
    image = volumes()["uint16-zyxc"]
    fov = _dataset(root, other=other).fov("FOV")
    fov.images = {"round1": image, "round2": np.roll(image, 2, axis=2), OTHER: image.copy()}
    fov.metadata = {name: metadata() for name in fov.images}
    return fov


def write_stage(root, image_compression=None):
    """FOV.run with the stage registered (translation registration), FOV.save_reference_image and FOV.register of the
    other round with checkpoints; the FOV folder."""
    from starfinder.dataset import CheckpointConfig, PipelineConfig
    from starfinder.registration import RegistrationRecipe, RegistrationStep, TranslationConfig
    recipe = RegistrationRecipe((RegistrationStep(TranslationConfig()),))
    checkpoints = CheckpointConfig(stages=("registered",), **_codec(image_compression))
    fov = _resident(root)
    fov.run(PipelineConfig(registration=recipe), checkpoints=checkpoints)
    fov.save_reference_image(**_codec(image_compression))
    fov.register(recipe, rounds=[OTHER], checkpoints=checkpoints)
    return fov.paths.fov_dir


def _zarr_codecs(array):
    """The codec list a Zarr array's metadata names: ``codecs`` of format 3, ``compressor`` and ``filters`` of 2."""
    array = Path(array)
    if (array / "zarr.json").is_file():
        return json.loads((array / "zarr.json").read_text())["codecs"]
    metadata = json.loads((array / ".zarray").read_text())
    return {"compressor": metadata["compressor"], "filters": metadata["filters"]}


def write_zarr(root):
    """The export's two Zarr outputs, which take no codec setting (CW1): I18, the standalone OME-Zarr of
    ``fusion_fixtures.quad`` fused as projection and stack (each entry's ``compression``, ``levels[].codec`` and the
    codecs its level arrays' metadata names), and I19, the SpatialData store of ``fusion_fixtures.export_folder`` (its
    ``spatialdata`` section's raster elements: ``levels``, ``array_metadata``, ``array_metadata_sha256`` and the
    codecs that array metadata names). The test package's fixtures are the running revision's."""
    import dask
    from starfinder.export import ExportConfig, FusionConfig, SpatialDataExportConfig, fuse_images
    from test.assembly_fixtures import SAMPLE
    from test.fusion_fixtures import export_folder, quad
    root = Path(root)
    with dask.config.set(scheduler="synchronous"):
        case = quad()
        section = fuse_images(case.placement, case.inputs(), output_dir=root / "fusion",
                              config=FusionConfig(dimensionality="both", scale_yx=1, ome_zarr=True)).section
        folder = export_folder(root / "export")
        folder.dataset.export(SAMPLE, placement=folder.placement_dir, output_dir=folder.export_dir,
                              config=ExportConfig(route="per_fov", export_spatialdata=True, fuse_labels=True,
                                                  spatialdata=SpatialDataExportConfig(zarr_format=3)))
    fusion = [{"path": entry["path"], "compression": entry["compression"],
               "levels": [level["codec"] for level in entry["levels"]],
               "stored": [_zarr_codecs(root / "fusion" / entry["path"] / str(i)) for i in range(len(entry["levels"]))]}
              for entry in section["outputs"] if entry["format"] == "ome_zarr"]
    record = json.loads((folder.export_dir / "export.json").read_text())
    spatialdata = [{"name": entry["name"], "levels": entry["levels"], "array_metadata": entry["array_metadata"],
                    "array_metadata_sha256": entry["array_metadata_sha256"],
                    "stored": _zarr_codecs((folder.export_dir / entry["array_metadata"]).parent)}
                   for entry in record["spatialdata"]["elements"]
                   if entry["kind"] in ("image", "labels") and entry["status"] != "omitted"]
    return {"fusion": fusion, "spatialdata": spatialdata}


#: The code identity of run_workflow's runs, workflow_fixture.CLEAN_CODE: a saved set or a resume refuses a dirty one,
#: and the runs do not depend on the state of the checkout.
CLEAN_CODE = {"version": "0.2.0", "version_source": "checkout", "git_commit": "0123456789abcdef0123456789abcdef01234567",
              "git_dirty": False}


def run_workflow(config, tasks):
    """Plan the workflow configuration file ``config`` and run each of ``tasks`` (ids) by its command in this process,
    with the code identity CLEAN_CODE; {task: exit status}."""
    import starfinder._records as records
    from starfinder.__main__ import main
    from starfinder.workflow import plan
    from starfinder.workflow._shape import COMMANDS
    records.code = lambda: dict(CLEAN_CODE)
    manifest = plan(config)
    path = Path(manifest["roots"]["output"]) / "workflow" / "manifest.json"
    kinds = {task["id"]: task["kind"] for task in manifest["tasks"]}
    return {task: main(["workflow", COMMANDS[kinds[task]], "--manifest", str(path), "--task", task]) for task in tasks}


def read_stage(root):
    """FOV.load_checkpoint("registered") and FOV.load_registered_round of the other round on a new FOV object of the
    stage write_stage wrote; {round: decoded identity}, and "metadata": each image's ImageMetadata."""
    fov = _dataset(root, other=True).fov("FOV")
    fov.load_checkpoint("registered")
    fov.load_registered_round(OTHER)
    images = {name: identity(fov.images[name]) for name in (*ROUNDS, OTHER)}
    return dict(images, metadata={name: asdict(fov.metadata[name]) for name in (*ROUNDS, OTHER)})


# --- the base reader (the tests' side) -----------------------------------------------------------------------------

#: The repository whose Git history holds BASE.
REPOSITORY = Path(__file__).resolve().parents[3]


def base_source(folder):
    """``src/python`` of BASE, whose tree (about 34 MB) is extracted with ``git archive`` from this clone into
    ``folder``; a skip naming the reason when Git or the revision is absent (a shallow clone, as in CI)."""
    import io
    import shutil
    import subprocess
    import tarfile
    import pytest
    git = shutil.which("git")
    if git is None:
        pytest.skip(f"the comparison with v0.3.0 ({BASE[:7]}) needs Git on PATH, which this environment lacks")
    found = subprocess.run([git, "-C", str(REPOSITORY), "cat-file", "-e", f"{BASE}^{{commit}}"], capture_output=True)
    if found.returncode != 0:
        pytest.skip(f"the comparison with v0.3.0 needs the revision {BASE[:7]}, which this clone does not hold "
                    "(a shallow clone)")
    archive = subprocess.run([git, "-C", str(REPOSITORY), "archive", "--format=tar", BASE], capture_output=True,
                             check=True)
    folder = Path(folder)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(folder, filter="data")
    return folder / "src" / "python"


def run_base(source, command, **arguments):
    """``command`` of this module run by the package at ``source`` (base_source) in a subprocess with one numerical
    thread; its result. The subprocess must have imported the package from ``source``."""
    import os
    import subprocess
    from starfinder._execution import THREAD_VARIABLES
    source = Path(source).resolve()
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1",
                       **dict.fromkeys(THREAD_VARIABLES, "1"))
    done = subprocess.run([sys.executable, __file__, command, json.dumps(arguments, default=str)], cwd=source,
                          env=environment, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    output = json.loads(done.stdout.splitlines()[-1])
    assert Path(output["package"]) == (Path(source) / "starfinder").resolve(), output["package"]
    return output["result"]


def main(argv):
    """``<command> <JSON arguments>``: run write_files, write_compressed, write_stage, read_stage, run_workflow or
    write_zarr and print its result as JSON, with the path of the package that ran it."""
    import starfinder
    command, arguments = argv[0], json.loads(argv[1])
    commands = {"write_files": write_files, "write_compressed": write_compressed, "read_stage": read_stage,
                "write_stage": lambda **a: str(write_stage(**a)), "run_workflow": run_workflow, "write_zarr": write_zarr}
    result = commands[command](**arguments)
    print(json.dumps({"package": str(Path(starfinder.__file__).resolve().parent), "result": result}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
