"""CW1 to CW6 and CR1 to CR4 of design/validation/compressed-image-outputs.md on hand-built fixtures (W-922).

The writers of the stage ``registered`` (I1, its snapshots I2), the reference image (I3) and the prepared images (I4)
take the codec ``image_compression``: ``zlib`` (deflate at level 6, no predictor; TIFF Compression 8), the default,
or ``none`` (Compression 1). The records name it, every reader takes either codec, and ``none`` writes the bytes of
v0.3.0 (``70edc72``). The comparisons with v0.3.0 run the base revision's package in a subprocess
(``image_compression_cases``) and skip, naming the reason, without Git or that revision (a shallow clone, as in CI).
The workflow's checks (the setting in the plan and the commands) are in test_workflow_image_compression.py.
"""
from dataclasses import asdict
import importlib.util
import json
import shutil
import warnings

import numpy as np
import pytest
import tifffile

from starfinder import _checkpoint
from starfinder.dataset import CheckpointConfig, _prepared
from starfinder.image import ImageMetadata
from starfinder.io import load_volume, load_volume_zyxc
from starfinder.io.tiff import _tiff_compression

from . import image_compression_cases as C

pytestmark = [pytest.mark.dataset, pytest.mark.io, pytest.mark.validation]

MV = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
SD = "the spatialdata and anndata extras (starfinder[spatialdata], starfinder[anndata]) are not installed"
#: The TIFF Compression tag of each setting (tifffile writes "zlib" as Adobe deflate, 8).
CODES = {"zlib": 8, "none": 1}


def tags(path):
    """(Compression, Predictor) of every page of a TIFF; the Predictor is 1 when the tag is absent."""
    with tifffile.TiffFile(path) as tif:
        return {(int(page.compression), int(page.tags["Predictor"].value) if "Predictor" in page.tags else 1)
                for page in tif.pages}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """``src/python`` of v0.3.0 (``base_reader``), or a skip naming why it is unavailable."""
    return C.base_source(tmp_path_factory.mktemp("base"))


# --- CW: writers and codecs ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("setting", ["zlib", "none", None])
def test_cw1_the_writers_of_i1_to_i4_write_their_codec(tmp_path, setting):
    """I1, I2, I3 (ZYX and the YX projection) and I4 of every dtype: Compression 8 and no predictor on every page by
    default (None) and with zlib, Compression 1 with none."""
    written = C.write_files(tmp_path, setting)
    assert len(written) == 5 * len(C.DTYPES)
    code = CODES[setting or "zlib"]
    for name in written:
        assert tags(tmp_path / name) == {(code, 1)}, name
        assert _tiff_compression(tmp_path / name) == (setting or "zlib")


def test_cw1_the_writers_that_compress_already_keep_zlib_and_the_others_stay_uncompressed(tmp_path):
    """I6 (labels), I7 and I9 (save_volume(compress=True) of a model input), I11 (the export's OME-TIFF writer) are
    Compression 8; I13 (the BigStitcher tiles) are Compression 1. None of these writers takes the codec setting."""
    from starfinder.io._ome import write_ome_tiff
    from starfinder.stitching import BigStitcherConfig, export_bigstitcher
    from .test_stitching_bigstitcher import golden_inputs, golden_table
    compressed = C.write_compressed(tmp_path / "compressed")
    for name in compressed:
        assert tags(tmp_path / "compressed" / name) == {(8, 1)}, name
    write_ome_tiff(tmp_path / "fused.ome.tif", C.volumes()["uint16-zyx"], metadata=C.metadata(), annotation={})
    assert {code for code, _ in tags(tmp_path / "fused.ome.tif")} == {8}
    table = golden_table(tmp_path)
    export_bigstitcher(table, golden_inputs(table), config=BigStitcherConfig(), tile_dir=tmp_path / "tiles",
                       save_dir=tmp_path / "tiles")
    tiles = sorted((tmp_path / "tiles").glob("tile_*.tif"))
    assert tiles and all(tags(path) == {(1, 1)} for path in tiles)


def test_cw1_the_synthetic_writers_stay_uncompressed(small_dataset):
    """I14: the session dataset's raw images, written by the synthetic writers, are Compression 1."""
    raw = sorted(small_dataset.rglob("*.tif"))
    assert raw and all({code for code, _ in tags(path)} == {1} for path in raw)


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=MV)
@pytest.mark.skipif(importlib.util.find_spec("spatialdata") is None or importlib.util.find_spec("anndata") is None,
                    reason=SD)
def test_cw1_the_zarr_outputs_keep_the_codec_of_v0_3_0(tmp_path, base):
    """I18 (the fused OME-Zarr: the codecs export.json records per level and in compression, and those of each
    level's array metadata) and I19 (the SpatialData store: each raster element's levels, array_metadata and its
    SHA-256, and the codecs that metadata names) equal those of v0.3.0 for the same input. The export takes no codec
    setting, so this holds in both."""
    new = C.write_zarr(tmp_path / "new")
    old = C.run_base(base, "write_zarr", root=str(tmp_path / "old"))
    assert len(new["fusion"]) == 2 and len(new["spatialdata"]) >= 2
    assert all(entry["levels"] == [entry["compression"]] for entry in new["fusion"])
    assert json.loads(json.dumps(new)) == old


@pytest.mark.parametrize("setting", ["zlib", "none"])
def test_cw2_each_reader_returns_the_written_array_and_metadata(tmp_path, setting):
    """write_registered_round, save_reference_image and _prepared.write_image, then their readers, for every dtype:
    the decoded identity (dtype, shape, C-order SHA-256) and the ImageMetadata of what was written."""
    from starfinder.dataset import Dataset, RoundState
    from starfinder.preprocessing import ProjectionConfig
    meta = C.metadata()
    for name, image in C.volumes().items():
        if image.ndim != 4:
            continue
        folder = tmp_path / name
        path = _checkpoint.write_registered_round(folder, "round1", image, meta, image_compression=setting)
        loaded = load_volume_zyxc(path)
        assert C.identity(loaded.image) == C.identity(image) and loaded.metadata == meta, name
        identity = dict(dataset_id="data", sample_id="sample", fov_id="FOV", subtile_id=None)
        record = _prepared.write_image(folder / "prepared", identity=identity, name="morph", image=image,
                                       metadata=meta, channels=[{"channel": "a"}, {"channel": "b"}], rotation=None,
                                       entry={}, image_compression=setting)
        read = _prepared.read_image(folder / "prepared", "morph", identity, ("a", "b"))
        assert C.identity(read["image"]) == C.identity(image) and read["metadata"] == meta, name
        assert record["image"]["compression"] == setting
        fov = Dataset(folder, folder / "out", "data", "sample", "run", RoundState(["round1"], reference_round="round1"),
                      ("a", "b")).fov("FOV")
        fov.images, fov.metadata = {"round1": image}, {"round1": meta}
        merged = fov.save_reference_image(image_compression=setting)
        loaded = load_volume(merged)
        assert C.identity(loaded.image) == C.identity(image.max(axis=3)) and loaded.metadata == meta, name
        projected = fov.save_reference_image(projection=ProjectionConfig(), image_compression=setting)
        plane = tifffile.imread(projected)
        assert C.identity(plane) == C.identity(image.max(axis=3).max(axis=0)), name
        with tifffile.TiffFile(projected) as tif:
            stored = json.loads(tif.pages[0].description)
        assert stored["axes"] == "YX" and ImageMetadata(**stored["starfinder_metadata"]) == meta.projected(method="max")


def test_cw3_the_opt_out_writes_the_bytes_of_v0_3_0(tmp_path, base):
    """With none, I1, I2, I3 (ZYX and YX) and I4 of every dtype are byte-equal to the files v0.3.0 writes for the
    same arrays and metadata, in this environment."""
    new = C.write_files(tmp_path / "new", "none")
    old = C.run_base(base, "write_files", root=str(tmp_path / "old"))
    assert len(new) == 5 * len(C.DTYPES)
    assert new == old


def test_cw4_the_writers_that_compress_today_keep_their_bytes(tmp_path, base):
    """save_volume(compress=True), now with its level 6 explicit, ZYX and ZYXC of every dtype (ZYXC also without
    metadata), and write_labels write the bytes v0.3.0 writes (tifffile's default level is 6 with the locked
    imagecodecs): the writers of I6 to I9."""
    new = C.write_compressed(tmp_path / "new")
    old = C.run_base(base, "write_compressed", root=str(tmp_path / "old"))
    assert len(new) == 3 * len(C.DTYPES) + 1
    assert new == old


@pytest.mark.contract
@pytest.mark.parametrize("value", ["zstd", "gzip", "", None, True])
def test_cw5_checkpoint_config_refuses_another_codec(value):
    with pytest.raises(ValueError, match=r"^image_compression must be zlib or none; got "):
        CheckpointConfig(image_compression=value)
    assert CheckpointConfig().image_compression == "zlib"


@pytest.mark.contract
def test_cw5_the_writers_refuse_another_codec(tmp_path):
    image = C.volumes()["uint8-zyxc"]
    with pytest.raises(ValueError, match="image_compression must be zlib or none; got 'zstd'"):
        _checkpoint.write_registered_round(tmp_path, "round1", image, C.metadata(), image_compression="zstd")
    fov = C._resident(tmp_path)
    with pytest.raises(ValueError, match="image_compression must be zlib or none; got 'deflate'"):
        fov.save_reference_image(image_compression="deflate")
    assert not any(tmp_path.rglob("*.tif"))


def test_cw6_the_default_is_zlib_level_6_without_a_predictor(tmp_path):
    """The default files equal tifffile.imwrite of the same array and metadata with compression zlib,
    compressionargs {"level": 6} and no predictor: the ZYX reference image, and the ZYXC round image, whose OME-XML
    UUID (a content UUID) is read back from the file."""
    meta = C.metadata()
    written = C.write_files(tmp_path / "default", None)
    for name, image in C.volumes().items():
        if image.ndim != 4:
            continue
        stage = tmp_path / "default" / name / "stage" / "registered" / "round1.ome.tif"
        reference = tmp_path / "default" / name / "reference-zyx" / "out" / "fovs" / "FOV" / "reference.tif"
        assert {f"{name}/stage/registered/round1.ome.tif", f"{name}/reference-zyx/out/fovs/FOV/reference.tif"} <= set(
            written)
        with tifffile.TiffFile(stage) as tif:
            uuid = tif.ome_metadata.split('UUID="', 1)[1].split('"', 1)[0]
        direct = tmp_path / "direct.ome.tif"
        tifffile.imwrite(direct, np.moveaxis(image, 3, 1), ome=True, compression="zlib",
                         compressionargs={"level": 6}, photometric="minisblack",
                         metadata={"axes": "ZCYX", "CommentAnnotation": {
                             "Namespace": "starfinder.image.ImageMetadata", "Value": json.dumps(asdict(meta))},
                             "UUID": uuid})
        assert stage.read_bytes() == direct.read_bytes(), name
        direct = tmp_path / "direct.tif"
        tifffile.imwrite(direct, image.max(axis=3), compression="zlib", compressionargs={"level": 6},
                         photometric="minisblack", metadata={"axes": "ZYX", "starfinder_metadata": asdict(meta)})
        assert reference.read_bytes() == direct.read_bytes(), name


# --- CR: records ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("setting", ["zlib", "none"])
def test_cr1_the_records_name_the_codec_of_each_file(tmp_path, setting):
    """run.json's config.checkpoints and outputs.reference_image, the stage record and registration.json name the
    setting, which is the codec read from each file's tags; the stage record keeps format version 3 and files
    entries of exactly {path, sha256}, the prepared record format version 1."""
    folder = C.write_stage(tmp_path, setting)
    run = json.loads((folder / "run.json").read_text())
    header = json.loads((folder / "checkpoints" / "registered" / "transforms.json").read_text())
    prepared = json.loads((folder / "other_rounds" / C.OTHER / "registration.json").read_text())
    assert run["config"]["checkpoints"]["image_compression"] == setting
    assert header["image_compression"] == setting and _checkpoint.stage_compression(header) == setting
    assert run["outputs"]["reference_image"]["compression"] == setting
    assert prepared["image"]["compression"] == setting
    assert header["format_version"] == 3 and prepared["format_version"] == 1
    assert all(set(entry) == {"path", "sha256"} for entry in header["files"])
    images = [folder / "checkpoints" / "registered" / entry["path"] for entry in header["files"]
              if entry["path"].endswith(".ome.tif")]
    assert len(images) == len(C.ROUNDS)
    for path in [*images, folder / run["outputs"]["reference_image"]["path"],
                 folder / "other_rounds" / C.OTHER / prepared["image"]["path"]]:
        assert _tiff_compression(path) == setting, path


@pytest.mark.parametrize("written_by", ["key removed", "v0.3.0"])
def test_cr2_records_without_the_key_load_as_uncompressed_without_a_warning(tmp_path, written_by, request):
    """A stage record and a registration.json without the codec key load; the stage reader takes the stage as
    uncompressed; the images are the ones a run with none writes."""
    expected = C.read_stage(C.write_stage(tmp_path / "none", "none").parents[2])
    if written_by == "key removed":
        root = tmp_path / "removed"
        shutil.copytree(tmp_path / "none", root)
        folder = root / "out" / "fovs" / "FOV"
        for path, keys in ((folder / "checkpoints" / "registered" / "transforms.json", ()),
                           (folder / "other_rounds" / C.OTHER / "registration.json", ("image",))):
            record = json.loads(path.read_text())
            (record[keys[0]] if keys else record).pop("compression" if keys else "image_compression")
            path.write_text(json.dumps(record))
    else:
        root = tmp_path / "old"
        C.run_base(request.getfixturevalue("base"), "write_stage", root=str(root))
        folder = root / "out" / "fovs" / "FOV"
    header = json.loads((folder / "checkpoints" / "registered" / "transforms.json").read_text())
    prepared = json.loads((folder / "other_rounds" / C.OTHER / "registration.json").read_text())
    assert "image_compression" not in header and "compression" not in prepared["image"]
    assert _checkpoint.stage_compression(header) == "none"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert C.read_stage(root) == expected


def test_cr3_v0_3_0_reads_a_compressed_stage(tmp_path, base):
    """At 70edc72, FOV.load_checkpoint("registered") (payload_entries included) restores a stage written now with
    zlib, and FOV.load_registered_round reads a prepared image written now: the arrays and metadata written."""
    C.write_stage(tmp_path, "zlib")
    expected = json.loads(json.dumps(C.read_stage(tmp_path)))
    assert C.run_base(base, "read_stage", root=str(tmp_path)) == expected


@pytest.mark.parametrize("setting", ["zlib", "none"])
def test_cr4_a_changed_byte_of_a_round_image_refuses_the_stage(tmp_path, setting):
    """A byte flipped in a round image (inside the compressed data with zlib) refuses the stage's read with the
    file's SHA-256 mismatch, as for an uncompressed one."""
    folder = C.write_stage(tmp_path, setting)
    path = folder / "checkpoints" / "registered" / "round2.ome.tif"
    with tifffile.TiffFile(path) as tif:
        offset = tif.pages[0].dataoffsets[0]
    data = bytearray(path.read_bytes())
    data[offset] ^= 0xFF
    path.write_bytes(bytes(data))
    fov = C._dataset(tmp_path, other=True).fov("FOV")
    with pytest.raises(ValueError, match=r"registered checkpoint file .*round2\.ome\.tif has SHA-256 [0-9a-f]{64}, "
                                         r"the record [0-9a-f]{64}"):
        fov.load_checkpoint("registered")
