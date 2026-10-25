"""W-433: the grid of a reference image without stored calibration is read from its TIFF header.

``starfinder.dataset._declared.reference_grid`` takes the shape of a file that stores no
``starfinder_metadata`` from its header (``starfinder.io.tiff._volume_header``) and reads no
pixel; a file with stored calibration, whose grid holds the SHA-256 of its pixels, and a file
whose header does not settle the shape ``load_volume`` gives are read whole as before.

The expected outcomes are explicit (W-574): per layout the ZYX shape or the refusal, and the pages
decoded (``EXPECTED_GRIDS``); per voxel setting the declared metadata or the sizes it lacks
(``DECLARED``, ``LACKS``); for the callers the declared grids of their samples. They hold what the
copied base-revision function (``ec4774a``) these tests compared with gave. Fixtures, all written in
the test: one small synthetic file per layout ``load_volume`` reads or refuses (``LAYOUTS``), and the
plain copy of W-390's synthetic sample scene (``sample-tiny-v1``, 2 × 2 FOVs of 5×36×36, its
reference images written as pages without metadata) for the export. Pages decoded are counted by
wrapping tifffile's pixel readers (``PixelReads``).
"""
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starfinder.dataset._declared import reference_grid, reference_volume
from starfinder.image import ImageMetadata, InvalidImageError
from starfinder.io import load_volume, save_volume
from starfinder.segmentation import ReferenceGrid

pytestmark = [pytest.mark.dataset, pytest.mark.io, pytest.mark.contract]
DATASET_ID, OUTPUT_ID, FOV = "data", "output", "FOV_001"
VOXELS = {"given": {"voxel_size_xy": 0.5, "voxel_size_z": 1.5}, "missing": {},
          "xy only": {"voxel_size_xy": 0.5}, "z only": {"voxel_size_z": 1.5}}
STORED = ImageMetadata("stored-frame", spacing_zyx=(2.0, 0.25, 0.25), spatial_unit="micrometer")
RNG = np.random.default_rng(433)


def pixels(shape, dtype=np.uint16):
    return RNG.integers(0, 1000, size=shape).astype(dtype)


def _two_series(path):
    with tifffile.TiffWriter(path) as tif:
        tif.write(pixels((2, 6, 6)), photometric="minisblack")
        tif.write(pixels((3, 6, 6)), photometric="minisblack")


def _write(kind):
    """A writer of one layout: (path) -> None."""
    def plain(shape, dtype=np.uint16, **kwargs):
        return lambda path: tifffile.imwrite(path, pixels(shape, dtype), metadata=None, photometric="minisblack",
                                             **kwargs)
    return {
        # Files that store no calibration and whose header settles the shape: no pixel is read.
        "ZYX stack (axes ZYX, save_volume without metadata)": lambda p: save_volume(pixels((5, 12, 10)), p),
        "ZYX stack (ImageJ)": lambda p: tifffile.imwrite(p, pixels((5, 12, 10)), imagej=True,
                                                         metadata={"axes": "ZYX"}),
        "ZYX stack (OME, singleton channel)": lambda p: tifffile.imwrite(p, pixels((1, 5, 12, 10)), ome=True,
                                                                         photometric="minisblack",
                                                                         metadata={"axes": "CZYX"}),
        "ZYX stack (shaped pages without axes, QYX)": lambda p: tifffile.imwrite(p, pixels((5, 12, 10)),
                                                                                 photometric="minisblack"),
        "single plane (YX, no metadata)": plain((12, 10)),
        "single plane (ZYX of one page)": lambda p: save_volume(pixels((1, 12, 10)), p),
        "paged without axis metadata (IYX, generic)": plain((5, 12, 10)),
        "paged without axis metadata (IYX, uniform)": plain((8, 48, 64)),
        "paged without axis metadata, zlib": plain((5, 32, 32), compression="zlib"),
        "paged without axis metadata, tiled": plain((3, 32, 32), tile=(16, 16)),
        "paged without axis metadata, int16": plain((5, 12, 10), np.int16),
        "four axes without metadata (read as ZYX pages)": plain((2, 3, 12, 10)),
        "RGB without metadata (YXS, read as ZYX)": lambda p: tifffile.imwrite(p, pixels((12, 10, 3), np.uint8),
                                                                              metadata=None, photometric="rgb"),
        # Files with stored calibration: read whole, as before (the grid holds the pixels' SHA-256).
        "stored calibration, ZYX": lambda p: save_volume(pixels((5, 12, 10)), p, metadata=STORED),
        "stored calibration, one plane": lambda p: save_volume(pixels((1, 12, 10)), p, metadata=STORED),
        # Files whose header does not settle what load_volume gives: read whole, as before.
        "float32 pages (finiteness needs the pixels)": plain((5, 12, 10), np.float32),
        "float64 pages (finiteness needs the pixels)": plain((5, 12, 10), np.float64),
        "float32 pages with NaN (refused)": lambda p: tifffile.imwrite(
            p, np.full((5, 12, 10), np.nan, np.float32), metadata=None, photometric="minisblack"),
        "ImageJ CYX, three channels (refused)": lambda p: tifffile.imwrite(p, pixels((3, 12, 10)), imagej=True),
        "OME CZYX, two channels (refused)": lambda p: tifffile.imwrite(p, pixels((2, 5, 12, 10)), ome=True,
                                                                       photometric="minisblack",
                                                                       metadata={"axes": "CZYX"}),
        "declared CZYX, two channels (refused)": lambda p: tifffile.imwrite(p, pixels((2, 5, 12, 10)),
                                                                            photometric="minisblack",
                                                                            metadata={"axes": "CZYX"}),
        "bool pages (refused)": plain((5, 12, 10), bool),
        "two series (refused)": _two_series,
    }[kind]


SETTLED = ("ZYX stack (axes ZYX, save_volume without metadata)", "ZYX stack (ImageJ)",
           "ZYX stack (OME, singleton channel)", "ZYX stack (shaped pages without axes, QYX)",
           "single plane (YX, no metadata)", "single plane (ZYX of one page)",
           "paged without axis metadata (IYX, generic)", "paged without axis metadata (IYX, uniform)",
           "paged without axis metadata, zlib", "paged without axis metadata, tiled",
           "paged without axis metadata, int16", "four axes without metadata (read as ZYX pages)",
           "RGB without metadata (YXS, read as ZYX)")
WITH_CALIBRATION = ("stored calibration, ZYX", "stored calibration, one plane")
# Floating-point files: load_volume refuses a non-finite value, and only the pixels show it.
FINITE_FLOAT = ("float32 pages (finiteness needs the pixels)", "float64 pages (finiteness needs the pixels)")
FLOAT = FINITE_FLOAT + ("float32 pages with NaN (refused)",)
REFUSED = ("ImageJ CYX, three channels (refused)", "OME CZYX, two channels (refused)",
           "declared CZYX, two channels (refused)", "bool pages (refused)", "two series (refused)")
UNSETTLED = FLOAT + REFUSED
LAYOUTS = SETTLED + WITH_CALIBRATION + UNSETTLED

NOT_FINITE = (InvalidImageError, "image must contain finite real numeric values")
AMBIGUOUS_C = (ValueError, "ambiguous C; select an explicit index")
#: Per layout: the ZYX shape of the grid, or load_volume's refusal (exception type, message) for every voxel setting;
#: and the pages reference_grid decodes, all in one whole read of the file (0: no read).
EXPECTED_GRIDS = {
    "ZYX stack (axes ZYX, save_volume without metadata)": ((5, 12, 10), 0),
    "ZYX stack (ImageJ)": ((5, 12, 10), 0),
    "ZYX stack (OME, singleton channel)": ((5, 12, 10), 0),
    "ZYX stack (shaped pages without axes, QYX)": ((5, 12, 10), 0),
    "single plane (YX, no metadata)": ((1, 12, 10), 0),
    "single plane (ZYX of one page)": ((1, 12, 10), 0),
    "paged without axis metadata (IYX, generic)": ((5, 12, 10), 0),
    "paged without axis metadata (IYX, uniform)": ((8, 48, 64), 0),
    "paged without axis metadata, zlib": ((5, 32, 32), 0),
    "paged without axis metadata, tiled": ((3, 32, 32), 0),
    "paged without axis metadata, int16": ((5, 12, 10), 0),
    "four axes without metadata (read as ZYX pages)": ((6, 12, 10), 0),
    "RGB without metadata (YXS, read as ZYX)": ((12, 10, 3), 0),
    "stored calibration, ZYX": ((5, 12, 10), 5),
    "stored calibration, one plane": ((1, 12, 10), 1),
    "float32 pages (finiteness needs the pixels)": ((5, 12, 10), 5),
    "float64 pages (finiteness needs the pixels)": ((5, 12, 10), 5),
    "float32 pages with NaN (refused)": (NOT_FINITE, 5),
    "ImageJ CYX, three channels (refused)": (AMBIGUOUS_C, 3),
    "OME CZYX, two channels (refused)": (AMBIGUOUS_C, 10),
    "declared CZYX, two channels (refused)": (AMBIGUOUS_C, 10),
    "bool pages (refused)": (NOT_FINITE, 5),
    "two series (refused)": ((ValueError, "ambiguous series; select an explicit index"), 0),
}
#: The metadata a file without stored calibration declares with both voxel sizes; the sizes each setting lacks.
DECLARED = ImageMetadata(f"declared:{DATASET_ID}/{OUTPUT_ID}/{FOV}", spacing_zyx=(1.5, 0.5, 0.5),
                         spatial_unit="micrometer")
LACKS = {"given": None, "missing": "voxel_size_xy and voxel_size_z", "xy only": "voxel_size_z",
         "z only": "voxel_size_xy"}
NO_CALIBRATION = ("{path} stores no calibration (no starfinder_metadata description), so its grid is declared by the "
                  "dataset configuration, which lacks {lacks}: set {lacks} (µm) in the workflow configuration, or the "
                  "Dataset fields of the same names")


def write_layout(root, kind):
    path = Path(root) / "fovs" / FOV / "reference.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(kind)(path)
    return path


class PixelReads:
    """The pages tifffile decodes while active: a whole-series read counts its pages, a page read one.

    Wraps ``TiffFile.asarray`` (series reads, ``imread``), ``TiffPage.asarray``,
    ``TiffFrame.asarray``, ``TiffPage.segments`` (tiled and compressed pages)
    and ``FileCache.read`` (a chunk of tifffile's Zarr view, which counts one);
    a read nested in another is not counted twice. ``whole`` lists the paths
    read whole (``TiffFile.asarray`` without a page key).
    """

    def __init__(self, monkeypatch):
        self.pages, self.whole, self._depth = 0, [], 0
        for owner, name in ((tifffile.TiffFile, "asarray"), (tifffile.TiffPage, "asarray"),
                            (tifffile.TiffFrame, "asarray"), (tifffile.TiffPage, "segments"),
                            (tifffile.FileCache, "read")):
            monkeypatch.setattr(owner, name, self._wrap(owner, name, getattr(owner, name)))

    def _wrap(self, owner, name, real):
        def wrapped(obj, *args, **kwargs):
            if self._depth == 0:
                if owner is tifffile.TiffFile:
                    series = kwargs.get("series")
                    series = obj.series[series or 0] if not isinstance(series, tifffile.TiffPageSeries) else series
                    key = args[0] if args else kwargs.get("key")
                    self.pages += len(series.pages) if key is None else len(np.atleast_1d(np.arange(
                        len(series.pages))[key]))
                    if key is None:
                        self.whole.append(Path(obj.filehandle.path).resolve())
                else:
                    self.pages += 1
            self._depth += 1
            try:
                return real(obj, *args, **kwargs)
            finally:
                self._depth -= 1
        return wrapped


def n_pages(path):
    with tifffile.TiffFile(path) as tif:
        return len(tif.pages)


@contextmanager
def counted(monkeypatch):
    with monkeypatch.context() as patch:
        yield PixelReads(patch)


def outcome(function, root, **voxels):
    """The grid, or (exception type, message), of ``function`` on the FOV of ``root``."""
    try:
        return function(root, FOV, dataset_id=DATASET_ID, output_id=OUTPUT_ID, **voxels)
    except Exception as error:  # noqa: BLE001 - compared as type and message
        return type(error), str(error)


# --- The same grid ------------------------------------------------------------------------------------------------

def expected_outcome(kind, voxels, path):
    """(shape, metadata, source, holds a hash) of the expected grid, or the expected (exception type, message)."""
    shape = EXPECTED_GRIDS[kind][0]
    if kind in WITH_CALIBRATION:
        return shape, STORED, f"file:{path.resolve()}", True
    if kind not in SETTLED + FINITE_FLOAT:
        return shape
    if LACKS[voxels] is not None:
        return ValueError, NO_CALIBRATION.format(path=path, lacks=LACKS[voxels])
    return shape, DECLARED, "declared", False


@pytest.mark.parametrize("voxels", list(VOXELS))
@pytest.mark.parametrize("kind", LAYOUTS)
def test_the_grid_equals_the_base_revisions(tmp_path, kind, voxels):
    """Every field of the grid (shape, metadata, source, whether it holds a hash), or the exception type and message,
    as stated in EXPECTED_GRIDS, DECLARED and LACKS; a grid equals reference_volume's."""
    path = write_layout(tmp_path, kind)
    result = outcome(reference_grid, tmp_path, **VOXELS[voxels])
    if isinstance(result, ReferenceGrid):
        assert (result.shape_zyx, result.metadata, result.source, result.sha256 is not None) == expected_outcome(
            kind, voxels, path)
        assert result == reference_volume(tmp_path, FOV, dataset_id=DATASET_ID, output_id=OUTPUT_ID,
                                          **VOXELS[voxels])[0]
    else:
        assert result == expected_outcome(kind, voxels, path)


def test_the_header_shape_is_the_loaded_shape(tmp_path):
    """The header reader gives load_volume's shape and metadata source where it settles them, and None elsewhere."""
    from starfinder.io.tiff import _volume_header
    for kind in LAYOUTS:
        path = write_layout(tmp_path / kind.replace(" ", "_"), kind)
        header = _volume_header(path)
        if kind in SETTLED + WITH_CALIBRATION:
            loaded = load_volume(path)
            assert header == (loaded.image.shape, loaded.diagnostics["metadata_source"] == "stored"), kind
        else:
            assert header is None or header[0] is None, kind


def rchar():
    with open("/proc/self/io") as handle:
        return next(int(line.split()[1]) for line in handle if line.startswith("rchar"))


@pytest.mark.skipif(not Path("/proc/self/io").is_file(), reason="needs the Linux I/O counters of /proc/self/io")
def test_the_header_reader_reads_only_the_header(tmp_path):
    """The bytes read (rchar) by the header reader of a file of 30 pages in 64 strips each are a fraction of the file:
    it reads without a buffer, where a buffered file reads a block at each seek to a page (more than the whole file
    here, and 1 MiB per seek on the SMB share)."""
    from starfinder.io.tiff import _volume_header
    path = tmp_path / "strips.tif"
    tifffile.imwrite(path, pixels((30, 64, 64), np.uint8), metadata=None, photometric="minisblack",
                     compression="packbits", rowsperstrip=1)
    before = rchar()
    assert _volume_header(path) == ((30, 64, 64), False)
    read = rchar() - before
    before = rchar()
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape == (30, 64, 64)
    buffered = rchar() - before
    assert read < path.stat().st_size / 4 < buffered


@pytest.mark.parametrize("voxels", ["given", "missing"])
def test_a_missing_file_raises_as_before(tmp_path, voxels):
    path = tmp_path / "fovs" / FOV / "reference.tif"
    assert outcome(reference_grid, tmp_path, **VOXELS[voxels]) == (
        FileNotFoundError, f"the reference image of FOV {FOV!r} is missing: {path} (the file the setting "
                           "reference_image 'fovs/{fov_id}/reference.tif' names)")


# --- No pixel is read ---------------------------------------------------------------------------------------------

def page_reads(monkeypatch, root, voxels):
    """(pages decoded, paths read whole) by reference_grid on the layout's file."""
    with counted(monkeypatch) as reads:
        outcome(reference_grid, root, **VOXELS[voxels])
    return reads.pages, reads.whole


def expected_reads(kind, path):
    """The pages EXPECTED_GRIDS states, in one whole read of the file, or no read."""
    pages = EXPECTED_GRIDS[kind][1]
    return pages, [path.resolve()] if pages else []


@pytest.mark.parametrize("voxels", ["given", "missing"])
@pytest.mark.parametrize("kind", SETTLED)
def test_no_page_is_decoded_for_an_integer_file_without_calibration(tmp_path, monkeypatch, kind, voxels):
    """0 pages decoded and no whole read for a file of an integer dtype without stored calibration whose header
    settles the shape (also when the voxel sizes are missing and it raises)."""
    path = write_layout(tmp_path, kind)
    assert load_volume(path).image.dtype.kind in "ui"
    assert page_reads(monkeypatch, tmp_path, voxels) == expected_reads(kind, path) == (0, [])


@pytest.mark.parametrize("voxels", ["given", "missing"])
@pytest.mark.parametrize("kind", FLOAT)
def test_a_floating_point_file_without_calibration_is_read_as_before(tmp_path, monkeypatch, kind, voxels):
    """A floating-point file without stored calibration is read whole, as before: load_volume refuses a file with a
    non-finite value, and only the pixels show that. Every page, in one whole read."""
    path = write_layout(tmp_path, kind)
    assert page_reads(monkeypatch, tmp_path, voxels) == expected_reads(kind, path) == (n_pages(path),
                                                                                       [path.resolve()])


@pytest.mark.parametrize("voxels", ["given", "missing"])
@pytest.mark.parametrize("kind", WITH_CALIBRATION)
def test_a_file_with_stored_calibration_is_read_as_before(tmp_path, monkeypatch, kind, voxels):
    """A file with stored calibration is read whole, as before: its grid holds the SHA-256 of its pixels. Every page,
    in one whole read."""
    path = write_layout(tmp_path, kind)
    assert page_reads(monkeypatch, tmp_path, voxels) == expected_reads(kind, path) == (n_pages(path),
                                                                                       [path.resolve()])


@pytest.mark.parametrize("kind", REFUSED)
def test_a_file_load_volume_refuses_is_read_as_before(tmp_path, monkeypatch, kind):
    """A file load_volume refuses (another axis, dtype or several series) is read as before, so that it raises as
    before: the pages EXPECTED_GRIDS states (none for two series, which tifffile refuses before a page)."""
    path = write_layout(tmp_path, kind)
    assert page_reads(monkeypatch, tmp_path, "given") == expected_reads(kind, path)


def test_the_counter_sees_every_reader(tmp_path, monkeypatch):
    """The counter counts load_volume, imread, a page read and the Zarr view of a tiled file."""
    path = write_layout(tmp_path, "paged without axis metadata, tiled")
    with counted(monkeypatch) as reads:
        load_volume(path)
    assert reads.pages == 3 and reads.whole == [path.resolve()]
    with counted(monkeypatch) as reads:
        tifffile.imread(path)
        with tifffile.TiffFile(path) as tif:
            tif.pages[1].asarray()
        with tifffile.imread(path, aszarr=True) as store:
            import zarr
            np.asarray(zarr.open(store, mode="r")[:])
    assert reads.pages >= 3 + 1 + 3 and reads.whole == [path.resolve()]


# --- The callers: the export with the assembly and the fusion on ---------------------------------------------------

EXTRAS = "the multiview-stitcher and anndata extras (starfinder[multiview-stitcher,anndata]) are not installed"
needs_extras = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None
                                  or importlib.util.find_spec("anndata") is None, reason=EXTRAS)
PARTS = {"assemble": "written", "fuse": "written", "segment": "off", "fuse_labels": "off", "export_spatialdata": "off"}


def whole_reads(reads, images):
    return {fov: reads.whole.count(path) for fov, path in images.items()}


def declared_grid(shape, frame, spacing):
    return ReferenceGrid(shape, ImageMetadata(f"declared:{frame}", spacing_zyx=spacing, spatial_unit="micrometer"),
                         "declared")


def golden_grid(fov):
    """The declared grid of a reference image of golden_sample: 8×48×64, voxel sizes 1.0 µm."""
    return declared_grid((8, 48, 64), f"data/out/{fov}", (1.0, 1.0, 1.0))


def grid_record(fov, part):
    """golden_grid as the export record writes it in the part's ``fovs`` entries."""
    record = {"shape_zyx": [8, 48, 64],
              "metadata": {"frame_id": f"declared:data/out/{fov}", "spacing_zyx": [1.0, 1.0, 1.0], "origin_zyx": None,
                           "direction_zyx": None, "spatial_unit": "micrometer"},
              "sha256": None, "source": "declared"}
    return {**record, "calibration_source": "declared"} if part == "assembly" else record


def golden_sample(root):
    """W-390's workflow sample (test_workflow_declared_grids): three FOVs of 8×48×64 uint8 reference images as pages
    without metadata, voxel sizes 1.0 µm, the fusion of reference_merged in both dimensionalities. Returns (output
    root, workflow configuration, {fov: reference image})."""
    from .assembly_fixtures import GOLDEN_IDS
    from .test_workflow_declared_grids import build
    out, config = build(root, "2d", stored=False)
    return out, config, {fov: (out / "fovs" / fov / "reference.tif").resolve() for fov in GOLDEN_IDS}


@needs_extras
@pytest.mark.stitching
def test_each_export_caller_reads_the_grid_without_pixels(tmp_path, monkeypatch):
    """assembly_inputs (from the molecule tables), fusion_inputs and the route after fusion's input grids read no
    reference image whole, and each gets the declared grid of every FOV (golden_grid)."""
    from starfinder.dataset.workflow import _assembly_block, _export_dataset
    from starfinder.export import ExportConfig, FusionConfig
    from starfinder.dataset._export import _input_grids, assembly_inputs, fusion_inputs
    from starfinder.stitching import load_placement
    out, config, images = golden_sample(tmp_path)
    dataset = _export_dataset(config, _assembly_block(config))
    dataset.load_codebook(tmp_path / "input" / "data" / "sample" / "genes.csv")
    placement = load_placement(out / "stitching" / "sample1")
    export = ExportConfig(fusion=FusionConfig(images=("reference_merged",)))
    callers = {"assembly_inputs": lambda: [i.grid for i in assembly_inputs(dataset, placement, None)],
               "fusion_inputs": lambda: [i.grid for i in fusion_inputs(dataset, placement, export)],
               "_input_grids": lambda: list(_input_grids(dataset, placement).values())}
    for name, call in callers.items():
        with counted(monkeypatch) as reads:
            grids = call()
        assert grids == [golden_grid(fov) for fov in images], name
        assert all(grid.source == "declared" for grid in grids), name
        assert whole_reads(reads, images) == {fov: 0 for fov in images}, name


@needs_extras
@pytest.mark.slow
@pytest.mark.stitching
@pytest.mark.workflow
def test_the_export_reads_no_reference_image_whole_for_its_grid(tmp_path, monkeypatch):
    """The workflow's reads_assignment with an assembly block, and its export_sample (Dataset.export with the
    assembly and the fusion on), read no reference image whole for its grid; the fusion reads every page. The
    figures are drawn (previews on, W-568), and the export record holds the declared grid of every FOV (grid_record)
    in the assembly and the fusion. That the plain sample's saved folders, tables and fused images equal those of
    the same sample with stored metadata is test_workflow_declared_grids.py's
    test_reads_assignment_and_export_sample_on_declared_grids."""
    import dask

    from .test_workflow_assembly_figures import export_sample, reads_assignment
    root = tmp_path
    out, config, images = golden_sample(root)
    # The figures reads_assignment draws on request (W-568, assembly.previews).
    config = {**config, "assembly": {**config["assembly"], "previews": True}}
    with counted(monkeypatch) as reads:
        figures = {fov: reads_assignment(root, out, fov, config) for fov in images}
    assert whole_reads(reads, images) == {fov: 0 for fov in images}
    assert {fov: list(drawn) for fov, drawn in figures.items()} == {fov: ["assignment.png"] for fov in images}
    assert all(png for drawn in figures.values() for png in drawn.values())
    with dask.config.set(scheduler="synchronous"), counted(monkeypatch) as reads:
        export_sample(out, config)
    folder = out / "export" / "sample1"
    record = json.loads((folder / "export.json").read_text())
    assert record["parts"] == PARTS
    assert whole_reads(reads, images) == {fov: 0 for fov in images}
    assert reads.pages >= sum(n_pages(path) for path in images.values())
    for part in ("assembly", "fusion"):
        assert [e["grid"] for e in record[part]["fovs"]] == [grid_record(fov, part) for fov in images], part
    for output in record["fusion"]["outputs"]:
        fused = tifffile.imread(folder / output["path"])
        assert fused.dtype == np.uint8 and fused.any()


@needs_extras
@pytest.mark.stitching
def test_a_float_reference_image_is_still_read_whole(monkeypatch, tmp_path):
    """W-390's sample scene, whose stains are float32 and written as pages without metadata: whether a float file is
    finite only its pixels show (load_volume refuses NaN), so assembly_inputs and fusion_inputs read it whole, as
    before, and get the declared grid of every FOV: 5×36×36, voxel sizes 1.0 and 0.5 µm."""
    from starfinder.export import ExportConfig, FusionConfig
    from starfinder.dataset._export import assembly_inputs, fusion_inputs
    from starfinder.stitching import load_placement
    from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene

    from .test_stitching_declared_grids import SAMPLE, scene_dataset
    book, config = sample_scene_preset("sample-tiny-v1")
    scene = generate_sample_scene(book, config=config)
    dataset = scene_dataset(tmp_path / "plain", scene, book, save_sample_scene(scene, tmp_path / "scene"),
                            stored=False)
    images = {fov: (dataset.output_root / "fovs" / fov / "reference.tif").resolve()
              for fov in scene.positions.fov_id}
    assert {load_volume(path).image.dtype for path in images.values()} == {np.dtype(np.float32)}
    placement = load_placement(dataset.output_root / "stitching" / SAMPLE)
    export = ExportConfig(fusion=FusionConfig(images=("reference_merged",)))

    def grids():
        return ([i.grid for i in assembly_inputs(dataset, placement, "default")]
                + [i.grid for i in fusion_inputs(dataset, placement, export)])

    with counted(monkeypatch) as reads:
        result = grids()
    expected = [declared_grid((5, 36, 36), f"{DATASET_ID}/{OUTPUT_ID}/{fov}", (1.0, 0.5, 0.5)) for fov in images]
    assert list(images) == ["FOV_001", "FOV_002", "FOV_003", "FOV_004"]
    assert result == expected * 2 and whole_reads(reads, images) == {fov: 2 for fov in images}


# --- The stitching's read_inputs -----------------------------------------------------------------------------------

@pytest.mark.stitching
@pytest.mark.parametrize("input_name", ["reference_merged", "reference_stain", "file:images/other/{fovID}.tif",
                                        "file:images/missing/{fovID}.tif"])
def test_read_inputs_reads_the_reference_image_whole_only_for_its_pixels(tmp_path, monkeypatch, input_name):
    """read_inputs on W-390's workflow sample (uint8 reference images as pages without metadata): only
    reference_merged, whose image it is, reads the reference image whole, and reference_stain, a file: input and a
    missing file: input read only its grid (no page of it decoded). Every StitchingInput holds the declared grid
    (golden_grid), its source, the pixels of the file it names and that file's SHA-256 (none for a missing file)."""
    import hashlib

    from starfinder.dataset._declared import reference_settings
    from starfinder.dataset._export import reference_stain_path
    from starfinder.dataset._stitching import read_inputs
    from starfinder.dataset.workflow import _assembly_block, _export_dataset
    out, config, images = golden_sample(tmp_path)
    reference = reference_settings(_export_dataset(config, _assembly_block(config)))
    files = {}
    for fov in images:
        grid = reference_grid(out, fov, **reference)
        stain = pixels(grid.shape_zyx)
        save_volume(stain[..., None], reference_stain_path(out, fov), metadata=grid.metadata)
        other = out / "images" / "other" / f"{fov}.tif"
        other.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(other, pixels(grid.shape_zyx), metadata=None, photometric="minisblack")
        files[fov] = {"reference_merged": (images[fov], tifffile.imread(images[fov])),
                      "reference_stain": (reference_stain_path(out, fov), stain),
                      "file:images/other/{fovID}.tif": (other, tifffile.imread(other)),
                      "file:images/missing/{fovID}.tif": (out / "images" / "missing" / f"{fov}.tif", None)}

    def whole_reads_of_references(reads):
        return {fov: reads.whole.count(path) for fov, path in images.items()}

    with counted(monkeypatch) as reads:
        result = read_inputs(out, list(images), input_name, reference=reference)
    assert list(result) == list(images)
    for fov in images:
        a, (path, pixels_of_file) = result[fov], files[fov][input_name]
        source = input_name if not input_name.startswith("file:") else f"file:{path}"
        file_sha256 = None if pixels_of_file is None else hashlib.sha256(path.read_bytes()).hexdigest()
        assert (a.fov_id, a.grid, a.source, a.file_sha256) == (fov, golden_grid(fov), source, file_sha256)
        assert (a.image is None) == (pixels_of_file is None) == input_name.startswith("file:images/missing")
        assert a.image is None or (a.image.dtype == pixels_of_file.dtype and np.array_equal(a.image, pixels_of_file))
        assert a.grid.source == "declared"
    assert whole_reads_of_references(reads) == {fov: 1 if input_name == "reference_merged" else 0 for fov in images}
