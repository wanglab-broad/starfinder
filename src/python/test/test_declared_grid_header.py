"""W-433: the grid of a reference image without stored calibration is read from its TIFF header.

``starfinder.dataset._declared.reference_grid`` takes the shape of a file that stores no
``starfinder_metadata`` from its header (``starfinder.io.tiff._volume_header``) and reads no
pixel; a file with stored calibration, whose grid holds the SHA-256 of its pixels, and a file
whose header does not settle the shape ``load_volume`` gives are read whole as before.

The oracle is the base revision's function (``ec4774a``), copied below as
``base_reference_grid``: the file loaded whole by ``load_volume``. Fixtures, all written in the
test: one small synthetic file per layout ``load_volume`` reads or refuses (``LAYOUTS``), and the
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

from starfinder.dataset._declared import declared_metadata, reference_grid, reference_volume
from starfinder.image import ImageMetadata
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


def base_reference_grid(output_root, fov_id, *, dataset_id, output_id, voxel_size_xy=None, voxel_size_z=None,
                        reference_image="images/ref_merged/{fov_id}.tif"):
    """The base revision's ``reference_grid`` (``reference_volume(...)[0]`` of ``ec4774a``), copied: the whole file."""
    from starfinder.segmentation._labels import grid_sha256
    path = Path(output_root) / reference_image.replace("{fov_id}", fov_id)
    if not path.is_file():
        raise FileNotFoundError(f"the reference image of FOV {fov_id!r} is missing: {path} (the file the setting "
                                f"reference_image {reference_image!r} names)")
    loaded = load_volume(path)
    if loaded.diagnostics["metadata_source"] == "stored":
        return ReferenceGrid(loaded.image.shape, loaded.metadata, f"file:{Path(path).resolve()}",
                             grid_sha256(loaded.image))
    missing = [key for key, value in zip(("voxel_size_xy", "voxel_size_z"), (voxel_size_xy, voxel_size_z))
               if value is None]
    if missing:
        raise ValueError(f"{path} stores no calibration (no starfinder_metadata description), so its grid is declared "
                         f"by the dataset configuration, which lacks {' and '.join(missing)}: set "
                         f"{' and '.join(missing)} (µm) in the workflow configuration, or the Dataset fields of the "
                         "same names")
    metadata = declared_metadata(dataset_id, output_id, fov_id, voxel_size_xy=voxel_size_xy,
                                 voxel_size_z=voxel_size_z)
    return ReferenceGrid(loaded.image.shape, metadata, "declared")


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


def write_layout(root, kind):
    path = Path(root) / "images" / "ref_merged" / f"{FOV}.tif"
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

@pytest.mark.parametrize("voxels", list(VOXELS))
@pytest.mark.parametrize("kind", LAYOUTS)
def test_the_grid_equals_the_base_revisions(tmp_path, kind, voxels):
    """Every field of the grid (shape, metadata, source, hash), or the exception type and message, as on the base."""
    write_layout(tmp_path, kind)
    result, base = outcome(reference_grid, tmp_path, **VOXELS[voxels]), outcome(base_reference_grid, tmp_path,
                                                                                 **VOXELS[voxels])
    assert type(result) is type(base)
    if isinstance(base, ReferenceGrid):
        assert (result.shape_zyx, result.metadata, result.source, result.sha256) == (
            base.shape_zyx, base.metadata, base.source, base.sha256)
        assert result == base == reference_volume(tmp_path, FOV, dataset_id=DATASET_ID, output_id=OUTPUT_ID,
                                                  **VOXELS[voxels])[0]
    else:
        assert result == base
    if kind in SETTLED:
        assert isinstance(base, ReferenceGrid) if voxels == "given" else base[0] is ValueError
    if kind in WITH_CALIBRATION:
        assert isinstance(base, ReferenceGrid) and base.metadata == STORED and base.sha256 is not None
    if kind in UNSETTLED:
        assert isinstance(base, ReferenceGrid) == (kind in FINITE_FLOAT and voxels == "given")
        assert kind in FINITE_FLOAT or "stores no calibration" not in base[1]


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
    result, base = outcome(reference_grid, tmp_path, **VOXELS[voxels]), outcome(base_reference_grid, tmp_path,
                                                                                **VOXELS[voxels])
    assert result == base and base[0] is FileNotFoundError and "reference_image" in base[1]


# --- No pixel is read ---------------------------------------------------------------------------------------------

def page_reads(monkeypatch, root, kind, voxels):
    """(result, base) PixelReads of reference_grid and of the base revision's on the layout's file."""
    with counted(monkeypatch) as reads:
        outcome(reference_grid, root, **VOXELS[voxels])
    with counted(monkeypatch) as base:
        outcome(base_reference_grid, root, **VOXELS[voxels])
    return reads, base


@pytest.mark.parametrize("voxels", ["given", "missing"])
@pytest.mark.parametrize("kind", SETTLED)
def test_no_page_is_decoded_for_an_integer_file_without_calibration(tmp_path, monkeypatch, kind, voxels):
    """0 pages decoded and no whole read for a file of an integer dtype without stored calibration whose header
    settles the shape (also when the voxel sizes are missing and it raises); the base revision read it whole."""
    path = write_layout(tmp_path, kind)
    assert load_volume(path).image.dtype.kind in "ui"
    reads, base = page_reads(monkeypatch, tmp_path, kind, voxels)
    assert (reads.pages, reads.whole) == (0, []) and base.pages == n_pages(path) > 0 and base.whole == [
        path.resolve()]


@pytest.mark.parametrize("voxels", ["given", "missing"])
@pytest.mark.parametrize("kind", FLOAT)
def test_a_floating_point_file_without_calibration_is_read_as_before(tmp_path, monkeypatch, kind, voxels):
    """A floating-point file without stored calibration is read whole, as on the base revision: load_volume refuses
    a file with a non-finite value, and only the pixels show that. The same pages, the same whole read."""
    path = write_layout(tmp_path, kind)
    reads, base = page_reads(monkeypatch, tmp_path, kind, voxels)
    assert (reads.pages, reads.whole) == (base.pages, base.whole) == (n_pages(path), [path.resolve()])


@pytest.mark.parametrize("voxels", ["given", "missing"])
@pytest.mark.parametrize("kind", WITH_CALIBRATION)
def test_a_file_with_stored_calibration_is_read_as_before(tmp_path, monkeypatch, kind, voxels):
    """A file with stored calibration is read whole, as on the base revision: its grid holds the SHA-256 of its
    pixels. The same pages, the same whole read."""
    path = write_layout(tmp_path, kind)
    reads, base = page_reads(monkeypatch, tmp_path, kind, voxels)
    assert (reads.pages, reads.whole) == (base.pages, base.whole) == (n_pages(path), [path.resolve()])


@pytest.mark.parametrize("kind", REFUSED)
def test_a_file_load_volume_refuses_is_read_as_before(tmp_path, monkeypatch, kind):
    """A file load_volume refuses (another axis, dtype or several series) is read as on the base revision, so that
    it raises as before."""
    write_layout(tmp_path, kind)
    reads, base = page_reads(monkeypatch, tmp_path, kind, "given")
    assert (reads.pages, reads.whole) == (base.pages, base.whole)


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


@contextmanager
def base_revision(monkeypatch):
    """The callers with the base revision's reference_grid (each resolves it from the module when called)."""
    import starfinder.dataset._declared as declared
    with monkeypatch.context() as patch:
        patch.setattr(declared, "reference_grid", base_reference_grid)
        yield


def golden_sample(root):
    """W-390's workflow sample (test_workflow_declared_grids): three FOVs of 8×48×64 uint8 reference images as pages
    without metadata, voxel sizes 1.0 µm, the fusion of reference_merged in both dimensionalities. Returns (output
    root, workflow configuration, {fov: reference image})."""
    from .assembly_fixtures import GOLDEN_IDS
    from .test_workflow_declared_grids import build
    out, config = build(root, "2d", stored=False)
    return out, config, {fov: (out / "images" / "ref_merged" / f"{fov}.tif").resolve() for fov in GOLDEN_IDS}


@needs_extras
@pytest.mark.stitching
def test_each_export_caller_reads_the_grid_without_pixels(tmp_path, monkeypatch):
    """assembly_inputs (from the molecule tables), fusion_inputs and the route after fusion's input grids: each read
    each reference image whole once on the base revision, for its grid alone; now none does, and each gets the
    base's grids."""
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
        with base_revision(monkeypatch), counted(monkeypatch) as base:
            base_grids = call()
        assert grids == base_grids and len(grids) == len(images), name
        assert all(grid.source == "declared" for grid in grids), name
        assert whole_reads(reads, images) == {fov: 0 for fov in images}, name
        assert whole_reads(base, images) == {fov: 1 for fov in images}, name


@needs_extras
@pytest.mark.slow
@pytest.mark.stitching
@pytest.mark.workflow
def test_the_export_reads_no_reference_image_whole_for_its_grid(tmp_path, monkeypatch):
    """The workflow's reads_assignment with an assembly block reads each reference image whole once on the base
    revision, for its grid alone, and its export_sample (Dataset.export with the assembly and the fusion on) twice
    (the assembly's grid and the fusion's grid); now neither does. The fusion reads the same pages; the saved
    folders, records, tables and fused images are the base's."""
    import shutil

    import dask

    from starfinder.assembly import load_assembly

    from .test_stitching_declared_grids import compare_tables
    from .test_workflow_assembly_figures import export_sample, reads_assignment
    root = tmp_path
    out, config, images = golden_sample(root)
    for name in ("result", "base"):
        with counted(monkeypatch) as reads:
            if name == "base":
                with base_revision(monkeypatch):
                    figures = {fov: reads_assignment(root, out, fov, config) for fov in images}
            else:
                figures = {fov: reads_assignment(root, out, fov, config) for fov in images}
        expected = {fov: 1 if name == "base" else 0 for fov in images}
        assert whole_reads(reads, images) == expected, name
        saved = {fov: {p.name: p.read_bytes() for p in sorted((out / "checkpoints" / fov).rglob("*")) if p.is_file()}
                 for fov in images}
        if name == "result":
            first = (figures, saved)
        else:
            assert (figures, saved) == first
    folders, records, whole, pages = {}, {}, {}, {}
    for name in ("result", "base"):
        with dask.config.set(scheduler="synchronous"), counted(monkeypatch) as reads:
            if name == "base":
                with base_revision(monkeypatch):
                    export_sample(out, config)
            else:
                export_sample(out, config)
        folders[name] = shutil.move(out / "export" / "sample1", root / f"export_{name}")
        records[name] = json.loads((folders[name] / "export.json").read_text())
        whole[name], pages[name] = whole_reads(reads, images), reads.pages
    assert records["result"]["parts"] == records["base"]["parts"] == PARTS
    assert whole == {"result": {fov: 0 for fov in images}, "base": {fov: 2 for fov in images}}
    total = sum(n_pages(path) for path in images.values())
    assert pages["base"] - pages["result"] == 2 * total and pages["result"] >= total
    for part in ("assembly", "fusion"):
        grids = {name: [e["grid"] for e in records[name][part]["fovs"]] for name in records}
        assert grids["result"] == grids["base"], part
    compare_tables(load_assembly(folders["result"]), load_assembly(folders["base"]))
    for output in records["result"]["fusion"]["outputs"]:
        a, b = (tifffile.imread(folders[name] / output["path"]) for name in ("result", "base"))
        assert a.dtype == b.dtype == np.uint8 and np.array_equal(a, b) and a.any()


@needs_extras
@pytest.mark.stitching
def test_a_float_reference_image_is_still_read_whole(monkeypatch, tmp_path):
    """W-390's sample scene, whose stains are float32 and written as pages without metadata: whether a float file is
    finite only its pixels show (load_volume refuses NaN), so assembly_inputs and fusion_inputs read it whole, as
    on the base revision, and get the base's grids."""
    from starfinder.export import ExportConfig, FusionConfig
    from starfinder.dataset._export import assembly_inputs, fusion_inputs
    from starfinder.stitching import load_placement
    from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene

    from .test_stitching_declared_grids import SAMPLE, scene_dataset
    book, config = sample_scene_preset("sample-tiny-v1")
    scene = generate_sample_scene(book, config=config)
    dataset = scene_dataset(tmp_path / "plain", scene, book, save_sample_scene(scene, tmp_path / "scene"),
                            stored=False)
    images = {fov: (dataset.output_root / "images" / "ref_merged" / f"{fov}.tif").resolve()
              for fov in scene.positions.fov_id}
    assert {load_volume(path).image.dtype for path in images.values()} == {np.dtype(np.float32)}
    placement = load_placement(dataset.output_root / "stitching" / SAMPLE)
    export = ExportConfig(fusion=FusionConfig(images=("reference_merged",)))

    def grids():
        return ([i.grid for i in assembly_inputs(dataset, placement, "default")]
                + [i.grid for i in fusion_inputs(dataset, placement, export)])

    with counted(monkeypatch) as reads:
        result = grids()
    with base_revision(monkeypatch), counted(monkeypatch) as base:
        expected = grids()
    assert result == expected and whole_reads(reads, images) == whole_reads(base, images) == {
        fov: 2 for fov in images}


# --- The stitching's read_inputs -----------------------------------------------------------------------------------

@pytest.mark.stitching
@pytest.mark.parametrize("input_name", ["reference_merged", "reference_stain", "file:images/other/{fovID}.tif",
                                        "file:images/missing/{fovID}.tif"])
def test_read_inputs_reads_the_reference_image_whole_only_for_its_pixels(tmp_path, monkeypatch, input_name):
    """read_inputs on W-390's workflow sample (uint8 reference images as pages without metadata): the base revision
    read each reference image whole for every input; now only reference_merged, whose image it is, reads it whole,
    and reference_stain, a file: input and a missing file: input read only its grid (no page of it decoded). Every
    StitchingInput (image, grid, source, file hash) equals the base's."""
    from starfinder._layout import _FovPaths
    from starfinder.dataset._declared import reference_settings
    from starfinder.dataset._stitching import REFERENCE_STAIN_FILE, read_inputs
    from starfinder.dataset.workflow import _assembly_block, _export_dataset
    out, config, images = golden_sample(tmp_path)
    reference = reference_settings(_export_dataset(config, _assembly_block(config)))
    for fov in images:
        grid = reference_grid(out, fov, **reference)
        save_volume(pixels(grid.shape_zyx + (1,)), _FovPaths(out, fov).checkpoint_dir / REFERENCE_STAIN_FILE,
                    metadata=grid.metadata)
        other = out / "images" / "other" / f"{fov}.tif"
        other.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(other, pixels(grid.shape_zyx), metadata=None, photometric="minisblack")

    def whole_reads_of_references(reads):
        return {fov: reads.whole.count(path) for fov, path in images.items()}

    with counted(monkeypatch) as reads:
        result = read_inputs(out, list(images), input_name, reference=reference)
    with base_revision(monkeypatch), counted(monkeypatch) as base:
        expected = read_inputs(out, list(images), input_name, reference=reference)
    assert list(result) == list(expected) == list(images)
    for fov in images:
        a, b = result[fov], expected[fov]
        assert (a.fov_id, a.grid, a.source, a.file_sha256) == (b.fov_id, b.grid, b.source, b.file_sha256)
        assert (a.image is None) == (b.image is None) == input_name.startswith("file:images/missing")
        assert a.image is None or (a.image.dtype == b.image.dtype and np.array_equal(a.image, b.image))
        assert a.grid.source == "declared"
    assert whole_reads_of_references(base) == {fov: 1 for fov in images}
    assert whole_reads_of_references(reads) == {fov: 1 if input_name == "reference_merged" else 0 for fov in images}
    if input_name != "reference_merged":
        assert base.pages - reads.pages == sum(n_pages(path) for path in images.values())
