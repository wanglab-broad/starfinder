"""The fused sample image: multiview-stitcher's ``fuse`` on the placement of either backend.

docs/sample-output-contract.md ("The fused image", "Fusion by multiview-stitcher", "The
fusion grid", "Dtypes written", "The 2D projection", "The scale factor", "The 3D stack",
"File layouts"). Starfinder builds the library's inputs from the placement (one image per
included FOV, its spacing as scale and its origin as translation, identity affine under the
key ``placed``), projects each FOV along Z for the projection route, requests the
library's level 1 of each FOV for a scale factor (``level1``), passes the output grid as
``output_stack_properties`` and calls ``multiview_stitcher.fusion.fuse`` once per file with
``weighted_average_fusion`` at interpolation order 1. It blends, interpolates and averages
nothing itself; it writes the fused array as OME-TIFF (and the library's OME-Zarr on
request). The library is imported only when ``fuse_images`` runs. Its own reading of a source file
(``_Reader``) gives the library one task per page, or per file for the projection, and keeps
each result from its first use until the last band of the file written that needs it, so
that in one fusion each page is read and decoded at most once.
"""
from __future__ import annotations

import itertools
import math
import threading
import uuid
import warnings
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from starfinder._records import atomic_path, publish, software
from starfinder.image import IncompatibleGeometryError
from starfinder.io import _ome
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import PlacementResult
from starfinder.stitching._placement import OWNING, grid_problems, same_frame_grid

from ._config import FusionConfig
from ._grid import AXES, FusedGrid, _names, fused_grid, included_fovs
from ._record import (
    ExportIncompleteWarning,
    Timer,
    as_json,
    directory_files,
    listed_files,
    read_export,
    remove_listed,
    section_sha256,
    update_section,
)

ALLOWED_DTYPES = ("uint8", "uint16", "float32")
EXTRA = "multiview-stitcher"
TRANSFORM_KEY = "placed"
FUSION_FUNC = "weighted_average_fusion"
INTERPOLATION_ORDER = 1
#: The library's default blending widths (multiview_stitcher.weights, 0.1.62), not passed and not a key.
BLENDING_WIDTHS_UM = {"z": 3, "y": 10, "x": 10}
SUM_BOUND = 2 ** 24
#: Output chunks of the lazy fused array: one plane band per chunk, so a band of a plane fuses only its own chunks.
CHUNKS = {"stack": {"z": 1, "y": 512, "x": 512}, "plane": {"y": 512, "x": 512}}
#: Chunks of the standalone OME-Zarr (t, c, (z), y, x), set by a dask rechunk before the library's writer (A12).
ZARR_CHUNKS = {"projection": {"t": 1, "c": 1, "y": 1024, "x": 1024},
               "stack": {"t": 1, "c": 1, "z": 16, "y": 512, "x": 512}}


@dataclass(frozen=True)
class FusionInput:
    """One included FOV's input of :func:`fuse_images`: its reference grid and its source images.

    ``images`` maps each source-image name of ``FusionConfig.images`` to the
    image on the FOV's reference grid: a ZYX (or YX for a one-plane grid)
    numpy array, or the path of a TIFF file read lazily. A name that is
    absent, or a path that does not exist, is a missing source image.

    Raises
    ------
    TypeError
        An empty or non-string ``fov_id``, a ``grid`` that is not a
        ReferenceGrid, or ``images`` that is not a mapping of names to arrays
        or paths.
    """

    fov_id: str
    grid: ReferenceGrid
    images: Mapping[str, Any]

    def __post_init__(self):
        if not isinstance(self.fov_id, str) or not self.fov_id:
            raise TypeError(f"fov_id must be a nonempty string; got {self.fov_id!r}")
        if not isinstance(self.grid, ReferenceGrid):
            raise TypeError(f"grid of FOV {self.fov_id!r} must be a ReferenceGrid")
        if not isinstance(self.images, Mapping):
            raise TypeError(f"images of FOV {self.fov_id!r} must map source-image names to arrays or paths")
        for name, value in self.images.items():
            if not isinstance(name, str) or not isinstance(value, (np.ndarray, str, Path)):
                raise TypeError(f"images of FOV {self.fov_id!r}: {name!r} must map to a numpy array or a path")
        object.__setattr__(self, "images", dict(self.images))


@dataclass(frozen=True)
class FusionRecord:
    """The ``fusion`` section ``fuse_images`` wrote to ``export.json`` and the files it wrote.

    ``status`` is ``written``, ``empty``, ``not_run`` or ``failed``;
    ``section`` the section as written (JSON values); ``paths`` the written
    files and folders, in writing order.
    """

    status: str
    section: Mapping[str, Any]
    paths: tuple[Path, ...]


@dataclass
class _Source:
    fov_id: str
    name: str
    value: Any
    path: Path | None
    shape: tuple[int, int, int] | None
    dtype: np.dtype | None


@dataclass
class _Plan:
    ids: list
    grid: FusedGrid | None
    by_fov: dict
    sources: dict          # (fov, name) -> _Source, present sources
    missing: list          # [{fov_id, image, path}]
    dtypes: dict           # name -> dtype
    placement_origins: dict = None  # fov -> (origin_z, origin_y, origin_x)


#: The axes tifffile reports for pages over Y and X of a file without axis metadata (generic and shaped pages).
PAGE_AXES = ("IYX", "QYX")


def _unnamed_axes(tif, series) -> bool:
    """Whether a series carries no axis metadata of any kind: tifffile's fallback series of a file no metadata format
    matched (``uniform`` or ``generic`` pages), or a shaped description without axes."""
    if series.kind in ("uniform", "generic"):
        return True
    if series.kind != "shaped":
        return False
    described = tif.shaped_metadata or ()
    return bool(described) and "axes" not in described[0]


def _source_shape(path):
    """(shape_zyx, dtype) of a one-channel TIFF from its header; a YX file has Z = 1.

    A series that carries no axis metadata (not OME, not ImageJ, no shaped
    description naming axes) and whose axes, as tifffile reports them, are
    exactly one page axis over Y and X (:data:`PAGE_AXES`) is read as ZYX. Any
    other file follows the axes it reports, as before.
    """
    import tifffile
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes, shape, dtype = series.axes, tuple(series.shape), np.dtype(series.dtype)
        pages = axes in PAGE_AXES and _unnamed_axes(tif, series)
    if pages:
        return (shape[0], shape[1], shape[2]), dtype
    kept = [(a, n) for a, n in zip(axes, shape) if not (n == 1 and a not in "YX")]
    if "".join(a for a, _ in kept) not in ("YX", "ZYX"):
        raise ValueError(f"source image {path} has axes {axes} {shape}; a source image is one channel, ZYX or YX")
    sizes = dict(kept)
    return (sizes.get("Z", 1), sizes["Y"], sizes["X"]), dtype


def _array_shape(value):
    if value.ndim == 2:
        return (1,) + tuple(value.shape)
    if value.ndim != 3:
        raise ValueError(f"a source array is ZYX or YX; got shape {value.shape}")
    return tuple(value.shape)


def sum_bound(dtype, n_planes) -> None:
    """The sum projection's bound: for an integer dtype, ``n_planes × max`` must be at most 2^24 (ValueError)."""
    dtype = np.dtype(dtype)
    if dtype.kind not in "ui":
        return
    largest = int(np.iinfo(dtype).max)
    if int(n_planes) * largest > SUM_BOUND:
        raise ValueError(f"a sum projection of {n_planes} planes of {dtype} can reach {int(n_planes) * largest}, "
                         f"above 2^24 = {SUM_BOUND}, so its sums would not be exact in float32 (at most "
                         f"{SUM_BOUND // largest} planes of {dtype})")


def _recorded_grid(placement, fov):
    from starfinder.assembly._ownership import recorded_grid
    return recorded_grid(placement, fov)


def check_assembly(placement, assembly) -> None:
    """An assembly given to a part must come from this placement: equal ``fovs`` and ``links`` SHA-256 (ValueError)."""
    from starfinder.assembly import AssemblyResult
    from starfinder.stitching._placement import table_sha256
    if not isinstance(assembly, AssemblyResult):
        raise TypeError("assembly must be an AssemblyResult or None")
    recorded = assembly.record.get("placement") or {}
    current = {"fovs_sha256": table_sha256(placement.fovs), "links_sha256": table_sha256(placement.links)}
    differ = {k: (recorded.get(k), v) for k, v in current.items() if recorded.get(k) != v}
    if differ:
        raise ValueError(f"the assembly of sample {assembly.sample!r} was made from another placement: " +
                         "; ".join(f"{k} recorded {a}, the placement {b}" for k, (a, b) in differ.items()))


def check_fusion(placement, inputs, *, config, assembly=None) -> _Plan:
    """The checks of ``fuse_images`` before anything is written; returns what it reads.

    Check 5 of docs/sample-output-contract.md ("Checks before anything is
    written"), then the source images' shapes and dtypes, the sum bound and
    the pyramid bound. Missing source images are returned, not raised: the
    part then cannot run.
    """
    from starfinder.stitching._review import require_approval
    if not isinstance(placement, PlacementResult):
        raise TypeError("placement must be a PlacementResult")
    if not isinstance(config, FusionConfig):
        raise TypeError("config must be a FusionConfig")
    config.__post_init__()
    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence of FusionInput")
    inputs = list(inputs)
    wrong = [i for i, item in enumerate(inputs) if not isinstance(item, FusionInput)]
    if wrong:
        raise TypeError(f"inputs must be FusionInput; items {_names(wrong)} are not")
    require_approval(placement)
    if assembly is not None:
        check_assembly(placement, assembly)
    ids = included_fovs(placement)
    given = [item.fov_id for item in inputs]
    repeated = sorted({f for f in given if given.count(f) > 1})
    if repeated:
        raise ValueError(f"each included FOV has exactly one FusionInput; repeated: {_names(repeated)}")
    foreign = [f for f in given if f not in ids]
    if foreign:
        raise ValueError(f"FusionInput names FOVs that are not included FOVs of the placement of sample "
                         f"{placement.sample!r}: {_names(foreign)}")
    without = [f for f in ids if f not in given]
    if without:
        raise ValueError(f"every included FOV needs a FusionInput; FOVs without one: {_names(without)}")
    by_fov = {item.fov_id: item for item in inputs}
    origins = {row.fov_id: (row.origin_z, row.origin_y, row.origin_x) for row in placement.fovs.itertuples(index=False)}
    if not ids:
        return _Plan([], None, {}, {}, [], {}, origins)
    for fov in ids:
        grid, recorded = by_fov[fov].grid, _recorded_grid(placement, fov)
        if recorded is None or not same_frame_grid(recorded[0], recorded[1], grid.shape_zyx, grid.metadata):
            raise IncompatibleGeometryError(
                f"the grid of FOV {fov} ({grid.shape_zyx}, {grid.metadata}) is not the grid the placement recorded "
                f"({None if recorded is None else (recorded[0], recorded[1])})")
    bad = {"unknown spacing": [], "a unit other than µm": [], "a direction other than the identity": []}
    for fov in ids:
        for what in grid_problems(by_fov[fov].grid.metadata):
            bad[what].append(fov)
    found = [f"{what}: {_names(fovs)}" for what, fovs in bad.items() if fovs]
    if found:
        raise IncompatibleGeometryError("fuse_images needs known spacing, unit µm and identity direction on every grid; "
                                        f"FOVs with {'; '.join(found)}")
    grid = fused_grid(placement, {fov: by_fov[fov].grid for fov in ids}, config=config)
    sources, missing = {}, []
    for name in config.image_names:
        for fov in ids:
            value = by_fov[fov].images.get(name)
            if value is None or (isinstance(value, (str, Path)) and not Path(value).is_file()):
                missing.append({"fov_id": fov, "image": name, "path": None if value is None else str(value)})
                continue
            if isinstance(value, np.ndarray):
                path, (shape, dtype) = None, (_array_shape(value), value.dtype)
            else:
                path = Path(value).resolve()
                shape, dtype = _source_shape(path)
            expected = by_fov[fov].grid.shape_zyx
            if shape != expected:
                where = "array" if path is None else str(path)
                raise IncompatibleGeometryError(f"source image {name!r} of FOV {fov} ({where}) has shape {shape}; "
                                                f"its reference grid {expected}")
            sources[(fov, name)] = _Source(fov, name, value, path, shape, np.dtype(dtype))
    dtypes = {}
    for name in config.image_names:
        found = {}
        for fov in ids:
            source = sources.get((fov, name))
            if source is not None:
                found.setdefault(source.dtype.name, []).append(fov)
        other = [d for d in found if d not in ALLOWED_DTYPES]
        if other:
            raise ValueError(f"source image {name!r} has dtype {other[0]} (FOVs {_names(found[other[0]])}); fuse_images "
                             f"accepts {list(ALLOWED_DTYPES)}")
        if len(found) > 1:
            raise ValueError(f"source image {name!r} mixes dtypes across FOVs (" +
                             "; ".join(f"{d}: {_names(f)}" for d, f in found.items()) +
                             f"); fuse_images accepts one of {list(ALLOWED_DTYPES)} for every FOV, without conversion")
        if found:
            dtypes[name] = np.dtype(next(iter(found)))
    if config.projection == "sum" and "projection" in config.dimensionalities:
        planes = max(by_fov[fov].grid.shape_zyx[0] for fov in ids)
        for name, dtype in dtypes.items():
            sum_bound(dtype, planes)
    for dimensionality in config.dimensionalities:
        _, _, shape = grid.level(dimensionality)
        bound = int(math.floor(math.log2(max(shape[1], shape[2]))))
        if config.pyramid_levels > bound:
            raise ValueError(f"pyramid_levels {config.pyramid_levels} exceeds floor(log2(max(N_y, N_x))) = {bound} of "
                             f"the {dimensionality} output level {shape[1]} × {shape[2]}")
    return _Plan(ids, grid, by_fov, sources, missing, dtypes, origins)


# --- The record --------------------------------------------------------------------------------------------

def _placement_entry(placement):
    from starfinder.stitching._placement import table_sha256
    review = placement.record.get("review") or {}
    hashes = table_sha256(placement.fovs), table_sha256(placement.links)
    approvals = [a for a in review.get("approvals", []) if (a.get("fovs_sha256"), a.get("links_sha256")) == hashes]
    return {"method": (placement.record.get("method") or {}).get("method"), "gauge": placement.record.get("gauge"),
            "review": review.get("state"), "approval": approvals[-1] if approvals else None,
            "fovs_sha256": hashes[0], "links_sha256": hashes[1]}


def _fov_entries(placement, plan, config):
    from starfinder.segmentation._labels import array_sha256
    from starfinder._records import file_sha256
    entries = []
    for row in placement.fovs.itertuples(index=False):
        origin = [float(getattr(row, f"origin_{a}")) for a in AXES]
        entry = {"fov_id": row.fov_id, "placement_status": row.status, "included": row.status in OWNING,
                 "reason": None if row.status in OWNING else row.status,
                 "origin_zyx": [v if math.isfinite(v) else None for v in origin], "grid": None, "images": {}}
        item = plan.by_fov.get(row.fov_id)
        if item is not None:
            grid = item.grid
            entry["grid"] = {"shape_zyx": list(grid.shape_zyx), "metadata": asdict(grid.metadata),
                             "sha256": grid.sha256, "source": grid.source}
            for name in config.image_names:
                source = plan.sources.get((row.fov_id, name))
                if source is None:
                    value = item.images.get(name)
                    entry["images"][name] = {"path": None if value is None or isinstance(value, np.ndarray)
                                             else str(value), "sha256": None, "missing": True}
                elif source.path is None:
                    entry["images"][name] = {"path": None, "sha256": array_sha256(source.value),
                                             "sha256_of": "array", "dtype": source.dtype.name}
                else:
                    entry["images"][name] = {"path": str(source.path), "sha256": file_sha256(source.path),
                                             "sha256_of": "file", "dtype": source.dtype.name}
        entries.append(entry)
    return entries


def _library_entry(config, grid, version):
    return {"multiview_stitcher": version, "fusion_func": FUSION_FUNC, "interpolation_order": INTERPOLATION_ORDER,
            "blending_widths_um": dict(BLENDING_WIDTHS_UM), "route": "level1" if config.factor_yx > 1 else "none",
            "inputs_2d": None if grid is None else bool(grid.one_plane), "transform_key": TRANSFORM_KEY}


def _grid_entry(grid):
    if grid is None:
        return None
    return {"shape_zyx": list(grid.shape_zyx), "spacing_zyx": list(grid.spacing_zyx),
            "origin_zyx": list(grid.origin_zyx), "inputs_2d": grid.one_plane,
            "slab_z": list(grid.slab_z) if grid.one_plane else None,
            "origins_z": dict(zip(grid.fov_ids, grid.origins_z)), "factor_yx": grid.factor_yx,
            "fov_ids": list(grid.fov_ids)}


def _section(status, placement, plan, config, assembly, outputs, timer, *, version=None, **extra):
    section = {"status": status, **extra,
               "assembly_sha256": None if assembly is None else section_sha256(as_json(assembly.record)),
               "placement": _placement_entry(placement), "fovs": _fov_entries(placement, plan, config),
               "library": _library_entry(config, plan.grid, version), "config": asdict(config),
               "grid": _grid_entry(plan.grid), "outputs": list(outputs), "software": software(),
               **timer.record(), "viewer_qualification": "deferred"}
    return as_json(section)


# --- Fusion ------------------------------------------------------------------------------------------------

class _Reader:
    """Starfinder's reading of the source files of one ``fuse_images`` call (docs/sample-output-contract.md, "Memory").

    A file is one task per page (a page is a Z plane), as one chunk per page
    whatever strips the page is stored in, or one task for the whole file, in
    the chunks of tifffile's Zarr view, when its pages are not its Z planes;
    its projection along Z is one task that reads the file's pages and
    projects them. The tasks open nothing but the file's header before the
    fused array is computed. A task's result is held from its first call
    until :meth:`plan` says no later band of the writer needs it, then
    released; a task the plan does not count is read at each call. Nothing
    is staged on disk.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._task_locks = {}
        self._held = {}
        self._remaining = {}
        self._files = {}
        self._prefix = f"starfinder-source-{uuid.uuid4().hex}"

    # Files and tasks.

    def _file(self, path, stack):
        """The open file of a path (its header read) and the lock its reads take."""
        import tifffile
        with self._lock:
            if path not in self._files:
                self._files[path] = (stack.enter_context(tifffile.TiffFile(path)), threading.Lock())
            return self._files[path]

    def _key(self, source, what, index=None):
        return f"{self._prefix}-{source.fov_id}-{source.name}-{what}" + ("" if index is None else f"-{index}")

    def _task(self, key, load, shape, dtype):
        import dask
        import dask.array as da
        # The key is bound into the function: as an argument dask would read it as a reference to the task itself.
        return da.from_delayed(dask.delayed(partial(self._get, key, load), pure=False)(dask_key_name=key),
                               shape=shape, dtype=dtype)

    def stack(self, source, stack):
        """The ZYX dask array of a source file: one chunk per page when its pages are its Z planes, else
        one task for the whole file in the chunks of tifffile's Zarr view (the chunks the fusion used
        before it had its own reader)."""
        # W-432; the reader before W-397
        import dask.array as da
        import tifffile
        import zarr
        tif, lock = self._file(source.path, stack)
        n_z, n_y, n_x = source.shape
        if _paged(tif, source.shape):
            planes = [self._task(self._key(source, "page", i), partial(_read_page, tif, lock, i, (1, n_y, n_x)),
                                 (1, n_y, n_x), source.dtype) for i in range(n_z)]
            return da.concatenate(planes, axis=0) if n_z > 1 else planes[0]
        with tifffile.imread(source.path, aszarr=True) as store:
            chunks = da.from_zarr(zarr.open(store, mode="r")).reshape(source.shape).chunks
        array = self._task(self._key(source, "file"), partial(_read_file, tif, lock, source.shape),
                           source.shape, source.dtype)
        return array.rechunk(chunks) if array.chunks != chunks else array

    def projection(self, source, stack, projection):
        """The projection along Z of a source file, as one task that reads its pages."""
        from starfinder.preprocessing.projection import ProjectionConfig, project_image
        tif, lock = self._file(source.path, stack)
        config = ProjectionConfig(method=projection, axis="z", output_dtype="float32" if projection == "sum" else None)
        dtype = project_image(np.zeros((1, 1, 1), source.dtype), config=config)[0].dtype
        load = partial(_read_projection, tif, lock, source.shape, config)
        return self._task(self._key(source, f"projection-{projection}"), load, source.shape[1:], dtype)

    def _get(self, key, load):
        """A task's result: held from its first call while the plan counts later calls, read once."""
        with self._lock:
            task_lock = self._task_locks.setdefault(key, threading.Lock())
        with task_lock:
            with self._lock:
                value = self._held.get(key)
            if value is None:
                value = load()
            with self._lock:
                remaining = self._remaining.get(key, 0) - 1
                self._remaining[key] = remaining
            if remaining > 0:
                self._hold(key, value)
            else:
                self._release(key)
            return value

    def _hold(self, key, value):
        with self._lock:
            self._held[key] = value

    def _release(self, key):
        with self._lock:
            self._held.pop(key, None)

    # The plan.

    def plan(self, array) -> None:
        """Count, for each source task, the bands of ``array`` the writer computes (``_ome.bands``) that need it."""
        from dask.core import get_dependencies
        from dask.utils import ensure_dict
        graph = ensure_dict(array.__dask_graph__())
        found = {}   # key -> frozenset of the source tasks it depends on (memoized over the whole graph)
        empty = frozenset()

        def tasks(key):
            todo = [key]
            while todo:
                current = todo[-1]
                if current in found:
                    todo.pop()
                    continue
                if isinstance(current, str) and current.startswith(self._prefix):
                    found[current] = frozenset((current,))
                    todo.pop()
                    continue
                dependencies = get_dependencies(graph, current)
                pending = [d for d in dependencies if d not in found]
                if pending:
                    todo.extend(pending)
                    continue
                todo.pop()
                sets = {found[d] for d in dependencies} - {empty}
                found[current] = empty if not sets else next(iter(sets)) if len(sets) == 1 else frozenset().union(*sets)
            return found[key]

        keys = array.__dask_keys__()

        def block_key(index):
            value = keys
            for i in index:
                value = value[i]
            return value

        edges = [np.cumsum((0,) + tuple(c)) for c in array.chunks]

        def blocks(axis, lo, hi):
            return [i for i in range(len(edges[axis]) - 1) if edges[axis][i] < hi and edges[axis][i + 1] > lo]

        counts = {}
        for z, y0, y1 in _ome.bands(array.shape):
            axes = ([blocks(0, z, z + 1)] if array.ndim == 3 else []) + [blocks(array.ndim - 2, y0, y1),
                                                                          range(array.numblocks[-1])]
            needed = set()
            for index in itertools.product(*axes):
                needed |= tasks(block_key(index))
            for key in needed:
                counts[key] = counts.get(key, 0) + 1
        with self._lock:
            for key, count in counts.items():
                self._remaining[key] = max(self._remaining.get(key, 0), 0) + count

    def clear(self) -> None:
        """Release every held result and forget the plan (the end of one fused file)."""
        with self._lock:
            self._held.clear()
            self._remaining.clear()


def _read_page(tif, lock, index, shape):
    """One page of a source file, decoded, as a one-plane ZYX array."""
    with lock:
        return tif.series[0].pages[index].asarray().reshape(shape)


def _paged(tif, shape) -> bool:
    """Whether the pages of a file's first series are its Z planes, in order (else the file is read whole)."""
    pages = tif.series[0].pages
    return len(pages) == shape[0] and all(page is not None for page in pages) \
        and math.prod(pages[0].shape) == shape[1] * shape[2]


def _read_file(tif, lock, shape):
    """A whole source file as ZYX: page by page when its pages are its planes, each page decoded once."""
    if _paged(tif, shape):
        return np.concatenate([_read_page(tif, lock, i, (1,) + tuple(shape[1:])) for i in range(shape[0])])
    with lock:
        return tif.series[0].asarray().reshape(shape)


def _read_projection(tif, lock, shape, config):
    """The projection along Z of a whole source file."""
    from starfinder.preprocessing.projection import project_image
    return project_image(_read_file(tif, lock, shape), config=config)[0]


def _lazy(source, stack, reader=None):
    """A ZYX dask array of a source: a file read lazily by ``reader`` (one task per page), or an array."""
    import dask.array as da
    if source.path is None:
        value = source.value
        return da.from_array(value if value.ndim == 3 else value[None])
    return (reader or _Reader()).stack(source, stack)


def _fov_image(source, lazy, grid, dimensionality, projection, reader=None, stack=None):
    """The array the library gets for one FOV: the stack (lazy), its plane, or its projection along Z (lazy for a
    file read by ``reader``)."""
    from starfinder.preprocessing.projection import ProjectionConfig, project_image
    if dimensionality == "projection":
        if source.path is not None and reader is not None:
            return reader.projection(source, stack, projection)
        config = ProjectionConfig(method=projection, axis="z", output_dtype="float32" if projection == "sum" else None)
        return project_image(np.asarray(lazy), config=config)[0]
    return lazy[0] if grid.one_plane else lazy


def _sims(plan, lazies, name, dimensionality, config, reader=None, stack=None):
    """The library's input images of one file, in placement table order (level 1 of each for k > 1)."""
    from multiview_stitcher import msi_utils
    from multiview_stitcher import spatial_image_utils as si_utils
    grid, k = plan.grid, config.factor_yx
    dims = ["y", "x"] if dimensionality == "projection" or grid.one_plane else ["z", "y", "x"]
    sims = []
    for fov in plan.ids:
        source = plan.sources[(fov, name)]
        array = _fov_image(source, lazies[(fov, name)], grid, dimensionality, config.projection, reader, stack)
        spacing = plan.by_fov[fov].grid.metadata.spacing_zyx
        origin = plan.placement_origins[fov]
        sim = si_utils.get_sim_from_array(array, dims=dims, scale={a: float(spacing[AXES.index(a)]) for a in dims},
                                          translation={a: float(origin[AXES.index(a)]) for a in dims},
                                          transform_key=TRANSFORM_KEY)
        if k > 1:
            msim = msi_utils.get_msim_from_sim(sim, scale_factors=[{"y": k, "x": k}])
            sim = msi_utils.get_sim_from_msim(msim, scale="scale1")
        sims.append(sim)
    return sims, dims


def output_stack_properties(grid: FusedGrid, dimensionality: str, dims) -> dict:
    """The output grid passed to the library: spacing, origin and shape of the output level on the axes fused."""
    spacing, origin, shape = grid.level(dimensionality)
    return {"spacing": {a: float(spacing[AXES.index(a)]) for a in dims},
            "origin": {a: float(origin[AXES.index(a)]) for a in dims},
            "shape": {a: int(shape[AXES.index(a)]) for a in dims}}


def _fuse_file(plan, lazies, name, dimensionality, config, reader=None, stack=None):
    """One fused image, lazily: (the library's fused image, dims, the recorded arguments)."""
    from multiview_stitcher import fusion
    sims, dims = _sims(plan, lazies, name, dimensionality, config, reader, stack)
    properties = output_stack_properties(plan.grid, dimensionality, dims)
    chunks = CHUNKS["stack" if len(dims) == 3 else "plane"]
    fused = fusion.fuse(sims, transform_key=TRANSFORM_KEY, fusion_func=fusion.weighted_average_fusion,
                        interpolation_order=INTERPOLATION_ORDER, output_stack_properties=properties,
                        output_chunksize=chunks)
    return fused, dims, {"output_stack_properties": properties, "output_chunksize": chunks}


def _library_grid(fused):
    from multiview_stitcher import spatial_image_utils as si_utils
    dims = si_utils.get_spatial_dims_from_sim(fused)
    return {"spacing": {a: float(v) for a, v in si_utils.get_spacing_from_sim(fused).items()},
            "origin": {a: float(v) for a, v in si_utils.get_origin_from_sim(fused).items()},
            "shape": {a: int(fused.sizes[a]) for a in dims}}


def _level_factors(shapes):
    """Per level and axis the factor 2^n of the halvings from the output level (an axis kept keeps its factor)."""
    factors, current = [], [1] * len(shapes[0])
    for previous, shape in zip(shapes, shapes[1:]):
        current = [f * (2 if b != a else 1) for f, a, b in zip(current, previous, shape)]
        factors.append(list(current))
    return factors


def level_geometry(spacing, origin, factors) -> tuple[list[float], list[float]]:
    """``S^(ℓ) = f·S`` and ``O^(ℓ) = O + S·(f − 1)/2`` per axis."""
    return ([f * s for f, s in zip(factors, spacing)],
            [o + s * (f - 1) / 2 for f, s, o in zip(factors, spacing, origin)])


def _write_tiff(path, fused, grid, dimensionality, config, name, placement, reader=None):
    from multiview_stitcher import msi_utils
    data = fused.data[0, 0]
    if reader is not None:
        reader.plan(data)
    spacing, origin, shape = grid.level(dimensionality)
    metadata = grid.level_metadata(dimensionality)
    levels, level_entries = [], []
    if config.pyramid_levels:
        msim = msi_utils.get_msim_from_sim(fused, scale_factors=[{"y": 2, "x": 2}] * config.pyramid_levels)
        for level in range(1, config.pyramid_levels + 1):
            levels.append(msi_utils.get_sim_from_msim(msim, scale=f"scale{level}").data[0, 0])
    for level, array in enumerate(levels, start=1):
        factors = [1, 2 ** level, 2 ** level]
        level_spacing, level_origin = level_geometry(spacing, origin, factors)
        level_entries.append({"level": level, "shape_zyx": [1] * (3 - array.ndim) + list(array.shape),
                              "factors_zyx": factors, "spacing_zyx": level_spacing, "origin_zyx": level_origin})
    from starfinder.stitching._placement import table_sha256
    annotation = {"sample": grid.sample, "image": name, "dimensionality": dimensionality,
                  "projection": config.projection if dimensionality == "projection" else None,
                  "factor_yx": grid.factor_yx, "placement_fovs_sha256": table_sha256(placement.fovs),
                  "placement_links_sha256": table_sha256(placement.links)}
    layout = _ome.write_ome_tiff(path, data, metadata=metadata, annotation=annotation, levels=levels)
    return {"levels": level_entries, "tile_yx": layout["tile_yx"], "compression": layout["compression"],
            "axes": layout["axes"], "annotation": annotation}


def _write_zarr(path, fused, dimensionality, config, spacing, origin):
    """The library's standalone OME-Zarr of a fused image, rechunked first, written into a staging folder."""
    import zarr
    from multiview_stitcher import ngff_utils
    from multiview_stitcher import spatial_image_utils as si_utils
    dims = si_utils.get_spatial_dims_from_sim(fused)
    chunks = ZARR_CHUNKS["stack" if "z" in dims else "projection"]
    chunks = {d: c for d, c in chunks.items() if d in fused.dims}
    tmp = atomic_path(path)
    try:
        ngff_utils.write_sim_to_ome_zarr(fused.chunk(chunks), str(tmp), overwrite=True,
                                         ngff_version=config.ome_zarr_version, show_progressbar=False)
        group = zarr.open_group(str(tmp), mode="r")
        attrs = dict(group.attrs)
        multiscales = (attrs.get("ome") or attrs)["multiscales"][0]
        arrays = [group[d["path"]] for d in multiscales["datasets"]]
        shapes = [list(a.shape[-len(dims):]) for a in arrays]
        factors = [[1] * len(dims)] + _level_factors(shapes)
        spatial = [AXES.index(a) for a in dims]
        levels = []
        for d, array, f in zip(multiscales["datasets"], arrays, factors):
            level_spacing, level_origin = level_geometry([spacing[i] for i in spatial], [origin[i] for i in spatial], f)
            codecs = _codec(array)
            levels.append({"path": d["path"], "shape": list(array.shape), "dtype": str(array.dtype),
                           "chunks": list(array.chunks), "codec": codecs, "factors": dict(zip(dims, f)),
                           "spacing": dict(zip(dims, level_spacing)), "origin": dict(zip(dims, level_origin))})
        files = directory_files(tmp)
        publish(tmp, path)
    finally:
        if Path(tmp).exists():
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    return {"files": files, "sha256": section_sha256(files), "zarr_format": int(arrays[0].metadata.zarr_format),
            "ngff_version": config.ome_zarr_version, "axes": [a["name"] for a in multiscales["axes"]],
            "levels": levels, "chunks": levels[0]["chunks"], "compression": levels[0]["codec"]}


def _codec(array):
    """The compressor of a Zarr array as text (format 2: ``compressor``; format 3: the bytes-to-bytes codecs)."""
    metadata = array.metadata
    if getattr(metadata, "zarr_format", None) == 2:
        compressor = metadata.compressor
        return None if compressor is None else repr(compressor.get_config() if hasattr(compressor, "get_config")
                                                    else compressor)
    return repr([c.to_dict() if hasattr(c, "to_dict") else repr(c) for c in metadata.codecs])


def fuse_images(placement: PlacementResult, inputs: Sequence[FusionInput], *, assembly=None,
                config: FusionConfig = FusionConfig(), output_dir) -> FusionRecord:
    """Fuse a sample's source images with multiview-stitcher and write them to its export folder.

    Applies the checks of docs/sample-output-contract.md before anything is
    written (review, an assembly of this placement, one FusionInput per
    included FOV on the grid the placement recorded, one spacing, one
    dimensionality, the dtypes uint8, uint16 and float32, the sum bound and
    the pyramid bound), then, per image name of ``config.images`` and per file
    of ``config.dimensionality``, builds the library's inputs (level 1 of each
    FOV for ``scale_yx = 1/k`` with ``k > 1``; each FOV's projection along Z
    for the projection), calls ``multiview_stitcher.fusion.fuse`` once with
    ``weighted_average_fusion`` at order 1 and the output grid of
    :func:`fused_grid`, and writes ``<image>_projection.ome.tif`` or
    ``<image>_stack.ome.tif`` (BigTIFF, 512 × 512 zlib tiles, band by band,
    ``pyramid_levels`` SubIFD levels), and with ``ome_zarr`` the library's
    ``.ome.zarr`` beside it. Finally it replaces the ``fusion`` section of
    ``output_dir/export.json``, leaving the other sections unchanged, after
    removing the files the earlier section listed.

    Without an included FOV it writes no image and records ``empty``
    (``no_included_fov``). When the ``multiview-stitcher`` extra is not
    importable, or a selected source image is missing, it writes no image,
    records ``not_run`` with the reason and issues
    :class:`ExportIncompleteWarning`. Any other error while fusing or writing
    records ``failed`` with the files completed and is raised again.

    Parameters
    ----------
    placement : PlacementResult
        Any placement; one whose review is required must hold an approval.
    inputs : Sequence[FusionInput]
        Exactly one per included FOV (``image_supported`` or ``metadata_only``).
    assembly : AssemblyResult or None
        Optional, only recorded (the SHA-256 of its section); its placement
        SHA-256 values must equal the placement's.
    config : FusionConfig
        Images, dimensionality, projection, scale and file options.
    output_dir : str or Path
        The sample's export folder ``<OUTPUT>/export/<sample>/``.

    Returns
    -------
    FusionRecord

    Raises
    ------
    TypeError
        An argument of the wrong type.
    PlacementReviewRequiredError
        Review is required and the placement holds no approval.
    ValueError
        An assembly of another placement; a missing, repeated or foreign
        FusionInput; two spacings or one-plane beside multi-plane FOVs; a dtype
        other than uint8, uint16 and float32, or mixed dtypes; a sum projection
        above the bound 2^24; ``pyramid_levels`` above
        ``floor(log2(max(N_y, N_x)))``.
    IncompatibleGeometryError
        An input grid that is not the grid the placement recorded, or without
        known spacing, unit µm and identity direction, or a source image of
        another shape than its grid.
    """
    timer = Timer()
    plan = check_fusion(placement, inputs, config=config, assembly=assembly)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_listed(output_dir, listed_files(read_export(output_dir), "fuse"))

    def finish(status, outputs, *, version=None, warn=None, **extra):
        section = _section(status, placement, plan, config, assembly, outputs, timer, version=version, **extra)
        update_section(output_dir, placement.sample, "fuse", section)
        if warn:
            warnings.warn(f"export part fuse could not run: {warn}", ExportIncompleteWarning, stacklevel=3)
        return FusionRecord(status, section, tuple(output_dir / o["path"] for o in outputs))

    if plan.grid is None:
        return finish("empty", [], reason="no_included_fov")
    try:
        from importlib.metadata import version as package_version

        import multiview_stitcher  # noqa: F401
        version = package_version("multiview-stitcher")
    except ImportError:
        return finish("not_run", [], reason="extra_missing", extra=EXTRA,
                      warn=f"extra_missing: the {EXTRA!r} extra is not installed")
    if plan.missing:
        described = ", ".join(f"{m['fov_id']} {m['image']}" for m in plan.missing)
        return finish("not_run", [], version=version, reason="source_image_missing", missing=plan.missing,
                      warn=f"source_image_missing ({described})")
    outputs = []
    try:
        with ExitStack() as stack:
            reader = _Reader()
            stack.callback(reader.clear)
            lazies = {key: _lazy(source, stack, reader) for key, source in plan.sources.items()}
            for name in config.image_names:
                for dimensionality in config.dimensionalities:
                    reader.clear()
                    fused, dims, arguments = _fuse_file(plan, lazies, name, dimensionality, config, reader, stack)
                    spacing, origin, shape = plan.grid.level(dimensionality)
                    dtype = np.dtype(fused.dtype).name
                    file_name = f"{name}_{dimensionality}.ome.tif"
                    entry = {"path": file_name, "format": "ome_tiff", "image": name, "dimensionality": dimensionality,
                             "projection": config.projection if dimensionality == "projection" else None,
                             "factor_yx": plan.grid.factor_yx, "shape_zyx": list(shape), "dtype": dtype,
                             "spacing_zyx": list(spacing), "origin_zyx": list(origin),
                             "slab_z": [origin[0] - spacing[0] / 2, origin[0] + spacing[0] / 2]
                             if shape[0] == 1 else None, "inputs_2d": len(dims) == 2,
                             "library_grid": _library_grid(fused), **arguments}
                    entry.update(_write_tiff(output_dir / file_name, fused, plan.grid, dimensionality, config, name,
                                             placement, reader))
                    from starfinder._records import file_sha256
                    entry["sha256"] = file_sha256(output_dir / file_name)
                    outputs.append(entry)
                    if config.ome_zarr:
                        zarr_name = f"{name}_{dimensionality}.ome.zarr"
                        written = _write_zarr(output_dir / zarr_name, fused, dimensionality, config, spacing, origin)
                        outputs.append({"path": zarr_name, "format": "ome_zarr", "image": name,
                                        "dimensionality": dimensionality, "projection": entry["projection"],
                                        "factor_yx": entry["factor_yx"], "shape_zyx": list(shape), "dtype": dtype,
                                        "spacing_zyx": list(spacing), "origin_zyx": list(origin), **written})
    except BaseException as error:
        finish("failed", outputs, version=version, error={"type": type(error).__name__, "message": str(error)})
        raise
    return finish("written", outputs, version=version)
