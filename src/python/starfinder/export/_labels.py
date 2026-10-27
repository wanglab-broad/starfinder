"""The assembled label export: the retained cells of the sample, each drawn whole, placed without blending.

docs/sample-output-contract.md ("Assembled label export"), rule LR4. Each FOV with a
retained cell has its territory image (the one assign sampled) mapped from ``cell_id`` to
the ``instance_id`` of its retained cells and to 0 elsewhere; these retained-cell images,
in placement table order, are fused by multiview-stitcher's ``fuse`` at interpolation
order 0 with ``_first_positive``, a small function passed to the library, on a grid
requested by ``output_spacing`` at ``LabelConfig.scale_yx``. Labels are never blended,
no owner mask is used and no ID is shifted. The result is written as ``cell_labels.ome.tif``
(uint32) with the ``labels`` section of ``export.json``. The library is imported only when
:func:`fuse_labels` fuses.

On the route after fusion (docs/sample-segmentation-contract.md, "The label images") the
one view passed to the same library call is the sample's territories (the cell run's label
file, or ``territories.ome.tif`` when assign expanded) with ``cell_id`` mapped to
``instance_id``; the section records the rule ``segmentation`` and ``segmentation_files``.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import _ome
from starfinder.segmentation._operations import retained_image
from starfinder.stitching import PlacementResult
from starfinder.stitching._placement import grid_problems

from ._config import FusionConfig, LabelConfig
from ._grid import AXES, IDENTITY, _names, fused_grid, included_fovs
from ._record import (
    ExportIncompleteWarning,
    Timer,
    as_json,
    listed_files,
    read_export,
    remove_listed,
    section_sha256,
    update_section,
)

EXTRA = "multiview-stitcher"
TRANSFORM_KEY = "placed"
RULE = "retained_cells"
#: The rule of the route after fusion: the sample's territories as the one view.
SEGMENTATION_RULE = "segmentation"
FUSION_FUNC = "_first_positive"
INTERPOLATION_ORDER = 0
ROUTE = "output_spacing"
LABEL_FILE = "cell_labels.ome.tif"
#: 2^24: every integer up to it is exact in float32, through which ``fuse`` passes the values.
LABEL_ID_BOUND = 2 ** 24
TOLERANCE_UM = 1e-9
#: Output chunks of the lazy label image, as for the fused image.
CHUNKS = {"stack": {"z": 1, "y": 512, "x": 512}, "plane": {"y": 512, "x": 512}}


@dataclass(frozen=True)
class LabelRecord:
    """The ``labels`` section ``fuse_labels`` wrote to ``export.json`` and the files it wrote.

    ``status`` is ``written``, ``empty``, ``not_run`` or ``failed``;
    ``section`` the section as written (JSON values); ``paths`` the written
    files (``cell_labels.ome.tif``, or none).
    """

    status: str
    section: Any
    paths: tuple[Path, ...]


def _first_positive(transformed_views):
    """The fusion function passed to the library: per voxel the first view's value that is valid and above 0.

    Views come in placement table order; a voxel where no view holds a valid
    positive value is NaN, which the library writes as 0. It selects one
    input value per voxel and computes none.
    """
    out = np.full(transformed_views[0].shape, np.nan, dtype=transformed_views.dtype)
    for view in transformed_views:
        take = np.isnan(out) & ~np.isnan(view) & (view > 0)
        out[take] = view[take]
    return out


def check_label_ids(cells) -> None:
    """Check 6 of the entry: at most 2^24 = 16,777,216 cell rows, so every ``instance_id`` is exact in float32.

    Reads the number of rows only. Raises ValueError naming the count.
    """
    check_label_count(len(cells))


def check_label_count(n: int) -> None:
    """L6 for ``n`` cell rows: ValueError naming the count above 2^24. On the route after fusion ``Dataset.export``
    calls it with the number of labels of the cell run, one cell row each, when that number is first known."""
    if n > LABEL_ID_BOUND:
        raise ValueError(f"the sample cell table has {n} rows, above 2^24 = {LABEL_ID_BOUND}: its instance_id values "
                         "would not be exact in float32, so the label export is refused")


def instance_ids(cells) -> np.ndarray:
    """The ``instance_id`` of each row of the sample cell table: ``1..N`` in its order (uint32)."""
    return np.arange(1, len(cells) + 1, dtype=np.uint32)


def retained_by_fov(cells) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per FOV the ``cell_id`` of its retained cells (sorted, int64) and their ``instance_id`` (uint32).

    A cell is retained when the cell table says ``assembly_status ==
    "retained"``; ownership is not recomputed.
    """
    if len(cells) == 0:
        return {}
    keep = cells["assembly_status"].eq("retained").fillna(False).to_numpy(dtype=bool)
    ids = instance_ids(cells)[keep]
    fovs = cells["fov_id"].astype(object).to_numpy()[keep]
    cell_ids = cells["cell_id"].astype("int64").to_numpy()[keep]
    out = {}
    for fov in dict.fromkeys(fovs.tolist()):
        rows = fovs == fov
        order = np.argsort(cell_ids[rows], kind="stable")
        out[fov] = (cell_ids[rows][order], ids[rows][order])
    return out


def label_grid(spacing_zyx, origins, shapes, k) -> dict:
    """The formula label grid in Y and X: spacing ``k·s``, origin ``o``, size ``floor((max_f (origin_f +
    (n_f − 1)·s) − o) / (k·s) + 1e-9) + 1`` (docs/sample-output-contract.md, "Scale factor 0.25")."""
    grid = {"spacing": {}, "origin": {}, "shape": {}}
    for i, a in ((1, "y"), (2, "x")):
        s = float(spacing_zyx[i])
        o = min(float(origin[i]) for origin in origins)
        last = max(float(origin[i]) + (shape[i] - 1) * s for origin, shape in zip(origins, shapes))
        grid["spacing"][a] = k * s
        grid["origin"][a] = o
        grid["shape"][a] = int(math.floor((last - o) / (k * s) + 1e-9)) + 1
    return grid


def _assembly_fovs(assembly):
    return {entry["fov_id"]: entry for entry in assembly.record.get("fovs") or []}


def _territory_source(entry):
    """``(path, sha256, source)`` of the territory file the label export reads for one FOV of the assembly record.

    The path the assembly recorded beside its SHA-256 (``conflicts=True`` and a
    saved assignment folder), else the file the assignment loader resolves from the
    recorded assignment folder (the expanded territories when assign expanded,
    the cell labels otherwise) with the SHA-256 its record holds. ``path`` is
    None for an assignment given in memory. Raises FileNotFoundError, carrying
    the path, when the folder's ``assignment.json`` is missing.
    """
    # §2.9
    from starfinder.assignment._persist import RECORD_FILE, _path
    from starfinder._records import file_sha256
    territories = entry.get("territories") or {}
    if territories.get("path"):
        return Path(territories["path"]), territories["sha256"], "assembly_record"
    assignment = entry.get("assignment") or {}
    folder = assignment.get("folder")
    if not folder:
        return None, None, "memory"
    folder = Path(folder)
    record_path = folder / RECORD_FILE
    if not record_path.is_file():
        raise FileNotFoundError(record_path)
    recorded = assignment.get("record_sha256")
    if recorded is not None and file_sha256(record_path) != recorded:
        raise ValueError(f"assignment record {record_path} has SHA-256 {file_sha256(record_path)}, the assembly "
                         f"record {recorded}")
    record = json.loads(record_path.read_text())
    files = record["files"]
    file_entry = files["territories"] if "territories" in files else record["inputs"]["cells"]["file"]
    return _path(folder, file_entry), file_entry["sha256"], "assignment_folder"


def _read_territory(path, sha256):
    """The territory labels of a file, its SHA-256 checked first (ValueError naming the path and both hashes)."""
    from starfinder.assignment import _persist as assignment_persist
    from starfinder._records import check_file
    check_file(path, sha256, "territory file")
    labels, _ = assignment_persist.read_labels(path)
    return labels


def _grids(placement, assembly, ids):
    """The ReferenceGrid of each included FOV: the grid the placement recorded, else the assembly's."""
    from starfinder.assembly._ownership import recorded_grid
    from starfinder.segmentation import ReferenceGrid
    recorded = _assembly_fovs(assembly)
    grids, without = {}, []
    for fov in ids:
        found = recorded_grid(placement, fov)
        if found is None and isinstance((recorded.get(fov) or {}).get("grid"), dict):
            grid = recorded[fov]["grid"]
            found = tuple(int(n) for n in grid["shape_zyx"]), ImageMetadata(**grid["metadata"])
        if found is None:
            without.append(fov)
            continue
        grids[fov] = ReferenceGrid(found[0], found[1], "declared")
    if without:
        raise ValueError(f"fuse_labels needs the grid of every included FOV; FOVs without one: {_names(without)}")
    bad = [fov for fov in ids if grid_problems(grids[fov].metadata)]
    if bad:
        raise IncompatibleGeometryError("fuse_labels needs known spacing, unit µm and identity direction on every "
                                        f"grid; FOVs without: {_names(bad)}")
    return grids


def _check(placement, assembly, config):
    from starfinder.assembly import AssemblyResult
    from starfinder.stitching._review import require_approval

    from ._fuse import check_assembly
    if not isinstance(placement, PlacementResult):
        raise TypeError("placement must be a PlacementResult")
    if not isinstance(assembly, AssemblyResult):
        raise TypeError("assembly must be an AssemblyResult")
    if not isinstance(config, LabelConfig):
        raise TypeError("config must be a LabelConfig")
    config.__post_init__()
    require_approval(placement)
    check_assembly(placement, assembly)
    check_label_ids(assembly.cells)


def after_fusion(assembly) -> bool:
    """Whether an assembly holds the cells of a sample segmentation (the route after fusion, ``mode`` ``cells``)."""
    return assembly.record.get("route") == "after_fusion" and assembly.record.get("mode") == "cells"


def _section(status, placement, assembly, config, timer, **extra):
    from starfinder.stitching._placement import table_sha256

    from starfinder._records import software
    flags = assembly.record.get("flags") or {}
    segmentation = after_fusion(assembly)
    section = {"status": status, **extra,
               "assembly_sha256": section_sha256(as_json(assembly.record)),
               "rule": SEGMENTATION_RULE if segmentation else RULE,
               "placement": {"fovs_sha256": table_sha256(placement.fovs),
                             "links_sha256": table_sha256(placement.links)},
               "conflicts_computed": bool(flags.get("conflicts", False)) and not segmentation,
               "labels_scale_yx": config.scale_yx,
               "factor_yx": config.factor_yx, "software": software(), **timer.record(),
               "viewer_qualification": "deferred"}
    if segmentation:
        # The label files the assembly read (each run's, and territories.ome.tif when assign expanded): T5.
        section["segmentation_files"] = list(assembly.record.get("segmentation_files") or [])
    section.setdefault("outputs", [])
    return as_json(section)


def sample_territory_source(assembly):
    """``(path, sha256, run)`` of the sample's territories, relative to the export folder, on the route after fusion.

    ``territories.ome.tif`` when the sample assignment expanded the cells
    (its entry of ``segmentation_files`` has run null), else the label file of
    the cell run (the run the sample assignment's ``territory_source`` names).
    Raises ValueError when the assembly record names neither.
    """
    files = list(assembly.record.get("segmentation_files") or [])
    expanded = [e for e in files if e.get("run") is None]
    if expanded:
        return expanded[0]["path"], expanded[0]["sha256"], None
    run = ((assembly.record.get("assignment") or {}).get("territory_source") or {}).get("run")
    entry = next((e for e in files if e.get("run") == run), None)
    if run is None or entry is None:
        raise ValueError(f"the assembly record names no label file of its cell run {run!r} among its "
                         f"segmentation_files {[e.get('path') for e in files]}")
    return entry["path"], entry["sha256"], run


def sample_label_grid(shape_zyx, spacing_zyx, origin_zyx, output_spacing_yx) -> dict:
    """The label grid of the route after fusion in Y and X: the output spacing (``k·s`` at the label scale, ``s``
    the acquired spacing), the territories' origin ``O``, size ``floor((N − 1)·S / (k·s) + 1e-9) + 1`` for
    territories of ``N`` pixels of spacing ``S`` (docs/sample-segmentation-contract.md, "The label images")."""
    grid = {"spacing": {}, "origin": {}, "shape": {}}
    for i, a, out in ((1, "y", output_spacing_yx[0]), (2, "x", output_spacing_yx[1])):
        grid["spacing"][a] = float(out)
        grid["origin"][a] = float(origin_zyx[i])
        grid["shape"][a] = int(math.floor((int(shape_zyx[i]) - 1) * float(spacing_zyx[i]) / float(out) + 1e-9)) + 1
    return grid


def _acquired_spacing(placement, assembly):
    """The acquired Y and X spacing ``s`` of the included FOVs (the grid the placement or the assembly records)."""
    ids = included_fovs(placement)
    grids = _grids(placement, assembly, ids)
    spacings = {tuple(float(v) for v in grids[f].metadata.spacing_zyx[1:]) for f in ids}
    if len(spacings) != 1:
        raise ValueError(f"fuse_labels needs one Y and X spacing of the included FOVs; they have {sorted(spacings)}")
    return next(iter(spacings))


def fuse_labels(placement: PlacementResult, assembly, *, config: LabelConfig = LabelConfig(),
                output_dir) -> LabelRecord:
    """Write the assembled label image of a sample: its retained cells, each drawn whole, without blending.

    Rule LR4 of docs/sample-output-contract.md ("Assembled label export"). Per
    included FOV in placement table order, its retained-cell image: the
    territory image assign sampled, read from the path the assembly record
    holds beside its SHA-256 (or, when that path is null, from the file the
    assignment loader resolves from the recorded assignment folder), with each
    ``cell_id`` of a cell the cell table marks ``retained`` in that FOV mapped
    to its ``instance_id`` (``1..N`` in cell table order) and every other value
    to 0. A FOV without a retained cell is not read and contributes zeros. The
    images are fused by ``multiview_stitcher.fusion.fuse`` at interpolation
    order 0 with ``_first_positive`` (the first positive view in table
    order; NaN, written as 0, where none is), at ``output_spacing`` ``k·s`` in
    Y and X (``scale_yx = 1/k``) and ``s_z`` in Z, as 2D for ``plane``
    territories and one-plane FOVs and as 3D for ``volume`` and ``extended``
    territories of multi-plane FOVs. The grid the library returns is checked
    against the formula grid (L5) within 1e-9 µm. The image is written as
    ``cell_labels.ome.tif`` (uint32, the OME-TIFF layout of the fused image; Z
    the slab for a 2D image), and the ``labels`` section of
    ``output_dir/export.json`` is replaced, leaving the other sections
    unchanged, after removing the files the earlier section listed.

    On the route after fusion (an assembly with ``route`` ``after_fusion``
    and ``mode`` ``cells``; docs/sample-segmentation-contract.md, "The label
    images") the one view is the sample's territories, the file of
    ``output_dir`` that the assembly record lists in ``segmentation_files``
    (``territories.ome.tif`` when assign expanded, else the cell run's
    ``segmentation_<run>.ome.tif``), its SHA-256 checked, with each retained
    cell's ``cell_id`` mapped to its ``instance_id``, passed to the same
    library call on the segmentation grid at the same ``output_spacing``
    ``k·s`` (``s`` the FOVs' acquired spacing). The grid is checked against
    the formula of that page (the territories' origin) and Z is the
    segmentation grid's slab. The section records the rule ``segmentation``,
    ``segmentation_files`` (the assembly's) and no FOV entry.

    A ``molecules_only`` sample writes no image and records ``empty``
    (``no_cells``); a placement without an included FOV records ``empty``
    (``no_included_fov``). When the ``multiview-stitcher`` extra is not
    importable, or a territory file of a FOV with a retained cell is missing,
    it writes no image, records ``not_run`` with the reason and issues
    :class:`ExportIncompleteWarning`. Any other error (a territory file whose
    SHA-256 differs from the record, a library error) records ``failed`` with
    the exception's type and message and is raised again.

    Parameters
    ----------
    placement : PlacementResult
        The placement the assembly was made from; one whose review is
        required must hold an approval.
    assembly : AssemblyResult
        The assembled sample, typically ``load_assembly(output_dir)``.
    config : LabelConfig
        ``scale_yx = 1/k`` in Y and X (default 0.25).
    output_dir : str or Path
        The sample's export folder ``<OUTPUT>/export/<sample>/``.

    Returns
    -------
    LabelRecord

    Raises
    ------
    TypeError
        An argument of the wrong type.
    PlacementReviewRequiredError
        Review is required and the placement holds no approval.
    ValueError
        An assembly of another placement; more than 16,777,216 cell rows;
        included FOVs of different spacings or planes; territories of mixed
        geometry; a territory file whose SHA-256 differs from the record; a
        library grid other than the formula grid.
    IncompatibleGeometryError
        A grid without known spacing, unit µm and identity direction, or a
        territory of another shape than its FOV's grid.
    """
    # §2.9
    timer = Timer()
    _check(placement, assembly, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_listed(output_dir, listed_files(read_export(output_dir), "fuse_labels"))

    def finish(status, *, warn=None, **extra):
        section = _section(status, placement, assembly, config, timer, **extra)
        update_section(output_dir, placement.sample, "fuse_labels", section)
        if warn:
            warnings.warn(f"export part fuse_labels could not run: {warn}", ExportIncompleteWarning, stacklevel=3)
        return LabelRecord(status, section, tuple(output_dir / o["path"] for o in section["outputs"]))

    if assembly.record.get("mode") == "molecules_only":
        return finish("empty", reason="no_cells")
    ids = included_fovs(placement)
    if not ids:
        return finish("empty", reason="no_included_fov")
    try:
        from importlib.metadata import version as package_version

        import multiview_stitcher  # noqa: F401
        version = package_version("multiview-stitcher")
    except ImportError:
        return finish("not_run", reason="extra_missing", extra=EXTRA,
                      warn=f"extra_missing: the {EXTRA!r} extra is not installed")
    library = {"multiview_stitcher": version, "fusion_func": FUSION_FUNC,
               "interpolation_order": INTERPOLATION_ORDER, "route": ROUTE, "transform_key": TRANSFORM_KEY}
    fovs, missing, outputs = [], [], []
    try:
        if after_fusion(assembly):
            # The route after fusion: the sample's territories as the one view.
            relative, sha256, _ = sample_territory_source(assembly)
            if not (output_dir / relative).is_file():
                missing.append({"fov_id": None, "path": str(output_dir / relative)})
                return finish("not_run", reason="territory_missing", missing=missing, library=library,
                              warn=f"territory_missing (the sample's territories {output_dir / relative})")
            image, entry = _fuse_sample_and_write(placement, assembly, config, output_dir, relative, sha256)
            outputs.append(entry)
            return finish("written", library=library, fovs=fovs, outputs=outputs, **image)
        retained = retained_by_fov(assembly.cells)
        recorded = _assembly_fovs(assembly)
        for fov in ids:
            if fov not in retained:
                continue
            try:
                path, sha256, source = _territory_source(recorded.get(fov) or {})
            except FileNotFoundError as error:
                missing.append({"fov_id": fov, "path": str(error.args[0]) if error.args else None})
                continue
            if path is None or not path.is_file():
                missing.append({"fov_id": fov, "path": None if path is None else str(path)})
                continue
            fovs.append({"fov_id": fov, "path": str(path), "sha256": sha256, "source": source,
                         "n_retained": int(len(retained[fov][0]))})
        if missing:
            described = ", ".join(f"{m['fov_id']} {m['path']}" for m in missing)
            return finish("not_run", reason="territory_missing", missing=missing, library=library,
                          warn=f"territory_missing ({described})")
        image, entry = _fuse_and_write(placement, assembly, config, ids, retained, fovs, output_dir)
        outputs.append(entry)
        return finish("written", library=library, fovs=fovs, outputs=outputs, **image)
    except BaseException as error:
        if not outputs:
            (output_dir / LABEL_FILE).unlink(missing_ok=True)
        finish("failed", library=library, fovs=fovs, outputs=outputs,
               error={"type": type(error).__name__, "message": str(error)})
        raise


def _fuse_and_write(placement, assembly, config, ids, retained, fovs, output_dir):
    """Read each territory, build the retained-cell images, fuse them and write the label image."""
    import dask.array as da
    from multiview_stitcher import fusion
    from multiview_stitcher import spatial_image_utils as si_utils

    from starfinder._records import file_sha256
    from starfinder.stitching._placement import table_sha256
    grids = _grids(placement, assembly, ids)
    grid = fused_grid(placement, grids, config=FusionConfig())
    k = config.factor_yx
    by_fov = {entry["fov_id"]: entry for entry in fovs}
    images, geometries = {}, {}
    for fov in ids:
        if fov not in by_fov:
            continue
        entry = by_fov[fov]
        territory = _read_territory(Path(entry["path"]), entry["sha256"])
        entry["sha256"] = file_sha256(entry["path"])
        shape = grids[fov].shape_zyx
        if territory.shape[1:] != tuple(shape[1:]) or territory.shape[0] not in (1, shape[0]):
            raise IncompatibleGeometryError(f"the territory of FOV {fov} ({entry['path']}) has shape "
                                            f"{territory.shape}; its grid {shape}")
        plane = territory.shape[0] == 1
        geometries[fov] = "plane" if plane or grid.one_plane else "volume"
        cell_ids, values = retained[fov]
        images[fov] = retained_image(territory[0] if plane else territory, cell_ids, values)
        del territory
    kinds = set(geometries.values())
    if len(kinds) > 1:
        planes = [f for f, g in geometries.items() if g == "plane"]
        raise ValueError("the territories of the FOVs with a retained cell mix plane and volume geometry; plane: "
                         f"{_names(planes)}, volume: {_names([f for f in geometries if f not in planes])}")
    # One-plane FOVs and plane territories give a 2D image; volume territories of multi-plane FOVs, and
    # multi-plane FOVs of which no territory is read (no retained cell), the stack's Z.
    two_d = grid.one_plane or kinds == {"plane"}
    dims = ["y", "x"] if two_d else ["z", "y", "x"]
    dimensionality = "projection" if two_d else "stack"
    origins = {row.fov_id: (row.origin_z, row.origin_y, row.origin_x) for row in placement.fovs.itertuples(index=False)}
    sims = []
    for fov in ids:
        shape = grids[fov].shape_zyx
        array = images.get(fov)
        if array is None:            # an included FOV without a retained cell is not read
            array = da.zeros(tuple(shape[1:]) if two_d else tuple(shape), dtype=np.uint32)
        spacing = grids[fov].metadata.spacing_zyx
        sims.append(si_utils.get_sim_from_array(array, dims=dims,
                                                scale={a: float(spacing[AXES.index(a)]) for a in dims},
                                                translation={a: float(origins[fov][AXES.index(a)]) for a in dims},
                                                transform_key=TRANSFORM_KEY))
    s = grid.spacing_zyx
    output_spacing = {"y": k * float(s[1]), "x": k * float(s[2])}
    if not two_d:
        output_spacing = {"z": float(s[0]), **output_spacing}
    chunks = CHUNKS["plane" if two_d else "stack"]
    fused = fusion.fuse(sims, transform_key=TRANSFORM_KEY, fusion_func=_first_positive,
                        interpolation_order=INTERPOLATION_ORDER, output_spacing=output_spacing,
                        output_chunksize=chunks)
    from ._fuse import _library_grid
    returned = _library_grid(fused)
    formula = label_grid(s, [origins[f] for f in ids], [grids[f].shape_zyx for f in ids], k)
    for a in ("y", "x"):
        if returned["shape"][a] != formula["shape"][a] or \
                abs(returned["spacing"][a] - formula["spacing"][a]) > TOLERANCE_UM or \
                abs(returned["origin"][a] - formula["origin"][a]) > TOLERANCE_UM:
            raise ValueError(f"the label grid the library returned ({returned}) is not the formula grid ({formula}) "
                             f"in {a} within {TOLERANCE_UM} µm")
    level = grid.level(dimensionality)
    z_spacing, z_origin, n_z = level[0][0], level[1][0], level[2][0]
    if not two_d:
        formula["spacing"]["z"], formula["origin"]["z"], formula["shape"]["z"] = z_spacing, z_origin, n_z
        if returned["shape"]["z"] != n_z or abs(returned["spacing"]["z"] - z_spacing) > TOLERANCE_UM or \
                abs(returned["origin"]["z"] - z_origin) > TOLERANCE_UM:
            raise ValueError(f"the label grid the library returned in z ({returned}) is not the fusion grid's Z "
                             f"(spacing {z_spacing}, origin {z_origin}, {n_z} planes)")
    spacing_zyx = [z_spacing, returned["spacing"]["y"], returned["spacing"]["x"]]
    origin_zyx = [z_origin, returned["origin"]["y"], returned["origin"]["x"]]
    shape_zyx = [n_z, returned["shape"]["y"], returned["shape"]["x"]]
    data = fused.data[0, 0]
    if np.dtype(data.dtype) != np.uint32:
        raise ValueError(f"the library returned the label image as {data.dtype}, not uint32")
    metadata = ImageMetadata(f"sample:{placement.sample}", tuple(spacing_zyx), tuple(origin_zyx), IDENTITY, "µm")
    annotation = {"sample": placement.sample, "image": "cell_labels", "rule": RULE, "factor_yx": k,
                  "placement_fovs_sha256": table_sha256(placement.fovs),
                  "placement_links_sha256": table_sha256(placement.links)}
    path = output_dir / LABEL_FILE
    layout = _ome.write_ome_tiff(path, data, metadata=metadata, annotation=annotation)
    written = _tiff_values(path)
    ids_input = sorted({int(v) for image in images.values() for v in np.unique(image) if v > 0})
    ids_output = sorted(int(v) for v in written if v > 0)
    absent = sorted(set(ids_input) - set(ids_output))
    conflicts = None
    if (assembly.record.get("flags") or {}).get("conflicts", False) and len(assembly.cells):
        flagged = assembly.cells["assembly_flags"].fillna("").astype(str).str.split(";").map(
            lambda flags: "conflict" in flags).to_numpy(dtype=bool)
        conflicts = int(np.isin(instance_ids(assembly.cells)[flagged], ids_output).sum())
    entry = {"path": LABEL_FILE, "format": "ome_tiff", "image": "cell_labels", "dimensionality": dimensionality,
             "inputs_2d": two_d, "factor_yx": k, "shape_zyx": shape_zyx, "dtype": "uint32",
             "spacing_zyx": spacing_zyx, "origin_zyx": origin_zyx,
             "slab_z": list(grid.slab_z) if two_d else None, "levels": [], "tile_yx": layout["tile_yx"],
             "compression": layout["compression"], "axes": layout["axes"], "annotation": annotation,
             "output_spacing": output_spacing, "output_chunksize": chunks, "sha256": file_sha256(path)}
    image = {"territory_geometry": next(iter(kinds)) if kinds else None,
             "grid": {"fusion": {"shape_zyx": list(grid.shape_zyx), "spacing_zyx": list(grid.spacing_zyx),
                                 "origin_zyx": list(grid.origin_zyx), "inputs_2d": grid.one_plane,
                                 "slab_z": list(grid.slab_z), "fov_ids": list(grid.fov_ids)},
                      "library": returned, "formula": formula},
             "n_ids_input": len(ids_input), "n_ids_output": len(ids_output), "absent_ids": absent,
             "n_absent_ids": len(absent), "n_conflict": conflicts}
    return image, entry


def read_sample_territory(path, sha256) -> np.ndarray:
    """The sample's territories of the route after fusion (ZYX, one plane), the file's SHA-256 checked first."""
    import tifffile

    from starfinder._records import check_file
    check_file(path, sha256, "territory file")
    labels = tifffile.imread(path)
    return np.ascontiguousarray(labels[np.newaxis] if labels.ndim == 2 else labels)


def retained_ids(cells) -> tuple[np.ndarray, np.ndarray]:
    """The ``cell_id`` of every retained cell of the sample (sorted, int64) and their ``instance_id`` (uint32)."""
    if len(cells) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.uint32)
    keep = cells["assembly_status"].eq("retained").fillna(False).to_numpy(dtype=bool)
    cell_ids = cells["cell_id"].astype("int64").to_numpy()[keep]
    order = np.argsort(cell_ids, kind="stable")
    return cell_ids[order], instance_ids(cells)[keep][order]


def _fuse_sample_and_write(placement, assembly, config, output_dir, relative, sha256):
    """The route after fusion: the sample's territories mapped to ``instance_id``, fused as one view, written."""
    from multiview_stitcher import fusion
    from multiview_stitcher import spatial_image_utils as si_utils

    from starfinder._records import file_sha256
    from starfinder.stitching._placement import table_sha256

    from ._fuse import _library_grid
    grid = assembly.record["assignment"]["grid"]
    shape, spacing, origin = grid["shape_zyx"], grid["metadata"]["spacing_zyx"], grid["metadata"]["origin_zyx"]
    path = output_dir / relative
    territory = read_sample_territory(path, sha256)
    if territory.shape != tuple(shape) or territory.shape[0] != 1:
        raise IncompatibleGeometryError(f"the sample's territories {path} have shape {territory.shape}; the "
                                        f"segmentation grid {tuple(shape)}")
    cell_ids, values = retained_ids(assembly.cells)
    image = retained_image(territory[0], cell_ids, values)
    del territory
    s_y, s_x = _acquired_spacing(placement, assembly)
    k = config.factor_yx
    output_spacing = {"y": k * s_y, "x": k * s_x}
    sim = si_utils.get_sim_from_array(image, dims=["y", "x"], scale={"y": float(spacing[1]), "x": float(spacing[2])},
                                      translation={"y": float(origin[1]), "x": float(origin[2])},
                                      transform_key=TRANSFORM_KEY)
    chunks = CHUNKS["plane"]
    fused = fusion.fuse([sim], transform_key=TRANSFORM_KEY, fusion_func=_first_positive,
                        interpolation_order=INTERPOLATION_ORDER, output_spacing=output_spacing,
                        output_chunksize=chunks)
    returned = _library_grid(fused)
    formula = sample_label_grid(shape, spacing, origin, (output_spacing["y"], output_spacing["x"]))
    for a in ("y", "x"):
        if returned["shape"][a] != formula["shape"][a] or \
                abs(returned["spacing"][a] - formula["spacing"][a]) > TOLERANCE_UM or \
                abs(returned["origin"][a] - formula["origin"][a]) > TOLERANCE_UM:
            raise ValueError(f"the label grid the library returned ({returned}) is not the formula grid ({formula}) "
                             f"in {a} within {TOLERANCE_UM} µm")
    # Z: the slab of the segmentation grid, as for the projection.
    z_spacing, z_origin = float(spacing[0]), float(origin[0])
    spacing_zyx = [z_spacing, returned["spacing"]["y"], returned["spacing"]["x"]]
    origin_zyx = [z_origin, returned["origin"]["y"], returned["origin"]["x"]]
    shape_zyx = [1, returned["shape"]["y"], returned["shape"]["x"]]
    data = fused.data[0, 0]
    if np.dtype(data.dtype) != np.uint32:
        raise ValueError(f"the library returned the label image as {data.dtype}, not uint32")
    metadata = ImageMetadata(f"sample:{placement.sample}", tuple(spacing_zyx), tuple(origin_zyx), IDENTITY, "µm")
    annotation = {"sample": placement.sample, "image": "cell_labels", "rule": SEGMENTATION_RULE, "factor_yx": k,
                  "placement_fovs_sha256": table_sha256(placement.fovs),
                  "placement_links_sha256": table_sha256(placement.links)}
    out = output_dir / LABEL_FILE
    layout = _ome.write_ome_tiff(out, data, metadata=metadata, annotation=annotation)
    written = _tiff_values(out)
    ids_input = sorted(int(v) for v in np.unique(image) if v > 0)
    ids_output = sorted(int(v) for v in written if v > 0)
    absent = sorted(set(ids_input) - set(ids_output))
    entry = {"path": LABEL_FILE, "format": "ome_tiff", "image": "cell_labels", "dimensionality": "projection",
             "inputs_2d": True, "factor_yx": k, "shape_zyx": shape_zyx, "dtype": "uint32",
             "spacing_zyx": spacing_zyx, "origin_zyx": origin_zyx,
             "slab_z": [z_origin - z_spacing / 2, z_origin + z_spacing / 2], "levels": [], "tile_yx": layout["tile_yx"],
             "compression": layout["compression"], "axes": layout["axes"], "annotation": annotation,
             "output_spacing": output_spacing, "output_chunksize": chunks, "sha256": file_sha256(out)}
    image_record = {"territory_geometry": "plane",
                    "grid": {"territories": {"path": relative, "sha256": sha256, "shape_zyx": list(shape),
                                             "spacing_zyx": list(spacing), "origin_zyx": list(origin)},
                             "library": returned, "formula": formula},
                    "n_ids_input": len(ids_input), "n_ids_output": len(ids_output), "absent_ids": absent,
                    "n_absent_ids": len(absent), "n_conflict": None}
    return image_record, entry


def _tiff_values(path) -> np.ndarray:
    """The distinct values of a written label image, read back plane by plane."""
    import tifffile
    values = set()
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        for i in range(len(series.pages)):
            values |= set(np.unique(series.asarray(key=i)).tolist())
    return np.array(sorted(values), dtype=np.int64)
