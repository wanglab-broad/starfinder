"""``Dataset.export``: the one sample-level export entry (design/specs/sample-output.md, "The export entry").

It runs the checks before anything is written, removes the files the previous record
lists for the parts that run or are off, records each part that will run as ``not_run``
with the reason ``export_not_finished`` (and each part that is off as ``off``), and calls
the parts in order, writing ``export.json`` after each part, whose section replaces its
entry. On the route per FOV: ``assemble`` (then ``save_assembly``), ``fuse``,
``fuse_labels`` and ``export_spatialdata``, with the part ``segment`` recorded ``off``
(reason ``route_per_fov``, or ``no_cells`` without an assignment, on either route). On the
route after fusion with an assignment (design/specs/sample-segmentation.md, "The parts on
each route"): ``fuse``, ``segment`` (``segment_fused``, or the saved sample segmentation
``reused``), ``assemble`` with the sample segmentation (or the ``molecules_only`` assembly
when ``segment`` could not run), then ``fuse_labels`` (the sample's territories as its one
view) and ``export_spatialdata``.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from starfinder.export._config import ExportConfig, image_name
from starfinder.export._fuse import FusionInput, check_assembly, check_fusion, fuse_images
from starfinder.export._labels import check_label_count, check_label_ids, fuse_labels
from starfinder.export._record import (
    PARTS,
    SECTIONS,
    ExportRecord,
    as_json,
    listed_files,
    pending_section,
    read_export,
    remove_listed,
    update_section,
    write_export,
)
from starfinder.export._segment import check_route, resolve_route, segment_off_section
from starfinder.stitching import PlacementResult
from starfinder.stitching._placement import OWNING


def _paths(dataset, fov):
    from starfinder._layout import _FovPaths
    return _FovPaths(Path(dataset.output_root), fov)


def reference_stain_path(output_root, fov) -> Path:
    """The saved reference stain of a FOV: ``fovs/<fov>/other_rounds/reference_stain/image.ome.tif``."""
    from starfinder._layout import fov_dir
    from starfinder.dataset._prepared import IMAGE_FILE, image_directory
    from starfinder.dataset.types import REFERENCE_STAIN
    return image_directory(fov_dir(output_root, fov), REFERENCE_STAIN) / IMAGE_FILE


def source_path(dataset, fov, entry) -> Path:
    """The file of one source-image entry of ``images`` for a FOV."""
    if entry == "reference_stain":
        return reference_stain_path(dataset.output_root, fov)
    if entry == "reference_merged":
        from starfinder._layout import reference_image_path
        return reference_image_path(dataset.output_root, fov, dataset.reference_image, dataset.reference_image_import)
    pattern = entry.split("=", 1)[1][len("file:"):].replace("{fov_id}", fov)
    path = Path(pattern)
    return path if path.is_absolute() else Path(dataset.output_root) / path


def _reference_grid(dataset, fov):
    """The FOV's grid: the stored metadata of its reference image (``reference_image``), else the voxel sizes."""
    from starfinder.dataset._declared import reference_grid, reference_settings
    return reference_grid(dataset.output_root, fov, **reference_settings(dataset))


def _genes(dataset):
    if dataset.readout_mode == "direct" and dataset.direct_panel is not None:
        return tuple(dict.fromkeys(dataset.direct_panel.genes))
    if dataset.codebook is not None:
        return tuple(dataset.codebook.genes)
    raise ValueError("Dataset.export with assignment=None reads molecule tables and needs the gene list: load the "
                     "codebook (or the direct panel)")


def assembly_inputs(dataset, placement, assignment) -> list:
    """One AssemblyInput per FOV of the placement that has a saved assignment folder (or molecule table).

    The folder is ``fovs/<fov_id>/assignment/<assignment>/``; with
    ``assignment=None`` the FOV's ``fovs/<fov_id>/molecules.csv`` is read
    with ``load_molecules``, which keeps the molecules' identities, and the
    namespace ``[dataset_id, sample_id, fov_id, null]``. On the workflow's
    legacy route (the dataset's ``_legacy_molecules``, set from the
    configuration) the one-based ``signal/<fov_id>_goodSpots.csv`` is read with
    ``import_legacy_molecules_csv`` instead. The grid is that of the FOV's reference
    image, the file the dataset's ``reference_image`` names (default
    ``fovs/<fov_id>/reference.tif``) or its ``reference_image_import``: its stored metadata, or the grid the
    dataset's ``voxel_size_xy`` and ``voxel_size_z`` declare for a file that
    stores none. A FOV without a saved input gets none;
    ``assemble`` then refuses an owning FOV without one.
    """
    from starfinder.assembly import AssemblyInput
    from starfinder.assignment import import_legacy_molecules_csv, load_molecules
    from starfinder.assignment._persist import RECORD_FILE, assignment_directory
    inputs = []
    genes = _genes(dataset) if assignment is None else None
    for fov in placement.fovs["fov_id"]:
        paths = _paths(dataset, fov)
        if assignment is not None:
            folder = assignment_directory(paths.fov_dir, assignment)
            if not (folder / RECORD_FILE).is_file():
                continue
            inputs.append(AssemblyInput(fov, _reference_grid(dataset, fov), assignment=folder))
            continue
        legacy = dataset._legacy_molecules
        path = paths.signal_csv("goodSpots") if legacy else paths.spot_table("molecules")
        if not path.is_file():
            continue
        namespace = json.dumps([dataset.dataset_id, dataset.sample_id, fov, None], separators=(",", ":"))
        reader = import_legacy_molecules_csv if legacy else load_molecules
        inputs.append(AssemblyInput(fov, _reference_grid(dataset, fov),
                                    molecules=reader(path, spot_namespace=namespace, genes=genes)))
    return inputs


def fusion_inputs(dataset, placement, config) -> list[FusionInput]:
    """One FusionInput per included FOV: its reference grid and the path of each selected source image."""
    inputs = []
    for fov, status in zip(placement.fovs["fov_id"], placement.fovs["status"]):
        if status not in OWNING:
            continue
        images = {image_name(entry): source_path(dataset, fov, entry) for entry in config.fusion.images}
        inputs.append(FusionInput(fov, _reference_grid(dataset, fov), images))
    return inputs


def _require_extras(config):
    """Check 3: the assemble part's extras (anndata; pyarrow for Parquet tables)."""
    if not config.assemble:
        return
    try:
        import anndata  # noqa: F401
    except ImportError as error:
        raise ModuleNotFoundError("the assemble part writes sample.h5ad and needs anndata; install the 'anndata' "
                                  "extra") from error
    if config.table_format == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError as error:
            raise ModuleNotFoundError("table_format 'parquet' needs pyarrow; install the 'checkpoint' extra or use "
                                      "table_format 'csv'") from error


def _saved_assembly(output_dir):
    """The saved assembly of the folder, or None when its export.json has no assembly section."""
    from starfinder.assembly import load_assembly
    section = read_export(output_dir).get("assembly")
    if not isinstance(section, dict) or section.get("stage") != "assembly":
        return None
    return load_assembly(output_dir)


def export_sample(dataset, sample: str, *, placement, assignment: str | None = "default",
                  config: ExportConfig = ExportConfig(), output_dir) -> ExportRecord:
    """The body of :meth:`starfinder.dataset.Dataset.export`."""
    from starfinder.assembly import assemble, save_assembly
    from starfinder.stitching import load_placement
    from starfinder.stitching._review import require_approval
    # 1. Config.
    if not isinstance(config, ExportConfig):
        raise TypeError("config must be an ExportConfig")
    for part in (config, config.assembly, config.fusion, config.labels, config.spatialdata):
        part.__post_init__()
    if not isinstance(sample, str) or not sample:
        raise ValueError("sample must be a nonempty string")
    if config.export_spatialdata:
        from starfinder.export._spatialdata import check_name
        check_name(sample, "sample name")
    if assignment is not None and (not isinstance(assignment, str) or not assignment):
        raise ValueError(f"assignment must be a saved assignment name or None; got {assignment!r}")
    if not isinstance(placement, PlacementResult):
        placement = load_placement(placement)
    if placement.sample != sample:
        raise ValueError(f"the placement is of sample {placement.sample!r}, not {sample!r}")
    output_dir = Path(output_dir)
    route, route_source = resolve_route(config)
    after_fusion = route == "after_fusion" and assignment is not None
    # 2. Review.
    require_approval(placement)
    # 3. Required extras.
    _require_extras(config)
    if after_fusion:
        return _export_after_fusion(dataset, sample, placement, assignment, config, output_dir, route_source)
    # 4. Assembly: computed in memory (assemble applies its own checks), or the saved one.
    result, status = None, None
    if config.assemble:
        result = assemble(placement, assembly_inputs(dataset, placement, assignment), config=config.assembly)
    else:
        needed = config.fuse_labels or config.export_spatialdata
        if needed and read_export(output_dir).get("assembly", {}).get("stage") != "assembly":
            raise ValueError(f"assemble is off and fuse_labels or export_spatialdata is on, but {output_dir} holds no "
                             "saved assembly; run with assemble on first")
        result = _saved_assembly(output_dir)
        if result is not None:
            check_assembly(placement, result)
        status = "reused" if result is not None else "off"
    # 5. Fusion inputs.
    inputs = None
    if config.fuse:
        inputs = fusion_inputs(dataset, placement, config)
        check_fusion(placement, inputs, config=config.fusion, assembly=None)
    # 6. Label IDs.
    if config.fuse_labels and result is not None:
        check_label_ids(result.cells)
    # 7. The route (design/specs/sample-segmentation.md, "The entry").
    check_route(config, assignment)
    segmentation = segment_off_section(route, route_source, assignment)

    record = read_export(output_dir)
    if record.get("sample") not in (None, sample):
        raise ValueError(f"{output_dir} holds the export of sample {record.get('sample')!r}, not {sample!r}")

    # Keep the folder equal to its record: remove what the previous record lists for each part that runs or is off.
    # Each part that will run is recorded not_run (export_not_finished) until it writes its own section, so a record
    # left by an export stopped during a part is not complete; a part that is off is recorded off, as at its turn.
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {"assemble": config.assemble, "fuse": config.fuse, "segment": False, "fuse_labels": config.fuse_labels,
            "export_spatialdata": config.export_spatialdata}
    for part in PARTS:
        if part == "assemble" and status == "reused":
            continue
        remove_listed(output_dir, listed_files(record, part))
        record.pop(SECTIONS[part], None)
        (record.get("parts") or {}).pop(part, None)
        record[SECTIONS[part]] = pending_section() if runs[part] else \
            (segmentation if part == "segment" else {"status": "off"})
    write_export(output_dir, record, sample=sample)

    # Part 1: assemble.
    if config.assemble:
        result = save_assembly(with_route(result, route, route_source), output_dir, table_format=config.table_format)
        update_section(output_dir, sample, "assemble", result.record, status="written", fill=False)
    elif status == "reused":
        update_section(output_dir, sample, "assemble", result.record, status="reused", fill=False)
    else:
        update_section(output_dir, sample, "assemble", {"status": "off"}, fill=False)
    # Part 2: fuse.
    if config.fuse:
        fuse_images(placement, inputs, assembly=result, config=config.fusion, output_dir=output_dir)
    else:
        update_section(output_dir, sample, "fuse", {"status": "off"}, fill=False)
    # The part segment of the route after fusion: off on this route.
    update_section(output_dir, sample, "segment", segmentation, fill=False)
    # Part 3: fuse_labels.
    if config.fuse_labels:
        fuse_labels(placement, result, config=config.labels, output_dir=output_dir)
    else:
        update_section(output_dir, sample, "fuse_labels", {"status": "off"}, fill=False)
    # Part 4: export_spatialdata, from the saved assembly and the sections of parts 2 and 3.
    if config.export_spatialdata:
        _spatialdata_part(result, config, output_dir)
        record = read_export(output_dir)
    else:
        record = update_section(output_dir, sample, "export_spatialdata", {"status": "off"})
    return ExportRecord(output_dir, record)


def _spatialdata_part(result, config, output_dir):
    """Part 4. An error inside it records the part ``failed`` with the exception's type and message, then raises."""
    from starfinder.export._record import Timer
    from starfinder.export._spatialdata import _check, _export
    timer = Timer()
    requested = [f"{name}_{d}" for name in config.fusion.image_names for d in config.fusion.dimensionalities]
    try:
        record = _check(result, config.spatialdata, output_dir)
        _export(result, record, config.spatialdata, output_dir, timer,
                requested_images=requested)
    except BaseException as error:
        if (read_export(output_dir).get("spatialdata") or {}).get("status") != "failed":
            update_section(output_dir, result.sample, "export_spatialdata",
                           {"status": "failed", "store": None, "outputs": [],
                            "error": {"type": type(error).__name__, "message": str(error)}})
        raise


# --- The route after fusion (design/specs/sample-segmentation.md, "The entry") -------------------------------------

GRID_TOLERANCE_UM = 1e-9


def with_route(result, route, route_source, **extra):
    """The assembly with the record's ``route`` and ``route_source`` (and ``extra`` keys) set by the export."""
    from dataclasses import replace

    from starfinder.assembly._fused import territories_of, with_territories
    return with_territories(replace(result, record=dict(result.record, route=route, route_source=route_source,
                                                        **extra)), territories_of(result))


def _input_grids(dataset, placement):
    return {fov: _reference_grid(dataset, fov) for fov, status in zip(placement.fovs["fov_id"],
                                                                     placement.fovs["status"]) if status in OWNING}


def _reuse(dataset, placement, config, output_dir, held):
    """The reuse conditions of ``segment: false`` (and ``assemble: false``), before anything is written."""
    from starfinder.export._grid import fused_grid
    from starfinder.export._segment import load_sample_segmentation
    from starfinder.stitching._placement import table_sha256
    where = f"export.segment: false reuses the saved sample segmentation of {output_dir}"
    try:
        saved = load_sample_segmentation(output_dir)
    except ValueError as error:
        raise ValueError(f"{where}, but {error}") from error
    recorded = saved.record["inputs"].get("placement") or {}
    current = {"fovs_sha256": table_sha256(placement.fovs), "links_sha256": table_sha256(placement.links)}
    differ = [key for key in current if recorded.get(key) != current[key]]
    if differ:
        raise ValueError(f"{where}, but it was made from another placement: {differ[0]} recorded "
                         f"{recorded.get(differ[0])}, the placement {current[differ[0]]}")
    spacing, origin, shape = fused_grid(placement, _input_grids(dataset, placement),
                                        config=config.fusion).level("projection")
    grid = saved.grid
    same = tuple(grid.shape_zyx) == tuple(shape) and all(
        abs(a - b) <= GRID_TOLERANCE_UM for a, b in zip((*grid.metadata.spacing_zyx, *grid.metadata.origin_zyx),
                                                        (*spacing, *origin)))
    if not same:
        raise ValueError(f"{where}, but it is on another grid ({grid.shape_zyx}, spacing "
                         f"{grid.metadata.spacing_zyx}, origin {grid.metadata.origin_zyx}) than this export's "
                         f"placement and scale_yx give ({tuple(shape)}, spacing {spacing}, origin {origin})")
    names = config.fusion.image_names
    missing = sorted({e["image"] for e in saved.record["inputs"].get("images") or []} - set(names))
    if missing:
        raise ValueError(f"{where}, but it was made from the images {missing}, which are not names of export.images "
                         f"{list(names)}")
    held["segmentation"] = saved
    if config.assemble:
        return
    from starfinder.assembly import load_assembly
    section = read_export(output_dir).get("assembly")
    if not isinstance(section, dict) or section.get("stage") != "assembly":
        raise ValueError(f"export.assemble: false reuses the saved tables of {output_dir}, but it holds no saved "
                         "assembly")
    files = [{"run": run["name"], "path": run["path"], "sha256": run["sha256"]} for run in saved.record["runs"]]
    used = [e for e in section.get("segmentation_files") or [] if e.get("run") is not None]
    if section.get("route") != "after_fusion" or used != files:
        raise ValueError(f"export.assemble: false reuses the saved tables of {output_dir}, but they were not made "
                         "from the reused sample segmentation: the assembly section's segmentation_files differ "
                         "from its label files")
    result = load_assembly(output_dir)
    check_assembly(placement, result)
    held["assembly"] = result


def _export_after_fusion(dataset, sample, placement, assignment, config, output_dir, route_source) -> ExportRecord:
    """``Dataset.export`` on the route after fusion with an assignment: checks 4 to 7, then the parts in order."""
    from starfinder.assembly import assemble, save_assembly
    from starfinder.export._segment import segment_fused
    from starfinder.segmentation import SampleSegmentation
    # 7. The route, with the reuse conditions when segment is off. On this route it comes before checks 4 and 5, so
    # that a configuration without the route's settings is refused naming them (RD1), not for its inputs.
    held = {}
    check_route(config, assignment, reuse=lambda: _reuse(dataset, placement, config, output_dir, held))
    # 4. Assembly: the molecule tables of every FOV; their molecules_only assembly is computed in memory, so that
    # every check of assemble runs before anything is written.
    inputs = assembly_inputs(dataset, placement, None)
    if config.assemble:
        assemble(placement, inputs, config=config.assembly)
    # 5. Fusion inputs.
    fusion = None
    if config.fuse:
        fusion = fusion_inputs(dataset, placement, config)
        check_fusion(placement, fusion, config=config.fusion, assembly=None)
    # 6. Label IDs (L6): with a reused segmentation the saved cell run gives the number of cells, one row per label,
    # so the bound is checked before anything is written. A fresh export knows it only after the part segment.
    if config.fuse_labels and "segmentation" in held:
        check_label_count(held["segmentation"].cells.n_labels)

    record = read_export(output_dir)
    if record.get("sample") not in (None, sample):
        raise ValueError(f"{output_dir} holds the export of sample {record.get('sample')!r}, not {sample!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {"assemble": config.assemble, "fuse": config.fuse, "segment": config.segment,
            "fuse_labels": config.fuse_labels, "export_spatialdata": config.export_spatialdata}
    for part in PARTS:
        if (part == "assemble" and not config.assemble) or (part == "segment" and not config.segment):
            continue   # reused: its files and section stay until the part's turn
        remove_listed(output_dir, listed_files(record, part))
        record.pop(SECTIONS[part], None)
        (record.get("parts") or {}).pop(part, None)
        record[SECTIONS[part]] = pending_section() if runs[part] else {"status": "off"}
    write_export(output_dir, record, sample=sample)

    # Part fuse.
    if config.fuse:
        fuse_images(placement, fusion, assembly=None, config=config.fusion, output_dir=output_dir)
    else:
        update_section(output_dir, sample, "fuse", {"status": "off"}, fill=False)
    # Part segment.
    assignment_settings = config.segmentation.assignment
    if config.segment:
        segmentation = segment_fused(output_dir, config=config.segmentation, sample=sample,
                                     identity=(dataset.dataset_id, dataset.sample_id))
        if config.fuse_labels and segmentation.status == "written":
            _check_label_bound(output_dir, sample, segmentation)
    else:
        saved = held["segmentation"]
        section = dict(saved.record, status="reused", assignment=as_json(asdict(assignment_settings)))
        fused = read_export(output_dir).get("fusion") or {}
        if fused.get("status") == "written":
            outputs = {o["image"]: o for o in fused.get("outputs") or []
                       if o.get("format") == "ome_tiff" and o.get("dimensionality") == "projection"}
            for entry in saved.record["inputs"].get("images") or []:
                now = outputs.get(entry["image"], {}).get("sha256")
                if now != entry["sha256"]:
                    failed = dict(section, status="failed", reason="fused_input_changed", image=entry["image"],
                                  recorded_sha256=entry["sha256"], fused_sha256=now)
                    message = (f"the reused sample segmentation was made from the fused projection of image "
                               f"{entry['image']!r} with SHA-256 {entry['sha256']}; this export fused it with "
                               f"SHA-256 {now} (fused_input_changed)")
                    failed["error"] = {"type": "ValueError", "message": message}
                    update_section(output_dir, sample, "segment", failed, fill=False)
                    raise ValueError(message)
        update_section(output_dir, sample, "segment", section, fill=False)
        segmentation = SampleSegmentation(saved.grid, saved.runs, as_json(section))
    # Part assemble.
    if config.assemble:
        if segmentation.status in ("written", "reused"):
            result = with_route(assemble(placement, inputs, config=config.assembly, segmentation=segmentation),
                                "after_fusion", route_source)
        else:
            # The part segment could not run: the molecules_only assembly, so that the files of the parts that ran
            # exist; the record says why.
            result = with_route(assemble(placement, inputs, config=config.assembly), "after_fusion", route_source,
                                reason=f"segment_{segmentation.status}")
        result = save_assembly(result, output_dir, table_format=config.table_format)
        update_section(output_dir, sample, "assemble", result.record, status="written", fill=False)
    else:
        result = held["assembly"]
        update_section(output_dir, sample, "assemble", result.record, status="reused", fill=False)
    # Part fuse_labels: the sample's territories as the one view.
    if config.fuse_labels:
        fuse_labels(placement, result, config=config.labels, output_dir=output_dir)
    else:
        update_section(output_dir, sample, "fuse_labels", {"status": "off"}, fill=False)
    # Part export_spatialdata, from the saved assembly and the sections of fuse and fuse_labels.
    if config.export_spatialdata:
        _spatialdata_part(result, config, output_dir)
    else:
        update_section(output_dir, sample, "export_spatialdata", {"status": "off"}, fill=False)
    return ExportRecord(output_dir, read_export(output_dir))


def _check_label_bound(output_dir, sample, segmentation):
    """L6 on a fresh export of the route after fusion, directly after the part ``segment`` and before ``assemble``
    writes (design/specs/sample-segmentation.md, "The label images"): the number of labels of the cell run, one cell
    row each. A refusal records the part ``fuse_labels`` ``failed`` with the error of L6 and raises; the files of
    ``fuse`` and ``segment`` stay, and no table, ``sample.h5ad``, label image or store is written."""
    try:
        check_label_count(segmentation.cells.n_labels)
    except ValueError as error:
        update_section(output_dir, sample, "fuse_labels",
                       {"status": "failed", "outputs": [], "error": {"type": type(error).__name__,
                                                                     "message": str(error)}}, fill=False)
        raise
