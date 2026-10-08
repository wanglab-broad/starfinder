"""``Dataset.export``: the one sample-level export entry (docs/sample-output-contract.md, "The export entry").

It runs the checks before anything is written, removes the files the previous record
lists for the parts that run or are off, and calls the parts in order: ``assemble``
(then ``save_assembly``), ``fuse``, ``fuse_labels`` and ``export_spatialdata``, writing
``export.json`` after each part.
"""
from __future__ import annotations

import json
from pathlib import Path

from starfinder.stitching import PlacementResult

from ._config import ExportConfig, image_name
from ._fuse import FusionInput, check_assembly, check_fusion, fuse
from ._grid import INCLUDED
from ._labels import check_label_ids, fuse_labels
from ._record import (
    PARTS,
    SECTIONS,
    ExportRecord,
    listed_files,
    read_export,
    remove_listed,
    update_section,
    write_export,
)

REFERENCE_STAIN_FILE = Path("other_rounds") / "reference_stain" / "image.ome.tif"
MOLECULE_CSV = "signal/{fov_id}_goodSpots.csv"


def _paths(dataset, fov):
    from starfinder.dataset._paths import _FovPaths
    return _FovPaths(Path(dataset.output_root), fov)


def source_path(dataset, fov, entry) -> Path:
    """The file of one source-image entry of ``images`` for a FOV."""
    paths = _paths(dataset, fov)
    if entry == "reference_stain":
        return paths.checkpoint_dir / REFERENCE_STAIN_FILE
    if entry == "reference_merged":
        return paths.ref_merged_tif
    pattern = entry.split("=", 1)[1][len("file:"):].replace("{fov_id}", fov)
    path = Path(pattern)
    return path if path.is_absolute() else Path(dataset.output_root) / path


def _reference_grid(dataset, fov):
    """The FOV's grid: the stored metadata of ``images/ref_merged/<fov_id>.tif``, else the dataset's voxel sizes."""
    from starfinder.dataset._declared import reference_grid
    return reference_grid(dataset.output_root, fov, dataset_id=dataset.dataset_id, output_id=dataset.output_id,
                          voxel_size_xy=dataset.voxel_size_xy, voxel_size_z=dataset.voxel_size_z)


def _genes(dataset):
    if dataset.readout_mode == "direct" and dataset.direct_panel is not None:
        return tuple(dict.fromkeys(dataset.direct_panel.genes))
    if dataset.codebook is not None:
        return tuple(dataset.codebook.genes)
    raise ValueError("Dataset.export with assignment=None reads molecule tables and needs the gene list: load the "
                     "codebook (or the direct panel)")


def assembly_inputs(dataset, placement, assignment) -> list:
    """One AssemblyInput per FOV of the placement that has a saved assignment folder (or molecule table).

    The folder is ``checkpoints/<fov_id>/assignment/<assignment>/``; with
    ``assignment=None`` the FOV's ``signal/<fov_id>_goodSpots.csv`` is read
    with ``molecule_table_from_csv``. The grid is that of
    ``images/ref_merged/<fov_id>.tif``: its stored metadata, or the grid the
    dataset's ``voxel_size_xy`` and ``voxel_size_z`` declare for a file that
    stores none. A FOV without a saved input gets none;
    ``assemble`` then refuses an owning FOV without one.
    """
    from starfinder.assembly import AssemblyInput
    from starfinder.assignment import molecule_table_from_csv
    from starfinder.assignment._persist import RECORD_FILE, assignment_directory
    inputs = []
    genes = _genes(dataset) if assignment is None else None
    for fov in placement.fovs["fov_id"]:
        paths = _paths(dataset, fov)
        if assignment is not None:
            folder = assignment_directory(paths.checkpoint_dir, assignment)
            if not (folder / RECORD_FILE).is_file():
                continue
            inputs.append(AssemblyInput(fov, _reference_grid(dataset, fov), assignment=folder))
            continue
        path = Path(dataset.output_root) / MOLECULE_CSV.format(fov_id=fov)
        if not path.is_file():
            continue
        namespace = json.dumps([dataset.dataset_id, dataset.sample_id, fov, None], separators=(",", ":"))
        inputs.append(AssemblyInput(fov, _reference_grid(dataset, fov),
                                    molecules=molecule_table_from_csv(path, spot_namespace=namespace, genes=genes)))
    return inputs


def fusion_inputs(dataset, placement, config) -> list[FusionInput]:
    """One FusionInput per included FOV: its reference grid and the path of each selected source image."""
    inputs = []
    for fov, status in zip(placement.fovs["fov_id"], placement.fovs["status"]):
        if status not in INCLUDED:
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
        from ._spatialdata import check_name
        check_name(sample, "sample name")
    if assignment is not None and (not isinstance(assignment, str) or not assignment):
        raise ValueError(f"assignment must be a saved assignment name or None; got {assignment!r}")
    if not isinstance(placement, PlacementResult):
        placement = load_placement(placement)
    if placement.sample != sample:
        raise ValueError(f"the placement is of sample {placement.sample!r}, not {sample!r}")
    output_dir = Path(output_dir)
    # 2. Review.
    require_approval(placement)
    # 3. Required extras.
    _require_extras(config)
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

    record = read_export(output_dir)
    if record.get("sample") not in (None, sample):
        raise ValueError(f"{output_dir} holds the export of sample {record.get('sample')!r}, not {sample!r}")

    # Keep the folder equal to its record: remove what the previous record lists for each part that runs or is off.
    output_dir.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        if part == "assemble" and status == "reused":
            continue
        remove_listed(output_dir, listed_files(record, part))
        record.pop(SECTIONS[part], None)
        (record.get("parts") or {}).pop(part, None)
    write_export(output_dir, record, sample=sample)

    # Part 1: assemble.
    if config.assemble:
        result = save_assembly(result, output_dir, table_format=config.table_format)
        update_section(output_dir, sample, "assemble", result.record, status="written", fill=False)
    elif status == "reused":
        update_section(output_dir, sample, "assemble", result.record, status="reused", fill=False)
    else:
        update_section(output_dir, sample, "assemble", {"status": "off"}, fill=False)
    # Part 2: fuse.
    if config.fuse:
        fuse(placement, inputs, assembly=result, config=config.fusion, output_dir=output_dir)
    else:
        update_section(output_dir, sample, "fuse", {"status": "off"}, fill=False)
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
    from ._record import Timer
    from ._spatialdata import _check, _export
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
