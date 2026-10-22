"""The assembly of the route after fusion: the sample's molecules assigned to the cells of the sample segmentation.

docs/sample-segmentation-contract.md ("The entry", "The sample's molecule table", "Outputs, column by
column"). The molecules are those of the ``molecules_only`` assembly, unchanged: a position belongs to
one FOV. Its retained molecules are converted to one ``MoleculeTable`` on the segmentation grid (MT1,
private: ``spot_namespace`` ``[dataset_id, sample_id, null, null]``, ``spot_id`` the row's position in
the sample molecule table, ``(sample_zyx − O) ⊘ S`` in float64), ``assign_molecules`` runs once on the
sample, and its rows are mapped back to the original keys by position. Every cell of the cell run is
one retained row; its ``fov_id`` is the owning FOV of its ownership centroid (F1), and no cell flag
applies (FL1).
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from starfinder.assignment import AssignmentConfig, MoleculeTable, assign_molecules
from starfinder.segmentation import ExpandLabelsConfig
from starfinder.stitching._placement import table_sha256

from ._assemble import (
    AXES,
    CELL_DTYPES,
    CELL_DTYPES_29,
    COUNT_DTYPES,
    MOLECULE_DTYPES_29,
    AssemblyResult,
    _counts_record,
    molecule_dtypes,
)
from ._ownership import ownership_map

ROUTE = "after_fusion"
CONVERSION = "MT1"
#: The §2.9 assignment columns of the sample molecule table, null on a molecule the sample assignment did not place.
ASSIGNED_COLUMNS = [c for c in MOLECULE_DTYPES_29 if c not in ("spot_namespace", "spot_id", "z", "y", "x",
                                                                "gene_id")]
#: The private attribute of an AssemblyResult of this route that holds the expanded territories and their grid's
#: metadata, for ``save_assembly`` to write ``territories.ome.tif``.
TERRITORIES = "_territories"
TERRITORIES_FILE = "territories.ome.tif"


def territories_of(result):
    """``(labels, metadata)`` of the expanded territories an AssemblyResult of this route carries, or None."""
    return getattr(result, TERRITORIES, None)


def with_territories(result, territories):
    """The result with the territories attached (``dataclasses.replace`` does not carry the attribute)."""
    if territories is not None:
        object.__setattr__(result, TERRITORIES, territories)
    return result


def assignment_config(value) -> AssignmentConfig:
    """The AssignmentConfig a ``segmentation`` record holds as JSON (``AssignmentConfig()`` when it holds none)."""
    from starfinder.assignment._config import CorrespondenceConfig
    if value is None:
        return AssignmentConfig()
    if isinstance(value, AssignmentConfig):
        return value
    fields = dict(value)
    expansion = fields.get("expansion")
    fields["expansion"] = None if expansion is None else ExpandLabelsConfig(**expansion)
    fields["correspondence"] = CorrespondenceConfig(**fields.get("correspondence", {}))
    return AssignmentConfig(**fields)


def used_assignment(config: AssignmentConfig, k: int) -> AssignmentConfig:
    """The assignment settings at factor ``k`` (LS2): an expansion in pixels is divided by ``k``; µm unchanged."""
    expansion = config.expansion
    if expansion is None or expansion.unit != "pixel" or k == 1:
        return config
    return replace(config, expansion=replace(expansion, distance=expansion.distance / k))


def _identity(namespace):
    return json.loads(namespace)[:2]


def _check(placement, inputs, segmentation):
    """The checks of the route's assembly, after ``assemble``'s own checks of the placement and inputs."""
    from starfinder.sample_output import SampleSegmentation
    if not isinstance(segmentation, SampleSegmentation):
        raise TypeError("segmentation must be a SampleSegmentation (segment_fused or load_sample_segmentation)")
    if segmentation.status not in ("written", "reused") or segmentation.grid is None or segmentation.cells is None:
        raise ValueError(f"the sample segmentation has no cells to assign to (its part segment is "
                         f"{segmentation.status!r}); assemble the molecules alone with segmentation=None")
    kinds = sorted({"assignment" if item.assignment is not None else "molecules" for item in inputs})
    if kinds != ["molecules"]:
        raise ValueError("with a segmentation the inputs are molecule tables (AssemblyInput(molecules=…)); the "
                         "sample's cells come from the sample segmentation, not from per-FOV assignments")
    frame = f"sample:{placement.sample}"
    if segmentation.grid.metadata.frame_id != frame:
        raise ValueError(f"the sample segmentation is on frame {segmentation.grid.metadata.frame_id!r}, not "
                         f"{frame!r}")
    recorded = ((segmentation.record.get("inputs") or {}).get("placement") or {})
    hashes = {"fovs_sha256": table_sha256(placement.fovs), "links_sha256": table_sha256(placement.links)}
    differ = [key for key, value in hashes.items() if recorded.get(key) not in (None, value)]
    if differ:
        raise ValueError(f"the sample segmentation was made from another placement: its {differ[0]} is "
                         f"{recorded[differ[0]]}, the placement's {hashes[differ[0]]}")


def sample_molecule_table(base: AssemblyResult, grid, namespace: str, population: str) -> tuple[MoleculeTable,
                                                                                                np.ndarray]:
    """MT1: the retained molecules of the ``molecules_only`` assembly as one MoleculeTable on the segmentation grid.

    Returns the table and the positions of its rows in the sample molecule
    table; ``spot_id`` is that position as text.
    """
    molecules = base.molecules
    rows = np.flatnonzero(molecules["assembly_status"].eq("retained").to_numpy(dtype=bool))
    sample_zyx = molecules[[f"sample_{a}" for a in AXES]].to_numpy(np.float64)[rows]
    zyx = (sample_zyx - np.asarray(grid.metadata.origin_zyx, np.float64)) / \
        np.asarray(grid.metadata.spacing_zyx, np.float64)
    table = pd.DataFrame({"spot_namespace": pd.array([namespace] * len(rows), dtype="string"),
                          "spot_id": pd.array([str(int(i)) for i in rows], dtype="string"),
                          "z": zyx[:, 0], "y": zyx[:, 1], "x": zyx[:, 2],
                          "gene_id": pd.array(molecules["gene_id"].astype(object).to_numpy()[rows].tolist(),
                                              dtype="string")})
    source = {"spot_namespace": namespace, "conversion": CONVERSION,
              "rows": "the retained molecules of the molecules_only assembly; spot_id is the row's position in "
                      "the sample molecule table, z, y, x = (sample_zyx - O) / S on the segmentation grid"}
    return MoleculeTable(table, tuple(base.genes), population, source), rows


def _scatter(n, rows, values, dtype):
    """A column of length ``n``, null except at ``rows``, where it holds ``values``."""
    out = pd.array([None] * n, dtype=dtype)
    if len(rows):
        out[rows] = pd.array(values, dtype=dtype)
    return out


def _nearest_owner(points_yx, geometry):
    """F1 outside every owning footprint: the owning FOV with the nearest footprint centre, ties by table order."""
    owning = geometry[geometry["owns_points"].to_numpy(dtype=bool)]
    centres = owning[["centre_y", "centre_x"]].to_numpy(np.float64)
    ids = owning["fov_id"].astype(object).to_numpy()
    squared = ((points_yx[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return ids[np.argmin(squared, axis=1)]   # argmin keeps the first of equal distances (table order)


def assemble_after_fusion(placement, inputs, config, segmentation, base: AssemblyResult) -> AssemblyResult:
    """The keyword ``segmentation`` of :func:`assemble`: ``base`` is the ``molecules_only`` assembly of ``inputs``."""
    grid = segmentation.grid
    cells_run, nuclei_run = segmentation.cells, segmentation.nuclei
    sample = placement.sample
    namespaces = [item.molecules.spot_namespace for item in inputs]
    dataset_id, sample_id = _identity(namespaces[0])
    labels_identity = _identity(cells_run.label_namespace)
    if labels_identity != [dataset_id, sample_id]:
        raise ValueError(f"the sample segmentation's labels are of [dataset_id, sample_id] {labels_identity}, the "
                         f"molecules of {[dataset_id, sample_id]}; segment_fused needs identity=(dataset_id, "
                         "sample_id) of the dataset")
    k = int(segmentation.record["inputs"]["grid"]["factor_yx"])
    given = assignment_config(segmentation.record.get("assignment"))
    namespace = json.dumps([dataset_id, sample_id, None, None], separators=(",", ":"))
    population = next(iter(item.molecules.population for item in inputs))
    table, rows = sample_molecule_table(base, grid, namespace, population)
    assigned = assign_molecules(table, cells_run, grid=grid, nuclei=nuclei_run, config=used_assignment(given, k))

    # The molecules: the molecules_only rows, with the sample assignment's columns at the rows it placed.
    molecules = base.molecules
    n = len(molecules)
    placed = assigned.molecules["assignment_status"].ne("outside_grid").to_numpy(dtype=bool)
    at = rows[placed]
    columns = {"fov_id": molecules["fov_id"]}
    for column in ("spot_namespace", "spot_id", "z", "y", "x", "gene_id"):
        columns[column] = molecules[column]
    for column in ASSIGNED_COLUMNS:
        columns[column] = _scatter(n, at, assigned.molecules[column].array[placed], MOLECULE_DTYPES_29[column])
    has_cell = pd.Series(columns["cell_id"]).notna().to_numpy(dtype=bool)
    columns["cell_namespace"] = pd.array(np.where(has_cell, cells_run.label_namespace, None), dtype="string")
    for column in ("sample_z", "sample_y", "sample_x", "owner_fov_id"):
        columns[column] = molecules[column]
    status = molecules["assembly_status"].astype(object).to_numpy().copy()
    reason = molecules["assembly_reason"].astype(object).to_numpy().copy()
    outside = rows[~placed]
    status[outside], reason[outside] = "excluded", "outside_grid"
    columns["assembly_status"] = status
    columns["assembly_reason"] = pd.array(list(reason), dtype="string")
    columns["assembly_flags"] = [""] * n
    dtypes = molecule_dtypes("cells")
    molecules = pd.DataFrame(columns, columns=list(dtypes)).astype(dtypes)

    # The cells: one retained row per cell of the cell run, in increasing cell_id.
    cells29 = assigned.cells.reset_index(drop=True)
    cell_ids = cells29["cell_id"].astype("int64").to_numpy()
    origin = np.asarray(grid.metadata.origin_zyx, np.float64)
    spacing = np.asarray(grid.metadata.spacing_zyx, np.float64)
    original = origin + cells29[[f"centroid_{a}" for a in AXES]].to_numpy(np.float64) * spacing
    grown = origin + cells29[[f"expanded_centroid_{a}" for a in AXES]].to_numpy(np.float64) * spacing
    original[:, 0] = np.nan   # the labels are a plane: no sample z
    grown[:, 0] = np.nan
    expanded = assigned.territories is not None
    use_expanded = config.centroid == "territory" and expanded
    own = grown if use_expanded else original
    ownership = ownership_map(placement, {item.fov_id: item.grid for item in inputs}, config=config)
    owner = ownership.owner(own) if len(own) else np.array([], dtype=object)
    fov_of = owner.copy()
    outside_every = np.array([v is None for v in owner], dtype=bool)
    if outside_every.any():
        fov_of[outside_every] = _nearest_owner(own[outside_every][:, 1:], ownership.geometry)
    front = pd.DataFrame({"fov_id": pd.array(list(fov_of), dtype="string"),
                          "cell_namespace": [cells_run.label_namespace] * len(cells29),
                          "cell_id": cells29["cell_id"].array,
                          "obs_name": [f"{sample}_{f}_{c}" for f, c in zip(fov_of, cell_ids.tolist())]})
    frame = pd.concat([front, cells29[[c for c in CELL_DTYPES_29 if c != "cell_id"]]], axis=1)
    for name, values in (("sample_centroid", original), ("sample_expanded_centroid", grown)):
        for i, a in enumerate(AXES):
            frame[f"{name}_{a}"] = values[:, i]
    frame["ownership_centroid"] = "expanded" if use_expanded else "original"
    for i, a in enumerate(AXES):
        frame[f"sample_ownership_centroid_{a}"] = own[:, i]
    frame["owner_fov_id"] = pd.array(list(owner), dtype="string")
    frame["assembly_status"] = "retained"
    frame["assembly_reason"] = pd.array([None] * len(frame), dtype="string")
    frame["assembly_flags"] = ""
    frame["conflict_with"] = ""
    frame["duplicate_of"] = pd.array([None] * len(frame), dtype="string")
    cells = frame[list(CELL_DTYPES)].astype(CELL_DTYPES).reset_index(drop=True)

    # The counts of the counted cells (every kept cell), with the cell's fov_id and the run's namespace.
    counts = assigned.counts.reset_index(drop=True)
    fov_by_cell = dict(zip(cell_ids.tolist(), fov_of))
    counts = pd.concat([pd.DataFrame({"fov_id": [fov_by_cell[int(c)] for c in counts["cell_id"]],
                                      "cell_namespace": [cells_run.label_namespace] * len(counts)}), counts], axis=1)
    counts = counts[list(COUNT_DTYPES)].astype(COUNT_DTYPES).reset_index(drop=True)

    # The fovs table: the cells whose fov_id is the FOV; the retained molecules after the sample assignment.
    fovs = base.fovs.copy()
    per_fov = cells["fov_id"].astype(object).value_counts()
    kept = molecules[molecules["assembly_status"].eq("retained").to_numpy(dtype=bool)]["fov_id"].astype(object)
    kept = kept.value_counts()
    fovs["n_cells"] = [int(per_fov.get(f, 0)) for f in fovs["fov_id"]]
    fovs["n_cells_retained"] = fovs["n_cells"]
    fovs["n_molecules_retained"] = [int(kept.get(f, 0)) for f in fovs["fov_id"]]
    fovs = fovs.astype(base.fovs.dtypes.to_dict())

    from starfinder._records import json_record
    record = dict(base.record)
    record.update({
        "mode": "cells", "route": ROUTE, "route_source": None,
        "config": dict(asdict(config), duplicate_threshold=None),
        "flags": {"computed": [], "conflicts": bool(config.conflicts), "applicable": False},
        "counts": _counts_record(cells, molecules, fovs, set(), False),
        "segmentation_files": [{"run": run["name"], "path": run["path"], "sha256": run["sha256"]}
                               for run in segmentation.record["runs"]],
        "assignment": assigned.record,
    })
    result = AssemblyResult(sample, fovs, cells, molecules, counts, base.genes, json_record(record))
    if expanded:
        return with_territories(result, (np.ascontiguousarray(assigned.territories, dtype=np.uint32), grid.metadata))
    return result
