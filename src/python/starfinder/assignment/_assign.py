"""The assign entry: direct territory assignment, one recorded expansion, correspondence and counts."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from starfinder.image import IncompatibleGeometryError
from starfinder.segmentation import ReferenceGrid, SegmentationResult, expand_labels
from starfinder.segmentation._labels import GEOMETRIES, _grid_record, _json, array_sha256

from ._config import (ASSIGNMENT_STATUSES, CELL_CORRESPONDENCE, COMPARTMENT_STATES, EXCLUSION_RATIONALE,
                      EXCLUSION_REASON, SAMPLING_RULE, AssignmentConfig)
from ._correspondence import check_supplied, match_nuclei
from ._molecules import MoleculeTable, _fov_identity

FORMAT_VERSION = 1
COMPARTMENTS = ("whole", "nucleus", "cytoplasm")
_MOLECULE_STATES = ("nucleus", "cytoplasm", "withheld", "no_nucleus", "unavailable")


def sample_labels(labels, positions_zyx, *, geometry: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a label image at molecule positions with the rule ``i = floor(c + 0.5)`` per axis.

    Voxel ``i`` covers ``[i − 0.5, i + 0.5)``, so a position at −0.5 samples voxel 0
    and one at ``n − 0.5`` is outside. ``volume`` and ``extended`` labels are read at
    ``z, y, x``; ``plane`` labels (Z=1) at ``y, x``, and ``z`` selects nothing (its
    bounds belong to the molecule grid, which the caller checks).

    Parameters
    ----------
    labels : numpy.ndarray
        ZYX label image (a plane is 1×Y×X).
    positions_zyx : array_like
        Finite zero-based positions, shape (m, 3).
    geometry : str
        ``volume``, ``plane`` or ``extended``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``values`` (the label at each sampled voxel, 0 outside), ``voxels``
        (int64 (m, 3), the sampled index per axis and −1 on an axis where it is
        outside; Z is 0 for a plane) and ``inside`` (bool, inside on every
        sampled axis).

    Raises
    ------
    ValueError
        Labels that are not ZYX, an unknown geometry or one that disagrees with
        Z, or positions that are not finite (m, 3).
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError("labels must be ZYX (a plane is 1×Y×X)")
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be volume, plane or extended; got {geometry!r}")
    if (geometry == "plane") != (labels.shape[0] == 1):
        raise ValueError(f"geometry {geometry} disagrees with labels of Z={labels.shape[0]}")
    positions = np.asarray(positions_zyx, np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("positions_zyx must be finite with shape (m, 3)")
    index = np.floor(positions + 0.5)
    sizes = np.asarray(labels.shape, np.float64)
    inside_axes = (index >= 0) & (index < sizes)
    if geometry == "plane":
        index[:, 0], inside_axes[:, 0] = 0, True
    voxels = np.where(inside_axes, index, -1).astype(np.int64)
    inside = inside_axes.all(axis=1)
    values = np.zeros(len(positions), labels.dtype)
    z, y, x = voxels[inside].T
    values[inside] = labels[z, y, x]
    return values, voxels, inside


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _record_json(record):
    from starfinder.io._checkpoint import _jsonable
    return json.loads(json.dumps(_jsonable(record), default=str))


def table_sha256(frame: pd.DataFrame) -> str:
    """SHA-256 of a table's column names, dtypes and CSV text (floats with 17 significant digits)."""
    header = json.dumps([[str(c), str(t)] for c, t in frame.dtypes.items()])
    text = frame.to_csv(index=False, float_format="%.17g")
    return hashlib.sha256((header + "|" + text).encode()).hexdigest()


def _projection_method(cells):
    """The projection that put a plane run on its grid: its record's, or the one its frame names."""
    projection = (cells.record.get("input") or {}).get("projection")
    if isinstance(projection, Mapping) and projection.get("method"):
        return projection["method"]
    frame = cells.grid.metadata.frame_id
    for method in ("max", "sum"):
        if frame.endswith(f"/projection:{method}"):
            return method
    return None


def _check_grid(cells, grid):
    """Check 4: the cell grid against the molecule run's; returns ``checked`` or ``declared_checked``."""
    expected = grid
    if cells.geometry == "plane" and grid.shape_zyx[0] > 1:
        method = _projection_method(cells)
        expected = grid.projected(method=method) if method else None
    if (expected is None or cells.grid.shape_zyx != expected.shape_zyx
            or cells.grid.metadata != expected.metadata):
        frame = cells.grid.metadata.frame_id
        raise IncompatibleGeometryError(
            f"{cells.geometry} labels of run {cells.record.get('run')!r} on grid {cells.grid.shape_zyx} "
            f"(frame {frame!r}) are not on the molecule grid {grid.shape_zyx} (frame {grid.metadata.frame_id!r})"
            + ("" if expected is None or expected is grid else
               f" projected to {expected.shape_zyx} (frame {expected.metadata.frame_id!r})"))
    return "declared_checked" if cells.grid.source == "declared" else "checked"


def _calibration(cells, grid):
    """The metadata that calibrates the cells, and its source: the cell grid, or the molecule grid of a projection."""
    def known(metadata):
        return metadata.spacing_zyx is not None and metadata.spatial_unit is not None
    if known(cells.grid.metadata):
        return cells.grid.metadata, "grid"
    if cells.geometry == "plane" and grid.shape_zyx[0] > 1 and known(grid.metadata):
        return grid.metadata, "projection_source"
    return None, None


def _expanded_by_segmentation(cells):
    return any(isinstance(op, Mapping) and op.get("operation") == "expand_labels"
               for op in cells.record.get("operations") or ())


def _territory_stats(labels):
    """Per positive value: the voxel count and the mean voxel index, accumulated per Z plane."""
    values, counts, sums = [], [], []
    yy, xx = np.indices(labels.shape[1:], dtype=np.float64)
    for z in range(labels.shape[0]):
        plane = labels[z].ravel()
        unique, inverse = np.unique(plane, return_inverse=True)
        n = np.bincount(inverse, minlength=len(unique))
        values.append(unique)
        counts.append(n)
        sums.append(np.stack([n * float(z), np.bincount(inverse, weights=yy.ravel(), minlength=len(unique)),
                              np.bincount(inverse, weights=xx.ravel(), minlength=len(unique))], axis=1))
    unique, inverse = np.unique(np.concatenate(values), return_inverse=True)
    size = np.zeros(len(unique), np.int64)
    np.add.at(size, inverse, np.concatenate(counts))
    total = np.zeros((len(unique), 3), np.float64)
    np.add.at(total, inverse, np.concatenate(sums))
    keep = unique > 0
    return unique[keep].astype(np.int64), size[keep], total[keep] / size[keep, None]


def _nullable(values, mask, dtype):
    """A pandas extension array of values, null where mask is False."""
    return pd.array([v if m else None for v, m in zip(np.asarray(values).tolist(), mask)], dtype=dtype)


@dataclass(frozen=True, eq=False)
class AssignmentResult:
    """The result of :func:`assign_molecules` for one FOV.

    ``molecules`` has one row per input molecule (its status, cell, nucleus and
    compartment); ``cells`` one row per territory (status, sizes, centroids,
    correspondence, compartments); ``counts`` the nonzero long counts (``cell_id``,
    ``gene_id``, ``compartment``, ``count``) of the kept cells; ``nuclei`` one row
    per nucleus, or None without nuclei. ``cell_labels`` are the original
    territories (the cell run's labels), ``territories`` the expanded ones (None
    without expansion) and ``nucleus_labels`` the nucleus labels (None without
    nuclei), all ``uint32`` ZYX. A cell's key is ``(cell_namespace, cell_id)``.
    ``record`` is the run record of docs/assignment-contract.md.

    Raises
    ------
    AssertionError
        A count identity of docs/assignment-contract.md ("Count accounting")
        fails, naming it; it can only come from a defect.
    """

    molecules: pd.DataFrame
    cells: pd.DataFrame
    counts: pd.DataFrame
    nuclei: pd.DataFrame | None
    cell_labels: np.ndarray
    territories: np.ndarray | None
    nucleus_labels: np.ndarray | None
    genes: tuple[str, ...]
    cell_namespace: str
    record: Mapping[str, Any]

    def __post_init__(self):
        check_identities(self)

    def matrix(self, compartment: str = "whole", *, cells: str = "kept") -> tuple[np.ndarray, pd.DataFrame]:
        """A dense cells × genes ``int64`` count matrix in ``genes`` order, with its cell-table rows.

        ``compartment`` is ``whole``, ``nucleus`` or ``cytoplasm``. ``cells="kept"``
        gives the kept cells; ``cells="all"`` adds the excluded cells to the
        ``whole`` matrix (their ``excluded_cell`` molecules), for the totals before
        and after the exclusion. A compartment matrix holds only the kept cells
        whose compartments are ``available``; the other cells are absent, never
        zero rows.

        Raises
        ------
        ValueError
            An unknown compartment or cells value.
        """
        if compartment not in COMPARTMENTS:
            raise ValueError(f"compartment must be whole, nucleus or cytoplasm; got {compartment!r}")
        if cells not in ("kept", "all"):
            raise ValueError(f"cells must be kept or all; got {cells!r}")
        table = self.cells
        if compartment == "whole":
            rows = table if cells == "all" else table[table.status.eq("kept").to_numpy(dtype=bool)]
            statuses = ("assigned", "excluded_cell") if cells == "all" else ("assigned",)
            molecules = self.molecules[self.molecules.assignment_status.isin(statuses).to_numpy(dtype=bool)]
            pairs = pd.DataFrame({"cell_id": molecules.cell_id.astype("int64"), "gene_id": molecules.gene_id})
        else:
            rows = table[(table.status.eq("kept") & table.compartments.eq("available")).to_numpy(dtype=bool)]
            counts = self.counts[self.counts.compartment.eq(compartment).to_numpy(dtype=bool)]
            pairs = counts.loc[counts.index.repeat(counts["count"].to_numpy()), ["cell_id", "gene_id"]]
            pairs = pairs.astype({"cell_id": "int64"})
        rows = rows.reset_index(drop=True)
        row_of = {int(c): i for i, c in enumerate(rows.cell_id)}
        gene_of = {g: j for j, g in enumerate(self.genes)}
        matrix = np.zeros((len(rows), len(self.genes)), np.int64)
        if len(pairs):
            np.add.at(matrix, (pairs.cell_id.map(row_of).to_numpy(np.int64),
                               pairs.gene_id.map(gene_of).to_numpy(np.int64)), 1)
        return matrix, rows

    def __repr__(self):
        counts = self.record.get("counts", {})
        return (f"AssignmentResult: {len(self.molecules)} molecules ({counts.get('assigned')} assigned), "
                f"{len(self.cells)} cells ({counts.get('cells_kept')} kept)")


def _tally(cell_ids, keys, counts=None):
    """{(cell_id, key...): total} from parallel sequences; counts default to 1 each."""
    totals = {}
    for i, key in enumerate(zip(cell_ids, *keys)):
        key = (int(key[0]), *key[1:])
        totals[key] = totals.get(key, 0) + (1 if counts is None else int(counts[i]))
    return totals


def check_identities(result: AssignmentResult) -> None:
    """The count identities 1–5 of docs/assignment-contract.md on a result; AssertionError names a failure."""
    molecules, cells, counts = result.molecules, result.cells, result.counts
    status = molecules.assignment_status.astype(object)
    if status.isna().any() or not status.isin(ASSIGNMENT_STATUSES).all():
        raise AssertionError("identity 1: every molecule has exactly one status of ASSIGNMENT_STATUSES")
    n = {s: int(status.eq(s).sum()) for s in ASSIGNMENT_STATUSES}
    if sum(n.values()) != len(molecules):
        raise AssertionError("identity 1: the status counts sum to the number of molecules")
    assigned = molecules[status.eq("assigned").to_numpy(dtype=bool)]
    if assigned.cell_id.isna().any():
        raise AssertionError("identity 5: every assigned molecule has a cell")
    kept = cells[cells.status.eq("kept").to_numpy(dtype=bool)]
    excluded = cells[cells.status.eq("excluded_no_nucleus").to_numpy(dtype=bool)]
    compartment = counts.compartment.astype(object)
    whole = counts[compartment.eq("whole").to_numpy(dtype=bool)]
    whole_counts = _tally(whole.cell_id, [whole.gene_id], whole["count"].to_numpy())
    if whole_counts != _tally(assigned.cell_id, [assigned.gene_id]):
        raise AssertionError("identity 2: the whole-cell counts are the assigned molecules per cell and gene")
    per_cell = _tally(whole.cell_id, [], whole["count"].to_numpy())
    kept_molecules = {(int(c),): int(m) for c, m in zip(kept.cell_id, kept.n_molecules) if m}
    if per_cell != kept_molecules or sum(per_cell.values()) != n["assigned"]:
        raise AssertionError("identity 2: the whole-cell counts sum to n_assigned and per kept cell to n_molecules")
    available = {int(c) for c in kept.cell_id[kept.compartments.eq("available").to_numpy(dtype=bool)]}
    parts = counts[compartment.isin(("nucleus", "cytoplasm")).to_numpy(dtype=bool)]
    if not {int(c) for c in parts.cell_id} <= available:
        raise AssertionError("identity 3: only available cells have nucleus and cytoplasm rows")
    split = _tally(parts.cell_id, [parts.gene_id], parts["count"].to_numpy())
    if split != {k: v for k, v in whole_counts.items() if k[0] in available}:
        raise AssertionError("identity 3: nucleus + cytoplasm = whole per available cell and gene")
    states = molecules.compartment.astype(object)
    if (states.notna() != status.eq("assigned")).any() or not states.dropna().isin(_MOLECULE_STATES).all():
        raise AssertionError("identity 3: every assigned molecule, and only those, has one compartment value")
    if counts.cell_id.isin(excluded.cell_id).any() or int(excluded.n_molecules.sum()) != n["excluded_cell"]:
        raise AssertionError("identity 4: excluded cells have no count rows and hold the excluded_cell molecules")
    if not counts.gene_id.isin(result.genes).all() or (counts["count"] <= 0).any():
        raise AssertionError("identity 5: every counted gene is in genes and every count row is nonzero")


def _check_inputs(molecules, cells, grid, nuclei, correspondence, config):
    """Checks 1–8 of docs/assignment-contract.md, in order, before any sampling."""
    if not isinstance(config, AssignmentConfig):
        raise TypeError("config must be an AssignmentConfig")
    config.__post_init__()
    config.correspondence.__post_init__()
    if not isinstance(molecules, MoleculeTable):
        raise TypeError("molecules must be a MoleculeTable (see molecule_table)")
    if not isinstance(cells, SegmentationResult):
        raise TypeError("cells must be a SegmentationResult")
    if nuclei is not None and not isinstance(nuclei, SegmentationResult):
        raise TypeError("nuclei must be a SegmentationResult or None")
    if not isinstance(grid, ReferenceGrid):
        raise TypeError("grid must be the molecule run's ReferenceGrid")
    if cells.target not in ("cell", "nucleus"):
        raise ValueError(f"cells must have target cell or nucleus; got {cells.target!r}")
    if nuclei is not None and nuclei.target != "nucleus":
        raise ValueError(f"nuclei must have target nucleus; got {nuclei.target!r}")
    check = _check_grid(cells, grid)
    if nuclei is not None:
        if nuclei.grid.shape_zyx != cells.grid.shape_zyx or nuclei.grid.metadata != cells.grid.metadata:
            extend = (" (a plane nucleus image with extended cells: extend the nuclei through z too)"
                      if nuclei.geometry == "plane" and cells.geometry == "extended" else "")
            raise IncompatibleGeometryError(
                f"nuclei on grid {nuclei.grid.shape_zyx} (frame {nuclei.grid.metadata.frame_id!r}) are not on the "
                f"cell grid {cells.grid.shape_zyx} (frame {cells.grid.metadata.frame_id!r}){extend}")
        if nuclei.grid.source == "declared":
            check = "declared_checked"
    identity = _fov_identity(molecules.spot_namespace)
    for result in (cells, nuclei):
        if result is None:
            continue
        try:
            namespace = json.loads(result.label_namespace)
        except ValueError:
            namespace = None
        if not isinstance(namespace, list) or namespace[:4] != identity:
            raise ValueError(f"labels of namespace {result.label_namespace!r} are not of the molecules' FOV "
                             f"{molecules.spot_namespace!r}")
    if _expanded_by_segmentation(cells):
        raise ValueError(f"labels of run {cells.record.get('run')!r} were expanded by segmentation, which keeps no "
                         "original mask; run segment without the expansion and set AssignmentConfig.expansion")
    calibration, source = _calibration(cells, grid)
    if config.expansion is not None:
        if config.expansion.unit == "um" and calibration is None:
            raise ValueError("an expansion in um needs a calibrated grid (spacing_zyx and spatial_unit); "
                             "give the distance in pixels on this uncalibrated grid")
        if config.expansion.unit == "pixel" and calibration is not None and not config.legacy_pixel_expansion:
            raise ValueError("a pixel expansion on a calibrated grid needs legacy_pixel_expansion=True; "
                             "give the distance in um")
    pairs = None
    if correspondence is not None:
        if nuclei is None:
            raise ValueError("a supplied correspondence needs nuclei")
        cell_ids = np.unique(cells.labels)
        nucleus_ids = np.unique(nuclei.labels)
        pairs = check_supplied(correspondence, nucleus_ids[nucleus_ids > 0], cell_ids[cell_ids > 0])
    return check, calibration, source, pairs


def _label_input(result):
    """The record's entry for one label run: identity, hashes, operations and the full record."""
    record = _record_json(result.record)
    return {"run": record.get("run"), "target": result.target, "geometry": result.geometry,
            "label_namespace": result.label_namespace, "labels_sha256": array_sha256(result.labels),
            "record_sha256": _sha256_json(record), "operations": record.get("operations", []),
            "saved_under_run": False, "file": {"path": None, "sha256": None}, "record": record}


def assign_molecules(molecules: MoleculeTable, cells: SegmentationResult, *, grid: ReferenceGrid,
                     nuclei: SegmentationResult | None = None, correspondence: pd.DataFrame | None = None,
                     config: AssignmentConfig = AssignmentConfig()) -> AssignmentResult:
    """Assign every molecule of one FOV to the territory that holds its sampled voxel.

    Applies the checks of docs/assignment-contract.md ("Checks the entry applies")
    in order; expands the cell territories once when ``config.expansion`` is set,
    keeping the original labels; samples each molecule at ``floor(c + 0.5)`` per
    axis; relates the nuclei to the territories by overlap (or by the supplied
    ``correspondence`` table); partitions compartments where the correspondence
    allows; excludes the cells without a matched nucleus (by default when nuclei
    are given); counts; and checks the count identities. Every molecule gets one
    status: ``assigned``, ``unassigned`` (outside every territory),
    ``excluded_cell`` (in an excluded cell) or ``outside_grid``. Nothing is
    filtered by size or count, nothing is repaired, and nothing is read or
    written.

    Parameters
    ----------
    molecules : MoleculeTable
        The molecules, zero-based on the molecule run's grid.
    cells : SegmentationResult
        The territories: a ``cell`` run, or a ``nucleus`` run used as territories.
    grid : ReferenceGrid
        The molecule run's reference grid (``FOV.reference_grid()``).
    nuclei : SegmentationResult, optional
        A ``nucleus`` run on the cell grid; enables correspondence, compartments
        and the exclusion.
    correspondence : pandas.DataFrame, optional
        Supplied ``nucleus_id``, ``cell_id`` pairs instead of the majority rule.
    config : AssignmentConfig
        Expansion, correspondence thresholds and the exclusion.

    Returns
    -------
    AssignmentResult

    Raises
    ------
    TypeError
        An argument of the wrong type.
    ValueError
        A wrong target, labels of another FOV, a cell run expanded by
        segmentation, an expansion unit the grid's calibration does not allow,
        or a supplied correspondence with unknown or repeated values.
    IncompatibleGeometryError
        Cells not on the molecule grid (or its projection), or nuclei not on
        the cell grid.
    """
    check, calibration, calibration_source, pairs = _check_inputs(molecules, cells, grid, nuclei, correspondence,
                                                                   config)
    cell_labels = cells.labels
    territories, expansion = None, None
    if config.expansion is not None:
        metadata = calibration if calibration is not None else cells.grid.metadata
        territories, expanded = expand_labels(cell_labels, metadata, config=config.expansion)
        expansion = {"function": "expand_labels", "mode": config.expansion.mode, "unit": config.expansion.unit,
                     "distance": config.expansion.distance,
                     "spacing_yx": None if calibration is None else list(calibration.spacing_zyx[1:]),
                     "physical_distance_yx": expanded["physical_distance_yx"],
                     "voxels_added": expanded["voxels_added"],
                     "original": {"path": None, "sha256": array_sha256(cell_labels)},
                     "expanded": {"path": None, "sha256": expanded["output"]}, "record": expanded}
    sampled = cell_labels if territories is None else territories
    geometry = cells.geometry

    # Sampling: the territory, the original territory and the nucleus at one voxel.
    table = molecules.table
    positions = table[["z", "y", "x"]].to_numpy(np.float64)
    values, voxels, inside = sample_labels(sampled, positions, geometry=geometry)
    if geometry == "plane":
        z_index = np.floor(positions[:, 0] + 0.5)
        z_inside = (z_index >= 0) & (z_index < grid.shape_zyx[0])
        inside = inside & z_inside
        voxels[~z_inside, 0] = -1
    values = np.where(inside, values, 0).astype(np.int64)
    original = (sample_labels(cell_labels, positions, geometry=geometry)[0].astype(np.int64)
                if territories is not None else None)
    nucleus_values = (sample_labels(nuclei.labels, positions, geometry=nuclei.geometry)[0].astype(np.int64)
                      if nuclei is not None else None)

    # Cells: sizes and centroids of both territories, correspondence and the exclusion.
    cell_ids, size, centroid = _territory_stats(cell_labels)
    if territories is not None:
        expanded_ids, expanded_size, expanded_centroid = _territory_stats(territories)
        if not np.array_equal(expanded_ids, cell_ids):
            raise AssertionError("expand_labels created or removed a territory value")
    if nuclei is not None:
        supplied = None if pairs is None else pd.DataFrame(
            {"nucleus_id": np.array(list(pairs), np.int64), "cell_id": np.array(list(pairs.values()), np.int64)})
        nucleus_table, per_cell = match_nuclei(nuclei.labels, sampled, config=config.correspondence,
                                               supplied=supplied)
        seeds = cells.record.get("seeds")
        if isinstance(seeds, Mapping) and seeds.get("label_rule") == "seed_values" \
                and seeds.get("sha256") == array_sha256(nuclei.labels):
            agrees = nucleus_table.status.eq("matched") & nucleus_table.cell_id.eq(nucleus_table.nucleus_id)
            nucleus_table["seed_value_agrees"] = pd.array(agrees.fillna(False).to_numpy(bool), dtype="boolean")
    else:
        nucleus_table = None
        per_cell = pd.DataFrame({"cell_id": pd.array(cell_ids, dtype="UInt32"),
                                 "n_nuclei": pd.array([None] * len(cell_ids), dtype="Int64"),
                                 "correspondence": pd.array(["unavailable"] * len(cell_ids), dtype="string"),
                                 "correspondence_flags": pd.array([None] * len(cell_ids), dtype="string"),
                                 "compartments": pd.array(["unavailable"] * len(cell_ids), dtype="string")})
    exclude = config.exclude_cells_without_nucleus
    exclusion_source = "default" if exclude is None else "config"
    exclude = nuclei is not None and (exclude is None or exclude)
    correspondence_of = per_cell.correspondence.to_numpy(object)
    excluded = exclude & (correspondence_of == "no_nucleus")
    excluded_ids = set(cell_ids[excluded].tolist())
    compartments_of = dict(zip(cell_ids.tolist(), per_cell.compartments.astype(object)))

    # Molecules: status, cell, nucleus and compartment.
    status = np.where(~inside, "outside_grid",
                      np.where(values == 0, "unassigned",
                               np.where(np.isin(values, list(excluded_ids)), "excluded_cell", "assigned")))
    assigned = status == "assigned"
    nucleus_cell = ({} if nucleus_table is None else
                    {int(n): int(c) for n, c in zip(nucleus_table.nucleus_id, nucleus_table.cell_id) if pd.notna(c)})
    compartment = []
    for is_assigned, cell, nucleus in zip(assigned, values.tolist(),
                                          nucleus_values.tolist() if nucleus_values is not None else [0] * len(values)):
        if not is_assigned:
            compartment.append(None)
            continue
        state = compartments_of[cell]
        if state == "available":
            state = "nucleus" if nucleus and nucleus_cell.get(nucleus) == cell else "cytoplasm"
        compartment.append(state)
    expanded_flag = territories is not None
    molecule_table = pd.DataFrame({
        "spot_namespace": table.spot_namespace.array, "spot_id": table.spot_id.array,
        "z": table.z.to_numpy(np.float64), "y": table.y.to_numpy(np.float64), "x": table.x.to_numpy(np.float64),
        "gene_id": table.gene_id.array,
        "voxel_z": _nullable(voxels[:, 0], voxels[:, 0] >= 0, "Int64"),
        "voxel_y": _nullable(voxels[:, 1], voxels[:, 1] >= 0, "Int64"),
        "voxel_x": _nullable(voxels[:, 2], voxels[:, 2] >= 0, "Int64"),
        "assignment_status": pd.array(status.tolist(), dtype="string"),
        "cell_id": _nullable(values, values > 0, "UInt32"),
        "in_expansion": (_nullable((values > 0) & (original == 0), inside, "boolean") if expanded_flag
                         else pd.array([None] * len(values), dtype="boolean")),
        "original_cell_id": (_nullable(original, inside, "UInt32") if expanded_flag
                             else pd.array([None] * len(values), dtype="UInt32")),
        "nucleus_id": (_nullable(nucleus_values, inside, "UInt32") if nucleus_values is not None
                       else pd.array([None] * len(values), dtype="UInt32")),
        "compartment": pd.array(compartment, dtype="string"),
    })

    # Cell table.
    in_cell = molecule_table.assignment_status.isin(("assigned", "excluded_cell")).to_numpy(dtype=bool)
    n_molecules = pd.Series(values[in_cell]).value_counts().reindex(cell_ids, fill_value=0).to_numpy(np.int64)
    plane = geometry == "plane"
    if calibration is not None:
        sz, sy, sx = calibration.spacing_zyx
        unit_size = sy * sx if plane else sz * sy * sx
        size_unit = f"{calibration.spatial_unit}^{2 if plane else 3}"
    else:
        unit_size, size_unit = np.nan, None
    nan = np.full(len(cell_ids), np.nan)
    cell_table = pd.DataFrame({
        "cell_id": pd.array(cell_ids, dtype="UInt32"),
        "status": pd.array(np.where(excluded, "excluded_no_nucleus", "kept").tolist(), dtype="string"),
        "exclusion_reason": pd.array([EXCLUSION_REASON if e else None for e in excluded], dtype="string"),
        "size_voxels": size.astype(np.int64),
        "expanded_size_voxels": (pd.array(expanded_size, dtype="Int64") if expanded_flag
                                 else pd.array([None] * len(cell_ids), dtype="Int64")),
        "size_physical": size * unit_size,
        "expanded_size_physical": expanded_size * unit_size if expanded_flag else nan,
        "centroid_z": centroid[:, 0], "centroid_y": centroid[:, 1], "centroid_x": centroid[:, 2],
        "expanded_centroid_z": expanded_centroid[:, 0] if expanded_flag else nan,
        "expanded_centroid_y": expanded_centroid[:, 1] if expanded_flag else nan,
        "expanded_centroid_x": expanded_centroid[:, 2] if expanded_flag else nan,
        "n_molecules": n_molecules,
        "n_nuclei": per_cell.n_nuclei.array,
        "correspondence": per_cell.correspondence.array,
        "correspondence_flags": per_cell.correspondence_flags.array,
        "compartments": per_cell.compartments.array,
    })

    # Counts: whole for the kept cells, nucleus and cytoplasm for the available ones; nonzero rows only.
    gene_order = {g: i for i, g in enumerate(molecules.genes)}
    counted = molecule_table[assigned]
    whole = counted.groupby([counted.cell_id.astype("int64"), counted.gene_id.astype(object)]).size()
    whole = whole.rename("count").reset_index().assign(compartment="whole")
    parts = counted[counted.compartment.isin(("nucleus", "cytoplasm")).to_numpy(dtype=bool)]
    parts = parts.groupby([parts.cell_id.astype("int64"), parts.gene_id.astype(object),
                           parts.compartment.astype(object)]).size().rename("count").reset_index()
    long = pd.concat([whole, parts], ignore_index=True)
    if len(long):
        long = long.assign(_c=long.compartment.map(COMPARTMENTS.index), _g=long.gene_id.map(gene_order))
        long = long.sort_values(["cell_id", "_c", "_g"], kind="mergesort")
    counts = pd.DataFrame({"cell_id": pd.array(long.cell_id.to_numpy(np.int64) if len(long) else [], dtype="UInt32"),
                           "gene_id": pd.array(long.gene_id.tolist() if len(long) else [], dtype="string"),
                           "compartment": pd.array(long.compartment.tolist() if len(long) else [], dtype="string"),
                           "count": np.asarray(long["count"] if len(long) else [], np.int64)})

    # The record.
    ids = json.loads(cells.label_namespace)
    kept = ~excluded
    n_status = {s: int((status == s).sum()) for s in ASSIGNMENT_STATUSES}
    by_compartment = {s: int(sum(1 for c in compartment if c == s)) for s in _MOLECULE_STATES}
    record = {
        "format_version": FORMAT_VERSION, "stage": "assignment",
        "dataset_id": ids[0], "sample_id": ids[1], "fov_id": ids[2], "subtile_id": ids[3], "name": None,
        "cell_namespace": cells.label_namespace,
        "territory_source": {"run": cells.record.get("run"), "target": cells.target,
                             "origin": "imported" if cells.record.get("import") is not None else "segmented",
                             "expanded_by_assign": expanded_flag},
        "grid": {**_grid_record(grid), "check": check},
        "inputs": {
            "molecules": {"population": molecules.population, "n": len(table), "sha256": molecules.sha256,
                          "source": _record_json(molecules.source)},
            "cells": _label_input(cells),
            "nuclei": None if nuclei is None else _label_input(nuclei),
            "correspondence": None if nuclei is None else {
                "source": "overlap" if correspondence is None else "supplied",
                "sha256": None if correspondence is None else table_sha256(
                    pd.DataFrame({"nucleus_id": sorted(pairs), "cell_id": [pairs[n] for n in sorted(pairs)]}))},
        },
        "config": {"expansion": None if config.expansion is None else _json(asdict(config.expansion)),
                   "legacy_pixel_expansion": config.legacy_pixel_expansion,
                   "correspondence": _json(asdict(config.correspondence)),
                   "exclude_cells_without_nucleus": bool(exclude), "exclusion_source": exclusion_source,
                   "exclusion_rationale": EXCLUSION_RATIONALE},
        "sampling": {"rule": SAMPLING_RULE, "sampled_axes": "yx" if plane else "zyx"},
        "expansion": _record_json(expansion) if expansion is not None else None,
        "calibration": "known" if calibration is not None else "unknown",
        "calibration_source": calibration_source, "size_unit": size_unit,
        "counts": {"molecules": len(table), **n_status,
                   "cells": len(cell_ids), "cells_kept": int(kept.sum()), "cells_excluded": int(excluded.sum()),
                   "nuclei": 0 if nucleus_table is None else len(nucleus_table),
                   "correspondence": {s: int((correspondence_of == s).sum()) for s in CELL_CORRESPONDENCE},
                   "compartments": {s: int((per_cell.compartments.to_numpy(object)[kept] == s).sum())
                                    for s in COMPARTMENT_STATES},
                   "assigned_by_compartment": by_compartment},
        "outcome": "ok" if len(cell_ids) else "empty",
        "files": {},
        "software": _software(),
    }
    return AssignmentResult(molecule_table, cell_table, counts, nucleus_table, cell_labels, territories,
                            None if nuclei is None else nuclei.labels, molecules.genes, cells.label_namespace,
                            record)


def _software():
    from starfinder.segmentation._import import _software as software
    return software()
