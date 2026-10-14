"""The sample measurements of the report (docs/report-algorithms.md, "Reads assigned to cells", "`cells.csv`",
"`mecr.csv`", "`reads_along_z.csv`", "Sample measurements").

Read from the export at its fixed place: ``tables/molecules.csv`` and ``tables/cells.csv``
with the checkpoint table reader's conventions and only the columns a definition names,
``sample.h5ad`` with h5py (choice A10, H1: ``X`` dense or CSR by its ``encoding-type``,
``obs_names`` and the ``obs`` columns by name; no AnnData object is built) and the fused
image of ``export.json`` ``fusion.outputs`` with tifffile, plane by plane. Each file is
checked against the SHA-256 the export records for it, hashed in blocks without holding its
bytes. :class:`_Measurements` computes the reads assigned to cells (per FOV, in total and
against ``Σ X``), the rows of ``cells.csv``, ``mecr.csv`` and ``reads_along_z.csv``, the
``summary`` keys of the sample's cells and the rows of ``unavailable.csv`` of the parts
``sample.*``; :meth:`_Measurements.figures` draws the per-cell distributions, the marker
patterns (with the data of the group selector) and the reads along Z (``_figures``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ._values import Unavailable, first, is_value, median

#: The slab of a marker zoom-in region and the band of its XZ view, µm (choice A6, SL1).
SLAB_UM = 3.0
#: The diameter of a molecule's point, µm, and its floor in pixels of the drawn image (SL1).
POINT_UM, POINT_FLOOR_PX = 1.5, 2.0
#: The colours of a marker group's genes, in group order (Okabe–Ito).
GENE_COLOURS = ("#E69F00", "#56B4E9", "#009E73", "#F0E442")
#: The genes of the figure without markers.
ABUNDANT_GENES = 3
#: The order of the cell size's units ("Cell size").
SIZE_UNITS = ("µm³", "µm²", "voxels")
_WRITTEN = ("written", "reused")
_MOLECULE_COLUMNS = ("fov_id", "z", "gene_id", "assignment_status", "cell_id", "cell_namespace", "sample_z",
                     "sample_y", "sample_x", "assembly_status")
_CELL_COLUMNS = ("cell_namespace", "cell_id", "status", "assembly_status")


def _get(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


# --- Readers ------------------------------------------------------------------------------------------------------

def read_columns(path, dtypes, columns) -> pd.DataFrame:
    """The named columns of a checkpoint table (``io/_checkpoint.py``: ``_read_table``'s conventions: strings read
    verbatim and unescaped, ``<NA>`` a missing value, floats round-trip), with their recorded dtypes, in file
    order; a column the table does not record is left out."""
    from starfinder.io._checkpoint import _CSV_DTYPES, NA_TOKEN, _unescape_strings
    path = Path(path)
    wanted = [c for c in dtypes if c in columns]
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=wanted, engine="pyarrow")
    else:
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != list(dtypes):
            raise ValueError(f"{path.name}: columns differ from the checkpoint header")
        strings = [c for c in wanted if dtypes[c] in ("string", "str")]
        frame = pd.read_csv(path, usecols=wanted, dtype={c: _CSV_DTYPES[dtypes[c]] for c in wanted},
                            keep_default_na=False, float_precision="round_trip",
                            na_values={c: [NA_TOKEN] for c in wanted if c not in strings})
        for column in strings:
            frame[column] = _unescape_strings(frame[column])
    return frame[wanted].astype({c: dtypes[c] for c in wanted}).reset_index(drop=True)


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _encoding(node):
    return _text(node.attrs.get("encoding-type", ""))


def _element(node):
    """An ``obs`` or ``var`` column, or an index, as a NumPy array (strings as objects, a missing value None or
    NaN), by its ``encoding-type``."""
    import h5py
    kind = _encoding(node)
    if isinstance(node, h5py.Dataset):
        if kind in ("string-array", "string") or node.dtype.kind in ("O", "S"):
            return np.asarray(node.asstr()[...], dtype=object)
        return np.asarray(node[...])
    if kind == "categorical":
        categories, codes = _element(node["categories"]), np.asarray(node["codes"][...])
        values = np.full(len(codes), None, dtype=object)
        values[codes >= 0] = categories[codes[codes >= 0]]
        return values
    if kind in ("nullable-integer", "nullable-boolean"):
        values, mask = np.asarray(node["values"][...]), np.asarray(node["mask"][...], dtype=bool)
        return np.where(mask, np.nan, values.astype(np.float64))
    raise ValueError(f"{node.name}: an element of encoding {kind!r} is not read")


@dataclass
class AnnDataArrays:
    """What the report reads of ``sample.h5ad``: the obs names, the named ``obs`` columns, the var names and ``X``
    (a NumPy array, or a SciPy CSR matrix)."""

    obs_names: np.ndarray
    obs: dict
    var_names: np.ndarray
    X: object = field(repr=False)

    @property
    def n_obs(self) -> int:
        return len(self.obs_names)

    def row_sums(self) -> np.ndarray:
        """``Σ_j X_ij`` per row, rounded to int64 (``X`` holds integer counts)."""
        sums = np.asarray(self.X.sum(axis=1), dtype=np.float64).reshape(-1)
        return np.rint(sums).astype(np.int64)

    def row_genes(self) -> np.ndarray:
        """``#{j : X_ij > 0}`` per row."""
        return np.asarray((self.X > 0).sum(axis=1)).reshape(-1).astype(np.int64)

    def total(self) -> int:
        """``Σ_ij X_ij`` rounded to int64."""
        return int(np.rint(float(self.X.sum())))

    def column(self, gene) -> np.ndarray:
        """The whole-cell counts of a gene (zeros for a gene ``X`` does not hold)."""
        index = np.flatnonzero(self.var_names == gene)
        if not len(index):
            return np.zeros(self.n_obs, dtype=np.float64)
        column = self.X[:, int(index[0])]
        return np.asarray(column.toarray() if hasattr(column, "toarray") else column, dtype=np.float64).reshape(-1)


def read_h5ad(path, obs_columns=("fov_id", "size_physical", "size_voxels")) -> AnnDataArrays:
    """``sample.h5ad`` read with h5py (choice A10, H1): ``X`` as written by ``assignment.write_h5ad`` (a dense
    array) or a CSR or CSC matrix by its ``encoding-type``; ``obs_names``, the named ``obs`` columns and
    ``var_names``. A named column ``obs`` does not hold raises KeyError."""
    import h5py
    with h5py.File(path, "r") as handle:
        obs, var = handle["obs"], handle["var"]
        obs_names = _element(obs[_text(obs.attrs["_index"])])
        var_names = _element(var[_text(var.attrs["_index"])])
        columns = {}
        for name in obs_columns:
            if name not in obs:
                raise KeyError(f"{path}: obs has no column {name!r}")
            columns[name] = _element(obs[name])
        node = handle["X"]
        kind = _encoding(node)
        if isinstance(node, h5py.Dataset):
            matrix = np.asarray(node[...], dtype=np.float64)
            if matrix.ndim != 2:
                matrix = matrix.reshape(len(obs_names), len(var_names))
        elif kind in ("csr_matrix", "csc_matrix"):
            from scipy import sparse
            shape = tuple(int(n) for n in node.attrs["shape"])
            arrays = (np.asarray(node["data"][...], dtype=np.float64), np.asarray(node["indices"][...]),
                      np.asarray(node["indptr"][...]))
            matrix = (sparse.csr_matrix if kind == "csr_matrix" else sparse.csc_matrix)(arrays, shape=shape).tocsr()
        else:
            raise ValueError(f"{path}: X of encoding {kind!r} is not read")
    return AnnDataArrays(np.asarray([str(v) for v in obs_names], dtype=object), columns,
                         np.asarray([str(v) for v in var_names], dtype=object), matrix)


# --- Definitions on plain values (the tests compute them on hand tables) -------------------------------------------

def plane_of(z) -> np.ndarray:
    """A molecule's plane ``floor(z + 0.5)`` (the §2.9 sampling rule)."""
    return np.floor(np.asarray(z, dtype=np.float64) + 0.5).astype(np.int64)


def reads_along_z(fov_ids, fov_of, z, planes) -> list[dict]:
    """``reads_along_z.csv``: per FOV of ``fov_ids`` (list order) the rows ``k_lo … k_hi`` with
    ``k_lo = min(0, min P)``, ``k_hi = max(n_z − 1, max P)`` and the molecules on each plane.

    ``fov_of`` and ``z`` are the table's ``fov_id`` and ``z`` of every molecule (retained or not); ``planes`` maps
    each FOV to ``n_z`` of its grid (None for a FOV whose export entry records no grid: its rows run over plane 0
    and its molecules' planes).
    """
    fov_of = np.asarray(fov_of, dtype=object)
    plane = plane_of(z)
    rows = []
    for fov in fov_ids:
        own = plane[fov_of == fov]
        n_z = planes.get(fov) or 1
        low = min(0, int(own.min())) if len(own) else 0
        high = max(n_z - 1, int(own.max())) if len(own) else n_z - 1
        counts = np.bincount(own - low, minlength=high - low + 1) if len(own) else np.zeros(high - low + 1, int)
        rows += [{"fov_id": fov, "z": low + k, "reads": int(counts[k])} for k in range(high - low + 1)]
    return rows


def mecr(a, b, threshold) -> dict:
    """``mecr.csv``'s values of a pair from the whole-cell counts ``a`` and ``b`` of the counted cells: the cells
    positive (count ≥ ``threshold``) for A, for B, for both, for either, and ``both / either`` (``undefined`` when
    ``either`` is 0)."""
    a = np.asarray(a, dtype=np.float64) >= threshold
    b = np.asarray(b, dtype=np.float64) >= threshold
    both, either = int((a & b).sum()), int((a | b).sum())
    return {"cells_a": int(a.sum()), "cells_b": int(b.sum()), "both": both, "either": either,
            "mecr": both / either if either else Unavailable("undefined", "either")}


def size_unit_of(record, n_z=None) -> str:
    """The unit of a finite ``size_physical`` of the cells a FOV's assignment record wrote: its ``size_unit``
    (``µm^3``, ``µm^2``) written ``µm³``, ``µm²``; a record without the key, the cell input's ``geometry``;
    without a record, the FOV's grid (one plane: a plane)."""
    unit = (record or {}).get("size_unit") if isinstance(record, dict) else None
    if isinstance(unit, str) and unit:
        return unit.replace("^3", "³").replace("^2", "²")
    geometry = _get(record, "inputs", "cells", "geometry")
    plane = geometry == "plane" if geometry is not None else n_z == 1
    return "µm²" if plane else "µm³"


def cell_rows(arrays: AnnDataArrays, units) -> list[dict]:
    """``cells.csv``: one row per counted cell in ``obs`` order; ``size`` is ``obs.size_physical`` with the unit of
    its FOV (``units``: FOV → unit) when finite, else ``obs.size_voxels`` with ``voxels``."""
    reads, genes = arrays.row_sums(), arrays.row_genes()
    physical = np.asarray(arrays.obs["size_physical"], dtype=np.float64)
    voxels = np.asarray(arrays.obs["size_voxels"], dtype=np.float64)
    rows = []
    for i, name in enumerate(arrays.obs_names):
        fov = str(arrays.obs["fov_id"][i])
        finite = bool(np.isfinite(physical[i]))
        rows.append({"obs_name": str(name), "fov_id": fov, "reads": int(reads[i]), "genes": int(genes[i]),
                     "size": float(physical[i]) if finite else float(voxels[i]),
                     "size_unit": units[fov] if finite else "voxels"})
    return rows


def abundant_genes(genes, retained_genes, n=ABUNDANT_GENES) -> tuple[str, ...]:
    """The ``n`` genes with the most retained molecules, in decreasing order of molecules, ties by the order of
    ``genes`` (``assembly.genes``); genes without a retained molecule are not named."""
    counts = pd.Series(np.asarray(retained_genes, dtype=object)).value_counts()
    order = {g: i for i, g in enumerate(genes)}
    present = [g for g in counts.index if g in order] + [g for g in counts.index if g not in order]
    ranked = sorted(present, key=lambda g: (-int(counts[g]), order.get(g, len(order))))
    return tuple(ranked[:n])


@dataclass(frozen=True)
class GroupFigureData:
    """The data of a marker figure's group selector: the group's name, the figure's title, its genes and their
    colours, in group order."""

    name: str
    title: str
    genes: tuple
    colours: tuple


def marker_groups(config, genes, retained_genes) -> list[GroupFigureData]:
    """One entry per marker group of the configuration, titled with its name; without markers, the figure of the
    three most abundant genes, titled "Most abundant genes (no markers supplied): A, B, C"."""
    if config.markers:
        return [GroupFigureData(m.name, m.name, m.genes, GENE_COLOURS[:len(m.genes)]) for m in config.markers]
    top = abundant_genes(genes, retained_genes)
    return [GroupFigureData("most_abundant", f"Most abundant genes (no markers supplied): {', '.join(top)}", top,
                            GENE_COLOURS[:len(top)])]


def point_diameter(um_per_px) -> float:
    """A point's diameter in pixels of the drawn image: ``max(2, 1.5 µm / p)``."""
    return max(POINT_FLOOR_PX, POINT_UM / um_per_px)


def slab(sample_z) -> tuple[float, float, float] | None:
    """``(z_c, z_c − slab/2, z_c + slab/2)`` with ``z_c`` the median ``sample_z`` of a region's retained molecules,
    or None for a region without one."""
    values = np.asarray(sample_z, dtype=np.float64)
    if not len(values):
        return None
    centre = median(values)
    return centre, centre - SLAB_UM / 2, centre + SLAB_UM / 2


# --- The collection's measurements --------------------------------------------------------------------------------

@dataclass
class MarkerFigure:
    """The figure of one marker group: the selector data, the whole-sample view and the zoom-in tiles
    (``(role, view)`` → :class:`._figures.FigureImage` or :class:`._values.Unavailable`, ``view`` ``xy`` or
    ``xz``)."""

    group: GroupFigureData
    whole: object
    zooms: dict


@dataclass
class SampleFigures:
    """The static figures of "Sample measurements": the per-cell distributions (reads, genes, then one cell size
    per unit in the order µm³, µm², voxels), the marker figures and the reads along Z; a part that cannot be drawn
    is its :class:`._values.Unavailable`."""

    per_cell: object
    groups: object
    markers: object
    reads_along_z: object


class _Measurements:
    """The sample measurements of a collection (``_collect._Collection``), each source read once."""

    def __init__(self, collection):
        self.c = collection
        self._cache = {}

    def _once(self, name, make):
        if name not in self._cache:
            self._cache[name] = make()
        return self._cache[name]

    # The sources -------------------------------------------------------------------------------------------

    def _export_file(self, name, kind):
        """The path and entry of a file of ``export.json`` ``assembly.files``, checked against its SHA-256; or the
        reason it cannot be read (the export's state first)."""
        c = self.c
        if c.e_state is not None:
            return c.e_state
        entry = _get(c.assembly, "files", name) or {}
        default = {"molecules": "tables/molecules.csv", "cells": "tables/cells.csv", "anndata": "sample.h5ad"}[name]
        path = c.export_dir / entry.get("path", default)
        missing = c.inputs.verify(path, kind, entry.get("sha256"))
        return missing if missing is not None else (path, entry)

    def molecules(self):
        """The export's molecule table, the columns the definitions name, or the reason it cannot be read."""
        def make():
            found = self._export_file("molecules", "export_table")
            if isinstance(found, Unavailable):
                return found
            path, entry = found
            return read_columns(path, entry.get("dtypes") or {}, _MOLECULE_COLUMNS)
        return self._once("molecules", make)

    def counted_keys(self):
        """``K``: the ``(cell_namespace, cell_id)`` of the cells of the export's cell table with ``assembly_status``
        ``retained`` and ``status`` ``kept``."""
        def make():
            found = self._export_file("cells", "export_table")
            if isinstance(found, Unavailable):
                return found
            path, entry = found
            cells = read_columns(path, entry.get("dtypes") or {}, _CELL_COLUMNS)
            keep = cells["assembly_status"].eq("retained").fillna(False).to_numpy(bool) & \
                cells["status"].eq("kept").fillna(False).to_numpy(bool)
            return pd.DataFrame({"cell_namespace": cells["cell_namespace"][keep].astype(object).to_numpy(),
                                 "cell_id": cells["cell_id"][keep].astype("int64").to_numpy()})
        return self._once("counted_keys", make)

    def anndata(self):
        """``sample.h5ad`` (:class:`AnnDataArrays`), or the reason it cannot be read."""
        def make():
            missing = first(self.c.e_state, self.c.no_cells)
            if missing is not None:
                return missing
            found = self._export_file("anndata", "anndata")
            return found if isinstance(found, Unavailable) else read_h5ad(found[0])
        return self._once("anndata", make)

    # Reads assigned to cells ------------------------------------------------------------------------------------

    def reads_in_cells(self):
        """FOV → ``reads_in_cells(f)``, the retained ``assigned`` molecules of counted cells; or the reason."""
        def make():
            missing = first(self.c.e_state, self.c.no_cells)
            if missing is not None:
                return missing
            molecules, keys = self.molecules(), self.counted_keys()
            missing = first(molecules, keys)
            if missing is not None:
                return missing
            chosen = molecules["assembly_status"].eq("retained").fillna(False).to_numpy(bool) & \
                molecules["assignment_status"].eq("assigned").fillna(False).to_numpy(bool) & \
                molecules["cell_id"].notna().to_numpy(bool)
            own = pd.DataFrame({"fov_id": molecules["fov_id"][chosen].astype(object).to_numpy(),
                                "cell_namespace": molecules["cell_namespace"][chosen].astype(object).to_numpy(),
                                "cell_id": molecules["cell_id"][chosen].astype("int64").to_numpy()})
            counted = own.merge(keys.drop_duplicates(), on=["cell_namespace", "cell_id"], how="inner")
            per_fov = counted["fov_id"].value_counts()
            return {fov: int(per_fov.get(fov, 0)) for fov in self.c.fov_ids}
        return self._once("reads_in_cells", make)

    def reads_in_cells_of(self, fov):
        values = self.reads_in_cells()
        return values if isinstance(values, Unavailable) else values[fov]

    def counted_cells(self):
        """``n_obs`` of ``sample.h5ad``, or the reason."""
        arrays = self.anndata()
        return arrays if isinstance(arrays, Unavailable) else arrays.n_obs

    def whole_count_sum(self):
        """``h5ad_whole_count_sum``: ``Σ_ij X_ij`` rounded to int64, or the reason."""
        arrays = self.anndata()
        return arrays if isinstance(arrays, Unavailable) else arrays.total()

    # cells.csv, mecr.csv, reads_along_z.csv ----------------------------------------------------------------------

    def size_units(self):
        """FOV → the unit of its cells' finite ``size_physical``: on the route per FOV its assignment record's, on
        the route after fusion ``µm²`` (the export's segmentation is a plane)."""
        c = self.c
        if c.route == "after_fusion":
            return {fov: "µm²" for fov in c.fov_ids}
        planes = self.planes()
        return {fov: size_unit_of((c.assignments.get(fov) or (None,))[0], planes.get(fov)) for fov in c.fov_ids}

    def cells_rows(self):
        """The rows of ``cells.csv``, or the reason there are none."""
        def make():
            arrays = self.anndata()
            if isinstance(arrays, Unavailable):
                return arrays
            units = self.size_units()
            for fov in {str(v) for v in arrays.obs["fov_id"]} - set(units):
                units[fov] = size_unit_of(None, self.planes().get(fov))
            return cell_rows(arrays, units)
        return self._once("cells_rows", make)

    def mecr_rows(self):
        """The rows of ``mecr.csv``: one per pair of ``marker_pairs``, each count the reason of ``sample.h5ad`` when
        it cannot be read."""
        config = self.c.config
        arrays = self.anndata()
        rows = []
        for gene_a, gene_b in config.marker_pairs:
            row = {"gene_a": gene_a, "gene_b": gene_b, "threshold": config.positive_threshold}
            if isinstance(arrays, Unavailable):
                row.update({c: arrays for c in ("cells_a", "cells_b", "both", "either")})
                row["mecr"] = first(row["both"], row["either"])
            else:
                row.update(mecr(arrays.column(gene_a), arrays.column(gene_b), config.positive_threshold))
            rows.append(row)
        return rows

    def planes(self):
        """FOV → ``n_z`` of its grid (``export.json`` ``assembly.fovs[*].grid.shape_zyx[0]``), None without one."""
        planes = {}
        for entry in (self.c.assembly or {}).get("fovs") or []:
            grid = entry.get("grid") if isinstance(entry, dict) else None
            shape = grid.get("shape_zyx") if isinstance(grid, dict) else None
            planes[entry.get("fov_id")] = int(shape[0]) if shape else None
        return planes

    def reads_along_z_rows(self):
        """The rows of ``reads_along_z.csv``, or the reason there are none."""
        def make():
            molecules = self.molecules()
            if isinstance(molecules, Unavailable):
                return molecules
            return reads_along_z(self.c.fov_ids, molecules["fov_id"].astype(object).to_numpy(),
                                 molecules["z"].to_numpy(np.float64), self.planes())
        return self._once("reads_along_z_rows", make)

    # report.json summary ---------------------------------------------------------------------------------------

    @staticmethod
    def total(fovs):
        """``reads_in_cells``: ``Σ_f reads_in_cells(f)`` over the FOVs of ``fovs.csv`` rows with a value, or the
        first reason of their cells."""
        cells = [row["reads_in_cells"] for row in fovs]
        values = [v for v in cells if is_value(v)]
        return sum(values) if values else (first(*cells) or Unavailable("undefined", "no FOV"))

    def summary(self, fovs):
        """The keys of the sample's cells: ``counted_cells``, the per-cell medians, ``reads_in_cells``,
        ``h5ad_whole_count_sum`` and ``reads_in_cells_equal``."""
        rows = self.cells_rows()

        def per_cell(column):
            if isinstance(rows, Unavailable):
                return rows
            return median([r[column] for r in rows]) if rows else Unavailable("undefined", "no counted cell")
        total = self.total(fovs)
        whole = self.whole_count_sum()
        equal = first(total, whole)
        return {"counted_cells": self.counted_cells(), "reads_per_cell_median": per_cell("reads"),
                "genes_per_cell_median": per_cell("genes"), "reads_in_cells": total, "h5ad_whole_count_sum": whole,
                "reads_in_cells_equal": equal if equal is not None else bool(total == whole)}

    # The marker patterns ------------------------------------------------------------------------------------------

    def fused_sources(self):
        """The fused image of the marker patterns, the first image of ``export.json`` ``fusion.outputs``: its
        projection entry and its stack entry (either None), each file checked against its SHA-256; or the reason
        the part has none."""
        def make():
            from starfinder.sample_output._record import part_status
            c = self.c
            if c.e_state is not None:
                return c.e_state
            status = part_status(c.export, "fuse")
            if status not in _WRITTEN:
                return Unavailable("part_not_written", f"fuse: {status}")
            outputs = [o for o in _get(c.export, "fusion", "outputs") or [] if isinstance(o, dict)]
            name = outputs[0].get("image") if outputs else None
            entries = [o for o in outputs if o.get("image") == name and o.get("format", "ome_tiff") == "ome_tiff"]
            projection = next((o for o in entries if o.get("dimensionality") == "projection"), None)
            stack = next((o for o in entries if o.get("dimensionality") == "stack"), None)
            if projection is None and stack is None:
                return Unavailable("part_not_written", "fuse: export.json fusion.outputs holds no image")
            for entry in (projection, stack):
                if entry is not None:
                    missing = c.inputs.verify(c.export_dir / entry["path"], "fused_image", entry.get("sha256"))
                    if missing is not None:
                        return missing
            return {"image": name, "projection": projection, "stack": stack}
        return self._once("fused_sources", make)

    def markers_missing(self):
        """The reason the marker patterns cannot be drawn for the sample, or None."""
        molecules = self.molecules()
        return first(self.fused_sources(), molecules if isinstance(molecules, Unavailable) else None)

    def has_stack(self):
        sources = self.fused_sources()
        stack = None if isinstance(sources, Unavailable) else sources["stack"]
        return stack is not None and int(stack["shape_zyx"][0]) > 1

    def regions(self):
        """The zoom-in regions: per example role its FOV and footprint ``(y0, y1, x0, x1)`` µm, or the reason the
        footprint is unavailable; the reason of the example choice when there is none."""
        c = self.c
        if c.example_reason is not None:
            return c.example_reason
        rows = {r["fov_id"]: r for r in c.fovs}
        regions = []
        for example in c.examples:
            row = rows[example.fov_id]
            footprint = [row[k] for k in ("footprint_y0", "footprint_y1", "footprint_x0", "footprint_x1")]
            missing = first(*footprint)
            regions.append((example.role, example.fov_id,
                            missing if missing is not None else tuple(float(v) for v in footprint)))
        return regions

    # The rows of unavailable.csv ----------------------------------------------------------------------------------

    def part_rows(self):
        """The rows of ``unavailable.csv`` of the parts ``sample.*`` with no cell for what they lack."""
        c = self.c
        rows = []
        cells = self.cells_rows()
        if isinstance(cells, Unavailable):
            rows.append(("sample.per_cell", cells, ""))
        markers = self.markers_missing()
        if markers is not None:
            rows += [("sample.markers", markers, ""), ("sample.marker_zoom", markers, "")]
        else:
            regions = self.regions()
            if isinstance(regions, Unavailable):
                rows.append(("sample.marker_zoom", regions, ""))
            else:
                stack = self.has_stack()
                for role, fov, footprint in regions:
                    key = f"fov_id={fov};role={role}"
                    if isinstance(footprint, Unavailable):
                        rows.append(("sample.marker_zoom", footprint, key))
                    elif not stack:
                        rows.append(("sample.marker_zoom", Unavailable("no_stack", "the export has no fused stack "
                                                                       "of more than one plane"), f"{key};view=xz"))
        if not c.config.marker_pairs:
            rows.append(("sample.mecr", first(c.e_state, c.no_cells) or
                         Unavailable("no_markers", "the configuration has no marker_pairs"), ""))
        along = self.reads_along_z_rows()
        if isinstance(along, Unavailable):
            rows.append(("sample.reads_along_z", along, ""))
        return rows

    # The figures ------------------------------------------------------------------------------------------------

    def figures(self) -> SampleFigures:
        """The static figures of "Sample measurements" (``_figures``), each the reason of its part when it cannot be
        drawn."""
        from . import _figures
        config = self.c.config
        cells = self.cells_rows()
        if isinstance(cells, Unavailable):
            per_cell = cells
        else:
            shown = [([r["reads"] for r in cells], "reads", "reads per cell", "molecules", True),
                     ([r["genes"] for r in cells], "genes", "genes per cell", "genes", True)]
            for unit in SIZE_UNITS:
                values = [r["size"] for r in cells if r["size_unit"] == unit]
                if values:
                    shown.append((values, f"size:{unit}", "cell size", unit, unit == "voxels"))
            # Each figure is drawn for its slot: rows of three, a row of fewer in half columns (_page.grid).
            per_cell = [_figures.histogram_figure(values, name, title, unit, integer=integer,
                                                  slot=_figures.grid_slot(min(3, len(shown) - 3 * (k // 3))))
                        for k, (values, name, title, unit, integer) in enumerate(shown)]
        along = self.reads_along_z_rows()
        reads_figure = along if isinstance(along, Unavailable) else _figures.reads_along_z_figure(along,
                                                                                                    self.planes())
        molecules = self.molecules()
        if isinstance(molecules, Unavailable):
            groups = markers = molecules
        else:
            retained = molecules["assembly_status"].eq("retained").fillna(False).to_numpy(bool)
            genes = list((self.c.assembly or {}).get("genes") or [])
            groups = marker_groups(config, genes, molecules["gene_id"][retained].astype(object).to_numpy())
            markers = self.markers_missing() or self._marker_figures(groups, molecules, retained)
        return SampleFigures(per_cell, groups, markers, reads_figure)

    def _marker_figures(self, groups, molecules, retained):
        from ._figures import FusedView, marker_figure
        sources = self.fused_sources()
        regions = self.regions()
        view = FusedView.read(self.c.export_dir, sources,
                              [] if isinstance(regions, Unavailable) else
                              [r for r in regions if not isinstance(r[2], Unavailable)], molecules, retained)
        config = self.c.config
        return [marker_figure(group, view, molecules, retained, regions, width=config.figure_width_px,
                              quality=config.jpeg_quality) for group in groups]
