"""The consistency check of the two routes (docs/sample-segmentation-contract.md, "The consistency check").

:func:`compare_routes` reads an export folder of each route for the same sample and reports
how their cells and molecules correspond. Each route's counted cells are placed on the
segmentation grid of the route after fusion: a cell of that route is its territory as it
is; a counted cell of the route per FOV is its territory placed as the label export places
it, a grid pixel ``v`` showing the FOV pixel ``floor(c + ½)`` with ``c = (O + v ⊙ S −
origin) ⊘ s`` in Y and X inside the FOV's voxel-centre extent, a cell shown at ``v`` when
any of its planes shows it there, each cell by itself. The share of a pair is the duplicate
rule's measure, :func:`starfinder.assembly._duplicates.share`: the shared pixels as a share
of the smaller cell. The report decides nothing and changes no file; it writes its tables
only into a folder ``output`` outside both export folders.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._record import EXPORT_FILE, read_export

#: The six molecule categories, in the order they are decided (retention, then the cell, then the match).
CATEGORIES = ("same_cell", "different_cells", "cell_on_one_route", "no_cell_on_both", "retained_on_one_route",
              "excluded_on_both")
#: The width of a bin of the share distribution.
BIN = 0.05
N_BINS = 20
#: The margin of a FOV's voxel-centre extent, in voxels (the duplicate rule's).
EPS = 1e-9
MEASURE = "starfinder.assembly._duplicates.share"
PAIR_COLUMNS = {"per_fov_cell_namespace": "string", "per_fov_cell_id": "Int64", "per_fov_obs_name": "string",
                "after_fusion_cell_namespace": "string", "after_fusion_cell_id": "Int64",
                "after_fusion_obs_name": "string", "per_fov_size": "int64", "after_fusion_size": "int64",
                "shared": "int64", "share": "float64", "matched": "bool", "centroid_distance": "float64",
                "per_fov_fov_edge": "bool", "per_fov_conflict": "bool", "per_fov_duplicate_of": "string"}
COUNT_COLUMNS = {"per_fov_cell_namespace": "string", "per_fov_cell_id": "Int64",
                 "after_fusion_cell_namespace": "string", "after_fusion_cell_id": "Int64", "per_fov_whole": "int64",
                 "after_fusion_whole": "int64", "whole_difference": "int64", "gene_difference": "int64"}
MOLECULE_COLUMNS = {"fov_id": "string", "spot_namespace": "string", "spot_id": "string", "gene_id": "string",
                    "per_fov_assembly_status": "string", "per_fov_cell_namespace": "string",
                    "per_fov_cell_id": "Int64", "after_fusion_assembly_status": "string",
                    "after_fusion_cell_namespace": "string", "after_fusion_cell_id": "Int64", "category": "string",
                    "route": "string"}
ROUTE_NAMES = ("per_fov", "after_fusion")


@dataclass(frozen=True)
class RouteComparison:
    """The report of :func:`compare_routes`.

    ``summary``: the counted cells of each route (``cells``), the pairs with a
    share above 0 (``pairs``), the matched pairs (``matched_pairs``), the
    cells of each route in no matched pair (``cells_in_no_matched_pair``),
    the cells of the route after fusion that share at least the threshold
    with two or more cells of the route per FOV
    (``after_fusion_cells_sharing_with_several``) and the reverse
    (``per_fov_cells_sharing_with_several``), the share distribution in bins
    of 0.05 (``share_bins``, the last bin closed), the retained molecules of
    each route and of both (``retained_molecules``) and the molecules of each
    category (``molecule_categories``). ``pairs``: one row per pair with a
    share above 0, both cell keys and ``obs_name``s, both sizes and the shared
    pixels on the segmentation grid, the share, ``matched``, the distance of
    the two ownership centroids in Y and X (µm), the flags ``fov_edge`` and
    ``conflict`` and ``duplicate_of`` of the cell of the route per FOV.
    ``counts``: per matched pair both whole counts, their difference (after
    fusion minus per FOV) and the sum over genes of the absolute per-gene
    differences. ``molecules``: per molecule key, in the order of the route
    per FOV's molecule table, its assembly status and cell on each route, its
    category and the route a category ``*_on_one_route`` names. ``record``:
    the threshold, the measure, the two folders and the SHA-256 of every file
    read.
    """

    summary: Mapping[str, Any]
    pairs: pd.DataFrame
    counts: pd.DataFrame
    molecules: pd.DataFrame
    record: Mapping[str, Any]


# --- The checks of the two folders ----------------------------------------------------------------------------

def _threshold(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0 < value <= 1:
        raise ValueError(f"threshold must be a number in (0, 1]; got {value!r}")
    return float(value)


def _route_of(folder: Path) -> tuple[dict, str]:
    """The ``export.json`` of a folder and the route its ``assembly`` section records (``per_fov`` when an
    assembly saved before the route existed records none)."""
    record = read_export(folder)
    assembly = record.get("assembly")
    if not isinstance(assembly, dict):
        raise ValueError(f"{folder} holds no assembly: no {EXPORT_FILE} with an 'assembly' section")
    return record, assembly.get("route") or "per_fov"


def _inside(path: Path, folder: Path) -> bool:
    return path == folder or folder in path.parents


def _check_output(output, folders) -> Path | None:
    """The output folder, refused when it is one of the export folders, lies inside one or is an export folder."""
    if output is None:
        return None
    path = Path(output).resolve()
    for folder in folders:
        if _inside(path, folder):
            raise ValueError(f"output {path} lies inside the export folder {folder}; compare_routes writes into a "
                             "folder outside both export folders")
    for parent in (path, *path.parents):
        if (parent / EXPORT_FILE).is_file():
            raise ValueError(f"output {path} lies inside the export folder {parent}; compare_routes writes into a "
                             "folder outside every export folder")
    return path


def _keys(molecules) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([molecules["spot_namespace"].astype(object).to_numpy(),
                                      molecules["spot_id"].astype(object).to_numpy()])


def _check_same_sample(results, records):
    """ValueError naming the difference unless the two folders hold one sample, placement, gene list and molecule
    population."""
    a, b = (results[r] for r in ROUTE_NAMES)
    if a.sample != b.sample:
        raise ValueError(f"the two folders hold different samples: {a.sample!r} (per_fov), {b.sample!r} "
                         "(after_fusion)")
    for key in ("fovs_sha256", "links_sha256"):
        values = [((records[r]["assembly"].get("placement") or {}).get(key)) for r in ROUTE_NAMES]
        if values[0] is None or values[0] != values[1]:
            raise ValueError(f"the two folders hold different placements: placement {key} {values[0]} (per_fov), "
                             f"{values[1]} (after_fusion)")
    if tuple(a.genes) != tuple(b.genes):
        raise ValueError(f"the two folders hold different gene lists: {list(a.genes)} (per_fov), {list(b.genes)} "
                         "(after_fusion)")
    keys = {}
    for route in ROUTE_NAMES:
        index = _keys(results[route].molecules)
        if index.has_duplicates:
            raise ValueError(f"the {route} molecule table holds a (spot_namespace, spot_id) key twice: "
                             f"{list(index[index.duplicated()][:3])}")
        keys[route] = index
    only = {route: keys[route].difference(keys[other]) for route, other in zip(ROUTE_NAMES, ROUTE_NAMES[::-1])}
    if len(only["per_fov"]) or len(only["after_fusion"]):
        raise ValueError("the two folders hold different molecule populations (spot_namespace, spot_id): "
                         f"{len(only['per_fov'])} keys only in per_fov (first {list(only['per_fov'][:3])}), "
                         f"{len(only['after_fusion'])} only in after_fusion (first {list(only['after_fusion'][:3])})")


# --- The cells on the segmentation grid -----------------------------------------------------------------------

def _grid(section) -> dict:
    grid = ((section or {}).get("inputs") or {}).get("grid")
    if not isinstance(grid, dict):
        raise ValueError("the after_fusion folder's segmentation section records no segmentation grid")
    return {"shape": tuple(int(n) for n in grid["shape_zyx"]), "spacing": tuple(float(v) for v in grid["spacing_zyx"]),
            "origin": tuple(float(v) for v in grid["origin_zyx"])}


def _indices(grid_origin, grid_spacing, n_grid, origin, spacing, n):
    """The grid indices inside a FOV's voxel-centre extent on one axis and the FOV index each shows,
    ``floor(c + ½)`` with ``c = (O + v·S − origin) / s``."""
    v = np.arange(n_grid, dtype=np.int64)
    c = (grid_origin + v * grid_spacing - origin) / spacing
    inside = (c >= -EPS) & (c <= n - 1 + EPS)
    return v[inside], np.floor(c[inside] + 0.5).astype(np.int64).clip(0, n - 1)


def _cell_key(namespace, cell_id) -> tuple[str, int]:
    return str(namespace), int(cell_id)


def _fov_grid(entry) -> tuple[tuple, tuple, tuple]:
    """``(shape, spacing, origin)`` of a FOV of the assembly record (its grid and its placement origin)."""
    grid = entry.get("grid")
    if not isinstance(grid, dict) or entry.get("origin_zyx") is None:
        raise ValueError(f"the per_fov assembly record holds no grid or origin of FOV {entry.get('fov_id')!r}")
    return (tuple(int(n) for n in grid["shape_zyx"]), tuple(float(v) for v in grid["metadata"]["spacing_zyx"]),
            tuple(float(v) for v in entry["origin_zyx"]))


def _per_fov_territory(entry):
    """The territories of a FOV of the route per FOV (the ones assign sampled), their SHA-256 checked first."""
    from ._labels import _read_territory, _territory_source
    path, sha256, source = _territory_source(entry)
    if path is None:
        raise ValueError(f"the per_fov assembly record names no territory file of FOV {entry.get('fov_id')!r} "
                         f"(its assignment was given in memory)")
    return np.asarray(_read_territory(path, sha256)), {"fov_id": entry.get("fov_id"), "path": str(path),
                                                       "sha256": sha256, "source": source}


def _place(per_fov, record, grid, fused, files):
    """Each counted cell of the route per FOV on the segmentation grid: its pixels, and the pixels it shares with
    each cell of the route after fusion. Returns ``(sizes, shared)`` keyed by the cell keys."""
    counted = per_fov.counted_cells()
    entries = {entry["fov_id"]: entry for entry in record.get("fovs") or []}
    _, n_y, n_x = grid["shape"]
    flat = fused.reshape(-1)
    sizes, shared = {}, Counter()
    for fov in dict.fromkeys(counted["fov_id"].astype(object).tolist()):
        rows = counted[counted["fov_id"].astype(object).eq(fov).to_numpy(dtype=bool)]
        namespaces = dict(zip(rows["cell_id"].astype("int64").tolist(), rows["cell_namespace"].astype(object)))
        ids = np.array(sorted(namespaces), np.int64)
        for cell in ids.tolist():
            sizes[_cell_key(namespaces[cell], cell)] = 0
        if fov not in entries:
            raise ValueError(f"the per_fov assembly record holds no entry of FOV {fov!r}")
        shape, spacing, origin = _fov_grid(entries[fov])
        territory, entry = _per_fov_territory(entries[fov])
        files.append(entry)
        if territory.ndim == 2:
            territory = territory[np.newaxis]
        gy, fy = _indices(grid["origin"][1], grid["spacing"][1], n_y, origin[1], spacing[1], shape[1])
        gx, fx = _indices(grid["origin"][2], grid["spacing"][2], n_x, origin[2], spacing[2], shape[2])
        if not len(gy) or not len(gx):
            continue
        pixel = (gy[:, None] * n_x + gx[None, :]).astype(np.uint64)
        found = []
        for plane in territory:
            values = plane[np.ix_(fy, fx)].astype(np.int64)
            hit = np.isin(values, ids) & (values > 0)
            # The pixel and the cell ID (uint32) in one unsigned 64-bit key, so that every ID comes back whole.
            found.append((pixel[hit] << np.uint64(32)) | values[hit].astype(np.uint64))
        keys = np.unique(np.concatenate(found)) if found else np.array([], np.uint64)
        cells = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
        values, n = np.unique(cells, return_counts=True)
        for cell, count in zip(values.tolist(), n.tolist()):
            sizes[_cell_key(namespaces[cell], cell)] = count
        other = flat[(keys >> np.uint64(32)).astype(np.int64)].astype(np.uint64)
        both = other > 0
        pairs, n = np.unique((cells[both].astype(np.uint64) << np.uint64(32)) | other[both], return_counts=True)
        for key, count in zip(pairs.tolist(), n.tolist()):
            cell = key >> 32
            shared[(_cell_key(namespaces[cell], cell), key & 0xFFFFFFFF)] += count
    return sizes, shared


def _fused_cells(after, territory):
    """The counted cells of the route after fusion and their pixels on the segmentation grid."""
    counted = after.counted_cells()
    values, n = np.unique(territory[territory > 0], return_counts=True)
    pixels = dict(zip(values.astype(np.int64).tolist(), n.tolist()))
    by_id = {}
    for namespace, cell in zip(counted["cell_namespace"].astype(object), counted["cell_id"].astype("int64").tolist()):
        by_id[cell] = _cell_key(namespace, cell)
    return by_id, {key: pixels.get(cell, 0) for cell, key in by_id.items()}


# --- The report -----------------------------------------------------------------------------------------------

def _match(pairs, threshold):
    """The matched pairs: share at least ``threshold`` and each cell the other's partner of highest share, ties by
    the smaller cell key."""
    best = {}
    for (a, b), (_, value) in pairs.items():
        for cell, partner in ((("per_fov", a), b), (("after_fusion", b), a)):
            held = best.get(cell)
            if held is None or value > held[0] or (value == held[0] and partner < held[1]):
                best[cell] = (value, partner)
    return {(a, b) for (a, b), (_, value) in pairs.items()
            if value >= threshold and best[("per_fov", a)][1] == b and best[("after_fusion", b)][1] == a}


def _cell_rows(result) -> dict:
    cells = result.cells
    keys = [_cell_key(ns, c) for ns, c in zip(cells["cell_namespace"].astype(object),
                                                   cells["cell_id"].astype("int64").tolist())]
    return dict(zip(keys, cells.to_dict("records")))


def _flags(value) -> set:
    return set() if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)) \
        else set(str(value).split(";"))


def _null(value):
    return None if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)) else value


def _distance(a, b) -> float:
    d = [float(a[f"sample_ownership_centroid_{x}"]) - float(b[f"sample_ownership_centroid_{x}"]) for x in "yx"]
    return math.hypot(*d)


def _frame(rows, columns) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(columns))
    return frame.astype(columns) if len(frame) else pd.DataFrame({c: pd.Series(dtype=t) for c, t in columns.items()})


def _whole(result) -> dict:
    """Per counted cell key ``{gene: count}`` of its whole counts."""
    counts = result.counts
    whole = counts[counts["compartment"].astype(object).eq("whole").to_numpy(dtype=bool)]
    out = {}
    for ns, cell, gene, count in zip(whole["cell_namespace"].astype(object), whole["cell_id"].astype("int64"),
                                     whole["gene_id"].astype(object), whole["count"].astype("int64")):
        genes = out.setdefault(_cell_key(ns, cell), Counter())
        genes[str(gene)] += int(count)
    return out


def _category(a_retained, b_retained, a_cell, b_cell, matched):
    """The category of a molecule and the route it names: retention, then the cell, then the match."""
    if a_retained and b_retained:
        if a_cell is not None and b_cell is not None:
            return ("same_cell" if (a_cell, b_cell) in matched else "different_cells"), None
        if a_cell is not None or b_cell is not None:
            return "cell_on_one_route", "per_fov" if a_cell is not None else "after_fusion"
        return "no_cell_on_both", None
    if a_retained or b_retained:
        return "retained_on_one_route", "per_fov" if a_retained else "after_fusion"
    return "excluded_on_both", None


def _molecule_cell(namespace, cell_id):
    cell_id = _null(cell_id)
    return None if cell_id is None else _cell_key(namespace, cell_id)


def _molecules(per_fov, after, matched):
    a = per_fov.molecules.reset_index(drop=True)
    b = after.molecules.reset_index(drop=True)
    position = pd.Series(np.arange(len(b)), index=_keys(b))
    order = position.reindex(_keys(a)).to_numpy()
    b = b.iloc[order].reset_index(drop=True)
    rows = []
    for x, y in zip(a.to_dict("records"), b.to_dict("records")):
        a_cell = _molecule_cell(x["cell_namespace"], x["cell_id"])
        b_cell = _molecule_cell(y["cell_namespace"], y["cell_id"])
        category, route = _category(x["assembly_status"] == "retained", y["assembly_status"] == "retained", a_cell,
                                    b_cell, matched)
        rows.append({"fov_id": x["fov_id"], "spot_namespace": x["spot_namespace"], "spot_id": x["spot_id"],
                     "gene_id": x["gene_id"], "per_fov_assembly_status": _null(x["assembly_status"]),
                     "per_fov_cell_namespace": None if a_cell is None else a_cell[0],
                     "per_fov_cell_id": None if a_cell is None else a_cell[1],
                     "after_fusion_assembly_status": _null(y["assembly_status"]),
                     "after_fusion_cell_namespace": None if b_cell is None else b_cell[0],
                     "after_fusion_cell_id": None if b_cell is None else b_cell[1],
                     "category": category, "route": route})
    return _frame(rows, MOLECULE_COLUMNS)


def _report(per_fov, after, sizes, shared, threshold):
    """The tables and the summary of the report."""
    from starfinder.assembly._duplicates import share
    pairs = {}
    for (a, b), voxels in shared.items():
        key = (("per_fov", a), ("after_fusion", b))
        # The duplicate rule's measure: the shared pixels as a share of the smaller cell.
        pairs[(a, b)] = (voxels, share(sizes, key, voxels))
    matched = _match(pairs, threshold)
    cells = {"per_fov": _cell_rows(per_fov), "after_fusion": _cell_rows(after)}
    rows = []
    for a, b in sorted(pairs):
        voxels, value = pairs[(a, b)]
        x, y = cells["per_fov"][a], cells["after_fusion"][b]
        flags = _flags(x["assembly_flags"])
        rows.append({"per_fov_cell_namespace": a[0], "per_fov_cell_id": a[1], "per_fov_obs_name": x["obs_name"],
                     "after_fusion_cell_namespace": b[0], "after_fusion_cell_id": b[1],
                     "after_fusion_obs_name": y["obs_name"], "per_fov_size": sizes[("per_fov", a)],
                     "after_fusion_size": sizes[("after_fusion", b)], "shared": voxels, "share": value,
                     "matched": (a, b) in matched, "centroid_distance": _distance(x, y),
                     "per_fov_fov_edge": "fov_edge" in flags, "per_fov_conflict": "conflict" in flags,
                     "per_fov_duplicate_of": _null(x["duplicate_of"])})
    pair_table = _frame(rows, PAIR_COLUMNS)
    whole = {"per_fov": _whole(per_fov), "after_fusion": _whole(after)}
    rows = []
    for a, b in sorted(matched):
        x, y = whole["per_fov"].get(a, Counter()), whole["after_fusion"].get(b, Counter())
        rows.append({"per_fov_cell_namespace": a[0], "per_fov_cell_id": a[1], "after_fusion_cell_namespace": b[0],
                     "after_fusion_cell_id": b[1], "per_fov_whole": sum(x.values()),
                     "after_fusion_whole": sum(y.values()), "whole_difference": sum(y.values()) - sum(x.values()),
                     "gene_difference": sum(abs(y[g] - x[g]) for g in set(x) | set(y))})
    count_table = _frame(rows, COUNT_COLUMNS)
    molecules = _molecules(per_fov, after, matched)
    counted = {route: {k for (r, k) in sizes if r == route} for route in ROUTE_NAMES}
    in_pair = {"per_fov": {a for a, _ in matched}, "after_fusion": {b for _, b in matched}}
    several = {route: Counter() for route in ROUTE_NAMES}
    for (a, b), (_, value) in pairs.items():
        if value >= threshold:
            several["after_fusion"][b] += 1
            several["per_fov"][a] += 1
    bins = [0] * N_BINS
    for (a, b), (voxels, _) in pairs.items():
        # Exact in integers: bin floor(share / 0.05), the last bin closed at 1.
        smaller = min(sizes[("per_fov", a)], sizes[("after_fusion", b)])
        bins[min(N_BINS - 1, (N_BINS * voxels) // smaller)] += 1
    retained = {route: molecules[f"{route}_assembly_status"].eq("retained").fillna(False).to_numpy(dtype=bool)
                for route in ROUTE_NAMES}
    categories = molecules["category"].astype(object).value_counts().to_dict()
    summary = {"cells": {route: len(counted[route]) for route in ROUTE_NAMES},
               "pairs": len(pairs), "matched_pairs": len(matched),
               "cells_in_no_matched_pair": {route: len(counted[route] - in_pair[route]) for route in ROUTE_NAMES},
               "after_fusion_cells_sharing_with_several": sum(1 for n in several["after_fusion"].values() if n >= 2),
               "per_fov_cells_sharing_with_several": sum(1 for n in several["per_fov"].values() if n >= 2),
               "share_bins": [{"low": round(i * BIN, 2), "high": round((i + 1) * BIN, 2), "pairs": n}
                              for i, n in enumerate(bins)],
               "retained_molecules": {"per_fov": int(retained["per_fov"].sum()),
                                      "after_fusion": int(retained["after_fusion"].sum()),
                                      "both": int((retained["per_fov"] & retained["after_fusion"]).sum())},
               "molecule_categories": {c: int(categories.get(c, 0)) for c in CATEGORIES}}
    return summary, pair_table, count_table, molecules


def _write(output: Path, report: RouteComparison) -> None:
    from starfinder.io._checkpoint import _write_table, write_json
    output.mkdir(parents=True, exist_ok=True)
    for name in ("pairs", "counts", "molecules"):
        _write_table(getattr(report, name), output, name, "csv")
    write_json({"summary": dict(report.summary), "record": dict(report.record)}, output / "summary.json")


def compare_routes(per_fov, after_fusion, *, threshold: float = 0.5, output=None) -> RouteComparison:
    """Report how the cells and molecules of the two routes of one sample correspond.

    Reads the export folder of the route per FOV (``per_fov``) and of the
    route after fusion (``after_fusion``) of the same sample. Both route's
    counted cells are placed on the segmentation grid of the route after
    fusion (the route per FOV's territories as the label export places them,
    each cell by itself) and the shared pixels of every pair are counted. The
    share of a pair is its shared pixels divided by the pixels of the smaller
    cell, the measure of the duplicate rule; a pair is matched when its share
    is at least ``threshold`` and each cell is the other's partner of highest
    share (ties by the smaller cell key ``(cell_namespace, cell_id)``). Each
    molecule key gets exactly one of the six categories ``same_cell``,
    ``different_cells``, ``cell_on_one_route``, ``no_cell_on_both``,
    ``retained_on_one_route`` and ``excluded_on_both``. The report decides
    nothing: no file of either folder changes, and the threshold only names
    the pairs counted as matched.

    Parameters
    ----------
    per_fov, after_fusion : str or Path
        The export folders of the route per FOV and of the route after fusion.
    threshold : float
        The share at which a pair of mutual best partners is matched, in
        (0, 1]; 0.5 by default, the duplicate rule's.
    output : str or Path, optional
        A folder outside both export folders (and outside every export
        folder) that receives ``pairs.csv``, ``counts.csv``,
        ``molecules.csv`` and ``summary.json``. Nothing is written without it.

    Returns
    -------
    RouteComparison
        The summary, the pairs, the counts per matched pair, the molecules and
        the record.

    Raises
    ------
    ValueError
        Before anything is written: a threshold outside (0, 1]; a folder
        without an assembly; two folders of the same route, or a folder of the
        other route; two samples, placements (the placement SHA-256 of both
        ``assembly`` sections), gene lists or molecule populations (the
        ``(spot_namespace, spot_id)`` keys); an ``after_fusion`` folder without
        a sample segmentation; ``output`` inside an export folder; a file whose
        SHA-256 differs from its record.
    """
    from starfinder.assembly import load_assembly
    from starfinder.segmentation._persist import file_sha256

    from ._fuse import _software
    from ._labels import read_sample_territory, sample_territory_source
    threshold = _threshold(threshold)
    folders = {"per_fov": Path(per_fov).resolve(), "after_fusion": Path(after_fusion).resolve()}
    records, routes = {}, {}
    for name, folder in folders.items():
        records[name], routes[name] = _route_of(folder)
    if routes["per_fov"] == routes["after_fusion"]:
        raise ValueError(f"the two folders are of the same route {routes['per_fov']!r}: {folders['per_fov']} and "
                         f"{folders['after_fusion']}; compare_routes compares a folder of each route")
    for name, folder in folders.items():
        if routes[name] != name:
            raise ValueError(f"{name} names {folder}, a folder of the route {routes[name]!r}")
    output = _check_output(output, folders.values())
    results = {name: load_assembly(folder) for name, folder in folders.items()}
    _check_same_sample(results, records)
    after = results["after_fusion"]
    if after.record.get("mode") != "cells":
        raise ValueError(f"the after_fusion folder {folders['after_fusion']} holds no sample segmentation (assembly "
                         f"mode {after.record.get('mode')!r}), so it has no cells and no segmentation grid")
    grid = _grid(records["after_fusion"].get("segmentation"))
    path, sha256, run = sample_territory_source(after)
    territory = read_sample_territory(folders["after_fusion"] / path, sha256)
    if territory.shape != grid["shape"]:
        raise ValueError(f"the territories {path} have shape {territory.shape}, the segmentation grid "
                         f"{grid['shape']}")
    files = [{"fov_id": None, "path": str(folders["after_fusion"] / path), "sha256": sha256, "run": run}]
    by_id, fused_sizes = _fused_cells(after, territory[0])
    sizes, shared = _place(results["per_fov"], results["per_fov"].record, grid, territory[0], files)
    sizes = {**{("per_fov", k): v for k, v in sizes.items()},
             **{("after_fusion", k): v for k, v in fused_sizes.items()}}
    # Pixels of a placed cell under a fused label that is no counted cell are not a pair.
    shared = {(a, by_id[b]): n for (a, b), n in shared.items() if b in by_id}
    summary, pairs, counts, molecules = _report(results["per_fov"], after, sizes, shared, threshold)
    record = {"threshold": threshold, "measure": MEASURE, "sample": after.sample,
              "placement": {k: records["after_fusion"]["assembly"]["placement"].get(k)
                            for k in ("fovs_sha256", "links_sha256")},
              "genes": list(after.genes), "grid": {k: list(v) for k, v in grid.items()},
              "folders": {name: {"path": str(folder),
                                 "export_sha256": file_sha256(folder / EXPORT_FILE),
                                 "files": {k: v["sha256"] for k, v in (results[name].record.get("files") or {}).items()}}
                          for name, folder in folders.items()},
              "territories": files, "software": _software()}
    report = RouteComparison(summary, pairs, counts, molecules, record)
    if output is not None:
        _write(output, report)
    return report
