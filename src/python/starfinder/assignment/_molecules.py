"""The molecule input of assign: a validated table of reads with zero-based positions and genes."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COLUMNS = ("spot_namespace", "spot_id", "z", "y", "x", "gene_id")
POPULATIONS = ("final", "called")
_KEYS = ["spot_namespace", "spot_id"]
_STRING = ("spot_namespace", "spot_id", "gene_id")
_COORDINATES = ("z", "y", "x")


def _fov_identity(spot_namespace):
    """The FOV identity list [dataset_id, sample_id, fov_id, subtile_id] of a spot namespace."""
    try:
        value = json.loads(spot_namespace)
    except (TypeError, ValueError):
        value = None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("spot_namespace must be the JSON list [dataset_id, sample_id, fov_id, subtile_id]; "
                         f"got {spot_namespace!r}")
    return value


def molecules_sha256(table: pd.DataFrame) -> str:
    """SHA-256 of the six molecule columns in canonical form, independent of row order.

    Rows are sorted by ``(spot_namespace, spot_id)``; then, per column, its name,
    its dtype and its values (float64 bytes for the coordinates, JSON text for the
    strings) are hashed.
    """
    ordered = table.sort_values(_KEYS, kind="mergesort")
    digest = hashlib.sha256()
    for column in COLUMNS:
        values = ordered[column]
        digest.update(f"{column}|{values.dtype}|".encode())
        if column in _COORDINATES:
            digest.update(np.ascontiguousarray(values.to_numpy(np.float64)).tobytes())
        else:
            digest.update(json.dumps(values.astype(object).tolist()).encode())
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class MoleculeTable:
    """The molecules of one FOV that assign places in cells.

    ``table`` has exactly the columns ``spot_namespace``, ``spot_id`` and
    ``gene_id`` (pandas string) and ``z``, ``y``, ``x`` (float64, zero-based voxel
    coordinates of the molecule run's reference grid), one row per molecule.
    ``genes`` is the ordered gene list of the count matrix (``Codebook.genes``,
    first-appearance order); ``population`` is ``final`` (the accepted reads) or
    ``called`` (the reads whose call is ``assigned``); ``source`` records where the
    molecules come from and holds ``spot_namespace``, the FOV's identity list
    ``[dataset_id, sample_id, fov_id, subtile_id]`` as JSON, which every row
    shares. ``sha256``, set at construction, is the SHA-256 of the six columns in
    canonical form: rows sorted by ``(spot_namespace, spot_id)``, then each column's
    name, dtype and values, so it does not depend on row order. Build it with
    :func:`molecule_table` or :func:`molecule_table_from_csv`.

    Raises
    ------
    TypeError
        table is not a DataFrame, or source is not a mapping.
    ValueError
        Other columns or dtypes, a null or repeated key, a row of another FOV, a
        non-finite coordinate (naming the first row), a null gene, a gene outside
        ``genes`` (naming the unknown genes), an empty or repeated gene list, or
        an unknown population.
    """

    table: pd.DataFrame
    genes: tuple[str, ...]
    population: str
    source: Mapping[str, Any]
    sha256: str = field(init=False)

    def __post_init__(self):
        table = self.table
        if not isinstance(table, pd.DataFrame):
            raise TypeError("table must be a pandas DataFrame")
        if tuple(table.columns) != COLUMNS:
            raise ValueError(f"the molecule table needs exactly the columns {list(COLUMNS)}; got {list(table.columns)}")
        for column in _STRING:
            if not isinstance(table[column].dtype, pd.StringDtype):
                raise ValueError(f"column {column} must have pandas string dtype")
        for column in _COORDINATES:
            if table[column].dtype != np.dtype("float64"):
                raise ValueError(f"column {column} must be float64")
        genes = self.genes
        if (not isinstance(genes, tuple) or not genes or not all(isinstance(g, str) and g for g in genes)
                or len(set(genes)) != len(genes)):
            raise ValueError("genes must be a nonempty tuple of distinct gene names")
        if self.population not in POPULATIONS:
            raise ValueError(f"population must be final or called; got {self.population!r}")
        if not isinstance(self.source, Mapping):
            raise TypeError("source must be a mapping")
        namespace = self.source.get("spot_namespace")
        _fov_identity(namespace)
        if table[_KEYS].isna().any().any() or (table.spot_id.str.len() == 0).any():
            raise ValueError("spot_namespace and spot_id must be present and nonempty")
        if table.duplicated(_KEYS).any():
            raise ValueError("molecule keys (spot_namespace, spot_id) must be unique")
        if not table.spot_namespace.eq(namespace).all():
            raise ValueError(f"every molecule must have the spot_namespace {namespace!r} of one FOV")
        finite = np.isfinite(table[list(_COORDINATES)].to_numpy(np.float64)).all(axis=1)
        if not finite.all():
            first = int(np.flatnonzero(~finite)[0])
            raise ValueError(f"molecule coordinates must be finite; row {first} "
                             f"({table.spot_id.iloc[first]}) is not")
        if table.gene_id.isna().any():
            first = int(np.flatnonzero(table.gene_id.isna().to_numpy())[0])
            raise ValueError(f"every molecule needs a gene_id; row {first} has none")
        unknown = sorted(set(table.gene_id) - set(genes))
        if unknown:
            raise ValueError(f"genes outside the gene list: {unknown}")
        object.__setattr__(self, "sha256", molecules_sha256(table))

    def __len__(self):
        return len(self.table)

    @property
    def spot_namespace(self) -> str:
        """The FOV's spot namespace, ``source["spot_namespace"]``."""
        return self.source["spot_namespace"]

    def __repr__(self):
        return f"MoleculeTable: {len(self.table)} molecules of {len(self.genes)} genes ({self.population})"


def _frame(spot_namespace, spot_id, z, y, x, gene_id):
    n = len(spot_id)
    return pd.DataFrame({
        "spot_namespace": pd.array([spot_namespace] * n, dtype="string"),
        "spot_id": pd.array(list(spot_id), dtype="string"),
        "z": np.asarray(z, np.float64), "y": np.asarray(y, np.float64), "x": np.asarray(x, np.float64),
        "gene_id": pd.array(list(gene_id), dtype="string"),
    })


def _genes(genes):
    if isinstance(genes, str):
        raise TypeError("genes must be a sequence of gene names, not one string")
    return tuple(genes)


def molecule_table(detection, reads, *, genes, population: str = "final") -> MoleculeTable:
    """The molecules of a detection and a read result, joined by ``(spot_namespace, spot_id)``.

    The join and its checks are those of :func:`starfinder.io.export_spots`.
    ``population="final"`` takes the accepted reads of a
    :class:`~starfinder.barcode.ReadFilteringResult` (the goodSpots population);
    ``"called"`` takes the reads whose ``call_status`` is ``assigned`` from a
    :class:`~starfinder.barcode.BarcodeDecodingResult` or a filtering result,
    before filtering. Positions are the detection's zero-based ``z, y, x``.

    Parameters
    ----------
    detection : SpotFindingResult
        The spots and their positions.
    reads : BarcodeDecodingResult or ReadFilteringResult
        The reads of those spots.
    genes : sequence of str
        The ordered gene list (``Codebook.genes``).
    population : str
        ``final`` (default) or ``called``.

    Raises
    ------
    TypeError
        detection or reads has another type.
    ValueError
        A population other than final or called, final without a filtering
        result, the join's errors (missing, foreign or repeated keys, another
        namespace), and every check of :class:`MoleculeTable`.
    """
    from starfinder.barcode import ReadFilteringResult
    from starfinder.io.spots import _join_spots

    if population not in POPULATIONS:
        raise ValueError(f"population must be final or called; got {population!r}")
    if population == "final" and not isinstance(reads, ReadFilteringResult):
        raise ValueError("population 'final' takes the accepted reads of a ReadFilteringResult")
    joined = _join_spots(detection, reads, accepted_only=population == "final")
    if population == "called":
        joined = joined.loc[joined.call_status.eq("assigned").to_numpy(dtype=bool)]
    table = _frame(detection.spot_namespace, joined.spot_id, joined.z, joined.y, joined.x, joined.gene_id)
    filter_config = asdict(reads.config) if isinstance(reads, ReadFilteringResult) else None
    source = {"spot_namespace": detection.spot_namespace, "detection": type(detection).__name__,
              "reads": type(reads).__name__, "filter_config": json.loads(json.dumps(filter_config, default=str)),
              "population": population}
    return MoleculeTable(table, _genes(genes), population, source)


def molecule_table_from_csv(path: Path | str, *, spot_namespace: str, genes) -> MoleculeTable:
    """The molecules of a legacy one-based ``x, y, z, gene`` CSV (MATLAB or ``export_spots``).

    Integer and float coordinates are both read; 1 is subtracted from each, and
    the row with zero-based index ``i`` gets the identity ``spot_id = "csv:<i>"``.
    The population is ``final`` (a goodSpots file); the source records the file
    and its SHA-256. It exists for the workflow adapter.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    ValueError
        A column of x, y, z, gene is missing, a coordinate is not numeric, and
        every check of :class:`MoleculeTable`.
    """
    from starfinder.dataset._run_record import _sha256

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"molecule CSV not found: {path}")
    frame = pd.read_csv(path, dtype={"gene": str}, keep_default_na=False, na_values=[""])
    missing = [c for c in ("x", "y", "z", "gene") if c not in frame]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}")
    coordinates = {}
    for axis in ("z", "y", "x"):
        values = pd.to_numeric(frame[axis], errors="coerce")
        if values.isna().any():
            bad = int(np.flatnonzero(values.isna().to_numpy())[0])
            raise ValueError(f"{path}: column {axis} is not a finite number in row {bad}")
        coordinates[axis] = values.to_numpy(np.float64) - 1
    table = _frame(spot_namespace, [f"csv:{i}" for i in range(len(frame))], coordinates["z"], coordinates["y"],
                   coordinates["x"], frame["gene"].astype(object).where(frame["gene"].notna(), None))
    source = {"spot_namespace": spot_namespace, "csv": str(path), "sha256": _sha256(path),
              "coordinates": "one-based"}
    return MoleculeTable(table, _genes(genes), "final", source)
