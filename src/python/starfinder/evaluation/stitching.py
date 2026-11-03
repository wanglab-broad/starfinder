"""Pure placement and same-gene overlap metrics in an explicit physical frame."""
import numpy as np
import pandas as pd

from starfinder.image import ImageMetadata, IncompatibleGeometryError

from ._result import EvaluationResult, _geometry, _result, _threshold
from .matching import match_points

__all__ = ["gauge_aligned_origin_error", "overlap_molecules"]


def _physical_geometry(reference, observed, units):
    if not isinstance(reference, ImageMetadata) or not isinstance(observed, ImageMetadata):
        raise IncompatibleGeometryError("physical evaluation requires both grids")
    if units == "voxel":
        raise IncompatibleGeometryError("stitching metrics require physical coordinate units")
    _geometry(reference, observed, units)


def _origin_table(table):
    columns = ["origin_z", "origin_y", "origin_x"]
    if not {"fov_id", *columns} <= set(table.columns):
        raise ValueError("placements require fov_id and origin_z, origin_y, origin_x")
    if not table.fov_id.is_unique or any(not isinstance(v, str) or not v for v in table.fov_id):
        raise ValueError("fov_id must contain unique nonempty strings")
    return table.set_index("fov_id")[columns].astype(float)


def gauge_aligned_origin_error(reference: pd.DataFrame, observed: pd.DataFrame, *, anchor: str,
                               reference_metadata: ImageMetadata, observed_metadata: ImageMetadata,
                               units: str) -> EvaluationResult:
    """Remove one anchor's translation before comparing physical origins (M62).

    Parameters
    ----------
    reference : pandas.DataFrame
        Truth origins with unique ``fov_id`` and finite ``origin_z, origin_y,
        origin_x`` columns, already in the supplied physical frame and unit.
    observed : pandas.DataFrame
        Estimated origins with the same columns. Absent rows or NaN origins
        are listed as missing placements and excluded from numerical summaries.
    anchor : str
        FOV present with finite origins in both populations. Its estimated
        minus truth offset is subtracted from every estimated origin.
    reference_metadata : ImageMetadata
        Complete calibrated common grid and frame for the truth coordinates.
    observed_metadata : ImageMetadata
        Equal calibrated grid and frame for the estimated coordinates.
    units : str
        Physical coordinate unit equal to both metadata spatial units.

    Returns
    -------
    EvaluationResult
        Median and maximum Euclidean origin error in the supplied unit,
        per-FOV errors, the removed offset, missing and unexpected placements.
        Missing placements produce ``missing`` status even when the remaining
        origins have defined errors; the summaries include the anchor.
    """
    _physical_geometry(reference_metadata, observed_metadata, units)
    ref, obs = _origin_table(reference), _origin_table(observed)
    if not np.isfinite(ref.to_numpy()).all() or np.isinf(obs.to_numpy()).any():
        raise ValueError("reference origins must be finite; observed origins may only be missing or finite")
    if anchor not in ref.index or anchor not in obs.index or not np.isfinite(obs.loc[anchor]).all():
        raise ValueError("anchor requires a finite origin in both placements")
    offset = obs.loc[anchor].to_numpy() - ref.loc[anchor].to_numpy()
    missing, rows = [], []
    for fov in ref.index:
        if fov not in obs.index or not np.isfinite(obs.loc[fov]).all():
            missing.append(fov)
        else:
            delta = obs.loc[fov].to_numpy() - offset - ref.loc[fov].to_numpy()
            rows.append({"fov_id": fov, "error": float(np.linalg.norm(delta)), "difference_zyx": delta.tolist()})
    errors = [r["error"] for r in rows]
    values = {"median_error": float(np.median(errors)), "max_error": float(np.max(errors))}
    return _result(values, {k: units for k in values},
                   {"total_reference": len(ref), "total_observed": len(obs), "evaluated": len(rows),
                    "missing_placements": len(missing)},
                   {"anchor": anchor, "gauge_offset_zyx": offset.tolist(), "units": units,
                    "frame_id": reference_metadata.frame_id, "gauge": "anchor translation only"},
                   details={"origins": rows, "missing_placements": missing,
                            "unexpected_placements": [f for f in obs.index if f not in ref.index]},
                   reasons={"placements": "reference FOV placements missing"} if missing else {},
                   status="missing" if missing else None)


def overlap_molecules(reference: pd.DataFrame, observed: pd.DataFrame, *, bounds_zyx,
                      threshold: float, reference_metadata: ImageMetadata,
                      observed_metadata: ImageMetadata, units: str) -> EvaluationResult:
    """Compare same-gene molecules inside a common physical overlap (M68).

    Parameters
    ----------
    reference : pandas.DataFrame
        Reference molecules with finite ``z, y, x`` coordinates already in the
        common physical frame and ``gene_id``. Missing genes never match.
    observed : pandas.DataFrame
        Observed molecules in that same physical frame, unit and table format.
    bounds_zyx : array_like
        Shape (2, 3) physical lower and upper bounds for the overlap population.
        Each axis uses the half-open interval [lower, upper). Equal bounds on
        any axis denote no overlap; reversed bounds are invalid.
    threshold : float
        Nonnegative inclusive Euclidean matching radius in the supplied unit.
    reference_metadata : ImageMetadata
        Complete calibrated common grid and frame for reference coordinates.
    observed_metadata : ImageMetadata
        Equal calibrated grid and frame for observed coordinates.
    units : str
        Physical coordinate unit equal to both metadata spatial units.

    Returns
    -------
    EvaluationResult
        Median and maximum matched distance and both directional match rates
        over all molecules inside the bounds, including missing-gene rows.
        Matching calls ``match_points`` with same-gene greedy one-to-one pairs;
        ties follow original row order. Pair indices are positional in the
        supplied tables. Empty denominators and no matches are undefined with
        reasons. Excluded and missing-gene populations remain counted.
    """
    _physical_geometry(reference_metadata, observed_metadata, units)
    _threshold(threshold)
    bounds = np.asarray(bounds_zyx, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
        raise ValueError("bounds_zyx must be finite (2, 3) lower/upper bounds without reversed axes")
    populations, positions = [], []
    for table in (reference, observed):
        if not {"z", "y", "x", "gene_id"} <= set(table.columns):
            raise ValueError("molecules require z, y, x and gene_id columns")
        points = table[["z", "y", "x"]].to_numpy(dtype=float)
        if not np.isfinite(points).all():
            raise ValueError("molecule coordinates must be finite")
        inside = np.all((points >= bounds[0]) & (points < bounds[1]), axis=1)
        positions.append(np.flatnonzero(inside))
        populations.append(table.iloc[positions[-1]].reset_index(drop=True))
    ref, obs = populations
    pairs = []
    for gene, part in ref.groupby("gene_id", sort=False):
        other = obs[obs.gene_id.eq(gene).fillna(False)]
        matches = match_points(part[["z", "y", "x"]].to_numpy(float),
                               other[["z", "y", "x"]].to_numpy(float),
                               policy="greedy", threshold=threshold, boundary="inclusive", units=units,
                               reference_metadata=reference_metadata, observed_metadata=observed_metadata)
        for i, j, distance in matches.details["matched_pairs"]:
            pairs.append((int(positions[0][part.index[i]]), int(positions[1][other.index[j]]), distance))
    pairs.sort(key=lambda p: (p[2], p[0], p[1]))
    distances = [p[2] for p in pairs]
    values = {"reference_match_rate": len(pairs) / len(ref) if len(ref) else None,
              "observed_match_rate": len(pairs) / len(obs) if len(obs) else None,
              "median_distance": float(np.median(distances)) if distances else None,
              "max_distance": float(np.max(distances)) if distances else None}
    counts = {"total_reference": len(reference), "total_observed": len(observed),
              "eligible_reference": len(ref), "eligible_observed": len(obs), "matched": len(pairs),
              "excluded_reference": len(reference) - len(ref), "excluded_observed": len(observed) - len(obs),
              "missing_gene_reference": int(ref.gene_id.isna().sum()),
              "missing_gene_observed": int(obs.gene_id.isna().sum())}
    return _result(values, {k: "fraction" if k.endswith("rate") else units for k in values}, counts,
                   {"bounds_zyx": bounds.tolist(), "population_boundary": "half-open",
                    "policy": "greedy", "threshold": float(threshold), "boundary": "inclusive",
                    "units": units, "frame_id": reference_metadata.frame_id, "gene": "same nonmissing gene_id"},
                   details={"matched_pairs": pairs})
