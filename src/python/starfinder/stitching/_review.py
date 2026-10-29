"""The review policy V1, the approval and the review gate (design/specs/stitching.md, "Failure policy").

The checks are those of design/specs/stitching-algorithms.md ("Placement-quality checks"); they
read only the record.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from numbers import Real

import numpy as np

from ._errors import PlacementReviewRequiredError
from ._placement import FLAGS, PlacementResult, table_sha256

REVIEW_FLAGS = ("correction_above_bound", "residual_above_tolerance", "nan_quality_link", "optimizer_removed_link")
# Review reasons in the order they are listed; supplied_placement stands alone.
REASONS = ("unplaced_fovs", "correction_above_bound", "residual_above_tolerance", "optimizer_removed_link")
SUPPLIED_REASON = "supplied_placement"


@dataclass(frozen=True)
class ReviewPolicy:
    """Review criterion V1: when a placement needs review before sample assembly.

    Parameters
    ----------
    max_correction_fraction : float
        Nonnegative correction bound as a fraction of nominal overlap width for placement review.
    max_residual_um : float
        Nonnegative kept-link residual bound for placement review, in µm.

    Notes
    -----
    Review is required when any FOV is not ``image_supported``; when a
    correction exceeds ``max_correction_fraction`` of the FOV's nominal overlap
    width in Y or X (the median over its registered pairs); when a kept link's
    residual exceeds ``max_residual_um`` (µm); or when the method removed a link
    in its optimization. Both defaults (0.5 and 1.0 µm) are the contract's
    proposals, tested at their boundary and not tuned. A supplied placement is
    always ``required``, with the reason ``supplied_placement``.

    Raises
    ------
    ValueError
        A value that is not a finite number ≥ 0.
    """

    max_correction_fraction: float = 0.5
    max_residual_um: float = 1.0

    def __post_init__(self):
        for name in ("max_correction_fraction", "max_residual_um"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) \
                    or value < 0:
                raise ValueError(f"{name} must be a finite number ≥ 0; got {value!r}")


def _flag_text(flags):
    return ";".join(f for f in FLAGS if f in flags)


def review_placement(result: PlacementResult, policy: ReviewPolicy = ReviewPolicy()) -> PlacementResult:
    """Apply the review criterion to a placement; return it with its flags and ``record["review"]``.

    The flags ``correction_above_bound``, ``residual_above_tolerance``,
    ``nan_quality_link`` and ``optimizer_removed_link`` are recomputed from the
    tables (``fallback`` and ``supplied`` are kept): the correction bound
    ``|correction| > f × w`` in Y or X, with ``w`` the median nominal overlap
    width of the FOV's registered pairs (links not ``pruned``) in that axis; a
    kept link's residual above ``max_residual_um`` flags both FOVs; a registered
    link with NaN quality or one the optimizer removed flags both FOVs.
    ``record["review"]`` becomes ``{"state", "reasons", "policy", "flagged",
    "approvals"}``: ``state`` ``not_needed`` or ``required``, ``reasons`` among
    ``unplaced_fovs``, ``correction_above_bound``, ``residual_above_tolerance``
    and ``optimizer_removed_link``, ``flagged`` the FOVs of each reason, and the
    approvals whose table SHA-256 still match. No status and no origin changes.

    A placement whose method is ``supplied`` keeps ``required`` with the reasons
    exactly ``[supplied_placement]`` and gains no flag, whatever its corrections:
    it has no kept link and no registered pair.

    Raises
    ------
    TypeError
        result is not a PlacementResult or policy not a ReviewPolicy.
    """
    if not isinstance(result, PlacementResult):
        raise TypeError("result must be a PlacementResult")
    if not isinstance(policy, ReviewPolicy):
        raise TypeError("policy must be a ReviewPolicy")
    fovs, links = result.fovs, result.links
    ids = list(fovs["fov_id"])
    flags = {fov: {f for f in value.split(";") if f and f not in REVIEW_FLAGS} for fov, value in
             zip(ids, fovs["flags"])}
    flagged = {}
    if result.record["method"]["method"] == "supplied":
        reasons = [SUPPLIED_REASON]
    else:
        def mark(flag, fov_list):
            for fov in fov_list:
                flags[fov].add(flag)

        registered = links[links["reason"] != "pruned"]
        above = []
        for fov, cy, cx in zip(ids, fovs["correction_y"], fovs["correction_x"]):
            mine = registered[(registered["fov_a"] == fov) | (registered["fov_b"] == fov)]
            if mine.empty or not (math.isfinite(cy) and math.isfinite(cx)):
                continue
            w_y = float(np.median(mine["overlap_y1"] - mine["overlap_y0"]))
            w_x = float(np.median(mine["overlap_x1"] - mine["overlap_x0"]))
            if abs(cy) > policy.max_correction_fraction * w_y or abs(cx) > policy.max_correction_fraction * w_x:
                above.append(fov)
        mark("correction_above_bound", above)
        residual = links[links["kept"] & (links["residual"] > policy.max_residual_um)]
        mark("residual_above_tolerance", list(residual["fov_a"]) + list(residual["fov_b"]))
        nan = registered[registered["quality"].isna()]
        mark("nan_quality_link", list(nan["fov_a"]) + list(nan["fov_b"]))
        removed = links[links["reason"] == "optimizer_removed"]
        mark("optimizer_removed_link", list(removed["fov_a"]) + list(removed["fov_b"]))
        candidates = {
            "unplaced_fovs": [fov for fov, status in zip(ids, fovs["status"]) if status != "image_supported"],
            "correction_above_bound": above,
            "residual_above_tolerance": [f for f in ids if "residual_above_tolerance" in flags[f]],
            "optimizer_removed_link": [f for f in ids if "optimizer_removed_link" in flags[f]],
        }
        reasons = [reason for reason in REASONS if candidates[reason]]
        flagged = {reason: candidates[reason] for reason in reasons}
    fovs = fovs.assign(flags=[_flag_text(flags[fov]) for fov in ids])
    hashes = table_sha256(fovs), table_sha256(links)
    approvals = [a for a in result.record["review"].get("approvals", []) if (a["fovs_sha256"], a["links_sha256"])
                 == hashes] if isinstance(result.record.get("review"), dict) else []
    review = {"state": "required" if reasons else "not_needed", "reasons": reasons, "policy": asdict(policy),
              "flagged": flagged, "approvals": approvals}
    return replace(result, fovs=fovs, record=dict(result.record, review=review))


def approve_placement(result: PlacementResult, *, reviewer: str, note: str) -> PlacementResult:
    """Record an approval of the placement as it is; return a new result.

    The approval holds the reviewer, the UTC time, the note and the SHA-256 of
    the per-FOV and link tables it approves (the bytes ``save_placement``
    writes). No origin, correction, status or flag changes.

    Raises
    ------
    TypeError
        result is not a PlacementResult or note not a string.
    ValueError
        An empty reviewer.
    """
    if not isinstance(result, PlacementResult):
        raise TypeError("result must be a PlacementResult")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer must be a nonempty string")
    if not isinstance(note, str):
        raise TypeError("note must be a string")
    review = dict(result.record.get("review") or {})
    approval = {"reviewer": reviewer, "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "note": note, "fovs_sha256": table_sha256(result.fovs), "links_sha256": table_sha256(result.links)}
    review["approvals"] = [*review.get("approvals", []), approval]
    return replace(result, record=dict(result.record, review=review))


def require_approval(result: PlacementResult) -> None:
    """Raise :class:`PlacementReviewRequiredError` with the reasons when review is required and unapproved.

    An approval counts only when its table SHA-256 equal those of the
    placement's current tables. Sample assembly calls this before it reads any
    origin.
    """
    review = result.record.get("review") or {}
    if review.get("state") != "required":
        return
    hashes = table_sha256(result.fovs), table_sha256(result.links)
    if any((a.get("fovs_sha256"), a.get("links_sha256")) == hashes for a in review.get("approvals", [])):
        return
    raise PlacementReviewRequiredError(
        f"the placement of sample {result.sample!r} needs review ({', '.join(review.get('reasons', []))}) and holds "
        "no approval of its current tables; inspect it and call approve_placement(result, reviewer=..., note=...)")
