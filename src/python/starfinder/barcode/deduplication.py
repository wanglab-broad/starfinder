"""Optional cross-channel deduplication: one amplicon read in two channels keeps one read.

docs/readout-contract.md, "Optional deduplication"; docs/readout-algorithms.md, "Deduplication".
The pairwise link (same WTA observed sequence within a distance) is the rule W-278
measured; the M/N exclusion, the grouping, the representative and the conflict rule are
the accepted proposals it did not measure.
"""

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from starfinder.spot_finding import SpotFindingResult
from .decoding import READOUT_MODES, BarcodeDecodingResult
from .extraction import IntensityExtractionResult
from .scoring import ReadScoringResult

#: Columns deduplicate_reads appends to the read table, in order.
DEDUPLICATION_COLUMNS = ("duplicate_group", "duplicate_of", "is_representative", "duplicate_reason")
#: Signal compatibility rules; same_sequence is the W-278 rule with the proposed M/N exclusion.
COMPATIBILITY = ("same_sequence",)
MERGED, CONFLICTING = "same_sequence_within_distance", "conflicting_calls"

# Raised in readout mode direct (docs/readout-contract.md, "Optional deduplication").
DIRECT_MODE = ("deduplication is not available in readout mode 'direct': different channels and rounds are "
               "different genes there, and no direct-mode read is merged or reassigned")


@dataclass(frozen=True)
class DeduplicationConfig:
    """Cross-channel duplicate rule; off unless a config is given (PipelineConfig.deduplication).

    distance_voxels is the inclusive Euclidean distance between candidate
    coordinates in zero-based voxel index space (default 1.0, the W-278 design
    choice; index space, not microns). compatibility "same_sequence" links two
    reads whose WTA observed color sequences are identical and contain no M or N.
    """

    distance_voxels: float = 1.0
    compatibility: str = "same_sequence"

    def __post_init__(self):
        d = self.distance_voxels
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or d < 0:
            raise ValueError("distance_voxels must be a finite nonnegative number of voxels")
        if self.compatibility not in COMPATIBILITY:
            raise ValueError(f"compatibility must be one of {COMPATIBILITY}")


@dataclass(frozen=True)
class ReadDeduplicationResult:
    """The read table with the deduplication columns appended; every row and identity unchanged.

    duplicate_group is the spot_id of the group's representative (missing for a
    read that is linked to no other), duplicate_of the representative's spot_id
    for a merged member (missing otherwise), is_representative is true for
    representatives and ungrouped reads, and duplicate_reason is "",
    same_sequence_within_distance or conflicting_calls. In a conflicting group
    every member stays a representative, and duplicate_group is the spot_id the
    representative rule would have chosen. counts holds total, groups (merged
    groups), merged_reads (reads that are not representatives) and
    conflicting_groups; diagnostics holds the cross-channel pairs within the
    distance (``pairs``) and their counts, and is empty for a result reloaded
    from a checkpoint. The repr is a one-line count summary.
    """

    table: pd.DataFrame
    spot_namespace: str
    channel_labels: tuple[str, ...]
    round_labels: tuple[str, ...]
    config: DeduplicationConfig
    counts: dict[str, int]
    readout_mode: str = "multiplexed"
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.config, DeduplicationConfig):
            raise TypeError("config must be DeduplicationConfig")
        if self.readout_mode not in READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {READOUT_MODES}; got {self.readout_mode!r}")
        if self.readout_mode == "direct":
            raise ValueError(DIRECT_MODE)
        if not isinstance(self.table, pd.DataFrame) or not set(DEDUPLICATION_COLUMNS).issubset(self.table):
            raise ValueError(f"a deduplicated read table needs the columns {', '.join(DEDUPLICATION_COLUMNS)}")
        if not self.table.spot_namespace.eq(self.spot_namespace).all():
            raise ValueError("deduplication namespace mismatch")

    def _summary(self):
        c = self.counts
        return (f"{c['total']} reads — {c['groups']} groups, {c['merged_reads']} merged, "
                f"{c['conflicting_groups']} conflicting")

    def __repr__(self):
        return f"ReadDeduplicationResult: {self._summary()}"


def deduplication_counts(table):
    """counts of a deduplicated table: total, groups, merged_reads and conflicting_groups."""
    reason = table.duplicate_reason
    return {"total": len(table),
            "groups": int(table.duplicate_group[reason.eq(MERGED)].nunique()),
            "merged_reads": int((~table.is_representative.astype(bool)).sum()),
            "conflicting_groups": int(table.duplicate_group[reason.eq(CONFLICTING)].nunique())}


def _compatible(sequence):
    return isinstance(sequence, str) and bool(sequence) and "M" not in sequence and "N" not in sequence


def _components(n, links):
    """Connected-component label (smallest member row) of each row over the linked pairs."""
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in links:
        a, b = root(i), root(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return [root(i) for i in range(n)]


def deduplicate_reads(
    result: BarcodeDecodingResult | ReadScoringResult,
    spots: SpotFindingResult,
    intensity_result: IntensityExtractionResult,
    *,
    config: DeduplicationConfig = DeduplicationConfig(),
    detection_round: str | None = None,
) -> ReadDeduplicationResult:
    """Group cross-channel reads of one amplicon and keep one original read per group.

    Pairs are candidates of the same detection round in different detection
    channels (the spot table's channel column); same-channel pairs are never
    grouped (that is the §2.7 merge_radius_zyx). Two reads are linked when their
    Euclidean distance in voxel index space is at most config.distance_voxels
    and their WTA observed color sequences are identical and contain no M or N.
    Groups are the connected components of the links. A group whose assigned
    members do not all share one entry_id is not merged (conflicting_calls).
    Otherwise its representative is one original candidate: among the members
    with an assigned call (all members if none is assigned), the one with the
    largest extracted sum in its own detection channel in the detection round,
    ties going to the earliest row of the spot table; the other members,
    unassigned ones included, are its duplicates. No read is averaged,
    re-extracted, removed or changed: the deduplication columns are appended
    (see ReadDeduplicationResult), and deduplication columns already in the
    table are replaced.

    result is the decoding (or scoring) result of the candidates in spots, whose
    identities must match the result and intensity_result in order.
    detection_round is the round of candidates without a round column (FOV
    passes the reference round); None takes the first round of the intensities.

    Raises
    ------
    TypeError
        Wrong argument types (a result without decoding).
    ValueError
        Readout mode direct, candidates without a channel column, mismatched
        identities, or a detection round the intensities do not hold.
    """
    if not isinstance(result, (BarcodeDecodingResult, ReadScoringResult)):
        raise TypeError("deduplication needs decoded reads: a BarcodeDecodingResult or ReadScoringResult")
    if not isinstance(spots, SpotFindingResult) or not isinstance(intensity_result, IntensityExtractionResult):
        raise TypeError("expected SpotFindingResult and IntensityExtractionResult")
    if not isinstance(config, DeduplicationConfig):
        raise TypeError("config must be DeduplicationConfig")
    if result.readout_mode == "direct":
        raise ValueError(DIRECT_MODE)
    result.__post_init__()
    intensity_result.__post_init__()
    config.__post_init__()
    table = result.table.drop(columns=[c for c in DEDUPLICATION_COLUMNS if c in result.table])
    table = table.reset_index(drop=True)
    frame = spots.spots.reset_index(drop=True)
    if (result.spot_namespace != spots.spot_namespace or result.spot_namespace != intensity_result.spot_namespace
            or tuple(table.spot_id) != tuple(frame.spot_id) or tuple(table.spot_id) != intensity_result.spot_ids):
        raise ValueError("read, candidate and intensity identities must match in order")
    if "channel" not in frame:
        raise ValueError("deduplication needs the detection channel of each candidate (a channel column in the "
                         "spot table)")
    labels = spots.diagnostics.get("channel_labels")
    position = {label: i for i, label in enumerate(intensity_result.channel_labels)}
    channel = frame.channel.to_numpy(dtype=np.int64)
    if labels is not None:
        if any(label not in position for label in labels):
            raise ValueError("candidate channel labels must be channels of the intensities")
        channel = np.array([position[labels[c]] for c in channel], dtype=np.int64)
    if len(channel) and (channel.min() < 0 or channel.max() >= len(intensity_result.channel_labels)):
        raise ValueError("candidate channels are outside the intensities")
    rounds = {label: j for j, label in enumerate(intensity_result.round_labels)}
    if "round" in frame:
        detected = frame["round"].astype(str).tolist()
    else:
        detected = [intensity_result.round_labels[0] if detection_round is None else detection_round] * len(frame)
    missing = sorted(set(detected) - set(rounds))
    if missing:
        raise ValueError(f"detection rounds {missing} are not rounds of the intensities")
    own = intensity_result.values[np.arange(len(frame)), channel, [rounds[r] for r in detected]]
    points = frame[["z", "y", "x"]].to_numpy(dtype=np.float64)
    observed = table.observed_color_sequence.astype(object).where(table.observed_color_sequence.notna(), None)
    observed = observed.tolist()
    d = float(config.distance_voxels)
    pairs = []
    for label in dict.fromkeys(detected):
        rows = np.flatnonzero(np.asarray(detected) == label)
        if len(rows) < 2:
            continue
        # The tree's inclusive radius, widened by a rounding margin; the exact distance decides below.
        found = cKDTree(points[rows]).query_pairs(d * (1 + 1e-9) + 1e-12, output_type="ndarray")
        for a, b in sorted(map(tuple, found.tolist())):
            i, j = int(rows[a]), int(rows[b])
            distance = float(np.sqrt(((points[i] - points[j]) ** 2).sum()))
            if channel[i] == channel[j] or distance > d:
                continue
            same = _compatible(observed[i]) and observed[i] == observed[j]
            pairs.append((i, j, label, distance, same))
    links = [(i, j) for i, j, _, _, same in pairs if same]
    component = _components(len(table), links)
    members = {}
    for i, c in enumerate(component):
        members.setdefault(c, []).append(i)
    group = [None] * len(table)
    of = [None] * len(table)
    representative = [True] * len(table)
    reason = [""] * len(table)
    assigned = table.call_status.eq("assigned").to_numpy(dtype=bool)
    entries = (table.entry_id if "entry_id" in table else table.gene_id).astype(object).tolist()
    spot_ids = table.spot_id.astype(str).tolist()
    for rows in members.values():
        if len(rows) < 2:
            continue
        calls = [i for i in rows if assigned[i]]
        pool = calls or rows
        # Largest own-channel sum in the detection round; ties go to the earliest row.
        chosen = max(pool, key=lambda i: (own[i], -i))
        conflicting = len({entries[i] for i in calls}) > 1
        for i in rows:
            group[i] = spot_ids[chosen]
            reason[i] = CONFLICTING if conflicting else MERGED
            if not conflicting and i != chosen:
                representative[i], of[i] = False, spot_ids[chosen]
    table["duplicate_group"] = pd.array(group, dtype="string")
    table["duplicate_of"] = pd.array(of, dtype="string")
    table["is_representative"] = np.asarray(representative, dtype=bool)
    table["duplicate_reason"] = pd.array(reason, dtype="string")
    pair_table = pd.DataFrame({
        "spot_id_a": pd.array([spot_ids[p[0]] for p in pairs], dtype="string"),
        "spot_id_b": pd.array([spot_ids[p[1]] for p in pairs], dtype="string"),
        "round": pd.array([p[2] for p in pairs], dtype="string"),
        "distance_voxels": np.array([p[3] for p in pairs], dtype=np.float64),
        "linked": np.array([p[4] for p in pairs], dtype=bool)})
    diagnostics = {"pairs": pair_table, "cross_channel_pairs": len(pairs), "linked_pairs": len(links),
                   "detection_rounds": list(dict.fromkeys(detected))}
    return ReadDeduplicationResult(table, result.spot_namespace, result.channel_labels, result.round_labels,
                                   config, deduplication_counts(table), result.readout_mode, diagnostics)


def cross_channel_pairs(spots, distance_voxels):
    """Cross-channel candidate pairs of one detection round within distance_voxels (inclusive), counted.

    The population summary reports it in every readout mode, direct included,
    where no read is merged; candidates without a channel column give None.
    """
    frame = spots.spots
    if "channel" not in frame:
        return None
    points = frame[["z", "y", "x"]].to_numpy(dtype=np.float64)
    channel = frame.channel.to_numpy()
    detected = frame["round"].astype(str).to_numpy() if "round" in frame else np.zeros(len(frame), dtype=object)
    d, count = float(distance_voxels), 0
    for label in dict.fromkeys(detected.tolist()):
        rows = np.flatnonzero(detected == label)
        if len(rows) < 2:
            continue
        for a, b in cKDTree(points[rows]).query_pairs(d * (1 + 1e-9) + 1e-12, output_type="ndarray").tolist():
            i, j = rows[a], rows[b]
            count += int(channel[i] != channel[j] and np.sqrt(((points[i] - points[j]) ** 2).sum()) <= d)
    return count
