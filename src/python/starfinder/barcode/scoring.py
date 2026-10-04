"""The shared read-QC score: a ranking of calls that never changes them.

docs/readout-contract.md, "Shared read-QC score"; docs/readout-algorithms.md, "Shared read-QC score".
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .codebook import Codebook
from .decoding import READOUT_MODES, BarcodeDecodingResult
from .extraction import IntensityExtractionResult
from ._direct import DirectPanel

#: Columns score_reads appends to the read table, in order.
SCORE_COLUMNS = ("qc_score", "qc_ambiguity_max", "qc_signal_to_background", "qc_rounds", "qc_reason")

# Raised when the extraction did not measure the local background (background=None, or a
# candidates checkpoint written before §2.8); the stage to rerun is extraction.
NO_BACKGROUND = ("scoring needs the local background measured at extraction, and these intensities have none "
                 "(NeighborhoodSumConfig.background=None, or a candidates checkpoint written without it); "
                 "rerun extraction with NeighborhoodSumConfig.background=LocalBackgroundConfig()")


@dataclass(frozen=True)
class ReadScoreConfig:
    """The shared read-QC score bgcorr_probability (W-278 design D1); no tunable parameter.

    The constants 1e-6 (additive) and 1e-12 (probability floor) are the
    codebook-aware decoder's.
    """

    method: str = field(default="bgcorr_probability", init=False)

    def __post_init__(self):
        if self.method != "bgcorr_probability":
            raise ValueError("method must be bgcorr_probability")


@dataclass(frozen=True)
class ReadScoringResult:
    """The read table with the score columns appended; every row and identity unchanged.

    The score columns are qc_score (lower ranks as more reliable),
    qc_ambiguity_max, qc_signal_to_background, qc_rounds (float64; NaN for a
    read without a score) and qc_reason ("" when scored, no_assignment or
    background_unavailable). counts holds total, scored, no_assignment and
    background_unavailable. The score is a ranking, not a calibrated probability,
    and sets no cutoff. The repr is a one-line count summary.
    """

    table: pd.DataFrame
    spot_namespace: str
    channel_labels: tuple[str, ...]
    round_labels: tuple[str, ...]
    config: ReadScoreConfig
    counts: dict[str, int]
    readout_mode: str = "multiplexed"

    def __post_init__(self):
        if not isinstance(self.config, ReadScoreConfig):
            raise TypeError("config must be ReadScoreConfig")
        if self.readout_mode not in READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {READOUT_MODES}; got {self.readout_mode!r}")
        if not isinstance(self.table, pd.DataFrame) or not set(SCORE_COLUMNS).issubset(self.table):
            raise ValueError(f"a scored read table needs the columns {', '.join(SCORE_COLUMNS)}")
        if not self.table.spot_namespace.eq(self.spot_namespace).all():
            raise ValueError("scoring namespace mismatch")

    def _summary(self):
        counts = self.counts
        unscored = [f"{k} {counts[k]}" for k in ("no_assignment", "background_unavailable") if counts.get(k)]
        return f"{counts['scored']} of {counts['total']} reads scored" + (" — " + ", ".join(unscored) if unscored else "")

    def __repr__(self):
        return f"ReadScoringResult: {self._summary()}"


def _components(values, background, box_voxels, assigned):
    """W-278 D1 and its components over the used rounds.

    values and background (N, C, U), box_voxels (N, U), assigned channel indices
    (N, U). v' = max(v - box_voxels x background, 0); p = (v'_a + 1e-6) /
    sum_c (v'_c + 1e-6); qc_score = sum_u -log max(p, 1e-12). ambiguity: max over
    u of the strongest other channel's clipped v' over the assigned channel's;
    signal-to-background: mean over u of the signed (v_a - box_voxels x
    background_a) / (box_voxels x background_a) (W-278 scripts/w278_lib.py,
    components).
    """
    n, _, u = values.shape
    rows, rounds = np.arange(n)[:, None], np.arange(u)[None, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        box_background = box_voxels[:, None, :] * background
        subtracted = values - box_background
        own = subtracted[rows, assigned, rounds]
        signal_to_background = (own / box_background[rows, assigned, rounds]).mean(axis=1)
        other = subtracted.copy()
        other[rows, assigned, rounds] = -np.inf
        ambiguity = (np.maximum(other.max(axis=1), 0) / np.maximum(own, 0)).max(axis=1)
        p = np.maximum(subtracted, 0) + 1e-6
        p = p / p.sum(axis=1, keepdims=True)
        score = -np.log(np.maximum(p[rows, assigned, rounds], 1e-12)).sum(axis=1)
    return score, ambiguity, signal_to_background


def score_reads(
    decoding_result: BarcodeDecodingResult,
    intensity_result: IntensityExtractionResult,
    *,
    reference: Codebook | DirectPanel,
    config: ReadScoreConfig = ReadScoreConfig(),
) -> ReadScoringResult:
    """Score every assigned read; append the score columns without changing any identity.

    The score of an assigned read (W-278 design D1, bgcorr_probability) is the
    decoder's probability NLL of the assigned entry recomputed on background-
    subtracted sums: per used round, v'_c = max(v_c - box_voxels x background_c,
    0) and p = (v'_a + 1e-6) / sum_c (v'_c + 1e-6) at the assigned channel a;
    qc_score = sum of -log max(p, 1e-12). Used rounds are every sequencing round
    in readout mode multiplexed, where the assigned channels are the decoded
    entry's colors (a rescued round is scored at the codeword's channel), and
    the candidate's own round in readout mode direct, where the assigned channel
    is its own channel. Components: qc_ambiguity_max, qc_signal_to_background
    (signed numerator) and qc_rounds. A read that is not assigned has NaN with
    qc_reason no_assignment; an assigned read with a used round without
    background (fewer ring voxels than min_voxels) has NaN with
    background_unavailable. gene_id, entry_id, call_status and call_type, and
    every other decoding column, are unchanged; score columns already in the
    table (a rescored pre_qc table) are replaced.

    reference is the Codebook (multiplexed) or the DirectPanel (direct) the reads
    were called with.

    Raises
    ------
    TypeError
        Wrong argument types, or a reference of the other readout mode.
    ValueError
        The intensities have no background measurements (rerun extraction), or
        identities or labels differ between the reads, intensities and reference.
    """
    if not isinstance(decoding_result, BarcodeDecodingResult) or not isinstance(
            intensity_result, IntensityExtractionResult):
        raise TypeError("expected BarcodeDecodingResult and IntensityExtractionResult")
    if not isinstance(config, ReadScoreConfig):
        raise TypeError("config must be ReadScoreConfig")
    mode = decoding_result.readout_mode
    expected = Codebook if mode == "multiplexed" else DirectPanel
    if not isinstance(reference, expected):
        raise TypeError(f"readout mode {mode!r} scores with a {expected.__name__} reference")
    decoding_result.__post_init__()
    intensity_result.__post_init__()
    config.__post_init__()
    if intensity_result.background is None or intensity_result.box_voxels is None:
        raise ValueError(NO_BACKGROUND)
    table = decoding_result.table.drop(columns=[c for c in SCORE_COLUMNS if c in decoding_result.table])
    table = table.reset_index(drop=True)
    if (decoding_result.spot_namespace != intensity_result.spot_namespace
            or tuple(table.spot_id) != intensity_result.spot_ids):
        raise ValueError("read and intensity identities must match in order")
    if (decoding_result.channel_labels != intensity_result.channel_labels
            or decoding_result.round_labels != intensity_result.round_labels):
        raise ValueError("read and intensity channel/round labels must match exactly")
    n = len(table)
    values, background, box_voxels = intensity_result.values, intensity_result.background, intensity_result.box_voxels
    assigned = table.call_status.eq("assigned").to_numpy(dtype=bool)
    if mode == "multiplexed":
        if (reference.channel_labels != intensity_result.channel_labels
                or reference.round_labels != intensity_result.round_labels):
            raise ValueError("codebook and intensity channel/round labels must match exactly")
        rounds = values.shape[2]
        channels = np.zeros((n, rounds), dtype=np.int64)
        for i in np.flatnonzero(assigned):
            channels[i] = [reference.color_to_channel[c] for c in table.decoded_color_sequence.iloc[i]]
        used_values, used_background, used_boxes = values, background, box_voxels
    else:
        labels = {label: i for i, label in enumerate(intensity_result.channel_labels)}
        positions = {label: j for j, label in enumerate(intensity_result.round_labels)}
        own = np.zeros(n, dtype=np.int64)
        channels = np.zeros((n, 1), dtype=np.int64)
        for i in np.flatnonzero(assigned):
            own[i], channels[i, 0] = positions[table["round"].iloc[i]], labels[table.channel.iloc[i]]
        rows = np.arange(n)
        used_values = values[rows, :, own][:, :, None]
        used_background = background[rows, :, own][:, :, None]
        used_boxes = box_voxels[rows, own][:, None]
    unavailable = assigned & np.isnan(used_background).any(axis=(1, 2))
    scored = assigned & ~unavailable
    score, ambiguity, signal_to_background = _components(used_values, used_background, used_boxes, channels)
    reason = np.where(scored, "", np.where(assigned, "background_unavailable", "no_assignment"))
    missing = np.full(n, np.nan)
    table["qc_score"] = np.where(scored, score, missing)
    table["qc_ambiguity_max"] = np.where(scored, ambiguity, missing)
    table["qc_signal_to_background"] = np.where(scored, signal_to_background, missing)
    table["qc_rounds"] = np.where(scored, float(used_values.shape[2]), missing)
    table["qc_reason"] = pd.array(reason, dtype="string")
    counts = {"total": n, "scored": int(scored.sum()), "no_assignment": int((~assigned).sum()),
              "background_unavailable": int(unavailable.sum())}
    return ReadScoringResult(table, decoding_result.spot_namespace, decoding_result.channel_labels,
                             decoding_result.round_labels, config, counts, mode)
