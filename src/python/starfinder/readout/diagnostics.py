"""The three §2.8 diagnostics: read inspection, population summaries and decision inspection.

docs/readout-contract.md, "Diagnostics". Each reads retained results only (no image
access and no rerun of a decision). inspect_read, plot_read and explain_read run on
demand; FOV.run never calls them, and it stores summarize_reads in run.json.
"""

from collections.abc import Mapping
from math import isfinite

import numpy as np
import pandas as pd

from starfinder.spot_finding import SpotFindingResult
from ._codebook_aware import _build_one_error_index, _candidate_sequences, _channel_probabilities, _score_candidates
from ._direct import DirectPanel
from ._layout import colors_have_ends, read_orientation, segment_bases, segment_colors
from .codebook import Codebook, encoding_spec
from .decoding import DECODING_METHODS, BarcodeDecodingResult
from .deduplication import (DEDUPLICATION_COLUMNS, DeduplicationConfig, ReadDeduplicationResult,
                            cross_channel_pairs, deduplication_counts)
from .extraction import IntensityExtractionResult
from .filtering import ReadFilteringResult
from .scoring import ReadScoringResult

STATUSES = ("assigned", "ambiguous", "no_signal", "unmatched")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# Read stages from the most downstream; a read result or FOV.results gives them.
_STAGES = (("filtering", ReadFilteringResult), ("deduplication", ReadDeduplicationResult),
           ("scoring", ReadScoringResult), ("decoding", BarcodeDecodingResult))


def _stages(reads):
    """{stage: result} of one read result, or of a mapping of stage names to results (FOV.results)."""
    if isinstance(reads, Mapping):
        found = {}
        for stage, kind in _STAGES + (("extraction", IntensityExtractionResult), ("spot_finding", SpotFindingResult)):
            value = reads.get(stage)
            if value is None:
                continue
            if not isinstance(value, kind):
                raise TypeError(f"{stage} must be a {kind.__name__}")
            found[stage] = value
        if not any(stage in found for stage, _ in _STAGES):
            raise ValueError("no read result: the mapping holds none of decoding, scoring, deduplication, filtering")
        return found
    for stage, kind in _STAGES:
        if isinstance(reads, kind):
            return {stage: reads}
    raise TypeError("reads must be a BarcodeDecodingResult, ReadScoringResult, ReadDeduplicationResult, "
                    "ReadFilteringResult or a mapping of stage names to them (FOV.results)")


def _table(stages):
    """The most downstream read table, which holds every earlier stage's columns."""
    return next(stages[stage].table for stage, _ in _STAGES if stage in stages)


def _mode(stages, table):
    for stage in ("deduplication", "scoring", "decoding"):
        if stage in stages:
            return stages[stage].readout_mode
    return "direct" if "own_channel_rank" in table else "multiplexed"


def _row(table, spot_id):
    match = np.flatnonzero(table.spot_id.astype(str).to_numpy() == str(spot_id))
    if len(match) != 1:
        raise ValueError(f"spot_id {spot_id!r} is not a read of this table")
    return int(match[0]), table.iloc[int(match[0])]


def _number(value):
    """A finite float, or None (JSON-safe)."""
    if value is None or value is pd.NA:
        return None
    value = float(value)
    return value if isfinite(value) else None


def _missing(value):
    return value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value))


def _text(value):
    return None if _missing(value) else str(value)


def _first_bases(segment, assigned_bases):
    """Allowed first bases (declared ends), else the assigned entry's, else A, C, G and T."""
    if segment.ends:
        return list(dict.fromkeys(first for first, _ in segment.ends))
    return [assigned_bases[0]] if assigned_bases else list("ACGT")


def _segment_reading(colors, segment, spec, encoding, firsts):
    """Bases of one segment's colors in read orientation; two_base gives "first:bases" per first base."""
    if not isinstance(colors, str) or not colors or any(c not in spec.alphabet for c in colors):
        return None
    if not spec.needs_first_base:
        return read_orientation(spec.decode(colors, encoding, None), encoding)
    return ";".join(f"{f}:{read_orientation(spec.decode(colors, encoding, f), encoding)}" for f in firsts)


# --- Read inspection ----------------------------------------------------------------------------

def inspect_read(intensity_result: IntensityExtractionResult, reads, spot_id: str, *,
                 reference: Codebook | DirectPanel) -> pd.DataFrame:
    """One read's measurements and colors, one row per (round, channel), in round-major order.

    Columns: spot_id, round, channel, color (the channel's color symbol, missing
    in readout mode direct), valid, box_voxels (voxels the box summed), sum (the
    extracted neighborhood sum), background (grey levels per voxel),
    background_sum (background x box_voxels), subtracted (sum - background_sum,
    signed), noise, probability (the decoders' channel probability of the round,
    (max(sum, 0) + 1e-6) / its round total), qc_probability (the same on the
    background-subtracted sums clipped at 0, as the shared score uses),
    observed_color and assigned_color (the round's WTA observed color, M for a
    tie, and the assigned entry's color; missing in direct mode), observed and
    assigned (whether this channel holds that color; in direct mode assigned
    marks the candidate's own round and channel), segment (the segment layout
    segment of the round) and, for that segment, observed_bases and
    assigned_bases in read orientation: for two_base, "first:bases" decoded from
    each allowed first base of the segment's declared ends (else the assigned
    entry's first base, else A, C, G and T), joined by ";"; for one_base the
    bases. Background columns are NaN when the extraction measured none.

    reads is a read result (decoding, scoring, deduplication or filtering) or a
    mapping of stage names to them, such as FOV.results; reference is the
    Codebook (multiplexed) or the DirectPanel (direct).
    """
    if not isinstance(intensity_result, IntensityExtractionResult):
        raise TypeError("intensity_result must be an IntensityExtractionResult")
    stages = _stages(reads)
    table = _table(stages)
    _, row = _row(table, spot_id)
    if str(spot_id) not in intensity_result.spot_ids:
        raise ValueError(f"spot_id {spot_id!r} is not a candidate of the intensities")
    i = intensity_result.spot_ids.index(str(spot_id))
    mode = _mode(stages, table)
    expected = Codebook if mode == "multiplexed" else DirectPanel
    if not isinstance(reference, expected):
        raise TypeError(f"readout mode {mode!r} inspects reads with a {expected.__name__} reference")
    channels, rounds = intensity_result.channel_labels, intensity_result.round_labels
    n_c, n_r = len(channels), len(rounds)
    values = intensity_result.values[i]
    box = (intensity_result.box_voxels[i].astype(float) if intensity_result.box_voxels is not None
           else np.full(n_r, np.nan))
    background = (intensity_result.background[i] if intensity_result.background is not None
                  else np.full((n_c, n_r), np.nan))
    noise = intensity_result.noise[i] if intensity_result.noise is not None else np.full((n_c, n_r), np.nan)
    probability = _channel_probabilities(values[None])[0]
    background_sum = background * box[None, :]
    subtracted = values - background_sum
    clipped = np.maximum(subtracted, 0) + 1e-6
    with np.errstate(invalid="ignore"):
        qc_probability = clipped / clipped.sum(axis=0, keepdims=True)
    color = [None] * n_c
    observed_colors = assigned_colors = [None] * n_r
    segment_of, observed_bases, assigned_bases = [None] * n_r, {}, {}
    own = None
    if mode == "multiplexed":
        if reference.channel_labels != channels or reference.round_labels != rounds:
            raise ValueError("codebook and intensity channel/round labels must match exactly")
        color = [None] * n_c
        for symbol, c in reference.color_to_channel.items():
            color[c] = symbol
        observed = _text(row.observed_color_sequence)
        assigned = _text(row.decoded_color_sequence)
        observed_colors = list(observed) if observed and len(observed) == n_r else [None] * n_r
        assigned_colors = list(assigned) if assigned and len(assigned) == n_r else [None] * n_r
        spec, encoding, layout = encoding_spec(reference.encoding), reference.encoding, reference.layout
        start = 0
        for segment in layout.acquired:
            for j in range(start, start + spec.colors_for(segment.bases)):
                segment_of[j] = segment.name
            start += spec.colors_for(segment.bases)
        entry_bases = {}
        if assigned is not None and "base_sequence" in reference.table:
            match = reference.table[reference.table.color_sequence == assigned]
            if len(match) and not _missing(match.base_sequence.iloc[0]):
                entry_bases = {name: read_orientation(bases, encoding)
                               for name, bases in segment_bases(match.base_sequence.iloc[0], layout).items()}
        observed_parts = segment_colors(observed, layout, spec) if observed_colors[0] is not None else {}
        assigned_parts = segment_colors(assigned, layout, spec) if assigned_colors[0] is not None else {}
        for segment in layout.segments:
            firsts = _first_bases(segment, entry_bases.get(segment.name))
            observed_bases[segment.name] = _segment_reading(observed_parts.get(segment.name), segment, spec,
                                                            encoding, firsts)
            assigned_bases[segment.name] = (entry_bases[segment.name] if segment.name in entry_bases else
                                            _segment_reading(assigned_parts.get(segment.name), segment, spec,
                                                             encoding, firsts))
    else:
        if not _missing(row.get("round")) and not _missing(row.get("channel")):
            own = (rounds.index(str(row["round"])) if str(row["round"]) in rounds else None,
                   channels.index(str(row["channel"])) if str(row["channel"]) in channels else None)
    records = []
    for j, round_label in enumerate(rounds):
        for c, channel in enumerate(channels):
            records.append(dict(
                spot_id=str(spot_id), round=round_label, channel=channel, color=color[c],
                valid=bool(intensity_result.valid[i, j]), box_voxels=box[j], sum=values[c, j],
                background=background[c, j], background_sum=background_sum[c, j], subtracted=subtracted[c, j],
                noise=noise[c, j], probability=probability[c, j], qc_probability=qc_probability[c, j],
                observed_color=observed_colors[j], assigned_color=assigned_colors[j],
                observed=color[c] is not None and color[c] == observed_colors[j],
                assigned=(own == (j, c)) if mode == "direct" else color[c] is not None and color[c] == assigned_colors[j],
                segment=segment_of[j], observed_bases=observed_bases.get(segment_of[j]),
                assigned_bases=assigned_bases.get(segment_of[j])))
    frame = pd.DataFrame(records)
    strings = ["spot_id", "round", "channel", "color", "observed_color", "assigned_color", "segment",
               "observed_bases", "assigned_bases"]
    frame = frame.astype({**{c: "string" for c in strings}, "valid": bool, "observed": bool, "assigned": bool,
                          **{c: "float64" for c in ("box_voxels", "sum", "background", "background_sum",
                                                     "subtracted", "noise", "probability", "qc_probability")}})
    frame["box_voxels"] = frame.box_voxels.astype("Int64")
    return frame


def plot_read(intensity_result: IntensityExtractionResult, reads, spot_id: str, *,
              reference: Codebook | DirectPanel, ax=None):
    """Draw inspect_read's table with matplotlib and return the Figure.

    One group of bars per round: each channel's sum, with the background sum
    (background x box_voxels) as a black tick; the observed channel is outlined
    and the assigned channel is marked with a star. ax None draws on a new
    Figure (matplotlib.figure.Figure, no pyplot state).
    """
    from matplotlib.figure import Figure
    table = inspect_read(intensity_result, reads, spot_id, reference=reference)
    if ax is None:
        figure = Figure(figsize=(1.6 + 1.4 * table["round"].nunique(), 3.2))
        ax = figure.subplots()
    else:
        figure = ax.figure
    rounds = list(dict.fromkeys(table["round"]))
    channels = list(dict.fromkeys(table.channel))
    width = 0.8 / len(channels)
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for c, channel in enumerate(channels):
        part = table[table.channel == channel]
        x = np.arange(len(rounds)) + (c - (len(channels) - 1) / 2) * width
        ax.bar(x, part["sum"].to_numpy(float), width, label=channel, color=palette[c % len(palette)],
               edgecolor=["black" if o else "none" for o in part.observed], linewidth=1.5)
        ax.scatter(x, part.background_sum.to_numpy(float), marker="_", color="black", s=80, zorder=3)
        peak = part["sum"].to_numpy(float)
        for k in np.flatnonzero(part.assigned.to_numpy(bool)):
            ax.annotate("*", (x[k], peak[k]), ha="center", va="bottom")
    ax.set_xticks(np.arange(len(rounds)), rounds)
    ax.set_ylabel("neighborhood sum")
    ax.set_title(f"read {spot_id}: bars sums, ticks background x box voxels, * assigned, outline observed",
                 fontsize="small")
    ax.legend(fontsize="x-small", ncols=len(channels))
    return figure


# --- Population summaries -----------------------------------------------------------------------

def _value_counts(series):
    counts = series.dropna().astype(str).value_counts()
    return {str(k): int(counts[k]) for k in sorted(counts.index)}


def summarize_reads(reads, *, intensity_result: IntensityExtractionResult | None = None,
                    spots: SpotFindingResult | None = None, distance_voxels: float | None = None) -> dict:
    """Population summary of a read table: a JSON-safe dict of counts, quantiles and medians.

    Keys: readout_mode; reads; call_status (each status, zeros included),
    failure_reason (nonempty reasons) and call_type counts; genes and entries
    (assigned reads per gene_id and per entry_id); qc_score (per call_type of
    the scored reads, n and the 5, 25, 50, 75 and 95 % quantiles) and scoring
    (scored, no_assignment, background_unavailable), None without scores;
    deduplication (total, groups, merged_reads, conflicting_groups) and
    filtering (total, accepted, rejected and rejection_reasons), None when the
    stage did not run; intensities (per round the valid and invalid
    measurements, the valid measurements without a local background, and per
    round and channel the medians of the valid sums, background and noise),
    None without intensity_result; cross_channel_pairs (distance_voxels and the
    number of candidate pairs of one detection round in different channels
    within it), None without spots with a channel column. The pair count is
    reported in every readout mode, direct included, where no read is merged;
    distance_voxels None is the deduplication config's when the reads were
    deduplicated, else DeduplicationConfig().distance_voxels. Undefined values
    are None.

    reads is a read result or a mapping of stage names to them, such as
    FOV.results, which also supplies intensity_result ("extraction") and spots
    ("spot_finding") when they are not given.
    """
    stages = _stages(reads)
    intensity_result = intensity_result if intensity_result is not None else stages.get("extraction")
    spots = spots if spots is not None else stages.get("spot_finding")
    table = _table(stages)
    assigned = table.call_status.eq("assigned")
    summary = {"readout_mode": _mode(stages, table), "reads": len(table),
               "call_status": {s: int(table.call_status.eq(s).sum()) for s in STATUSES},
               "failure_reason": _value_counts(table.failure_reason[table.failure_reason.fillna("").ne("")]),
               "call_type": _value_counts(table.call_type) if "call_type" in table else {},
               "genes": _value_counts(table.gene_id[assigned]),
               "entries": _value_counts(table.entry_id[assigned]) if "entry_id" in table else {}}
    summary["qc_score"] = summary["scoring"] = None
    if "qc_score" in table:
        scored = table[table.qc_score.notna()]
        summary["qc_score"] = {
            call_type: dict(n=len(part), **{f"q{round(q * 100):02d}": _number(np.quantile(part.qc_score, q))
                                            for q in QUANTILES})
            for call_type, part in sorted(scored.groupby(scored.call_type.astype(str)), key=lambda kv: kv[0])}
        reason = table.qc_reason.fillna("")
        summary["scoring"] = {"scored": int(reason.eq("").sum()), "no_assignment": int(reason.eq("no_assignment").sum()),
                              "background_unavailable": int(reason.eq("background_unavailable").sum())}
    summary["deduplication"] = (deduplication_counts(table) if set(DEDUPLICATION_COLUMNS).issubset(table)
                                else None)
    summary["filtering"] = None
    if "accepted" in table:
        reasons = [r for joined in table.rejection_reasons.dropna() for r in str(joined).split(";") if r]
        accepted = int(table.accepted.sum())
        summary["filtering"] = {"total": len(table), "accepted": accepted, "rejected": len(table) - accepted,
                                "rejection_reasons": _value_counts(pd.Series(reasons, dtype="string"))}
    summary["intensities"] = None
    if intensity_result is not None:
        r = intensity_result
        valid = r.valid
        unavailable = (np.isnan(r.background).all(axis=1) & valid if r.background is not None
                       else np.zeros_like(valid))
        medians = {}
        for j, round_label in enumerate(r.round_labels):
            rows = valid[:, j]
            medians[round_label] = {}
            for c, channel in enumerate(r.channel_labels):
                entry = {"sum": _number(np.median(r.values[rows, c, j])) if rows.any() else None}
                for name in ("background", "noise"):
                    array = getattr(r, name)
                    part = array[rows, c, j] if array is not None else np.array([])
                    part = part[np.isfinite(part)]
                    entry[name] = _number(np.median(part)) if len(part) else None
                medians[round_label][channel] = entry
        summary["intensities"] = {
            "candidates": len(r.spot_ids),
            "valid": {label: int(valid[:, j].sum()) for j, label in enumerate(r.round_labels)},
            "invalid": {label: int((~valid[:, j]).sum()) for j, label in enumerate(r.round_labels)},
            "background_unavailable": {label: int(unavailable[:, j].sum()) for j, label in enumerate(r.round_labels)},
            "medians": medians}
    summary["cross_channel_pairs"] = None
    if spots is not None:
        if distance_voxels is None:
            distance_voxels = (stages["deduplication"].config.distance_voxels if "deduplication" in stages
                               else DeduplicationConfig().distance_voxels)
        pairs = cross_channel_pairs(spots, distance_voxels)
        if pairs is not None:
            summary["cross_channel_pairs"] = {"distance_voxels": float(distance_voxels), "pairs": pairs}
    return summary


# --- Decision inspection ------------------------------------------------------------------------

def _decoder(stages, table):
    if "decoding" in stages:
        return stages["decoding"].config
    if "probability_nll" in table:
        return None, "codebook_aware"
    if "own_channel_rank" in table:
        return None, "direct"
    return None, "wta"


def _color_probabilities(i, intensity_result, reference, decoding):
    """(R,) per-round color probabilities of read i in color order 1-4, or None when unavailable."""
    if intensity_result is not None and isinstance(reference, Codebook):
        values = np.maximum(intensity_result.values[i][[reference.color_to_channel[c] for c in "1234"], :], 0)
        return _channel_probabilities(values[None])[0]
    if decoding is not None and "probabilities" in decoding.diagnostics and isinstance(reference, Codebook):
        probabilities = decoding.diagnostics["probabilities"][i]
        return probabilities[[reference.color_to_channel[c] for c in "1234"], :]
    return None


def explain_read(reads, spot_id: str, *, intensity_result: IntensityExtractionResult | None = None,
                 reference: Codebook | DirectPanel | None = None) -> pd.DataFrame:
    """The ordered decisions of one read and their inputs, one row per decision.

    Columns: step (1, 2, ...), stage (decoding or assignment, end_bases,
    scoring, deduplication, filtering), item, value and limit (float64, NaN
    when not numeric), relation (how value is compared with limit), passed
    (nullable Boolean; missing when not decided by a comparison) and detail.
    Decoding lists the decoder, the required rounds (with intensity_result),
    the observed sequence and its exact lookup, then for codebook_aware the
    candidates with their scores (from decoder diagnostics, diagnostics=True,
    or recomputed from intensity_result and reference) and, for a read that is
    not an exact call, every gate value against its limit (max_hamming,
    max_corrected_round_margin, min_score_delta, max_correction_penalty,
    min_geomean_probability), and the call. Direct assignment lists the panel
    lookup and the own round. Then the per-segment end-base checks (with a
    codebook whose layout declares ends), the score components, the
    deduplication group and representative, and each filter predicate with
    its reason.

    reads is a read result or a mapping of stage names to them; FOV.results
    gives every stage, the decoder config and the intensities. A single
    filtering or scoring result has no decoder config, so the gate limits
    are then missing.
    """
    stages = _stages(reads)
    intensity_result = intensity_result if intensity_result is not None else stages.get("extraction")
    table = _table(stages)
    position, row = _row(table, spot_id)
    i = (intensity_result.spot_ids.index(str(spot_id))
         if intensity_result is not None and str(spot_id) in intensity_result.spot_ids else None)
    mode = _mode(stages, table)
    decoding = stages.get("decoding")
    config = _decoder(stages, table)
    method = config[1] if isinstance(config, tuple) else DECODING_METHODS[type(config)].name
    config = None if isinstance(config, tuple) else config
    rows = []

    def add(stage, item, value=np.nan, limit=np.nan, relation="", passed=None, detail=""):
        rows.append(dict(stage=stage, item=item, value=np.nan if _missing(value) else float(value),
                         limit=np.nan if _missing(limit) else float(limit), relation=relation,
                         passed=pd.NA if passed is None else bool(passed), detail=detail))

    def gate(item, value, limit, relation):
        decided = None
        if not _missing(value) and not _missing(limit):
            decided = value <= limit if relation == "<=" else value >= limit
        add("decoding", item, value, limit, relation, decided)

    status, call_type = str(row.call_status), _text(row.get("call_type"))
    gene, entry, reason = _text(row.gene_id), _text(row.get("entry_id")), _text(row.failure_reason) or ""
    if mode == "multiplexed":
        add("decoding", "decoder", detail=method)
        if i is not None:
            valid = intensity_result.valid[i]
            add("decoding", "valid_rounds", valid.sum(), len(valid), "==", bool(valid.all()),
                "every round must be valid, else unmatched with invalid_measurement")
            zero = int((np.maximum(intensity_result.values[i], 0).sum(axis=0) == 0).sum())
            add("decoding", "zero_signal_rounds", zero, 0, "==", zero == 0,
                "a round summing to 0 in every channel gives no_signal with zero_signal_round")
        observed = _text(row.observed_color_sequence)
        add("decoding", "observed_color_sequence", detail=observed or "")
        if isinstance(reference, Codebook):
            hit = reference.seq_to_entry.get(observed) if observed else None
            add("decoding", "exact_match", passed=hit is not None,
                detail=(f"entry {hit} gene {reference.seq_to_gene[observed]}" if hit is not None
                        else "not in the codebook"))
        if observed and ("M" in observed or "N" in observed):
            add("decoding", "tied_or_unknown_rounds", passed=False,
                detail="rounds " + ",".join(str(k + 1) for k, c in enumerate(observed) if c in "MN")
                + " tie (M) or are unknown (N)")
        if method == "codebook_aware":
            candidates = None
            if decoding is not None and "candidates" in decoding.diagnostics:
                frame = decoding.diagnostics["candidates"]
                candidates = frame[frame.spot_id.astype(str) == str(spot_id)]
                candidates = [(r.color_sequence, r.gene_id, r.probability_nll, r.geomean_probability)
                              for r in candidates.itertuples()]
            probabilities = (_color_probabilities(i, intensity_result, reference, decoding) if i is not None
                             else None)
            if candidates is None and probabilities is not None and observed:
                index = _build_one_error_index(reference.seq_to_gene, 4, len(observed))
                seqs = _candidate_sequences(observed, index, reference.seq_to_gene,
                                            max_hamming=config.max_hamming if config else 1)
                scored = _score_candidates(probabilities, seqs)
                candidates = [(r.seq, reference.seq_to_gene[r.seq], r.score, r.geomean_prob)
                              for r in scored.itertuples()]
            if candidates is None:
                add("decoding", "candidates", detail="unavailable: decode with diagnostics=True or pass "
                                                     "intensity_result and reference")
            else:
                for rank, (seq, candidate_gene, score, geomean) in enumerate(candidates):
                    entry_of = reference.seq_to_entry.get(seq) if isinstance(reference, Codebook) else None
                    add("decoding", "candidate", score, detail=f"rank {rank}: {seq} gene {candidate_gene}"
                        + (f" entry {entry_of}" if entry_of else "") + f", geomean probability {geomean:.6g}")
            if call_type != "exact":
                if config is not None and not config.allow_rescue:
                    add("decoding", "allow_rescue", passed=False, detail="rescue is off")
                gate("max_hamming", row.get("hamming_to_wta"), config.max_hamming if config else None, "<=")
                gate("max_corrected_round_margin", row.get("corrected_round_margin"),
                     config.max_corrected_round_margin if config else None, "<=")
                gate("min_score_delta", row.get("score_delta"), config.min_score_delta if config else None, ">=")
                penalty = None
                decoded = _text(row.decoded_color_sequence)
                if probabilities is not None and decoded and observed and len(decoded) == len(observed):
                    changed = [k for k, (o, d) in enumerate(zip(observed, decoded)) if o not in "MN" and o != d]
                    penalty = sum(-np.log(max(probabilities[int(decoded[k]) - 1, k], 1e-12))
                                  + np.log(max(probabilities[int(observed[k]) - 1, k], 1e-12)) for k in changed)
                gate("max_correction_penalty", penalty, config.max_correction_penalty if config else None, "<=")
                gate("min_geomean_probability", row.get("geomean_probability"),
                     config.min_geomean_probability if config else None, ">=")
        add("decoding", "call", passed=status == "assigned",
            detail=f"{status} {call_type or ''}".strip() + (f" -> {gene} (entry {entry})" if gene else "")
            + (f"; corrected rounds {_text(row.get('corrected_rounds'))}"
               if _text(row.get("corrected_rounds")) else "") + (f"; {reason}" if reason else ""))
    else:
        own_round, own_channel = _text(row.get("round")), _text(row.get("channel"))
        add("assignment", "decoder", detail=method)
        mapped = (reference.gene_of.get((own_round, own_channel)) if isinstance(reference, DirectPanel) else gene)
        add("assignment", "panel", passed=mapped is not None,
            detail=f"{own_round}/{own_channel} -> " + (mapped if mapped else "no gene (unmapped_channel)"))
        if i is not None and own_round in intensity_result.round_labels:
            j = intensity_result.round_labels.index(own_round)
            add("assignment", "own_round_valid", passed=bool(intensity_result.valid[i, j]),
                detail="the own round must be valid, else invalid_measurement")
        add("assignment", "own_channel_rank", row.get("own_channel_rank"), detail="1 = brightest in its round")
        add("assignment", "own_channel_fraction", row.get("own_channel_fraction"))
        add("assignment", "call", passed=status == "assigned",
            detail=f"{status} {call_type or ''}".strip() + (f" -> {gene} (entry {entry})" if gene else "")
            + (f"; {reason}" if reason else ""))
    if mode == "multiplexed":
        observed = _text(row.observed_color_sequence)
        if isinstance(reference, Codebook) and any(s.ends for s in reference.layout.segments):
            spec = encoding_spec(reference.encoding)
            parts = (segment_colors(observed, reference.layout, spec)
                     if observed and len(observed) == len(reference.round_labels) else {})
            for segment in reference.layout.segments:
                if segment.ends:
                    colors = parts.get(segment.name)
                    add("end_bases", f"segment {segment.name}",
                        passed=colors_have_ends(colors, segment.ends, spec, reference.encoding),
                        detail=f"colors {colors or 'unavailable'}; allowed (first, last) "
                        + ", ".join(first + last for first, last in segment.ends) + "; diagnostic unless "
                        "exclude_invalid_endpoints")
        elif "endpoint_valid" in table:
            add("end_bases", "endpoint_valid", passed=bool(row.endpoint_valid))
    if "qc_score" in table:
        scored = not _missing(row.qc_score)
        add("scoring", "qc_score", row.qc_score, passed=None,
            detail="lower ranks as more reliable; a ranking, not a probability" if scored
            else f"not scored: {_text(row.qc_reason)}")
        for item in ("qc_ambiguity_max", "qc_signal_to_background", "qc_rounds"):
            add("scoring", item, row.get(item))
    if set(DEDUPLICATION_COLUMNS).issubset(table):
        group, of = _text(row.duplicate_group), _text(row.duplicate_of)
        dedup_reason = _text(row.duplicate_reason) or ""
        add("deduplication", "group", detail=f"group {group} ({dedup_reason})" if group else "not linked")
        add("deduplication", "representative", passed=bool(row.is_representative),
            detail=f"duplicate of {of}" if of else ("every member kept: conflicting_calls"
                                                    if dedup_reason == "conflicting_calls" else ""))
    if "accepted" in table:
        filtering = stages.get("filtering")
        if filtering is not None:
            fc = filtering.config
            add("filtering", "call_status", passed=status in fc.call_statuses,
                detail=f"{status} in {', '.join(fc.call_statuses)}")
            for name, (lo, hi) in fc.score_bounds.items():
                for limit, relation in ((lo, ">="), (hi, "<=")):
                    if limit is not None:
                        value = row.get(name)
                        add("filtering", f"score:{name}", value, limit, relation,
                            False if _missing(value) else (value >= limit if relation == ">=" else value <= limit))
            if "is_representative" in table:
                add("filtering", "duplicate", passed=bool(row.is_representative) or not fc.exclude_duplicates,
                    detail=f"exclude_duplicates={fc.exclude_duplicates}")
            if "endpoint_valid" in table:
                add("filtering", "endpoint", passed=bool(row.endpoint_valid) or not fc.exclude_invalid_endpoints,
                    detail=f"exclude_invalid_endpoints={fc.exclude_invalid_endpoints}")
        add("filtering", "accepted", passed=bool(row.accepted), detail=_text(row.rejection_reasons) or "")
    frame = pd.DataFrame(rows, columns=["stage", "item", "value", "limit", "relation", "passed", "detail"])
    frame.insert(0, "step", np.arange(1, len(frame) + 1, dtype=np.int64))
    return frame.astype({"stage": "string", "item": "string", "value": "float64", "limit": "float64",
                         "relation": "string", "passed": "boolean", "detail": "string"})
