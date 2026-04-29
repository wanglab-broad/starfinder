"""Codebook-aware barcode decoding from per-spot intensity tensors.

This module keeps STARfinder's winner-take-all exact codebook behavior as the
baseline, then optionally rescues invalid reads when a nearby codebook sequence
is both unique and well supported by the raw channel intensities.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import exp, log

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = [
    "spot_id",
    "color_seq_wta",
    "gene_wta",
    "decoded_seq",
    "gene",
    "call_type",
    "reject_reason",
    "hamming_to_wta",
    "corrected_rounds",
    "score",
    "score_delta",
    "geomean_prob",
    "min_round_margin",
    "corrected_round_margin",
    "mean_total_intensity",
]


def _validate_codebook(
    seq_to_gene: dict[str, str],
    n_channels: int,
    n_rounds: int,
) -> None:
    for seq in seq_to_gene:
        if len(seq) != n_rounds:
            raise ValueError(
                f"Codebook sequence {seq!r} has length {len(seq)}, "
                f"expected {n_rounds}"
            )
        for color in seq:
            if not color.isdigit():
                raise ValueError(f"Codebook sequence {seq!r} contains {color!r}")
            channel = int(color)
            if channel < 1 or channel > n_channels:
                raise ValueError(
                    f"Codebook sequence {seq!r} uses channel {color}, "
                    f"outside 1..{n_channels}"
                )


def channel_probabilities(intensity_tensor: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert raw `(N, C, R)` intensities to per-round channel probabilities."""
    if intensity_tensor.ndim != 3:
        raise ValueError(
            f"Expected intensity tensor shape (N, C, R), got {intensity_tensor.shape}"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")

    values = np.asarray(intensity_tensor, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 0.0, None)

    adjusted = values + eps
    totals = adjusted.sum(axis=1, keepdims=True)
    return adjusted / totals


def wta_color_sequences(probs: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Return WTA color sequences plus per-round margin diagnostics."""
    if probs.ndim != 3:
        raise ValueError(f"Expected probabilities shape (N, C, R), got {probs.shape}")

    n_spots, n_channels, n_rounds = probs.shape
    if n_channels == 0:
        raise ValueError("probabilities must have at least one channel")

    color_matrix = np.empty((n_spots, n_rounds), dtype=object)
    diagnostics: dict[str, np.ndarray] = {}
    margins = np.full((n_spots, n_rounds), np.nan, dtype=np.float64)

    labels = np.array([str(i + 1) for i in range(n_channels)], dtype=object)
    for round_idx in range(n_rounds):
        round_probs = probs[:, :, round_idx]
        finite = np.isfinite(round_probs).all(axis=1)

        order = np.argsort(round_probs, axis=1)
        top_idx = order[:, -1]
        second_idx = order[:, -2] if n_channels > 1 else order[:, -1]
        top_prob = round_probs[np.arange(n_spots), top_idx]
        second_prob = round_probs[np.arange(n_spots), second_idx]
        margin = top_prob - second_prob

        tie_mask = np.isclose(round_probs, top_prob[:, None], rtol=0.0, atol=1e-12).sum(axis=1) > 1
        normal_mask = finite & ~tie_mask

        color_matrix[:, round_idx] = "M"
        color_matrix[~finite, round_idx] = "N"
        color_matrix[normal_mask, round_idx] = labels[top_idx[normal_mask]]

        margins[:, round_idx] = margin
        diagnostics[f"round{round_idx}_top_channel"] = top_idx + 1
        diagnostics[f"round{round_idx}_top_prob"] = top_prob
        diagnostics[f"round{round_idx}_second_prob"] = second_prob
        diagnostics[f"round{round_idx}_margin"] = margin

    color_seq = np.array(["".join(row) for row in color_matrix], dtype=object)
    diagnostics["min_round_margin"] = (
        np.nanmin(margins, axis=1) if n_rounds else np.full(n_spots, np.nan)
    )
    return color_seq, pd.DataFrame(diagnostics)


def build_one_error_index(
    seq_to_gene: dict[str, str],
    n_channels: int,
    n_rounds: int,
) -> dict[str, list[str]]:
    """Map every one-color-error sequence to valid codebook candidates."""
    _validate_codebook(seq_to_gene, n_channels, n_rounds)

    labels = [str(i + 1) for i in range(n_channels)]
    index: dict[str, set[str]] = defaultdict(set)
    for seq in seq_to_gene:
        chars = list(seq)
        for round_idx, original in enumerate(chars):
            for label in labels:
                if label == original:
                    continue
                mutated = chars.copy()
                mutated[round_idx] = label
                index["".join(mutated)].add(seq)

    return {key: sorted(values) for key, values in index.items()}


def _known_hamming(observed: str, candidate: str, unknown_chars: str) -> int:
    return sum(
        obs not in unknown_chars and obs != cand
        for obs, cand in zip(observed, candidate)
    )


def _unknown_count(observed: str, unknown_chars: str) -> int:
    return sum(char in unknown_chars for char in observed)


def candidate_sequences(
    wta_seq: str,
    one_error_index: dict[str, list[str]],
    seq_to_gene: dict[str, str],
    unknown_chars: str = "MN",
    max_hamming: int = 1,
) -> list[str]:
    """Return valid codebook candidates near an observed WTA sequence."""
    if max_hamming < 0:
        raise ValueError("max_hamming must be non-negative")
    if wta_seq in seq_to_gene:
        return [wta_seq]

    unknowns = _unknown_count(wta_seq, unknown_chars)
    if unknowns == 0 and max_hamming == 1:
        return one_error_index.get(wta_seq, [])

    candidates = []
    for seq in seq_to_gene:
        if len(seq) != len(wta_seq):
            continue
        known_mismatches = _known_hamming(wta_seq, seq, unknown_chars)
        total_edits = known_mismatches + unknowns
        if total_edits <= max_hamming:
            candidates.append(seq)

    return sorted(candidates)


def _sequence_score(
    probs_for_spot: np.ndarray,
    seq: str,
    eps: float,
    rounds: Iterable[int] | None = None,
) -> float:
    score = 0.0
    selected_rounds = range(len(seq)) if rounds is None else rounds
    for round_idx in selected_rounds:
        channel = int(seq[round_idx]) - 1
        probability = float(probs_for_spot[channel, round_idx])
        score += -log(max(probability, eps))
    return score


def score_candidates(
    probs_for_spot: np.ndarray,
    candidates: list[str],
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Score candidate codebook sequences for one spot."""
    if probs_for_spot.ndim != 2:
        raise ValueError(
            f"Expected one-spot probabilities shape (C, R), got {probs_for_spot.shape}"
        )
    if not candidates:
        return pd.DataFrame(columns=["seq", "score", "geomean_prob"])

    n_channels, n_rounds = probs_for_spot.shape
    rows = []
    for seq in candidates:
        if len(seq) != n_rounds:
            raise ValueError(
                f"Candidate sequence {seq!r} has length {len(seq)}, expected {n_rounds}"
            )
        for color in seq:
            channel = int(color) - 1
            if channel < 0 or channel >= n_channels:
                raise ValueError(
                    f"Candidate sequence {seq!r} uses channel {color}, "
                    f"outside 1..{n_channels}"
                )
        score = _sequence_score(probs_for_spot, seq, eps)
        rows.append(
            {
                "seq": seq,
                "score": score,
                "geomean_prob": exp(-score / n_rounds) if n_rounds else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["score", "seq"]).reset_index(drop=True)


def _base_row(
    spot_id: object,
    wta_seq: str,
    gene_wta: str | None,
    min_round_margin: float,
    mean_total_intensity: float,
) -> dict[str, object]:
    return {
        "spot_id": spot_id,
        "color_seq_wta": wta_seq,
        "gene_wta": gene_wta,
        "decoded_seq": None,
        "gene": None,
        "call_type": "no_call",
        "reject_reason": "",
        "hamming_to_wta": np.nan,
        "corrected_rounds": "",
        "score": np.nan,
        "score_delta": np.nan,
        "geomean_prob": np.nan,
        "min_round_margin": min_round_margin,
        "corrected_round_margin": np.nan,
        "mean_total_intensity": mean_total_intensity,
    }


def _reject(row: dict[str, object], reason: str) -> dict[str, object]:
    row["reject_reason"] = reason
    return row


def decode_codebook_aware(
    intensity_tensor: np.ndarray,
    seq_to_gene: dict[str, str],
    *,
    spot_ids: np.ndarray | list | None = None,
    max_hamming: int = 1,
    unknown_chars: str = "MN",
    min_corrected_round_margin: float | None = 0.20,
    min_score_delta: float = 0.25,
    min_geomean_prob: float = 0.45,
    max_correction_penalty: float = 1.50,
    allow_exact: bool = True,
    allow_rescue: bool = True,
) -> pd.DataFrame:
    """Decode intensities using exact WTA calls plus gated codebook rescue."""
    if intensity_tensor.ndim != 3:
        raise ValueError(
            f"Expected intensity tensor shape (N, C, R), got {intensity_tensor.shape}"
        )
    if max_hamming < 0:
        raise ValueError("max_hamming must be non-negative")

    n_spots, n_channels, n_rounds = intensity_tensor.shape
    if spot_ids is None:
        spot_ids = np.arange(n_spots)
    if len(spot_ids) != n_spots:
        raise ValueError(f"spot_ids length {len(spot_ids)} does not match {n_spots}")

    _validate_codebook(seq_to_gene, n_channels, n_rounds)

    if n_spots == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    probs = channel_probabilities(intensity_tensor)
    wta_seqs, diagnostics = wta_color_sequences(probs)
    one_error_index = build_one_error_index(seq_to_gene, n_channels, n_rounds)

    values = np.asarray(intensity_tensor, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 0.0, None)
    mean_total_intensities = values.sum(axis=1).mean(axis=1)

    rows = []
    for spot_idx, spot_id in enumerate(spot_ids):
        wta_seq = str(wta_seqs[spot_idx])
        gene_wta = seq_to_gene.get(wta_seq)
        row = _base_row(
            spot_id,
            wta_seq,
            gene_wta,
            float(diagnostics.iloc[spot_idx]["min_round_margin"]),
            float(mean_total_intensities[spot_idx]),
        )
        probs_for_spot = probs[spot_idx]

        if gene_wta is not None and allow_exact:
            score = _sequence_score(probs_for_spot, wta_seq, eps=1e-12)
            row.update(
                {
                    "decoded_seq": wta_seq,
                    "gene": gene_wta,
                    "call_type": "exact",
                    "hamming_to_wta": 0,
                    "score": score,
                    "score_delta": np.inf,
                    "geomean_prob": exp(-score / n_rounds) if n_rounds else np.nan,
                }
            )
            rows.append(row)
            continue

        if not allow_rescue:
            rows.append(_reject(row, "rescue_disabled"))
            continue

        candidates = candidate_sequences(
            wta_seq,
            one_error_index,
            seq_to_gene,
            unknown_chars=unknown_chars,
            max_hamming=max_hamming,
        )
        candidates = [seq for seq in candidates if seq != wta_seq]
        if not candidates:
            rows.append(_reject(row, "no_candidate"))
            continue

        scores = score_candidates(probs_for_spot, candidates)
        best = scores.iloc[0]
        score_delta = (
            float(scores.iloc[1]["score"] - best["score"])
            if len(scores) > 1
            else np.inf
        )
        if np.isfinite(score_delta) and score_delta < min_score_delta:
            rows.append(_reject(row, "ambiguous_candidate"))
            continue

        best_seq = str(best["seq"])
        known_changed_rounds = [
            round_idx
            for round_idx, (observed, decoded) in enumerate(zip(wta_seq, best_seq))
            if observed not in unknown_chars and observed != decoded
        ]
        unknown_rounds = [
            round_idx for round_idx, observed in enumerate(wta_seq) if observed in unknown_chars
        ]
        total_edits = len(known_changed_rounds) + len(unknown_rounds)
        if total_edits > max_hamming:
            rows.append(_reject(row, "too_many_edits"))
            continue

        corrected_margin = np.nan
        if known_changed_rounds:
            corrected_margin = float(
                max(
                    diagnostics.iloc[spot_idx][f"round{round_idx}_margin"]
                    for round_idx in known_changed_rounds
                )
            )
            if (
                min_corrected_round_margin is not None
                and corrected_margin > min_corrected_round_margin
            ):
                rows.append(_reject(row, "corrected_round_margin_too_high"))
                continue

        changed_rounds = known_changed_rounds
        correction_penalty = 0.0
        if changed_rounds:
            best_changed_score = _sequence_score(
                probs_for_spot, best_seq, eps=1e-12, rounds=changed_rounds
            )
            wta_changed_score = _sequence_score(
                probs_for_spot, wta_seq, eps=1e-12, rounds=changed_rounds
            )
            correction_penalty = best_changed_score - wta_changed_score
            if correction_penalty > max_correction_penalty:
                rows.append(_reject(row, "correction_penalty_too_high"))
                continue

        geomean_prob = float(best["geomean_prob"])
        if geomean_prob < min_geomean_prob:
            rows.append(_reject(row, "geomean_prob_too_low"))
            continue

        corrected_rounds = sorted(set(known_changed_rounds + unknown_rounds))
        call_type = "rescued_unknown" if unknown_rounds else f"rescued_h{len(known_changed_rounds)}"
        row.update(
            {
                "decoded_seq": best_seq,
                "gene": seq_to_gene[best_seq],
                "call_type": call_type,
                "hamming_to_wta": len(known_changed_rounds),
                "corrected_rounds": ",".join(str(idx) for idx in corrected_rounds),
                "score": float(best["score"]),
                "score_delta": score_delta,
                "geomean_prob": geomean_prob,
                "corrected_round_margin": corrected_margin,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


__all__ = [
    "OUTPUT_COLUMNS",
    "build_one_error_index",
    "candidate_sequences",
    "channel_probabilities",
    "decode_codebook_aware",
    "score_candidates",
    "wta_color_sequences",
]
