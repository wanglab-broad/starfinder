"""Rerunnable read filtering; rejected identities remain available."""

from collections import Counter
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from ._encoding import decode_color_sequence
from ._layout import colors_have_ends, segment_colors
from .codebook import Codebook, encoding_spec
from .decoding import BarcodeDecodingResult
from .scoring import ReadScoringResult


@dataclass(frozen=True)
class ReadFilterConfig:
    """Explicit status and inclusive named score bounds.

    score_bounds maps a score column to (lower, upper), with None for no bound.
    NaN fails a requested score predicate. Endpoint checks are diagnostic-only
    unless exclude_invalid_endpoints is True. end_bases means first/last base of
    the whole sequence decoded from start_base: the one-segment shortcut. Segment
    ends declared on the codebook layout are checked per segment instead (see
    filter_reads); exclusion needs one of the two.
    """

    call_statuses: tuple[str, ...] = ("assigned",)
    score_bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    end_bases: str | None = None
    start_base: str = "C"
    exclude_invalid_endpoints: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.call_statuses, tuple)
            or len(set(self.call_statuses)) != len(self.call_statuses)
            or any(
                s not in ("assigned", "unmatched", "ambiguous", "no_signal")
                for s in self.call_statuses
            )
        ):
            raise ValueError("invalid call_statuses")
        for key, bounds in self.score_bounds.items():
            if key not in (
                "wta_l2_nll",
                "probability_nll",
                "score_delta",
                "geomean_probability",
                "min_round_margin",
                "corrected_round_margin",
                "mean_total_intensity",
                "hamming_to_wta",
            ):
                raise ValueError(f"unknown score predicate {key}")
            if (
                len(bounds) != 2
                or any(
                    v is not None and (not isinstance(v, (float, int)) or np.isnan(v))
                    for v in bounds
                )
                or (None not in bounds and bounds[0] > bounds[1])
            ):
                raise ValueError("invalid score bounds")
        if self.start_base not in ("A", "C", "G", "T") or (
            self.end_bases is not None
            and (len(self.end_bases) != 2 or any(c not in "ACGT" for c in self.end_bases))
        ):
            raise ValueError("invalid endpoint bases")
        if type(self.exclude_invalid_endpoints) is not bool:
            raise ValueError("exclude_invalid_endpoints must be Boolean")


@dataclass(frozen=True)
class ReadFilteringResult:
    """Complete annotated table, accepted view and nullable summary fractions.

    The repr is a one-line accepted fraction and rejection-reason count. It
    never reports precision or accuracy, which require truth.
    """

    table: pd.DataFrame
    spot_namespace: str
    config: ReadFilterConfig
    counts: dict[str, int]
    fractions: dict[str, float | None]
    diagnostics: dict

    def _summary(self):
        fraction = self.fractions.get("accepted")
        text = (
            f"{self.counts['accepted']} accepted / {self.counts['total']} "
            f"({'undefined' if fraction is None else f'{fraction:.1%}'}), "
            f"rejected {self.counts['rejected']}"
        )
        reasons = Counter(
            reason
            for joined in self.table.rejection_reasons.dropna()
            for reason in joined.split(";")
            if reason
        )
        if reasons:
            text += " — " + ", ".join(f"{r} {n}" for r, n in reasons.most_common())
        return text

    def __repr__(self):
        return f"ReadFilteringResult: {self._summary()}"

    @property
    def accepted(self) -> pd.DataFrame:
        """Copy of accepted rows retaining original identities and score columns."""
        return self.table.loc[self.table.accepted].copy()


def filter_reads(
    decoding_result: BarcodeDecodingResult | ReadScoringResult,
    *,
    config: ReadFilterConfig = ReadFilterConfig(),
    codebook: Codebook | None = None,
) -> ReadFilteringResult:
    """Apply independent predicates without extracting or decoding again.

    With a codebook whose layout declares segment ends, each read's observed
    colors are cut into the layout's segments and each segment with ends is
    checked on its own: decoded from each allowed first base, it must end in the
    paired last base (endpoint_valid_<segment>; endpoint_valid when all pass).
    Reads with M or N colors fail. config.end_bases (the one-segment shortcut)
    cannot be combined with layout ends. Membership never depends on the check.
    A ReadScoringResult (score_reads) is filtered like its decoding result, and
    its score columns are kept in the table.
    """
    if not isinstance(decoding_result, (BarcodeDecodingResult, ReadScoringResult)) or not isinstance(
        config, ReadFilterConfig
    ):
        raise TypeError("expected BarcodeDecodingResult or ReadScoringResult and ReadFilterConfig")
    if codebook is not None and not isinstance(codebook, Codebook):
        raise TypeError("codebook must be Codebook or None")
    decoding_result.__post_init__()
    config.__post_init__()
    segments = ()
    if codebook is not None:
        codebook.__post_init__()
        if codebook.round_labels != decoding_result.round_labels:
            raise ValueError("codebook and decoding round labels must match exactly")
        segments = tuple(s for s in codebook.layout.segments if s.ends)
    if segments and config.end_bases is not None:
        raise ValueError("end_bases is the one-segment shortcut; the codebook layout already declares "
                         "segment ends")
    if config.exclude_invalid_endpoints and config.end_bases is None and not segments:
        raise ValueError("endpoint exclusion requires end_bases or segment ends on the codebook layout")
    table = decoding_result.table.copy()
    reasons = [[] for _ in range(len(table))]

    def reject(mask, reason):
        for i in np.flatnonzero(np.asarray(mask)):
            reasons[i].append(reason)

    reject(~table.call_status.isin(config.call_statuses), "call_status")
    for name, (lo, hi) in config.score_bounds.items():
        if name not in table:
            raise ValueError(f"score {name!r} unavailable for this decoder")
        value = table[name]
        passed = value.notna()
        if lo is not None:
            passed &= value >= lo
        if hi is not None:
            passed &= value <= hi
        reject(~passed, f"score:{name}")
    diagnostics = {}
    if config.end_bases is not None:

        def endpoint(seq):
            if pd.isna(seq) or not seq or any(c not in "1234" for c in seq):
                return False
            bases = decode_color_sequence(seq, config.start_base)
            return bool(bases and bases[0] + bases[-1] == config.end_bases)

        table["endpoint_valid"] = table.observed_color_sequence.map(endpoint).astype(bool)
    elif segments:
        spec = encoding_spec(codebook.encoding)
        rounds = len(codebook.round_labels)

        def segment_valid(seq, segment):
            if pd.isna(seq) or len(seq) != rounds:
                return False
            colors = segment_colors(seq, codebook.layout, spec)[segment.name]
            return colors_have_ends(colors, segment.ends, spec, codebook.encoding)

        valid = pd.Series(True, index=table.index)
        diagnostics["endpoint_valid_segment_counts"] = {}
        for segment in segments:
            column = f"endpoint_valid_{segment.name}"
            table[column] = table.observed_color_sequence.map(
                lambda seq: segment_valid(seq, segment)).astype(bool)
            diagnostics["endpoint_valid_segment_counts"][segment.name] = int(table[column].sum())
            valid &= table[column]
        table["endpoint_valid"] = valid.astype(bool)
    if "endpoint_valid" in table:
        diagnostics["endpoint_valid_count"] = int(table.endpoint_valid.sum())
        if config.exclude_invalid_endpoints:
            reject(~table.endpoint_valid, "endpoint")
    table["accepted"] = pd.Series([not r for r in reasons], index=table.index, dtype=bool)
    table["rejection_reasons"] = pd.Series(
        [";".join(r) for r in reasons], index=table.index, dtype="string"
    )
    total, accepted = len(table), int(table.accepted.sum())
    fractions = {"accepted": accepted / total if total else None}
    diagnostics["undefined_fraction_reasons"] = {} if total else {"accepted": "empty_population"}
    if "endpoint_valid" in table:
        fractions["endpoint_valid"] = diagnostics["endpoint_valid_count"] / total if total else None
        if not total:
            diagnostics["undefined_fraction_reasons"]["endpoint_valid"] = "empty_population"
    return ReadFilteringResult(
        table,
        decoding_result.spot_namespace,
        config,
        {"total": total, "accepted": accepted, "rejected": total - accepted},
        fractions,
        diagnostics,
    )
