"""Rerunnable read filtering; rejected identities remain available."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from ._encoding import decode_color_sequence
from .decoding import BarcodeDecodingResult


@dataclass(frozen=True)
class ReadFilterConfig:
    """Explicit status and inclusive named score bounds.

    score_bounds maps a score column to (lower, upper), with None for no bound.
    NaN fails a requested score predicate. Endpoint checks are diagnostic-only
    unless exclude_invalid_endpoints is True. end_bases means first/last base.
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
        if type(self.exclude_invalid_endpoints) is not bool or (
            self.exclude_invalid_endpoints and self.end_bases is None
        ):
            raise ValueError("endpoint exclusion requires end_bases")


@dataclass(frozen=True)
class ReadFilteringResult:
    """Complete annotated table, accepted view and nullable summary fractions."""

    table: pd.DataFrame
    spot_namespace: str
    config: ReadFilterConfig
    counts: dict[str, int]
    fractions: dict[str, float | None]
    diagnostics: dict

    @property
    def accepted(self) -> pd.DataFrame:
        """Copy of accepted rows retaining original identities and score columns."""
        return self.table.loc[self.table.accepted].copy()


def filter_reads(
    decoding_result: BarcodeDecodingResult, *, config: ReadFilterConfig = ReadFilterConfig()
) -> ReadFilteringResult:
    """Apply independent predicates without extracting or decoding again."""
    if not isinstance(decoding_result, BarcodeDecodingResult) or not isinstance(
        config, ReadFilterConfig
    ):
        raise TypeError("expected BarcodeDecodingResult and ReadFilterConfig")
    decoding_result.__post_init__()
    config.__post_init__()
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
    if config.end_bases is not None:
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
