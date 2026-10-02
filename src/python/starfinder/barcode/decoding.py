"""Independent WTA and codebook-aware decoding with retained identities."""

from dataclasses import KW_ONLY, dataclass, field
from numbers import Integral
from typing import Callable
import numpy as np
import pandas as pd

from starfinder._registry import Dependency, check_name, spec_for
from .codebook import Codebook, encoding_spec
from .extraction import IntensityExtractionResult
from ._codebook_aware import (
    _decode_codebook_aware,
    _channel_probabilities,
    _wta_color_sequences,
    _build_one_error_index,
    _candidate_sequences,
    _score_candidates,
)


class InvalidIntensityError(ValueError):
    """Intensity values violate the declared decoding input policy."""


def _policy(config):
    if config.negative_policy not in ("reject", "clip_negative"):
        raise ValueError("negative_policy must be reject or clip_negative")
    if type(config.diagnostics) is not bool:
        raise ValueError("diagnostics must be Boolean")


@dataclass(frozen=True)
class WtaDecoderConfig:
    """Exact WTA calls; score is summed -log(max/(L2+epsilon)) across rounds.

    Normalization is max / (sqrt(sum(squares)) + 1e-6), matching legacy scores.
    Exact maximum ties are ambiguous. A zero-total round yields no_signal.
    """

    negative_policy: str = "reject"
    diagnostics: bool = False
    method: str = field(default="wta", init=False)

    def __post_init__(self):
        _policy(self)


@dataclass(frozen=True)
class CodebookAwareDecoderConfig:
    """Probability-score decoder with exact calls and conservative rescue gates.

    Probabilities use additive 1e-6, candidate scores sum -log(p) with 1e-12
    floor. max_corrected_round_margin is an UPPER bound. Exact calls bypass
    rescue gates. Candidate ties remain ambiguous even if min_score_delta=0.
    """

    max_hamming: int = 1
    max_corrected_round_margin: float | None = 0.20
    min_score_delta: float = 0.25
    min_geomean_probability: float = 0.45
    max_correction_penalty: float = 1.50
    allow_exact: bool = True
    allow_rescue: bool = True
    negative_policy: str = "reject"
    diagnostics: bool = False
    method: str = field(default="codebook_aware", init=False)

    def __post_init__(self):
        _policy(self)
        if (
            isinstance(self.max_hamming, bool)
            or not isinstance(self.max_hamming, Integral)
            or self.max_hamming < 0
        ):
            raise ValueError("max_hamming must be a nonnegative integer")
        for key in (
            "max_corrected_round_margin",
            "min_score_delta",
            "min_geomean_probability",
            "max_correction_penalty",
        ):
            v = getattr(self, key)
            if v is None and key == "max_corrected_round_margin":
                continue
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not np.isfinite(v)
                or v < 0
                or (key in ("max_corrected_round_margin", "min_geomean_probability") and v > 1)
            ):
                raise ValueError(f"invalid {key}")
        if type(self.allow_exact) is not bool or type(self.allow_rescue) is not bool:
            raise ValueError("allow_exact/allow_rescue must be Boolean")


@dataclass(frozen=True)
class BarcodeDecodingResult:
    """One row per identity, labeled axes, effective config and optional diagnostics.

    table carries observed/decoded_color_sequence, nullable gene_id, call_status,
    failure_reason, and method-specific named scores. Diagnostics expose
    probabilities (N,C,R), per_round and candidates tables only when requested.
    The repr is a one-line read count by call_status without table rows.
    """

    table: pd.DataFrame
    spot_namespace: str
    channel_labels: tuple[str, ...]
    round_labels: tuple[str, ...]
    config: WtaDecoderConfig | CodebookAwareDecoderConfig
    diagnostics: dict

    def __post_init__(self):
        required = {
            "spot_id",
            "spot_namespace",
            "observed_color_sequence",
            "decoded_color_sequence",
            "gene_id",
            "call_status",
            "failure_reason",
        }
        if not required.issubset(self.table) or not self.table.columns.is_unique:
            raise ValueError("invalid decoding table schema")
        if (
            not isinstance(self.table.spot_id.dtype, pd.StringDtype)
            or self.table.spot_id.isna().any()
            or not self.table.spot_id.is_unique
            or (self.table.spot_id.str.len() == 0).any()
        ):
            raise ValueError("decoding spot IDs must be unique nonempty strings")
        if not self.table.spot_namespace.eq(self.spot_namespace).all():
            raise ValueError("decoding namespace mismatch")
        if not self.table.call_status.isin(
            ["assigned", "unmatched", "ambiguous", "no_signal"]
        ).all():
            raise ValueError("invalid call_status")

    def _summary(self):
        counts = self.table.call_status.value_counts()
        statuses = [
            f"{s} {int(counts[s])}"
            for s in ("assigned", "ambiguous", "no_signal", "unmatched")
            if s in counts.index
        ]
        return f"{len(self.table)} reads" + (" — " + ", ".join(statuses) if statuses else "")

    def __repr__(self):
        return f"BarcodeDecodingResult: {self._summary()}"


def _run_codebook_aware(values, codebook, config, spot_ids, diagnostics):
    """Codebook-aware decoding table (before the shared required-round rules)."""
    table = _decode_codebook_aware(
        values,
        codebook.seq_to_gene,
        spot_ids=spot_ids,
        max_hamming=config.max_hamming,
        min_corrected_round_margin=config.max_corrected_round_margin,
        min_score_delta=config.min_score_delta,
        min_geomean_prob=config.min_geomean_probability,
        max_correction_penalty=config.max_correction_penalty,
        allow_exact=config.allow_exact,
        allow_rescue=config.allow_rescue,
    )
    table = table.rename(
        columns={
            "color_seq_wta": "observed_color_sequence",
            "decoded_seq": "decoded_color_sequence",
            "gene": "gene_id",
            "reject_reason": "failure_reason",
            "score": "probability_nll",
            "geomean_prob": "geomean_probability",
        }
    )
    table["call_status"] = np.where(table.gene_id.notna(), "assigned", "unmatched")
    table.loc[table.failure_reason.eq("ambiguous_candidate"), "call_status"] = "ambiguous"
    # Tied observed rounds that cannot be uniquely rescued stay ambiguous.
    tied = table.observed_color_sequence.str.contains("M", na=False) & table.gene_id.isna()
    table.loc[tied, "call_status"] = "ambiguous"
    return table


def _run_wta(values, codebook, config, spot_ids, diagnostics):
    """WTA decoding table (before the shared required-round rules)."""
    norm = np.sqrt((values**2).sum(axis=1)) + 1e-6
    maximum = values.max(axis=1)
    ties = (values == maximum[:, None, :]).sum(axis=1) > 1
    colors = np.asarray(list("1234"), dtype=object)[values.argmax(axis=1)]
    colors[ties] = "M"
    observed = np.array(["".join(row) for row in colors], dtype=object)
    with np.errstate(divide="ignore"):
        scores = -np.log(maximum / norm)
    scores[ties] = np.inf
    genes = [codebook.seq_to_gene.get(seq) for seq in observed]
    table = pd.DataFrame(
        {
            "spot_id": spot_ids,
            "observed_color_sequence": observed,
            "decoded_color_sequence": [
                s if g is not None else None for s, g in zip(observed, genes)
            ],
            "gene_id": genes,
            "call_status": ["assigned" if g is not None else "unmatched" for g in genes],
            "failure_reason": ["" if g is not None else "not_in_codebook" for g in genes],
            "wta_l2_nll": scores.sum(axis=1),
            "call_type": ["exact" if g is not None else "no_call" for g in genes],
        }
    )
    table.loc[ties.any(axis=1), ["call_status", "failure_reason"]] = [
        "ambiguous",
        "tied_channels",
    ]
    if config.diagnostics:
        diagnostics["wta_round_l2_nll"] = scores
    return table


@dataclass(frozen=True)
class DecodingSpec:
    """Registered decoder: stable name, private implementation and declared capabilities.

    run(values, codebook, config, spot_ids, diagnostics) receives the clipped
    (N, 4, R) sums in color order 1-4 and returns the decoder's table before the
    shared required-round rules; decode_barcodes calls it after the checks below,
    callers never call it directly. modes are the readout modes the decoder
    supports; encodings the encoding symbol kinds (EncodingSpec.symbols) it
    decodes; rescue whether it can change an observed color; score_columns the
    numeric columns it writes. requires lists optional dependencies.
    """

    name: str
    run: Callable[..., pd.DataFrame]
    _: KW_ONLY
    modes: frozenset[str]
    encodings: frozenset[str]
    rescue: bool
    score_columns: tuple[str, ...]
    requires: tuple[Dependency, ...] = ()

    def __post_init__(self):
        check_name(self.name, "decoding method")
        if not callable(self.run):
            raise TypeError(f"decoding method {self.name!r} run must be callable")
        for name in ("modes", "encodings"):
            value = getattr(self, name)
            if not isinstance(value, frozenset) or any(not isinstance(v, str) or not v for v in value):
                raise TypeError(f"{name} of {self.name!r} must be a frozenset of nonempty strings")
        if not self.modes:
            raise ValueError(f"modes of {self.name!r} must not be empty")
        if not isinstance(self.rescue, bool):
            raise TypeError(f"rescue of {self.name!r} must be Boolean")
        if not isinstance(self.score_columns, tuple) or any(
                not isinstance(c, str) or not c for c in self.score_columns):
            raise TypeError(f"score_columns of {self.name!r} must be a tuple of column names")
        if not isinstance(self.requires, tuple) or not all(isinstance(d, Dependency) for d in self.requires):
            raise TypeError(f"decoding method {self.name!r} requires must be a tuple of Dependency")


#: Decoders by exact config type (docs/readout-contract.md, "What a decoder declares").
DECODING_METHODS: dict[type, DecodingSpec] = {
    WtaDecoderConfig: DecodingSpec(
        "wta", _run_wta, modes=frozenset({"multiplexed"}), encodings=frozenset({"color"}),
        rescue=False, score_columns=("wta_l2_nll",)),
    CodebookAwareDecoderConfig: DecodingSpec(
        "codebook_aware", _run_codebook_aware, modes=frozenset({"multiplexed"}),
        encodings=frozenset({"color"}), rescue=True,
        score_columns=("probability_nll", "score_delta", "geomean_probability", "min_round_margin",
                       "corrected_round_margin", "mean_total_intensity", "hamming_to_wta")),
}


def decode_barcodes(
    intensity_result: IntensityExtractionResult,
    codebook: Codebook,
    *,
    config: WtaDecoderConfig | CodebookAwareDecoderConfig,
) -> BarcodeDecodingResult:
    """Decode without changing extraction; reject invalid labels/values explicitly.

    The decoder is looked up in DECODING_METHODS by exact config type; a codebook
    whose encoding kind (EncodingSpec.symbols) the decoder does not declare raises
    TypeError. Any unavailable or zero-total round prevents assignment. WTA ties
    use exact equality; probability ties use absolute tolerance 1e-12, preserving
    the two methods' historical tie detection. A uniquely supported codebook
    rescue may resolve an observed tie; equally scored candidates never assign a
    gene. The table reports the decoded entry_id beside its gene_id.
    """
    if not isinstance(intensity_result, IntensityExtractionResult) or not isinstance(
        codebook, Codebook
    ):
        raise TypeError("expected IntensityExtractionResult and Codebook")
    spec = spec_for(DECODING_METHODS, config, "decoding method", TypeError, "unsupported decoder config")
    intensity_result.__post_init__()
    codebook.__post_init__()
    config.__post_init__()
    encoding = encoding_spec(codebook.encoding)
    if encoding.symbols not in spec.encodings:
        raise TypeError(
            f"decoder {spec.name!r} does not decode {encoding.symbols!r} encodings "
            f"(codebook encoding {encoding.name!r}; it decodes {sorted(spec.encodings)})"
        )
    result = intensity_result
    if (
        result.channel_labels != codebook.channel_labels
        or result.round_labels != codebook.round_labels
    ):
        raise ValueError("codebook and intensity channel/round labels must match exactly")
    values = result.values
    negative = values[values < 0]
    if len(negative) and config.negative_policy == "reject":
        raise InvalidIntensityError("negative intensities require explicit clip_negative policy")
    diagnostics = {
        "negative_policy": config.negative_policy,
        "clipped_count": len(negative),
        "clipped_range": (float(negative.min()), float(negative.max())) if len(negative) else None,
    }
    # Numerical mechanics operate in fixed color-symbol order, independent of acquisition order.
    values = np.maximum(values[:, [codebook.color_to_channel[c] for c in "1234"], :], 0)
    probs = _channel_probabilities(values)
    observed, margins = _wta_color_sequences(probs)
    n, _, r = values.shape
    no_signal = (values.sum(axis=1) == 0).any(axis=1)
    invalid = ~result.valid.all(axis=1)
    strings = [
        "spot_id",
        "spot_namespace",
        "observed_color_sequence",
        "decoded_color_sequence",
        "gene_id",
        "entry_id",
        "call_status",
        "failure_reason",
        "call_type",
        "corrected_rounds",
    ]
    table = spec.run(values, codebook, config, result.spot_ids, diagnostics)
    table["spot_namespace"] = result.spot_namespace
    for mask, status, reason in [
        (no_signal, "no_signal", "zero_signal_round"),
        (invalid, "unmatched", "invalid_measurement"),
    ]:
        table.loc[mask, ["call_status", "failure_reason", "call_type"]] = [
            status,
            reason,
            "no_call",
        ]
        table.loc[mask, ["gene_id", "decoded_color_sequence"]] = None
    # Competition and rescue are between entries (color sequences); report the decoded one.
    table.insert(int(table.columns.get_loc("gene_id")) + 1, "entry_id",
                 table.decoded_color_sequence.map(codebook.seq_to_entry))
    for col in strings:
        if col in table:
            table[col] = table[col].astype("string")
    for col in table.columns:
        if col not in strings and col != "gene_wta":
            table[col] = table[col].astype("float64")
    if "gene_wta" in table:
        table["gene_wta"] = table.gene_wta.astype("string")
    if config.diagnostics:
        # Public probabilities follow acquisition channel order, never implicit color order.
        inverse = np.argsort([codebook.color_to_channel[c] for c in "1234"])
        diagnostics["probabilities"] = probs[:, inverse, :]
        per_round = []
        candidates = []
        index = _build_one_error_index(codebook.seq_to_gene, 4, r)
        for i, spot_id in enumerate(result.spot_ids):
            for j, label in enumerate(result.round_labels):
                per_round.append(
                    dict(
                        spot_id=spot_id,
                        spot_namespace=result.spot_namespace,
                        round_label=label,
                        top_color=str(int(margins.iloc[i][f"round{j}_top_channel"])),
                        top_probability=margins.iloc[i][f"round{j}_top_prob"],
                        second_probability=margins.iloc[i][f"round{j}_second_prob"],
                        margin=margins.iloc[i][f"round{j}_margin"],
                    )
                )
            seqs = _candidate_sequences(
                str(observed[i]),
                index,
                codebook.seq_to_gene,
                max_hamming=config.max_hamming
                if isinstance(config, CodebookAwareDecoderConfig)
                else 0,
            )
            scored = _score_candidates(probs[i], seqs)
            for rank, row in enumerate(scored.to_dict("records")):
                candidates.append(
                    dict(
                        spot_id=spot_id,
                        spot_namespace=result.spot_namespace,
                        color_sequence=row["seq"],
                        gene_id=codebook.seq_to_gene[row["seq"]],
                        probability_nll=row["score"],
                        geomean_probability=row["geomean_prob"],
                        rank=rank,
                    )
                )
        diagnostics["per_round"] = pd.DataFrame(
            per_round,
            columns=[
                "spot_id",
                "spot_namespace",
                "round_label",
                "top_color",
                "top_probability",
                "second_probability",
                "margin",
            ],
        ).astype(
            {
                "spot_id": "string",
                "spot_namespace": "string",
                "round_label": "string",
                "top_color": "string",
                "top_probability": "float64",
                "second_probability": "float64",
                "margin": "float64",
            }
        )
        diagnostics["candidates"] = pd.DataFrame(
            candidates,
            columns=[
                "spot_id",
                "spot_namespace",
                "color_sequence",
                "gene_id",
                "probability_nll",
                "geomean_probability",
                "rank",
            ],
        ).astype(
            {
                "spot_id": "string",
                "spot_namespace": "string",
                "color_sequence": "string",
                "gene_id": "string",
                "probability_nll": "float64",
                "geomean_probability": "float64",
                "rank": "int64",
            }
        )
        diagnostics["candidate_order"] = (
            "probability_nll then color_sequence; tied candidates are ambiguous"
        )
    return BarcodeDecodingResult(
        table,
        result.spot_namespace,
        result.channel_labels,
        result.round_labels,
        config,
        diagnostics,
    )
