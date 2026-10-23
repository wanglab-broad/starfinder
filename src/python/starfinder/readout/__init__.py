"""Validated codebooks and independent intensity extraction, decoding, scoring, deduplication and filtering."""

from ._encoding import bases_to_colors, colors_to_bases
from ._layout import BarcodeLayout, Segment
from .codebook import ENCODINGS as _ENCODINGS
from .codebook import (
    Codebook,
    EncodingConfig,
    EncodingSpec,
    OneBaseEncodingConfig,
    load_codebook,
)
from .extraction import (
    LocalBackgroundConfig,
    NeighborhoodSumConfig,
    IntensityExtractionResult,
    extract_intensities,
)
from ._direct import DirectDecoderConfig, DirectPanel, load_direct_panel
from .decoding import DECODING_METHODS as _DECODING_METHODS
from .decoding import (
    WtaDecoderConfig,
    CodebookAwareDecoderConfig,
    ReadDecodingResult,
    DecodingSpec,
    InvalidIntensityError,
    decode_direct,
    decode_reads,
)
from .scoring import ReadScoreConfig, ReadScoringResult, score_reads
from .deduplication import DeduplicationConfig, ReadDeduplicationResult, deduplicate_reads
from .filtering import ReadFilterConfig, ReadFilteringResult, filter_reads
from .spots import export_spots
from .diagnostics import explain_read, inspect_read, plot_read, summarize_reads

#: The barcode encoding registry, mapping each exact frozen encoding config type to its
#: EncodingSpec (two_base, one_base). Codebook, load_codebook, decode_reads and the
#: workflow adapter derive their encoding sets from it.
ENCODINGS = _ENCODINGS

#: The decoder registry, mapping each exact frozen decoder config type to its DecodingSpec
#: (wta, codebook_aware for readout mode multiplexed; direct for readout mode direct).
#: decode_reads, decode_direct, PipelineConfig.decoding, FOV.run, the workflow adapter and
#: the checkpoint reader derive their decoder sets and mode checks from it.
DECODING_METHODS = _DECODING_METHODS

__all__ = [
    "BarcodeLayout",
    "Codebook",
    "CodebookAwareDecoderConfig",
    "DECODING_METHODS",
    "DecodingSpec",
    "DeduplicationConfig",
    "DirectDecoderConfig",
    "DirectPanel",
    "ENCODINGS",
    "EncodingConfig",
    "EncodingSpec",
    "IntensityExtractionResult",
    "InvalidIntensityError",
    "LocalBackgroundConfig",
    "NeighborhoodSumConfig",
    "OneBaseEncodingConfig",
    "ReadDecodingResult",
    "ReadDeduplicationResult",
    "ReadFilterConfig",
    "ReadFilteringResult",
    "ReadScoreConfig",
    "ReadScoringResult",
    "Segment",
    "WtaDecoderConfig",
    "bases_to_colors",
    "colors_to_bases",
    "decode_direct",
    "decode_reads",
    "deduplicate_reads",
    "explain_read",
    "export_spots",
    "extract_intensities",
    "filter_reads",
    "inspect_read",
    "load_codebook",
    "load_direct_panel",
    "plot_read",
    "score_reads",
    "summarize_reads",
]
