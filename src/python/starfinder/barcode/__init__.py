"""Validated codebooks and independent intensity extraction, decoding and filtering."""

from ._encoding import encode_bases, decode_color_sequence
from ._layout import BarcodeLayout, Segment
from .codebook import ENCODINGS as _ENCODINGS
from .codebook import (
    Codebook,
    EncodingConfig,
    EncodingSpec,
    OneBaseEncodingConfig,
    load_codebook,
)
from .extraction import NeighborhoodSumConfig, IntensityExtractionResult, extract_intensities
from .decoding import DECODING_METHODS as _DECODING_METHODS
from .decoding import (
    WtaDecoderConfig,
    CodebookAwareDecoderConfig,
    BarcodeDecodingResult,
    DecodingSpec,
    InvalidIntensityError,
    decode_barcodes,
)
from .filtering import ReadFilterConfig, ReadFilteringResult, filter_reads

#: The barcode encoding registry, mapping each exact frozen encoding config type to its
#: EncodingSpec (two_base, one_base). Codebook, load_codebook, decode_barcodes and the
#: workflow adapter derive their encoding sets from it.
ENCODINGS = _ENCODINGS

#: The decoder registry, mapping each exact frozen decoder config type to its DecodingSpec
#: (wta, codebook_aware). decode_barcodes, PipelineConfig.decoding, the workflow adapter and
#: the checkpoint reader derive their decoder sets from it.
DECODING_METHODS = _DECODING_METHODS

__all__ = [
    "BarcodeDecodingResult",
    "BarcodeLayout",
    "Codebook",
    "CodebookAwareDecoderConfig",
    "DECODING_METHODS",
    "DecodingSpec",
    "ENCODINGS",
    "EncodingConfig",
    "EncodingSpec",
    "IntensityExtractionResult",
    "InvalidIntensityError",
    "NeighborhoodSumConfig",
    "OneBaseEncodingConfig",
    "ReadFilterConfig",
    "ReadFilteringResult",
    "Segment",
    "WtaDecoderConfig",
    "decode_barcodes",
    "decode_color_sequence",
    "encode_bases",
    "extract_intensities",
    "filter_reads",
    "load_codebook",
]
