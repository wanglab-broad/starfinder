"""Barcode processing: encoding, decoding, codebook lookup, and filtering."""

from starfinder.barcode.codebook import load_codebook
from starfinder.barcode.encoding import (
    BASE_PAIR_TO_COLOR,
    COLOR_TO_BASE_PAIRS,
    COLOR_TO_CHANNEL,
    decode_color_seq,
    encode_bases,
)
from starfinder.barcode.extraction import extract_from_location
from starfinder.barcode.filtering import filter_reads

__all__ = [
    "BASE_PAIR_TO_COLOR",
    "COLOR_TO_BASE_PAIRS",
    "COLOR_TO_CHANNEL",
    "decode_color_seq",
    "encode_bases",
    "extract_from_location",
    "filter_reads",
    "load_codebook",
]
