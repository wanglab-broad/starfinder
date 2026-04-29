"""Barcode processing: encoding, decoding, codebook lookup, and filtering."""

from starfinder.barcode.codebook import load_codebook
from starfinder.barcode.codebook_aware import (
    build_one_error_index,
    candidate_sequences,
    channel_probabilities,
    decode_codebook_aware,
    score_candidates,
    wta_color_sequences,
)
from starfinder.barcode.encoding import (
    BASE_PAIR_TO_COLOR,
    COLOR_TO_BASE_PAIRS,
    COLOR_TO_CHANNEL,
    decode_color_seq,
    encode_bases,
)
from starfinder.barcode.extraction import extract_from_location, extract_intensity_tensor
from starfinder.barcode.filtering import filter_reads

__all__ = [
    "BASE_PAIR_TO_COLOR",
    "COLOR_TO_BASE_PAIRS",
    "COLOR_TO_CHANNEL",
    "build_one_error_index",
    "candidate_sequences",
    "channel_probabilities",
    "decode_color_seq",
    "decode_codebook_aware",
    "encode_bases",
    "extract_from_location",
    "extract_intensity_tensor",
    "filter_reads",
    "load_codebook",
    "score_candidates",
    "wta_color_sequences",
]
