"""Two-base color-space encoding and decoding for STARmap barcodes.

Reference: src/matlab/EncodeBases.m, src/matlab/DecodeCS.m

STARmap uses a differential encoding where each color represents the
transition between two consecutive DNA bases. A sliding 2-base window
maps base pairs to colors 1-4.
"""

# Forward lookup: base pair -> color
#: Ordered nucleotide pair to STARmap color string (1..4).
_BASE_PAIR_TO_COLOR = {
    "AA": "1", "CC": "1", "GG": "1", "TT": "1",
    "AC": "2", "CA": "2", "GT": "2", "TG": "2",
    "AG": "3", "CT": "3", "GA": "3", "TC": "3",
    "AT": "4", "CG": "4", "GC": "4", "TA": "4",
}

# Reverse lookup: color -> candidate base pairs
#: STARmap color string (1..4) to compatible nucleotide-pair strings.
_COLOR_TO_BASE_PAIRS = {
    "1": ["AA", "CC", "GG", "TT"],
    "2": ["AC", "CA", "GT", "TG"],
    "3": ["AG", "CT", "GA", "TC"],
    "4": ["AT", "CG", "GC", "TA"],
}

# Color to 0-based channel index
#: STARmap color string (1..4) to zero-based array channel index (0..3).
_COLOR_TO_CHANNEL = {"1": 0, "2": 1, "3": 2, "4": 3}


def encode_bases(sequence: str) -> str:
    """Encode a DNA sequence to a color-space sequence.

    Pure encoding with no reversal. Applies a sliding 2-base window
    and maps each pair to a color digit via _BASE_PAIR_TO_COLOR.

    Parameters
    ----------
    sequence : str
        DNA sequence (e.g., "CGCAC"). Must be at least 2 characters.

    Returns
    -------
    str
        Color sequence of length len(sequence) - 1 (e.g., "4422").

    Examples
    --------
    >>> encode_bases("CGCAC")
    '4422'

    Raises
    ------
    KeyError
        A pair contains characters outside uppercase A/C/G/T.

    Notes
    -----
    Inputs shorter than two bases return an empty string; case is not normalized.
    """
    colors = []
    for i in range(len(sequence) - 1):
        pair = sequence[i : i + 2]
        colors.append(_BASE_PAIR_TO_COLOR[pair])
    return "".join(colors)


def decode_color_sequence(color_seq: str, start_base: str) -> str:
    """Decode a color sequence back to a DNA barcode.

    Uses chain tracking: given the start base, each color digit constrains
    the next base pair (must start with the previous base). This makes
    decoding deterministic — exactly one candidate pair matches at each step.

    Parameters
    ----------
    color_seq : str
        Color sequence (e.g., "4422"). Each character must be "1"-"4".
    start_base : str
        Known first base of the barcode (e.g., "C").

    Returns
    -------
    str
        Decoded DNA barcode (e.g., "CGCAC"). Length = len(color_seq) + 1.

    Examples
    --------
    >>> decode_color_sequence("4422", "C")
    'CGCAC'

    Raises
    ------
    KeyError
        A color is not in 1..4 (including M/N).
    IndexError
        The initial base does not match any candidate for a nonempty input.

    Notes
    -----
    Empty color_seq returns an empty string, not start_base.
    """
    barcode = ""
    ref_base = start_base

    for j, color in enumerate(color_seq):
        candidates = _COLOR_TO_BASE_PAIRS[color]
        for pair in candidates:
            if pair[0] == ref_base:
                if j == 0:
                    barcode = pair
                else:
                    barcode += pair[1]
                break
        ref_base = barcode[-1]

    return barcode
