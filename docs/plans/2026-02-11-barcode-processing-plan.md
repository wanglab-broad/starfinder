# Phase 4: Barcode Processing — Implementation Plan

## Context

Phase 3 (Spot Finding & Extraction) is complete. `find_spots_3d()` detects spots and `extract_from_location()` reads per-channel intensities, returning per-round color assignments ("1"–"4", "M", "N"). **Phase 4 completes the spot→gene pipeline** by:

1. Encoding DNA barcodes to color-space sequences (for codebook lookup)
2. Decoding detected color sequences back to DNA (for structural validation)
3. Loading the codebook CSV into a bidirectional lookup
4. Filtering detected reads against the codebook and assigning gene names

The MATLAB implementation spans 4 files (~260 lines). All dependencies (`pandas`, `csv`) are already available.

---

## Scope

**In scope**: Encoding, decoding, codebook loading, reads filtering
**Out of scope**: Preprocessing (Phase 5), multi-round concatenation logic (Phase 6 — Dataset class handles round iteration)

---

## Files to Create / Modify

```
src/python/starfinder/barcode/
├── __init__.py           # UPDATE: add new exports
├── encoding.py           # NEW: encode_bases, decode_color_seq, constants
├── codebook.py           # NEW: load_codebook
├── filtering.py          # NEW: filter_reads
└── extraction.py         # (existing, no changes)

src/python/starfinder/testdata/
└── synthetic.py          # MODIFY: import encoding from barcode.encoding

src/python/test/
├── test_encoding.py      # MODIFY: update imports
├── test_barcode.py       # NEW: tests for decode, codebook, filtering
```

---

## Step 1: Encoding Module

### `barcode/encoding.py`

**Relocate** `BASE_PAIR_TO_COLOR` and encoding logic from `testdata/synthetic.py` to their proper home. Add the decoding function.

**Reference**: `src/matlab/EncodeBases.m` (34 lines) + `src/matlab/DecodeCS.m` (95 lines)

#### Constants (move from `testdata/synthetic.py`)

```python
BASE_PAIR_TO_COLOR = {
    "AA": "1", "CC": "1", "GG": "1", "TT": "1",
    "AC": "2", "CA": "2", "GT": "2", "TG": "2",
    "AG": "3", "CT": "3", "GA": "3", "TC": "3",
    "AT": "4", "CG": "4", "GC": "4", "TA": "4",
}

COLOR_TO_BASE_PAIRS = {
    "1": ["AA", "CC", "GG", "TT"],
    "2": ["AC", "CA", "GT", "TG"],
    "3": ["AG", "CT", "GA", "TC"],
    "4": ["AT", "CG", "GC", "TA"],
}

COLOR_TO_CHANNEL = {"1": 0, "2": 1, "3": 2, "4": 3}
```

#### `encode_bases(sequence: str) -> str`

Pure encoding (no reversal). Sliding 2-base window → color lookup.

- `"CGCAC"` → CG=4, GC=4, CA=2, AC=2 → `"4422"`
- Matches MATLAB `EncodeBases.m` exactly

#### `decode_color_seq(color_seq: str, start_base: str) -> str`

Decode color sequence back to DNA barcode using chain tracking.

**Algorithm** (matches MATLAB `DecodeCS.m` lines 68-93):
1. Set `ref_base = start_base`
2. For each color digit in `color_seq`:
   - Get 4 candidate base pairs from `COLOR_TO_BASE_PAIRS[digit]`
   - Find the pair whose first base == `ref_base`
   - First position: append full pair (2 chars); subsequent: append second char only
   - Update `ref_base` to last character of barcode so far
3. Return decoded barcode string

**Example**: `decode_color_seq("4422", "C")` → `"CGCAC"`

### Update `testdata/synthetic.py`

- Remove `BASE_PAIR_TO_COLOR` and `COLOR_TO_CHANNEL` definitions
- Import them from `barcode.encoding` instead
- Keep `encode_barcode_to_colors()` as a convenience wrapper (reverse + encode)

### Update `test/test_encoding.py`

- Change imports from `starfinder.testdata.synthetic` to `starfinder.barcode.encoding`
- The `encode_barcode_to_colors` tests stay since that function stays in testdata

---

## Step 2: Codebook Module

### `barcode/codebook.py`

**Reference**: `src/matlab/LoadCodebook.m` (30 lines)

```python
def load_codebook(
    path: str | Path,
    do_reverse: bool = True,
    split_index: int | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load codebook CSV and build bidirectional gene↔color_seq lookup.

    Returns (gene_to_seq, seq_to_gene).
    """
```

**Algorithm** (matches MATLAB):
1. Read CSV with columns `gene,barcode` from `path`
2. For each row:
   a. Optionally reverse barcode (`do_reverse=True` by default)
   b. Encode with `encode_bases()` → color sequence
3. If `split_index` is set:
   a. Remove character at that position
   b. Split into front/back at that position
   c. Concatenate back+front (flip)
4. Build `gene_to_seq` and `seq_to_gene` dicts

**Key detail**: `do_reverse=True` means the barcode is reversed **before** encoding. This matches how `testdata/synthetic.py` works and is the standard STARmap convention where sequencing reads the barcode in reverse.

**CSV format** (from `tests/fixtures/synthetic/mini/codebook.csv`):
```
gene,barcode
GeneA,CACGC
GeneB,CATGC
```

---

## Step 3: Filtering Module

### `barcode/filtering.py`

**Reference**: `src/matlab/FilterReads.m` (63 lines)

```python
def filter_reads(
    spots: pd.DataFrame,
    gene_to_seq: dict[str, str],
    seq_to_gene: dict[str, str],
    end_bases: str | None = None,
    start_base: str = "C",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Filter spots by codebook matching and assign gene names.

    Returns (good_spots, stats).
    """
```

**Algorithm** (matches MATLAB `FilterReads.m`):
1. Get `color_seq` column from spots DataFrame
2. Check which color sequences exist in `seq_to_gene` → `in_codebook` boolean mask
3. **Diagnostic only** (if `end_bases` provided):
   a. Decode color sequences to DNA barcodes via `decode_color_seq(seq, start_base)`
   b. Check start/end base patterns (e.g., "CC" → barcode starts with C, ends with C)
   c. Compute % with correct form, % of correct-form reads in codebook
4. Filter spots by `in_codebook` mask
5. Assign `gene` column from `seq_to_gene` lookup
6. Return filtered DataFrame + stats dict

**Stats dict keys**: `"in_codebook"`, `"correct_form"` (if end_bases given), `"validated"` (if end_bases given)

**Critical MATLAB behavior**: The actual filtering is **only** by codebook membership. End-base validation is diagnostic (printed stats), not a filter. This matches the MATLAB comment on line 4: "reads are actually filtered by whether they are in the codebook."

---

## Step 4: Package Wiring

### Update `barcode/__init__.py`

```python
from starfinder.barcode.codebook import load_codebook
from starfinder.barcode.encoding import (
    BASE_PAIR_TO_COLOR, COLOR_TO_CHANNEL, encode_bases, decode_color_seq,
)
from starfinder.barcode.extraction import extract_from_location
from starfinder.barcode.filtering import filter_reads
```

### Update `starfinder/__init__.py`

Add `load_codebook`, `filter_reads` to top-level exports.

---

## Step 5: Tests

### `test/test_barcode.py`

| Test | What it checks |
|------|---------------|
| **Encoding** | |
| `test_encode_bases_known` | `encode_bases("CGCAC")` → `"4422"` (no reversal) |
| `test_encode_bases_length` | Output length = input length - 1 |
| **Decoding** | |
| `test_decode_known` | `decode_color_seq("4422", "C")` → `"CGCAC"` |
| `test_encode_decode_roundtrip` | encode(barcode) → decode(result, barcode[0]) → barcode |
| `test_decode_all_codebook_entries` | All 8 mini codebook barcodes roundtrip correctly |
| **Codebook** | |
| `test_load_codebook_mini` | Load mini codebook.csv, check gene_to_seq has 8 entries |
| `test_codebook_gene_lookup` | `gene_to_seq["GeneA"]` == `"4422"` |
| `test_codebook_seq_lookup` | `seq_to_gene["4422"]` == `"GeneA"` |
| **Filtering** | |
| `test_filter_reads_basic` | Spots with valid color_seq get gene assigned |
| `test_filter_reads_invalid` | Spots with non-codebook color_seq are excluded |
| `test_filter_reads_stats` | Stats dict has correct keys and values |
| **End-to-end** | |
| `test_ground_truth_pipeline` | Mini dataset: detect spots → extract colors → filter → check genes match ground truth |

### Update `test/test_encoding.py`

Update import paths: `from starfinder.barcode.encoding import BASE_PAIR_TO_COLOR, encode_bases`
Keep `encode_barcode_to_colors` import from `testdata.synthetic` (it stays there as convenience wrapper).

---

## Verification

```bash
# Run barcode tests
uv run pytest test/test_barcode.py -v

# Run encoding tests (updated imports)
uv run pytest test/test_encoding.py -v

# Run full suite
uv run pytest test/ -v
```

**Success criteria**:
- All tests pass (existing + new)
- Encode→decode roundtrip works for all 8 codebook entries
- Mini dataset end-to-end: spots → extraction → filtering → genes match ground truth
- No import breakage from moving encoding constants

---

## Key Design Decisions

1. **Move encoding table to `barcode/encoding.py`**: The two-base encoding is fundamentally a barcode concept, not a test-data concept. `testdata/synthetic.py` will import from `barcode.encoding`.

2. **`encode_bases()` has no reversal**: Pure encoding matching MATLAB `EncodeBases.m`. The reversal is a codebook-loading concern (controlled by `do_reverse` in `load_codebook()`). `testdata.encode_barcode_to_colors()` remains as a convenience that reverses then encodes.

3. **Filtering is codebook-only**: End-base validation is diagnostic, not filtering — matching MATLAB behavior exactly.

4. **Multi-segment support deferred**: `FilterReadsMultiSegment.m` adds complexity for split barcodes. The `split_index` param in `load_codebook()` provides basic support; full multi-segment filtering can be added when needed.

5. **`decode_color_seq` is deterministic**: Given a start base, there's exactly one valid decoding per color sequence (the chain constraint means only one of the 4 candidate pairs matches at each position).
