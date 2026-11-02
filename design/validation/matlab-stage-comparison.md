# Validation design: Python and MATLAB stage comparison on a literal fixture

% Status: Proposed (W-212, 2026-11-02, on dev at f35abad)

This page fixes, before any comparison runs, how four Python stages are compared with the
MATLAB functions they replace: spot detection, integer translation registration, extraction
with color calling, and the codebook with read filtering. It fixes the fixture, every axis,
index, channel and color mapping, the codebook, each check's tolerance and the sequence that
regenerates the committed MATLAB reference outputs. The default test suite checks Python
against those references without MATLAB installed.

The comparison is adapted from the "Frozen comparison protocol" of W-171
(`docs/foundation-qualification.md` on `codex/chapter-ii-m1-m2` at `17e8dda`). Its storage,
export and reload sections are dropped: W-199 tests the checkpoints and W-170 owns the exports.

## What this design is and is not

It checks that, on two isolated integer peaks and a known integer shift, the coordinates,
corrections, calls and genes of the Python stages line up with those of MATLAB once the
mappings below are applied. It does not claim that the Python and MATLAB algorithms are
equivalent in general, it uses no real data, and it changes no MATLAB function, config key or
file and coordinate contract. The tolerances measure software precision
(`docs/contributing.md`, "Expected values and tolerances"), not accuracy. W-649's U9 applies
this protocol and its tool to Position400 afterwards; that comparison is outside this page.

## The fixture

`matlab-stage-comparison-v1` uses no random numbers and no TIFF. It is built by `fixture(depth)`
in `src/python/scripts/matlab_stage_comparison.py` at two depths, Z=5 and Z=1:

* Python images are `uint16` ZYXC of shape `(Z, 9, 11, 4)`, all pixels zero except those below.
* Rounds, in acquisition order: `round10`, `round2`. Channels, in image order: `ch02`, `ch00`,
  `ch03`, `ch01`. Neither order is sorted, so a code path that sorts labels fails.
* Calibration is unknown; coordinates are voxel indices in frame `stage-comparison/reference`.
* Centers, zero-based ZYX: A = `(Z//2, 2, 3)`, B = `(Z//2, 6, 7)`, zero = `(0, 0, 0)`,
  tie = `(Z//2, 4, 5)`.
* Detection image (round `round10`): A has 7 in channel 1 and B has 11 in channel 0.
* Extraction image of `round10`: the detection image plus 5 in channels 0 and 1 at the tie.
  The tie is only in the extraction input, so it cannot change the detected population.
* Image of `round2`: A has 9 in channel 0, B has 13 in channel 1, the tie has 5 in channels 0
  and 1.
* Registration: the reference is the detection image summed over channels as `float64` ZYX; the
  moving image is the reference rolled by `+1` along Y and `-1` along X. No signal crosses a
  border, so the roll loses nothing.

The fixture's SHA-256 is computed over each array's name, dtype, shape and C-order bytes, in a
fixed order, for both depths. The references record it and the test recomputes it.

## Mappings

(matlab-stage-mapping)=

| Quantity | MATLAB form | Python form | Mapping applied before comparing |
| --- | --- | --- | --- |
| image axes | `(row, column, Z, C)`, row = Y, column = X | `(Z, Y, X, C)` | MATLAB array = Python array transposed `(1, 2, 0, 3)`; a 3-D MATLAB image back to Python: transpose `(2, 0, 1)` |
| Z=1 images | `(row, column)` after MATLAB drops the trailing singleton axis | `(1, Y, X)` | add the Z axis in front |
| point coordinates | one-based `(x, y, z)` | zero-based `(z, y, x)` | Python = MATLAB reversed, minus 1 |
| registration correction | `params.shifts` = correction `[row, column, z]`; Z=1: `[row, column]`, Z correction 0 | `displacement_zyx` of the pull map; correction = minus displacement | correction ZYX = `shifts[[3, 1, 2]]` (one-based positions) |
| detection channel | one-based channel of the unreordered detection image | zero-based `channel` | Python = MATLAB minus 1 |
| extraction channels | input channels reordered `[2 1 4 3]`; the winner `m` is one-based in that order | channel index of the image | winner `m` means Python channel `[2 1 4 3](m) - 1` |
| color symbols | winner `m` is color symbol `m` | `color_to_channel` = `{1: 1, 2: 0, 3: 3, 4: 2}` | the reorder above makes MATLAB winner `m` the Python channel of color `m` |
| color sequence | `round10` call followed by `round2` call, string concatenation | `observed_color_sequence` | equal strings |
| no call | `"M"` with score `Inf` (zero or tied neighbourhood) | `call_status` `no_signal` (zero) or `ambiguous` (tie) | each side checked against its own expectation; the statuses are not treated as equal |

The codebook is `gene-A` = `12` and `gene-B` = `21`. In MATLAB it is
`dictionary([EncodeBases("AAC") EncodeBases("ACC")], ["gene-A" "gene-B"])`, so the two codes are
independent two-base encodings of `AAC` and `ACC`, without reversal. In Python it is a
`Codebook` with those two color sequences, the rounds and channels above and the
`color_to_channel` above. `FilterReads` is called with end bases `["AC"]`; its terminal-base
statistics are diagnostic only.

## The stages compared

Each stage runs on the same fixture in both languages. Python uses today's package; MATLAB uses
the functions in `src/matlab/` through the entry script `workflow/scripts/compare_matlab_stages.m`.

* **Detection.** Python: `find_spots` with `LocalMaximaConfig("adaptive", 0.1)`
  (`min_distance_voxels=1`, `exclude_border=True`) on the detection image. MATLAB:
  `SpotFindingMax3D(image, "adaptive", 0.1)` on the same image in YXZC.
* **Registration.** Python: `estimate_transform` with `TranslationConfig()` and
  `apply_transform` with the returned application config. MATLAB: `DFTRegister3D` then
  `DFTApply3D`; at Z=1 `DFTRegister2D` then `DFTApply2D`.
* **Extraction and color calling.** Python: `extract_intensities` with
  `NeighborhoodSumConfig(neighborhood_radius_zyx=(r, r, r))` on the four supplied centers, then
  `decode_reads` with `WtaDecoderConfig()`. MATLAB: `ExtractFromLocation` per round on the
  reordered image with one-based `x, y, z` and `voxel_size = [r r r]`. Both radii `r = 0` and
  `r = 1` run.
* **Codebook and filtering.** Python: `filter_reads` with the default `ReadFilterConfig()`.
  MATLAB: `FilterReads` on A and B only, with the calls of the same radius. MATLAB's
  `DecodeCS` does not accept `M`, so the zero and tie reads are not passed to `FilterReads`;
  Python keeps them as rejected rows. `EncodeBases("AAC")`, `EncodeBases("ACC")` and Python's
  `bases_to_colors` give `12` and `21`.

## Checks and tolerances

(matlab-stage-checks)=

| Check | Depth | What is compared | Comparison |
| --- | --- | --- | --- |
| D1 | 5 | Python and MATLAB detections, each against the literal centers A and B | a one-to-one match within Euclidean distance `1e-9` voxel; no extra or duplicate detection; exact channel, A 1 and B 0 |
| D2 | 1 | MATLAB detection | not compared: `SpotFindingMax3D` and `regionprops3` are volumetric; the Python detection is still checked against the literal centers |
| R1 | 5, 1 | correction ZYX of Python and of MATLAB | exact `(0, -1, +1)` |
| R2 | 5, 1 | the full registered image of MATLAB and of Python, borders included, against the literal reference, and against each other | `atol=1e-12, rtol=0` |
| E1 | 5, 1; r 0, 1 | Python neighbourhood sums against the literal values | exact |
| E2 | 5, 1; r 0, 1 | calls: A `12`, B `21` on both sides; MATLAB zero and tie `MM` with score `Inf`; Python zero `no_signal`, tie `ambiguous` | exact |
| E3 | 5, 1; r 0, 1 | summed score of A and B: MATLAB's per-round scores summed, Python's `wta_l2_nll`, and the analytic `log1p(1e-6/7) + log1p(1e-6/9)` (A) and `log1p(1e-6/11) + log1p(1e-6/13)` (B); each MATLAB per-round score against its term `log1p(1e-6/a)` | `atol=1e-12, rtol=0` |
| F1 | 5, 1; r 0, 1 | MATLAB `goodSpots` and Python accepted reads | both molecules kept with exact genes, A `gene-A` and B `gene-B`; MATLAB rows matched to A and B by coordinate within `1e-9`; no other row |
| F2 | 5, 1 | `EncodeBases` and `bases_to_colors` of `AAC` and `ACC` | exact `12`, `21` |
| N1 | 5 | the comparison with one mapping broken: axes swapped, a one-voxel index offset, or the extraction channel order not applied | the comparison fails |

Matching is by coordinate and never by row position or a fabricated identifier. `1e-12` bounds
double-precision FFT roundoff on amplitudes of at most 11; it is not a registration error. The
integer sums are exact in MATLAB's `single` intermediate.

## The committed references

The references are small JSON files under `src/python/test/data/matlab_reference/`:
`provenance.json`, `z5.json` and `z1.json`. `z5.json` and `z1.json` hold MATLAB's outputs in
MATLAB's own form (one-based XYZ, YXZ images, reordered-channel winners); the mappings above are
applied when comparing, so that a broken mapping fails. Scores of `Inf` are written as the
string `"Inf"`. `provenance.json` records the MATLAB release and toolboxes (`ver`), the licence
checkouts, the thread count, the launcher (`matlab_launcher: path`), the exact MATLAB command,
the fixture SHA-256, this page's SHA-256 at the time of `prepare`, the source hashes and the
SHA-256 of each raw MATLAB output. Hand-written or source-inferred references are not accepted:
the files come only from `export-reference` on a real MATLAB run.

The default-tier test `src/python/test/test_matlab_stage_comparison.py` runs the Python stages
on the fixture and applies every check above to the references; it needs no MATLAB.
`src/python/test/test_matlab_invocation.py` checks, with `subprocess` mocked, that the entry
script is started through `run_matlab_scripts` and `workflow/rules/matlab_launcher.py`.

## Regeneration sequence

Run from `src/python`, with a new external directory, on a host with MATLAB R2023b, the Image
Processing and Statistics and Machine Learning toolboxes and a licence:

1. `uv run python scripts/matlab_stage_comparison.py prepare <dir>` writes the fixture as
   MATLAB inputs, the entry script's config, a Snakefile that includes
   `workflow/rules/common.smk` and calls `run_matlab_scripts`, and `frozen.json` (fixture, page
   and source hashes).
2. `STARFINDER_MATLAB_EXECUTABLE=<matlab> uv run snakemake --snakefile <dir>/Snakefile --cores 1
   --directory <dir>` runs `compare_matlab_stages` once through the launcher with
   `matlab_launcher: path` and `matlab_single_thread: true`. It writes the exact command
   (`matlab-command.json`), MATLAB's `ver` output and the raw outputs into `<dir>`.
3. `uv run python scripts/matlab_stage_comparison.py export-reference <dir>` checks the frozen
   hashes and the MATLAB runtime, then writes the three reference files. It writes nothing else
   in the repository.
4. `uv run python scripts/matlab_stage_comparison.py compare <dir>` runs the comparison against
   the committed references and writes `<dir>/comparison.json`; the default test runs the same
   comparison.

Rerunning the sequence changes only the reference files. A missing MATLAB output is an error,
never a skipped success. A change of the fixture changes its hash, and the test then fails until
the references are regenerated.
