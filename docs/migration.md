# Python migration guide

This is a coordinated breaking Python cleanup: update maintained imports and
callers together. Replaced interfaces have no compatibility aliases. **FOV stays
FOV.** MATLAB APIs, shared MATLAB-facing workflow keys, CSV coordinate bases and
filenames remain unchanged. Historical scripts and outputs retain their original
provenance; they are not examples of the current API.

## Mechanical names and ownership

| Before | Current owner/interface |
| --- | --- |
| `dataset.STARMapDataset`, `LayerState`, `layers` | `dataset.Dataset`, `RoundState`, `rounds` |
| `dataset.Codebook` | `barcode.Codebook` |
| `io.load_multipage_tiff`, `load_image_stacks`, `save_stack` | `io.load_volume`, `load_round`, `save_volume` |
| `min_max_normalize`, `histogram_match`, `morphological_reconstruction`, `tophat_filter` | `preprocessing.normalize_intensity`, `match_histogram`, `reconstruct_background`, `filter_tophat` |
| `utils.make_projection` | `preprocessing.project_image` |
| `spotfinding.find_spots_3d` | `spot_finding.find_spots` |
| `decode_color_seq` | `barcode.decode_color_sequence` |
| `benchmark.synthetic` | `synthetic` |
| `registration.metrics`, benchmark truth comparisons | Pure domain functions in `evaluation` |
| Separate benchmark drivers and `starfinder-generate` | `starfinder synthetic generate`, `starfinder benchmark run/evaluate/report` |

These relocations do not make the old positional arguments or return values valid.
Use typed operation configs and structured results. See the [full export
inventory](api/inventory.rst) and [coordination member map](coordination.md).
Registration internals, benchmark helpers and barcode lookup constants are private.

## Before and after Python calls

The **before** snippets below are historical text, not runnable imports.

### Load and preprocess

Before:

```text
image = load_multipage_tiff(path)
image = min_max_normalize(image)
```

After:

```python
from starfinder.io import load_volume
from starfinder.preprocessing import normalize_intensity, MinMaxNormalizationConfig

loaded = load_volume(path)
image = normalize_intensity(loaded.image, config=MinMaxNormalizationConfig(
    output_dtype="uint8", output_range=(0, 255)))
metadata = loaded.metadata
```

Loaders preserve dtype by default and return `ImageLoadResult`. Conversion is
explicit; physical calibration is never inferred from shape. See [contracts](api/contracts.md).

### Estimate, then apply registration

Before:

```text
shift = phase_correlate(reference, moving)
registered = apply_shift(moving, -shift)
```

After (given same-grid ZYX arrays):

```python
from starfinder.image import ImageMetadata
from starfinder.registration import TranslationConfig, estimate_transform, apply_transform

result = estimate_transform(reference, moving, config=TranslationConfig(),
    reference_metadata=ImageMetadata("reference"),
    moving_metadata=ImageMetadata("moving"))
registered = apply_transform(moving, result.transform, config=result.application_config)
```

The transform already corrects moving to reference. Do not negate it. Dense
transforms are pull fields, not forward scene perturbations. Estimation errors
are explicit; optional recovery belongs to [coordination](coordination.md).

### Keep extraction, decoding and filtering separate

Before:

```text
calls, scores = extract_from_location(images, locations, ...)
good_reads = filter_reads(calls, ...)
```

After (given labeled rounds, a SpotFindingResult and a validated Codebook):

```python
from starfinder.barcode import (NeighborhoodSumConfig, WtaDecoderConfig,
    ReadFilterConfig, extract_intensities, decode_barcodes, filter_reads)

intensities = extract_intensities(rounds, spots,
    config=NeighborhoodSumConfig(neighborhood_radius_zyx=(1, 2, 2)))
decoded = decode_barcodes(intensities, codebook, config=WtaDecoderConfig())
filtered = filter_reads(decoded, config=ReadFilterConfig())
accepted = filtered.accepted
```

All rows retain `(spot_namespace, spot_id)`, including ambiguous/unmatched calls.
Filtering keeps rejection reasons. Export joins by identity rather than row order.
Rerun filtering without decoding, or decoding without repeating extraction.

### Coordinate a pipeline

Before: `dataset.fov(id).run_streaming(...)` and separate batch stage calls.
After: `dataset.fov(id).run(pipeline, execution=ExecutionConfig("streaming"))`.
`PipelineConfig` specifies scientific stages; `ExecutionConfig` specifies residency.
Use `from_workflow_config(config, rule)` at the shared YAML boundary. Full
construction and recovery examples are in [coordination](coordination.md).

### Generate, evaluate and report

`synthetic.generate_dataset(codebook, config, fov_ids=...)` returns arrays and
truth records, or streams each round to a callback. The CLI persists generated
inputs; benchmark cases explicitly select them.
`evaluation` accepts supplied results/truth and explicit matching/units; it does
not rerun algorithms. [Benchmark](benchmark.md) documents immutable processing
runs, checksum validation, and separate saved-output evaluations/reports.

### Generate synthetic data with one generator

The historical generator (`SyntheticConfig`, integer-center `render_spots`,
hash-seeded registration pairs) is removed. Benchmark data and test fixtures
come from the formed-scene generator, with benchmark presets that keep the
historical shapes, amplicon counts, seeds and shift ranges. There are no aliases.

| Before | After |
| --- | --- |
| `SyntheticConfig`, `get_preset_config(name)` | `benchmark_scene_preset(name, dtype=...)` returns `(Codebook, FormedSceneConfig)`; adjust with `dataclasses.replace`; `SCENE_PRESETS` lists every preset |
| `generate_dataset(config, preset=...)` | `generate_dataset(codebook, config, fov_ids=..., preset=..., on_round=...)` |
| `generate_registration_pairs([preset], seed=...)` | `generate_registration_pair(preset, deformation=..., seed=...)`, one call per pair |
| `generate_displacement_field`, `DEFORMATION_CONFIGS` | `deformation_geometry(name, shape)` with `GeometryConfig` affine, polynomial and RBF terms; `forward_displacement(transform, shape)`. Magnitudes are reduced where needed for invertibility; notably `multi_point` uses 1.3–10.5 px instead of 5–20 px below `tissue` (see the synthetic API) |
| `render_spots`, `generate_volume` | `generate_formed_scene` (single scenes) |
| `generate_codebook(n)` list of `(gene, barcode)` | `generate_codebook(n)` returns a `Codebook` with `base_sequence` |
| `SyntheticDataset.spot_truth`, `perturbations`, `molecular_truth`, `config` | `formed`, `round_truth`, `spot_truth`, per-FOV `provenance`; `historical_truth` keeps the ground_truth.json keys |
| `--dtype uint8` default; registration uint8 only | uint16 default in both modes; `--dtype uint8` scales intensities by 1/16 |

Images change: uint16 with a camera offset, a spatially varying background,
Poisson plus read noise, lognormal brightness, crosstalk and a round trend
replace the constant background 20, Gaussian σ 10 and uint8 200–255 spots.
Positions and shifts are continuous, spots are no longer dropped at the border,
and the 8-gene test codebook gains GeneI–GeneL so every channel is used in
every round. Outputs add `formed.csv` and `round_truth.csv`; registration
pairs add continuous forward fields for every deformation.

### Save and reload pipeline checkpoints

An earlier development branch had an artifact and provenance contract with
contract IDs, reference chains, per-file manifests, immutable directories and
HTML reports. It was never merged, and it has been replaced by three plain
checkpoints and one run record. There are no aliases.

| Development branch | Current interface |
| --- | --- |
| `starfinder.provenance` run recorder and event snapshots | `run.json`, written by `FOV.run(..., checkpoints=CheckpointConfig())` |
| `io.checkpoints`, `FOV.load_image_checkpoint` | `FOV.save_checkpoint("registered")`, `FOV.load_checkpoint("registered")` |
| `io.candidates` combined candidates and signals | `candidates.csv` or `.parquet`, reloaded as `SpotFindingResult` and `IntensityExtractionResult` |
| `io.molecules`, final checkpoints and `MoleculeIndex` | `pre_qc` checkpoint for decoding; final outputs stay the workflow CSVs |
| `reporting` HTML summaries | None; read `run.json` or the checkpoint tables directly |

Images are TIFF through `save_volume`, tables are CSV unless Parquet is
requested, and the location is `<output_root>/checkpoints/<fov_id>/`. Spot
identity, input SHA-256 hashes, atomic writes and reruns of decoding or
filtering without images are kept. See [checkpoints](checkpoints.md).

## Intentional behavior changes — not mechanical equivalence

| Area | Change and consequence |
| --- | --- |
| I/O / preprocessing | Preserve loaded dtype; conversion, cropping and channel selection are explicit. Constant normalization groups map to the lower endpoint even with SNR gating. Float64 computation can change quantization boundaries. Slice morphology avoids uint16 signed overflow; projection preserves singleton Z and uses wider sums without display scaling. |
| Detection | Singleton-Z local maxima operate in YX. Empty tables are typed, identities/geometry explicit. Distinct landmark and pipeline detector policies remain distinct. |
| Translation | Singleton axes return zero; odd-length peak wrapping is corrected. Signed fractional Fourier output uses the real inverse FFT rather than magnitude. Even half-period backend signs and Nyquist behavior are documented rather than hidden. |
| Transform application | Integer output rounds once with nearest-even ties and saturation; floating output retains signed interpolation/overshoot. No silent method fallback; unsupported geometry/backend/dimensions fail explicitly. |
| Barcodes | Validate codebook collisions and label alignment; neighborhoods use explicit ZYX radii and subpixel/boundary policy. Preserve ambiguous/unmatched/rejected identities instead of dropping them. Scores and endpoint filtering have explicit meanings. |
| Coordination | Merged registration images sum in float64. Rectangular subtiles cover both axes and remainders. Batch/streaming honor the same stage flags, unlike legacy forced/omitted stages. |
| Synthetic | One formed-scene generator with keyed SHA-256/PCG64 streams: byte-repeatable across processes for a pinned NumPy build on the same CPU, but every image and truth record differs from the historical generator. Appearance defaults are uncalibrated and do not establish molecular truth. |
| Evaluation | Centered NCC has no epsilon bias. Missing/failed shifts, zero denominators and constant images are undefined rather than zero/passing. Shift errors preserve floats; matching thresholds/policies are explicit. |
| Benchmark / recipes | Failures retain requested/actual method identity. Evaluation/reporting reuse saved artifacts. Optional legacy experiments remain recipes with prerequisites, not validated research results. |

Detailed numerical policies, tolerances and edge cases remain canonical in
[contracts](api/contracts.md), [evaluation](api/evaluation.registration.rst),
[synthetic](api/synthetic.rst) and [benchmark recipes](benchmark-recipes.md).
This guide does not claim bitwise equivalence or MATLAB runtime validation.
Scientific qualification remains separate; notably, molecular truth and
calibration of synthetic appearance remain unresolved.
