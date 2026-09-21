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

Persistent `FOV.run(..., provenance=RunRecorder(...))` now saves the complete
candidate/signal checkpoint before decoding/QC by default. Prepare the optional
`checkpoint` extra first, or set `RunRecorder(..., save_candidates_signals=False)`
to omit trace persistence while retaining decoded/final outputs. In-memory runs without a recorder do not
infer a destination. Use `io.load_candidate_checkpoint` to rerun existing
decoding/filtering without images; see [checkpoint usage](candidate-checkpoints.md).

### Coordinate a pipeline

Before: `dataset.fov(id).run_streaming(...)` and separate batch stage calls.
After: `dataset.fov(id).run(pipeline, execution=ExecutionConfig("streaming"))`.
`PipelineConfig` specifies scientific stages; `ExecutionConfig` specifies residency.
Use `from_workflow_config(config, rule)` at the shared YAML boundary. Full
construction and recovery examples are in [coordination](coordination.md).

### Generate, evaluate and report

`synthetic.generate_dataset(config)` returns arrays and truth records in memory.
For the versioned clean formed-amplicon model, use
`synthetic.generate_formed_scene(codebook, config=FormedSceneConfig(...))`.
It preserves full stable-ID formed/per-round truth with independent streams;
the historical generator remains available with its historical limitations.
See the [synthetic API](api/synthetic.rst) for the distinct result and kernel
contracts. No historical fixture or hash-derived seed is retroactively qualified.
`render_spots` accepts an identity-bearing scene table instead of integer tuples.
The CLI persists generated inputs; benchmark cases explicitly select them.
`evaluation` accepts supplied results/truth and explicit matching/units; it does
not rerun algorithms. [Benchmark](benchmark.md) documents immutable processing
runs, checksum validation, and separate saved-output evaluations/reports.

## Intentional behavior changes — not mechanical equivalence

| Area | Change and consequence |
| --- | --- |
| I/O / preprocessing (W-137) | Preserve loaded dtype; conversion, cropping and channel selection are explicit. Constant normalization groups map to the lower endpoint even with SNR gating. Float64 computation can change quantization boundaries. Slice morphology avoids uint16 signed overflow; projection preserves singleton Z and uses wider sums without display scaling. |
| Detection (W-138) | Singleton-Z local maxima operate in YX. Empty tables are typed, identities/geometry explicit. Distinct landmark and pipeline detector policies remain distinct. |
| Translation (W-139) | Singleton axes return zero; odd-length peak wrapping is corrected. Signed fractional Fourier output uses the real inverse FFT rather than magnitude. Even half-period backend signs and Nyquist behavior are documented rather than hidden. |
| Transform application (W-140) | Integer output rounds once with nearest-even ties and saturation; floating output retains signed interpolation/overshoot. No silent method fallback; unsupported geometry/backend/dimensions fail explicitly. |
| Barcodes (W-141) | Validate codebook collisions and label alignment; neighborhoods use explicit ZYX radii and subpixel/boundary policy. Preserve ambiguous/unmatched/rejected identities instead of dropping them. Scores and endpoint filtering have explicit meanings. |
| Coordination (W-142) | Merged registration images sum in float64. Rectangular subtiles cover both axes and remainders. Batch/streaming honor the same stage flags, unlike legacy forced/omitted stages. |
| Synthetic (W-143) | Fractional scene centers render analytically. Integer-center rendering and historical randomness remain; namespace extraction does not establish molecular truth or cross-process reproducibility. |
| Evaluation (W-144) | Centered NCC has no epsilon bias. Missing/failed shifts, zero denominators and constant images are undefined rather than zero/passing. Shift errors preserve floats; matching thresholds/policies are explicit. |
| Benchmark / recipes (W-145–W-146) | Failures retain requested/actual method identity. Evaluation/reporting reuse saved artifacts. Optional legacy experiments remain recipes with prerequisites, not validated research results. |

Detailed numerical policies, tolerances and edge cases remain canonical in
[contracts](api/contracts.md), [evaluation](api/evaluation.registration.rst),
[synthetic](api/synthetic.rst) and [benchmark recipes](benchmark-recipes.md).
This guide does not claim bitwise equivalence or MATLAB runtime validation.
W-92/W-93/W-94/W-124 and Chapter II retain scientific qualification; notably,
historical process-dependent synthetic hash seeds and scientific molecular-truth
qualification remain unresolved.

### Explicit formed-scene background and noise controls

Use `BackgroundConfig`, `TextureConfig` and `NoiseConfig` with
`generate_formed_scene` for the frozen processed-image background/noise model.
The historical `SyntheticConfig.background_std` remains unused and
`render_spots(add_noise=False)` still draws background with SD `background/4`.
Those APIs and their saved fixtures retain historical values; neither is a
noise-off oracle. There is no automatic mapping from the unused field to a
new stochastic component. In the formed API, every background/noise enable flag
defaults false and disables its contribution regardless of retained parameters.
Enable `NoiseConfig.independent_enabled` with explicit `sigma` for additive
residual noise; structured background widths/brightness are separate controls.
See [the API](api/synthetic.rst) for order, identity and provenance contracts.

Formed-scene generator version 4 adds optional `GeometryConfig`. Clean numerical
images remain unchanged. Round truth adds explicit destination `frame_id`;
transforms retain realized coefficients and inverse diagnostics. `scene.metadata`
is the reference grid; use `scene.round_metadata[label]` when saving transformed
rounds. Historical `SyntheticConfig` geometry and fixtures remain separate.
