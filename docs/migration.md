# Python migration guide

Use this guide when updating callers of replaced Python interfaces. FOV keeps
its spelling. The [API reference](api/python.rst) gives current signatures;
the [full migration record](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/migration.md)
keeps historical details and intermediate formats. MATLAB interfaces and their
files retain their own conventions; see the [boundary reference](api/contracts.md#matlab-boundary).

## Quick start

Replace imports first, then adapt calls to typed configs and result objects.
For example, replace `load_multipage_tiff(path)` and `min_max_normalize(image)`:

```python
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from starfinder.io import load_volume, save_volume
from starfinder.preprocessing import MinMaxNormalizationConfig, normalize_min_max

with TemporaryDirectory() as directory:
    path = Path(directory) / "image.tif"
    save_volume(np.arange(24, dtype=np.uint16).reshape(2, 3, 4), path)
    loaded = load_volume(path)
    image = normalize_min_max(loaded.image, config=MinMaxNormalizationConfig(
        output_dtype="uint8", output_range=(0, 255)))
    metadata = loaded.metadata
```

## Imports and member names

Each row gives an old name and its current owner. Call signatures may also
need adjustment in the stage entries below.

### Coordination

| Before | After |
| --- | --- |
| `dataset.STARMapDataset`, `LayerState`, `layers` | `dataset.Dataset`, `RoundState`, `rounds` |
| `dataset.Codebook` | `readout.Codebook` |
| `io.read_checkpoint` | `dataset.load_checkpoint` |
| `dataset.read_checkpoint` | `dataset.load_checkpoint` |
| `FOV.save_spots` | `FOV.export_spots` (see the molecule-table entry below) |
| `FOV.create_subtiles` | `FOV.save_subtiles` |
| `Dataset.channel_record` | Private; use `Dataset.channel_info` and `Dataset.channel_labels` to inspect channels |

### Image I/O and preprocessing

| Before | After |
| --- | --- |
| `io.load_multipage_tiff`, `load_image_stacks`, `save_stack` | `io.load_volume`, `load_round`, `save_volume` |
| `min_max_normalize`, `histogram_match`, `morphological_reconstruction`, `tophat_filter` | `preprocessing.normalize_min_max`, `match_histogram`, `correct_background`, `filter_tophat` |
| `utils.make_projection` | `preprocessing.project_image` |
| `preprocessing.normalize_intensity`, the root export `starfinder.normalize_intensity` and `FOV.normalize_intensity` | `preprocessing.normalize_min_max`, root `starfinder.normalize_min_max` and `FOV.normalize_min_max` |
| `preprocessing.reconstruct_background`, the root export `starfinder.reconstruct_background` and `FOV.reconstruct_background` | `preprocessing.correct_background`, root `starfinder.correct_background` and `FOV.correct_background` |
| `preprocessing.summary_stage` | `preprocessing.summary_recipe` |
| `preprocessing.supplied_section` | `preprocessing.fit_supplied_section` |
| `preprocessing.read_histograms` | `preprocessing.load_histograms` |
| `preprocessing.write_histograms` | `preprocessing.save_histograms` |
| `preprocessing.read_supplied_statistics` | `preprocessing.load_supplied_statistics` |
| `preprocessing.write_supplied_statistics` | `preprocessing.save_supplied_statistics` |

### Registration

| Before | After |
| --- | --- |
| `registration.metrics`, benchmark truth comparisons | Pure domain functions in `evaluation` |
| `dataset.RegistrationRecipe`, `RegistrationStep`, `RecoveryConfig`, `ExternalReference` | `registration.RegistrationRecipe`, `RegistrationStep`, `RecoveryConfig`, `ExternalReference` |
| `FOV.register_rounds(recipe, *, rounds, reference=None, checkpoints=None)` | `FOV.register(recipe, rounds=..., reference=..., checkpoints=...)` |

### Spot finding

| Before | After |
| --- | --- |
| `spotfinding.find_spots_3d` | `spot_finding.find_spots` |
| `spot_finding.resolve_weights` | `spot_finding.resolve_model` |
| `spot_finding.fetch_weights` | `spot_finding.fetch_model` |
| `spot_finding.KNOWN_WEIGHTS` | `spot_finding.KNOWN_MODELS` |
| `spot_finding.KnownWeights` | `spot_finding.KnownModel` |
| `spot_finding.WeightsFile` | `spot_finding.ModelFile` |
| `spot_finding.MissingWeightsError` | `spot_finding.MissingModelError` |
| `spot_finding.WeightsHashMismatchError` | `spot_finding.ModelHashMismatchError` |

### Readout

| Before | After |
| --- | --- |
| `decode_color_seq` | `readout.colors_to_bases` |
| `io.export_spots` | `readout.export_spots` |
| `starfinder.barcode` (the module, its submodules and the root export `starfinder.barcode`) | `starfinder.readout` |
| `starfinder.evaluation.barcode` | `starfinder.evaluation.readout` |
| pytest marker `barcode` (`-m barcode`) | pytest marker `readout` (`-m readout`) |
| `readout.decode_barcodes` and the root export `starfinder.decode_barcodes` | `readout.decode_reads`, root `starfinder.decode_reads` |
| `FOV.decode_barcodes` | `FOV.decode_reads` |
| `readout.encode_bases` | `readout.bases_to_colors` |
| `readout.decode_color_sequence` | `readout.colors_to_bases` |
| `readout.BarcodeDecodingResult` | `readout.ReadDecodingResult` |
| `readout.assign_direct` | `readout.decode_direct` |
| `readout.DirectAssignmentConfig` | `readout.DirectDecoderConfig` |

### Segmentation and assignment

| Before | After |
| --- | --- |
| `segmentation.labels_to_grid` | `segmentation.resample_labels` |
| `segmentation.normalize_percentiles` (the function) | `segmentation.normalize_percentile_unclipped` |
| `assignment.molecule_table_from_csv` | `assignment.import_legacy_molecules_csv` |
| `assignment.sample_labels` (the function) | `assignment.labels_at_positions` |
| `FOV.assign` | `FOV.assign_molecules` |

### Synthetic data

| Before | After |
| --- | --- |
| `synthetic.FormedSceneConfig` | `synthetic.AmpliconSceneConfig` |
| `synthetic.FormedScene` | `synthetic.AmpliconScene` |
| `synthetic.generate_formed_scene` | `synthetic.generate_amplicon_scene` |
| `synthetic.formed_scene_preset` | `synthetic.amplicon_scene_preset` |
| `synthetic.save_formed_scene` | `synthetic.save_amplicon_scene` |
| `synthetic.CellularSceneConfig` (the sample scene's cells, `SampleSceneConfig.cells`) | `synthetic.CellsConfig`; `CellularSceneConfig` now means the former `ContextSceneConfig` |
| `synthetic.ContextSceneConfig` | `synthetic.CellularSceneConfig` |
| `synthetic.ContextScene` | `synthetic.CellularScene` |
| `synthetic.generate_context_scene` | `synthetic.generate_cellular_scene` |
| `synthetic.save_context_scene` | `synthetic.save_cellular_scene` |
| `synthetic.save_context_dataset` | `synthetic.save_cellular_dataset` |
| `synthetic.context_scene_preset` | `synthetic.cellular_scene_preset` |
| `synthetic.CONTEXT_PRESETS` | `synthetic.CELLULAR_PRESETS` |
| `synthetic.CONTEXT_PRESET_VERSION` | `synthetic.CELLULAR_PRESET_VERSION` |
| `synthetic.CONTEXT_TRUTH_VERSION` | `synthetic.CELLULAR_TRUTH_VERSION` |
| `synthetic.generate_dataset` | `synthetic.generate_amplicon_dataset` |
| `synthetic.closed_form_bank` | `synthetic.generate_closed_form_bank` |

### Sample outputs and reports

| Before | After |
| --- | --- |
| `sample_output.SampleSegmentation` | `segmentation.SampleSegmentation` |
| `starfinder.sample_output` (the module and its submodules) | `starfinder.export` |
| `report.read_report` | `report.load_report` |
| `assembly.sample_anndata` | `assembly.write_sample_h5ad` |
| `stitching.bigstitcher_macro` | `stitching.export_bigstitcher` |
| `export.fuse` | `export.fuse_images` |

### Evaluation and command line

| Before | After |
| --- | --- |
| `benchmark.synthetic` | `synthetic` |
| Separate benchmark drivers and `starfinder-generate` | `starfinder synthetic generate`, `starfinder benchmark run/evaluate/report` |

## Image I/O and preprocessing calls

### Preprocessing recipe

Replace fixed `PipelineConfig` preprocessing slots with a
`PreprocessingRecipe` in `PipelineConfig.preprocessing`.

(migration-preprocessing-recipe)=

| Before | After |
| --- | --- |
| `PipelineConfig.normalization` | `PreprocessingStep(MinMaxNormalizationConfig(...))` in `preprocessing.steps` |
| `PipelineConfig.histogram` | `PreprocessingStep(HistogramMatchingConfig(...))` in `preprocessing.steps` |
| `PipelineConfig.histogram_reference_channel` | `HistogramMatchingConfig.reference_channel` |
| `PipelineConfig.reconstruction` | `PreprocessingStep(ReconstructionConfig(...))` in `preprocessing.steps` |
| `PipelineConfig.reconstruction_after_registration` | A reconstruction step in `preprocessing.post_registration` |
| `PipelineConfig.tophat` | `PreprocessingStep(TophatConfig(...))` in `preprocessing.steps` |
| `PipelineConfig.projection` | Use an output projection view instead |
| `FOV.match_histogram(reference_channel=...)` | `FOV.match_histogram(config=HistogramMatchingConfig(reference_channel=...))` |
| `STEPS`, `StepSpec`, `RecipeStep` | `PREPROCESSING_METHODS`, `PreprocessingSpec`, `PreprocessingStep` |

The [preprocessing guide](preprocessing-contract.md) shows recipe construction.
For `save_reference_image`, replace assumptions about the earlier full ZYXC
image with the selected ZYX detection view; the current default file is
`fovs/<fov>/reference.tif`.

## Registration calls

### Registration recipe

| Before | After |
| --- | --- |
| `PipelineConfig(registration=(RegistrationStep(...),))` | Wrap steps in `RegistrationRecipe(...)` |
| `PipelineConfig(registration=())` | `PipelineConfig(registration=None)` |
| `fov.register(RegistrationStep(...))` | `fov.register(RegistrationRecipe((RegistrationStep(...),)))` |
| Per-step image selection and warp | Recipe `signal` and `warp`; step-specific `signal` where needed |
| Earlier merged-channel sum | `RegistrationSignalConfig(mode="sum")`; the default selection is maximum |

### Translation displacement

Replace `TranslationTransform.correction_zyx` with `displacement_zyx` and negate
the old stored value when constructing the new transform. Pass the returned
transform directly to `apply_transform`, with its application config. The
[registration guide](registration-contract.md) shows the call pair.
For changed composition and resampling behavior, consult the
[application specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration.md).

## Detection and readout calls

| Before | After |
| --- | --- |
| Subclassing a detector config to select a method | Use a registered config type; see [adding a method](method-registry.md) |
| `PipelineConfig(detection=LocalMaximaConfig(...))` | `PipelineConfig(spot_finding=LocalMaximaConfig(...))` |
| Several detection rounds decoded as multiplexed reads | Detect in the reference round for multiplexed readout; use `readout_mode="direct"` for a direct assay |
| `voxel_size=(...)` extraction arguments | `NeighborhoodSumConfig(neighborhood_radius_zyx=(...))` |
| Combined extraction and barcode calls | `extract_intensities`, then `decode_reads`, then `filter_reads` |
| `Codebook.table['gene']`, `['color_seq']` | `gene_id`, `color_sequence` |
| Truth `codeword` | `color_sequence` in amplicon truth |
| Gene-indexed codebook entries | `entry_id` identifies entries; `gene_id` identifies genes |

The [readout guide](readout-contract.md) shows the current calls. Inspect
`observed_color_sequence` and `decoded_color_sequence` on decoded reads.

### Barcode layout

Replace a workflow `load_codebook.split_index` workaround that supplied a
zero-based Python index with the shared one-based MATLAB position. Direct
Python `EncodingConfig.split_index` remains zero-based; use `BarcodeLayout`
for explicit segment layouts. See the [readout reference](readout-algorithms.md).

## Segmentation, assignment and sample calls

| Before | After |
| --- | --- |
| Assuming labels are an untyped array on an implicit grid | Use `SegmentationResult` and the [label import functions](segmentation-contract.md) |
| Raw channel lists passed to segmentation | Build `SegmentationInput` with the [input functions](segmentation-contract.md) |
| Older mask/assignment workflow outputs | Use the [assignment guide](assignment-contract.md) for saved assignments and cell counts |
| Sample projection export with existing per-FOV assignments | Set `ExportConfig(route="per_fov")`, or YAML `export.route: per_fov` |
| Segmentation of the fused projection | Configure `export.segmentation`; see [sample segmentation](sample-segmentation-contract.md) |
| Per-FOV expression files on the saved-assignment route | Read `fovs/<fov>/assignment/<name>/assignment.json` and its saved tables |
| Assignment figures produced implicitly on that route | Enable `assembly.previews` |
| Earlier exact-containment correspondence default | Set `CorrespondenceConfig(outside_tolerance=0.0)` to retain it |

## Dataset, files and saved records

### Dataset and FOV coordination migration

| Before | After |
| --- | --- |
| `layers.seq`, `layers.other`, `layers.ref` | `rounds.sequencing_rounds`, `rounds.other_rounds`, `rounds.reference_round` |
| `all_layers`, `to_register` | `all_rounds`, `moving_rounds` |
| `STARMapDataset.from_config(config)` | `from_workflow_config(config, rule).dataset` |
| `load_raw_images`, `enhance_contrast`, `hist_equalize`, `morph_recon`, `tophat` | `load_images`, `normalize_min_max`, `match_histogram`, `correct_background`, `filter_tophat` |
| `global_registration()`, `local_registration(method=...)` | `register(RegistrationRecipe(...))` with the selected method configs |
| `save_signal`, `save_ref_merged`, `save_log`, `save_score_log` | `export_spots`, `save_reference_image`, and the checkpointed run's `run.json` |
| `run_streaming(...)`, `run_streaming_gr(...)` | `run(pipeline, execution=ExecutionConfig("streaming"))` |
| `all_spots`, `good_spots` | `spot_result`, `intensity_result`, `decoding_result`, `filtering_result.accepted` |
| `global_shifts`, `local_registered` | `registration_results`, `registration_chains`, `registration_attempts` |
| `run` loading other rounds | Call `prepare_morphology` separately |
| `save_processing_log`, `save_diagnostics` | `run(..., checkpoints=CheckpointConfig(stages=()))` for `run.json` |
| Morphology text log | `save_registration_record()` for `nuclei_registration.json` |
| `checkpoints/<fov>/run.json` | `fovs/<fov>/run.json`; stage files are under `checkpoints/` below that FOV folder |
| Reference file under `images/ref_merged/` | `fovs/<fov>/reference.tif`, or configured reference file/import |
| Number-only channel folders | Channel labels in saved prepared-image folders |

### Molecule and read tables

| Before | After |
| --- | --- |
| `export_spots("goodSpots")`, `signal/<fov>_goodSpots.csv` | `export_spots()` or slot `"molecules"`, `fovs/<fov>/molecules.csv` |
| `export_spots("allSpots")` | `export_spots("reads")`, `fovs/<fov>/reads.csv` |
| One-based `x,y,z,gene` | Zero-based `z,y,x,gene_id`, with `spot_namespace,spot_id` identity |
| `import_legacy_molecules_csv` for new Python outputs | `load_molecules(path, genes=...)`; retain the legacy importer for legacy files |

(readout-checkpoints)=
### Checkpoints

Replace manual parsing of old checkpoint headers with
`dataset.load_checkpoint` or `FOV.load_checkpoint`. Current checkpoint headers
are version 3; current `run.json` is version 2. The
[checkpoint guide](checkpoints.md) shows reload calls, and its
[reference](checkpoints-reference.md) lists saved files.

### One Python configuration dialect

For `backend: python`, replace MATLAB-style blocks with native stage settings:

| Before | After |
| --- | --- |
| `enhance_contrast`, `hist_equalize`, `morph_recon`, `tophat` | `preprocessing.steps` with the corresponding registered methods |
| Resident-subtile `morph_recon` | `preprocessing.post_registration` |
| `global_registration`, `local_registration` | `registration.steps` |
| `ref_img`, `mov_img` | `registration.signal` |
| Demons method named by variant | `method: demons`, with `variant` |
| `intensity_estimation`, `intensity_threshold`, local-maxima `min_distance` | `threshold_mode`, `threshold_value`, `min_distance_voxels` |
| `dapi_round` | `reference_stains` entries |
| `OUTPUT/documents/genes.csv` as count-matrix authority | The sequencing codebook; align count columns by gene name |
| Subset switches for Python membership | `input_root/fovs.csv` and `selected_fovs` |
| A percent-style FOV pattern | A `{i}` pattern |
| Hand-built round-first copies | Set `image_dir_template="{fov}/{round}"` for FOV-first inputs |

The [workflow guide](workflows.md) and
[reference](workflow-configuration.md) show complete native configurations.

## Runtime error messages

If your code matches error text, update these eight messages. Exception types
and the conditions that raise them stay the same. Braced fields below keep their
runtime values.

| Source | Before | After |
| --- | --- | --- |
| `assembly/_assemble.py` | `only counted cells (retained, §2.9 kept) have count rows` | `only counted cells (retained, assignment kept) have count rows` |
| `dataset/workflow.py` | `the segmentation block is run by FOV.segment (the §2.13 rule); the stardist_segmentation rule translates rules.stardist_segmentation.parameters only: remove the block or the rule` | `the segmentation block is run by FOV.segment (the workflow orchestration rule); the stardist_segmentation rule translates rules.stardist_segmentation.parameters only: remove the block or the rule` |
| `dataset/workflow.py` | `the assignment block is run by FOV.assign_molecules (the §2.13 rule); the reads_assignment rule translates rules.reads_assignment.parameters only: remove the block or the rule` | `the assignment block is run by FOV.assign_molecules (the workflow orchestration rule); the reads_assignment rule translates rules.reads_assignment.parameters only: remove the block or the rule` |
| `segmentation/_learned.py` | `block fields together with a scale other than 1 are not supported (W-309): nothing has tested that combination` | `block fields together with a scale other than 1 are not supported: nothing has tested that combination` |
| `segmentation/_learned.py` | `diameter None (no rescale) is refused with do_3d=True (W-306)` | `diameter None (no rescale) is refused with do_3d=True` |
| `stitching/_bigstitcher.py` | `{path}: view {view} composes {names} to a linear part {off:.3g} off the identity; §2.10 admits translation only (tolerance {LINEAR_TOLERANCE})` | `{path}: view {view} composes {names} to a linear part {off:.3g} off the identity; stitching admits translation only (tolerance {LINEAR_TOLERANCE})` |
| `stitching/_bigstitcher.py` | `{what}: the shift's linear part is {off:.3g} off the identity; §2.10 admits translation only` | `{what}: the shift's linear part is {off:.3g} off the identity; stitching admits translation only` |
| `stitching/_methods.py` | `transform_kind of {self.name!r} must be 'translation', the only kind §2.10 admits` | `transform_kind of {self.name!r} must be 'translation', the only kind stitching admits` |

## Synthetic and evaluation calls

| Before | After |
| --- | --- |
| `SyntheticConfig`, `get_preset_config(name)` | `benchmark_scene_preset(name)` returns a codebook and config |
| `generate_registration_pairs(...)` | One `generate_registration_pair(...)` call per pair |
| `generate_displacement_field`, `DEFORMATION_CONFIGS` | `deformation_geometry` and `forward_displacement` |
| `render_spots`, `generate_volume` | `generate_amplicon_scene` |
| `generate_codebook(n)` returning tuples | A `Codebook` result |
| Default uint8 synthetic images | Default uint16; select `dtype` for the desired output |
| `gene_accuracy`, `color_seq_accuracy` | `gene_id_accuracy`, `color_sequence_accuracy` |

The [synthetic guide](synthetic-specification.md) describes the current generator;
regenerate historical generator outputs rather than expecting the same images.

The [full migration record](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/migration.md)
keeps intermediate changes and detailed format transitions.
