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
| `FOV.create_subtiles`, `FOV.save_subtiles`, `FOV.from_subtile`, `dataset.SubtileConfig` (with `compute_windows`), `dataset.CropWindow`, `Dataset.subtile` | Removed with the Python subtile route (D5): a FOV in tiles is `dataset.tiled` (`tile_grid`, `run_tile_global`, `run_tile`, `merge_tiles`), and `io.read_window` reads a window of the chunked checkpoint; in the workflow, `processing.mode: tiled` |
| `FOV.subtile_id` | `FOV.tile`: the tile id `r<row>c<col>` or None, the fourth element of a tile's spot namespace; the records keep their `subtile_id` field, null for a whole FOV |
| `FOV.paths.subtile_dir` (`output/subtile/<fov_id>/`) and the `subtile_<n>/` folder of a FOV | Removed with the subtile files; a tiled run's files are `fovs/<fov_id>/tiled/` and `fovs/<fov_id>/tiles/<tile>/` |
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
| The workflow report of a run with `assignment` and `report` blocks and no `assembly` block (a sample without cells, its Segmentation and Assignment steps "not run") | The report names the `assignment` block's `name` and shows the cells, the cell run and the per-FOV segment and assign steps ({ref}`reference <workflow-keys-report>`). Its `report` task now also reads the `assignment` block, so every report task has a new cache token and an engine that compares it reruns the task once; a run with an `assembly` block keeps `assembly.assignment` |
| The report page's Files → Configuration row "Hash" (the first FOV's `workflow_config.sha256`) | The row is labelled "Hash (FOVs ran under)". It shows the hash every FOV's `run.json` holds, which is the `resolved.json` hash when the FOVs ran, not the hash of the file now beside it; when two FOVs hold different hashes it shows "several; see each FOV's run.json". No other row of the page changes |
| An export whose placement needs review (`PlacementReviewRequiredError`): the task failed with exit status 1, Nextflow retried it `--max_retries` times, and no sample report was made | The export task exits 4 (held for review), after making the sample report from the FOVs' records and `placement.json` when every other task the report needs has succeeded, and its failed record says `held`. Nextflow does not retry it. The report's Placement section shows the review's state, reasons, flagged FOVs and approvals. After `approve_placement` and `save_placement`, a rerun of the placement task keeps the approval ({ref}`the route <workflow-placement-review>`). A report with an export is unchanged |
| The molecules panel of `plot_assignment` (panel 2 of `assignment.png` and `assignment_single_layer.png`, and of the workflow's `assignment.previews` and `export.previews`) drew the `unassigned` markers over the `assigned` ones | It draws the molecules in the order `unassigned`, `excluded_cell`, `assigned`, so the assigned molecules are on top; the molecules the assembly excluded stay grey and last. The legend order (`assigned`, `unassigned`, `excluded`), the colours, the sizes, the outlines and the other panels do not change |
| Nextflow with `--max_retries 0` resubmitted a failed task once, at twice its memory and runtime (Nextflow read the 0 as unset and applied its own one retry) | A failed task runs once and the run finishes after the other tasks in flight; the exit status and `<task>.failed.json` are those of the failure. `--max_retries` above 0 retries as before, scaling memory and runtime by the attempt |

### Evaluation and command line

| Before | After |
| --- | --- |
| `benchmark.synthetic` | `synthetic` |
| Separate benchmark drivers and `starfinder-generate` | `starfinder synthetic generate` and the experiment commands below |
| `python -m starfinder.benchmark._trial trial.json` | `starfinder trial run --config trial.json` |
| `starfinder.benchmark._experiment.expand_experiment(path, output)` | `starfinder experiment expand --config PATH --output DIR` (a new or empty output folder) |
| Evaluating saved workflow outputs, checkpoints or trial outputs | `starfinder experiment evaluate --sources PATH --metrics PATH --output DIR` |
| The engine comparison of `scripts/workflow_golden.py` | `starfinder.evaluation.workflow` (the script calls it) |

See the {doc}`experiments guide <benchmark/experiments>` and the
{doc}`report guide <benchmark/report>`.

### Removed benchmark lifecycle

W-781 removed the old benchmark lifecycle, with no alias or compatibility
shim. Each name below is gone; use its replacement. The new commands neither
read nor write the old run, evaluation and report folders. To read an old
run, check out the revision that wrote it. Each source link points to the file
at `8094b30`, the last revision that has it.

| Removed | Replacement |
| --- | --- |
| `starfinder benchmark run --config --input-root --output-root --owner [--run-id] [--resume]` | `starfinder experiment expand --config PATH --output DIR`, then one `starfinder trial run --config trial-NNNNNN.json` per process; the {doc}`workflow <workflow/index>` for FOV and downstream work. There is no resume: a trial refuses an output folder that is not empty |
| `starfinder benchmark evaluate --run-dir DIR` | `starfinder experiment evaluate --sources PATH --metrics PATH --output DIR` over saved trial, workflow and checkpoint outputs |
| `starfinder benchmark report --evaluation-dir DIR` | `starfinder experiment report --evaluation DIR --output DIR [--protocol PATH]` |
| `benchmark.BenchmarkCase`, `benchmark.BenchmarkTrialResult` ([`_records.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/src/python/starfinder/benchmark/_records.py)) | A trial mapping (`method`, `inputs`, `seed`, `root_output_path`) and its `record.json` (`starfinder.benchmark.trial/1`) |
| `benchmark.run_benchmark`, `evaluate_benchmark`, `report_benchmark` ([`_lifecycle.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/src/python/starfinder/benchmark/_lifecycle.py), [`_adapters.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/src/python/starfinder/benchmark/_adapters.py), [`_pipeline.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/src/python/starfinder/benchmark/_pipeline.py), [`_storage.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/src/python/starfinder/benchmark/_storage.py)) | The three commands above |
| The registration evaluation adapter ([`_legacy_evaluation.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/src/python/starfinder/benchmark/_legacy_evaluation.py)) | An explicit projection and detection, then `starfinder.evaluation` metrics in `experiment evaluate` |
| The printed registration quality report ([`_reporting.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/src/python/starfinder/benchmark/_reporting.py)) | `experiment report`, which shows undefined values with their reasons |
| The case builders and profiles: [`benchmarks/recipes.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/benchmarks/recipes.py), [`benchmarks/configs/registration.json`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/benchmarks/configs/registration.json), [`benchmarks/configs/pipeline.json`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/benchmarks/configs/pipeline.json) and the source map [`benchmarks/configs/source-map.json`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/benchmarks/configs/source-map.json) | An explicit experiment or trial mapping, such as the one {download}`experiment_inputs.py <examples/experiment_inputs.py>` writes |
| The saved-table report [`benchmarks/report_saved.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/benchmarks/report_saved.py) | `experiment report`; its `tables/resources.csv` keeps each source's resource scope |
| The preprocessing comparison [`benchmarks/preprocessing_synthetic.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/benchmarks/preprocessing_synthetic.py) and its report [`benchmarks/preprocessing_report.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/benchmarks/preprocessing_report.py) | Preprocessing and spot-finding trials, `experiment evaluate` with `starfinder.evaluation.preprocessing` and `spot_finding`, and `experiment report`. No threshold or development-seed selection is reinstated |
| The examples [`docs/examples/benchmark_lifecycle.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/docs/examples/benchmark_lifecycle.py), [`docs/examples/benchmark_recipes.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/docs/examples/benchmark_recipes.py) and [`docs/examples/preprocessing_comparison.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/docs/examples/preprocessing_comparison.py) | {download}`experiment_inputs.py <examples/experiment_inputs.py>` and {download}`experiment_report_inputs.py <examples/experiment_report_inputs.py>` |
| The guides [`docs/benchmark.md`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/docs/benchmark.md) and [`docs/benchmark-recipes.md`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/docs/benchmark-recipes.md), the `benchmark-profiles` directive ([`benchmark_profiles.py`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/docs/_ext/benchmark_profiles.py)), the notes [`design/notes/benchmark-recipes.md`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/design/notes/benchmark-recipes.md) and [`design/notes/benchmark-navigation/`](https://github.com/wanglab-broad/starfinder/blob/8094b30c6c57486f307aa6198dfb5e33b86c5241/design/notes/benchmark-navigation/index.md) | The {doc}`Benchmark section <benchmark/index>` |

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
| One detection per voxel of a tied local maximum | One representative per regional maximum; plateau and shoulder rows are removed and spot identities renumber ([rule](spot-finding-algorithms.md#local-maxima)) |
| `PipelineConfig(detection=LocalMaximaConfig(...))` | `PipelineConfig(spot_finding=LocalMaximaConfig(...))` |
| Spotiflow and Piscis on the CPU only; `device="cuda"` raised | `device="cuda"` (`ExecutionConfig`, `find_spots`, or `processing.spot_finding.device` in the workflow) runs them on the GPU; `local_maxima` and `starfish_log` refuse it ([execution device](spot-finding-contract.md#execution-device)) |
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

The [workflow guide](workflow/index.md) and
[reference](workflow/reference.md) show complete native configurations.

(workflow-python-shape)=

### The Python configuration shape

The `starfinder workflow` commands read a configuration in the Python shape of
the [workflow contract](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/orchestration.md):
the dataset keys at the top level, one block per step (`processing`,
`morphology`, `segmentation`, `assignment`, `stitching`, `assembly`, `export`,
`report`) and an `execution` section. A present block enables its step; the
`rules` of the MATLAB dialect and their `run` flags go. `backend` defaults to
`python`: `starfinder workflow plan` refuses a configuration without `backend`
that holds a key of the MATLAB dialect, asking for `backend: matlab`, and it
refuses each key that left the Python dialect, naming where it went. There is
no shim. The table below is generated from the contract's migration table:

```{workflow-migration}
```

## Runtime error messages

Some exception messages have changed while their types and trigger conditions stay
the same. Code that matches error text should use exception types instead. The
[message migration record](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/migration.md#runtime-error-messages)
lists the old and current strings.

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
