# Segmentation reference

The methods, options, functions, records, files and workflow keys of segmentation. The
method and option tables are read from the code when the site is built. The
{doc}`segmentation guide <segmentation-contract>` explains how to use them; the
{doc}`segmentation API <api/segmentation>` documents every class.

## Methods

```{method-table} starfinder.segmentation.SEGMENTATION_METHODS
```

Each method's name is its `method` value in the workflow's `segmentation` block.

## Method options

### StarDist

StarDist predicts star-convex objects, nuclei above all, with a pretrained or
user-trained model; the model's dimensionality decides between 2D and 3D. See
{py:class}`~starfinder.segmentation.StarDistConfig`.

```{config-options} starfinder.segmentation.StarDistConfig
```

### Cellpose

Cellpose segments nuclei or whole cells from stains in 2D and 3D. See
{py:class}`~starfinder.segmentation.CellposeConfig`.

```{config-options} starfinder.segmentation.CellposeConfig
```

### Seeded watershed

The seeded watershed grows one cell from each nucleus of the seed run over the
smoothed stain, and each cell keeps its nucleus's value. See
{py:class}`~starfinder.segmentation.SeededWatershedConfig`.

```{config-options} starfinder.segmentation.SeededWatershedConfig
```

## Plans

A plan is an ordered tuple of runs; each run names its channels with `InputChannel`. See
{py:class}`~starfinder.segmentation.SegmentationPlan`,
{py:class}`~starfinder.segmentation.SegmentationRun` and
{py:class}`~starfinder.segmentation.InputChannel`.

```{config-options} starfinder.segmentation.SegmentationRun
```

```{config-options} starfinder.segmentation.InputChannel
```

## Import

A run imports a mask with `LabelImportConfig`; `import_labels` imports one onto a grid
without a FOV. See {py:class}`~starfinder.segmentation.LabelImportConfig` and
{py:func}`~starfinder.segmentation.import_labels`.

```{config-options} starfinder.segmentation.LabelImportConfig
```

## Input and label functions

Each function can be called on its own and returns its result with a record. The first
two make a run's channel through `InputChannel.prepare`; `expand_labels` and
`extend_labels_through_z` are a run's `operations`.

| Function | Config | What it does |
| --- | --- | --- |
| `composite_nuclei_amplicon` | `CompositeConfig` | One 8-bit image in which both the nuclei and the amplicon signal are bright |
| `enhance_with_flamingo` | `FlamingoEnhancementConfig` | An 8-bit DAPI image dimmed where the Flamingo stain is bright |
| `normalize_percentile_unclipped` | | Percentile normalization to float32 without clipping |
| `rescale_input` | | Rescales an image by per-axis factors, with its metadata |
| `resample_labels` | | Maps a label image onto another shape by nearest neighbour |
| `expand_labels` | `ExpandLabelsConfig` | Grows each label into the background, in each plane or in 3D |
| `extend_labels_through_z` | `ZExtensionConfig` | Extends plane labels through Z where a stain is foreground |

### Composite input

Stretch settings for the nuclei–amplicon composite. See
{py:class}`~starfinder.segmentation.CompositeConfig`.

```{config-options} starfinder.segmentation.CompositeConfig
```

### Flamingo enhancement

Settings for enhancing the nuclear stain with Flamingo. See
{py:class}`~starfinder.segmentation.FlamingoEnhancementConfig`.

```{config-options} starfinder.segmentation.FlamingoEnhancementConfig
```

### Label expansion

Settings for growing label territories. See
{py:class}`~starfinder.segmentation.ExpandLabelsConfig`.

```{config-options} starfinder.segmentation.ExpandLabelsConfig
```

### Z extension

Settings for extending plane labels through a stain volume. See
{py:class}`~starfinder.segmentation.ZExtensionConfig`.

```{config-options} starfinder.segmentation.ZExtensionConfig
```

## Known models

`KNOWN_MODELS` lists the pretrained models that Starfinder can fetch and verify;
`starfinder weights list` prints them with their local state.

| Method | Model | Dimensionality | Channels |
| --- | --- | --- | --- |
| `stardist` | `2D_versatile_fluo` | 2D | 1 |
| `cellpose` | `cpsam_v2` | 2D network; 3D by planes with `do_3d` | 1 to 3 |

See {py:class}`~starfinder.segmentation.KnownModel` and
{py:func}`~starfinder.segmentation.resolve_model`.

## Label images

Every label image, computed or imported, is a C-contiguous `uint32` ZYX array: 0 is
background and each positive value is one object; a plane is 1×Y×X.
{py:func}`~starfinder.segmentation.to_label_dtype` converts a method's or a file's
integer labels to `uint32` without wrapping. Label values stay as the method returned
them.

| Geometry | Labels |
| --- | --- |
| `volume` | A 3D result on Z>1 |
| `plane` | A result on one plane, Z=1 |
| `extended` | Plane labels extended through Z by `extend_labels_through_z` |

## Records

`SegmentationResult.record`, written as `segmentation.json`, holds:

| Key | Contents |
| --- | --- |
| `format_version`, `stage` | `1` and `"segmentation"` |
| `dataset_id`, `sample_id`, `fov_id`, `subtile_id`, `run` | The identities of the label namespace |
| `target`, `geometry`, `label_namespace` | As on the result |
| `grid` | The grid's `shape_zyx`, `metadata`, `source` and `sha256` |
| `input` | A computed run's input: its SHA-256, shape, dtype, metadata, projection and channels, with the file's path and SHA-256 when saved; `null` for an import |
| `seeds` | The seed run, its label namespace and labels' SHA-256, and the method's label rule; or `null` |
| `methods` | The method's provenance entry, with its model files (`artifacts`), `execution` and `effective` parameters; empty for an import |
| `import` | An import only: the file, its SHA-256, the source dtype and shape, where the metadata came from, and the relabel map |
| `operations` | The label operations applied, each with its config and record |
| `outcome` | `ok`, or `empty` when no label remains |
| `labels` | The labels' dtype, SHA-256, `n_labels` and `max_label`, with the file's path and SHA-256 when saved |
| `software` | The software versions |
| `upstream` | Runs of `FOV.segment`: the SHA-256 of the FOV's preprocessing and registration records |

## Files

| File | Written by | Contents |
| --- | --- | --- |
| `segmentation/<run>/labels.tif` | `FOV.segment` with `checkpoints` | The labels, ZYX `uint32`, with the grid's metadata |
| `segmentation/<run>/input.ome.tif` | the same, for a computed run | The segmentation input, ZYXC |
| `segmentation/<run>/segmentation.json` | the same | The run record |
| `images/stardist_segmentation/{fovID}.tif` | the workflow rule `stardist_segmentation` | The labels, YX or ZYX as the input file |
| `images/stardist_segmentation/{fovID}.json` | the same | The run record |

The `segmentation/` folders are inside the FOV's folder, `<output_root>/fovs/<fov_id>/`
by default ({doc}`checkpoint reference <checkpoints-reference>`); `FOV.load_segmentation(name)` reads them. The
`images/` paths are under the workflow's output folder.

## Workflow keys

### The segmentation block

With `backend: python`, the top-level `segmentation` block holds a plan; no rule runs it yet.

| Key | Meaning |
| --- | --- |
| `device` | The device of the plan's runs |
| `runs` | The runs, in order |
| `runs[].name`, `target`, `seeds` | As in `SegmentationRun` |
| `runs[].inputs` | A list of `InputChannel` fields; `prepare` names `composite_nuclei_amplicon` or `enhance_with_flamingo` with its config's fields |
| `runs[].projection` | The run's Z projection, with `ProjectionConfig` fields |
| `runs[].operations` | A list of label operations, each `expand_labels` or `extend_labels_through_z` with its config's fields |
| `runs[].method` | A method of the table above, or `import` |
| The config's fields | The fields of that method's config, or of `LabelImportConfig` |

{doc}`workflow-configuration` describes the block.

### Legacy StarDist keys

The rule turns `rules.stardist_segmentation.parameters` into one StarDist run:

| Legacy key | Becomes |
| --- | --- |
| `stardist_base_path`, `stardist_model_name` | The model: the folder `<base>/<name>`, or the known model of that name when the base path is the weights cache's `stardist` folder |
| `prob_thresh`, `nms_thresh` | The StarDist thresholds |
| `rescale` | `scale` 0.5 when true, 1.0 when false |
| `expand_labels`, `distance` | One planar expansion by `distance` pixels |
| `segmentation_input_folder` | The input channel's role: `composite` for `overlay`, the default, and `nuclear` for any other folder |
| `target` (`backend: python`) | The run's target; without it, `cell` for `overlay` and `nucleus` otherwise |
| `device` (`backend: python`) | The device |

The run record keeps the legacy keys under `workflow.legacy`.
