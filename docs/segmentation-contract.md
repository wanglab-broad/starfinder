# Segmentation contract: label images, methods, inputs and environments

Status: Proposed

This page proposes the §2.9 segmentation contract: a segment entry separate from
`FOV.run`, one label-image contract for every method and for imported masks, a
segmentation registry on the shared mechanism of {doc}`method-registry`, the reusable
input-preparation and label functions that are not methods, the model rules, the device
and environment plan, and the workflow translation of the legacy keys. Its registry
fields, lookup, dependency and provenance rules are those of {doc}`method-registry`,
used unchanged. The current behavior is recorded in {doc}`segmentation-baseline`; the
numerical behavior of each method and function is in {doc}`segmentation-algorithms`.
Nothing here is implemented and nothing is renamed. Assignment, nucleus–cell
correspondence, compartments, counts and exports belong to W-308, which builds on the
label contract below and does not redefine it.

Evidence:

* W-306, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-306/20261004T194036Z-74e12949`
  (found with `timeout 30 find …/runs/W-306 -maxdepth 2 -name segmentation-manifest.json`):
  `notes.md`, `tables/segmentation-calls.csv`, `cost-summary.csv`, `comparisons.csv`,
  `models.csv`, `parity.csv`, `label-dtype-probe.json`, `loading-probes.json`,
  `fov-memory-estimate.json`, `environment-packages.csv`, the seam tables, and
  `provisioning/` (`lock-diff.txt`, `spike-added-packages.txt`, `freeze.txt`).
* W-305: the compatibility matrix
  `runs/W-305/20261004T0510Z-stardist-tf-matrix/matrix.md` (rows L, A, A1, A2, B, C, D),
  the model inventory `runs/W-305/20261004T0630Z-summary/models.md` and the examples
  `runs/W-305/20261004T0630Z-summary/examples.md`, all under
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/`.

W-306 is dependency selection: it ranks no segmenter, sets no default and makes no
accuracy claim. Its counts and agreements show that a model loaded or that two results
agree, nothing more. This page makes no accuracy claim either.

## Settled decisions this page follows

The W-307 issue restates five decisions from the W-152 comment "§2.9 planning
decisions (2026-10-03)"; this page follows them as stated there:

1. **D1** The segment entry is separate from `FOV.run` and `PipelineConfig`, with its
   own run record.
2. **D2** A method is an in-process callable that produces one label image from an
   input image (and optional seeds).
3. **D3** External-mask import produces the same label contract; import is not a
   method.
4. **D4** The composite, the Flamingo enhancement, normalization and rescaling of the
   input, label expansion and the culture extension through z are reusable functions,
   not methods.
5. **D5, D6** The device values and the environments follow the W-305 and W-306
   evidence.

The comment's own text and D7 were not among this issue's inputs; nothing below relies
on D7. The rules Jiahao set after W-305 (2026-10-04) also hold:

* Models in scope are the trained StarDist `3D_spleen` (LN), the pretrained StarDist
  `2D_versatile_fluo` (tissue-2D) and Cellpose `cpsam_v2`; `2D_brain_overlay_05` and the
  other trained models are out.
* The input scale follows the model's training data. It is an explicit, recorded
  parameter, with no default carried over from another model.
* Cell outlines are never computed on a shrunk grid. Nuclei may be detected on a shrunk
  input, but their labels come back on the full-resolution grid, where the watershed
  runs with its seeds and its stain.
* A model's stored `thresholds.json` is the primary setting; any other threshold is a
  sensitivity setting and is recorded as such.
* One environment holds StarDist 0.9.2 with TensorFlow 2.20.0, Cellpose and the §2.7
  detectors next to the locked packages, with one locked change (tensorboard 2.21.0 to
  2.20.0) and the `+cu126` torch build on GPU hosts. This page proposes the lock change;
  no issue makes it.

## Terms

* A **segmentation input** is one ZYXC image on one grid whose channels each carry a
  declared **role** (`nuclear`, `cytoplasm`, `membrane`, `amplicon`, `composite`).
* A **label image** is an integer ZYX image in which 0 is background and each positive
  value is one object.
* The **target** of a label image is what its objects are: `nucleus` or `cell`.
* The **geometry** of a label image is how its Z extent was obtained: `volume` (a true
  3D result on Z>1), `plane` (a 2D result on Z=1), or `extended` (a 2D result extended
  through z by `extend_labels_through_z`).
* **Seeds** are a label image of another run on the same grid that a method grows
  from (the nuclei of a seeded watershed).
* A **segmentation run** is one call of the segment entry for one FOV that yields one
  label image; it has a name (`nucleus`, `cell` or another) that keys its result.
* A **label operation** maps a label image to a label image (expansion, extension
  through z); an **input function** maps images to an image (composite, Flamingo
  enhancement, normalization, rescaling). Neither is a method.

## Names

The entry and result names, with the options considered:

| Option | Names | Effect on existing files | Effect on the lock | Effect on the engines |
| --- | --- | --- | --- | --- |
| **S1. A `starfinder.segmentation` module (recommended)** | Module `starfinder.segmentation`; function `segment`; result `SegmentationResult`; input `SegmentationInput`; registry `SEGMENTATION_METHODS` with `SegmentationSpec`; FOV coordination `FOV.segment(plan)` with `SegmentationPlan` and `SegmentationRun`, results in `FOV.segmentation_results`; import `import_labels`. | New files only, plus one new method and one attribute on `FOV` (`dataset/fov.py`); `docs/api/segmentation.rst`, `docs/api/inventory.rst` and `docs/api/python-index.rst` gain the exports, which `docs/check_reference.py` requires. No existing name changes. | None: the backends are imported lazily inside the methods. | The workflow adapter (`dataset/workflow.py`) gains `_run_segmentation`; MATLAB is untouched. |
| S2. A `starfinder.labels` module | Module `starfinder.labels`; function `predict_labels`; result `LabelImage`; FOV coordination `FOV.label(plan)`. | The same files. "Labels" also names `reads_assignment.py`'s `seg_label` column and the checkpoint `labels`, so the module name collides with W-308's assignment vocabulary. | None. | The same adapter function under another name. |

**Recommendation: S1.** The stage is called segmentation in the chapter, the registry
pattern names registries after stages (`SPOT_FINDING_METHODS`, `DECODING_METHODS`), and
`segment` reads as the verb next to `find_spots` and `decode_barcodes`.

New names under S1:

| Concept | Name | Kind |
| --- | --- | --- |
| Entry | `segment(segmentation_input, *, config, target, seeds=None, device="cpu", label_namespace)` | function |
| Result | `SegmentationResult` | frozen dataclass |
| Input | `SegmentationInput` | frozen dataclass |
| Registry | `SEGMENTATION_METHODS: dict[type, SegmentationSpec]` | registry |
| Methods | `StarDistConfig` (`stardist`), `CellposeConfig` (`cellpose`), `SeededWatershedConfig` (`seeded_watershed`) | configs |
| Import | `import_labels`, `LabelImportConfig` | function, config |
| Input functions | `composite_nuclei_amplicon` (`CompositeConfig`), `enhance_with_flamingo` (`FlamingoEnhancementConfig`), `normalize_percentiles`, `rescale_input` | functions, configs |
| Label functions | `expand_labels` (`ExpandLabelsConfig`), `extend_labels_through_z` (`ZExtensionConfig`), `labels_to_grid` | functions, configs |
| Models | `KNOWN_MODELS`, `resolve_model`, `ModelFile`, `KnownModel` | table, function, dataclasses |
| Errors | `SegmentationBackendUnavailableError`, `MissingModelError`, `ModelHashMismatchError` | exceptions |
| Coordination | `FOV.segment(plan, *, device="cpu", checkpoints=None)`, `FOV.segmentation_results`, `FOV.load_segmentation(name)`; `SegmentationPlan`, `SegmentationRun`, `InputChannel` | method, attribute, dataclasses |

## The segment entry

### Function

```python
def segment(segmentation_input: SegmentationInput, *, config, target: str,
            seeds: SegmentationResult | None = None, device: str = "cpu",
            label_namespace: str) -> SegmentationResult: ...
```

`config` is a frozen config registered in `SEGMENTATION_METHODS`. `segment` applies the
stage checks ("Checks the stage wrapper applies" below), calls the method's `run`, checks
and converts the label image, and returns a `SegmentationResult` whose record holds the
uniform provenance entry. It never reads files, never writes files and never applies a
label operation; those belong to the coordination layer or to the caller.

### Coordination per FOV

```python
@dataclass(frozen=True)
class InputChannel:
    role: str                          # nuclear, cytoplasm, membrane, amplicon or composite
    round: str | None = None           # a loaded round in the reference frame; None with reference_merged
    channel: str | int | None = None   # a label of Dataset.channel_labels(round), or an index
    reference_merged: bool = False     # the reference round's channel maximum (save_reference_image "merged")
    prepare: CompositeConfig | FlamingoEnhancementConfig | None = None  # an input function and its own sources

@dataclass(frozen=True)
class SegmentationRun:
    name: str                          # snake_case; the key of FOV.segmentation_results and the record folder
    target: str                        # nucleus or cell
    inputs: tuple[InputChannel, ...]
    method: object                     # a SEGMENTATION_METHODS config, or a LabelImportConfig
    seeds: str | None = None           # name of an earlier run of the same plan
    projection: ProjectionConfig | None = None   # optional Z projection of the input (legacy maximum_projection)
    operations: tuple = ()             # ExpandLabelsConfig, ZExtensionConfig, applied in order after the method

@dataclass(frozen=True)
class SegmentationPlan:
    runs: tuple[SegmentationRun, ...]  # in order; seeds name earlier runs
```

`FOV.segment(plan, *, device="cpu", checkpoints=None)` runs each run in order: it
assembles the `SegmentationInput` from the FOV's images, calls `segment` (or
`import_labels`), applies the run's label operations, stores the result in
`FOV.segmentation_results[run.name]` and, when `checkpoints` is given, writes the run's
files ("Saved format"). It never runs registration, detection or decoding, and
`PipelineConfig` gains no field (D1).

The three calls per FOV are:

1. `fov.run(pipeline, execution=…, checkpoints=…)` registers and processes the
   sequencing rounds as today and leaves the reference round's image and metadata
   resident (streaming keeps the reference image; {doc}`coordination`).
2. `fov.register_rounds(recipe, rounds=[…])` brings the loaded morphology rounds into
   the reference frame through their shared stain ({doc}`coordination`, "Other rounds
   and external references").
3. `fov.segment(plan, device=…, checkpoints=…)` builds each segmentation input from the
   reference-frame images and runs the plan.

A plan whose inputs use only the reference round (tissue-2D PI in a sequencing round)
needs no `register_rounds` call; a plan that imports external masks needs neither of the
first two calls beyond loading the reference metadata.

### What the segment entry needs from `FOV.run` and `FOV.register_rounds`

| From | What | Used for |
| --- | --- | --- |
| `FOV.run` | `fov.metadata[reference_round]`: the reference `ImageMetadata` (frame, spacing, origin, direction) and the reference grid `images[reference_round].shape[:3]` | the grid and metadata every segmentation input and label image must have |
| `FOV.run` | the reference round's current image (its detection image after `run`), whose channel maximum is the reference merged image, ZYX, exactly as `save_reference_image(reference_image="merged")` writes it | the amplicon channel of the composite (`InputChannel(reference_merged=True)`) and the stain of a seeded watershed on the amplicon signal (tissue-2D) |
| `FOV.run` | `preprocessing_record` and `registration_record` | copied by reference (their hashes) into the segmentation record, so the input's processing is traceable |
| `FOV.register_rounds` | each registered morphology round's image (every channel resampled once into the reference frame) and its metadata, which equals the reference metadata | the `nuclear`, `cytoplasm` and Flamingo channels |
| `FOV.register_rounds` | `registration_record["rounds"][round]`: the recipe summary, the reference label and `reference_sha256` | the input-channel identity in the segmentation record |
| `Dataset` | `channel_labels(round)` and `other_channel_order` | resolving `InputChannel.channel` by label; an unknown label raises `ValueError` |

`FOV.segment` raises `ValueError` naming the round when an input round is not loaded, is
a morphology round that has not been registered, or has metadata different from the
reference round's; and `IncompatibleGeometryError` when a grid differs. A run's
`projection` is the only route to a plane from a volume, and the record keeps it.

### What the assignment entry receives

W-308 builds the assignment entry on these, and on nothing else from segmentation:

* one `SegmentationResult` per run: `labels` (`uint32` ZYX on the reference grid, or Z=1
  for a `plane` or projected run), `metadata` (equal to the reference metadata, or its
  `projected` form), `target`, `geometry`, `label_namespace`, `n_labels`, `max_label`;
* `record["operations"]`: the label operations already applied, so expansion is applied
  once and the assignment never expands again;
* for a seeded run, `record["seeds"]`: the seed run's name and labels SHA-256, and the
  correspondence rule of the method (`seeded_watershed`: cell k contains nucleus k,
  `record["seed_correspondence"] = "label"`); for other runs `None`;
* the saved files of each run and their SHA-256 values, so a later assignment can load
  them with `FOV.load_segmentation(name)` without the images.

The molecules' `z, y, x` (index coordinates of the reference grid, from the spot table)
index these label images directly. How a 3D molecule is assigned to a `plane` label
image, and how nuclei and cells correspond, are W-308's.

### Run record

Each run writes one `segmentation.json` (when checkpoints are on) and keeps the same
mapping in `SegmentationResult.record`:

```json
{
  "format_version": 1,
  "stage": "segmentation",
  "dataset_id": "...", "sample_id": "...", "fov_id": "FOV_001", "subtile_id": null,
  "run": "nucleus", "target": "nucleus", "geometry": "volume",
  "label_namespace": "[\"dataset\", \"sample\", \"FOV_001\", null, \"nucleus\"]",
  "input": {
    "path": "input.ome.tif", "sha256": "<C-order bytes>", "file_sha256": "...",
    "shape_zyxc": [50, 512, 512, 1], "dtype": "uint8", "metadata": {"frame_id": "..."},
    "projection": null,
    "channels": [{"role": "nuclear", "round": "round4", "channel": "ch04", "reference_merged": false,
                  "prepare": null, "registration": {"reference": "round1:ch04", "reference_sha256": "..."},
                  "sha256": "..."}]
  },
  "seeds": null,
  "methods": [{"stage": "segmentation", "method": "stardist", "config_type": "...", "implementation": "...",
               "config": {}, "requires": {"stardist": "0.9.2", "tensorflow": "2.20.0"},
               "artifacts": [{"name": "stardist/3D_spleen", "path": "...", "sha256": "...", "source": "path"}],
               "execution": {"device": "cpu", "framework": {}, "threads": {}},
               "effective": {"scale_zyx": [1.0, 1.0, 1.0], "prob_thresh": 0.6429, "threshold_source": "stored"}}],
  "operations": [{"operation": "expand_labels", "config": {"distance": 4, "unit": "pixel", "mode": "planar"}}],
  "outcome": "ok",
  "labels": {"path": "labels.tif", "dtype": "uint32", "sha256": "...", "file_sha256": "...",
             "n_labels": 425, "max_label": 425},
  "software": {"starfinder": "...", "git_commit": "...", "packages": {}}
}
```

* `methods` holds the uniform entry of {doc}`method-registry`, with `stage`
  `"segmentation"`, plus `execution` (as in {doc}`spot-finding-contract`) and
  `effective` (the parameters the method resolved: scale, thresholds and their source,
  tiles). An imported run has `methods: []` and an `import` entry instead.
* `input.sha256` is the SHA-256 of the segmentation input's C-order bytes (dtype and
  shape included, as the golden tests hash arrays); each channel also has its own.
* `outcome` is `ok` or `empty` (no object; not an error).
* `software` has the same content as the `run.json` written by `FOV.run`
  ({doc}`checkpoints`); `run.json` itself does not change.

## Label image

| Element | Rule |
| --- | --- |
| Array | Integer ZYX, C-contiguous. 0 is background; each positive value is one object. A plane is 1×Y×X, never YX. |
| dtype | The label dtype rule below (recommended: `uint32`). |
| Grid | Equal to the segmentation input's ZYX shape. A method whose library returns another shape (Cellpose drops Z for one plane, W-306 notes section 6) has it restored by the wrapper; any other shape raises `ValueError`. |
| Metadata | The input's `ImageMetadata`, unchanged: the same `frame_id`, spacing, origin and direction. For the reference grid this is the reference round's metadata; for a projected run, its `projected(method=…)` form. |
| Target | `nucleus` or `cell`, declared by the run and checked against the method's `targets`. |
| Geometry | `volume` when a method ran in 3D on Z>1; `plane` when the input has Z=1; `extended` after `extend_labels_through_z`, whose record keeps the source plane run. |
| Identities | A label's identity is `(label_namespace, value)`. `label_namespace` is a JSON list `[dataset_id, sample_id, fov_id, subtile_id, run name]`, built like the spot namespace. Values are kept as the method returned them; the wrapper never relabels, so a seeded watershed keeps its seeds' values. Values are unique within one result and mean nothing across runs except through a recorded seed correspondence. |
| Provenance | `SegmentationResult.record` ("Run record"). |
| Saved format | The saved-format option below (recommended: a ZYX TIFF with metadata, and the JSON record beside it). |
| Empty result | Allowed: an all-zero image with `outcome` `empty`. |

```python
@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray                 # uint32 ZYX, 0 background
    metadata: ImageMetadata            # equal to the input's
    target: str                        # nucleus or cell
    geometry: str                      # volume, plane or extended
    label_namespace: str
    record: Mapping[str, Any]          # the run record above
    diagnostics: Mapping[str, Any]     # method-specific counts and timings

    @property
    def n_labels(self) -> int: ...     # number of distinct positive values
    @property
    def max_label(self) -> int: ...
```

`__post_init__` checks the dtype, the absence of negative values, the target and the
geometry, and that `labels.shape` equals `metadata`'s grid when one is recorded.

### Label dtype rule

The libraries return different dtypes: StarDist int32, or uint16 when `3D_spleen` finds
nothing; Cellpose uint16, switching to uint32 above 65,535 masks with a warning; the
current script casts to uint16 and wraps (W-306 notes section 6,
`tables/label-dtype-probe.json`: 70,000 objects survive in int32 and uint32, and the cast
maps 65,536 to 0).

| Option | Rule | Effect on existing files | Effect on the lock | Effect on the engines |
| --- | --- | --- | --- | --- |
| **L1. Always `uint32` (recommended)** | Every result and every saved label image is `uint32`; the wrapper converts the library result explicitly and raises `ValueError` on a negative value or a value above 2³²−1. | `images/stardist_segmentation/{fovID}.tif` changes from uint16 to uint32 when the rule runs through the package; `reads_assignment.py` reads it with `imread` and indexes with it, which works for both. Files written before keep loading. | None. | tifffile and napari read uint32 TIFF and Fiji opens it as a 32-bit image; no MATLAB script reads these files. |
| L2. Smallest unsigned type that holds `max_label` | uint16 up to 65,535 labels, else uint32. | The common case keeps today's uint16 files. | None. | Every consumer must accept two dtypes, and a digest depends on the object count's range as well as the labels. |
| L3. `int32`, as StarDist returns | Signed 32-bit. | uint16 files become int32. | None. | Signed labels admit negative values that mean nothing; Cellpose results need a conversion that can overflow above 2³¹−1. |

**Recommendation: L1** (W-306 unresolved choice 2). One dtype makes digests and
consumers simple, holds any realistic FOV (the 50×1496×1496 estimate has far fewer than
2³² voxels), and never wraps. A caller who needs uint16 converts explicitly with a check.

### Saved format

| Option | Files per run | Effect on existing files | Effect on the lock | Effect on the engines |
| --- | --- | --- | --- | --- |
| **F1. TIFF and a JSON record (recommended)** | `<checkpoint dir>/segmentation/<run>/labels.tif` (ZYX, uint32, zlib, written by `save_volume` with `ImageMetadata` in the JSON description), `input.ome.tif` (the segmentation input, ZYXC OME-TIFF by `save_volume`) and `segmentation.json` (the run record). The workflow keeps writing `images/stardist_segmentation/{fovID}.tif` for the legacy rule. | None beyond the dtype of L1; `load_volume` reads the label file with its metadata. `run.json` and the existing checkpoint stages are untouched. | None (tifffile is a core dependency). | Snakemake declares TIFF outputs as today; Fiji opens the files; MATLAB reads TIFF. |
| F2. OME-Zarr label image | A `labels/<run>` group in an OME-Zarr image with the NGFF label metadata. | New format beside the TIFF images; `reads_assignment.py` could not read it. | `ome-zarr` is in the lock only through the `spatialdata` extra; it would become a dependency of segmentation. | Snakemake must declare directories; MATLAB cannot read it without an add-on. |
| F3. Checkpoint stage in `FOV.run`'s format | A new `segmentation` checkpoint stage beside `registered`, `candidates` and `pre_qc`. | `CheckpointConfig.stages` and `FORMAT_VERSION` change; `test_checkpoints.py`, which fixes the stage list, needs an edit. | None. | As F1. |

**Recommendation: F1.** It keeps the file type every current consumer reads, keeps
segmentation out of `FOV.run`'s checkpoint format (D1), and writes the input image the
labels were computed from, so a run can be checked against its input's hash.
`FOV.load_segmentation(name)` reads `labels.tif` and `segmentation.json`, checks the
recorded SHA-256 values and returns the `SegmentationResult`.

## Two targets

| Option | Representation | Effect on existing files | Effect on the lock | Effect on the engines |
| --- | --- | --- | --- | --- |
| **T1. Two runs (recommended)** | One `SegmentationResult` per target, as two runs of one plan; the cell run names the nucleus run as `seeds` when its method grows from seeds. | Two folders, `segmentation/nucleus/` and `segmentation/cell/`; the legacy single label file stays for the legacy rule. | None. | One Snakemake output per run; a run can be recomputed alone. |
| T2. One result with two label images | `SegmentationResult` holds `nuclei` and `cells` arrays from one call. | One folder with two label files. | None. | One rule output pair. A method then returns two images, which contradicts D2, and a nucleus-only or cell-only method fills one slot with nothing. |

**Recommendation: T1.** It is what D2 implies, it lets each target come from a
different method (StarDist nuclei, watershed or Cellpose cells) or from an import, and
the seed link records the correspondence that W-308 needs.

## Method input

```python
@dataclass(frozen=True)
class SegmentationInput:
    image: np.ndarray                  # ZYXC, finite, real dtype; Z=1 for a plane
    metadata: ImageMetadata            # the grid and frame (the reference metadata in a FOV run)
    roles: tuple[str, ...]             # one role per channel, in channel order
    sources: tuple[Mapping, ...]       # per channel: the InputChannel fields and its SHA-256
```

* Roles are `nuclear`, `cytoplasm`, `membrane`, `amplicon` and `composite`; each role
  appears at most once. A method declares the roles it accepts and those it requires.
* Seeds are a `SegmentationResult` on the same grid and metadata, with target `nucleus`.
* **The input as a saved artifact.** `FOV.segment` assembles the input from the
  reference-frame images, applying each channel's `prepare` function and the run's
  `projection`, then writes it as `segmentation/<run>/input.ome.tif` with `save_volume`
  (ZYXC, the input's dtype, its `ImageMetadata`) before the method runs. It is in the
  reference frame (or its projection). Its SHA-256 (C-order bytes) and the file's
  SHA-256 are in the run record, and each label file's record names that hash, so a
  label image is tied to the exact input it was computed from. The direct call
  `segment` does not write it; it records the array hash.

## Method registry

`SEGMENTATION_METHODS` uses the mechanism of {doc}`method-registry` unchanged: an exact
frozen config type maps to a frozen `SegmentationSpec`; lookup uses `spec_for`,
`config_type_for` and `names` of `starfinder._registry`; the YAML `method` key holds
`spec.name`, which equals each config's `method` discriminator.

```python
@dataclass(frozen=True)
class SegmentationSpec:
    # Shared fields (method registry page, "Fields of the shared spec")
    name: str                                   # snake_case, unique in the stage
    run: Callable[..., tuple[np.ndarray, dict]] # run(image, config, context) -> (labels, details)
    _: KW_ONLY
    requires: tuple[Dependency, ...] = ()       # imported lazily; extra "stardist" or "cellpose"
    min_shape_zyx: tuple[int, int, int] = (1, 1, 1)
    # Stage-specific fields
    targets: frozenset[str]                     # subset of {"nucleus", "cell"}
    roles: frozenset[str]                       # accepted channel roles
    required_roles: tuple[frozenset[str], ...]  # each entry: one of these roles must be present
    seeds: str                                  # "none", "optional" or "required"
    dimensions: frozenset[int]                  # 2: Z=1 runs as a plane; 3: Z>1 runs as a volume
    models: bool                                # True when the config names a model (known name or path)
    devices: frozenset[str]                     # subset of {"cpu", "cuda"}
```

`run` receives the validated ZYXC image, the config and a `MethodContext`
(`roles`, `seeds` as an array or `None`, `metadata`, `device`, the resolved model folder
or file) and returns a label array and a details mapping (`effective` parameters,
`library_dtype`, counts). The stage wrapper calls it; callers never do.

### Registered methods

Every registered method takes an image (and, for the watershed, seeds) and returns one
label image on the input grid. Nothing that maps labels to labels or images to images
is registered.

| Name | Config | Targets | Roles (required) | Seeds | Dimensions | Models | Devices | `requires` (extra) | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stardist` | `StarDistConfig` | nucleus, cell | `nuclear` or `composite` (exactly one of them) | none | {2, 3}, decided by the model | yes | cpu, cuda | `stardist`, `csbdeep`, `tensorflow` (`stardist`) | W-306 notes sections 3, 6, 8; `models.csv` |
| `cellpose` | `CellposeConfig` | nucleus, cell | `cytoplasm` or `nuclear`; optional second channel `nuclear` with `cytoplasm` | none | {2, 3} (3 needs `do_3d=True`) | yes | cpu, cuda | `cellpose`, `torch` (`cellpose`) | W-306 sections 6, 8, 10 |
| `seeded_watershed` | `SeededWatershedConfig` | cell | one of `cytoplasm`, `membrane`, `amplicon`, `composite` | required | {2, 3} | no | cpu | none | W-306 section 7 |

The library defaults, stored thresholds and boundary behaviors behind the fields below
are from W-306 (notes sections 6 to 8, `models.csv`, `segmentation-calls.csv`) and the
W-305 inventory (`models.md`). They were observed on one host and one field of view per
context, without annotation; the full list of limitations is under "Device and
environments".

Config fields (frozen dataclasses, validated in `__post_init__`):

**`StarDistConfig`** (`method="stardist"`)

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | `None` | A known pretrained model name (`2D_versatile_fluo`); exactly one of `model` and `model_path`. |
| `model_path` | `None` | Folder of a user-trained model (`config.json`, `thresholds.json`, `weights_best.h5`), such as `3D_spleen`. |
| `model_sha256` | `None` | Optional mapping of file name to expected SHA-256 for `model_path`; checked when given. |
| `scale` | required | Input scale passed to `predict_instances(scale=…)`: a number scales Y and X (Z factor 1), a 3-tuple is per axis ZYX. No default (W-305 rule). |
| `prob_thresh`, `nms_thresh` | `None` | `None` uses the model's `thresholds.json`; a number is a sensitivity override and is recorded with `threshold_source: "override"`. |
| `normalize_percentiles` | `(1.0, 99.8)` | csbdeep percentile normalization over the whole image, as the current script. |
| `n_tiles` | `None` | `None` is the current script's tiling: 1×4×4 for a volume, 2×2 for a plane. A tuple is passed as given (ZYX; Y, X for a plane). |

**`CellposeConfig`** (`method="cellpose"`)

| Field | Default | Meaning |
| --- | --- | --- |
| `model`, `model_path`, `model_sha256` | `None` | As for StarDist; the known model is `cpsam_v2`. |
| `diameter` | required | Object diameter [pixels] passed to `eval`; the image is rescaled by 30/`diameter`. `None` must be given explicitly and means no rescale; it is rejected with `do_3d=True` (W-306 section 6). |
| `do_3d` | `False` | 3D mode on Z>1; a Z>1 input with `False` raises. |
| `anisotropy` | `None` | Z spacing over Y, X spacing; required with `do_3d=True`. |
| `flow_threshold` | 0.4 | Library default, recorded. |
| `cellprob_threshold` | 0.0 | Library default, recorded. |
| `tile_overlap` | 0.1 | Library default, recorded (W-306 choice 8). |
| `bfloat16` | `True` | Library default, recorded; float32 was not measured. |
| `min_size` | 15 | Library default [pixels], recorded. |
| `normalize_percentiles` | `(1.0, 99.0)` | Passed to Cellpose as a new mapping on every call, because `CellposeModel.eval` writes into its module-level default (W-306 section 6). |

**`SeededWatershedConfig`** (`method="seeded_watershed"`)

| Field | Default | Meaning |
| --- | --- | --- |
| `sigma_um` | 1.5 | Gaussian smoothing of the stain [µm], converted per axis with the input's spacing; an axis of length 1 is not smoothed. Provisional, from the W-306 prototype; W-305 used 1.514 µm on tissue-2D. |
| `threshold` | `"otsu"` | Otsu's threshold of the smoothed stain, or a number on the `img_as_float` scale. |
| `compactness` | 0.0 | Passed to `skimage.segmentation.watershed`. |
| `connectivity` | 1 | Passed to `skimage.segmentation.watershed`. |

The watershed needs a spacing: an input whose metadata has no `spacing_zyx` raises
`ValueError`.

### Checks the stage wrapper applies to every method

In this order, before and after `spec.run`:

1. **Config.** `spec_for(SEGMENTATION_METHODS, config, …)` with `TypeError("no
   segmentation method is registered for …")`; `LabelImportConfig` raises
   `TypeError("import_labels imports masks; it is not a segmentation method")`.
2. **Target.** `target` is in `spec.targets`, else `ValueError`.
3. **Input.** A `SegmentationInput` with a finite ZYXC image, one role per channel,
   known and unique roles, every required role present and no role outside
   `spec.roles` (`ValueError` naming the role).
4. **Dimensionality.** Z=1 needs 2 in `spec.dimensions`, Z>1 needs 3; for model methods
   the model's own dimensionality must match too (StarDist `config.json` `n_dim`), so a
   3D model rejects Z=1 and a 2D model rejects Z>1 (`IncompatibleGeometryError`; W-306
   choice 4). The shape must be at least `min_shape_zyx`.
5. **Seeds.** Required, optional or refused per `spec.seeds`; a seed image must be an
   integer `SegmentationResult` with target `nucleus`, the input's shape and equal
   metadata (`IncompatibleGeometryError`).
6. **Device.** `device` in `spec.devices` ("Device and environments").
7. **Dependencies.** `require(spec, "segmentation method",
   SegmentationBackendUnavailableError)`, naming the module and the extra.
8. **Model.** For `spec.models`, `resolve_model` checks the files and their SHA-256
   before any library call ("Models"), so Cellpose's silent fallback for a missing path
   (W-306 section 6) cannot happen.
9. **Run.** `spec.run(image, config, context)`.
10. **Output.** An integer array of the input's ZYX shape (a Z axis the library dropped
    is restored), no negative value, converted to `uint32` by the dtype rule; the
    details' `effective` parameters are recorded.
11. **Record.** The uniform provenance entry with `artifacts` (model files) and
    `execution`; `outcome` `empty` when no label remains.

No method gets an image without its roles, and none returns a label image off the input
grid. There is no foreground gate (baseline discrepancy 2).

## External-mask import

```python
@dataclass(frozen=True)
class LabelImportConfig:
    path: str                          # a TIFF readable by load_volume, YX or ZYX
    target: str                        # nucleus or cell
    geometry: str | None = None        # None: plane for Z=1, volume for Z>1; or extended
    frame: str = "reference"           # the file is declared to be on the reference grid
    relabel: bool = False              # False: keep the file's values

def import_labels(path, *, metadata: ImageMetadata, target: str, geometry=None,
                  relabel=False, label_namespace: str) -> SegmentationResult: ...
```

Import is not a method (D3): it is not in `SEGMENTATION_METHODS`, it takes no image, and
`segment` refuses its config. It produces the same `SegmentationResult`, so the
assignment cannot tell an imported run from a computed one except by its record.

Validation, in order: the file exists and is read with its byte order converted to
native (the W-305 culture references are big-endian; W-306 notes section 6); a YX image
becomes 1×Y×X; the dtype is integer (a float or boolean mask raises `TypeError`); no
value is negative; the shape equals the grid of `metadata` (`IncompatibleGeometryError`);
when the TIFF carries `starfinder_metadata`, it must equal `metadata`, otherwise the
caller's `metadata` is taken as a declaration and recorded as such
(`"metadata_source": "declared"`); values above 2³²−1 raise. With `relabel=True`, values
are mapped to 1…n in increasing order and the map is recorded.

Provenance: the run record has `methods: []` and an `import` entry with the path, the
file's SHA-256, the array's SHA-256, the source dtype and shape, `metadata_source`,
`relabel` and its map, and the target and geometry. CellProfiler outputs, the legacy
`images/stardist_segmentation` files and the W-305 references all enter this way.

## Input preparation and label functions

These are plain functions (D4), callable on their own and from `FOV.segment` through
`InputChannel.prepare` and `SegmentationRun.operations`. Each returns its result and a
record mapping (`{"function": name, "config": …, "inputs": [sha256 …], "output":
sha256}`), which the run record keeps.

| Function | Signature | Parameters and units | What it records |
| --- | --- | --- | --- |
| `composite_nuclei_amplicon` | `(nuclear, amplicon, *, config=CompositeConfig()) -> np.ndarray` | `nuclear_quantile` 0.005 and `amplicon_quantile` 0.001 (fractions; the stretch uses q and 1 − q of the whole volume); ZYX inputs on one grid; output uint8 ZYX | the config, the four quantile values reached, both input hashes |
| `enhance_with_flamingo` | `(nuclear, flamingo, *, config=FlamingoEnhancementConfig()) -> np.ndarray` | `flamingo_quantile` 0.005, `nuclear_quantile` 0.001 (fractions), `median_radius_px` 1 (pixels, a disk in each plane); output uint8 ZYX | the config, the quantile values, both input hashes |
| `normalize_percentiles` | `(image, *, p_low=1.0, p_high=99.8, axes=None) -> np.ndarray` | percents of the intensity distribution over `axes` (default all spatial axes); output float32, unclipped | `p_low`, `p_high` and the two values reached |
| `rescale_input` | `(image, metadata, *, scale_zyx) -> tuple[np.ndarray, ImageMetadata]` | per-axis factors (dimensionless); linear interpolation with anti-aliasing; the metadata's spacing is divided by the factors and its `frame_id` extended | factors, input and output shapes |
| `labels_to_grid` | `(labels, metadata, *, target_metadata, target_shape) -> np.ndarray` | nearest neighbour through physical coordinates, onto exactly `target_shape` | both shapes and frames |
| `expand_labels` | `(labels, metadata, *, config=ExpandLabelsConfig(distance, unit, mode)) -> np.ndarray` | `distance` in `pixel` or `um`; `mode` `planar` (each Z plane, the current behavior) or `volumetric` (3D Euclidean distance in physical units); no default distance | the config and the number of voxels added |
| `extend_labels_through_z` | `(labels_2d, stain, metadata, *, config=ZExtensionConfig()) -> np.ndarray` | `median_um`, `threshold` (`"otsu"` or a number), `min_area_um2`, `dilation_um`, `fill_holes` (twice for cells); legacy values in pixels are translated by the workflow adapter with the configured spacing | the config, the threshold reached, the source plane run's hash; the result's geometry becomes `extended` |

Rules:

* `rescale_input` and `labels_to_grid` serve nucleus detection on a shrunk input by a
  method without its own scale parameter. `labels_to_grid` maps through physical
  coordinates onto the exact target shape, so the odd-size grid change of the current
  round trip cannot happen. A cell run, or a run with seeds, rejects an input whose
  metadata is not the reference grid. StarDist and Cellpose scale internally (`scale`,
  `diameter`), which W-306 found returns labels on the input grid, so they never use
  these two functions.
* `expand_labels` is the only expansion. It is applied once, by a run's `operations`,
  and recorded; `expand_labels(mode="planar", unit="pixel")` reproduces the current
  script and the golden digests.
* `extend_labels_through_z` takes a `plane` label image (Z=1) and a ZYX stain on the
  same Y, X grid and returns a ZYX label image whose geometry is `extended`. It is the
  Python form of `create_3d_segmentation.m` per FOV ({doc}`segmentation-baseline`),
  without the `Cyto` subtraction, which is a compartment (W-308).

**Reuse of `PREPROCESSING_METHODS`.** None of these functions is registered as a
preprocessing method:

* A preprocessing step returns an image of its input's shape within one round
  ({doc}`preprocessing-contract`, `StepResult`). The composite combines two rounds
  (the morphology DAPI and the reference merged image) into one new image, and the
  Flamingo enhancement combines two channels into one, so neither fits.
* The composite and Flamingo stretches are not `percentile_normalization`: they use
  linear-interpolated quantiles and map onto the dtype range before converting to
  float, while `percentile_normalization` uses inverted-CDF percentiles and keeps the
  dtype. Reusing it would change the pinned golden digests.
* StarDist needs csbdeep's float32, unclipped normalization at 1 and 99.8 to keep the
  W-306 parity; `percentile_normalization` keeps the input dtype, so `normalize_percentiles`
  is separate.
* `rescale_input` changes the grid, which no preprocessing step may do.
* `expand_labels`, `labels_to_grid` and `extend_labels_through_z` operate on labels.

What is reused is an application policy, not a method: a run's optional `projection` is
a `ProjectionConfig` applied with `starfinder.preprocessing.project_image`, as
`save_reference_image` does, and the projected metadata is `metadata.projected(…)`.

## Models

Evidence: W-306 `tables/models.csv` (SHA-256 recomputed from the staged files) and
W-305 `20261004T0630Z-summary/models.md` (the inventory with configuration).

### Known-models table

`KNOWN_MODELS` lists pretrained models that Starfinder can verify. It sets no default
model.

| Method | Model | Files (bytes, SHA-256) | Source | Dimensionality, channels | Stored thresholds | Training pixel size |
| --- | --- | --- | --- | --- | --- | --- |
| `stardist` | `2D_versatile_fluo` | `config.json` (1,021, `836da16282c3e0db1ba2e58f377e977419887ace8c737ebde16a094d58d50f74`), `thresholds.json` (39, `5cd6aac6e923f8659b63a9e297485920d45a59dad35371ddb8b6ae4be398d805`), `weights_best.h5` (5,771,480, `42202bd269c8106782316f1a2c75afb3f5ffa65e525c2e155ee0ced3a95da349`) | `stardist-models` release v0.1, `python_2D_versatile_fluo.zip` (SHA-256 `4ad678d0758eed6e55625f1b5ae30771e59adb79f1239e09b9772eac8846c3dd`, the hash StarDist 0.9.2 registers) | 2D YX, 1 channel; grid (2, 2); 32 rays | prob 0.4791, nms 0.3 | not documented (fluorescent nuclei, DSB 2018 subset); train patch 256×256 |
| `cellpose` | `cpsam_v2` | `cpsam_v2` (1,233,586,851, `0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667`) | `https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam_v2` (the Cellpose 4.2.1.1 URL), fetched in W-305 | 2D network, 3D by planes; 1 to 3 channels | none; library defaults flow 0.4, cellprob 0.0 | none: images are rescaled to a 30-pixel diameter |

Models live in the §2.7 weights cache (`STARFINDER_WEIGHTS_DIR`, default
`~/.cache/starfinder/weights`) as `<root>/<method>/<model>/`, verified by
`resolve_model` the way `resolve_weights` verifies detector weights
({doc}`learned-detectors`). Fetching is an explicit command
(`starfinder weights fetch stardist 2D_versatile_fluo`), extended from §2.7; for
`cpsam_v2` the published hash could not be checked without network access (W-306
notes section 8), so its table hash is the W-305 download's.

### User-trained models

A user-trained model is given by `model_path`, never by name. `resolve_model` requires
the folder and the files the library loads (`config.json`, `thresholds.json` and
`weights_best.h5` for StarDist; the file itself for Cellpose), computes each file's
SHA-256, compares them with `model_sha256` when it is given, and records them in the
provenance entry's `artifacts` (`name`, `path`, `sha256`, `source: "path"`).
`3D_spleen` is such a model: trained in the lab (W-305 inventory
`~/wanglab/Tools/stardist_models/3D_spleen`), files `config.json` (1,206,
`130a82e9…`), `thresholds.json` (40, `b350f3e6…`) and `weights_best.h5` (18,244,960,
`30a5bbc8987dd7e58b6db57705780d8adb317c64ed7d82fcc4369eb7f6da8091`), 3D ZYX with 1
channel, grid (1, 4, 4), 96 rays, stored anisotropy (2.286, 1, 1), train patch
32×128×128, stored thresholds prob 0.6429 and nms 0.5; its training pixel size and
stain are not recorded. The documentation lists these values; `KNOWN_MODELS` does not,
because Starfinder cannot fetch it.

### No entry downloads a model

`segment`, `FOV.segment`, `import_labels` and the workflow adapter never download.
StarDist models are loaded with `StarDist2D(None, name=…, basedir=…)` or `StarDist3D`,
which read the folder and attempt no download; Cellpose with
`CellposeModel(pretrained_model=<absolute file path>)` after the file and its hash are
checked, with `HF_HUB_OFFLINE=1` set for the call (W-306 `tables/loading-probes.json`).
`StarDist2D.from_pretrained` and a bare Cellpose model name are never used. A missing
file raises `MissingModelError` naming the method, the model or path and the fetch
command; a hash difference raises `ModelHashMismatchError` naming the file and both
hashes.

### Input scale

`StarDistConfig.scale` and `CellposeConfig.diameter` are required: none of the three
models records a training pixel size, so no scale can be derived from the spacing. The
values W-305 found (`3D_spleen` 1.0 on LN; `2D_versatile_fluo` 0.25 on the tissue-2D PI
stain; Cellpose diameter 240 for culture cells) are observations on one field of view
each, not recommended settings, and the documentation says so. W-306 recorded how the
outputs move with the scale (LN at scale 0.5, 0.75 and 1.0: 3, 253 and 426 labels; GPU
rows of `segmentation-calls.csv`).

## Device and environments

### Device

| Option | Rule | Effect on existing files | Effect on the lock | Effect on the engines |
| --- | --- | --- | --- | --- |
| **V1. A `device` keyword of the segment entry (recommended)** | `segment(…, device=…)` and `FOV.segment(…, device=…)` accept `"cpu"` (default) and `"cuda"`; a method accepts only the devices of `spec.devices`. `ExecutionConfig.device` and `_execution.DEVICES` stay CPU-only for `FOV.run`. | `_execution.py` gains a per-stage device check and a TensorFlow entry in `execution_record`; `test_spot_finding_registry.py::test_device_accepts_only_cpu` and `test_spot_finding_workflow_key.py::test_the_rule_level_device` pass unchanged. | None. | A rule passes `device` from its own Python-only parameter; GPU rules need Snakemake GPU resources (§2.13). |
| V2. `ExecutionConfig.device` accepts `"cuda"` | `_execution.DEVICES` becomes `("cpu", "cuda")`; spot finding and `FOV.run` reject `"cuda"` in their wrappers. | Those two tests need named edits (construction no longer raises; the stage does). | None. | One device key for every rule. |

**Recommendation: V1.** D1 separates the segment entry from `FOV.run`, so it need not
share `FOV.run`'s execution settings, and no existing test changes. The spot-finding
contract expected segmentation to reuse `ExecutionConfig.device`; the worker notes list
this as an open choice.

What is recorded (the execution entry of the provenance entry): `device`; `framework`
with its name (`tensorflow` or `torch`), version, CUDA build (`None` for CPU builds) and,
for `"cuda"`, the GPU name and driver version; for TensorFlow, whether memory growth was
enabled; and `threads` as in {doc}`spot-finding-contract` plus TensorFlow's intra- and
inter-op thread counts. Starfinder enables TensorFlow memory growth before the first
model call on `"cuda"` and changes no thread setting.

Behavior to state with the device (W-306 notes sections 4 and 5, `comparisons.csv`):

* For a fixed device and environment, labels are bitwise repeatable across processes
  and between one and four CPU threads, in the one repeat case per method.
* CPU and GPU results are not equal: `3D_spleen` gave 425 vs 426 labels on LN
  `dapi_round4`, all 425 matched at IoU ≥ 0.5 with median 0.9997; Cellpose cells 34 vs 34
  with median 0.9952. A validation compares CPU with GPU by label count (within max(1,
  1 %)), matched fraction (at least 99 % at IoU ≥ 0.5) and median matched IoU (at least
  0.99), the tolerances W-306 supports; exact digests hold on CPU only.
* StarDist's non-maximum suppression runs on the CPU even on `"cuda"` (38.7 s of CPU
  time in a 54.6 s GPU call).
* TensorFlow keeps the GPU memory it grows to (8.9 GB after the tiled LN calls, 17.1 GB
  after the untiled one), so one learned backend runs per process on the GPU (W-306
  choice 6).
* `3D_spleen` on LN 50×512×512 takes 226 s on one CPU thread, 58 s on four and 54.6 s on
  the GPU; `cpsam_v2` on a 1024² two-channel culture image at diameter 240 takes 203 s,
  69 s and 1.5 s; Cellpose 3D mode on 42×512×512 is projected at 3,015–3,833 s per CPU
  call and was run on the GPU only (6.2 s) (`cost-summary.csv`).

### Environments

| Option | Plan | Effect on existing files | Effect on the lock | Effect on the engines |
| --- | --- | --- | --- | --- |
| **E1. One environment, backends as optional extras (recommended)** | Extras `stardist` and `cellpose` in `pyproject.toml`, resolved in the one `uv.lock` with the §2.7 extras; GPU hosts install the `+cu126` torch build and TensorFlow's `and-cuda` extra in their own environment. | `src/python/pyproject.toml` and `src/python/uv.lock` (below). | tensorboard 2.21.0 → 2.20.0 is the only locked version change; 20 packages are added (W-305 rows B, C, D; W-306 notes section 1). | `stardist_segmentation` drops its `conda:` directive and runs in the project environment; the legacy conda environment is kept only to regenerate the parity outputs. |
| E2. Separate environments per backend | The lock is unchanged; StarDist keeps the conda environment `{envs_path}/stardist`, Cellpose gets another; the package calls them out of process. | Workflow environment files per backend. | None. | A method would not be an in-process callable, which contradicts D2; the legacy environment has Python 3.9 and no Starfinder. |
| E3. Backends as default dependencies | The same packages in `[project] dependencies`. | `pyproject.toml`, `uv.lock`. | The same version change; every install gains about 1.9 GB (TensorFlow) and 0.2 GB (Cellpose) (W-306 `environment-packages.csv`). | Every rule's environment holds TensorFlow. |

**Recommendation: E1**, the W-305 row D environment that W-306 adopted (W-306 notes
section 10, unresolved choice 1). W-306 audited it: all 312 locked packages installed,
exactly three differing (tensorboard 2.20.0 and the `+cu126` torch and torchvision), 36
packages added in the GPU spike, and every call of `segmentation-calls.csv` ran there;
StarDist, Cellpose, TensorFlow and PyTorch import together in one process.

**Versions** (W-306 notes section 10, `provisioning/freeze.txt`): StarDist 0.9.2,
csbdeep 0.8.2 (already locked), TensorFlow 2.20.0, Keras 3.15.1, tensorboard 2.20.0;
Cellpose 4.2.1.1 with the locked torch 2.7.1 and torchvision 0.22.1. Rows A1 and A2 of
the W-305 matrix exclude the alternatives: TensorFlow 2.21.0 needs h5py < 3.15 (the lock
has 3.15.1), and 2.19 needs numpy < 2.2 (the lock has 2.2.6). In these versions the
step-by-step prototype reproduces the legacy script exactly on CPU (W-306 section 3).

**Cellpose is integrated**, as W-306 recommends (notes section 10): it loads offline from
an explicit path, runs in the same environment, repeats exactly for a fixed device,
covers 2D and 3D (by planes) and costs about 1.5 s per 1024² image on the GPU. The
contract absorbs the five behaviors W-306 lists: the silent fallback for a missing model
path (check 8), the uint16 to uint32 switch (the dtype rule), the dropped Z axis for one
plane (check 10), a diameter that must be explicit (`diameter` required) and its CPU cost
(documented; 3D mode is GPU-only in practice). Whether tile seams change dense real
images is open (W-306 choice 10); `tile_overlap` is recorded and the input window is
fixed by the FOV.

**The proposed `pyproject.toml` change.** For the W-309 gate; not made by any §2.9
specification issue:

```toml
[project.optional-dependencies]
# Segmentation backends (§2.9). TensorFlow 2.20.0 and torch 2.7.1 have no cp314
# wheels, so the extras are empty on Python 3.14 and later.
stardist = [
    "stardist==0.9.2; python_version < '3.14'",
    "csbdeep==0.8.2; python_version < '3.14'",
    "tensorflow==2.20.0; python_version < '3.14'",
]
cellpose = [
    "cellpose==4.2.1.1; python_version < '3.14'",
    "torch==2.7.1; python_version < '3.14'",
    "torchvision==0.22.1; python_version < '3.14'",
]

[tool.uv]
# huggingface-hub (pulled in by piscis) is held at the version W-266 measured;
# newer releases would also move the locked click and idna. keras and tensorboard
# are held at the versions W-305 and W-306 measured with TensorFlow 2.20.0.
constraint-dependencies = ["torch==2.7.1", "torchvision==0.22.1", "huggingface-hub==1.16.1",
                           "keras==3.15.1", "tensorboard==2.20.0"]
```

The two extras are added after `piscis`; nothing else in the file changes. torch and
torchvision are listed in `cellpose` for the same reason as in the §2.7 extras: uv applies
`[tool.uv.sources]` (the CPU index on Linux) to direct dependencies only.

**The expected `uv.lock` change**, from the W-305 resolution rows B and C and the W-306
audit (`provisioning/lock-diff.txt`, `spike-added-packages.txt`):

* `tensorboard` 2.21.0 → 2.20.0 (its wheel URL and hash), the only change to a locked
  package; spotiflow, which requires tensorboard, accepts 2.20.0.
* 20 new packages: `stardist` 0.9.2, `tensorflow` 2.20.0, `keras` 3.15.1, `astunparse`
  1.6.3, `flatbuffers` 25.12.19, `gast` 0.7.0, `google-pasta` 0.2.0, `libclang` 18.1.1,
  `ml-dtypes` 0.6.0, `namex` 0.1.0, `opt-einsum` 3.4.0, `optree` 0.20.0, `termcolor` 3.3.0,
  `wheel` 0.48.0 (TensorFlow's closure); `cellpose` 4.2.1.1, `fastremap` 1.20.0,
  `fill-voids` 2.1.2, `opencv-python-headless` 5.0.0.93, `roifile` 2026.9.22,
  `segment-anything` 1.0 (Cellpose's closure).
* The `starfinder` package entry gains the `stardist` and `cellpose` optional
  dependencies and the two constraints in its metadata.
* No NVIDIA library and no `triton`: the lock keeps the CPU torch build; the NVIDIA CUDA
  12.6 libraries and triton 3.3.1 (3.72 GB, W-306 notes section 1) come only with the
  `+cu126` build on GPU hosts.

Verification for whoever applies it: `uv lock`; `uv export --locked --all-extras
--all-groups --no-hashes` before and after, whose difference must be exactly the list
above; `uv sync --extra stardist --extra cellpose` in a fresh environment, then the
routine test gate and the `learned` parity test. W-305 resolved only Python 3.12 on Linux
x86_64; `uv lock` resolves every Python version from 3.10 and every platform, where it may
add platform-specific entries or find no TensorFlow 2.20.0 wheel. If it does, the marker of
the `stardist` extra narrows to the versions with wheels, and the gate reviews the
narrower marker.

**Limitations of the evidence** (W-305 `matrix.md`, "What this does not show"; W-306
notes section 12): `uv lock` was not run; TensorFlow 2.20.0 is built against CUDA 12.5.1
and cuDNN 9 and ran on torch's CUDA 12.6 libraries only in the W-305 smoke tests and the
W-306 runs; GPU memory with both frameworks in one process, UGER or GCP GPU nodes and
whole fields of view were not examined (the 50×1496×1496 `3D_spleen` call is an estimate:
3,442 MiB for four dense arrays, about 12,500 MiB peak RSS and 1,930 s by linear
extrapolation, `fov-memory-estimate.json`); one host (GP099-29C, RTX A5000, driver 580);
few images, one field of view per context, and no annotation, so no accuracy statement;
the four-thread run covers one case per model and the scale and threshold rows ran on the
GPU only; the 3D seam conclusion rests on one LN crop; the CPU costs of GPU-only calls
are projections; no published hash was checked for `cpsam_v2`; the legacy environment's
parity timings come from a network share and are not costs.

## Workflow configuration

A Python-only top-level `segmentation` block describes a plan:

```yaml
segmentation:
  device: cpu
  runs:
    - name: nucleus
      target: nucleus
      inputs:
        - {role: nuclear, round: round4, channel: ch04}
      method: stardist
      model_path: /absolute/stardist_models/3D_spleen
      scale: 1.0
      operations:
        - {operation: expand_labels, distance: 4, unit: pixel, mode: planar}
    - name: cell
      target: cell
      seeds: nucleus
      inputs:
        - {role: amplicon, reference_merged: true}
      method: seeded_watershed
      sigma_um: 1.5
```

Each run's keys besides `name`, `target`, `inputs`, `seeds`, `projection`, `operations`
and `method` are exactly the init fields of the method's config (or of
`LabelImportConfig` with `method: import`); `method` takes the `SEGMENTATION_METHODS`
names, and a default-tier test keeps the schema list equal to the registry, as for the
other stages ({doc}`method-registry`, "YAML naming per stage"). The block is rejected
unless `backend: python`.

**Translation of the legacy keys.** The workflow adapter (`dataset/workflow.py`, a new
`_segmentation` beside `_nuclei_registration`) turns `rules.stardist_segmentation.parameters`
into a one-run plan when the `segmentation` block is absent:

| Legacy key | Translation |
| --- | --- |
| `stardist_base_path`, `stardist_model_name` | `model_path = <base>/<name>`; the files' SHA-256 values are recorded. A name equal to a known model with the base path of the weights cache resolves through `KNOWN_MODELS`. |
| `prob_thresh`, `nms_thresh` | `prob_thresh`, `nms_thresh`; recorded with `threshold_source: "stored"` when they equal the model's `thresholds.json`, else `"override"`. |
| `rescale` | `true` becomes `scale` 0.5 in Y and X inside StarDist, so the labels are rendered on the input grid instead of the legacy round trip; the change is recorded in `docs/migration.md`. `false` becomes `scale` 1.0. |
| `expand_labels`, `distance` | `operations: [{operation: expand_labels, distance: <distance>, unit: pixel, mode: planar}]` when `expand_labels` is true. |
| `segmentation_input_folder` | `overlay` (default): one `composite` channel prepared by `composite_nuclei_amplicon` from the nuclear stain and the reference merged image; `DAPI`: one `nuclear` channel; `flamingo/enhanced_DAPI`: one `nuclear` channel prepared by `enhance_with_flamingo`; any other folder: that image, read as one `nuclear` channel declared to be on the reference grid. |
| (none) | `target`: a new Python-only key of the legacy block; without it, `cell` for `overlay` and `nucleus` otherwise, recorded as `"target_source": "legacy_default"`. |
| `create_nuclei_amplicon_overlay.parameters.maximum_projection`, top-level `maximum_projection` | the run's `projection` (`ProjectionConfig()` when true). |
| `rotate_angle`, `dapi_round` | not read by segmentation: inputs come from the FOV's rounds after `FOV.run`'s rotation and registration; the adapter checks that exactly one input file matches and raises naming the matches. |

`rules.reads_assignment.parameters.expand_labels` and `dilation_distance` are W-308's;
this page only requires that a label image is expanded once.

**Which rule changes belong to §2.9 and which to §2.13.** This page assigns to §2.9 the
changes that keep every rule's name, inputs and outputs, and to §2.13 the changes to rule
topology, names, outputs and engine settings, following the W-152 boundary comment
(2026-09-28) as the issue cites it; the comment's text was not among this issue's
inputs, so W-309 checks the split.

| Change | Section |
| --- | --- |
| The bodies of `stardist_segmentation.py`, `create_nuclei_amplicon_overlay.py` and `enhance_dapi_with_flamingo.py` become adapter calls (like `nuclei_registration.py`), keeping the rule names, inputs, outputs and file names; the composite and the Flamingo enhancement keep their golden digests | §2.9 |
| `stardist_segmentation` drops `conda: {envs_path}/stardist` and runs in the project environment with the `stardist` extra (needed by the line above) | §2.9 |
| The schema gains the Python-only `segmentation` block, the legacy block's `target` key and a Python-only `device`; a default-tier test keeps the method list equal to the registry | §2.9 |
| A method-agnostic rule per run with outputs `images/segmentation/<run>/{fovID}.tif` and the record; retiring `stardist_segmentation` | §2.13 |
| Declaring `nuclei_registration`'s image outputs; connecting `enhance_dapi_with_flamingo` to them; retiring `rotate_nuclei` as a segmentation input | §2.13 |
| GPU resources and one-backend-per-process rules on GPU nodes | §2.13 |
| Removing `create_segmentation_preview.py` | §2.13 |
| Removing the second expansion in `reads_assignment.py` | W-308 and §2.13 |

MATLAB rules, shared MATLAB keys and filenames do not change.

## Tests the implementation changes

* `test/test_segmentation_golden.py`: every pinned digest stays. Named edits: tests are
  added that compute the composite and the Flamingo enhancement with
  `composite_nuclei_amplicon` and `enhance_with_flamingo` and assert
  `COMPOSITE_DIGESTS` and `FLAMINGO_DIGEST`; that `expand_labels(mode="planar",
  unit="pixel", distance=4)` on the stand-in labels, cast to `uint16`, gives
  `LABELS_3D[(False, True)]` and `LABELS_2D[(False, True)]`; and that `labels_to_grid` on
  the even 16×32×32 → 16×64×64 case gives the values of `RESTORED_LABELS`. When the script
  bodies become adapter calls, `run_script` is replaced by those package calls and
  `test_the_helper_follows_the_script` is removed, because the cited lines leave the
  script; `legacy_stardist_steps` stays as the frozen legacy reference, with the four
  legacy-behavior tests.
* A new `learned` parity test copies `parity/parity_expected.npz` of the W-306 run into
  `src/python/test/data/segmentation_parity.npz` (357 KB, sha256 `4a5df997…`) and runs the
  `stardist` method on CPU with `scale` 1.0, `normalize_percentiles` (1, 99.8), the
  script's tiling, the stored thresholds and, for the `expand1` arrays, planar expansion
  by 4 pixels. It asserts exact equality, after a cast to `uint16`, with the six
  `rescale0` arrays of P1, P2 and P3. `2D_versatile_fluo` comes from the weights cache,
  `3D_spleen` from a path in an environment variable. The `rescale1` arrays are not
  reproduced (the package has no legacy round trip) and stay records.
* A subsystem marker `segmentation` is added to `pyproject.toml` and
  `SUBSYSTEM_MARKERS` (`test/conftest.py`); the golden test keeps `workflow` and gains it.
* New tests: registry and checks (each check above with its error), the dtype rule
  (int32, uint16 and uint32 inputs, a negative value, the 65,536 case), import
  validation, the run record and `FOV.load_segmentation` round trip, model resolution with
  missing and changed files, and the workflow key.
* `test_spot_finding_registry.py`, `test_spot_finding_workflow_key.py`,
  `test_checkpoints.py`, `test_coordination_contract.py`, `test_fov.py`, `test_e2e.py`,
  `test_workflow_scripts.py` and every other existing test pass unchanged.

## `docs/migration.md` entries

The implementation adds: the `starfinder.segmentation` module and its names; the
`segmentation` YAML block; the legacy translation, with two intentional changes, the
output dtype of `images/stardist_segmentation/{fovID}.tif` (uint16 → uint32) and
`rescale: true` rendered on the input grid; no foreground gate, so an image without
foreground gives an empty label image instead of an error; the `stardist` and
`cellpose` extras; the `device` keyword of the segment entry.

## Exclusions

No assignment, correspondence, compartment, count or export specification (W-308); no
transcript-based or combined segmentation, training or fine-tuning; no managed
CellProfiler execution (its masks enter through `import_labels`); no accuracy claim and
no comparison between segmenters (E04, W-123); no default model and no recommended scale.
