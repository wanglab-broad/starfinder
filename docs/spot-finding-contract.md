# Spot-finding contract

Status: Accepted (W-268, 2026-10-01, at b5fcb7f); amended at the W-276 review, 2026-10-01

This page proposes the §2.7 spot-finding contract: four selectable pipeline
methods (Starfinder local maxima, the native Starfish LoG, native-3D Spotiflow
and Piscis stack mode) in 3D and Z=1, explicitly named pretrained weights,
per-channel overrides with recorded effective settings, detection in a named set
of rounds, compact diagnostics, and the dependency plan for the learned
detectors. It builds on the {doc}`method-registry` design, whose registry fields,
lookup, dependency and provenance rules it uses unchanged. The current behavior is
recorded in {doc}`spot-finding-baseline`; the numerical methods and the
engineering validation design are in {doc}`spot-finding-algorithms`. Nothing here
is implemented yet.

The evidence comes from W-266 (versions, weights, CPU behavior and Starfish parity;
run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-266/20260930T225252Z-967e52bd`,
handoff `worker-notes.md`, tables `detectors.csv` and `known-weights.csv`) and
W-265 (GPU conditions on GP099; `runs/W-265/20260930T2137Z-gpu-matrix/matrix.md`).

## Settled decisions this page follows

Decided by Jiahao in the §2.7 planning session (W-152 comment "§2.7 planning
decisions (2026-09-30)"):

1. **D1** The Starfish LoG is a native reimplementation of starfish
   `BlobDetector` (`is_volume=True`, no reference image, one round and channel at
   a time), with parity against starfish `1fb00cbc` saved as test data.
2. **D2** §2.7 is CPU-only; the `device` setting accepts only `"cpu"` and is a
   cross-stage execution setting.
3. **D3** Starfinder manages the weights: an explicit fetch command, a known-weights
   table with SHA-256 that lists known weights and sets no default, loading from
   explicit paths that bypass the libraries' caches; `FOV.run` never downloads.
   Tests skip when an extra is not installed and fail when it is installed but its
   weights are missing.
4. **D4** One extra per learned detector; torch and torchvision from the PyTorch
   CPU index on Linux; torch 2.7.1 pinned through `[tool.uv]
   constraint-dependencies`; the lock change is approved at W-268 and applied as a
   batch-preparation commit.
5. **D5** Six known weights; for Z=1 Spotiflow uses a 2D model and a model of the
   wrong dimensionality raises an error (no automatic switching); Piscis uses one
   file in stack mode (3D) and plane mode (Z=1); LoG squeezes a single plane;
   native `scale` parameters are explicit, default 1 and are recorded; native
   parameter defaults are kept and recorded per channel.
6. **D6** Detection in a named set of rounds, default the reference round only;
   §2.8 owns the readout mode, gene mapping and direct-mode wiring; the barcode-mode
   namespace, `spot_id` and columns stay unchanged, joins use
   `(namespace, spot_id)`, the round is explicit and coincident candidates are
   never merged.
7. **D7** The golden test of {doc}`spot-finding-baseline`.
8. **D8** The validation design S1–S16 of {doc}`spot-finding-algorithms`.

Jiahao accepted on 2026-10-01 the W-266 recommendations that this page uses: the
learned extras carry the marker `python_version < "3.14"`; the Piscis model is
always named, with the `Piscis` class and threshold 0.5, and the contract does not
choose between `20230905` and `20251212`; Piscis loads through a hash-checked
absolute-path wrapper; Piscis stack-mode Z is specified as it is (integer Z,
vertically aligned spots merge); the wrapper raises an error on Spotiflow shapes
that would otherwise return nothing; `scale=1` is fixed and resampling is
explicit; LoG keeps starfish's absolute threshold on dtype-scaled [0, 1] images;
the local-maxima high-density noise threshold is flagged, not changed; the
training pixel sizes retrieved by the operator are accepted, with both Piscis
models recorded as "not published by the authors"; Piscis seam behavior is
documented here.

## Registry entries

Spot-finding methods are registered in `SPOT_FINDING_METHODS`, a public
module-level `dict[type, SpotFindingSpec]` in `starfinder.spot_finding`
(registry move 4). The shared fields (stable `name`, exact config type as the key,
`run`, `requires`, `min_shape_zyx`), the lookup helpers, the dependency rule and
the provenance entry are those of {doc}`method-registry` ("Fields of the shared
spec", "Shared and stage-specific capabilities" and "Provenance in run.json").
Each config's `method` discriminator equals its spec name. `SpotFindingSpec` adds:

| Field | Meaning |
| --- | --- |
| `pipeline` | `True` when `FOV.find_spots` and `PipelineConfig.spot_finding` accept the method. |
| `dimensions` | `frozenset` of `2` and/or `3`: `2` means a Z=1 input is detected as a YX plane; `3` means Z>1 is detected in 3D. |
| `output_columns` | The spot-table columns besides `spot_id`, in order; a trailing `?` marks an optional column. |
| `weights` | `True` when the config names pretrained weights from the known-weights table. |

`run(image, config, context)` is the private method function. It returns the
method's table (the declared columns, without `spot_id`) and its per-channel
diagnostics. `find_spots` calls it after the checks below, then adds `spot_id`,
validates the columns and assembles the diagnostics; callers never call it
directly. `min_shape_zyx` applies to 3D inputs, and a Z=1 input is checked against
its last two entries. A smaller input, or Z=1 for a method without `2` in
`dimensions`, raises `IncompatibleGeometryError` before the method runs.

| Name | Config | `pipeline` | `dimensions` | `min_shape_zyx` | `output_columns` | `requires` (extra) | `weights` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `local_maxima` | `LocalMaximaConfig` (exists; gains `merge_radius_zyx`) | yes | 2, 3 | (1, 1, 1) | `z, y, x, channel, peak_intensity?` | none | no |
| `noise_landmark` | `NoiseLandmarkConfig` (exists) | no | 2, 3 | (1, 1, 1) | `z, y, x` | none | no |
| `percentile_centroid` | `PercentileCentroidConfig` (exists) | no | 2, 3 | (1, 1, 1) | `z, y, x` (fractional) | none | no |
| `starfish_log` | `StarfishLogConfig` (new) | yes | 2, 3 | (1, 1, 1) | `z, y, x, channel, peak_intensity, radius` | none | no |
| `spotiflow` | `SpotiflowConfig` (new) | yes | 2, 3 | (7, 6, 6), and the model's minimum (below) | `z, y, x` (fractional), `channel, peak_intensity, probability` | `spotiflow`, `torch` (`spotiflow`) | yes |
| `piscis` | `PiscisConfig` (new) | yes | 2, 3 | (2, 1, 1) | `z` (integer in stack mode), `y, x` (fractional), `channel, peak_intensity` | `piscis`, `torch` (`piscis`) | yes |

Column meanings: coordinates are zero-based voxel indices in float64, with the
voxel centre at the integer. W-266's mean biases on the isolated-spot scenes were
at most 0.13 voxels on every axis (Spotiflow `smfish_3d` X: +0.12), except Piscis
stack-mode Z (−0.32 to −0.43, from its integer Z), so no offset is applied; `channel` is int64;
`peak_intensity` is the original pixel value at the coordinate rounded half to
even; `radius` is starfish's blob radius `round(σ·√ndim)` in voxels (float64);
`probability` is Spotiflow's spot-wise heatmap probability (`details.prob`,
float64, in [0, 1]). Piscis 1.1.0 returns coordinates only (`Piscis.predict` gives
scores only through its intermediate feature maps), so it has no score column.

The names, fields and defaults of the three existing configs do not change, except
the new optional `LocalMaximaConfig.merge_radius_zyx` (default `None`, the current
behavior). Exact-type lookup rejects subclasses, which `isinstance` accepts today;
no caller, script or test subclasses them at `42f652d`.

**Dependency error.** `SpotFindingBackendUnavailableError(ImportError)`, exported
by `starfinder.spot_finding`, is raised through `require(spec, "spot-finding
method", SpotFindingBackendUnavailableError)`, for example `spot-finding method
'spotiflow' requires spotiflow; install the 'spotiflow' extra
(starfinder[spotiflow])`. On Python 3.14 and later, where the extras install
nothing, the message adds `(the extra is not available on Python 3.14 and later)`.
Importing `starfinder.spot_finding` never imports an optional dependency, and
constructing or validating a config never needs one.

## Configs of the new methods and the W-218 option

Every config is a frozen dataclass validated in `__post_init__`, with
`channel_labels: tuple[str, ...] | None = None` as today and a `method` field
(`init=False`) equal to the spec name. Units are voxels unless stated.

| Config | Fields (default) |
| --- | --- |
| `LocalMaximaConfig` | unchanged fields, plus `merge_radius_zyx: tuple[float, float, float] \| None = None`: the opt-in W-218 within-channel merge ({doc}`spot-finding-algorithms`, "Local maxima"). Positive finite radii; the Z radius is unused for Z=1. |
| `StarfishLogConfig` | `min_sigma`, `max_sigma`: a number or a ZYX 3-tuple, required; `num_sigma: int`, required; `threshold: float`, required (starfish `BlobDetector` has no defaults for these four); `overlap: float = 0.5`; `exclude_border: bool \| int = False` (the starfish and `blob_log` defaults). |
| `SpotiflowConfig` | `model: str`, required (a `spotiflow` row of the known-weights table); `prob_thresh: float \| None = None` (the threshold stored with the weights); `min_distance: int = 1`; `exclude_border: bool = False`; `subpix: bool \| None = None` (the model configuration's choice); `n_tiles: tuple[int, ...] \| None = None` (the library's choice); `scale: float = 1.0` (only 1 is accepted). |
| `PiscisConfig` | `model: str`, required (a `piscis` row of the known-weights table); `threshold: float = 0.5`; `min_distance: int = 1`; `input_size: int \| None = None` (the tile side in pixels passed to the `Piscis` constructor, which rounds it to a multiple of 8; `None` keeps the model's 256); `scale: float = 1.0` (only 1 is accepted). The `Piscis` class is used, never `PiscisLegacy`. |

A model name absent from the known-weights table, or a row of the other method,
raises `ValueError` at construction, listing the known names. There is no default
model: every run names its weights.

## Detection plan: per-channel overrides and rounds

`PipelineConfig.spot_finding` accepts a registered config with `pipeline=True` (as
today for `LocalMaximaConfig`) or a `SpotFindingPlan`. The field was named
`detection` until the W-276 review (2026-10-01), which renamed it to the stage name
({doc}`migration`).

Spot finding has a plan, not a recipe. A recipe is a stage's ordered chain of
method steps, as in `PreprocessingRecipe` and `RegistrationRecipe`; spot finding
has no such chain, since it runs one method per run, with per-channel overrides
and a set of rounds, so its stage configuration is a `SpotFindingPlan`.

```python
@dataclass(frozen=True)
class ChannelOverride:
    channel: str                    # a channel label
    config: SpotFindingConfig       # same exact config type as the plan's config

@dataclass(frozen=True)
class SpotFindingPlan:
    config: SpotFindingConfig       # the method and its settings for every channel
    channel_overrides: tuple[ChannelOverride, ...] = ()
    rounds: tuple[str, ...] | None = None   # None: the reference round only
```

* A bare config means `SpotFindingPlan(config)`; `PipelineConfig` stores what it
  was given, so existing constructions and comparisons are unchanged.
* An override replaces the whole config for one channel. Its config must have the
  plan config's exact type (one method per run), and its `channel_labels` must be
  `None` or equal to the plan's. Channels must be unique and must be labels of
  `Dataset.channel_order` (checked by `FOV.run` and `FOV.find_spots`, which fill
  `channel_labels` from it as today). Direct `find_spots` accepts a plan without
  `rounds` and needs `config.channel_labels` when it has overrides. An override
  may change any setting except one that changes the output columns (for
  `local_maxima`, `measure_peak_intensity` must equal the plan's), and plan
  validation raises `ValueError` naming the channel and the field otherwise.
* The effective settings are recorded for every channel, overridden or not:
  `diagnostics["effective_settings"][<label>]` is the effective config serialized
  as `run.json` serializes dataclasses, including native defaults that `None`
  resolved (for example Spotiflow's stored `prob_thresh` 0.4 for `smfish_3d`).
* `rounds` is described under "Detection in several rounds".

## Workflow configuration

The `spot_finding` block of a Python rule keeps `run` and `ref_round` (which must
equal the dataset reference round) and gains, as Python-only keys:

| Key | Meaning |
| --- | --- |
| `method` | A `SPOT_FINDING_METHODS` name with `pipeline=True`, through `config_type_for()`; default `local_maxima`. |
| the config's init fields | Every other key is an init field of the selected config (YAML lists become tuples), for example `exclude_border`, `merge_radius_zyx`, `model`, `prob_thresh`, `threshold`, or one of the `local_maxima` legacy aliases below. Unknown keys raise `ValueError`. |
| `channel_overrides` | A mapping from channel label to a mapping of config fields; each entry becomes `ChannelOverride(label, replace(config, **fields))`. |
| `rounds` | A list of round labels; omitted means the reference round only. |

The legacy keys `intensity_estimation`, `intensity_threshold` and
`min_distance` are aliases for `local_maxima` only, and keep their meaning.
`min_distance` is also a native init field of two other configs. The adapter
handles each key by method, and this table is the only rule:

| Key | `local_maxima` (also when `method` is omitted) | `spotiflow`, `piscis` | `starfish_log` |
| --- | --- | --- | --- |
| `intensity_estimation` | Alias of `threshold_mode`; `ValueError` together with `threshold_mode` | `ValueError` (not a field) | `ValueError` (not a field) |
| `intensity_threshold` | Alias of `threshold_value`; `ValueError` together with `threshold_value` | `ValueError` (not a field) | `ValueError` (not a field) |
| `min_distance` | Alias of `min_distance_voxels`; `ValueError` together with `min_distance_voxels` (today `min_distance_voxels` silently wins) | The native field `min_distance` of `SpotiflowConfig` and `PiscisConfig`, passed unchanged | `ValueError` (not a field) |
| `min_distance_voxels` | The field | `ValueError` (not a field) | `ValueError` (not a field) |

`channel_overrides` entries follow the same rule for the selected method. A
rule-level Python-only key `device` sets `ExecutionConfig.device`.

```yaml
rsf_single_fov:
  parameters:
    device: cpu                       # the only accepted value in §2.7
    spot_finding:
      run: true
      ref_round: round1
      method: spotiflow
      model: smfish_3d                # required: every run names its weights
      prob_thresh: null               # null: the threshold stored with the weights
      channel_overrides:
        ch03: {prob_thresh: 0.5}
```

The schema (`workflow/schemas/config.schema.yaml`, `spot_finding_params`) gains
`method` with an enum kept equal to the pipeline methods by a default-tier test, as
`test_preprocessing_workflow_key.py` does for preprocessing, declares the
Python-only keys, and drops `local` from `intensity_estimation`, which neither
Python nor MATLAB accepts. MATLAB rules, shared keys and filenames do not change.

## Execution device

`ExecutionConfig` gains `device: str = "cpu"`. In §2.7 any other value raises
`ValueError("device must be 'cpu'; §2.7 runs on CPU only")`. Direct `find_spots`
takes the same keyword (`device="cpu"`). The setting belongs to execution, not to
a method, so §2.9 segmentation reuses it unchanged; it is recorded in `run.json`
with the rest of `ExecutionConfig`.

Each detection records an execution entry in its diagnostics and in its provenance
entry (`"execution"`):

* `device`: `"cpu"`;
* `framework`: for torch methods, `torch.__version__` (for example `2.7.1+cpu`),
  `torch.version.cuda` (`None` for CPU builds), and whether MKL and MKL-DNN are
  available; `None` for methods without a framework;
* `threads`: `torch.get_num_threads()` and `torch.get_num_interop_threads()` when
  torch is loaded, and the values of `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `NUMBA_NUM_THREADS` (`None`
  when unset).

Starfinder does not change thread settings. W-266 found that `OMP_NUM_THREADS=1`
holds a whole Spotiflow process to one thread, while `torch.set_num_threads(1)`
alone leaves 15 OS threads and a CPU/wall ratio of 1.9, and that no MKL
workaround is needed. The documentation therefore tells users to set the thread
variables, and the record shows what was in effect.

Enabling `"cuda"` later is a separate decision for §2.9 or after. W-265 showed
that torch 2.7.1+cu118 uses the RTX A5000 on GP099's driver 470 without changing
the lock (the constraint pins the version, and a GPU user swaps in the `+cu118`
build). Before a CUDA run can be accepted, three points need a decision: TF32
convolutions are on by default and changed a conv3d layer by 8.0e-4, so the
learned methods need TF32 off or tolerances instead of digests; a CUDA process
used 4.7 GB of host RSS, above the 4 GiB target; and the GPU is shared.

## Pretrained weights

### Known-weights table

`KNOWN_WEIGHTS` in `starfinder.spot_finding` maps `(method, model)` to a frozen
entry. It lists the weights Starfinder knows how to fetch and verify. It sets no
default. The six entries are W-266's `known-weights.csv`, whose SHA-256 values
were recomputed from the staged files and match the published hashes:

| Method | Model | Dimensionality | Source and revision | File Starfinder downloads (SHA-256, bytes) | File the library loads (SHA-256) | Training pixel size | Native threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spotiflow | `synth_3d` | 3D | `spotiflow-models` release 0.6.0, `synth_3d.zip` (library MD5 `a031f128…`) | `2468125c…`, 263,107,693 | `best.pt` `846d1ef4…` | 0.2 µm voxels (synthetic) | `prob_thresh` 0.3 |
| spotiflow | `smfish_3d` | 3D | release 0.6.0, `smfish_3d.zip` (MD5 `c5ab30ba…`) | `a6c79f76…`, 263,106,200 | `best.pt` `1fdfd62c…` | 0.13 µm YX, 0.48 µm Z | `prob_thresh` 0.4 |
| spotiflow | `general` | 2D | release 0.6.0, `general.zip` (MD5 `9dd31a36…`) | `1da93a82…`, 87,885,382 | `best.pt` `1c357546…` | 0.04 to 0.34 µm (mixed) | `prob_thresh` 0.5 |
| spotiflow | `hybiss` | 2D | release 0.6.0, `hybiss.zip` (MD5 `254afa97…`) | `d6221339…`, 87,945,684 | `best.pt` `fa5d5cb3…` | 0.15, 0.32 and 0.34 µm | `prob_thresh` 0.532 |
| piscis | `20230905` | 2D network; stack mode links planes | Hugging Face `wniu/Piscis` at `9bdefc72cb`, `models/20230905.pt` | `57177963…`, 30,077,822 | the same file | not published by the authors | `threshold` 0.5 (`Piscis` class) |
| piscis | `20251212` | 2D network; stack mode links planes | `wniu/Piscis` at `9bdefc72cb`, `models/20251212.pt` | `e4ec9fe6…`, 30,143,014 | the same file | not published by the authors | `threshold` 0.5 (`Piscis` class) |

The table in the code holds the full 64-character hashes, the URLs, the extracted
file list of each Spotiflow archive with its hashes (recorded by the Spotiflow
implementation issue from the verified archives), the observed minimum input shape
(Spotiflow 3D models: Z ≥ 7 and Y, X ≥ 8; 2D models: Y, X ≥ 6; Piscis: Z ≥ 2 in
stack mode, 1×1 in plane mode) and the provenance of each training pixel size
(operator-retrieved on 2026-10-01 from the Spotiflow documentation; not verified by
a worker). The Piscis library defaults to `20251212` and the E02 proposal named
`20230905`; this contract does not choose between them.

### Fetching, cache and verification

* **Explicit fetch.** `starfinder weights fetch <method> <model> [--dir DIR]`
  (and `fetch_weights(method, model, *, directory=None)`) downloads the source file
  to a temporary name in the target directory and checks its SHA-256 and, for
  Spotiflow archives, the library-registered MD5, and it checks every file the table
  lists for the model, as `verify` and detection do. Only then does it extract (Spotiflow)
  or move (Piscis) the file into place, and it writes a small `starfinder-weights.json`
  with the entry and the per-file hashes. An existing model folder gets the same full
  check, and a present file is never overwritten: a folder whose listed files all verify
  is returned without a download; when listed files are missing and every present one
  verifies, only the missing files are restored from one verified download and the
  record is rewritten; a changed present file raises `WeightsHashMismatchError` and a
  folder without the record raises `FileExistsError`, and neither downloads or changes
  anything.
  `starfinder weights list` prints the table and the local state;
  `starfinder weights verify [<method> <model>]` re-hashes local copies. Only these
  commands use the network.
* **Cache location (proposed; open choice for Jiahao).** `STARFINDER_WEIGHTS_DIR`
  when set, otherwise `$XDG_CACHE_HOME/starfinder/weights` (default
  `~/.cache/starfinder/weights`), with the layout `<root>/<method>/<model>/`:
  the extracted Spotiflow folder, or `<model>.pt` for Piscis. The libraries' own
  caches (`~/.spotiflow`, `~/.piscis/models`, the Hugging Face cache) are never read
  or written.
* **Verification on every run.** Before loading, the wrapper resolves the folder,
  checks that every listed file exists and recomputes its SHA-256 (the loaded
  files are 30 to 142 MB; the cost is not measured yet and is recorded by the
  implementation). Only then does it construct the model from those files. The
  library has already been imported by the registry's `require()` (next
  section); verification comes before model construction, which is the step that
  could otherwise reach a library cache or the network.
* **Loading route.** Spotiflow: `Spotiflow.from_folder(<folder>,
  map_location="cpu")` (W-266). Piscis: `Piscis(model_name=str(<folder> /
  <model>))`, an absolute path without `.pt`, which piscis 1.1.0 resolves to that
  file through pathlib (W-266, accepted). Because the wrapper has already checked
  the file, Piscis never reaches its fallback of JAX conversion or download. An
  upstream request for an explicit path argument is recommended. A loaded model is
  cached per process by `(method, model, SHA-256)`, so the channels and rounds of
  one run load it once.
* **`FOV.run` never downloads.** No detection code path imports the fetch module.
  The validation (S14) runs a detection with network access patched to raise.

### Errors

| Situation | Error |
| --- | --- |
| The extra is not installed | `SpotFindingBackendUnavailableError` (an `ImportError`) naming the module and the extra |
| The weights folder or a listed file is missing | `MissingWeightsError` (a `FileNotFoundError`) naming the method, model, expected path and the fetch command |
| A file's SHA-256 differs from the table | `WeightsHashMismatchError` (a `ValueError`) naming the file and both hashes |
| The model name is unknown, or belongs to the other method | `ValueError` at config construction, listing the known names |

The order at run time keeps the registry mechanism of {doc}`method-registry`
unchanged: (1) `require(spec, ...)` imports each declared dependency, as
`_registry.require` does today, and raises `SpotFindingBackendUnavailableError`
when one is missing; (2) `resolve_weights(method, model)` resolves the cache
folder and recomputes every listed SHA-256, without using the library; (3) the
model is constructed from the verified files (`Spotiflow.from_folder`, the
absolute-path `Piscis`). A missing extra is therefore reported before missing or
changed weights. `resolve_weights` performs step 2 on its own, so its errors can
be tested without the extras. Tests skip when an extra is not installed (`pytest.importorskip`) and fail
with `MissingWeightsError` when the extra is installed but the weights are not
fetched (D3).

### Provenance

Each verified file adds one `artifacts` entry to the method's provenance entry
({doc}`method-registry`, "Provenance in run.json"): `name` (`<method>/<model>`),
`path` (the absolute path loaded), `sha256`, `source` (the URL) and `revision`.
The same entries, the training pixel size and its provenance appear in
`diagnostics["model"]`. `requires` records the installed versions of `spotiflow`
or `piscis` and of `torch`.

## Z=1, dimensionality, scale and tiling

| Method | Z=1 | Z>1 | Wrong dimensionality or small inputs | `scale` | Tiling |
| --- | --- | --- | --- | --- | --- |
| `local_maxima` | YX plane, z=0 (unchanged) | 3D | none; 1 < Z ≤ 2 × `min_distance_voxels` with `exclude_border=True` stays empty, as today, and is recorded as a warning | none | none: the whole image |
| `starfish_log` | single plane squeezed to 2D, z=0 (starfish) | 3D `blob_log` | a 3-tuple `min_sigma` or `max_sigma` with Z=1 raises `IncompatibleGeometryError` (starfish raises a SciPy `RuntimeError` there) | none; σ in voxels, per axis when a tuple | none, so the result equals starfish's whole-volume result; memory is estimated in advance (below) |
| `spotiflow` | needs a 2D model (`general`, `hybiss`) on the squeezed plane | needs a 3D model (`synth_3d`, `smfish_3d`) | a 3D model on Z=1 or on Z < 7, a 2D model on Z>1, or Y or X below the model's minimum raises `IncompatibleGeometryError`; Spotiflow would otherwise return nothing without error (W-266, accepted) | `scale` must be 1 (`ValueError` otherwise; resampling is explicit) | native `n_tiles` (None: chosen from `max_tile_size`; CPU fallback tiles 2048×2048 in 2D and 128×256×256 in 3D); the effective tiling is recorded |
| `piscis` | plane mode (`stack=False`) on the squeezed plane | stack mode (`stack=True`); Z is an integer and vertically aligned spots merge across planes | Z ≥ 2 for stack mode is guaranteed by the dispatch | `scale` must be 1 | native tiles of side T = `round(input_size / scale)` (256 unless `input_size` is set), overlap `rint(0.1 × T)` on axes longer than T, one keep-boundary per overlap; the boundaries are recorded |

No method switches models or modes behind the user's back: the model named in the
config must fit the input, and the only automatic choice is Piscis's plane or
stack mode, which uses the same file (D5). Explicit resampling, when needed, is
the caller's preprocessing step, with coordinates mapped back by the caller. W-266
measured what `scale` ≠ 1 does: Spotiflow raises for 3D models and with sub-pixel
refinement, and Piscis shifts coordinates by about −0.5 px (scale 0.5) or up to
+0.19 px (scale 2) and loses spots depending on the model.

**Piscis seams.** A spot whose prediction lies within a fraction of a pixel of a
keep-boundary can be lost or reported once per adjoining tile: at most two
candidates on an edge and four at a corner in W-266's seam probe. On a 512-pixel
axis the native boundaries are at 242.5 and 472.5. Spots 0.5 px or more from a
boundary gave exactly one candidate, displaced by at most 0.147 px. The contract
keeps Piscis's behavior unchanged and records the keep-boundaries per axis in the
diagnostics, so seam candidates can be identified. Merging seam repeats would be a
Starfinder-side deduplication, which is left as an open choice ({doc}`spot-finding-algorithms`).

**Native thresholds** (units and defaults; details in {doc}`spot-finding-algorithms`):

| Method | Threshold | Units | Native default |
| --- | --- | --- | --- |
| `local_maxima` | `threshold_mode`, `threshold_value` | robust σ (`noise`) or fraction of a maximum | `noise`, 5.0 |
| `starfish_log` | `threshold` | absolute scale-normalized LoG response of the dtype-scaled [0, 1] image (about 0.5 × spot amplitude in 2D and 0.53 × in 3D at matched σ) | none (required) |
| `spotiflow` | `prob_thresh` | probability on the sigmoid heatmap, [0, 1] | stored per model: 0.3, 0.4, 0.5, 0.532 |
| `piscis` | `threshold` | value of the max-pooled sigmoid label map, [0, 1], kept when strictly above | 0.5 |

**LoG intensity scaling.** Integer images are scaled by their dtype maximum to
float32 in [0, 1], as `skimage.img_as_float32` does in a starfish `ImageStack`. Float
images in [0, 1] are used as float32. Float images outside [0, 1] raise
`ValueError`, as starfish's `ImageStack` does. `peak_intensity` still reports the
original pixel value.

## Detection in several rounds

This section is the interface proposed to §2.8. §2.8 owns the readout mode, gene
mapping, direct-mode wiring in `FOV.run` and direct-mode extraction semantics, and
may amend this interface through its own gate. It did so in {doc}`readout-contract`
("Amendments to the option-A interface"), accepted at W-280 and implemented in
W-293: the rules below carry the three amendments.

`SpotFindingPlan.rounds` names the rounds to detect in. `None`, the default, means
the reference round only: every current result, identity, column, checkpoint and
golden digest stays the same. An explicit tuple must hold unique labels of
`RoundState.all_rounds`. `FOV.run` detects each listed round after that round's
registration and post-registration steps, on its detection image, so all
coordinates are on the reference grid. A listed round whose metadata differs from
the reference round's metadata raises `IncompatibleGeometryError`. Each round is
detected with the same plan (method, overrides), and thresholds are computed per
round from that round's image.

### Identity representations

| Option | Representation | Effect on the golden digests | Effect on the `candidates` checkpoint | Effect on §2.8 |
| --- | --- | --- | --- | --- |
| **A. One candidate set with an explicit `round` column (recommended)** | One `SpotFindingResult` for all listed rounds, in `FOV.run` round order (reference first). It adds a `round` column (pandas string), and `spot_id` runs `"0"`…`"N-1"` over the combined table, so the reference round's identities are unchanged whenever it is listed. The namespace is unchanged. | None with the default. With an explicit `rounds`, the table gains the `round` column; the reference-round rows without that column equal the golden table exactly. | Same files and `FORMAT_VERSION` 2: the table carries the extra string column, and `candidates.json` adds optional keys. A reader at `42f652d` loads every row, with `round` as an ordinary spot column, so nothing is lost silently. | A single table to link across rounds or to map to genes; `(namespace, spot_id)` is unique; the round is a column; coincident candidates of different rounds are separate rows. |
| B. One result per round, with the round in the namespace | `FOV.spot_results[round]`: the reference round keeps today's namespace and identities; another round's namespace appends the round label (`[dataset, sample, fov, subtile, round]`), with its own `"0"`…`"n-1"`. | None with the default; the reference result stays bit-identical in multi-round runs. | Several tables per stage. `FORMAT_VERSION` becomes 3, so an older reader cannot silently load only the reference round. `test_registration_recipe.py:520-525` and `test_registration_validation.py:433`, which pin version 2 and the rejection of 3, need named edits. | Per-round results that §2.8 must concatenate. Each round's identities are independent of the other rounds' detections. The asymmetric namespaces (reference without a round) have to be parsed or special-cased. |
| C. Round-qualified identities | One table and namespace, with `spot_id` `"round2:17"` for non-reference rounds and the reference unchanged. | As A. | As A. | Identity carries meaning that must be parsed; the reference and other rounds use different forms. Not recommended. |

**Recommendation: A.** It satisfies every D6 constraint without a format-version
change or an edit to an existing test, it gives §2.8 the round as data rather than
as a parsed string, and it keeps the reference round's identities when that round
is listed. Its cost is that a non-reference round's identities depend on how many
candidates the earlier rounds produced. Option B avoids that dependence at the
cost of a format version and a second result container.

### Rules for option A

* Coincident candidates of different rounds, or of different channels, are never
  merged; each keeps its own row, round and channel.
* **Amended by §2.8 (a mode check).** A `round` column requires
  `Dataset.readout_mode="direct"`, and `direct` mode requires a `round` column (a
  plan with explicit `rounds`; listing only the reference round is allowed). In
  `multiplexed` mode, with a plan that names `rounds`, `FOV.run` raises
  `ValueError("decoding candidates from several detection rounds needs a readout
  mode (§2.8): set readout_mode='direct' on the dataset for direct readout")` when
  decoding is enabled, and `FOV.decode_barcodes` raises the same error for a spot
  table with a `round` column, so a multi-round set is never decoded as barcodes.
  In `direct` mode a spot table without a `round` column raises `ValueError`.
* **Amended by §2.8 (extraction).** In `multiplexed` mode extraction accepts a
  multi-round candidate set unchanged: it reads neighbourhood sums at every
  candidate's coordinates in every sequencing round, and the `round` column rides
  along. In `direct` mode each candidate is extracted in its own round only; the
  values of its other rounds are `0.0` with `valid=False`.
* **Amended by §2.8 (unmapped rounds).** In `direct` mode a candidate detected in a
  round, or a channel, that the panel does not name is `unmatched` with
  `unmapped_channel`, not an error.
* In a multi-round result the per-channel diagnostics (`thresholds`, `counts`,
  `outcomes`, `noise`) move under `diagnostics["rounds"][<round>]`. A single-round
  result keeps them at the top level, as today.

## Candidates checkpoint

The `candidates` stage keeps its files, table layout, dtype map and
`FORMAT_VERSION` 2 (recommended option A):

* the table may carry `round` (string), `radius` and `probability` (float64) as
  spot columns; `candidates_frame` already copies every spot column, and `string`
  and `float64` are supported checkpoint dtypes;
* `candidates.json` adds the optional keys `detection_rounds` (`null` or the list
  of rounds), `detection_plan` (the channel overrides as `{channel, config}`
  entries), `execution` (the execution entry) and `weights` (the `artifacts`
  entries);
* `_detectors()` is replaced by the name-to-type map from `SPOT_FINDING_METHODS`
  (`config_type_for`), and the override configs are rebuilt the same way;
* reload: `read_checkpoint(..., "candidates")` returns a `SpotFindingResult` whose
  table equals the written one exactly (CSV and Parquet), whose config and plan
  equal the original, and whose diagnostics keep the JSON-representable entries.
  A version-2 checkpoint written at `42f652d` loads as before, with
  `detection_rounds` taken as `null`.

`run.json` keeps `format_version` 1. Its `config.pipeline.spot_finding` holds the
config or plan, `config.execution` holds `device`, and the detection step record
gains the `methods` list of provenance entries (registry move 5).

## Diagnostics

`SpotFindingResult.diagnostics` keeps today's keys for a single-round result
(`method`, `channel_labels`, `thresholds`, `coordinate_units`,
`singleton_z_policy`, `measurements`) and adds:

| Key | Content |
| --- | --- |
| `counts` | Candidates per channel (per round under `rounds` in a multi-round result). |
| `outcomes` | Per channel: `ok` (at least one candidate), `empty` (the method ran and found nothing) or `constant` (a constant channel, which yields no candidates without running the method). A failure is never an outcome: it raises (dependency, weights, geometry, invalid input or a library exception wrapped with its message). |
| `effective_settings` | Per channel, the effective config with resolved native defaults. |
| `noise` | Local maxima, every threshold mode: per channel `zero_fraction`, `median`, `mad`, `threshold` (the §2.5 noise-mode record). A `SpotFindingWarning` (a `UserWarning`) naming round and channel is emitted, and its message added to `warnings`, when MAD is 0 or more than half of the voxels are zero; the threshold does not change. |
| `warnings` | The warning messages of this call. |
| `software` | Versions of starfinder, NumPy, SciPy and scikit-image, and of the method's library and torch. |
| `model` | Learned methods: method, model, the `artifacts` entries and the training pixel size with its provenance. |
| `execution` | The execution entry ("Execution device"). |
| `native` | Per channel, the minimum, median and maximum of each native score or size column (`probability`, `radius`). |
| `geometry` | LoG: the scale-space memory estimate; Spotiflow: the effective `n_tiles`; Piscis: the tile size and keep-boundaries per axis. |
| `rounds` | Multi-round results: per-round `thresholds`, `counts`, `outcomes` and `noise`. |

`plot_detections(image, result, *, channel, z, yx_window=None, round=None,
ax=None)` draws the detections of one channel on one slice or crop with
matplotlib (a core dependency) and returns the axes. It is never called by
`FOV.run`; overlays are written only when a caller asks for them.

## Dependency plan

The batch-preparation commit, applied only after W-268 approves it explicitly,
changes `src/python/pyproject.toml` and `src/python/uv.lock` and nothing else:

```toml
[project.optional-dependencies]
# Learned spot detectors (§2.7). torch has no cp314 wheel at 2.7.1, so the extras
# are empty on Python 3.14 and later.
spotiflow = [
    "spotiflow==0.6.5; python_version < '3.14'",
    "torch==2.7.1; python_version < '3.14'",
    "torchvision==0.22.1; python_version < '3.14'",
]
piscis = [
    "piscis==1.1.0; python_version < '3.14'",
    "torch==2.7.1; python_version < '3.14'",
    "torchvision==0.22.1; python_version < '3.14'",
]

[tool.uv]
constraint-dependencies = ["torch==2.7.1", "torchvision==0.22.1"]

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

* torch and torchvision are listed in each extra because uv applies
  `[tool.uv.sources]` to direct dependencies only; `explicit = true` keeps every
  other package on PyPI.
* The detector libraries are pinned exactly, because the loading routes rely on
  their 0.6.5 and 1.1.0 internals (Spotiflow's folder layout; Piscis's
  `MODELS_DIR` path resolution).
* Starfish LoG needs no extra, and starfish stays out of the dependencies (its
  `docutils<0.20` pin conflicts with the lock).
* W-266 measured the closure. Spotiflow alone adds 32 packages, 859 MB on disk
  (197 MB without torch and torchvision); Piscis alone adds 16 packages, 904 MB;
  both together add 43 packages, 1.07 GB. No locked package changed in the spike
  environment, which holds exactly this set (torch 2.7.1+cpu, torchvision
  0.22.1+cpu, spotiflow 0.6.5, piscis 1.1.0).
* **Python marker.** The lock covers Python ≥ 3.10 up to 3.14, and torch 2.7.1 has
  wheels for cp39 to cp313 only. The extras therefore carry `python_version <
  "3.14"` (W-266 choice 1, accepted). On 3.14 the learned methods raise
  `SpotFindingBackendUnavailableError` with the note above.
* **GPU users** on GP099 replace torch with the same version's `+cu118` build in
  their own environment (W-265). The constraint keeps the version aligned. The
  lock stays CPU.

The full commit text, with the verification commands, is the draft
`drafts/batch-prep-dependency-commit.md` in the W-267 run directory.

## `docs/migration.md` entries

The implementation adds these entries:

1. **Spot-finding registry.** `SPOT_FINDING_METHODS` and `SpotFindingSpec`;
   exact-type lookup, so subclasses of the detection configs are no longer
   accepted; `SpotFindingBackendUnavailableError`.
2. **New methods and extras.** `starfish_log` (`StarfishLogConfig`, no extra,
   starfish `BlobDetector` parity), `spotiflow` and `piscis` (`SpotiflowConfig`,
   `PiscisConfig`; extras `spotiflow` and `piscis`); `KNOWN_WEIGHTS`, `starfinder
   weights fetch`, `MissingWeightsError`, `WeightsHashMismatchError`; `FOV.run`
   never downloads weights.
3. **Workflow keys.** The `spot_finding` block's `method` key and config fields,
   `channel_overrides`, `rounds` and the rule-level `device`. The legacy keys are
   aliases for `local_maxima` only, and `min_distance` is the native field of
   `spotiflow` and `piscis` (the method-aware table of
   {doc}`spot-finding-contract`). **Intentional change:** for `local_maxima`,
   `min_distance` together with `min_distance_voxels`, or another legacy key
   together with its field, now raises instead of silently preferring one;
   `local` leaves the schema enum.
4. **Detection plan.** `SpotFindingPlan` and `ChannelOverride`; `find_spots` and
   `PipelineConfig.spot_finding` accept a plan; the `round` column of multi-round
   results; decoding a multi-round set raises in readout mode `multiplexed` and is
   direct readout in readout mode `direct` (§2.8, W-293).
5. **Execution.** `ExecutionConfig.device` (only `"cpu"`) and the execution
   record.
6. **Diagnostics.** The new keys. **Intentional change:** `SpotFindingWarning` when
   a channel's MAD is 0 or more than half of its voxels are zero, where nothing was
   reported before; the thresholds and detections are unchanged.
7. **W-218 option.** `LocalMaximaConfig.merge_radius_zyx`, off by default; the
   default behavior is unchanged unless W-268 decides otherwise.
8. **Evaluation.** `localization_errors` and `classify_detections` in
   `starfinder.evaluation.spot_finding`.
9. **Checkpoints.** The optional `candidates.json` keys and spot columns;
   `FORMAT_VERSION` stays 2.

## Tests the implementation changes

* `test/test_spot_finding_golden.py`: every pinned digest stays. The diagnostics
  change makes one named edit: `test_noise_mode_mad_zero_passes_silently` becomes
  a test that the MAD-0 channel keeps threshold 0 and now emits one
  `SpotFindingWarning` per affected channel. If the registry move changes how a
  local-maxima config is built, only the body of `detection_config` may change.
* `test/test_registration_recipe.py:520-525` and
  `test/test_registration_validation.py:433` stay unchanged with option A; option
  B would need the named edits listed in its row.
* Every other existing test passes byte for byte unchanged. Each draft
  implementation issue names its set, following {doc}`method-registry`
  ("Migration plan", move 4).

## Exclusions

No training or fine-tuning, no fifth detector family, no shared DoG or PSF filters,
no generic spot fitting or spot masks, and no cross-channel deduplication (§2.8).
No readout-mode configuration, gene mapping or direct-mode wiring in `FOV.run`
(§2.8). No GPU execution, and no default method or model chosen from comparative
data; method comparisons belong to E02.
