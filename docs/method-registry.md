# Method registry: adding a method

This page is the guide to adding a processing method to a Starfinder stage. Each stage
selects its method by the exact type of a frozen config, through one plain registry per
stage and a small shared helper, `starfinder._registry`. The page lists the seven
registries, the steps every new method takes, the hooks a few stages add, and the checks
the public entry points already make before a method runs. It replaces the decision
record that introduced the registries; see [History](#history).

Paths below are relative to `src/python/starfinder/` unless they start with `docs/`,
`workflow/` or `test/` (`src/python/test/`).

## Terms

* a **stage** is a pipeline stage, such as preprocessing, registration or spot finding;
* a **method** is a registrable algorithm of a stage;
* a **recipe** is a stage's ordered configuration (`PreprocessingRecipe`,
  `RegistrationRecipe`), and a **step** is one position in a recipe that names a method.
  Recipe positions are never called stages.

Spot finding and segmentation have a plan, not a recipe: `SpotFindingPlan` runs one method
per run with per-channel overrides ({doc}`spot-finding-contract`), and `SegmentationPlan`
holds named runs ({doc}`segmentation-contract`). Saved formats keep the words `steps` and
`step` where they used them before these terms were chosen.

## The seven registries

Each registry is a public module-level `dict` from an exact frozen config type to the
stage's own frozen spec dataclass, re-exported by the stage package.

| Registry | Public module | Spec type | Defined in | Public entry point that runs a method |
| --- | --- | --- | --- | --- |
| `PREPROCESSING_METHODS` | `starfinder.preprocessing` | `PreprocessingSpec` | `preprocessing/steps.py` | `run_step` (and the recipe runners that call it) |
| `REGISTRATION_METHODS` | `starfinder.registration` | `RegistrationSpec` | `registration/_methods.py` | `estimate_transform` |
| `SPOT_FINDING_METHODS` | `starfinder.spot_finding` | `SpotFindingSpec` | `spot_finding/_methods.py` | `find_spots` (and `FOV.find_spots`) |
| `DECODING_METHODS` | `starfinder.readout` | `DecodingSpec` | `readout/decoding.py` | `decode_reads` (multiplexed), `decode_direct` (direct) |
| `ENCODINGS` | `starfinder.readout` | `EncodingSpec` | `readout/codebook.py` | `load_codebook` and `Codebook`, through `encoding_spec` |
| `SEGMENTATION_METHODS` | `starfinder.segmentation` | `SegmentationSpec` | `segmentation/_methods.py` | `segment` (and `FOV.segment`) |
| `STITCHING_METHODS` | `starfinder.stitching` | `StitchingSpec` | `stitching/_methods.py`, filled by `stitching/_multiview.py` | `stitch` |

`ENCODINGS` registers barcode encodings rather than algorithms that process an image; it
uses the same lookup and naming rules. Its stage-specific fields are in the
[readout specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/readout.md#encoding-registry).

## The shared mechanism

The registries stay plain dictionaries. No lookup may cache a registry or copy it into a
second list: tests insert fixture methods with `monkeypatch.setitem` and expect every
lookup to see them. The helper `starfinder._registry` holds only functions that read such a
mapping, plus the `Dependency` record:

* `Dependency(module, distribution, extra=None)`: an optional dependency, imported lazily
  and recorded with its installed version;
* `check_name(name, what)`: the name pattern `[a-z][a-z0-9]*(_[a-z0-9]+)*`;
* `check_shared(spec, what)`: the shared fields below, called from a spec's `__post_init__`;
* `spec_for(registry, config, what, error, message)`: the spec of `type(config)` exactly;
  subclasses are not matched;
* `config_type_for(registry, name, what, error, message)`: the config type whose spec has
  this name, which every YAML and checkpoint lookup uses;
* `names(registry)`: the registered names in registry order;
* `check_shape(spec, shape)`: the `dimensions` and `min_shape_zyx` geometry check;
* `require(spec, what, error)`: imports each declared dependency, naming the module and
  the install extra when one is missing;
* `provenance(spec, config, stage)`: the uniform entry of
  [Provenance in run.json](#provenance-in-runjson).

`what` is the noun of the stage's messages ("registration method", "spot-finding
method"); each stage passes its own exception type, so caller-visible errors stay the
stage's own.

### Fields of the shared spec

| Shared element | Where it lives | Rule |
| --- | --- | --- |
| Stable name | `spec.name` | Lowercase snake_case, unique within the registry; the YAML `method` value and the provenance `method`. A released name never changes; a renamed method keeps the old name as a legacy alias in the workflow adapter. |
| Exact config type | the registry key | A frozen dataclass whose `__post_init__` validates it. Lookup uses `type(config)` exactly. In every registry except preprocessing the config carries a `method` discriminator whose value equals `spec.name`. |
| Implementation | `spec.run` | A callable with the stage's own signature. The stage's entry point calls it; callers never do. |
| Optional dependencies | `spec.requires` | A tuple of `Dependency`, imported only when the method runs. Importing the package or constructing a config never needs them. |
| Smallest input | `spec.min_shape_zyx` | The smallest accepted size of each spatial axis. `check_shape`, segmentation and stitching check a Z=1 input against the last two entries; preprocessing's `run_step` checks all three entries for every input. Not present in `DecodingSpec` and `EncodingSpec`, which take no image. |

`check_shared` validates the name, a callable `run`, `requires` and `min_shape_zyx`; the
preprocessing, registration, spot-finding, segmentation and stitching specs call it.
`DecodingSpec` and `EncodingSpec` call `check_name` and check their own callables.

### Shared and stage-specific capabilities

A capability is shared only when it means the same in every stage and a generic check
uses it: `requires`, `min_shape_zyx` and, for registration, spot finding, segmentation
and stitching, `dimensions` (2 when Z=1 runs as a plane, 3 when Z>1 runs as a volume);
`PreprocessingSpec` has no `dimensions` field. Everything else
is a field of the stage spec, documented with the spec's docstring and its stage
contract: for example `category`, `scope`, `post_registration` and `supplied` for
preprocessing ([preprocessing specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/preprocessing.md)); `step_kind`, `transform_kind` and `space`
for registration ({doc}`registration-algorithms`); `pipeline`, `output_columns` and
`weights` for spot finding ([spot-finding specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/spot-finding.md#registry-entries)); `modes`, `encodings`, `rescue`
and `score_columns` for decoding ([readout specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/readout.md#encoding-registry)); `targets`, `roles`,
`required_roles`, `seeds`, `models` and `devices` for segmentation
([segmentation specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/segmentation.md#method-registry)); `devices` and `transform_kind` for stitching
([stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md)). A stage may add a field without changing the helper.

## Adding a method: the mandatory steps

Every new method takes these seven steps. The stage contract states the method's
algorithm, limits and validation; this list covers only how it is registered and reached.

1. **The config.** A `@dataclass(frozen=True)` config in the stage package whose
   `__post_init__` validates every field, with a `method` discriminator
   (`field(default=<name>, init=False)`) in every stage except preprocessing. Its init fields are
   exactly the YAML keys of the method (YAML lists become tuples).
2. **The run function.** A private function with the stage's `run` signature, in its own
   private module or next to the registry. It imports its optional dependencies inside the
   function, never at module level, and it may assume the checks of
   [The validation the entry points already do](#the-validation-the-entry-points-already-do).
3. **The spec and registry entry.** One entry `Config: Spec(name, run, ...)` in the
   registry, declaring `requires`, `min_shape_zyx` and every stage-specific capability.
   The spec's `__post_init__` rejects an invalid declaration when the module is imported.
4. **The union alias.** Where the stage has an annotation alias of its registered configs
   (`RegistrationConfig` in `registration/_methods.py`, `SpotFindingConfig` in
   `spot_finding/_methods.py`), add the config to it; a test keeps each alias equal to the
   registry keys. The other stages have no alias.
5. **`__init__` and `__all__`.** Import the config in the stage's `__init__.py` and add it
   to `__all__`. The run function stays private.
6. **The API page.** Add the config to the sorted autosummary list of the stage's page in
   `docs/api/` (for example `docs/api/registration.rst`) and name it in the page's prose.
   `docs/check_reference.py` requires each `__all__` to equal the union of the autosummary
   lists, so a missing entry fails the reference audit.
7. **The schema enum or `$defs`.** `workflow/schemas/config.schema.yaml` cannot import the
   registry. Add the name to the stage's method enum (spot finding, decoding,
   segmentation, stitching) or, for preprocessing and registration, add a
   `preprocessing_step_<name>` or `registration_step_<name>` definition with the config's
   fields and list it in the `oneOf`. A default-tier test per stage keeps the schema equal
   to the registry and its config fields (for example
   `test/test_registration_workflow_key.py`, `test/test_segmentation_workflow.py`).

Nothing else needs a method list: the workflow adapter (`dataset/workflow.py`), the
checkpoint readers (`_checkpoint.py`), the benchmark adapter (`benchmark/_adapters.py`)
and the recipe and plan types derive their method sets from the registry through
`spec_for` and `config_type_for`.

## Method-specific hooks

Some stages need one more entry beside the registry. Each is kept next to the code that
reads it, and a new method adds to it only when its stage has the hook.

* **`_CONSTANT`** (`spot_finding/_methods.py`). Maps a per-channel spot-finding config to
  the function that records a constant channel without running the method; `find_spots`
  checks `skips_constant(config)` first. A per-channel method without an entry receives its
  constant channels like any other channel.
* **`LENGTH_FIELDS`** (`export/_segment.py`). For each segmentation method, the settings the
  sample-segmentation route converts from the acquired grid to the segmentation grid, and
  their rule (`length`, `area`, `factor_yx`, `length_yx` or `unchanged`). A test
  (`test/test_sample_segmentation_segment.py`) keeps its keys equal to
  `SEGMENTATION_METHODS`, so every new segmentation method needs an entry, even an empty
  rule set.
* **Stitching's side-effect registration.** `STITCHING_METHODS` is created empty in
  `stitching/_methods.py`; each method's module adds its own entry when it is imported, as
  `stitching/_multiview.py` does for `multiview_stitcher`. `stitching/__init__.py` imports
  that module, so the entry exists once `starfinder.stitching` is imported. A new stitching
  method registers itself the same way and is imported by `stitching/__init__.py`.
* **`check_shape`** (`_registry.py`). The geometry check of a spec's `dimensions` and
  `min_shape_zyx`; `estimate_transform` and `find_spots` call it. Segmentation
  (`_check_dimensions` in `segmentation/_segment.py`) and stitching (`_check_shapes` in
  `stitching/_stitch.py`) apply the same rules with their own messages, so a new method of
  these four stages only declares the two fields. Preprocessing's `run_step` has its own
  check: `PreprocessingSpec` has no `dimensions` field, and every input, Z=1 included, must
  reach all three entries of `min_shape_zyx`, or `run_step` raises `ValueError`.
* **`check_shared`** (`_registry.py`). Called from the spec's `__post_init__`, so a new method
  gets it by constructing its spec; a new stage's spec calls it first and adds its own
  checks.

## The validation the entry points already do

A method's `run` is called only after its stage's entry point has made these checks, so it
does not repeat them:

* **`estimate_transform`**: `spec_for` (exact type, `InvalidRegistrationConfigError`), both
  images finite ZYX arrays, equal grids and their metadata, `check_shape`, then `require`
  (`RegistrationBackendUnavailableError`). Estimation errors are wrapped as
  `RegistrationEstimationError`.
* **`find_spots`**: the image, `spec_for`, the metadata and spot namespace, the device, the
  plan (no rounds; channel labels and overrides), `require_method`, then `check_shape`.
* **`run_step`**: `spec_for`, the volume and every axis against `min_shape_zyx`, `require`,
  and the reference and supplied-statistics rules of the spec's `scope`.
* **`decode_reads` and `decode_direct`**: `spec_for` and the readout mode against
  `spec.modes`; `decode_reads` also checks the codebook's encoding kind against
  `spec.encodings`.
* **`segment`**: `spec_for`, the target against `spec.targets`, the input's channel roles,
  the device, `require`, the model, the dimensions and minimum shape, the seeds, and the
  label namespace.
* **`stitch`**: `spec_for`, the review policy, fallback, table, sample and inputs, the
  grids, the tile shapes, then `require` (`StitchingBackendUnavailableError`).

## YAML naming per stage

In every stage the YAML key `method` holds a registered `spec.name`, and the other keys of
a method's mapping are exactly the init fields of its config. The adapter resolves the name
with `config_type_for` over the stage's registry, so it sees only the methods registered
when it runs. Under `backend: python` only these registry-based keys are accepted; the
legacy keys stay accepted, with their meaning, for MATLAB-backend configurations, and their
translation lives in `dataset/_legacy_config.py`, not in the registries.

## Provenance in run.json

A stage that records its method calls `provenance(spec, config, stage)`, which returns one
uniform entry: `stage`, `method` (`spec.name`), `config_type` (through the public module),
`implementation` (the qualified name of `spec.run`), `config` (as `run.json` serializes
dataclasses), `requires` (each declared distribution with its installed version, or `null`)
and `artifacts` (empty). `(stage, method)` identifies a method, so names need only be
unique within a stage. Spot finding (`dataset/fov.py`), segmentation and stitching write
the entry, and a stage may add fields to it, such as spot finding's model `artifacts` and
`execution`; {doc}`checkpoints-reference` lists the fields of the `run.json` records.

## Third-party methods

Third-party methods are not a supported extension point. The registries are public, mutable
dictionaries only so that tests can register fixture methods. A method inserted at run time
passes the same spec validation and exact-type lookup, and its provenance entry records its
implementation's qualified name, so an out-of-tree method is identifiable in `run.json`. The
workflow adapter resolves only names present when it runs, so Snakemake rules see built-in
methods only. There is no entry-point discovery and no plugin system. Supporting third-party
methods later means adding a documented `register()` helper and a stability promise for the
spec fields; that is a separate decision.

## History

This page was the decision record of the registry mechanism, accepted at
the registration acceptance review on 2026-09-30 and amended at the spot-finding validation
review on 2026-10-01. The decisions were made
by Jiahao on 2026-09-28 and 2026-09-29: separate stage contracts with one
shared mechanism, the `*_METHODS` registry names, the recipe names, formats unchanged by the
renames, and the eight choices of the method-registry design (among them public registries and no third-party
extension point).

% W-246 (https://linear.app/jiahaoh/issue/W-246), W-276 (https://linear.app/jiahaoh/issue/W-276), W-152, W-240

The full record, including its "Decisions recorded" (numbered 1 to 7), "Where method sets are
hard-coded today", the "Decoding" survey, "The preprocessing renames", the "Migration plan"
(moves 0 to 5), the section on what the registration and spot-finding specifications adopt, and the "Comparison with
starfish", is kept in Git: see
[`docs/method-registry.md` at `17d1a9c`](https://github.com/wanglab-broad/starfinder/blob/17d1a9c/docs/method-registry.md).
Other pages and test docstrings that cite a decision or a move by number refer to that
version.

% §2.6, §2.7: the record's section "What the §2.6 and §2.7 specifications adopt"
