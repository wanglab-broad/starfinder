# Method registry across stages

Status: Accepted (W-246, 2026-09-30, at 47057aa); amended at the W-276 review, 2026-10-01

This page decides how processing methods are registered, named in workflow YAML
and recorded in provenance, before §2.6 adds registration methods and §2.7 adds
selectable spot-finding methods. It shares one small registry mechanism between
stages and leaves each stage's own contract in place. It proposes no code change
by itself; the implementation belongs to the §2.6 specification
({doc}`registration-contract`) and later issues. The preprocessing registry it
generalizes is specified in {doc}`preprocessing-contract`.

## Terms

These terms were decided by Jiahao on 2026-09-29 (W-152) and are used
throughout:

* a **stage** is a pipeline stage: preprocessing, registration, spot finding;
* a **method** is a registrable algorithm of a stage;
* a **recipe** is a stage's ordered configuration;
* a **step** is one position in a recipe and names a method. Recipe positions
  are never called stages.

Spot finding has a plan, not a recipe. A recipe is an ordered chain of method
steps; spot finding has no such chain, since it runs one method per run, with
per-channel overrides and a set of rounds, so its stage configuration is a
`SpotFindingPlan` ({doc}`spot-finding-contract`, "Detection plan"). Its pipeline
field is `PipelineConfig.spot_finding`, named after the stage like the module, the
YAML block and the registry; it was named `detection` until the W-276 review
(2026-10-01).

## Scope

The registry covers the stages whose method is chosen by the exact type of a
frozen config:

| Stage | Registry | Spec type | Recipe and step | Config types today |
| --- | --- | --- | --- | --- |
| Preprocessing | `starfinder.preprocessing.PREPROCESSING_METHODS` (renamed from `STEPS`) | `PreprocessingSpec` (renamed from `StepSpec`) | `PreprocessingRecipe` with `PreprocessingStep` (renamed from `RecipeStep`) | seven preprocessing configs |
| Registration | `starfinder.registration.REGISTRATION_METHODS` (new) | `RegistrationSpec` (new) | `RegistrationRecipe` (new) with `RegistrationStep` (exists, not renamed) | `TranslationConfig`, `DemonsConfig`, `TpsConfig`, `CpdConfig` |
| Spot finding | `starfinder.spot_finding.SPOT_FINDING_METHODS` (new) | `SpotFindingSpec` (new) | none: a `SpotFindingPlan`, not a recipe (one method per run, per-channel overrides, rounds) | `LocalMaximaConfig`, `NoiseLandmarkConfig`, `PercentileCentroidConfig` |
| Segmentation | `starfinder.segmentation.SEGMENTATION_METHODS` (§2.9) | `SegmentationSpec` | none: a `SegmentationPlan` of named runs, not a recipe (one method or one mask import per run; {doc}`segmentation-contract`) | `SeededWatershedConfig`; the learned `StarDistConfig` and `CellposeConfig` follow with their backends |

The registries of this table are public module-level constants. The module stays
`starfinder.spot_finding`.

Decoding, segmentation and stitching (§§2.8–2.10) adopt the same mechanism
when they add methods; their current method lists are recorded below so that
they are not missed. The registry does not cover:

* application policies that are not methods, such as `WarpConfig.backend`
  (`translation`, `scipy`, `simpleitk`), `ImageLoadConfig` and `ProjectionConfig`;
* the fixed internal uses of one method, such as the landmark detection that
  TPS and CPD run with `NoiseLandmarkConfig`;
* stage order, the pipeline itself, or any method's numerical behavior.

There is no generic pipeline or DAG engine and no entry-point plugin discovery.

## Decisions recorded

These decisions were made by Jiahao in the §2.6 planning session on
2026-09-28 and after the W-240 checkpoint on 2026-09-29 (W-152 comment
"§2.6 method-registry naming and W-240 open choices"); they are settled and
this page builds on them.

1. **Separate stage contracts, one shared mechanism.** Preprocessing,
   registration and spot finding keep their own interfaces, results and
   enforcement wrappers. Only the registry mechanism is shared, through a
   private helper module `starfinder._registry` that each stage uses with its
   own spec type.
2. **Registry names.** `PREPROCESSING_METHODS` (renamed from `STEPS`),
   `REGISTRATION_METHODS` and `SPOT_FINDING_METHODS`, public module-level
   constants. Spec types `PreprocessingSpec` (renamed from `StepSpec`),
   `RegistrationSpec` and `SpotFindingSpec`.
3. **Recipes.** `PreprocessingRecipe` with `PreprocessingStep` (renamed from
   `RecipeStep`); `RegistrationRecipe` with `RegistrationStep`, which is not
   renamed. The ordered list of every recipe is `steps`. `StepContext`,
   `StepResult`, `run_step`, `step_spec` and `step_config_type` keep their
   names.
4. **`RegistrationRecipe`.** The §2.6 global and local steps form a
   `RegistrationRecipe`: the reference round, the registration signal, the
   ordered steps, composed transforms and one final resampling. It replaces
   `PipelineConfig.registration: tuple[RegistrationStep, ...]`. Its fields are
   in {doc}`registration-contract`.
5. **One registration module.** Registration keeps one public module,
   `starfinder.registration`. Transform persistence stays in `starfinder.io`
   and registration metrics stay in `starfinder.evaluation`. The spot-finding
   module stays `starfinder.spot_finding`.
6. **Formats unchanged by the renames.** YAML keys and saved formats do not
   change: they keep `steps` and `step`, for example the `preprocessing.steps`
   YAML list, `preprocessing_record["recipe"]["steps"]`, the `step` and
   `stage` fields of the per-round records in `run.json` and the `run.json`
   `steps` list. Those saved names are kept even where the new terms would
   choose another word.
7. **The W-240 open choices.** The eight choices recorded with this page's
   first version were accepted as recommended: public registries (1); no
   third-party extension point (2); provenance entries on the existing
   `run.json` `steps` records (3); the schema's registration method enum
   widened to the accepted names (4); mechanical test edits for a rename, with
   no alias (5); one `demons` method with a `variant` field (6); exact-type
   lookup for spot finding (7); supplied-statistics hooks on the preprocessing
   spec (8).

## The shared mechanism

### Registry shape

Each stage owns one module-level mapping from an exact frozen config type to
its spec, as preprocessing does today:

```python
PREPROCESSING_METHODS: dict[type, PreprocessingSpec]   # starfinder.preprocessing (renamed from STEPS)
REGISTRATION_METHODS: dict[type, RegistrationSpec]     # starfinder.registration
SPOT_FINDING_METHODS: dict[type, SpotFindingSpec]      # starfinder.spot_finding
```

The mapping stays a plain `dict`. Existing tests insert fixture methods into
the preprocessing registry with `monkeypatch.setitem` and expect every lookup
to see them, so no lookup may be cached or copied into a second list.

The private helper holds only functions that read such a mapping:

```python
# starfinder/_registry.py (private)
@dataclass(frozen=True)
class Dependency:
    module: str               # imported lazily, e.g. "SimpleITK"
    distribution: str         # recorded with its version, e.g. "SimpleITK"
    extra: str | None = None  # install hint, e.g. "local-registration"

def check_name(name: str, what: str) -> None: ...
def spec_for(registry, config, what: str, error: type[Exception] = TypeError): ...
def config_type_for(registry, name: str, what: str, error: type[Exception] = ValueError) -> type: ...
def names(registry) -> tuple[str, ...]: ...
def require(spec, what: str, error: type[Exception]) -> None: ...
def provenance(spec, config, stage: str) -> dict: ...
```

`what` is the noun used in the stage's messages ("preprocessing step",
"registration method", "spot-finding method"); preprocessing keeps "step"
because its current messages use it. The helper builds the messages the stages
raise today, for example
`unknown preprocessing step 'x'` and `no preprocessing step is registered for
SubTophat (lookup uses the exact config type)`. A stage whose current error type
or message differs (for example `InvalidRegistrationConfigError("expected a
typed registration config")`) passes its own exception type and keeps its
message, so no caller-visible error changes.

### Fields of the shared spec

Every stage spec is its own frozen dataclass. It has these shared fields, which
the helper reads, plus its stage-specific fields:

| Shared element | Where it lives | Rule |
| --- | --- | --- |
| Stable name | `spec.name` | Lowercase snake_case, matching `[a-z][a-z0-9]*(_[a-z0-9]+)*`; unique within the stage's registry; used as the YAML `method` value and in provenance. Once released, a name never changes; a renamed method keeps the old name as a legacy alias in the workflow adapter. |
| Exact config type | the registry key | Lookup uses `type(config)` exactly; subclasses are not matched. The config is a frozen dataclass whose `__post_init__` validates it. Where a stage's configs carry a `method` discriminator (registration, spot finding, decoding), its value equals `spec.name`. |
| Implementation | `spec.run` | A callable with the stage's own signature. The stage's enforcement wrapper calls it; callers never call it directly. |
| Declared capabilities | stage spec fields | Declarations the stage wrapper, the workflow adapter or the docs read; see the next section. |
| Optional dependencies | `spec.requires: tuple[Dependency, ...] = ()` | Imported lazily inside the implementation. `require()` imports each module when the method runs and raises the stage's error, naming the module and the extra, if one is missing. Importing the stage package never imports an optional dependency. Constructing or validating a config never needs the dependency. |
| Provenance entry | `provenance(spec, config, stage)` | The uniform entry described in [Provenance in run.json](#provenance-in-runjson). |

`PreprocessingSpec` keeps the positional constructor of `StepSpec`,
`PreprocessingSpec(name, run, category, scope, dtype_policy="preserve")`; its new
shared fields are keyword-only with defaults, so every existing construction
stays valid after the rename.

### Shared and stage-specific capabilities

| Capability | Shared or stage | Meaning | Values today |
| --- | --- | --- | --- |
| `requires` | shared | Optional dependencies (above) | `SimpleITK` for `demons`; none elsewhere; §2.6 adds `itk-elastix` for `rigid`, `affine` and `bspline` |
| `min_shape_zyx` | shared | Smallest accepted size of each spatial axis; the stage wrapper rejects smaller inputs with its geometry error before running. For registration methods that accept Z=1 as 2D, a Z=1 input is checked against the last two entries ({doc}`registration-contract`) | `(4, 4, 4)` demons; `(2, 2, 2)` TPS and CPD; `(1, 1, 1)` everywhere else; §2.6 adds `(4, 16, 16)` for `rigid`, `affine` and `bspline` |
| `category`, `scope`, `dtype_policy` | preprocessing | As in the {doc}`preprocessing-contract` | unchanged |
| `post_registration` | preprocessing | Whether the step may run in a recipe's `post_registration` list | `True` only for reconstruction |
| `supplied` | preprocessing | `None` when the step cannot use a supplied-statistics file; otherwise a `SuppliedSpec` (below) with the step's hooks for that file | set for histogram matching, scalar background and percentile normalization |
| `transform_kind` | registration | Kind of transform the estimator returns | `translation` or `dense`; §2.6 adds `affine` and `bspline` |
| `step_kind` | registration | `global` or `local`; decides the allowed step sequences of a `RegistrationRecipe` | `global` for translation; `local` for demons, TPS and CPD |
| `dimensions` | registration | Whether Z=1 is estimated as 2D (`2`) and Z>1 in 3D (`3`) | `{2, 3}` translation; `{3}` demons, TPS and CPD today; §2.6 adds `2` for demons |
| `space` | registration | `index` or `physical` estimation coordinates | `index` for the four current methods; `physical` for the §2.6 methods |
| `pipeline` | spot finding | Whether `FOV.find_spots` and `PipelineConfig.spot_finding` accept it | `True` only for `local_maxima` |
| `output_columns` | spot finding | Columns of the spot table besides `spot_id` | `z, y, x, channel[, peak_intensity]` or `z, y, x` |
| `targets`, `roles`, `required_roles`, `seeds`, `dimensions`, `models`, `devices` | segmentation | Label targets, accepted and required channel roles, whether seeds are refused, optional or required, Z=1 as a plane (`2`) and Z>1 as a volume (`3`), whether the config names a model, and the devices ({doc}`segmentation-contract`, "Method registry") | `seeded_watershed`: `cell`; one stain of `cytoplasm`, `membrane`, `amplicon`, `composite`; seeds required; `{2, 3}`; no model; `cpu` |

A capability is shared only when it means the same thing in every stage and a
generic check can use it. Anything else stays in the stage spec. A stage may add
fields without changing the helper. The first version of this page proposed a
registration field `application_backend`; the §2.6 contract drops it because a
`RegistrationRecipe` resamples once with its own `WarpConfig`.

`SuppliedSpec` is a preprocessing-only frozen dataclass. It holds the
per-method code that `preprocessing/supplied.py` now selects by config type:

```python
@dataclass(frozen=True)
class SuppliedSpec:
    per_round: bool                               # fitted values per round; False: reference round only (histogram matching)
    fit: Callable[[Any, HistogramSummary, str | None], dict]  # params and fitted values from merged histograms
    validate: Callable[[Mapping, np.dtype, int], None]        # checks one section of a read file
    check_params: Callable[[Any, Mapping, str], None]         # section params against the recipe's config
```

The hooks live next to each step's implementation (`normalization.py`,
`background.py`, which already hold `_supplied_range` and
`_supplied_background`), so `steps.py` never imports `supplied.py`. Each hook
keeps the error messages of the branch it replaces.

## Stage contracts left untouched

The registry changes how a method is found, not what it does or returns.

* **Preprocessing methods.** `run(volume, config, context) -> StepResult`,
  `run_step()` and its checks, `StepContext`, `PreprocessingStep` (renamed from
  `RecipeStep`) and `PreprocessingRecipe` stay as specified in the
  {doc}`preprocessing-contract`.
* **Registration methods.** `estimate_transform(reference, moving, *, config,
  reference_metadata, moving_metadata) -> RegistrationResult`,
  `apply_transform(moving, transform, *, config)`, the transform types, the
  error types and `RegistrationDiagnostics(method, backend, ...)` stay. The
  §2.6 contract ({doc}`registration-contract`) adds transform kinds and
  replaces the per-step `warp` of `RegistrationStep` with the recipe's single
  final resampling.
* **Spot finding.** `find_spots(image, *, config, metadata, spot_namespace) ->
  SpotFindingResult`, the stable `spot_id` identities and the result's
  validation stay.

## YAML naming per stage

In every stage, the YAML key `method` holds a registered `spec.name`, and the
other keys of a method's mapping are exactly the init fields of its config
dataclass (YAML lists become tuples). The name lookup is `config_type_for()`
over the stage's registry. Legacy keys stay accepted and keep their meaning;
their translation lives in the workflow adapter (`dataset/workflow.py`), not in
the registry.

| Stage | Current YAML | Registry-based YAML | Legacy compatibility |
| --- | --- | --- | --- |
| Preprocessing | `preprocessing.steps[].method: <method name>` (Python only) | unchanged | `enhance_contrast`, `hist_equalize`, `morph_recon`, `tophat` and `snr_threshold` keep mapping to recipe 1; the explicit key stays mutually exclusive with them |
| Registration | `global_registration` and `local_registration` blocks; `method` in `translation`, `demons`, `diffeomorphic`, `symmetric`, `fast_symmetric`, `tps`, `cpd`; MATLAB-style setting names such as `grid_spacing` and `beta` | a Python-only `registration` key for the `RegistrationRecipe`, whose `steps` list names methods by `REGISTRATION_METHODS` names ({doc}`registration-contract`, "Workflow configuration") | the two legacy blocks map to a recipe of two steps (global, then local). The demons variant names stay legacy aliases for `method: demons` with `variant: <name>`; the MATLAB-style setting names and the legacy defaults (CPD `detection_noise_sigma=3`, `grid_spacing_voxels=32`) stay in the adapter |
| Spot finding | `spot_finding` block with `intensity_estimation`, `intensity_threshold`, `min_distance`/`min_distance_voxels`; always `local_maxima` | an optional `method` key naming a `SPOT_FINDING_METHODS` name, default `local_maxima`, with that config's fields | the legacy keys stay valid for `local_maxima` only and are rejected with any other method |
| Segmentation | `rules.stardist_segmentation.parameters`; always StarDist | a Python-only `segmentation` block whose runs name a `SEGMENTATION_METHODS` name (or `import`) in `method`, with that config's fields ({doc}`segmentation-contract`, "Workflow configuration") | the legacy block translates into one `stardist` run in the workflow adapter ({doc}`segmentation-contract`, "Translation of the legacy keys") |

The static schema `workflow/schemas/config.schema.yaml` cannot import the
registry. Each stage's schema list is kept equal to its registry by a
default-tier test, as `test_preprocessing_workflow_key.py` does for the
preprocessing registry.
MATLAB rules, shared MATLAB keys and filenames do not change.

## Third-party methods

Third-party methods are not a supported extension point in Chapter II. The
registries are public, mutable dictionaries only so that tests can register
fixture methods, as the preprocessing tests already do. A method inserted at run
time passes the same spec validation and exact-type lookup, and its provenance
entry records its implementation's qualified name, so an out-of-tree method is
identifiable in `run.json`. The workflow adapter resolves only names present
when it runs, so Snakemake rules see built-in methods only. There is no
entry-point discovery. Supporting third-party methods later means adding a
documented `register()` helper and a stability promise for the spec fields;
that is a separate decision.

## Provenance in run.json

Each method invocation that the stage wrapper runs produces one uniform entry:

```json
{
  "stage": "registration",
  "method": "demons",
  "config_type": "starfinder.registration.DemonsConfig",
  "implementation": "starfinder.registration._demons.estimate_demons",
  "config": {"variant": "demons", "iterations": [100, 50, 25], "smoothing_sigma": 1, "pyramid_mode": "antialias", "method": "demons"},
  "requires": {"SimpleITK": "2.5.3"},
  "artifacts": []
}
```

* `stage` is `preprocessing`, `registration` or `spot_finding` (later
  `decoding`, `segmentation`, `stitching`); `(stage, method)` identifies a
  method, so names only need to be unique within a stage.
* `config` is the config as `run.json` already serializes dataclasses.
* `requires` maps each declared optional distribution to its installed version.
* `artifacts` lists files a method reads besides its image, each with `path`
  and `sha256`; it is empty today and reserved for §2.7 pretrained weights.

The entries are added to the existing `steps` records of `run.json`, as a list
`methods` on the record of the step that ran them: one entry for a
preprocessing step or a detection, and one per attempt, in order, for a
registration with recovery. The top-level fields of `run.json` and
`format_version` stay unchanged, and so do the stage records it already holds:
`preprocessing.rounds[...]` with `step` and `config`, `registration[...]` with
`requested_method`, `actual_method`, `config` and `outcome`, and
`config.pipeline`. The {doc}`checkpoints` page documents the field when it is
implemented.

## Where method sets are hard-coded today

Paths are relative to `src/python/starfinder/` unless they start with
`workflow/` or `benchmarks/`; line numbers are at the starting revision `1b4db5a`.

### Preprocessing

| Place | What is hard-coded | How it will be derived |
| --- | --- | --- |
| `preprocessing/steps.py:124-132` (`STEPS`) | the method set | This is the registry, renamed `PREPROCESSING_METHODS` in move 1; it stays the source of truth. |
| `preprocessing/steps.py:139-166` (`step_spec`, `step_config_type`) | lookups over `STEPS` | Already derived; they call `spec_for()` and `config_type_for()` with unchanged messages. |
| `preprocessing/steps.py:55-74` (`StepSpec`) | the name pattern | `check_name()` in `PreprocessingSpec`; the pattern is unchanged. |
| `preprocessing/supplied.py:90-114` (`supplied_section`) | exact-type branches for `PercentileNormalizationConfig`, `ScalarBackgroundConfig` and `HistogramMatchingConfig` that fit each section, the rule that only histogram matching takes `reference_round`, and the fallback error | `step_spec(config).supplied`: its `fit` hook builds the section; `reference_round` is accepted only when `per_round` is `False`; `supplied is None` raises the current `has no supplied statistics` error. |
| `preprocessing/supplied.py:163-164` (`_SECTION_VALIDATORS`), used at line 195 | the steps whose file sections can be validated | The `validate` hook of `PREPROCESSING_METHODS[step_config_type(name)].supplied`; a name whose spec has `supplied=None` keeps the current error. |
| `preprocessing/supplied.py:280-294` (`read_supplied_statistics`) | exact-type branches that compare section params with the recipe's config (`p_low`/`p_high`, `percentile`, `reference_channel`) and skip the per-round check for histogram matching | The `check_params` hook of the step's `supplied`, and its `per_round` flag for the round check. |
| `preprocessing/background.py:136`, `:167` (`scalar_background_histograms`) | a single-type check and the literal name `scalar_background` | Stays single-method (a public helper of one step); the literal becomes `step_spec(config).name`. |
| `preprocessing/steps.py:279-282` (`PreprocessingRecipe.__post_init__`) | only `ReconstructionConfig` may run after registration | The steps whose spec has `post_registration=True`; the message is built from their config names, so with reconstruction alone it stays `post_registration may contain only ReconstructionConfig steps`. |
| `benchmarks/preprocessing_synthetic.py:505` | passes `reference_round` to `supplied_section()` only for `HistogramMatchingConfig` | `per_round` of `step_spec(entry.config).supplied`. |
| `preprocessing/steps.py:135-136` (`_supplied`), `dataset/fov.py:358`, `dataset/fov.py:634` | none; they read the config's `fit` field | Stay: a config field, not a method list. |
| `dataset/workflow.py:120-143` (`_explicit_recipe`) | none; it uses `step_config_type` | Already derived. |
| `dataset/workflow.py:96-113` (`_legacy_recipe`) | the four legacy keys and their fixed configs | Stays explicit: it is the frozen legacy translation of recipe 1, not a method list. |
| `workflow/schemas/config.schema.yaml:381-505` | one definition per step and the `oneOf` list | Stays static; `test_preprocessing_workflow_key.py` keeps it equal to `PREPROCESSING_METHODS`. |
| `dataset/fov.py:299-330` (`normalize_intensity`, `match_histogram`, `reconstruct_background`, `filter_tophat`) | one fixed config per convenience method | Stays: public per-method wrappers, not a list. |

### Registration

| Place | What is hard-coded | How it will be derived |
| --- | --- | --- |
| `registration/_api.py:39-40` | accepted config types in `estimate_transform` | `spec_for(REGISTRATION_METHODS, config, ...)` raising `InvalidRegistrationConfigError` with the current message. |
| `registration/_api.py:50-53` | minimum axis size by type (demons 4, other local methods 2) | `spec.min_shape_zyx`, checked with `IncompatibleGeometryError`. |
| `registration/_api.py:54-126` | the `isinstance` dispatch chain and each branch's backend and `WarpConfig` | `spec.run` (one private estimator per method, in its own module); the `diagnostics.backend` value is returned by the estimator as today. Until move 3, the estimator also returns its `WarpConfig` as today. |
| `registration/_types.py:112` | the config union in `RegistrationDiagnostics.effective_config` | An annotation only; one alias defined next to `REGISTRATION_METHODS`, with a test that its members are the `REGISTRATION_METHODS` keys. |
| `dataset/config.py:14` (`_REGISTRATION_CONFIGS`), used at lines 32 and 48 | accepted types for `RecoveryConfig.alternatives` and `RegistrationStep.config` | `type(config) in REGISTRATION_METHODS` through `spec_for()`, keeping `TypeError` and the messages `invalid recovery configuration` and `unsupported registration config`. |
| `dataset/config.py:24`, `dataset/config.py:40` | config unions in annotations | The same alias as above. |
| `dataset/config.py:119`, `dataset/config.py:137-139` | `registration: tuple[RegistrationStep, ...]` | Replaced by the `RegistrationRecipe` (decision 4); the recipe validates each step's config through `REGISTRATION_METHODS`. |
| `dataset/workflow.py:38-72` (`_registration`) | the `if`/`elif` chain over `translation`, `tps`, `cpd`, `demons` and the three demons variants | `config_type_for(REGISTRATION_METHODS, name, ...)` for registered names, plus an adapter-local legacy alias table for the demons variants and the legacy CPD defaults; unknown names keep `ValueError('unknown registration method ...')`. |
| `dataset/workflow.py:40` | legacy default methods (`demons` local, `translation` global) | Stay in the adapter as legacy defaults. |
| `dataset/workflow.py:91` | warp backend by method name | Until move 3, derived from the estimator's returned `WarpConfig` backend through a per-`transform_kind` table; move 3 replaces it with the recipe's single `warp`. |
| `dataset/workflow.py:204` (`registration_keys`) | the four config types whose fields are accepted | The init fields of every `REGISTRATION_METHODS` key, plus the adapter's legacy names. |
| `io/_checkpoint.py:343-346` (`_registration_results`) | `dict(translation=..., demons=..., tps=..., cpd=...)` | The name-to-type map from `REGISTRATION_METHODS` (`config_type_for()`); saved `method` values equal spec names by the discriminator rule. |
| `benchmark/_adapters.py:5-19` (`_CONFIGS`, `_config`) | name-to-type map for benchmark cases | `config_type_for(REGISTRATION_METHODS, ...)`, keeping `ValueError('unsupported registration method: ...')`. |
| `registration/_config.py:31`, `:48`, `:74`, `:99` | the `method` discriminator defaults | Stay (checkpoints and `run.json` store them); a test asserts each equals its spec name. |
| `workflow/schemas/config.schema.yaml:576-579` | `local_registration.method` enum `demons`, `tps`, `cpd` | Widened to the accepted names (the registered names plus the legacy demons aliases) and kept equal to them by a new default-tier test (W-240 choice 4, accepted); the change is in `workflow/`, so it belongs to move 3. |

### Spot finding

| Place | What is hard-coded | How it will be derived |
| --- | --- | --- |
| `spot_finding/__init__.py:57`, `:80`, `:97` | the `method` discriminator defaults | Stay; a test asserts each equals its spec name. |
| `spot_finding/__init__.py:126` (`SpotFindingResult.__post_init__`) | accepted config types | `type(config) in SPOT_FINDING_METHODS`. |
| `spot_finding/__init__.py:163-177` (`find_spots`) | the config union and the accepted types | `spec_for(SPOT_FINDING_METHODS, config, ...)` keeping `TypeError("unsupported detection config")`; the annotation uses one alias. |
| `spot_finding/__init__.py:187-228` | the `isinstance` dispatch | `spec.run` per method; the shared table and diagnostics assembly stay in `find_spots`. |
| `io/_checkpoint.py:378-381` (`_detectors`) | name-to-type map for saved detection configs | The name-to-type map from `SPOT_FINDING_METHODS`. |
| `dataset/config.py:120`, `dataset/config.py:126` | `PipelineConfig.spot_finding` accepts only `LocalMaximaConfig` | Types whose spec has `pipeline=True`. |
| `dataset/fov.py:477-486` (`FOV.find_spots`) | only `LocalMaximaConfig` | The same `pipeline` capability, keeping the current message for other types. |
| `dataset/workflow.py:232` | the YAML block always builds `LocalMaximaConfig` | The optional `method` key through `config_type_for(SPOT_FINDING_METHODS, ...)`, default `local_maxima`. |
| `benchmark/_legacy_evaluation.py:19`, `:35` | fixed `PercentileCentroidConfig` and the literal `percentile_centroid` | Stay: one fixed legacy evaluation, not a list. |

### Decoding

`barcode/decoding.py:169` and `:207` (accepted types and dispatch),
`dataset/config.py:121` and `:127` (`PipelineConfig.decoding`) and
`io/_checkpoint.py:439` (`dict(wta=..., codebook_aware=...)`) hard-coded the two
decoders. The first version of this page placed this registry in §2.9 and named
it `DECODERS`; §2.8 adds it as `DECODING_METHODS`, following the `*_METHODS`
pattern of decision 2 ({doc}`readout-contract`, "Encoding registry"; W-292).

| Place | What was hard-coded | How it is derived |
| --- | --- | --- |
| `barcode/decoding.py` (`decode_barcodes`) | accepted types and the `isinstance` dispatch | `spec_for(DECODING_METHODS, config, ...)`, keeping `TypeError("unsupported decoder config")`; `spec.run` per decoder. A codebook whose encoding kind (`EncodingSpec.symbols`) is not in `spec.encodings` raises `TypeError`. |
| `dataset/config.py` (`PipelineConfig.decoding`) | the two config types | The `DECODING_METHODS` keys. |
| `io/_checkpoint.py` (`_read_pre_qc`) | `dict(wta=..., codebook_aware=...)` | The name-to-type map of `DECODING_METHODS`. |
| `dataset/workflow.py` | always `WtaDecoderConfig(diagnostics=True)` | The Python-only `decoding` key names a `DECODING_METHODS` method (default `wta`); the schema list is kept equal to the registry by a test. |

`DecodingSpec` declares `modes` (readout modes), `encodings` (encoding symbol
kinds), `rescue` and `score_columns`. W-293 registers `direct`
(`DirectAssignmentConfig`, mode `direct`, no encoding), which `assign_direct` runs;
`FOV.run`, `FOV.decode_barcodes`, `decode_barcodes`, `assign_direct` and the
workflow adapter check the dataset's readout mode against `modes`. The barcode encodings are registered the same
way in `ENCODINGS` (`EncodingSpec`), whose stage-specific fields are listed in
{doc}`readout-contract`; since W-304 they include `table(config)`, the encoding's
table (`two_base`: the configurable `EncodingConfig.pair_to_color`; `one_base`:
`base_to_color`). Neither registry has `min_shape_zyx`: decoders and
encodings take no image.

## The preprocessing renames

Move 1 renames three public preprocessing names and changes nothing else:

| Old name | New name | Kind |
| --- | --- | --- |
| `STEPS` | `PREPROCESSING_METHODS` | registry, `starfinder.preprocessing` |
| `StepSpec` | `PreprocessingSpec` | spec type, `starfinder.preprocessing` |
| `RecipeStep` | `PreprocessingStep` | recipe entry, `starfinder.preprocessing` |

`PreprocessingRecipe`, `StepContext`, `StepResult`, `run_step`, `step_spec`,
`step_config_type`, the recipe field `steps` and every YAML key and saved field
keep their names (decisions 3 and 6). Error messages keep their wording except
the one that names the class: `requires RecipeStep entries`
(`preprocessing/steps.py:276`) becomes `requires PreprocessingStep entries`; no
test matches that message at `1b4db5a`.

These are all references at the starting revision `1b4db5a` (line numbers in
parentheses):

| File | References |
| --- | --- |
| `src/python/starfinder/preprocessing/steps.py` | `StepSpec` class (55) and uses (124-131, 139, 140); `STEPS` (124, 147, 155, 162); `RecipeStep` class (230, 231) and uses (265, 266, 275, 276) |
| `src/python/starfinder/preprocessing/__init__.py` | import (10, 11); comment and `STEPS` alias (16, 17); `__all__` (19) |
| `src/python/starfinder/dataset/workflow.py` | import (10); constructions (113, 139); docstring (121) |
| `benchmarks/preprocessing_synthetic.py` | import (48); constructions (125-142) |
| `docs/api/preprocessing.rst` | prose (8, 14); autosummary entries (87, 97, 98) |
| `docs/api/inventory.rst` | inventory entries (159, 169, 170) |
| `docs/api/python-index.rst` | index entries (102, 127, 128) |
| `docs/examples/percentile_two_pass.py` | import (17); constructions (56, 57) |
| `docs/migration.md` | preprocessing recipe table and example (125-146, 157) |
| `docs/preprocessing-contract.md` | interface listings and prose (45, 52, 57, 93, 99, 100, 253, 259) |
| `workflow/schemas/config.schema.yaml` | comments only (381, 383) |

**Tests that need only an import edit.** In each of these files the only edit
is the import of the renamed names and the same identifier replacement where
the file uses them (listed); no assertion, fixture value, message pattern or
expected output changes:

| Test file | Import lines | Uses of the renamed names |
| --- | --- | --- |
| `test_background_subtraction.py` | 16, 17 | 70, 203, 266, 350 |
| `test_checkpoints.py` | 575 | 586 |
| `test_coordination_contract.py` | 18 | 50 |
| `test_e2e.py` | 4 | 229, 300 |
| `test_percentile_normalization.py` | 18, 19 | 78, 296-481 (15 lines) |
| `test_preprocessing_golden.py` | 115 | 117-119 |
| `test_preprocessing_recipe.py` | 19, 20 | 27-29, 61-198 (13 lines), 273, 285 |
| `test_preprocessing_workflow_key.py` | 11, 12 | docstring and comments (1, 32, 50); 21, 22, 60-78, 100-102, 167 |
| `test_projection_views.py` | 21 | 117 |
| `test_recipe_sources.py` | 18 | 28, 29, 79-124 (9 lines), 185, 186, 239, 240 |
| `test_summaries.py` | 12 | 48 |

Every other test file stays byte-for-byte unchanged, including
`test_preprocessing.py`, `test_preprocessing_synthetic_evaluation.py`,
`test_registration_golden.py` and all registration tests. There is no
compatibility alias; the project avoids shims for replaced Python APIs. The
`docs/api` lists are alphabetical and checked by `docs/check_reference.py`, so
the renamed entries move to their sorted positions. `docs/migration.md` gains
one entry for the three renames.

`RegistrationStep` is not renamed, so no registration file changes for a
rename.

## Migration plan

Each move is one reviewable change. Moves 0, 1, 2, 4 and 5 change no behavior;
move 3 is the §2.6 recipe, whose behavior changes are specified in
{doc}`registration-contract`. Every listed test file exists at `1b4db5a` and
passes after the move byte-for-byte unchanged, unless an edit is named. The
strict docs build, `docs/check_reference.py` and the default and extended
pytest tiers pass after each move.

### Move 0: the helper

Add `starfinder/_registry.py` and a new `test/test_registry.py` for the helper
alone (name pattern, exact-type lookup, duplicate names, lazy dependency error,
provenance entry). No existing file changes.

### Move 1: preprocessing renames and `PREPROCESSING_METHODS`

Apply the preprocessing renames listed above. `step_spec`, `step_config_type`
and the `PreprocessingSpec` name check call the helper; `PreprocessingSpec`
gains keyword-only `requires`, `min_shape_zyx`, `post_registration` and
`supplied`; the recipe's post-registration check reads `post_registration`. The
type branches of `supplied_section()` and `read_supplied_statistics()` and the
`_SECTION_VALIDATORS` table are replaced by the `SuppliedSpec` hooks of the
three methods that have one; the public functions keep their signatures and
messages.

The named edit is the import edit of the eleven test files listed in "The
preprocessing renames". With that edit only, these tests must pass:
`test_preprocessing_golden.py` (the exact SHA-256 digests),
`test_preprocessing_recipe.py` (the registry table, derived name lookup,
monkeypatched methods, exact-type lookup and messages),
`test_preprocessing_workflow_key.py` (schema consistency with the registry),
`test_background_subtraction.py` and `test_percentile_normalization.py`
(supplied-statistics fitting, reading, validation and their messages),
`test_recipe_sources.py`, `test_projection_views.py`, `test_checkpoints.py`,
`test_coordination_contract.py`, `test_e2e.py` and `test_summaries.py`. These
must pass byte-for-byte unchanged: `test_preprocessing.py`,
`test_preprocessing_synthetic_evaluation.py` (the benchmark in
`benchmarks/preprocessing_synthetic.py`) and `test_registration_golden.py`.

### Move 2: registration `REGISTRATION_METHODS`

Add `RegistrationSpec` and `REGISTRATION_METHODS` with the four current
methods, move each `isinstance` branch of `estimate_transform` into a private
estimator, and derive every registration place listed above except the
`PipelineConfig` field, which move 3 replaces.

Tests that must pass byte-for-byte unchanged:
`test_registration_contract.py` (config validation, discriminators, missing
SimpleITK error, 2D rejection, removed exports), `test_registration.py`,
`test_registration_resampling.py`, `test_demons.py`, `test_pointset.py`,
`test_benchmark.py` (unsupported methods and fallback), `test_benchmark_recipes.py`,
`test_coordination_contract.py` (workflow translation of `cpd` and `tps`,
recovery, unknown keys), `test_checkpoints.py` (dense transforms round trip and
the `run.json` fields), `test_recipe_sources.py`, `test_summaries.py`,
`test_e2e.py`, `test_fov.py`, `test_evaluation.py` and the registration golden
test `test_registration_golden.py` (W-245).

### Move 3: the RegistrationRecipe (specified by W-245)

This move changes the `PipelineConfig` field, `RegistrationStep`'s fields, the
registered checkpoint format and the `run.json` config layout, so it is not a
registry move. {doc}`registration-contract` specifies it and names its test
changes ("Tests the implementation changes"). The tests that construct
`PipelineConfig(registration=...)` or read the field are its input:
`test_checkpoints.py` (61, 146, 450, 587), `test_e2e.py` (230, 301),
`test_recipe_sources.py` (126 to 321), `test_summaries.py` (49, 155),
`test_projection_views.py` (181), `test_benchmark_recipes.py` (127) and
`test_coordination_contract.py` (65 to 71, 221 to 243).

### Move 4: spot-finding `SPOT_FINDING_METHODS`

Add `SpotFindingSpec` and `SPOT_FINDING_METHODS` with the three current
methods, split the dispatch into private method functions and derive every
spot-finding place listed above. `PipelineConfig.spot_finding` and
`FOV.find_spots` still accept only `local_maxima`, the one method with
`pipeline=True`.

Tests that must pass byte-for-byte unchanged:
`test_spot_finding_contracts.py`, `test_spotfinding.py`, `test_checkpoints.py`
(detection config round trip), `test_extraction.py`, `test_evaluation.py`,
`test_preprocessing_golden.py` (the local-maxima noise thresholds),
`test_pointset.py` (noise landmarks inside TPS and CPD), `test_fov.py`,
`test_e2e.py` and `test_coordination_contract.py`.

Exact-type lookup rejects subclass instances of the three spot-finding
configs, which `isinstance` accepts today. No caller, script or test subclasses
them at `1b4db5a`; the change is recorded in `docs/migration.md`.

### Move 5: provenance entries

The stage wrappers add the `methods` list to their `steps` records in
`run.json`. `test_checkpoints.py`, whose run-record test fixes the top-level
field set of `run.json`, `test_recipe_sources.py` and
`test_coordination_contract.py` must pass byte-for-byte unchanged; new tests
check the entries.

## What the §2.6 and §2.7 specifications adopt

The §2.6 specification (W-245) and the §2.7 spot-finding contract
specification of task group 1 adopt the following from this page.

Both specifications:

* use the terms stage, method, recipe and step as decided;
* register every new method in their stage's registry with a stable snake_case
  name, an exact frozen config type and a `method` discriminator equal to the
  name;
* declare `requires` for every optional dependency, import it lazily and raise
  the stage's dependency error naming the module and the extra;
* declare `min_shape_zyx` and their stage-specific capabilities;
* take the YAML `method` values from the registry and state the legacy keys
  they keep;
* record the uniform provenance entry;
* name, for each implementation issue they draft, the tests that pass
  unchanged, following the migration plan.

The §2.6 specification ({doc}`registration-contract`) also:

* uses `RegistrationRecipe` with `RegistrationStep` entries in `steps`, and
  defines the recipe's reference round, signal, allowed step sequences,
  transform composition and single final resampling;
* derives the five registration places named in its scope (`registration/_api.py`,
  `dataset/config.py`, `io/_checkpoint.py`, `benchmark/_adapters.py` and
  `dataset/workflow.py`) and the schema enum from `REGISTRATION_METHODS`;
* defines the `transform_kind` values of the new methods and the composition
  rules between kinds;
* keeps the four current methods' names, defaults and results unchanged.

The §2.7 specification also:

* registers its four methods in `SPOT_FINDING_METHODS` and sets
  `pipeline=True` for those that `PipelineConfig.spot_finding` and
  `FOV.find_spots` accept;
* adds the optional `method` key of the `spot_finding` block and keeps the
  legacy keys for `local_maxima`;
* identifies pretrained weights in the `artifacts` of the provenance entry
  (name, path and SHA-256) and defines where they are loaded from;
* introduces a spot-finding dependency error type, since spot finding has none
  today.

## Comparison with starfish

W-225 compared starfish with Starfinder in the thesis reference
`docs/chapter-II-reference/survey-starfish-design-comparison.md` (thesis commit
`6dc7179`), which audits starfish `main` at `1fb00cbc`. The statements below
follow that reference and the starfish source at that revision.

starfish organizes methods into component families, such as `Filter`,
`LearnTransform`, `ApplyTransform`, `FindSpots`, `DecodeSpots`, `DetectPixels`
and `Segment`. Each family has a base class with its own `run` signature, for
example `FilterAlgorithm.run(stack, *args) -> Optional[ImageStack]`. An
algorithm is a class of its family that holds its parameters as constructor
arguments. Each family package imports its implementations and builds its
`__all__` from the subclasses of the family base. The family bases use the
`AlgorithmBase` metaclass (`starfish/core/pipeline/algorithmbase.py`), which
wraps `run` to update a log. W-225 recommends mapping each Starfinder config
type to its implementation in one place, and not copying that metaclass logging.

| starfish pattern | Position here | Reason |
| --- | --- | --- |
| Separate component families with their own `run` signatures | Adopted | It matches decision 1: preprocessing, registration and spot finding keep their own interfaces. A single generic signature would hide the reference/moving pair of registration and the spot table of detection. |
| One list of implementations per family, from which documentation and tools derive their choices | Adopted as the per-stage registry, in a different form | starfish derives the list from the subclasses a package happens to import. Here the list is one explicit mapping from exact config type to spec, as W-225 recommends, and the YAML lookup, schema tests and checkpoint readers derive from it. |
| Provenance logging in the `AlgorithmBase` metaclass | Rejected | The wrapper writes a log entry only when `run` returns a value and the class name contains `ApplyTransform`, `Filter`, `FindSpots`, `DecodeSpots` or `DetectPixels`. `LearnTransform` and `Segment` runs, and in-place filters that return `None`, are not logged, and no input identity is recorded. W-225 lists this as a pattern not to copy. Here each stage wrapper records the uniform entry explicitly for every invocation of every registered method, in the per-FOV `run.json` of W-156 and W-199, which also holds input hashes, the Git commit and package versions. Returned arrays stay plain NumPy. |
| Algorithms as classes that hold parameters, gathered by subclassing | Rejected | Starfinder selects methods by frozen config types and runs plain functions. A method set that depends on which subclasses were imported conflicts with exact-type lookup and with a single explicit list. |
| A recipe and command-line layer over the component families | Rejected | W-225 records that starfish removed its recipe and component command-line layer in 0.1.5, and that its remaining WDL workflow pins a 0.1.0 container. Starfinder keeps its maintained Snakemake workflow, a fixed stage order in `FOV.run` and stage-specific recipes (`PreprocessingRecipe`, `RegistrationRecipe`). The issue also excludes a generic pipeline engine. |
