# Method registry: adding a method

Use a stage's registry to connect a new method's configuration, implementation
and capabilities to its public entry point. Start with a nearby implementation
in that stage, then update the workflow and documentation alongside it.
Paths below are relative to `src/python/starfinder/` unless shown otherwise.

## Quick start

Inspect the registration methods and their implementation locations:

```python
from starfinder.registration import REGISTRATION_METHODS

for config_type, spec in REGISTRATION_METHODS.items():
    print(spec.name, config_type.__name__, spec.run.__module__)
```

## What to change

1. **Config.** Add a frozen dataclass in the stage package. Describe its options
   and validation in the docstring, following a neighbouring config.
2. **Implementation.** Write the stage-specific run function. Use the stage
   spec's callable signature and import optional backends inside the execution path.
3. **Registry.** Add the config and spec to the stage's registry. Declare its
   capabilities and dependencies using that stage's spec type.
4. **Annotations and hooks.** Update a config union where the stage has one.
   Spot finding's `_CONSTANT` hook handles constant-channel shortcuts;
   segmentation's `export/_segment.py:LENGTH_FIELDS` describes sample-grid
   scaling; stitching imports method modules through its `__init__.py`.
   Compare the corresponding existing method before adding an entry.
5. **Exports.** Add the config to the stage's `__init__.py` and `__all__`,
   then to its sorted autosummary list under `docs/api/`.
6. **Workflow.** Update the stage's enum or step definition in
   `workflow/schemas/config.schema.yaml` and its workflow-key tests.
7. **Checks and pages.** Add focused method tests, run the reference audit and
   update the stage's guide. Generated method tables pick up the registry entry.

For exact lookup, geometry, validation and naming rules, consult the
[registry specification notes](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/method-registry.md#the-shared-mechanism)
and the stage spec's API docstring.

## Find the stage

| Registry | Module | Spec | Public entry |
| --- | --- | --- | --- |
| `PREPROCESSING_METHODS` | `preprocessing` | `PreprocessingSpec` | `run_step` |
| `REGISTRATION_METHODS` | `registration` | `RegistrationSpec` | `estimate_transform` |
| `SPOT_FINDING_METHODS` | `spot_finding` | `SpotFindingSpec` | `find_spots` |
| `DECODING_METHODS` | `readout` | `DecodingSpec` | `decode_reads`, `decode_direct` |
| `ENCODINGS` | `readout` | `EncodingSpec` | `load_codebook`, `Codebook` |
| `SEGMENTATION_METHODS` | `segmentation` | `SegmentationSpec` | `segment` |
| `STITCHING_METHODS` | `stitching` | `StitchingSpec` | `stitch` |

`ENCODINGS` describes barcode encodings. The other registries connect processing
methods. The shared helper is `starfinder._registry`; the stage modules own their
spec types and execution signatures.

## What you get

The config becomes available through the stage package and its API reference.
Workflow parsing and checkpoint reconstruction use registry lookups. A stage
that writes method provenance records the method name, config and implementation
path; see the {doc}`checkpoint reference <checkpoints-reference>`.

## Limits

This is a guide to adding built-in methods. There is no supported third-party
plugin interface. Check a stage's callable and capability definitions rather
than assuming that a declaration suitable for one stage fits another.

## How it is checked

Registry tests exercise lookup and declarations; workflow-key tests compare the
schema with the stage configs. `docs/check_reference.py` checks public exports
against the API lists. Follow [contributing](contributing.md) for test selections.

The [registry notes](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/method-registry.md)
keep the detailed mechanism, with its
[history](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/method-registry-history.md).
