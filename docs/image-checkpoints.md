# Optional image checkpoints

`starfinder.io.save_image_checkpoint` writes an explicit per-FOV HDF5 artifact;
`load_image_checkpoint` reloads it without applying any processing. Direct TIFF
loading remains available. h5py is already a locked runtime dependency; using
HDF5 and saving images are optional operations, with no automatic format migration.
These APIs implement image cases I1/I2/X1 of
[starfinder.artifacts/1](artifact-contracts.md).

## Prepare inputs

Load each supplied round with `load_round` or each single-channel layer with
`load_volume`. Construct an `ImageLayer` with its `ImageLoadResult`, round label,
roles and `ImageProcessingState(loaded.metadata)`. Pass all available sequencing,
stain and registration-reference layers explicitly; no directory scan discovers
unrequested stains. A layer may have multiple roles. `RoundState` supplies the
ordered sequencing/other rounds and reference identity. One layer represents one
round; combine channels explicitly through the existing loader when needed.

Call `save_image_checkpoint` with `stage="prepared_input"`, explicit
sample/dataset/FOV/run identities, effective configuration and code revision or
patch identity. Source records carry catalog/URI, selection and checksum (or a
reason it is unverified). The loader's source paths and diagnostics retain
channel selection, original geometry, crop and conversion information. For
in-memory arrays, record shape/dtype and checksum scope; acquisition provenance
is not inferred from intensities. Saving does not hash or dereference external
source/provenance URIs automatically.

Each layer preserves ZYX or ZYXC shape, channel labels, dtype and exact finite
values, including negative values and signed zero. All five `ImageMetadata`
fields are required; unknown physical fields remain `None`. Stains/references
may have different shapes, frames and channel counts. They are neither resampled
nor normalized by storage. A sequencing consumer needs explicit ZYXC arrays and
compatible geometry; storing several arrays together does not establish alignment.

## Save actual registered state

Use the distinct `registered_images` stage for already-processed arrays.
For each saved round, supply `ImageProcessingState` with source metadata,
ordered completed operations/effective configurations, successfully applied
`RegistrationResult` objects, registration attempts and actual terminal state.
`apply_transform` operation configs must match the saved application results,
in order. Compact corrections and dense fields retain their dtype, direction,
units, geometry, application policy and diagnostics. The reader restores their
types using a fixed constructor allowlist; it never loads pickle or arbitrary code.

Terminal states distinguish `applied`, `reference_unchanged`, `skipped`, `failed`
and `not_registered`. Non-applied states need a reason. A failed round may retain
an earlier successful transform; failed attempts are separate from applied results.
A frame label alone cannot establish an applied registration. Operations are
caller-supplied history, not a reconstruction from output pixels: preserve them
when processing, and link the exact `RunRecorder` manifest through its URI and
SHA-256. Parent run/artifact IDs plus manifest SHA-256 bind prepared input lineage.
The writer does not mutate terminal run manifests or invent missing history.

`load_image_checkpoint` returns an `ImageCheckpoint` retaining every saved layer,
roles and history. Partial/failed round inventories remain inspectable.
`checkpoint.sequencing_images(require_registered=True)` returns ordered
`ImageLoadResult` values directly consumable by `extract_intensities`; it rejects
missing/unavailable rounds, unapplied registration, differing sequencing labels
or geometry, and single-channel ZYX layers that were not explicitly stacked.

For coordination, `dataset.fov(id).load_image_checkpoint(path,
require_registered=True)` loads a complete checkpoint into an empty FOV. Identity,
round membership/order and sequencing channels must match the dataset. It restores
images, metadata, registration results/attempts and loader diagnostics; the full
original history remains under `fov.image_checkpoint`. Prepared FOV loading permits
different frames before explicit registration. Incompatible downstream geometry
still errors in the consuming operation. Existing FOV state is never overwritten.

Continue a registered FOV with `find_spots`, `extract_intensities`,
`decode_barcodes` and `filter_reads` as desired. Loading does not run these methods
or reapply transforms. `FOV.run` still means executing its explicitly supplied
pipeline; it is not an implicit restart engine. No streaming image sink or
checkpoint scheduler is introduced: retain/collect the layers you explicitly
choose to save.

## Files, integrity and cost

A fresh directory contains `artifact.json` and `images.h5`, with image datasets
at `/layers/layerNNNN/image`. Labels never become HDF5 paths. Dense fields use
separate datasets under the same opaque layer key. Image chunks are
`min(ZYX, (8,64,64))` with channel chunk 1; dense component chunks are 3.
Compression is lossless gzip level 4 with shuffle. Actual settings, h5py version,
shape/dtype, artifact/layer bindings and the HDF5 byte size/SHA-256 are recorded.

The manifest is published last. Existing directories raise `FileExistsError`;
failed writes leave inspectable incomplete directories and are not retried or
overwritten. Missing manifests/components raise `FileNotFoundError`. Invalid
schema/version, missing metadata, stage/config/geometry mismatches, component
corruption, unsafe paths, nonlocal HDF5 links and inconsistent bindings raise
`ValueError` before any payload is returned. Supply `sha256=` when loading to
verify the manifest against its parent or handoff, and `expected_stage=` to bind
the requested stage. Without a trusted manifest checksum, the file's metadata
is validated structurally but cannot be authenticated against malicious rewriting.
External provenance/parent links are retained; use `read_run(..., sha256=...)`
separately to validate the referenced run.

The bounded example writes fresh 3D and Z=1 inputs, runs existing registration and
barcode processing with provenance, saves/reloads registered images, and asserts
literal values plus exact downstream equality. It records observed I/O times and
file sizes in `summary.json`. These tiny costs are not an E10 storage comparison,
chunk tuning study, scientific accuracy result or qualification of real data.

From `src/python` in the prepared environment:

```bash
uv run python ../../docs/examples/image_checkpoints.py /external/new-run/image-example
uv run pytest test/test_image_checkpoints.py -v
```

```{literalinclude} examples/image_checkpoints.py
:language: python
```
