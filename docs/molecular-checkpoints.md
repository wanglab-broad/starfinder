# Decoded, final and sample molecule checkpoints

The `starfinder.artifacts/1` D1/S1 stages preserve existing decoder and filter
semantics. They add persistence and access, not a decoder, cell assignment or
sample assembly policy. The optional `checkpoint` extra supplies PyArrow; no
backend is installed during processing. See [the contract](artifact-contracts.md)
and [candidate checkpoints](candidate-checkpoints.md).

## Separate populations

`save_decoded_checkpoint(directory, spots, decoded, codebook, ...)` saves one
pre-QC call per candidate, complete coordinates/geometry, codebook snapshot,
method-specific scores/configuration and optional decoding diagnostics. Candidate
order is authoritative; signal-shaped diagnostics are aligned by identity.
`load_decoded_checkpoint` reconstructs existing `BarcodeDecodingResult`,
`SpotFindingResult` and `Codebook` objects. It needs neither images nor candidate
signal files. Filtering can be rerun directly on `.decoded`.

`save_final_checkpoint(directory, filtered, decoded_source=..., config=..., code=...)`
saves a distinct accepted table and complete QC decision table, counts, fractions,
undefined-fraction reasons and filter configuration. It checks those decisions
against the existing `filter_reads` implementation; it never redecodes signals.
`load_final_checkpoint` verifies the pinned pre-QC source and returns the complete
`ReadFilteringResult` plus `.pre_qc`. All-rejected and empty results are complete,
typed artifacts. Empty acceptance fractions remain `None`, not zero.

A persistent `FOV.run(..., provenance=RunRecorder(...))` saves pre-QC immediately
after decoding and final output after filtering. `fov.decoded_checkpoint_path`
and `fov.final_checkpoint_path` expose the manifests. Both batch and streaming
modes use these hooks. A save failure propagates and records a failed stage;
completed earlier checkpoints remain available. Runs without a recorder stay
in memory. Disabling candidate persistence **does not disable decoded/final
saving** or its PyArrow requirement when those stages run.

```python
from starfinder.io import (
    checkpoint_reference, save_decoded_checkpoint, load_decoded_checkpoint,
    save_final_checkpoint, load_final_checkpoint,
)
from starfinder.barcode import filter_reads

pre_path = save_decoded_checkpoint(
    output / "decoded", spots, decoded, codebook,
    candidate_source=checkpoint_reference(candidate_path),
    dataset_id="dataset", sample_id="sample", FOV="FOV-1", run_id="run-1",
    config=effective_config, code=code_identity,
)
pre = load_decoded_checkpoint(pre_path)
filtered = filter_reads(pre.decoded)
final_path = save_final_checkpoint(
    output / "final", filtered, decoded_source=checkpoint_reference(pre_path),
    config=effective_config, code=code_identity,
)
final = load_final_checkpoint(final_path)
rows = final.molecule_table()  # accepted; accepted_only=False includes rejected
```

## Exact source lookup and geometry

An `ArtifactReference` pins an explicit manifest path, SHA-256, run ID and artifact
ID. `checkpoint_reference` creates it from completed bytes; readers verify the
pin, components and identity. Paths are source locators, not guessed FOV names.
Moving a bundle requires preserving its source locators or explicitly publishing
new references; readers do not search for a latest or similarly named file.

A final molecule row contains calls/scores, acceptance/rejection reasons, float64
zero-based `z,y,x`, dataset/sample/FOV/frame, source run/candidate-artifact IDs,
namespace/spot ID, and decoded run/artifact IDs. Original signals are not copied.
The locator is `(source_run_id, candidate_artifact_id, spot_namespace, spot_id)`;
genes, coordinates and row positions are never source keys.

```python
trace = final.pre_qc.source_trace(spot_namespace=namespace, spot_id="A")
# available, reason, candidate, values(C,R), valid(R), metadata,
# round_labels, channel_labels, codebook and source/config context
```

If candidate saving was disabled, the decoded writer instead requires
`candidate_source=None, trace_unavailable_reason="candidates_signals_disabled"`.
Known rows return `available=False` with that reason. Unknown identities, changed
checksums or missing promised files fail explicitly. Reloading calls/QC still
works when signal files are unavailable; trace resolution checks them on demand.

Coordinates stay in their actual FOV frame. Physical spacing/origin/direction/unit
remain unknown when not supplied. `comparison_metadata()` exposes ZYX/base-zero
voxel coordinates, NCR/CR/NR signal/validity axes, zero-based channel/round indices,
ordered acquisition labels, codebook, encoding and color-to-channel mapping.
Backends can compare these representations without sharing a file format; this
is not executed MATLAB parity evidence. Observed and decoded color calls, optional
nucleotide interpretation, assigned genes, QC and independent truth are distinct.
No nucleotide truth or gene assignment is inferred by the persistence layer.

Optional `links` mappings on decoded/final artifacts preserve existing H5AD,
assignment or registered-image references unchanged; a link is not validation of
that external file. Pipeline checkpoints retain actual registration results and
attempts (including dense arrays) in decoded metadata. They apply no transform
on reload. Cell labels/assignment meaning, overlap ownership, reconciled sample
populations and common-frame transforms remain §§2.8–2.10/W-152 decisions.

## Sample and section access

`save_molecule_index(path, sources, dataset_id=..., sample_id=..., section_id=None)`
writes an ordered list of pinned final artifacts. An optional section label is
only explicit grouping, not an inferred section transform. Duplicate FOV/subtile
entries or repeated candidate sources are rejected as ambiguous. Colliding local
spot IDs across FOVs remain distinct through source references and namespaces.

```python
from starfinder.io import save_molecule_index, load_molecule_index

index_path = save_molecule_index(
    output / "sample.json", tuple(checkpoint_reference(p) for p in final_paths),
    dataset_id="dataset", sample_id="sample", section_id="section-1",
)
index = load_molecule_index(index_path)
whole = index.read_table(accepted_only=False)
for batch in index.iter_batches(batch_size=1024, accepted_only=False):
    rows, geometry_and_codebook, links = batch.table, batch.comparison, batch.links
trace = index.source_trace(run_id=source_run_id,
    candidate_artifact_id=candidate_artifact_id,
    spot_namespace=namespace, spot_id=spot_id)
```

Batches follow index order then candidate order and never span FOVs. Empty FOVs
yield one typed empty batch. Concatenation equals whole-table reads exactly for
compatible column schemas. Different decoder columns require FOV-scoped batches;
a whole-table read rejects incompatible schemas rather than filling absent scores.
Acquisition mappings and geometry stay per-FOV, never one implied common tensor.
Memory is bounded by the largest whole FOV plus one yielded row batch; this
implementation is not row-group streaming and makes no large-sample performance
claim. An index needs at least one artifact; a complete empty FOV is supported.

## Layout and integrity

Each fresh decoded/final directory has a manifest published last, lossless Zstd
Parquet scalar tables (row groups ≤65,536) and non-pickled NPY diagnostic arrays.
Declared pandas dtypes, null masks, signed zero, special floating scores and
metadata tuple types are retained. Table row indices explicitly restore order;
physical Parquet row order is not identity. Safe typed metadata is whitelisted.
Component paths, size, checksum, physical dtype, artifact binding and stage are
verified before success. Existing directories are never overwritten. A trusted
manifest hash additionally binds metadata; checksums alone are not authentication.
Final loads verify QC populations against pre-QC calls; source lookup verifies the
exact candidate checkpoint. Failed writes never publish a complete result.

## Bounded acceptance example

```bash
uv run python ../../docs/examples/molecular_checkpoints.py /external/new-run/molecular-example
uv run pytest test/test_molecular_checkpoints.py -v
```

The literal W-154 oracle has A signals `[[0,9],[7,0],[0,0],[0,0]]`, color call `12`,
gene-A and default acceptance; B has no signal and is retained as rejected.
3D/Z=1, empty, all-rejected and omitted-trace FOVs are saved independently.
Sample sources are deliberately reversed. Equality is exact, with no tolerance
for persistence changes. No historical image files, random seed or calibrated
scientific population is used.

```{literalinclude} examples/molecular_checkpoints.py
:language: python
```
