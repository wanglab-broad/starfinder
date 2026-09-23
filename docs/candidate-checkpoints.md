# Candidates and signals checkpoints

`starfinder.io.save_candidate_checkpoint` saves the complete **pre-decoding/QC
rejection** population. Detection and extraction remain separate functions;
loading never repeats either operation. The return value reports the manifest
path and actual total bytes (components plus manifest). Use
`load_candidate_checkpoint(path, sha256=trusted_manifest_hash)` to reconstruct a
whole FOV as existing `SpotFindingResult`, `IntensityExtractionResult` and optional
`Codebook` objects. A codebook is unnecessary for extraction, but decoding needs
an explicitly supplied or saved codebook.

Install the optional `checkpoint` extra during environment setup, separately from
execution/validation. It pins PyArrow 23.0.0. Missing PyArrow raises ImportError;
there is no installation or CSV fallback inside the API. Prepared locked batch
environments must include that extra alongside their existing selections.

## Saving and population

A persistent `FOV.run(..., provenance=RunRecorder(...))` saves this checkpoint
by default immediately after complete extraction, before decoding/filtering.
The run links the standalone manifest and its component checksums. A storage
failure remains visible, records a failed run and prevents downstream decoding.
Partial extraction does not publish a complete checkpoint. Failed downstream
processing does not erase an already completed pre-rejection checkpoint.

Set `RunRecorder(..., save_candidates_signals=False)` to disable this provisional
default. The run records the requested policy, omission reason and zero saved
bytes. `fov.candidate_checkpoint_save` holds the actual path/size/reason. Runs
without a recorder remain in-memory and report `no_persistent_run_destination`;
they do not invent an output location or run identity. Separate direct extraction
calls remain pure/in-memory until you explicitly call the writer, whose
`enabled=True` default may likewise be overridden with `enabled=False`.
Saving defaults remain subject to W-171 storage qualification; tiny-fixture costs
are not a universal retention policy.

An optional candidate-only diagnostic is still available via
`export_spots(spots, None, path, columns=["spot_namespace", "spot_id", "x", "y", "z"])`.
That existing CSV adapter exports **one-based XYZ** for MATLAB-facing inspection.
It has no signals/validity and is not a reloadable combined checkpoint. The
Parquet checkpoint preserves **zero-based float64 ZYX** and full geometry,
including unknown calibration. No decoding threshold or QC population changes.

## Physical layout and integrity

The [v1 contract](artifact-contracts.md#candidate-and-signal-layout-w-159) defines
one fresh directory containing `artifact.json`, `candidates.parquet`,
`signals.parquet`, `validity.parquet` and optional `codebook.parquet`.
The writer preserves all supported scalar candidate columns: pandas strings,
nullable integers/Booleans/floats and NumPy Boolean/integer/float widths.
Object, nested and categorical columns are rejected explicitly. DataFrame indices
are not identities; candidate order is stored as `candidate_index`. Ordered
labels, detector/extractor settings, diagnostics, geometry, sources, code context
and codebook mapping/encoding accompany the tables. Tuple metadata stays tuples.

Signals use long identity/channel/round keys, reconstructed as float64 NCR;
validity uses identity/round keys, reconstructed as Boolean NR. Physical row order
is irrelevant. A false validity bit retains its finite stored signal value;
it is not replaced by zero. Empty populations retain typed tables and nonempty
channel/round axes. Stable identities join signals to candidates even when the
supplied result orders differ. Duplicate/missing/foreign identities, missing rows,
invalid indices, nonfinite signals, unsupported versions, wrong physical dtypes,
checksums or dimensions fail before returning success.

Parquet uses lossless Zstandard and row groups at most 65,536 rows. Actual engine
version and storage settings are recorded. Components bind to artifact identity;
the reader checks their size, hash and schema. Writers never overwrite an
existing directory and publish the complete manifest last. Missing files raise
FileNotFoundError; malformed or corrupt payloads raise ValueError. Pin a trusted
manifest hash when external identity must be verified, as component checksums
alone do not authenticate edited metadata.

## Source trace and downstream reruns

```python
from starfinder.io import load_candidate_checkpoint
from starfinder.barcode import decode_barcodes, filter_reads, WtaDecoderConfig

saved = load_candidate_checkpoint(checkpoint_path, sha256=manifest_sha256)
trace = saved.source_trace(
    run_id=source_run_id, candidate_artifact_id=source_artifact_id,
    spot_namespace=namespace, spot_id=spot_id,
)
redecoded = decode_barcodes(saved.intensities, saved.codebook, config=WtaDecoderConfig())
filtered = filter_reads(redecoded)
```

Lookup requires the exact pinned artifact/run plus namespace/ID, never a latest
FOV or nearest coordinate. It returns the candidate, copied C×R signals and R
validity, geometry, ordered labels, optional codebook and source/config context.
Unknown IDs or a foreign locator raise ValueError. Disabled saving reports an
unavailable reason in `CandidateSaveResult`/the run; there is no checkpoint to
resolve. Decoded/final persistence and ordered sample access are described in
[molecular checkpoints](molecular-checkpoints.md). Sample assembly remains separate.

## Bounded example

The executable example reuses W-154's independent literal arithmetic: A has
`[[0,9],[7,0],[0,0],[0,0]]`, B has no signal, and the nontrivial channel mapping
assigns A to `gene-A`. Saved 3D and Z=1, empty and invalid cases rerun existing
decoding/filtering exactly in the same environment. No serialization tolerance,
scientific accuracy threshold, historical TIFF or random seed is introduced.

```bash
uv run python ../../docs/examples/candidate_checkpoints.py /external/new-run/candidate-example
uv run pytest test/test_candidate_checkpoints.py -v
```

```{literalinclude} examples/candidate_checkpoints.py
:language: python
```

The [bounded foundation qualification](foundation-qualification.md) measures
saving/override costs and documents the retained traceability default. Its
alternative dtype/layout probes do not change the canonical float64 signal
schema; production cost and optimal chunking remain unmeasured.
