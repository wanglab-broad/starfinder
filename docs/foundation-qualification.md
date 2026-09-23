# Bounded foundation qualification

W-171 qualifies software persistence and selected backend arithmetic, using
`starfinder.artifacts/1` and `starfinder.sample_export/1`. It does not establish
molecular truth, production storage performance or complete viewer support.
Fiji repair/execution/acceptance remains deferred to open W-170 under the
2026-09-23 scope decision. A successful Python round trip cannot close that gap.

## Frozen comparison protocol

`foundation-backend-v1` uses no RNG or historical TIFFs. Python inputs are
uint16 ZYXC `(5,9,11,4)` and `(1,9,11,4)`, with ordered rounds
`(round10,round2)` and channels `(ch02,ch00,ch03,ch01)`. Calibration is unknown;
coordinates are voxel indices in `foundation/reference`. The independent
zero-based centers are A=`(Z//2,2,3)` and B=`(Z//2,6,7)`. A has round/channel
values 7 at channel 1 then 9 at channel 0; B has 11 at channel 0 then 13 at
channel 1. All other pixels are zero. Additional supplied candidates at
`(0,0,0)` and `(Z//2,4,5)` test no signal and a two-channel tie; the tie is
in a separate extraction input so it cannot change the detector population.

This clean condition compares two isolated, interior, integer peaks without
asserting equivalence of the two general detection algorithms. Python uses
`LocalMaximaConfig('adaptive',0.1,min_distance_voxels=1,exclude_border=True)`;
MATLAB uses `SpotFindingMax3D(...,'adaptive',0.1)`. MATLAB's output is one-based
XYZ with one-based channels; convert explicitly to zero-based ZYX/channel.
Match each result independently to the two literal centers using Euclidean
voxel distance at most `1e-9`, requiring a bijection, no duplicates or extras,
and the exact channel. Never join detector rows by position or fabricated IDs.
`regionprops3` detection is qualified only for Z=5; Z=1 detection equivalence
is unexecuted because the MATLAB API is volumetric. Z=1 registration/extraction
and code calling are still executed.

This translation condition tests a known integer correction on a nonnegative
image with zero margins. Shift the summed reference by `(0,+1,-1)` without
cropping signal; both backends must estimate correction `(0,-1,+1)` exactly.
Python uses its default FFT translation/apply path; MATLAB uses
`DFTRegister3D`/`DFTApply3D`, or `DFTRegister2D`/`DFTApply2D` at Z=1.
MATLAB YXZ is Python ZYX transposed `(1,2,0)`; map correction YXZ back to ZYX.
Compare the entire output, including zeroed borders, to the literal reference
with `rtol=0, atol=1e-12`. The tolerance bounds double FFT roundoff on amplitudes
at most 11, not registration error. Signed/fractional Fourier and general
preprocessing/local registration are outside this comparison: MATLAB magnitude
and Python signed-real output have different contracts.

This extraction condition uses supplied centers, separately from detection,
and clipped radius `(0,0,0)` or `(1,1,1)` in Python ZYX, reordered YXZ for
MATLAB. MATLAB receives XYZ+1. Python preserves float64 NCR; MATLAB's API
returns only per-round winner/score, not the raw trace. Compare Python's full
trace to literal sums exactly, and MATLAB's exposed calls/scores to independent
expectations. Map color symbols `1,2,3,4` to Python channels `1,0,3,2`; reorder
MATLAB input channels to `[2,1,4,3]` so its one-based winners represent the same
symbols. Unique calls A=`12`, B=`21` are exact. Per-round WTA score is
`log1p(1e-6/amplitude)`; summed score tolerance is `atol=1e-12,rtol=0`.
Integer sums are exactly representable before MATLAB's single intermediate.
Zero/tied neighborhoods return MATLAB `M`/infinite score; Python preserves
`no_signal`/`ambiguous` rather than treating those statuses as equivalent.

This codebook/filter condition retains the uniquely called A/B population,
with codebook `gene-A=12` and `gene-B=21`. `EncodeBases('AAC')=12` and
`EncodeBases('ACC')=21` are independent nucleotide expectations with no reversal.
MATLAB `FilterReads` and Python default filtering must retain both and their
exact genes after coordinate matching; terminal-base checks are diagnostic.
MATLAB's `DecodeCS` does not support `M`, so the zero/tie population is measured
at extraction and explicitly excluded from MATLAB filtering. Python retains
those rejected rows. No all-population or rescued-decoder parity is claimed.

## Storage and reload protocol

Rerun retained I1/I2/C1/C2/C3/D1/S1/X1 tests and the image/candidate/molecular
examples in the pinned environment; no serialization tolerance is permitted.
Test inherited raster/export behavior independently, including literal E1–E9,
source hashes, affine geometry, cell/population/null relationships and regional
chunk reads. Reused results need matching code, input, config and environment
identities; report controller checks separately from worker checks.

The bounded cost example measures fresh writes and reads, bytes and exact
round trips for HDF5 image dtype/chunk/compression variants, and Parquet signal
schema/codec/row-group/partition variants. The image ramp is
`100*z+10*y+x`, ZYX `(9,48,48)`, one channel, cast exactly to uint16/float32/
float64. Signal rows use the two-candidate literal checkpoint, repeated to
256 distinct candidate IDs for the partition diagnostic. These are cost
probes, not alternate checkpoint schemas or a production sweep. Compare
actual candidate saving enabled/disabled, retain the explicit omission reason,
and record decoding/filtering equality. Timer results are single observations,
not throughput estimates; filesystem caching and container overhead dominate
small fixtures. Float32 signal experiments require exact representability and
do not authorize changing the canonical float64 signal contract.

The default remains candidate/signal saving for persistent runs because it
retains pre-rejection traceability. An explicit
`RunRecorder(...,save_candidates_signals=False)` remains available when that
cost is unwanted. The decision must be accompanied by measured local costs in
the external packet; it is not a universal retention or storage-superiority claim.

## Execution and host compatibility

The shared Snakemake `run_matlab_scripts` helper resolves
`STARFINDER_MATLAB_EXECUTABLE`, then `matlab` on `PATH`, then the existing Broad
setup when present. A missing explicitly selected executable fails rather than
selecting another installation. Calls use `-singleCompThread -batch`; MATLAB
errors propagate to Snakemake. Shared MATLAB function signatures, config keys
and coordinate/file contracts are unchanged. Qualification requires MATLAB
R2023b with Image Processing and Statistics toolboxes and an actual license
checkout; the Python/export environment remains the separately pinned W-168
family (Python 3.12, SpatialData 0.2.5, Zarr 2.18.7, numcodecs 0.15.1).

Run the maintained preparation/comparison example from `src/python` with a
fresh external directory; it emits a Snakefile which includes the same common
helper, plus immutable input/config/protocol hashes. Execute the printed
Snakemake command before comparing, with the fixed CPU/thread/no-GPU limits.
Missing MATLAB output is an error, never a skipped success.

```bash
uv run python ../../docs/examples/foundation_qualification.py prepare /external/new-run/backend
STARFINDER_MATLAB_EXECUTABLE=/usr/local/MATLAB/R2023b/bin/matlab \
  uv run snakemake --snakefile /external/new-run/backend/Snakefile --cores 1 \
  --directory /external/new-run/backend
uv run python ../../docs/examples/foundation_qualification.py compare /external/new-run/backend
uv run python ../../docs/examples/foundation_qualification.py storage /external/new-run/storage
```

Saved numerical output, manifests and reports remain external. Jiahao owns
retention through thesis/publication; backup and public reproducibility remain
unverified. The controller owns full checks, commit, final rendering and
independent acceptance; future lifecycle events cannot be certified in advance.
