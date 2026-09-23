# Sample export and viewer contract

**Contract ID: `starfinder.sample_export/1`.** This W-168 specification freezes
inputs, populations, raster rules and evidence before W-169/W-170 implementation.
It extends [artifact contracts](artifact-contracts.md), without changing Python
ZYX/ZYXC, MATLAB interfaces or the original checkpoints. An export is a derived
sample/section view, not evidence that segmentation or assembly is correct.
W-152/Jiahao owns unresolved §§2.8–2.10 decisions; W-57 owns scientific acceptance.

## Payloads, populations and ownership

The exporter accepts explicitly selected, checksummed saved artifacts and a
caller-supplied assembly description. A `MoleculeIndex` only groups sources:
its FOV coordinates are **not** assembled sample coordinates. Require a named
sample/section frame and an explicit source-to-target mapping per selected
source, even for an identity mapping. Never infer identity from equal shapes.
The assembly description records `sample_id`, nullable `section_id`,
`target_frame_id`, `coordinate_unit`, and per-source `artifact_id`, `sha256`,
`source_frame_id`, `transform_kind`, `matrix_zyx` (finite invertible 4×4 for
affine maps), `direction="source_to_target"`, and provenance/config hashes.
Coordinates are zero-based voxel centers. A dense backward registration field
is not a forward point map: require upstream mapped points and their transform
provenance, or reject unsupported mapping. Already registered rasters retain
their operation history without reapplying it. All supplied mappings must refer
to the actual saved coordinate frame, not its pre-registration predecessor.

A single-FOV input may contribute to a named sample export; no per-FOV SpatialData
API or export fan-out is introduced.

| Payload / relationship | Required semantics | Scientific owner / implementation |
| --- | --- | --- |
| Final molecule points | Exactly the saved `final_accepted` identities; preserve `spot_namespace`, `spot_id`, FOV/subtile, call, method-specific scores, QC decisions and original coordinates. Add mapped sample coordinates separately. No new filtering. | §2.8, W-152/Jiahao; W-164 source, W-170 adapter |
| Pre-QC/rejected molecules | Counts, reasons and checksummed pre-QC/QC source links remain accessible. They are not silently mixed into final points. An optional diagnostic point layer must be named `pre_qc`, with its population declared. | §2.8; W-164/W-170 |
| Unassigned final molecules | Retain every final molecule even when it has no cell. `assignment_status=unassigned` means an upstream assignment explicitly returned no cell; `unavailable` means assignment was not supplied. Both have null cell keys with a reason. `call_status=assigned` means a gene call, never cell assignment. | §2.9, W-118/W-123 and W-152/Jiahao; W-170 validates supplied links |
| Assignment links | One row per supplied molecule key; nullable `(cell_namespace,cell_id)`, source artifact/hash and upstream method/config. Duplicate or dangling non-null assignments fail. Missing rows are explicitly `unavailable`, not inferred background assignment. | §2.9; no new assignment algorithm |
| Cells and expression | Retain supplied H5AD matrix, gene ordering/IDs, dtype, counts/normalization semantics and metadata. Join by explicit cell key, never row order, spatial proximity or molecule recounting. `X` meaning and source selection are mandatory. | §2.9 and §2.8 count population; W-170 adapter |
| Mask/table relation | Background `0`; positive integer instance IDs. An explicit bijective map relates `(mask namespace, local label)` to cell key and exported `uint32` instance ID. SpatialData table `region` names the labels element; `instance_id` is its positive label; `obs_names` is unique. | §2.9; W-169 raster, W-170 table binding |
| Mask-only cells | Retain labels and a metadata row marked `expression_status=unavailable`; do not turn missing expression into measured zero. Keep unavailable rows in metadata, outside the measured-expression table. | §2.9; W-170 |
| Table-only cells | Retain original H5AD and a separate unmatched-cell table with reason; never fabricate a mask voxel or a SpatialData region link. If the caller declares a complete mask/expression relation, absence is an error. | §2.9; W-170 |
| Zero-count cells | Supplied measured zero rows stay in the expression table and retain mask links. Distinguish these from unavailable expression. | §2.9; W-170 |
| Reference/stain images | Explicit artifact, layer, round/channel selection, roles and ordered channel names; only selected reference/stain layers. No all-round integrated image export. | §2.4/§2.9; W-170 |
| Transform/source lookup | Preserve real operation history, source/output metadata and transform direction. Point identity resolves final → decoded → candidate checkpoint → original NCR trace and validity, using W-164 lookup. Record omitted traces with the existing reason. A missing promised source/hash mismatch fails. | §2.10/W-152 for assembly; W-164/W-170 |

Empty points/cells, all-rejected final output and explicitly absent optional
payloads are valid and separately counted. A requested payload with missing bytes
is an error. Every export manifest reports input/output counts, all missing
populations and reasons. Missing-label accounting cannot make label loss a pass.
Store expression outside the raster view; Fiji expression integration is excluded.

### Namespaces and unresolved scientific policy

A supplied global cell key may map to multiple source labels only if a separately
approved assembly artifact explicitly declares that biological equivalence. This
contract does not merge them. In the ordinary disjoint case, sort unique source
keys lexicographically by `(namespace, integer local_label)` and assign export IDs
`1..N`, retaining the reversible mapping; reject `N > 2**32-1`. Equal local labels
in distinct FOV namespaces remain distinct. A reused *global* cell key, overlapping
positive masks from different cells, ambiguous duplicate molecules, missing frame
mapping or contradictory assignments fails before serialization. Do not choose a
winning FOV, blend labels, deduplicate molecules or move a cell to a nearby hole.

Concrete blockers returned to W-152/Jiahao are: which overlap observation survives
(§2.10/E07), whether two masks are one biological cell (§2.9/E04), which molecular
population contributes to a supplied expression matrix (§2.8), and how an unknown
source frame relates to the sample (§2.10). A caller-supplied, qualified assembly
can settle those facts; a hand-built disjoint fixture can test the software.
General real-sample assembly stays blocked until those facts are supplied.

## Shared geometry and deterministic resolution fallback

W-169 owns **one** preparation function used by both viewing roles. Inputs are
finite numeric images in ZYXC, integer nonnegative labels in ZYX, explicit common
metadata, and requested integer power-of-two factors `(fz,fy,fx)`. Z=1 requires
`fz=1`; no projection. Floating, negative or Boolean masks fail. Native labels
must have at least one voxel per declared mask cell. Identity/factor-one output
is byte/value/dtype exact, apart from the separately recorded namespace remap.

V1 accepts already aligned rasters with identical extent, origin, direction,
spacing and shape; it does not resample independent source grids or implement
stitching. Mismatched grids require an upstream saved aligned artifact. For
portable NGFF/Fiji viewing, physical rasters require positive spacing, finite
origin and identity direction in the target frame. Rotated/sheared frames must
be explicitly regridded upstream or fail this export profile. Preserve the
original metadata and actual transform chain regardless. Unknown physical
calibration is permitted only with an explicit common **index** frame: omit
physical units and mark calibration unknown, never write invented micrometers.

For shape `n`, source voxel centers at index `i` have support `[i-.5,i+.5)`.
For each trial factor `f`, require exact divisibility `n[a] % f[a] == 0` on every
axis. Thus every output block contains exactly `fz*fy*fx` input voxels and the
extent remains exact. Output centers map to source indices
`i_source = f*i_output + (f-1)/2`. Physical spacing is `spacing*f`, origin is
`origin + spacing*(f-1)/2`; index-frame NGFF uses the same scale/translation.
Do not apply this translation twice when adding SpatialData transforms.

1. Compute image block means directly from the native input, with float64
   accumulation. Reduced output is float32, except float64 input remains float64.
   Never average channels together. No clipping, normalization or interpolation
   between labels. Preserve requested and achieved dtype and factors.
2. Each output mask voxel takes the most frequent input label in its block,
   **including background**. On a count tie choose the smallest numeric label.
   Record mixed-positive blocks and tied blocks. This is deterministic categorical
   block resampling; it cannot create a label outside its source support.
3. Check the complete positive-label set and all fidelity criteria below. If any
   check fails, discard both trial images and masks and replace every factor
   greater than one by half. Also halve after an indivisible grid trial. Recompute
   directly from native input; never average already reduced levels.
4. Repeat to `(1,1,1)`. Record each trial, failed checks and selected factors.
   Native failure is a hard error with no complete export published. Do not
   suppress missing labels or reserve arbitrary displaced voxels. This conservative
   policy may keep full resolution even when a more complex placement could work.
5. Coarser levels use absolute factors twice the last accepted factors on each
   non-singleton axis, again from native input and with the same checks. On failure,
   stop the pyramid and record `coarser_level_not_preservable`; never emit the bad
   level or duplicate the preceding level. Every emitted level preserves all IDs.
   A single-level store is valid, but cannot satisfy a test requiring multiresolution.

The number of base attempts is at most `1 + max(log2(requested_factor))`.
Resource limits also apply to native fallback; if it exceeds the agreed budget,
fail with required native shape/bytes, rather than increasing resources. The
mask and all associated images always share the achieved geometry at each level.

### Frozen numerical checks

These are engineering fidelity bounds, not biological accuracy thresholds.
Compute in source index coordinates so unknown calibration is not fabricated.
All source/derived arrays, label sets and mappings are saved or checksummed.

| Quantity | Acceptance rule at every emitted level |
| --- | --- |
| Native round trip | Exact shape, dtype, values, IDs, channel order, geometry and source references |
| Labels | Positive ID sets exactly equal; background remains zero; each label has at least one voxel; no new ID or merge |
| Extent and grid | All corner coordinates agree within `atol=1e-9`, `rtol=0` in source-index units; image/mask transforms identical |
| Support | Every labeled output block intersects native support of that same label; overlap fraction strictly positive |
| Spatial fidelity | For each cell, centroid displacement at most `norm(f-1)/2 + 1e-9` source voxels; symmetric Hausdorff distance between native and output voxel-center sets at most `norm(f-1) + 1e-9`. Map output centers back to native index coordinates before measurement. Failed bounds cause fallback/truncation. |
| Intensity | Independent block-mean oracle: float32 `atol=1e-6, rtol=1e-6`; float64 `atol=1e-12, rtol=1e-12`; categorical and factor-one results exact |
| Affine point mapping | Float64 homogeneous source-to-target mapping and inverse agree to `atol=1e-9, rtol=0`; no one-based offset; invalid/singular transform fails |
| Source immutability | All source component hashes identical before/after; regional reads equal the corresponding complete-array slices exactly |
| Associations | Exact molecule keys, null masks, statuses and source lookup; exact expression values/gene order/cell-region-instance joins; no expression recalculation |

## Literal development oracles

The machine-readable [literal cases](examples/sample_export_cases.json) retain
these expected values without implementing the W-169 algorithm.
`sample-export-contract-v1` is a hand-built software fixture specification, not
cell truth. No RNG or external TIFF is used. Labels below are native integer
IDs; array rows are Y, columns X and 2D retains singleton Z. Image channel zero
is `I[z,y,x]=100*z+10*y+x`, channel one is `I+1000`. All input voxels are exact
float32; `channels=(reference,stain)`. Native geometry is identity, spacing
`(2,1,1)`, origin `(10,20,30)`, unit `micrometer`, with explicit target frame
`literal-sample`. Unknown-calibration variants use null physical fields and an
explicit index frame. No biological calibration is implied by literal units.

| Case | Purpose and native labels | Requested → achieved / independent expectation |
| --- | --- | --- |
| E1 adjacent 2D | Distinct neighboring cells test valid reduction. ZYX `(1,2,4)`, rows `[1,1,2,2]`, repeated twice. | `(1,2,2)` succeeds; mask `[[[1,2]]]`, channel-zero means `[5.5,7.5]`, origin `(10,20.5,30.5)`; centroids exact |
| E2 small cell | One small cell tests majority loss detection. ZYX `(1,2,4)`, rows `[1,0,2,2]`, `[0,0,2,2]`. | `(1,2,2)` would erase `1`; fallback `(1,1,1)` exact; no relocation into background |
| E3 collision 2D | Two adjacent IDs competing for one block test tie behavior and impossibility. ZYX `(1,2,2)`, rows `[1,2]`, `[1,2]`. | Trial `(1,2,2)` chooses `1`; fails ID-set check; native fallback preserves both; pyramid stops |
| E4 adjacent 3D | True volumetric neighbors test Z reduction. ZYX `(2,2,4)`, both planes repeat E1. | `(2,2,2)` succeeds; labels `[[[1,2]]]`, channel-zero means `[55.5,57.5]`, origin `(11,20.5,30.5)` |
| E5 small/collision 3D | Two tiny cells test 3D disappearance. ZYX `(2,2,2)` all zero except `[0,0,0]=1`, `[1,1,1]=2`. | `(2,2,2)` chooses background; fallback native; unchanged coordinates and IDs |
| E6 namespace | Disjoint sources test equal local labels without biological merging. Two `(1,2,2)` all-`1` blocks, namespace `A` then `B`, occupy X `0:2` and `2:4`. | Export IDs `1,2`; reversible map `(A,1)→1`, `(B,1)→2`; overlap instead of disjoint placement must fail |
| E7 coarser failure | E1 at native base tests safe pyramid termination. | Native and `(1,2,2)` levels pass; proposed `(1,4,4)` is indivisible and terminates with reason |
| E8 invalid native | A declared cell `3` absent from E1 tests irrecoverable loss. | Hard failure at factor one; no successful export or synthetic voxel |
| E9 empty/unknown | All-zero `(1,2,4)` and zero molecule/cell rows test empty populations; unknown physical calibration tests metadata honesty. | Empty ID set survives reduction; units absent, index frame explicit; no fabricated physical scale |

For link tests supply final molecule keys `(A,7),(B,7),(A,8)`, mapped sample
coordinates `(0,0,0),(0,0,2),(0,1,1)`; the first two link to cells `(A,1),(B,1)`
and the third is explicitly unassigned. Separate unavailable-assignment and
omitted-trace variants retain the same points. Measured expression rows are
`(A,1):[2,0]`, `(B,1):[0,0]`, genes `(gene-A,gene-B)`; no equality with the
three illustrative points is claimed. Add a mask-only cell and a table-only
cell in separate link-validation cases, recording their missingness. Duplicate
keys, dangling assignments, bad hashes and absent transforms are rejection cases.
W-169/W-170 save exact JSON/NPY/Parquet component hashes before executing checks.

## One raster store, two viewing roles

W-170 writes one sample/section SpatialData Zarr v2 store. Its raster elements
are OME-NGFF `0.4` groups, opened directly by Fiji; there is no second raster
copy or independent OME writer. Use the pinned SpatialData writer once with the
already prepared multiscale arrays; do not let it generate a second pyramid.
Verify its stored datasets equal W-169's outputs. Points/tables and mappings are
additional elements/sidecars, not duplicated rasters. Unsupported writer behavior
blocks W-170; do not switch codecs or create a second writer silently.

```text
export/
  export-manifest.json          # source references, maps, geometry, level decisions
  open-in-fiji.json             # relative store/group paths, dimensions, units
  README.html                  # standalone opening instructions
  sample.zarr/
    images/reference/          # NGFF multiscales + channel metadata; levels 0,1,...
    images/stain/              # selected optional role, same grid as associated mask
    labels/cells/              # NGFF labels; exact uint32 IDs, background zero
    points/molecules/          # final points, QC, assignment and source keys
    tables/cell_expression/    # measured rows, region + instance_id binding
  cell-map.parquet             # reversible IDs; expression availability
  unmatched-cells.parquet      # explicit table-only population, if present
```

Element names are stable opaque keys; original round/channel/source names remain
metadata, never unescaped paths. Images serialize **CZYX**, labels **ZYX**; this is
a boundary transpose of canonical ZYXC. Preserve singleton Z. Chunks are clipped
to shape from image `(1,8,32,32)` / label `(8,32,32)`. Select Zarr v2 C-order,
`dimension_separator="/"`, no sharding, default fill `0`, lossless
`Blosc(cname="lz4", clevel=5, shuffle=1, blocksize=0)` via `numcodecs=0.15.1`.
Assert these values from every `.zarray`, including coarser levels, after writing.
A metadata-only finalization step must encode the actual per-level center
translations as well as SpatialData element-to-sample transforms; do not accept
the upstream writer's shape-derived scale alone as proof of alignment. Reopen
through both consumers to detect lost half-voxel offsets.
All metadata must remain readable without consolidated metadata; consolidation
may be added only after all writes. NGFF level transforms are scale then
translation, with C scale `1`/translation `0`. No time/round axis is implied.
Selected reference and stain roles retain their separate source identities.

Open `sample.zarr` using `spatialdata.read_zarr` and
`napari_spatialdata.Interactive` in the pinned environment; select the named
sample coordinate system and load reference, stain, cells and molecules. Use
cell-expression annotations to select a cell by instance ID and verify the
linked expression row. The Fiji role is image/mask inspection only: launch the
pinned Fiji, choose **Plugins → BigDataViewer → HDF5/N5/Zarr/OME-NGFF Viewer**
(the exact menu label must be captured during qualification), enter the path to
`sample.zarr`, then select `images/reference`, `images/stain` and `labels/cells`
as listed in `open-in-fiji.json`. Direct group paths remain the discoverable
entry points if the container root is not automatically recognized as an image.
Do not ask Fiji to interpret SpatialData expression tables.

## Pinned environment and qualification gate

The candidate is Python `3.12.12`, SpatialData `0.2.5`, OME-Zarr `0.10.3`, napari
`0.5.6`, napari-spatialdata `0.5.3`, PyQt5 `5.15.11`, Qt `5.15.16`, SIP `12.17.0`,
Zarr `2.18.7` and numcodecs `0.15.1`. The complete selected constraints are in
[export-viewer-requirements.txt](export-viewer-requirements.txt); the operator must
resolve and preserve a separate hash lock for all transitive packages. This
older Zarr-2 family is intentional. SpatialData `0.2.5` uses Xarray below
`2024.10`; Xarray `2024.7.0` satisfies DataTree `0.0.15`. A newer unconstrained
plugin or Zarr version is not interchangeable. Optional `spatialdata-io`, PyTorch
and GPU extras are not required by this export profile.

Primary dependency metadata:
[SpatialData 0.2.5](https://pypi.org/pypi/spatialdata/0.2.5/json),
[napari-spatialdata 0.5.3](https://pypi.org/pypi/napari-spatialdata/0.5.3/json),
[OME-Zarr 0.10.3](https://pypi.org/pypi/ome-zarr/0.10.3/json),
[Zarr 2.18.7](https://pypi.org/pypi/zarr/2.18.7/json).
Metadata compatibility and a resolved lock are setup evidence, not runtime success.

Select the preserved Fiji/ImageJ `2.14.0/1.54f` distribution with N5 Viewer
`6.1.1`, N5 `3.2.0`, N5-Zarr `1.3.4`, N5-Universe `1.6.0`, N5-ImgLib2 `7.0.0`,
N5-Blosc `1.1.1`, N5-IJ `4.2.1`, BigDataViewer core `10.4.12-tmpfix` and bundled
Java `1.8.0_66`. Freeze hashes of the complete distribution, native codec libraries
and actual Java executable before dependent use. Copy into an isolated setup if
changes are necessary; do not update the retained Fiji installation. The
[N5 reader documentation](https://imagej.net/libs/n5) describes NGFF opening;
actual plugin/codec/layout interoperability remains a W-170 runtime gate.

On GP099-29C the selected Python environment belongs under the external batch
artifacts, e.g. `artifacts/environment/export-viewer-v1/.venv`, with its own
`pyproject.toml`, lock, wheel hashes and setup manifest. Preserve batch-1 `.venv`
and `src/python/uv.lock`. Setup is a separately recorded operator phase after the
W-168 checkpoint; no install/sync during validation. Set `PYTHONPATH` to the
current authorized worktree's `src/python`, verify imported source paths and use
`UV_PROJECT_ENVIRONMENT` pointing to the isolated environment for dependent jobs.
Full STARfinder checks also require its existing docs/checkpoint/SimpleITK extras;
record their resolved versions when forming that environment. A requirements-only
viewer resolution does not qualify the combined STARfinder environment.

The host preflight found no Xvfb and did not qualify the forwarded display.
Qualify a virtual display/software OpenGL setup (`LIBGL_ALWAYS_SOFTWARE=1`,
`QT_API=pyqt5`, `CUDA_VISIBLE_DEVICES=""`) before viewer use. Native Qt libraries,
Java/Blosc and software rendering require a real process probe. Keep CPU affinity
`0`, numerical threads `1`, `1800 s/check`, `5400 s/worker`, RSS target `4 GiB`,
new artifacts `1 GiB`, arrays at most `32×64×64`, at most four channels/rounds.
Use local `/tmp` for temporary profiles and `/usr/bin/time -v` for checks; RSS is
KiB and this measurement is not a cgroup cap. Resource overrun stops the check.

### Actual reopening checklist (W-170)

Use E1/E4 for valid reduction, E2/E5 for native fallback, E7 for multiresolution,
and E9 for empty/unknown calibration. Each displayed condition explains its
purpose in one English sentence. Run the actual applications on saved exports:

1. Record exact process commands, interpreter/application/plugin/Qt/OpenGL
   versions, environment, input/store/file hashes and time/RSS/storage.
2. In napari, reopen through SpatialData; inspect separate channels, Z slices,
   2D pan/zoom, and 3D volume/labels/points on E4. Capture real canvas screenshots
   and event/selection state. Check explicit point coordinates and cell links.
3. In Fiji/N5 Viewer, open each raster group; change Z and zoom/region, inspect
   each resolution level, channel identity, calibrated or unknown units and
   image/mask alignment. Capture viewer window and selected dataset/level state.
   BDV provides 3D reslicing; qualify napari's volumetric 3D rendering separately.
4. Compare actual reopened voxel values/IDs and regional slices with native and
   prepared expectations. A small regional read must request fewer data chunks
   than the full multichunk volume (use a bounded `9×48×48` literal ramp extension
   if E4 fits one chunk); record chunk keys/counts, not only elapsed time.
5. Verify an assigned cell selection resolves its expression row, an unassigned
   point remains visible, and a source lookup retrieves the original trace or
   its explicit omission reason. Check all counts and null semantics numerically.
6. Save application logs, screenshots and criterion results. A disabled GPU must
   still yield a nonblank software-rendered canvas. Imports, schemas, reader-only
   checks and HTML browser opening cannot substitute for these sessions. Missing
   required capability blocks W-170. Additional human desktop inspection is optional.

## Saved evidence and delivery interface

Use `schema_name="starfinder.export_evidence"`, `schema_version=1`. Each immutable
`evidence.json` contains `issue_id`, `run_id`, UTC date, host, source commit/dirty
patch hash, contract/source hashes, environment/lock/source-path identity,
`inputs` (path, size, SHA-256, dataset version, selection, axes/dtype/shape/frame,
seed or no-RNG reason), requested/effective configuration, ordered `conditions`,
`checks`, output references/hashes, resources, limitations and ownership/retention.
Each condition has `id`, English `purpose`, input IDs, independent expected
values, observed values, units/tolerances, `status` (`passed`, `failed`,
`unexecuted`, `unavailable`), reason, table/figure refs and producer issue.
Checks contain argv/cwd/env, timestamps, exit code, log/hash, RSS KiB, artifact
bytes, and executed/reused status with exact reuse identity. Unknown is not pass.

| Producer | Saved evidence consumed by delivery |
| --- | --- |
| W-168 | Specification, literal expected cases, selected dependency metadata/lock, source inspection, focused checks and offline browser evidence; no new exporter results claimed |
| W-169 | Native/prepared arrays, all grid trials/fallback reasons, per-cell fidelity/counts, source hashes and independent expectations |
| W-170 | Serialized store/manifest, reopened numerical comparisons, actual napari/Fiji events/screenshots/versions and link checks |
| W-171 | Storage sizes/times, exact round-trip/downstream results, actual bounded Python/MATLAB results, mapping/matching protocol and check-reuse identity audit |
| W-172 | Integration of those saved results plus frozen runner telemetry; no processing rerun to fill missing data, no prediction of its own final acceptance |

The scientific renderer command contract is the same per owning issue:

```bash
uv run python /external/issue-run/deliver.py \
  --context /external/issue-run/delivery/context.json \
  --output /external/issue-run/delivery/packet
```

`context.json` is runner-owned: `issue`, `commit`, `status="validated"`, `checks`.
The issue-aware dispatcher must reject an unknown issue or missing producer
rather than emit an empty successful packet. `deliver.py` consumes adjacent saved
`evidence.json` and frozen specification/input snapshots; it never computes
processing results. W-168 implements its own entry point; each successor supplies
and executes its applicable implementation before independent acceptance.

Required outputs are standalone English `review.html`, `manifest.json` with
`code_commit` and `report_sha256`, and `browser-evidence.json` with `passed`, actual
command/version/input hashes and screenshot paths. Hash the manifest in an external
sidecar, avoiding self-hash cycles. Bind every controller check/log/hash; absent,
failed or altered required checks fail delivery. Report figures/tables must be
readable with networking disabled and no live kernel. Open with the real offline
browser, record exit/DOM/screenshot evidence and inspect the screenshot. Show
full host paths in fenced code blocks, host name and local opening/copy commands;
private paths alone do not establish public reproducibility. Preserve old packets.

At the handoff, the controller validates/commits W-168, invokes its pinned bootstrap,
independently reviews acceptance, then stops before W-169. The operator reconciles
the **stopped** configuration/guidance, pins all applicable issue-aware delivery
hooks and newly qualified environment/data identities, reads back prerequisites,
and only then continues. Never mutate a running config or clear STOP from a
worker. Empty hooks cannot qualify delivery. The pre-job frozen runner telemetry
schema remains authoritative: requested/observed routing, session/attempts,
identity/check reuse keys, usage coverage/deduplication, repairs/escalations,
resources and real commit/delivery/acceptance/Linear chronology. Unobserved
recovery branches remain unvalidated. W-172's terminal operational report is
supplementary to the scientific packet; final terminal evidence is appended only
after completion. No new human approval gate is introduced by this checkpoint.

Controller-owned integration gates remain default/extended pytest and strict
Sphinx/reference audit from [contributing](contributing.md). Focused W-168 checks
validate the specification fixture, dependency constraints and delivery behavior;
they do not run W-169/W-170 algorithms or W-171 MATLAB comparisons. Implementation
Done does not close W-57, W-93, human review or scientific acceptance.

## Python raster preparation

`starfinder.raster.prepare_rasters` implements the shared grid policy. Select
image roles and channels explicitly before calling it: each mapping value is an
`ImageLoadResult` with ZYXC data, ordered channel labels, source paths, and
upstream transform history in its diagnostics. Pass the same `ImageMetadata`
for labels; mismatched frame, geometry or shape fails before reduction.

```python
from starfinder.raster import RasterConfig, prepare_rasters

prepared = prepare_rasters(
    {"reference": selected_reference, "stain": selected_stain},
    aligned_labels,
    metadata=common_metadata,
    declared_ids=(1, 2),
    config=RasterConfig(factors_zyx=(1, 2, 2), coarser_levels=2,
                        coordinate_space="physical", max_output_bytes=1048576),
)
base = prepared.levels[0]
```

This condition requests lateral reduction while preserving separate channels and
all declared cells. The caller owns saved-artifact checksum verification and
selection; preparation neither loads artifacts nor replays a transform history.
Use `coordinate_space="index"` only with an explicit common named index frame
and all physical metadata fields `None`.

Every `RasterLevel` has aligned `images`, `labels`, `metadata`, absolute
`factors_zyx`, and `index_to_source_zyx`. The last is an output-to-native affine
center map, **not** a forward assembly transform. Physical `metadata` already
incorporates its offset. Index metadata stays uncalibrated; use the explicit
matrix for scale/translation without adding units. The result retains native
metadata, requested configuration, every trial and its failure reasons, per-cell
counts/support/centroid/Hausdorff metrics, dtypes, and total output array bytes.
`max_output_bytes` is an optional output-array budget, not an RSS cap; exhausting
it, including during native fallback, raises `RasterPreparationError` with
required shape/bytes and trial diagnostics. Input validation/native missing IDs
also fail explicitly, without a partial successful result.

`remap_labels` accepts disjoint masks that already occupy one aligned grid and
an exact explicit map from `(mask_namespace, local_label)` to
`(cell_namespace, cell_id)`. Both cell-key components are nonempty strings.
It returns `uint32` labels and reversible rows sorted by namespace/local ID.
Overlap and reused global cell keys fail; no assembly or biological equivalence
is inferred. Perform this remap separately and save its rows before preparation.
Preparation alone preserves the supplied mask dtype and IDs, including at native
resolution. Sources are not mutated. Expression/count/assignment inputs are not
accepted or recalculated. W-170 owns serialization and table binding.

## Sample serializer and explicit-level napari adapter

`starfinder.sample_export.export_sample` accepts the prepared `RasterResult`,
a `SampleExportConfig`, checksummed `SavedFile` references for raster inputs,
and a Parquet cell map. Optional molecular input is the existing `MoleculeIndex`;
assignments are a separately checksummed Parquet table and expression is a
checksummed H5AD. Every molecular source requires its explicit affine mapping.
The map's cell keys and instance IDs must be unique; all emitted mask levels
must contain exactly those positive IDs. Missing assignment rows become
`unavailable`; explicit `unassigned` rows stay distinct. Measured zero expression
rows are retained, mask-only cells stay unavailable, and table-only rows are
listed in `unmatched-cells.parquet` with the original H5AD source retained.
The destination must not exist. Writing occurs in a sibling temporary directory;
failed writes are not published as completed exports.

This opening condition uses each existing stored level's own calibrated map to
avoid the pinned default napari multiscale reader's lost half-voxel translations:

```python
import napari
from starfinder.sample_export import open_sample_viewer

viewer = open_sample_viewer('/path/to/export', level=0)
napari.run()
```

Use the layer list to select image channels and label levels together. This
adapter retains SpatialData's UI but exposes raster levels as single-resolution
napari layers with public `scale` and `translate`. Points remain in the common
frame. It does not implement automatic zoom-based level switching. Select a
label ID to show its cell key and original expression row in the **Cell and
molecule source** dock. Selecting a point resolves its source checkpoint and
trace, or its explicit omission reason. `lookup_export_trace` provides the same
source lookup without a GUI. Missing source bytes or wrong hashes raise errors.

NGFF dataset transforms contain absolute common-frame scale/translation for
Fiji. SpatialData's top-level element transform retains base-grid-to-sample
geometry. The adapter reads the dataset transforms directly, avoiding double
application. Default SpatialData/napari multiscale display is not qualified for
these stores: do not add the same raster again through the default element list.
Opening paths for the pinned Fiji N5 Viewer are in `open-in-fiji.json` and the
standalone export `README.html`. Actual Fiji interoperability remains a separate
runtime gate; these instructions do not certify an unexecuted consumer.

The current pinned Fiji N5 Viewer cannot yet parse the generated SpatialData
raster groups. This is a required interoperability blocker, not a successful
Fiji opening example. The manifest marks viewer qualification as unavailable;
software round trips and successful explicit-level napari viewing do not remove
that gate. In particular, the SpatialData top-level affine extension and NGFF
consumer interpretation still require reconciliation without duplicate rasters
or changed dependency pins.
