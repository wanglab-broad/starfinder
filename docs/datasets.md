# Dataset catalog

This is the canonical human-readable inventory of STARfinder development and
historical benchmark inputs. The [Chapter II register](https://linear.app/jiahaoh/document/chapter-ii-benchmark-datasets-and-experiments-c9b30ef7a1e5)
owns D01–D08, experiment readiness and scientific qualification.
[W-92](https://linear.app/jiahaoh/issue/W-92) owns real-input lineage;
[W-93](https://linear.app/jiahaoh/issue/W-93) owns synthetic truth, historical
provenance and calibration. Availability does **not** qualify an input for an
experiment. In particular, no entry here qualifies D04.

## Identity, versions and verification

Recovered source: repository `AGENTS.md` at
`43c95c636af3eaf949826080d06573909af41282`, committed
2026-09-15 22:09:11 −04:00 (2026-09-16 UTC). At baseline `912d788`, no maintained
successor inventory was found in the repository documentation; the live register
remains the scientific authority. Historical names are retained as version keys,
not assigned competing benchmark IDs. D01/D02/D03/D06 retain their existing IDs;
`aging`, named synthetic presets and registration derivatives identify catalog
versions, not newly selected benchmark datasets.

Last bounded verification: **2026-09-21**, GP099-29C, under
[W-153](https://linear.app/jiahaoh/issue/W-153). Evidence, exact sampled paths,
metadata/configuration hashes and fixture checksums are in `inventory.json`,
`metadata-details.json` and `manifest.json` under:

```text
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-153/20260921T021542Z-59970909/
```

These are private host paths, not public reproducibility or backup guarantees.
Header reads do not load image pixels. Except for the small test fixture, headers
were sampled, not exhaustively checked across rounds/FOVs/channels. Real-image
bytes were not hashed, processed, downloaded or scientifically evaluated.

States used here:

- **Historical/unverified:** recovered metadata or generator configuration only.
- **Available:** a local path and representative headers/metadata were read;
  completeness, lineage and fitness remain unverified.
- **Development-ready:** a pinned fixture passes a specified software check;
  this does not establish molecular or biological truth.
- **Missing:** absent at the particular inspected location, not globally absent.
- **Calibration/evaluation-qualified** and **retired** require an explicit owner
  decision and versioned evidence; no such promotion is made by this catalog.

Unless stated otherwise, physical spacing, units, acquisition date, original
archive-member identity, masks, independent biological units and evaluation split
are **unknown**. TIFF resolution tags with `ResolutionUnit=1` do not establish
physical calibration. Shapes below are explicitly **ZYX**, per channel; combined
Python rounds are **ZYXC**. “2D” in a dataset name does not mean Z=1.

## Real sequencing inputs

Local root `R = /home/unix/jiahao/wanglab/Data/Processed/sample-dataset`.
Historical common accession: **10.5281/zenodo.11176779**. Its association with each
local file/version has not been verified; do not treat that citation as a checked
archive-to-file lineage. These are real STARmap-related sequencing inputs;
precise assay/specimen and upstream processing provenance require W-92. Filenames
include `cmle`, but the exact restoration recipe/version is unknown. They are
inputs to STARfinder, not established raw acquisition data.

| Register / local version | State; location | FOVs and rounds | Sampled shape / dtype | Codebook and channels | Intended use / limits |
| --- | --- | --- | --- | --- | --- |
| D01 / `tissue-2D` | Available; `R/tissue-2D` | 56 `tile_*` directories in round1; round1–4; historical reference round1 | `(30,3072,3072)`, uint8 | `genes.csv`: 64 rows, no header; first round/FOV has ch00–ch04 | Real molecular anchor, provisional calibration candidate; not qualified, no matching cell truth established |
| D02 / `cell-culture-3D` | Available; `R/cell-culture-3D` | 70 Position351–420 in round1; round1–6; historical reference round1 | `(30,1496,1496)`, uint8 | `genes.csv`: 998 rows, no header; first round/FOV has ch00–ch04 | Culture/assignment/segmentation candidate; 3D stack is not independent 3D cell-boundary truth |
| D03 / `LN` | Available; `R/LN` | 64 Position001–064 in round1; round1–4; historical reference round4 | `(50,1496,1496)`, uint8 | `genes.csv`: **62 rows**, first field has BOM; first round/FOV has ch00–ch03 | Volumetric real candidate; historical 61-gene count is unresolved, not corrected by assuming every row is a distinct gene |
| `aging` / historical local sample | Available; `R/aging`; no new benchmark ID | Six Position400–405 in round1; round1–9; historical full count 848, reference round1 | `(36,2048,2048)`, uint8 | `genes.csv`: **14,242 rows** of probe/barcode entries; ch00–ch04 in sampled FOV | Historical 2,044-gene, two-segment example; probe rows are not gene counts; membership in D08 is not a selection decision |

The codebook row counts above are bounded CSV observations, not validated decoding
schemas. Completeness of later-round FOV/channel sets was not checked. Historical
sequencing channel order was `ch00,ch02,ch01,ch03`; observed ch04 must not be
silently included or interpreted as a sequencing channel. Explicit selection is
required by [image contracts](api/contracts.md).

Historical parameters are preserved here as **unverified settings**, not physical
measurements or defaults: cell-culture-3D `voxel_size=(1,2,2)`, end `CC`, adaptive
0.2; tissue-2D `(1,1,1)`, end `CC`, adaptive 0.4; LN `(1,1,1)`, start `A`, end `AC`,
adaptive 0.2; aging `(0.35,0.14,0.14)`, split index 5, ends `CC`/`TT`, adaptive
0.2. Axis order and units of those historical `voxel_size` tuples were not
specified. They must not be promoted into calibrated image spacing.

`dataset-info.json` exists for D01/D02 and describes one selected FOV despite
larger image-directory counts. It lists additional protein/organelle rounds,
rotation −90°, and D01 maximum projection. These are configuration requests, not
proof those operations have already been applied. Representative protein/organelle
headers remain 3D; local configuration paths beginning `~/sample-dataset` are
historical aliases, not the actual host root above. W-92 must reconcile source,
processing stage, assay identity, geometry and masks before qualification.

## Existing register entries without a new selected input

| ID | Identity / lineage | State and limits |
| --- | --- | --- |
| D04 | Condition-specific processed-image synthetic molecular foundation; W-93 | Historical generators below are related development/history inputs, **not frozen D04**. Actual rendered truth, independent streams/splits, historical provenance and calibration remain open |
| D05 | Leica SP8 raw/Huygens pairs; [W-114](https://linear.app/jiahaoh/issue/W-114) | Historical/unverified here; exact accession/local input, FOV/round/channel/codebook, shape/dtype/spacing and paired lineage unknown. Huygens is a comparator, not truth |
| D07 | Derived from D01 tile_1, all four rounds: Z `[0:30]`, Y/X `[1280:1792]` | Selected recipe, **not materialized or verified here**; expected ZYX `(30,512,512)` from source indexing, dtype preserved by recipe. Origin/spacing, source hashes, signal suitability and export path remain unverified; not an independent biological sample |
| D08 | Unselected pool of additional real/public inputs | No new selection, accession, local path or dimensional metadata assigned; qualify a specific question and input before inclusion |

## Historical synthetic sequencing inputs

These are synthetic Gaussian **processed-image** scenes, not simulated original
RNA populations, raw microscopy, cell masks or calibrated tissue. Local benchmark
root `B = /home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark`.
The source preset configuration is `starfinder.synthetic._presets` with
`SyntheticConfig`; versioned fixture metadata is distinct from current generator
code. All listed stored sequencing versions declare four rounds/four channels,
two FOVs (`FOV_001`, `FOV_002`) and seed 42 unless marked unknown. Physical spacing
and units are unknown; use voxel-index geometry, not invented micrometres.

| Stable catalog version / register | Location and state | ZYX / dtype | Identity/configuration and limits |
| --- | --- | --- | --- |
| `sequencing/tiny` | Current source preset; historical/unverified stored dataset (`B/e2e/data/tiny` missing) | Source `(8,128,128)`, uint8 | 8 genes, 10 spots/FOV; current default seed 42 is not a historical artifact seed; bounded example exception only |
| D06 / `sequencing/small/truth-v2.0` | `tests/fixtures/synthetic/small`; development-ready for existing end-to-end regression checks | All 32 TIFFs `(16,256,256)`, uint8 | Metadata version 2.0, seed 42, 8 genes, 50 spots/FOV; codebook and truth tracked, TIFFs ignored; software orchestration only |
| `sequencing/medium/truth-v1.0` | `tests/fixtures/synthetic/medium` metadata; TIFFs available in original checkout, absent in dedicated worktree | Sampled `(32,512,512)`, uint8 | Metadata says preset `custom`, seed 42, 8 genes; historical 100 spots/FOV; not copied/executed in this batch |
| `sequencing/large/truth-v1.0` | `B/e2e/data/large`; available | Sampled `(30,1024,1024)`, uint8 | Metadata `custom`, seed 42, 64 genes; historical 2,000 spots/FOV conflicts with current source preset 1,500; generator revision/effective configuration unverified |
| `sequencing/tissue/historical` | `B/e2e/data/tissue`; available headers | Sampled `(30,3072,3072)`, uint8 | Historical 64 genes, 14,000 spots/FOV, two FOVs; truth version/seed/full effective config not read in bounded verification |
| `sequencing/thick_medium/truth-v1.0` | `B/e2e/data/thick_medium`; available | Sampled `(100,1024,1024)`, uint8 | Metadata `custom`, seed 42, 64 genes; historical 5,200 spots/FOV; historical generator revision unverified |

`B/e2e/data/small` and `medium` are missing at those exact paths; this does not
contradict the repository fixtures. The historical medium count of 100 spots/FOV
also differs from the current source preset's 400; never substitute a newly
generated preset for existing fixture bytes. For large/tissue/thick_medium, `codebook.csv`
and full file-set completeness remain unverified. Additional discovered
`B/e2e_LR/data/large` and `thick_medium` truth metadata declare v1.0 and seeds
123/789 respectively, shapes matching their names. Treat these as distinct
historical versions, not replacements for the seed-42 inputs; image headers,
lineage, codebooks and effective deformation recipes were not checked there.

The current generator records `molecular_truth=None`, integer centers and rounded
deformations, and uses a process-dependent `hash()` for registration seeds.
`background_std` is unused. A passing software regression check does not repair
or qualify these historical assumptions. W-155/W-157 specify and implement the
new development model; W-167 returns independent evidence to W-93. Historical
v1/v2 provenance, the large-shift discrepancy and later calibration stay with W-93.

## Historical registration derivatives

| Stable version / parent | Path under B; state | Sampled ZYX / dtype | Lineage and limitations |
| --- | --- | --- | --- |
| `D01/registration/round1-round2` | `registration/data/real/tissue_2D`; available | `(30,3072,3072)`, uint8 | Metadata: tile_1, round1 reference, round2 moving, 4 source channels, unknown true shift |
| `D02/registration/round1-round2` | `registration/data/real/cell_culture_3D`; available | `(30,1496,1496)`, uint8 | Metadata: Position351, round1/round2, 4 source channels, unknown true shift |
| `D03/registration/round1-round2` | `registration/data/real/LN`; available | `(50,1496,1496)`, uint8 | Metadata: Position001, round1/round2, 4 source channels, unknown true shift |

Each has single-channel `ref.tif`/`mov.tif`. Historical AGENTS called them “MIP
extractions”, but both inspected headers retain Z. The projection/merge axis and
exact source-channel composition, extraction script revision, source hashes and
calibration are unverified. Do not label these Z-projected images. No codebook is
attached to these registration pairs, and they provide no molecular truth.

Synthetic registration versions live at `B/registration/data/synthetic/<preset>`.
All six directories exist; two moving-image headers per directory were read:

| Stable version | ZYX / dtype | Metadata checked |
| --- | --- | --- |
| `registration/tiny/historical` | `(8,128,128)`, uint8 | seed 42; 10 points |
| `registration/small/historical` | `(16,256,256)`, uint8 | seed 42; 50 points |
| `registration/medium/historical` | `(32,512,512)`, uint8 | seed 42; 400 points |
| `registration/large/historical` | `(30,1024,1024)`, uint8 | seed 42; 1,500 points |
| `registration/tissue/historical` | `(30,3072,3072)`, uint8 | Full truth/config not read |
| `registration/thick_medium/historical` | `(100,1024,1024)`, uint8 | Full truth/config not read |

State: **available**, not evaluation-qualified. The first four JSON records list
one reference and seven independent moving cases: shift, polynomial small/large,
Gaussian small/large, multi-point and linear small. These are single-channel
registration comparisons, not eight sequencing rounds. They preserve field-file
names and recorded shifts; field contents, complete image sets and generating
revision/effective hash seeds remain unverified. Assay/codebook: not applicable.
Physical spacing: unknown. Intended use: historical registration development;
reusing accuracy claims requires W-93/W-94 provenance and protocol review.

## Development fixtures and resource boundaries

New fixture deliveries must add a version here with source/parent or generator
revision, effective config/order/seeds, stage, identities, axes/shape/dtype,
spacing/units (or unknown), intended use, limitations, verification date and an
external manifest link. Preserve older versions and hashes. Per-run logs, images
and measurements remain outside Git; do not make this catalog a run diary.
No new generated scientific fixture is introduced by W-153.

### Artifact contract example v1

`docs/examples/artifact_contracts.py` defines the hand-constructed synthetic
development example `artifact-contract-v1` for [artifact contracts](artifact-contracts.md).
Source is this versioned script, not a historical generator or D04. It uses
no random streams (seed not applicable), accession or external input. One sample
has distinct FOV namespaces for Z=1 and Z=3, two explicitly ordered rounds
(`round10`, `round2`), four channels (`ch02`, `ch00`, `ch03`, `ch01`) and a one-gene
codebook with explicit nonidentity color mapping. Arrays are uint16 ZYXC
`(1,4,5,4)` or `(3,4,5,4)`; extraction is float64 NCR `(2,4,2)` with Boolean NR
validity. Prepared/reference, transformed image, candidate/signal, decoded and
accepted in-memory stages are exercised. Image spacing/units are unknown;
a separate analytic geometry example supplies spacing `(2,3,4)` and units `um`
without implying calibration of the images. Bounded numerical verification on
2026-09-21 passed; format round trips remain unverified.

Intended use is numerical/schema development, not assay realism, molecular
truth or a storage round-trip qualification. No TIFF fixture is generated or
replaced. Verification date and source/config hashes belong to the W-154 manifest:
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-154/20260921T024246Z-52ac45ba/implementation-manifest.json`.
Subsequent storage implementations must retain this example's independent
expectations and record their own format/reload evidence.

### Synthetic specification oracle v1

`synthetic-contract-v1` is defined by
[`docs/examples/synthetic_specification.py`](examples/synthetic_specification.py)
and the [versioned model specification](synthetic-specification.md). It contains
hand-calculated spot/signal/geometry expectations and development stream probes
(root seed 42, scene key `formed-v1`, SHA-256/PCG64). It has no external data,
accession, image files, physical calibration or historical fixture ancestry.
The oracle uses float64 arithmetic and four-element noise probes; it does not
render an image or establish molecular truth. Its declared downstream clean
presets are `formed-small-v1` (8,32,32) and `formed-z1-v1` (1,32,32), float32
ZYXC, three explicitly ordered rounds, four channels and two genes, with all
effects disabled. They are **specified, not generated or qualified** here;
W-157 must record actual generator/version/configuration and output hashes.
They do not replace D06 or qualify D04. Round/channel/codebook mappings and
independent tolerances are canonical in the specification.

W-155 bounded verification evidence and source/config hashes (2026-09-21):
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-155/20260921T031105Z-dd5b2c86/implementation-manifest.json`.
Scientific calibration, backend image rendering and storage round trips remain
unverified by this oracle; ownership/retention follows the catalog convention.

### Provenance literal example v1

`provenance-literal-v1` is a hand-constructed software fixture defined by
`docs/examples/provenance.py` and `src/python/test/test_provenance.py` (2026-09-21).
It has dataset/sample `literal/sample`, FOV `FOV` in tests or `FOV-Z1/FOV-Z3`
in the example, rounds `(round10,round2)`, channels `(b,a,d,c)` and one gene
with color sequence `11` and the standard color-to-channel mapping. Supplied
uint16 ZYXC arrays are `(1,7,9,4)` or `(3,7,9,4)`; the local-registration
diagnostic test uses `(4,7,9,4)` to meet the existing Demons minimum. One channel
has a single value 7 at `(Z//2,3,4)` in both rounds; the empty variant is zero.
There is no generator, random seed, accession, historical TIFF ancestor or
scientific truth claim. Prepared, registered, candidate/signal, decoded and
filtered **in-memory** states supply provenance records; images/tables are not
checkpointed by the recorder. Geometry is unknown except for the explicit
metadata test `(spacing,origin,direction,unit)=((2,3,4),(10,20,30),diag(1,-1,-1),um)`.
That literal calibration is a software oracle, not a measurement. New temporary
TIFFs in the loader-attribution test are derived only from these literal arrays;
no existing fixture TIFF is regenerated.

Source/config hashes, execution evidence and limits are in the external W-156
`implementation-manifest.json` under
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-156/20260921T032857Z-dbab02d9/`.
Intended use is metadata/integrity/failure regression. No D04 qualification,
image/table reload equivalence or molecular acceptance is implied.

### Formed-amplicon clean development v1

`formed-small-v1` and `formed-z1-v1` are new synthetic **development** scenes
from `starfinder.synthetic.generate_formed_scene`, generator version `1`, under
`starfinder.synthetic/1` (specification revision
`f9512694a0960c10ce5236efbaaf9d6f425c1d8a`). They have no accession, external
input, historical fixture ancestor or calibrated D04 status. The configuration
is supplied by `formed_scene_preset`: shape ZYX `(8,32,32)` or `(1,32,32)`,
float32 ZYXC, N=8 uniform formed positions, root seed 42, scene `formed-v1`,
development split, sample `sample`, FOV `FOV_001`. Each preset name is its dataset
version. Round order is `(round10,round2,round1)`, channels
`(ch02,ch00,ch03,ch01)`, colors map `1→1,2→0,3→3,4→2`; supplied codebook rows are
`gene-A=123`, `gene-B=214` with equal abundance. Brightness is 100, axial/lateral
sigma and elongation 1, angle 0; all effects disabled. Physical spacing/origin/
direction/units are unknown; lengths are voxel indices. Z=1 samples the same
3D kernel rather than projecting a volume.

The generated stage is a clean processed image plus complete formed/per-round
truth, float64 NCR intended/pre-mix/realized amplitudes, stable IDs, visibility,
configuration and SHA-256/PCG64 stream descriptors. No eligibility or biological
RNA truth is inferred. Historical fixture TIFFs are unchanged. These fixtures
exercise numerical/software reproducibility; they do not qualify assay realism,
historical seeds, scientific accuracy, or checkpoint reloads.

`docs/examples/formed_scene.py` also defines independent two-object 3D/Z=1
extraction oracles, shape `(3,7,9)` or `(1,7,9)`, float32, A=8, widths=.25,
explicit centers `(Z//2,2,2)` and `(Z//2,4,6)` and supplied gene-A/gene-B IDs.
They use the same labels/mapping with no random property draws. Detector IDs are
distinct; literal extracted signals and decoded calls are independently checked.

Bounded verification date: 2026-09-21. Exact generator source/patch identities,
configurations, rendered output hashes and environment are recorded in the W-157
external `implementation-manifest.json` and `preset-artifacts/` under
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-157/20260921T040822Z-08c03032/`.
This private evidence retains Jiahao's ownership/retention and unverified backup
status; it does not establish public reproducibility.

### Existing fixture inventory

The small D06 fixture's 32 existing TIFFs occupy 33,645,376 bytes. W-153 checked
all headers and SHA-256 hashes, compared truth/codebook bytes against the original
checkout, then copied the existing TIFFs into this dedicated worktree. No TIFF
was regenerated. The fixture manifest is in the evidence directory above.
A fresh checkout lacks ignored TIFFs; directory existence alone is insufficient.
The medium fixture's 32 TIFFs occupy 268,611,392 bytes at the original location;
only representative headers/metadata were inspected, and it is not approved for
execution by this batch's small-fixture exception.

Existing baseline test allocations must be distinguished from new fixture limits:

| Existing fixture/check | Bounded allocation / source |
| --- | --- |
| D06 consumers | `(16,256,256)` per channel, four channels/four rounds; `test/conftest.py` and small-fixture consumers |
| Pointset tests | `(16,128,128)`; `test/test_pointset.py` |
| Compression test | `(10,128,128)` uint8; `test/test_io.py` |
| Legacy deformation helper tests | `(10,100,100,3)` float32 fields; `test/test_benchmark_synthetic.py`; inside the existing pointset spatial envelope |
| Tiny generation/examples | `(8,128,128)` per channel, up to four channels/four sequencing rounds; `test/test_benchmark_synthetic.py`, `docs/examples/quickstart.py` |
| Contract tests | Image/coordination/synthetic tests use spatial shapes at most `(8,32,32)`; registration pair test creates seven independent moving cases, not one eight-round sequencing fixture |

The historical `256×1024 float32` benchmark-allocation exception was not found as
an allocation in the current tests; do not recreate it merely because old
instructions mention it. Source preset lookups of medium/large sizes do not
allocate their images. Test code/config identities belong in each run manifest.
New images remain at most `(32,64,64)` ZYX with four rounds/four channels. Existing
exceptions do not authorize new large fixtures, sweeps or real-image processing.
Use the current AGENTS.md/issue profile for time, CPU, memory and storage controls.

### Image checkpoint literal v1

`image-contract-v1` is the hand-constructed software fixture in
`docs/examples/image_checkpoints.py` and `src/python/test/test_image_checkpoints.py`.
It extends the W-154 artifact arithmetic without using a historical TIFF,
external accession, random generator or molecular truth. Sample `sample` has
`FOV-Z1`/`FOV-Z3` in the example (test FOV `FOV`), sequencing order
`(round10,round2)` and channels `(ch02,ch00,ch03,ch01)`. Example images are uint16
ZYXC `(1,4,5,4)` or `(3,4,5,4)`, with value 7 in reference channel 1 at
`(Z//2,2,1)` and value 9 in moving channel 0 at `(Z//2,1,2)`. The explicit
correction `(0,1,-1)` aligns them. Codebook `gene-A=12` uses the nonidentity
mapping `1→1,2→0,3→3,4→2`. Physical calibration is unknown; no seed applies.
Prepared and registered HDF5 stages, a run record and exact extraction/decoding/
filtering equivalence are saved externally.

I1 test variants use literal `arange` uint8/uint16/int16/float32/float64 arrays,
signed values and signed zero, the specification's explicit geometry
`((2,3,4),(10,20,30),diag(1,-1,-1),um)`, plus a ZYX `(1,2,3)` uint8 stain and
`(2,3,4)` float32 registration reference with distinct unknown frames. These are
software geometry oracles, not calibration measurements. I2 dense fields are
float32/float64 ZYX3; the half-index ramp independently requires nearest-even
`[0,2,2,4,0]`. Tests also retain partial/failed state and reject corrupt components.
New temporary TIFFs are derived only from these literal arrays; no existing
fixture bytes are regenerated. Maximum new image bounds remain `(3,4,5,4)`.

Verification date: 2026-09-21. Source/config/input/output hashes, exact commands,
measured resources and small-fixture storage costs are recorded in
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-158/20260921T044025Z-6f7ab365/implementation-manifest.json`
and `image-example-handoff/summary.json`. This is software round-trip evidence, not
E10 format superiority, D04 qualification or public reproducibility. Owner,
retention and unverified backup status follow this catalog's convention.

### Candidate checkpoint literal extension v1

W-159 reuses `artifact-contract-v1` above without regenerating image fixtures.
`docs/examples/candidate_checkpoints.py` saves its complete two-candidate table,
float64 `(2,4,2)` signals and Boolean `(2,2)` validity in Parquet, with empty
N=0 and invalid-A-round2 variants. Geometry stays unknown. The nonlexical round/
channel labels, nonidentity codebook mapping and literal gene-A/no-signal oracle
are unchanged. Tests add nullable Int16/Boolean/string, uint8/float32 optional
columns, signed zero, physical row reordering and malformed artifacts; these are
schema/integrity probes, not molecular truth. The batch/streaming save-policy
checks reuse the bounded provenance literal fixture above. No seed, accession,
historical TIFF or new scientific population applies.

Focused format evidence and saved output hashes belong to W-159's external
`implementation-manifest.json` and `candidate-example-handoff/summary.json` under
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-159/20260921T051139Z-97c371d3/`.
Small-file sizes/timings are illustrative; W-171 owns storage qualification.
Owner Jiahao, retention through thesis/publication, backup unverified. These
private development artifacts do not establish public reproducibility.

### Saved formed development v1

`saved-formed-z3-v1` and `saved-formed-z1-v1` are the explicit two-object
historical integration fixtures from `docs/examples/saved_synthetic.py` at
commit `1bc7783`; the current example produces v2 below. They derive from
the formed development v1 model above, generator version 1 / synthetic contract
`starfinder.synthetic/1`, with no external accession or historical TIFF ancestor.
Root seed 42, scene `formed-v1`, development split, sample `sample`, FOV `FOV_001`;
three ordered rounds `(round10,round2,round1)`, four channels
`(ch02,ch00,ch03,ch01)`, mapping `1→1,2→0,3→3,4→2`. Explicit `formed-A`/`formed-B`
centers are `(Z//2,2,2)`/`(Z//2,4,6)` with genes `gene-A=123`/`gene-B=214`.
Brightness is 8, axial/lateral sigma .25, elongation 1, angle 0, all effects off.
Shape is float32 ZYXC `(3,7,9,4)` or `(1,7,9,4)`; physical calibration is unknown.
This is the same sampled 3D model for Z=1, not a projection or calibrated D04.

The script saves complete formed/per-round truth and float64 NCR intended,
pre-mix and realized signals; prepared/registered HDF5, run provenance and
pre-rejection Parquet checkpoint; and example-only uninterrupted decoding/QC
comparison tables. These comparison tables are not a new decoded/final artifact
API. It independently requires two objects, six history rows, literal active
amplitudes 8 and inactive 0; exact image/extraction/downstream round trips use
separate processes. Registration is an explicit zero-correction clean control.
Report-only display projections preserve the original saved Z geometry.

Verification date 2026-09-21; effective config/order/stream descriptors,
generator/source checksums, output identities and limitations belong to the
external W-160 manifest under
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-160/20260921T054407Z-db98e9ef/`.
Intended use: bounded persistence/integration and human software review, not
scientific accuracy, historical seed qualification or public reproducibility.
Jiahao owns retention through thesis/publication; backup unverified.


### Saved formed development v2

`saved-formed-z9-v2` and `saved-formed-z1-v2` replace v1 for the maintained
saved example. Parent model remains formed development v1, generator 1,
`starfinder.synthetic/1`; no external data or calibrated D04 ancestor.
Float32 ZYXC shapes are `(9,32,32,4)` and the intentional `(1,32,32,4)`.
Two simulated IDs `spot-A`/`spot-B` have centers `(Z//2,10,10)`/`(Z//2,22,22)`,
gene truth `gene-A=123`/`gene-B=214`, peak brightness 8, axial sigma 1,
lateral sigma 1.25, elongation 1 and angle 0. Support is the closed four-sigma
ellipsoid with zero outside; the 3D support fits, while Z=1 samples its center
slice without integration or renormalization and records axial truncation.

Root seed 42, explicit identity/placement, scene `formed-v1`, development split,
sample `sample`, FOV `FOV_001`, ordered rounds `(round10,round2,round1)`, channels
`(ch02,ch00,ch03,ch01)`, color mapping and molecular processing are unchanged.
Effects/noise/background/deformation remain disabled; physical calibration is
unknown. Simulation and detection namespaces remain distinct, with explicit
coordinate correspondence rather than row-position joins.

The maintained example independently checks every image voxel against sampled
Gaussian values (absolute tolerance 1e-6, relative tolerance 0), literal truth and
exact checkpoint/downstream round trips in separate processes. Per-round TIFF
inspection exports preserve all float32 values and plane labels in ImageJ ZCYX
storage. The [Fiji recipe](fiji-inspection.md) reads canonical ZYXC HDF5 directly;
no checkpoint layout/schema is replaced. The viewer's absent-calibration default
is explicitly removed. Browser/Fiji runtime evidence is recorded per delivery.

State: development-ready, not scientifically calibrated/evaluation-qualified.
Verification date 2026-09-21; source/config/stream/seed and file hashes, fresh-process
checks and viewer evidence are in the W-175 manifest under
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-175/20260921T180549Z-round2/`.
Original v1 artifacts remain intact. Intended use: small software integration and
human inspection. Jiahao owns retention through thesis/publication; backup and
public-release reproducibility remain unverified.


### Saved formed development v3

`saved-formed-z9-v3` and `saved-formed-z1-v3` retain all v2 numerical settings
and molecular processing. Ground-truth IDs are now `gt-A`/`gt-B`; detector display
IDs `spot-1`/`spot-2` derive from the stable zero-based detector IDs, never row
positions after sorting/filtering. Coordinate correspondence is explicitly
`spot-1 ↔ gt-B`, `spot-2 ↔ gt-A`. Simulation and detection namespaces stay separate.

Saved simulation truth includes literal nucleotide expectations `gt-A=CCAG` and
`gt-B=CAAT` under start base C, corresponding to GT colors `123` and `214`.
The model itself specifies color-space truth, not independently sampled DNA.
Report barcodes use the existing nucleotide decoder on observed saved color calls,
not the gene assignment. All detections include GT/color/barcode/gene comparison
and a separate filter status/reason. WTA is exact matching with no correction or
matching distance; endpoint filtering is disabled. Earlier packets remain intact.

### Controlled readout development v1

`readout-contract-z3-v1` and `readout-contract-z1-v1` are software arithmetic
fixtures defined by `docs/examples/readout_effects.py`, parent formed development
v1 / `starfinder.synthetic/1`, generator version 2. Frozen specification revision
is `f9512694a0960c10ce5236efbaaf9d6f425c1d8a`. No external input, accession,
calibration, historical TIFF ancestor or biological RNA truth is claimed.
Existing clean presets retain their numerical values with this generator;
version 2 adds optional readout configuration/provenance, not new clean images.

Each case has one explicit `gt-A` at `(Z//2,3,4)`, gene-A color codeword `222`,
A=8, widths/elongation=1, angle=0; float32 ZYXC `(3,7,9,4)` or `(1,7,9,4)`.
Physical spacing/units remain unknown; coordinates/widths are voxel indices.
Seed 42, scene `formed-v1`, development split, sample `sample`, FOV `FOV_001`;
round order `(round10,round2,round1)`, channels `(ch02,ch00,ch03,ch01)` and
mapping `1→1,2→0,3→3,4→2`. Z=1 samples the same 3D kernel with axial truncation.

Cases retain clean, trend b=.5, middle-round dropout or weakening factor .25,
and loss from index 1. Independent expected active histories are [8,8,8],
[8,4,2], [8,0,2], [8,1,2], [8,0,0]. Mixing adds M[1,0]=.25 to identity;
the combined case also uses gain .5, trend .5 and middle weakening .25.
All expectations are binary-representable and checked exactly, including
rendered centers/extraction. Full intended/pre-mix/realized signals, formed and
round truth, requested/effective config and keyed SHA-256/PCG64 streams survive.
Background, noise and deformation remain disabled. A supplied detector point
`spot-1` corresponds explicitly to `gt-A`; neither ID nor row ordering establishes
correspondence. Tests use intermediate probabilities to verify independent
round.dropout/round.weakening/round.loss streams and cross-process invariance.

Source/config hashes, environment, measured checks and example output identities
are retained in `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-161/20260921T195700Z-c102e072/implementation-manifest.json`.
Intended use is bounded effective-model development, not calibrated evaluation.
No historical fixture bytes change. Owner Jiahao; retain through thesis/publication;
backup and public-release reproducibility remain unverified.

### Structured background development v1

`structured-background-development-v1` is the bounded W-162 in-memory example
in `docs/examples/background_noise.py`, using synthetic/1 and generator version 3.
It inherits seed 42, scene `formed-v1`, development split, sample/FOV identity,
three ordered rounds, four channels, codebook and mapping from `formed_scene_preset`.
Shapes are `(3,7,9)` and `(1,7,9)` ZYX, float64 ZYXC, unknown physical calibration,
N=0 (background-only oracle). It independently checks a normalized X gradient
with intercept 1/slopes (0,0,8); a height-8 Gaussian at (0,2,2) of width (1,1,1),
including tails beyond four sigma; and two uniform texture blobs with widths
(1,3,3)/height 5 and unit destination weights, with residual alpha=4/sigma=2.
Background/noise enabled individually as shown in the source; geometry is identity.
No historical TIFFs or measured backgrounds are consumed or regenerated.

`test_background_noise.py` also uses bounded `(3,7,9)` molecule/isolation probes
and `(8,32,32)` residual-statistics arrays (three rounds, four channels), literal
A5/A6 arithmetic, explicit blob/count-density laws and independent SHA-256/PCG64
expectations. Configurations and tolerances are maintained in that test source;
run manifests pin source hashes, commands, measurements and provenance. No fixture
bytes are committed. Analytic backgrounds are not cells, empirical tissue or
calibrated evaluation data; W-93/W-167 qualification remains separate. Latent
records, standardized noise stream hashes and pre-noise/final image hashes are
retained independently. Owner Jiahao; external evidence retained through
thesis/publication, backup unverified.

### Shared geometry development v1

`geometry-contract-z3-v1` and `geometry-contract-z1-v1` are bounded in-memory
fixtures in `docs/examples/formed_geometry.py`, synthetic/1 generator version 4,
parent formed development v1, frozen specification revision
`f9512694a0960c10ce5236efbaaf9d6f425c1d8a`. Shapes `(3,7,9)` and `(1,7,9)` ZYX,
float64 ZYXC, three rounds/four channels and mapping inherited from
`formed_scene_preset`. Seed 42, development split, scene `formed-v1`, sample
`sample`, FOV `FOV_001`; physical calibration unknown, lengths in voxel indices.
One `gt-A`, gene-A=123, center `(Z//2,2,3)`, A=8 and unit widths/elongation,
angle zero. A height-8, unit-width analytic background region shares its center;
destination weights `(1,0,0,0)` separate it from the first-round molecule channel.

Round translations are `(0,0,.5)`, identity and `(0,0,-3.5)`. One local control
at the reference center, scale 2, has vector `(0,0,.25)` in round10 and zero in
later rounds. Expected X positions are 3.75, 3 and -0.5; the final out-of-frame
center still contributes at x=0. Background uses the inverse map; puncta retain
fixed shapes about mapped centers. No noise/readout effects, external data,
legacy TIFF generation, empirical realism or D04 qualification is implied.
Clean image arrays retain their previous values; version 4 adds geometry
configuration, frame and inverse diagnostics to truth/provenance.

`test_formed_geometry.py` adds explicit fractional shifts, independently computed
local/inverse expectations and seed-42 geometry/noise isolation probes, all within
`(3,7,9)` and `(1,7,9)`. Source/config hashes and measured outcomes are recorded in
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-163/20260921T204405Z-ab340050/implementation-manifest.json`.
No fixture image bytes are committed. Owner Jiahao; retain evidence through
thesis/publication; backup and public-release reproducibility unverified.

### Molecular checkpoint literal extension v1

`molecular-contract-v1` reuses `artifact-contract-v1` (W-154) arithmetic in
`docs/examples/molecular_checkpoints.py`: uint16 ZYXC `(3,4,5,4)` or `(1,4,5,4)`,
R=2, C=4, candidates A/B, radius-zero float64 NCR signals, ordered rounds
`(round10,round2)` and channels `(ch02,ch00,ch03,ch01)`. Color mapping
`1→1,2→0,3→3,4→2` yields A=`12`/gene-A and B=no_signal. Physical calibration
is unknown. No accession, random seed, historical TIFF regeneration or new
molecular truth applies. FOV namespaces explicitly separate colliding local IDs.

Normal, empty, all-rejected and explicitly omitted-trace variants save distinct
pre-QC/final artifacts and a reversed sample/section index. Tests additionally
cover invalid measurement, tied calls, codebook-aware diagnostics, physical
row reordering and corrupt/missing sources. Literal source/config hashes, exact
commands/environment, measured resources and output checksums belong to the
external W-164 implementation manifest under
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-164/20260921T210202Z-781f6135/`.
These are development software fixtures, not calibrated evaluation data or
MATLAB parity evidence. Existing H5AD/assignment links are retained, not scientifically
qualified. Owner Jiahao; retain through thesis/publication; backup and public
reproducibility remain unverified.

### Saved summary development v1

`saved-summary-v1` in `docs/examples/run_summaries.py` reuses saved-formed-v3,
without changing its model or existing bytes. A deliberate TPS failure reloads
the Z=9 prepared checkpoint: two objects cannot provide sufficient landmarks.
The true run is failed with partial extraction and unavailable decoded/final
metrics. Successful Z=9/Z=1 runs retain decoded/final artifacts from the current
API. Seed 42, three rounds, four channels, `(9,32,32,4)` / `(1,32,32,4)` float32
and unknown calibration remain unchanged. No historical TIFFs are regenerated.
`inputs.json` pins saved inputs before rendering; external W-165 manifests pin
source/config/output hashes and commands. These are software fixtures, not
qualified evaluation data. Owner Jiahao; retain through thesis/publication;
backup unverified.

### Controlled development package v1

`controlled-development-v1-{size}-{condition}` is defined by
`synthetic.development_scene_preset` and the [preset guide](development-presets.md).
It packages formed synthetic/1 generator 4, frozen specification
`f9512694a0960c10ce5236efbaaf9d6f425c1d8a`, seed 42, scene key
`controlled-development-v1`, development split and unknown physical calibration.
Sizes are ZYX `(1,32,32)`, `(9,32,32)`, `(9,48,48)`, three rounds/four channels,
float32 images and float64 NCR truth. The guide pins explicit gt-A/gt-B positions,
codebook/order/mapping and every parameter for clean, 19 individual-factor and
one combined condition. Namespaces vary; unrelated latents within a size do not.
Structured backgrounds share the molecule map; Z=1 is sampling, not projection.
No historical inputs/TIFFs are consumed or replaced, and no empirical calibration,
cell truth or evaluation qualification is implied. Saved HDF5/Parquet/NPZ/JSON
and inspection HTML stay external. The W-166 run manifest records source/config,
outputs, exact validation and measured costs under
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-166/20260921T220125Z-c16d44c7/`.
Owner Jiahao; retain through thesis/publication; backup/public reproducibility
unverified. W-167 owns independent qualification; W-93/W-57 remain open.

### Synthetic qualification edge probe v1

`test_synthetic_qualification.py` retains four explicit formed objects in float64
ZYXC `(3,7,9,4)` and `(1,7,9,4)`: edge at `(Z//2,2,-.5)`, far at
`(Z//2,2,-20)`, overlap-A/B both at `(Z//2,4,6)`. All use gene-A=123,
A=8, unit widths/elongation, angle 0, seed 42 and scene `formed-v1` in the
development split. Middle-round dropout is certain; persistent loss starts at
round index 2. Four formed rows and twelve histories survive, including the
invisible object. Empty variants preserve typed tables and `(0,4,3)` arrays.
No historical TIFFs, calibrated inputs or evaluation scenes are used.

The [qualification example](synthetic-qualification.md) independently audits
all 63 saved controlled-development-v1 presets and checks exact fresh-process
reproduction; it does not replace or regenerate their saved bytes. New combined
float64 checks reuse the small/Z=1 preset constants, with an independent bisection
inverse and keyed noise oracle. Source/input/config hashes, commands, resources
and qualification limits are pinned in the external W-167 delivery manifest.
Owner Jiahao; retain through thesis/publication; backup/public reproducibility
unverified. Development correctness does not qualify D04 or close W-93/W-57.

### Sample export literal specification v1

`sample-export-contract-v1` defines E1–E9 in the
[sample export contract](sample-export-contract.md). These are literal software
oracles, not segmented biological truth: ZYX shapes from `(1,2,2)` to `(2,2,4)`,
uint32 labels, two float32 channels with `I=100*z+10*y+x` and `I+1000`, no RNG.
A later multichunk regional-read probe may use `(9,48,48)` with the same formula.
Explicit literal physical/index geometry, IDs, populations, expected reduced
values, native fallback and hard failures are specified before implementation.
W-168 saves the JSON specification and hashes externally; W-169/W-170 must save
component/config hashes before reuse and cannot claim those computations ran
merely because expected values exist. No historical TIFF bytes are regenerated.
Owner Jiahao; retain through thesis/publication; backup/public reproducibility
unverified. Scientific population/overlap decisions remain W-152/W-57 scope.

The W-169 raster checks materialize the existing E1–E9 specification, preserving
its version and independent expectations. `test/test_raster.py` additionally
uses a deterministic `(1,2,8)` one-cell sparse tail to reject centroid/Hausdorff
distortion even when IDs survive, integer/float dtype variants, and invalid-grid
and resource-budget variants. These are no-RNG software edge cases, not new
biological truth. External W-169 manifests pin native/prepared NPY bytes,
configuration, source metadata, remap rows, trial metrics and specification hashes.
No existing TIFF is read or regenerated by these focused checks.
