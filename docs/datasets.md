# Dataset catalog

This is the human-readable inventory of STARfinder development and historical
benchmark inputs. Register IDs D01–D08, experiment readiness and scientific
qualification are owned by the Chapter II benchmark register, which is maintained
outside this repository. Availability does **not** qualify an input for an
experiment. In particular, no entry here qualifies D04.

## Identity, versions and verification

Historical names are kept as version keys, not assigned competing benchmark IDs.
D01/D02/D03/D06 keep their register IDs; `aging`, named synthetic presets and
registration derivatives identify catalog versions, not newly selected benchmark
datasets.

Last bounded verification: **2026-09-21**, host GP099-29C. The evidence (sampled
paths, metadata and configuration hashes, fixture checksums) is kept with the run
records outside the repository. Host paths below are private locations, not public
reproducibility or backup guarantees. Header reads do not load image pixels.
Headers were sampled, not exhaustively checked across rounds/FOVs/channels.
Real-image bytes were not hashed, processed, downloaded or scientifically evaluated.

States used here:

- **Historical/unverified:** recovered metadata or generator configuration only.
- **Available:** a local path and representative headers/metadata were read;
  completeness, lineage and fitness remain unverified.
- **Development-ready:** a pinned input passes a specified software check;
  this does not establish molecular or biological truth.
- **Missing:** absent at the particular inspected location, not globally absent.
- **Calibration/evaluation-qualified** and **retired** require an explicit owner
  decision and versioned evidence; this catalog makes no such promotion.

Unless stated otherwise, physical spacing, units, acquisition date, original
archive-member identity, masks, independent biological units and evaluation split
are **unknown**. TIFF resolution tags with `ResolutionUnit=1` do not establish
physical calibration. Shapes below are explicitly **ZYX**, per channel; combined
Python rounds are **ZYXC**. “2D” in a dataset name does not mean Z=1.

## Real sequencing inputs

Local root `R = /home/unix/jiahao/wanglab/Data/Processed/sample-dataset`.
Historical common accession: **10.5281/zenodo.11176779**. Its association with each
local file/version has not been verified; do not treat that citation as a checked
archive-to-file lineage. These are real STARmap-related sequencing inputs; the
precise assay/specimen and upstream processing provenance remain unverified.
Filenames include `cmle`, but the exact restoration recipe/version is unknown.
They are inputs to STARfinder, not established raw acquisition data.

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
historical aliases, not the actual host root above. Source, processing stage,
assay identity, geometry and masks must be reconciled before qualification.

## Register entries without a selected local input

| ID | Identity / lineage | State and limits |
| --- | --- | --- |
| D04 | Condition-specific processed-image synthetic molecular foundation | The historical and current generators below are related development inputs, **not frozen D04**. Actual rendered truth, independent streams/splits, historical provenance and calibration remain open |
| D05 | Leica SP8 raw/Huygens pairs | Historical/unverified; exact accession/local input, FOV/round/channel/codebook, shape/dtype/spacing and paired lineage unknown. Huygens is a comparator, not truth |
| D07 | Derived from D01 tile_1, all four rounds: Z `[0:30]`, Y/X `[1280:1792]` | Selected recipe, **not materialized or verified**; expected ZYX `(30,512,512)` from source indexing, dtype preserved by recipe. Origin/spacing, source hashes, signal suitability and export path remain unverified; not an independent biological sample |
| D08 | Unselected pool of additional real/public inputs | No selection, accession, local path or dimensional metadata assigned; qualify a specific question and input before inclusion |

## Current synthetic presets

The current generator is the formed-scene model in `starfinder.synthetic`
(see the [synthetic API](api/synthetic.rst) and the
[model definition](synthetic-specification.md)). Its benchmark presets are
generated on demand with `starfinder synthetic generate`; no generated images
are stored in the repository. The test suite generates `small` (e2e, seed 42)
once per session under pytest's temporary directory (`src/python/test/conftest.py`).

| Preset (`benchmark-presets-v1`) | ZYX / dtype | Amplicons per FOV, FOVs | Seed |
| --- | --- | --- | --- |
| `tiny` | `(8,128,128)`, uint16 | 10, 2 | 42 |
| `small` | `(16,256,256)`, uint16 | 50, 2 | 42 |
| `medium` | `(32,512,512)`, uint16 | 400, 2 | 42 |
| `large` | `(30,1024,1024)`, uint16 | 1,500, 2 | 123 |
| `tissue` | `(30,3072,3072)`, uint16 | 14,000, 2 | 456 |
| `thick_medium` | `(100,1024,1024)`, uint16 | 5,200, 2 | 789 |

These are uncalibrated development inputs: appearance defaults are chosen to be
plausible, not fitted to real data, and they carry no molecular truth claim.
Their images differ from every historical synthetic version below, even where the
shape, count and seed match. No register ID is assigned to them; whether D06 moves
to `benchmark-presets-v1/small` is a register decision.

## Historical synthetic sequencing inputs

These are synthetic Gaussian **processed-image** scenes from the historical
generator (`SyntheticConfig`, removed from the package), not simulated original
RNA populations, raw microscopy, cell masks or calibrated tissue. Local benchmark
root `B = /home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark`. All listed
stored sequencing versions declare four rounds/four channels, two FOVs
(`FOV_001`, `FOV_002`) and seed 42 unless marked unknown. Physical spacing and
units are unknown; use voxel-index geometry, not invented micrometres.

| Stable catalog version / register | Location and state | ZYX / dtype | Identity/configuration and limits |
| --- | --- | --- | --- |
| `sequencing/tiny` | Historical/unverified stored dataset (`B/e2e/data/tiny` missing) | `(8,128,128)`, uint8 | 8 genes, 10 spots/FOV |
| D06 / `sequencing/small/truth-v2.0` | Historical test fixture, formerly `tests/fixtures/synthetic/small` (TIFFs were never tracked); no longer in the repository | All 32 TIFFs `(16,256,256)`, uint8 | Metadata version 2.0, seed 42, 8 genes, 50 spots/FOV; used by end-to-end regression checks until the current generator replaced it |
| `sequencing/medium/truth-v1.0` | Historical; metadata formerly under `tests/fixtures/synthetic/medium`, TIFFs in the original checkout only | Sampled `(32,512,512)`, uint8 | Metadata says preset `custom`, seed 42, 8 genes; historical 100 spots/FOV |
| `sequencing/large/truth-v1.0` | `B/e2e/data/large`; available | Sampled `(30,1024,1024)`, uint8 | Metadata `custom`, seed 42, 64 genes; historical 2,000 spots/FOV differs from the historical preset's 1,500; generator revision/effective configuration unverified |
| `sequencing/tissue/historical` | `B/e2e/data/tissue`; available headers | Sampled `(30,3072,3072)`, uint8 | Historical 64 genes, 14,000 spots/FOV, two FOVs; truth version/seed/full effective config not read in bounded verification |
| `sequencing/thick_medium/truth-v1.0` | `B/e2e/data/thick_medium`; available | Sampled `(100,1024,1024)`, uint8 | Metadata `custom`, seed 42, 64 genes; historical 5,200 spots/FOV; historical generator revision unverified |

`B/e2e/data/small` and `medium` are missing at those exact paths. The historical
medium count of 100 spots/FOV also differs from the historical preset's 400; never
substitute a newly generated preset for existing stored bytes. For
large/tissue/thick_medium, `codebook.csv` and full file-set completeness remain
unverified. Additional `B/e2e_LR/data/large` and `thick_medium` truth metadata
declare v1.0 and seeds 123/789 respectively, with shapes matching their names.
Treat these as distinct historical versions, not replacements for the seed-42
inputs; image headers, lineage, codebooks and effective deformation recipes were
not checked there.

The historical generator recorded `molecular_truth=None`, used integer centers and
rounded deformations, left `background_std` unused, and derived registration seeds
from a process-dependent `hash()`. These stored versions keep those limitations;
the current generator does not reproduce them, and a passing software check does
not qualify them.

## Historical registration derivatives

| Stable version / parent | Path under B; state | Sampled ZYX / dtype | Lineage and limitations |
| --- | --- | --- | --- |
| `D01/registration/round1-round2` | `registration/data/real/tissue_2D`; available | `(30,3072,3072)`, uint8 | Metadata: tile_1, round1 reference, round2 moving, 4 source channels, unknown true shift |
| `D02/registration/round1-round2` | `registration/data/real/cell_culture_3D`; available | `(30,1496,1496)`, uint8 | Metadata: Position351, round1/round2, 4 source channels, unknown true shift |
| `D03/registration/round1-round2` | `registration/data/real/LN`; available | `(50,1496,1496)`, uint8 | Metadata: Position001, round1/round2, 4 source channels, unknown true shift |

Each has single-channel `ref.tif`/`mov.tif`. Historical notes called them “MIP
extractions”, but both inspected headers retain Z. The projection/merge axis and
exact source-channel composition, extraction script revision, source hashes and
calibration are unverified. Do not label these Z-projected images. No codebook is
attached to these registration pairs, and they provide no molecular truth.

Synthetic registration versions from the historical generator live at
`B/registration/data/synthetic/<preset>`. All six directories exist; two
moving-image headers per directory were read:

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
reusing their accuracy claims requires provenance and protocol review.

## Adding a version

A new stored input or fixture adds a version here with its source/parent or
generator revision, effective configuration and seeds, identities, axes, shape and
dtype, spacing/units (or unknown), intended use, limitations and verification
date. Keep older versions and their identities. Per-run logs, images and
measurements stay outside Git; this catalog is not a run diary.
