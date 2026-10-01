# Spot-finding baseline

Status: Accepted (W-268, 2026-10-01, at b5fcb7f); amended at the W-276 review, 2026-10-01

This page records how spot finding behaves at revision `42f652d` (branch
`runner/s27-spec-20260930`, on `dev` after the §2.6 work), before the Chapter II
§2.7 work changes it. It is the reference that the golden test
`src/python/test/test_spot_finding_golden.py` pins. The proposed replacement is
described in {doc}`spot-finding-contract` and {doc}`spot-finding-algorithms`;
neither is accepted yet. Paths are relative to `src/python/starfinder/` unless
they start with `src/matlab/`, `workflow/` or `docs/`, and line numbers are at
`42f652d`.

Note: this page keeps the field name of `42f652d`. The W-276 review (2026-10-01)
renamed `PipelineConfig.detection` to `PipelineConfig.spot_finding`, and
`run.json` now records `config.pipeline.spot_finding`; see {doc}`migration`.

## Local maxima

`find_spots(image, *, config, metadata, spot_namespace)` in
`starfinder.spot_finding` ({doc}`api/spot_finding`) detects spots in one ZYX or
ZYXC image. `LocalMaximaConfig` is the pipeline detector; two registration
landmark detectors share the module (below).

| Field | Behavior at `42f652d` |
| --- | --- |
| Inputs | A finite real ZYX or ZYXC array (`_validate_image`), an `ImageMetadata` and a nonempty namespace string (`spot_finding/__init__.py:161-185`). The input is never modified; thresholds are computed in float64. Each channel is detected on its own; there is no deduplication across channels. A channel whose maximum is 0 is skipped, but its threshold is still recorded (`:214-215`). |
| Threshold modes and units | `threshold_mode` and `threshold_value` (`:41-67`, `:203-213`). `noise` (default, `threshold_value=5.0`): `median + value × 1.4826 × MAD` over every voxel of the channel, so `value` is in robust standard deviations; it is the only mode without an upper bound on `value`. `adaptive`: `value × channel maximum`; `adaptive_round`: `value × maximum of the whole image` (all channels of the round passed in); `global`: `value × dtype maximum`, for uint8 and uint16 only (`ValueError` otherwise). The last three take a fraction in [0, 1]. A maximum is kept only when it is strictly above the threshold (`skimage.feature.peak_local_max`, `image > threshold`). |
| Detection | `peak_local_max(channel, min_distance=min_distance_voxels, threshold_abs=threshold, exclude_border=...)` (scikit-image 0.26.0, `:152-158`). A voxel is a maximum when it equals the maximum of the cube of side `2 × min_distance_voxels + 1` around it (`maximum_filter`, mode `nearest`). Every voxel of a plateau that equals its neighbourhood maximum is returned, so a flat top gives several tied maxima. With `min_distance_voxels > 1`, maxima closer than that (Chebyshev distance) to a brighter one are removed; with 1 nothing is removed. A constant channel gives no maxima. |
| Border exclusion | `exclude_border=True` (default) removes maxima within `min_distance_voxels` of every face, Z included for Z>1; `False` keeps them. A volume with 1 < Z ≤ 2 × `min_distance_voxels` therefore always gives an empty table with the default. |
| Separation | `min_distance_voxels` (positive integer, default 1) sets the neighbourhood, the spacing rule and the border width together. |
| Z=1 | A singleton Z is detected as a YX plane (only the YX border is excluded) and returned with z=0 (`:152-158`). |
| Output columns | `spot_id` (pandas string), `z`, `y`, `x` (float64, integer-valued voxel indices), `channel` (int64, index into `diagnostics["channel_labels"]`), `peak_intensity` (float64, the original pixel value at the maximum; dropped with `measure_peak_intensity=False`). No score, size or sub-voxel coordinate. Rows are ordered by channel, then by decreasing intensity within a channel. |
| Identities and namespace | `spot_id` is `"0"`…`"n-1"` in table order (`:229`); identities are stable within a namespace through reordering and subsetting, but not across settings or images. Consumers join on `(spot_namespace, spot_id)` (`SpotFindingResult`, `:104-149`). |
| Diagnostics | `method`, `channel_labels`, `thresholds` (one per channel), `coordinate_units` (`voxel_index`), `singleton_z_policy` and `measurements` (`:230-234`). There are no per-channel counts, no median, MAD or zero fraction, no effective settings beyond the config, no software identity and no warnings. An empty result is a typed empty table; it cannot be told apart from a channel that was skipped. |
| `FOV.find_spots` | Accepts only `LocalMaximaConfig` (`TypeError("FOV detection requires LocalMaximaConfig")`, `dataset/fov.py:775-776`). Detects the reference round only (`self.images[self.rounds.reference_round]`), fills `channel_labels` from `Dataset.channel_order`, and uses the namespace `json.dumps([dataset_id, sample_id, fov_id, subtile_id])` (`dataset/fov.py:766-784`). `FOV.run` calls it once, after the reference round's preprocessing (`dataset/fov.py:982-983`); every sequencing round is then extracted at those reference coordinates (`dataset/fov.py:984-985`). |
| `PipelineConfig.detection` | `LocalMaximaConfig` or `None`; any other type raises `TypeError("detection requires its typed operation config")` (`dataset/config.py:207`, `:213-222`). |
| YAML keys and translation | The `spot_finding` block of a Python rule: `run`, `ref_round` (must equal the dataset reference round, else `ValueError`), `intensity_estimation` → `threshold_mode` (default `noise`), `intensity_threshold` → `threshold_value` (default 5.0), `min_distance` or `min_distance_voxels` → `min_distance_voxels` (default 1; when both are given `min_distance_voxels` wins without notice) (`dataset/workflow.py:279`, `:308-309`, `:319`). `exclude_border` and `measure_peak_intensity` are not accepted (unknown keys raise), so YAML runs always exclude the border. The schema (`workflow/schemas/config.schema.yaml:814-827`) declares `run`, `ref_round`, `intensity_threshold` and `intensity_estimation` with the enum `local`, `global`, `noise`, `adaptive`, `adaptive_round`; `local` is accepted by the schema and rejected by Python (`unknown threshold_mode`) and by MATLAB (no such case). |
| `candidates` checkpoint | `write_candidates` (`io/_checkpoint.py:456-469`) writes `candidates.<csv|parquet>`, the wide table of `candidates_frame` (`:196-220`: `spot_namespace`, `spot_id`, the spot columns, then `sig_<round>_<channel>` and `valid_<round>` when extracted), and `candidates.json` with `format_version` 2 (`FORMAT_VERSION`, `:20`; readers accept 1 and 2, `:22`), the dtype map, `spot_namespace`, `spot_columns`, `metadata`, `detection_config`, the JSON-representable `detection_diagnostics` and the extraction header. `_read_candidates` (`:472-492`) rebuilds the config through `_detectors()` (`:450-453`), a hard-coded map from the saved `method` (`local_maxima`, `noise_landmark`, `percentile_centroid`) to the three config types, and requires one namespace for every row. See {doc}`checkpoints`. |
| Provenance | `run.json` stores the pipeline config and the spot count (`dataset/_run_record.py`). The registry's `provenance()` entry exists (`_registry.py:112-122`) with an `artifacts` list reserved for §2.7 weights, but nothing calls it for detection. |
| MATLAB counterpart | `STARMapDataset.SpotFinding` (`src/matlab/STARMapDataset.m:708-758`) selects the channels whose name contains `seq` in `ref_layer` (default the reference round) and calls `SpotFindingMax3D` (`src/matlab/SpotFindingMax3D.m`): `imregionalmax` (connected regional maxima, 26-connectivity in 3D) above `intensity_threshold × channel maximum` (`adaptive`, MATLAB default with 0.2) or `× 255 / 65535` (`global`), strictly above; `regionprops3` centroids cast to `int16`, 1-based `x, y, z`, with `MaxIntensity` and `Channel`. A plateau is one regional maximum and so one spot at its rounded centroid; there is no `noise` mode, no border exclusion and no minimum distance. |

## Registration landmark detectors in the same module

| Config | Use | Behavior |
| --- | --- | --- |
| `NoiseLandmarkConfig(noise_sigma=5, min_distance_voxels=1)` | Fixed internal use by TPS and CPD (`registration/_landmarks.py:42-50`, `registration/_cpd.py:486-490`), with `noise_sigma` from the registration's `detection_noise_sigma` | Noise-mode maxima per channel with the border always excluded; for ZYXC input, maxima of different channels within `min_distance_voxels` (Euclidean) are deduplicated by keeping the lower concatenated index. Columns `spot_id, z, y, x`. |
| `PercentileCentroidConfig(threshold_percentile=99.5)` | The legacy benchmark evaluation (`benchmark/_legacy_evaluation.py:19`) | Channels summed, voxels above the percentile labelled with face connectivity (`scipy.ndimage.label`), intensity-weighted centroids (fractional). Columns `spot_id, z, y, x`. |

Neither is accepted by `FOV.find_spots` or `PipelineConfig.detection`; both are
saved and reloaded through `_detectors()`.

## Spot metrics in `starfinder.evaluation`

`evaluate_spots(detected, truth, **matching)` (`evaluation/spot_finding.py`)
calls `match_points(truth, detected, policy, threshold, units, ...)`
(`evaluation/matching.py`): `greedy` or `nearest_candidate` one-to-one matching
with an inclusive or exclusive threshold, optional eligibility masks, and recall,
precision and mean matched distance. It reports no per-axis localization error, no
percentile of the matched distance and no duplicate classification; W-266 computed
per-axis errors and the W-218 classification below from the matched pairs in its
own scripts.

## W-218 re-measurement on current code

W-218 reported 68 candidates on one FOV of the `small` benchmark preset in the
W-217 notebook: 45 matched, 22 duplicates, 1 spurious and 5 missed amplicons, and
attributed the duplicates to tied or split maxima of one multi-voxel amplicon.
This issue re-measured it on the current code.

* **Commit and environment.** `42f652d`, source unchanged; the locked project
  environment (Python 3.12.12, NumPy 2.2.6, SciPy 1.17.0, scikit-image 0.26.0,
  pandas 3.0.0), one thread.
* **Fixture.** `formed_scene_preset("small")`: one FOV, 16×256×256 voxels, four
  rounds, four channels, 50 amplicons, seed 42, uint16, generated in memory
  (round 1 SHA-256 `5f2ec890…`). No volume was saved.
* **Pipeline.** As in the foundation tour notebook: `FOV.run` with a translation
  `RegistrationRecipe`, `LocalMaximaConfig(threshold_mode="noise",
  threshold_value=5.0, min_distance_voxels=1)` (border excluded),
  `NeighborhoodSumConfig((1, 2, 2))` and `WtaDecoderConfig()`.
* **Matching policy.** `evaluate_spots` with policy `greedy`, threshold 5.0
  voxels, boundary `exclusive`, units `voxel` (index space), truth = the formed
  amplicon centres with `center_in_bounds` in round 1 (all 50 eligible). A
  candidate is *matched* when it is in a greedy pair; *duplicate* when it is
  unmatched and lies within 5 voxels of a matched amplicon with the same decoded
  color sequence; *spurious* otherwise (the notebook's classification).
* **Script and record.** `w218/w218_remeasure.py` and `w218/w218-remeasure.json`
  in the run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-267/20261001T014815Z-af149185`.
  The four runs took 12.8 s and 429 MB peak RSS in all (`/usr/bin/time -v`).

| Run (one change each) | Candidates | Matched | Duplicate (same channel / other channel) | Spurious | Missed | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Notebook settings | 68 | 45 | 22 (0 / 22) | 1 | 5 | 0.662 | 0.90 |
| `exclude_border=False` | 73 | 50 | 22 (0 / 22) | 1 | 0 | 0.685 | 1.00 |
| Generator channel mixing off | 46 | 45 | 0 | 1 | 5 | 0.978 | 0.90 |
| Legacy recipe-1 uint8 min–max normalization first | 158 | 45 | 30 (2 / 28) | 83 | 5 | 0.285 | 0.90 |

Findings:

* The W-218 counts reproduce exactly on current code.
* **Every duplicate is a cross-channel candidate.** Each of the 22 lies in the
  channel next to its amplicon's matched candidate (at most 1.0 voxel away in Y
  and X and 2.0 in Z), and 19 of them have peak values of 196–256 against
  1152–2347 for the amplicon in its own channel. In the other 3 the greedy match
  went to the dim copy and the bright one counts as the duplicate. The §2.12
  generator mixes 5 % of each channel into the next (`crosstalk=0.05`). With the
  mixing switched off (`ReadoutEffectsConfig.mixing_enabled=False`; disabled
  effects consume no random draws, so nothing else changes), all 22 disappear.
  The duplicates are crosstalk copies that local maxima detects correctly in
  the next channel. They are not tied or split maxima of one channel. Removing
  them needs a decision across channels.
* **Same-channel tied maxima** appear after the legacy uint8 normalization:
  2 duplicates, each an exact tie at a neighbouring voxel (offsets (1, 0, 0) and
  (0, 1, −1)), because truncation to 8 bits flattens bright tops. The uint8
  run also gives 83 spurious noise-mode maxima at grey levels 15–16 just above
  thresholds of 11.4–23.8.
* **The 5 missed amplicons** lie within 0.6 voxels of a face (z≈0, z≈15,
  y≈0.2). They are removed by `exclude_border=True` and all are found with
  `False`.
* **Within-channel merge check (post hoc).** A within-channel non-maximum
  suppression with radius (2, 2, 2) voxels (`w218/w218_nms_feasibility.py`),
  applied to the classified candidates without re-decoding, removes 3
  candidates in the notebook run and 8 in the uint8 run. All of them are
  duplicates: in the uint8 run, both same-channel ties and 6 copies that fall in
  the same other channel. No matched candidate is removed.

The proposed W-218 option and its validation are in {doc}`spot-finding-algorithms`
("Local maxima" and check S16).

## Discrepancies with the agreed §2.7 scope

| # | Discrepancy | Proposed resolution |
| --- | --- | --- |
| 1 | Method dispatch by `isinstance` and hard-coded method sets: `SpotFindingResult.__post_init__` (`spot_finding/__init__.py:126`), `find_spots` (`:176`, dispatch `:186-228`), `_detectors()` (`io/_checkpoint.py:450-453`), `PipelineConfig.detection` (`dataset/config.py:207`, `:214`), `FOV.find_spots` (`dataset/fov.py:775`) and the YAML adapter (`dataset/workflow.py:319`). `isinstance` also accepts subclasses. | `SPOT_FINDING_METHODS` (registry move 4, {doc}`method-registry`): exact-type lookup, one private run function per method, every place derived from the registry ({doc}`spot-finding-contract`, "Registry entries"). |
| 2 | Only local maxima is accepted by `FOV.find_spots` and `PipelineConfig.detection`. | Every method with `pipeline=True` is accepted: `local_maxima`, `starfish_log`, `spotiflow`, `piscis`. The landmark detectors stay `pipeline=False`. |
| 3 | Detection runs on the reference round only, while direct readout needs candidates in each relevant round. | A named set of detection rounds, default the reference round only (unchanged), with an explicit `round` column and unchanged barcode-mode identities; proposed to §2.8 ({doc}`spot-finding-contract`, "Detection in several rounds"). |
| 4 | W-218 duplicate maxima. Re-measured above: the 22 duplicates on the `small` scene are cross-channel crosstalk copies; same-channel ties appear after uint8 quantization; the 5 misses are border exclusion. | An explicit, opt-in within-channel merge `merge_radius_zyx` in `LocalMaximaConfig` (off by default, so the golden test is unchanged) resolves tied and split maxima; `exclude_border` becomes reachable from YAML for the border misses. The cross-channel copies are left to §2.8, because this issue excludes cross-channel deduplication; that split is an open choice for Jiahao ({doc}`spot-finding-algorithms`, "Local maxima"). |
| 5 | `exclude_border` (and `measure_peak_intensity`) cannot be set from YAML. | The `spot_finding` block accepts every init field of the selected method's config; the legacy aliases stay for `local_maxima` ({doc}`spot-finding-contract`, "Workflow configuration"). |
| 6 | A noise-mode MAD of 0 (more than half the voxels equal to the median, for example zeros after background removal) makes the threshold equal to the median and passes silently (golden test: channel 2, 861 maxima in 3D). | Record the zero fraction, median, MAD and threshold per channel and round, and warn when MAD is 0 or more than half the voxels are zero; the threshold itself does not change ({doc}`preprocessing-algorithms`, "Interaction with noise-mode detection"). |
| 7 | There is no spot-finding dependency error; spot finding has no optional dependency today. | `SpotFindingBackendUnavailableError(ImportError)`, raised through the registry's `require()` naming the module and the extra. |
| 8 | No weights provenance: the `artifacts` list of the provenance entry is empty and nothing records model files. | A known-weights table with SHA-256, an explicit fetch command, hash-checked loading from explicit paths, and `artifacts` entries ({doc}`spot-finding-contract`, "Pretrained weights"). |
| 9 | `min_distance` and `min_distance_voxels` together are accepted, and `min_distance` is ignored without notice. | Reject the pair with `ValueError`, as a duplicate setting. |
| 10 | The schema's `intensity_estimation` enum contains `local`, which neither Python nor MATLAB accepts; the schema has no `method` and no `min_distance` keys. | Drop `local` from the enum; declare `method` (kept equal to the pipeline methods by a default-tier test) and the Python-only keys. |
| 11 | Python returns every voxel of a plateau, while MATLAB `imregionalmax` returns one spot per plateau at its rounded centroid. | Recorded. The opt-in merge of item 4 removes plateau ties; MATLAB parity is not a goal of §2.7. |
| 12 | The noise threshold uses the whole channel, so at high spot density it rises above the spot peaks (W-266: empty result on the 1×64×64 isolated-spot plane at 0.024 spots per voxel). | Flagged (W-266 choice 8, accepted); no change in §2.7. The diagnostics of item 6 make it visible, and validation fixtures for local maxima on Z=1 use a stated lower density. |
| 13 | Diagnostics have no per-channel counts, no effective settings, no software or model identity and do not distinguish an empty success from a skipped channel. | The diagnostics of {doc}`spot-finding-contract` ("Diagnostics"). |
| 14 | There is no device or thread record for detection. | The cross-stage `ExecutionConfig.device` (only `"cpu"` in §2.7) and an execution record with the framework build and thread settings. |
| 15 | Coordinates are integer-valued; nothing reports a native score or size. | Unchanged for local maxima; the new methods add fractional coordinates and native scores or sizes as declared `output_columns`. |
| 16 | Evaluation has no per-axis localization error and no duplicate classification. | Add two metrics to `starfinder.evaluation.spot_finding` for the validation design ({doc}`spot-finding-algorithms`). |

## Golden test

`src/python/test/test_spot_finding_golden.py` pins the current local-maxima
behavior on its own seeded fixture of 12×48×48 voxels and four uint16 channels
(seed 20261001), and on its plane z=6 as the Z=1 variant:

* channel 0 holds a saturated amplicon clipped to 3000, whose plateau gives tied
  maxima, a spot on the z=0 face, and a dim punctum between the noise thresholds
  for `threshold_value` 4 and 5;
* channel 1 holds a two-lobe amplicon with lobes 2.4 voxels apart, which gives
  split maxima with `min_distance_voxels=1` and one maximum with 2, and a spot on
  the y=0 face;
* channel 2 is zero in about two thirds of its voxels, so its noise-mode MAD and
  threshold are 0;
* channel 3 is dim (its maximum is below the image maximum, so `adaptive` and
  `adaptive_round` differ) and has spots on the x=47 face and at x=1.

It pins with exact SHA-256 digests:

* (a) the `find_spots` table and threshold diagnostics for the four threshold
  modes (`noise` 5.0, `adaptive` 0.2, `adaptive_round` 0.2, `global` 0.01), with
  `exclude_border` true and false, in 3D and Z=1 (16 cases, 12 to 1342 rows);
* (b) the `candidates` table after `FOV.run`, written as a CSV checkpoint and
  reloaded: the `candidates.csv` digest, and the reloaded table, which equals the
  table of (a);
* (c) the same digests when the pipeline comes from the legacy YAML keys through
  `from_workflow_config`, with `min_distance` and with `min_distance_voxels`, for
  the eight cases the legacy keys can express (`exclude_border` true).

Three tests show that changing `threshold_value` (5 → 4), `min_distance_voxels`
(1 → 2) or `exclude_border` (true → false) changes the noise-mode table digest,
in 3D and Z=1. Two tests document legacy behavior that §2.7 changes or keeps:
the MAD-0 channel's threshold is 0 and no warning is emitted, and the saturated
plateau gives more than one candidate. Every detection configuration is built by
one helper, `detection_config`, which holds the only imports of detection config
types and the only `PipelineConfig(detection=...)`. Learned methods and LoG are
not in the golden test.

Three single-thread runs (`OMP_NUM_THREADS` and the other thread variables 1,
`taskset -c 0`) each passed all 59 tests. In three further processes, every pinned
value was recomputed with byte-identical output (`golden/compute_pins.py`,
`golden/runs/` in the run directory), so the test uses exact equality.
