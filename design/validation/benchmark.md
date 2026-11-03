# Benchmark validation design

**Status: Proposed — W-771; W-775 reviews this design, not scientific results.**

Base: `7cd1e7d712210b684f488a525d214667ffd766b0`. The
[contract](../specs/benchmark.md) states present and proposed APIs; the
[baseline](../baselines/benchmark.md) maps every retiring test. This record
covers all **38 distinct short-set IDs**, then accompanying M12/M13/M73/M74.
Repeated use across experiments shares one metric check, not a new algorithm.
A test named “existing” already pins the stated behavior; a test named “new”
is a proposed implementation requirement, not a reported execution result.

## Precision, populations and bounded fixtures

Follow `docs/contributing.md`, “Expected values and tolerances”:

| Code | Comparison |
| --- | --- |
| D | Exact integer/count/label/status/Boolean, file/record hash or same-environment identity. |
| F64 | `rtol=1e-9, atol=1e-12` for computed float64 values. |
| F32 | `rtol=1e-6, atol=1e-6` for computed float32 values. |
| µm | Absolute declared coordinate/length precision `1e-9 µm`. |
| Existing | Keep the named test's existing bound unchanged; do not widen it to this table. |

Use each result's dtype, including quantiles, not the input image dtype. Integer
sums/coordinates stay exact where integer-valued. Null plus an explicit reason
is a discrete expected result. No empty truth, no valid overlap or no defined
score is converted to a successful zero. Keep excluded-population counts. All
new arrays fit 32×64×64, at most four rounds/channels, generated in-session;
use hand-built tables/labels wherever possible. New tests get subsystem markers
and measured timings at implementation. This design adds no executable tests.

Test paths below are relative to `src/python/test/`. Abbreviations:

* ES = `test_evaluation_short_set.py`.
* EP = `test_evaluation_preprocessing.py`.
* EV = `test_evaluation.py`.
* RQ = `test_registration_qc.py`.
* RM = `test_readout_metrics.py`.

## The 38 selected metrics

| ID | Known-answer fixture and expected result | Existing pin or proposed named check | Tolerance |
| --- | --- | --- | --- |
| M01 | Two truth points at X=0,10; three detections at X=0,20,30, all Z=Y=0; greedy inclusive radius .5: one match, recall 1/2. Empty truth null; masks change denominator. | Existing EP `test_f1_known_answer_and_empty_populations` exercises this matching; EV `test_matching_policies_and_eligibility`, `test_matching_boundary_and_subpixel`, `test_empty_populations` pin matching directly; ES dense-oracle tests preserve candidate/tie parity. | D pairs/counts, F64 recall; oracle same-environment identity D. |
| M02 | Same fixture: precision 1/3; empty detections null for precision. | Same existing matching tests as M01. | D counts, F64 precision. |
| M03 | Same fixture: F1=2/5; nonempty truth/no detections 0; empty or entirely ineligible truth null. | Existing EP `test_f1_known_answer_and_empty_populations`. | F64, D undefined state. |
| M05 | Two truth, four detections scores 12,8,4,4; thresholds 15,10,5,2 give (recall,precision)=(0,null),(.5,1),(.5,.5),(1,.5); area .75. Supplied legacy rows additionally give .625. | Existing EP `test_scored_auprc_known_answer_subsets_and_undefined`, including tied scores, eligibility and empty cases. | F64. |
| M07 | No-background arm, constant truth background 2: estimate 0, RMSE 2, bias -2. Background arm with raw 10, bg_corrected 7, truth 2: RMSE 1, bias 1 in input units. Normalized case follows the fitted interpolation map, not input-unit subtraction. | Existing script `test_preprocessing_synthetic_evaluation.py::test_normalized_truth_follows_the_fitted_value_map` and EP `test_moved_metrics_and_sweep_equal_pre_move_scenes` retain normalized-stage/oracle coverage. **New** EP `test_background_known_answer_and_stage_units` pins these analytic constants. | F64; preserved move oracle D. |
| M08 | One truth at (1,6,6) in (3,13,13,1). Inner ±(1,3,3) has image 7 and truth signal 100; the outer shell has equal numbers of image 1 and 3, truth signal 0. Peak 7, shell median 2, quiet std 1: contrast 5. Make quiet image constant: null contrast and zero-noise fraction 1. | Existing EP move oracle preserves current calculations. **New** EP `test_puncta_contrast_known_shell_and_zero_noise` pins analytic shell/noise; add edge-clipped shell case with explicitly counted voxels. | F64, D missing. |
| M09 | Three isolated truth spots and two rounds/channels yield group medians 2,4,6,8 of on-channel sums: std/mean=sqrt(5)/5. Use isolated impulses with nonoverlapping default extraction boxes, the same sequences as M15 but `color_sequence`; empty truth null. | Existing EP move oracle; **new** EP `test_truth_intensity_spread_known_group_medians` through public `direct_metrics`, not a copy of its helper. | F64; oracle D. |
| M10 | Four truth, three spatial matches, accepted correct/wrong/rejected matched reads plus accepted distant detection: correct 1/4, wrong 1, false 1, accepted 3; rejected match consumes a pair. | Existing ES `test_m10_accepted_reads_and_truth_denominator`; EP `test_sweep_read_classification_parity_with_wrong_false_rejected_and_color_calls` preserves the old classification and color diagnostic. | F64 fraction; D counts/status. |
| M15 | Radius-zero integer sums on (1,1,7,2), two rounds: group medians 2,4,6,8; sqrt(5)/5. All detected spots with observed sequences, including rejected reads. Empty population null. | Existing EP `test_detected_intensity_spread_known_group_medians_and_fov_variation`. | F64. |
| M16 | Per-FOV values [1,3]: CV=.5; singleton [2]: 0; [0,0], empty or [1,null]: null; negative fails. Edge and dense populations evaluated separately, never pooled. | Existing EP `test_detected_intensity_spread_known_group_medians_and_fov_variation`; **new** EP `test_fov_spread_strata_and_fixed_conditions` checks caller stratification. | F64, D errors/missing. |
| M17 | `call_status` rows [assigned,unmatched,assigned,unmatched], accepted flags [true,false,false,true]: unmatched rate 1/2 over all four. Empty table null. | **New** `test_evaluation_readout_summary.py::test_status_counts_and_unmatched_denominator`; source is decoded read table, not accepted-only molecules. | F64; D counts. |
| M18 | Same table: assigned count 2 regardless of acceptance; empty table count 0. | Same new status-count test as M17. | D. |
| M21 | Estimated pull field = truth + (.3,0,0), spacing (2,1,1): median/p95/max .3 voxel, .6 physical; corruption outside mask ignored; empty mask null. | Existing RQ `test_displacement_field_error_statistics`. | Existing absolute 1e-12; keep unchanged. |
| M24 | Two held-out reference landmarks (0,0,0),(0,0,10); before observed (0,0,5),(0,0,15), after exact reference; radius .5 yields before recall 0, after 1. Fit IDs explicitly disjoint from held-out IDs for TPS/CPD. | Existing EV `test_pure_report_no_detection_or_io` and matching tests pin supplied-landmark evaluation; **new** `test_evaluation_registration_protocol.py::test_held_out_landmarks_before_after` pins split provenance. `evaluate_landmark_alignment` is called separately before/after. | F64 rates; D population IDs. |
| M26 | Existing translated ramp on (1,32,32), shift Y=5: common valid domain 27×32, NCC after 1, before computed Pearson on that same domain, gain=after-before. Constant signal/null and no overlap/null. | Existing RQ `test_translation_qc_on_matched_domains`, `test_undefined_qc_values_are_none_with_a_reason`. | Existing after error ≤1e-12 and existing before/gain identities; future standalone analytic fixture F64. |
| M29 | (4,16,16) dense field X=.5 except X=-3 at (2,8,8); one folded voxel, fraction 1/1024 across full grid. | Existing RQ `test_dense_qc_summary_folds_projections_and_diagnostics`. | Existing bound unchanged; count D, computed fraction F64 for future extraction. |
| M34 | Four direct molecules, three tiled: two direct a's share one neighbor at (1,1,1), one b match, one wrong gene. Direct fraction .75, reverse 2/3, relative count difference .25 and ratio .75; `1+.5e-9` remains inside 1+1e-9 Chebyshev threshold. | Existing ES `test_m34_chebyshev_any_neighbour_both_directions`. | F64 fractions, D counts; matching radius is contract 1+1e-9, not a float assertion tolerance. |
| M36 | Existing five detections/two truth: radius 5 exclusive gives matched 2, duplicate 1, spurious 2, same-group duplicate 1; inclusive makes duplicate 2, spurious 1, other-group duplicate 1. | Existing `test_spot_finding_metrics.py::test_classify_detections_rules`. | D counts/classes/boundary. |
| M37 | Two spatially matched pairs, decoded genes [a,null] vs truth [a,b]: gene accuracy .5, confusion includes missing call; no matches null. Unmatched truth affects recall, not conditional gene accuracy. | Existing EV `test_decoding_known_missing_and_zero_denominator`; preserve its confusion-producing API. **New** `test_evaluation_readout_summary.py::test_gene_confusion_and_unmatched_truth` pins the confusion table explicitly. | F64, D confusion counts. |
| M39 | Separated/reversed/tied correct vs incorrect scores give AUROC 1/0/.5; lower-is-better scores correct [1,3,5,7], incorrect [2,6,8] give 2/3 and independent Hanley–McNeil SE formula. Missing class null with reason. | Existing RM `test_auroc_of_separated_reversed_and_tied_scores`, `test_hanley_mcneil_standard_error`, `test_an_empty_class_gives_an_undefined_auroc_with_a_reason`. | Existing 1e-12 for SE/nontrivial AUC; D states. |
| M40 | Scores 1..10 lower better, errors at 3,6,9,10: errors at retention .5/.8/.9/1 are 1/5,2/8,3/9,4/10. Half-credit tie at boundary gives .5/5 in second case. | Existing RM `test_error_at_retention_on_ten_calls`. | Existing assertions unchanged; F64 for new adapters. |
| M41 | Four saved rows with higher-better scores [4,3,2,1], status [assigned,unmatched,assigned,assigned], genes [a,null,b,a], truth correctness [true,false,false,true]. Inclusive cut-offs [5,3,1]: retention [0,.5,1], unmatched [null,.5,.25], retained gene counts [{},{a:1},{a:2,b:1}], truth error [null,.5,.5]. Per-gene retained fractions at cutoff 3: a=.5,b=0. | **New** `test_evaluation_readout_curves.py::test_cutoff_curve_known_rows`. Flip orientation with negated scores, tied threshold inclusion, missing scores counted/excluded with reason, no truth omits error; no optimal cut-off returned. | F64, D counts/status. |
| M54 | Truth two separate 4-voxel labels; predictions one identical 4-voxel label and one disjoint false label: TP=1,FP=1,FN=1,F1=.5 at each IoU [.5,.6,.7,.8,.9]. Boundary fixture: one prediction exactly half of sole truth, IoU=.5, gives F1=1 at .5 and 0 above .5. Both empty null. | **New** `test_evaluation_segmentation.py::test_instance_f1_iou_grid`. Proposed matching extends existing L12 descending-IoU greedy with label-ID tie break and inclusive boundary; no claim of optimal matching. | F64 F1; D counts/matches. |
| M55 | Existing (1,12,24) hand labels: six truth, eight predicted; split truth IDs 2 and 6, split count 2 (disconnected voxels retain one ID). | Existing `test_integration_cellular_chain.py::test_step1_measures_on_hand_built_labels` and helper `match_nuclei:875`; move/reuse bounded pin for new public evaluator, retaining integration test. For exact half-ownership ties the existing helper overwrites owner in sorted traversal; expose this discrepancy for gate review before generalizing. | D labels/counts. |
| M56 | Same existing fixture: predicted ID 14 covers two truth nuclei at ≥half each, merge count 1 and merged-truth count 2. | Same existing hand-label test as M55; metric M56 counts predicted merges, not merged truth. | D. |
| M58 | Nonzero label voxel counts [2,4,6,8] on a small label image: count 4, median 5, linear Q1=3.5,Q3=6.5,IQR=3. Empty labels count 0 and null sizes. | **New** `test_evaluation_segmentation.py::test_label_size_summary`. Count per distinct ID including disconnected pieces; background excluded. | D count; F64 quantiles. |
| M60 | Existing assignment boxes give exact per-nucleus cell/share/background and per-cell n_nuclei in `BOXES_NUCLEI`/`BOXES_CELLS`; the 51/49 split fixture selects cell 3 with share .51. | Existing `test_assignment.py::test_a5_to_a7_nucleus_table_and_cell_rows`, `test_a7_a_nucleus_split_51_49`. Evaluation reads these records; no second assignment algorithm. | Existing pins unchanged; new summary F64 shares, D IDs/counts. |
| M62 | Truth origins A=(0,0,0),B=(0,0,10) µm; estimated A=(5,2,1),B=(5,2,13). Anchor A and subtract its offset: errors [0,2], median 1,max 2 µm. | **New** `test_evaluation_stitching.py::test_gauge_aligned_origin_error`. Missing anchor/grid rejects; missing placements listed, not silently averaged away. | µm. |
| M63 | Kept link: origins b-a=(0,0,12), nominal b-a=(0,0,10), shift=(0,0,1): residual 1 µm. Unkept link null. | Existing `test_stitching_placement.py::test_residual_boundary_and_optimizer_removed_links` pins residuals 1 and 1+TINY with zero placement corrections, and NaN for an unkept link; retain this metric pin. **New** `test_evaluation_stitching.py::test_saved_kept_link_residual` tests only reading saved values/units/null status. | µm, D kept/status. |
| M68 | Common physical overlap: A same-gene points X=[0,10], B X=[.2,10.4], Euclidean inclusive radius .5 µm: both match rates 1, distances [.2,.4], median .3,max .4. Add duplicate at .3: one-to-one remains two matches, reverse rate 2/3. | **New** `test_evaluation_stitching.py::test_overlap_molecules_common_frame`; explicit half-open overlap population bounds, gene mismatch, frame mismatch and no-overlap null cases. | µm distances; F64 rates; D counts. |
| M69 | MATLAB correction (2,-3,1) in row,column,Z converts to detected (-1,-2,3); second round observed-minus-reference (1,2,2) gives L2=3; with first round difference 0, mean L2=1.5, max axis=2. Missing round status missing. | Existing ES `test_m69_matlab_axis_sign_and_one_based_cancellation`; retain W-212 G1 tests too. | F64, D missing count. |
| M70 | MATLAB one-based XYZ rows convert to (2,10,30),(4,20,40),(6,25,50). Four Python rows include two nearby a's, matching b and wrong gene: two greedy matches at radius .5, MATLAB fraction 2/3, Python fraction .5. | Existing ES `test_m70_m71_legacy_boundary_and_one_to_one`; unconverted inputs yield zero matches, both-empty null. | F64, D pairs/counts. |
| M71 | Same three-vs-four fixture: Python/MATLAB count ratio 4/3; zero MATLAB denominator null even if Python nonempty. | Existing ES `test_m70_m71_legacy_boundary_and_one_to_one`; **new** parameter case there for one-empty reference. | F64, D missing. |
| M75 | Saved attempt records: failure OOM, failure method error, success; failure count 2, reasons each 1; one explicit engine retry caused by h_vmem gives memory retry count 1. Final success does not erase failures; unavailable attempt logs mean unknown retry count. | Existing `test_workflow_completion.py::test_an_api_call_that_raises_removes_the_attempts_outputs_and_writes_a_failed_record` pins failed-record behavior. **New** `test_evaluation_resources.py::test_attempts_and_memory_retries` uses synthetic saved logs, no scheduler. | D. |
| M76 | Existing complete task passes; same-size changed bytes fail only with hash check; missing output/stale fingerprint fail. Two saved output inventories with identical declared relative paths/bytes/hash match; change one byte in one output and comparison reports that file. | Existing `test_workflow_completion.py::test_verify_reports_a_complete_task_with_its_records_token`, `test_verify_reports_a_damaged_or_stale_task_as_not_complete_with_a_fresh_rerun_token`; **new** `test_evaluation_workflow.py::test_cross_engine_output_identity` extracted from R8 comparison behavior, plus `test_workflow_completion.py::test_verify_hash_detects_same_size_change` for byte-only corruption. | D hashes/paths/status; cross-host numerical comparison needs an explicitly frozen E05 tolerance per quantity, never automatic relaxation of raw hash equality. |
| M77 | Attributed billing rows for this run USD [1.25,2.50,-.25] including credit give 3.50; another run's 9.00 excluded; unknown attribution or missing bill null, mixed currencies refused. | **New** `test_evaluation_resources.py::test_attributed_billing_total` on synthetic saved billing rows; actual experiment requires owner-supplied billing evidence. | Exact decimal currency arithmetic; F64 only if serialized as computed binary float, D currency/attribution. |
| M78 | Frozen human-step definition: manual Fiji placement edit, manual acceptance, automated export. Count 2 human actions, keep operator/time/source for each; no log gives null, not zero. | **New** `test_evaluation_resources.py::test_manual_step_count` with synthetic action ledger. | D count/classification; experiment classification reviewed by owner. |
| M79 | Held-out measured [10,20,40], predicted [8,22,50]: relative absolute errors [.2,.1,.25], median .2,max .25, underestimates 1. A row measured zero is undefined/excluded with count, not a zero error. | **New** `test_evaluation_resources.py::test_held_out_prediction_errors`: training dataset IDs disjoint from evaluation dataset ID; separate wall/RSS targets and units; no actual fit/sweep in unit test. | F64, D dataset separation/counts. |

## Accompanying measurements

| ID | Known-answer fixture | Pin / proposed check | Tolerance |
| --- | --- | --- | --- |
| M12 | Raw positive uint16 [1,2,3,4], processed [0,2,65535,0]: two positive→zero out of four, .5. Raw zero becoming zero does not count; denominator still all voxels. | EP existing move oracle; **new** EP `test_clipping_and_saturation_known_counts` through `direct_metrics`, with truth empty and explicit images. | F64 fraction, D counts. |
| M13 | Same processed array: one uint16 maximum out of four, .25. Float/signed outputs give zero under existing definition. | Same new clipping/saturation test and existing oracle. | F64. |
| M73 | Mock perf_counter=13, start=10; own CPU user/system=2/1, child=.5/.25: wall_s=3, cpu_s=3.75. | Existing `test_benchmark_trial.py::test_private_trial_subprocess` pins measured nonnegative fields; **new** `test_process_resources.py::test_resource_field_scopes` pins analytic clock/rusage fixture without a timing accuracy assertion. | F64; recorded units/status D. |
| M74 | Mock own maxrss=100, children maxrss=150 KiB: peak_rss_kib=150, not 250. | Same proposed process-resource test. Actual timing/memory require measured invocation logs, not this mocked fixture. | D integer KiB. |

## Evaluation, records and retirement checks

Retain all existing `test_benchmark_trial.py` checks: subprocess execution,
failed method/input record, full product/ordering/hash stability and condition
merge/relocation. Add source-preserving evaluate-only tests with processing
entry points patched to raise: workflow output, trial and each checkpoint stage;
corrupt/missing payload, unknown header and v1/v2 external-inventory limitation;
failed source vs failed metric vs undefined metric. Hash original records before
and after. Reports must not run detection to recreate an overlay; explicitly
saved diagnostic artifacts must be verified. Keep all plotted metrics/units,
counts, failure causes and denominators in the saved evaluation.

The baseline's 74 individual test dispositions are migration requirements, not
permission to delete tests now. R5's useful-metric extraction and R7's synthetic
writer move must precede deleting their providers. Every R1–R12 dependency search
must be repeated at implementation base, including notebooks and documentation.

## Differences needing gate decisions

The short-set copy says “at most two primary and three secondary” but E14 lists
three primary IDs (M70/M01/M02), and E04 lists four secondary IDs. Preserve the
listed 38 metrics here; W-775 should settle whether M01/M02 are one paired endpoint
and whether E04's M60 is accompanying. W-771 does not silently drop a metric.

M55 planning says “at least half” ownership; the existing helper's single-owner
dictionary assigns an exact 50/50 predicted-label tie to the later truth ID.
The hand-built existing test does not resolve that semantic ambiguity. Proposed
public behavior counts each qualifying ownership relation, so one predicted
label may be owned by both at the exact half boundary; add
`test_evaluation_segmentation.py::test_half_ownership_ties` with two truth labels
each owning two half-shared predicted labels, expecting two splits. This change
must be explicitly accepted before extraction; until then keep historical helper
behavior and report the difference. M54 matching extends L12's greedy IoU policy;
it must be approved with the inclusive grid, not described as existing full-grid
coverage.

These are design review questions, not contradictions with W-771's requirement
to write Proposed records. No scientific thresholds are selected by fixture
precision, no real-data threshold sweep is reinstated, and pre-W-93 results
remain pilots.
