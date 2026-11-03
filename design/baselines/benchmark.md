# Benchmark baseline and retirement map

**Status: Proposed — W-771, for W-775 review; nothing is retired by this record.**

Baseline: `7cd1e7d712210b684f488a525d214667ffd766b0`, including W-769
(`528875e`) and W-770 (`7cd1e7d`). Paths and line numbers are pinned to that tree,
not moving references. W-124's copied 2026-11-02 decisions set R1–R12, D5 and D9.
See the [contract](../specs/benchmark.md) and [validation](../validation/benchmark.md).

## Present code and discrepancies from the decisions

| Subject | Current source pin | Decision / proposed resolution |
| --- | --- | --- |
| Old API | `src/python/starfinder/benchmark/__init__.py:1–6`, `_records.py`, `_lifecycle.py`; `src/python/starfinder/__main__.py:22–37,77–94` | Five public lifecycle objects and the benchmark commands still exist. R1 follows W-775 and replacement coverage; W-771 changes no executable behavior. |
| Stack trial | `src/python/starfinder/benchmark/_trial.py:51–80,83–111,124–188` | D5 entry exists for registration/preprocessing only, with one shared grid. E02 detection and E04 segmentation are future adapters, not present capabilities. No new workflow kind is justified. |
| Trial failure/provenance | `_trial.py:162–188` | Method/input errors are recorded, but parse/reservation errors and killed processes may lack records. No copied input archive, code/host identity, truth list, schema validation of condition, or resume. Add explicit evaluation/source provenance rather than claim old lifecycle guarantees survive unchanged. |
| Expansion | `src/python/starfinder/benchmark/_experiment.py:28–66` | Cartesian product and hashes exist; method overrides only, no scene generation, launch or engine scheduling. `name` is optional in code. Same-directory expansion overwrites configs and leaves stale files. A future CLI must state its overwrite policy explicitly. |
| Resources | `src/python/starfinder/_resources.py:6–12`; `workflow/_task.py:245–254` | D5 shares field names and implementation. Wall interval differs from cumulative CPU/RSS scope; RSS is max(self,children), not process-tree sum. Keep scope visible, especially E15. |
| W-769 metrics | `evaluation/matching.py:10,106`; `evaluation/readout.py:259`; `evaluation/registration.py:722` under `src/python/starfinder/` | M10, M34, M70/M71 and M69 are in the package; KD-tree greedy matches dense-oracle behavior. M34 is many-to-one Chebyshev, M70 one-to-one Euclidean. Do not unify them. |
| M69 presentation | `evaluation/registration.py:722–766` | Planning asks per-round shift norms. Code returns signed per-round differences, max-axis and mean-L2 summaries; derive per-round L2 in evaluation presentation without changing sign/axis conventions. |
| W-770 extraction | `src/python/starfinder/evaluation/preprocessing.py:44–202`, `evaluation/spot_finding.py:14–123`; `synthetic/_preprocessing.py:59,133` | D9 metrics/conditions moved. Preprocessing has its own module because matching/registration/spot/readout modules do not own background image metrics. Script imports these and M10; script output/selection behavior remains. D9 extraction alone does not retire R4. |
| Remaining metrics | Contract's ownership table | M17/M18/M60/M63 have source tables/records but need evaluation reducers. M41/M54/M55/M56/M58/M62/M68/M79 have no complete public evaluator here; M75 retry aggregation, M76 cross-engine comparison and M77/M78 evidence readers are pending. Label as prerequisites for relevant experiments. |
| Verify | `src/python/starfinder/workflow/_verify.py:24–60,89–117` | Planning shorthand “verify --final checks hashes” differs from code: `--hash` must be supplied too. Temporary outputs are exempt; verification writes its own record and does not compare two engines. Use `--hash --final`, preserve source immutability during evaluate-only. |
| Checkpoints | `src/python/starfinder/_checkpoint.py:611–700`; `workflow/_task.py:48,245` | `load_checkpoint` restores three stages; v3 verifies payload hashes, v1/v2 cannot. Changed `blocks_sha256` refuses verified resume. Read checkpoints directly for evaluation; do not alter resume. |
| R8 | `src/python/scripts/workflow_golden.py:740–861` | Comparisons already distinguish same-environment identity from numerical cross-environment comparisons. Reuse by extraction, retain script; there is no pure E05 engine-output comparator in evaluation yet. |
| R11 | `src/python/scripts/x9_local_registration.py:47–120` | Copied decision says wait for W-765; W-765 is now merged (`f35abad`). Dependency satisfied, but this is not an instruction to delete a script in W-771. Retention/removal is an explicit W-775 decision. |
| R12 | `docs/benchmark/`, `docs/benchmark.md`, `docs/benchmark-recipes.md`, `design/notes/benchmark-recipes.md`, `design/notes/benchmark-navigation/` | Current site and notes describe the old lifecycle. Rewrite site from landed implementation; archive stale notes before gate-authorized removal. Never publish this specification under docs. |
| G1 | `src/python/scripts/matlab_stage_comparison.py`; `workflow/scripts/compare_matlab_stages.m`; `design/validation/matlab-stage-comparison.md` | W-212 stays E14's coordinate/channel check, with its tests. M70 agreement cannot replace G1. No MATLAB change or execution. |

## Search method and retirement decisions

The R1–R12 table below includes exact repository-root commands and every file
returned, with matching line numbers, including notebook JSON, CI, examples,
historical notes and test-fixture provenance. Commands use `git ls-files -z`
so ignored tracked files and hidden CI files are included; the starting tree had
no untracked source. `rg` is not limited to Python packages. A filename inventory
supplements content hits for definitions or pages that never name themselves.
Full hit text and commands are retained in W-771's external `search-*.txt` and
`searches.json`. Broad hits are deliberately retained: a recorded artifact path,
old baseline, string guard, lock URL or log path is not a live API caller.

In every inventory row: executable imports/calls must migrate before removal;
current guides/API lists/examples must describe the landed replacement; historical
baselines, source hashes, expected-value provenance and old notebook outputs stay
historical. No old scientific baseline is rewritten to pretend it used new code.
R6 and wait/keep rows retain their callers. The test-disposition table below is
part of the retirement table and overrides any blanket deletion of a test file.

| ID | Proposed disposition and dependency | Caller / page treatment |
| --- | --- | --- |
| R1 | Retire the five public lifecycle objects, old CLI internals, `_lifecycle`, `_records`, `_adapters`, `_pipeline` once replacement behavior is tested. Keep `_trial`/`_experiment`. | Replace benchmark calls in CLI, examples and tests with trial/workflow/evaluate-only/report entries. `tests/qc_benchmark.ipynb` and `tests/qc_registration.ipynb` import and execute the old lifecycle: preserve historical outputs, update imports only under current notebook policy; full runnable migration needs separate explicit scope. Do not add shims to keep historical cells runnable. |
| R2 | Retire `_legacy_evaluation` and `_print_quality_report`. | Pure evaluation already owns metrics; preserve projection and undefined-value behavior in replacements. `_reporting.py` cannot be removed before R7's inspection helper moves. |
| R3 | Retire `benchmarks/recipes.py`, `report_saved.py` and three `configs/*.json` after preserving original source/provenance at this revision. | Remove profile directive/extension and its `docs/conf.py` registration after site replacement; update `docs/examples/benchmark_recipes.py`, `docs/recipes.md`, `design/notes/recipes.md` and references. `.github/workflows/docs.yml:65` is a broad-search false positive: it runs `docs/examples/recipes.py`, not `benchmarks/recipes.py`; leave it and deployment guards unchanged. |
| R4 | Retire two preprocessing scripts after D9 and replacement report/evaluation coverage. | Preserve package metrics, synthetic conditions and test-only pre-move oracle; remove obsolete threshold-selection protocol coverage for dropped decisions. `docs/examples/preprocessing_comparison.py` and notes are callers, not exceptions. `benchmarks/revision.py` remains used by R9/R10; its source inventory will need the deleted-source entries reconciled. |
| R5 | Retire `run_codebook_aware_benchmark.py`, `diagnose_codebook_aware_rescues.py`, `qc_codebook_aware_rescues.py` only after useful metrics/diagnostics have package homes. | Inventory `synthetic_metrics`, `real_metrics`, count correlations/Jaccard, confidence/score summaries and H1 tensor diagnostics before extraction. Keep saved tensor I/O/montage behavior used by `generate_decoding_example_montages.py` and `_decoding_inputs.py`; move shared dependencies before deleting scripts. Do not add dropped metrics to the short set. |
| R6 | Keep `starfinder.evaluation`, sole numerical metric home. | Existing callers/tests stay except old orchestrator imports in tests mapped below. Resource/billing provenance stays external. |
| R7 | Move benchmark `_synthetic_io`, useful `_storage` helpers and `generate_inspection_image` into `starfinder.synthetic`. | There is already `synthetic/_storage.py`; integrate without overwriting it. Update synthetic CLI and `_write_dataset` test import. Benchmark-only storage functions can retire with R1; retain generation manifests, dtype/layout, no-overwrite and identity checks. Lockfile `_storage` hits are unrelated; no dependency edit. |
| R8 | Keep `scripts/workflow_golden.py`; extract engine comparison into package evaluation for E05. | Keep golden/rehearsal callers and fixture archives; script calls shared comparison. Do not repurpose historical normalized goldens as raw cross-engine hash equality. |
| R9 | Wait for W-93: keep `benchmarks/image_statistics.py`. | Keep statistics tests and synthetic-calibration consumers; migrate only imports required by R4. No real-image rerun. |
| R10 | Wait for W-544/W-749: keep appearance/cellular/cellular-chain evidence scripts. | Preserve associated validation-table rows, preset tests and their authorization/resource distinctions. No evidence generation in retirement. |
| R11 | W-765 dependency now satisfied; default keep X9 pending explicit gate disposition. | `test_x9_arguments.py` and orchestration validation reference remain. Retirement draft cannot infer permission from merged dependency alone. |
| R12 | Rewrite Benchmark site after implementation; remove stale notes only after archive and link migration. | Include all four `docs/benchmark/` pages and all six navigation notes, even those with no content-search hit; do not delete the new spec/baseline/validation. Preserve historical links or retarget them to archived source. |

## Tests that read design records

Before rewriting, the repository-wide `record-readers` search found no test
reading `design/specs/benchmark.md`, directly or through `docs_pages.design`.
The old record had no labelled parsed table. No label or row shape was removed.
`src/python/test/docs_pages.py` supplies path helpers; existing design readers
parse other explicitly named records, not all specifications.

`test_benchmark_recipes.py::test_profiles_and_complete_source_map` indirectly
checks existence of `design/specs/benchmark.md` via the source-map target, so the
path stays. It does not parse the page. R3's future removal replaces that test's
profile/archive requirements as listed below. `docs/check_reference.py:80` checks
the site's Benchmark navigation entry and does not parse this spec. R12 must
retain a valid implemented Benchmark landing page. No test changes are needed
for W-771's record rewrite.

## Test disposition rules

The following table names every test in a retiring script/API test file and the
individual tests in shared files that directly touch a retired API. Replacement
names prefixed **new** are proposed requirements, not claims that those tests
exist. All paths are under `src/python/test/`. Keep existing assertions and
markers for retained behavior; changed record/column layouts require the named
migration only. Do not relax numerical tolerances or rewrite expected values.

| Existing test | Named replacement or reason |
| --- | --- |
| `test_benchmark.py::test_saved_lifecycle_and_roundtrip` | **new** `test_experiment_evaluate.py::test_saved_only_and_record_identity`: preserve evaluate/report separation, source immutability, transform metadata and JSON identity; use explicit input archive, not old copied-input layout. |
| `test_benchmark.py::test_resume_and_integrity` | **new** `test_experiment_evaluate.py::test_integrity_and_source_identity`; retain workflow resume tests; drop old lifecycle resume identity because that executor retires. |
| `test_benchmark.py::test_failures_fallback_and_undefined` | **new** `test_experiment_evaluate.py::test_failed_recovered_and_undefined`: consume workflow recovery attempts, failed status and null reasons; no fallback logic in trial executor. |
| `test_benchmark.py::test_evaluation_failure_is_distinct` | **new** `test_experiment_evaluate.py::test_metric_failure_is_distinct` retains separate processing/evaluation status. |
| `test_benchmark.py::test_float_transform_and_dense_artifacts` | **new** `test_benchmark_trial.py::test_float_transform_and_dense_artifacts` preserves fractional metadata and NPZ field identity through saved trial output. |
| `test_benchmark.py::test_cli_lifecycle_and_help` | **new** `test_experiment_cli.py::test_commands_help_and_exit_status` exercises proposed commands plus retained synthetic CLI; old benchmark commands must reject. |
| `test_benchmark.py::test_removed_surface_and_invalid_configs` | **new** `test_experiment_cli.py::test_removed_api_and_invalid_config`; retain removed-name checks and explicit unknown-field errors; old input-root confinement assertion drops because private trial accepts absolute paths, while evaluate source path validation gets its own test. |
| `test_benchmark.py::test_demons_saved_field` | **new** `test_benchmark_trial.py::test_demons_saved_field` with installed SimpleITK, checking saved transform and evaluate-only use; do not substitute an import-only pass. |
| `test_benchmark.py::test_interrupted_trial_preserved` | **new** `test_benchmark_trial.py::test_nonempty_output_is_preserved`: private trial refuses nonempty output; old resume protocol drops. |
| `test_benchmark_recipes.py::test_profiles_and_complete_source_map` | Drop fixed 56-row profile/source-map count because R3 removes those profiles; **new** `test_experiment_cli.py::test_explicit_config_and_archive_links` checks explicit replacement config and archive targets before deletion. |
| `test_benchmark_recipes.py::test_pipeline_shared_lifecycle_and_saved_only` | **new** `test_experiment_evaluate.py::test_workflow_batch_streaming_saved_only`: preserve batch/streaming read parity, source immutability and missing-file failure on workflow outputs. |
| `test_benchmark_recipes.py::test_pipeline_rejects_missing_sources_and_unsafe_paths` | **new** `test_experiment_evaluate.py::test_missing_sources_and_unsafe_paths`; preserve existing workflow planner path/config validation, remove retired adapter calls. |
| `test_benchmark_recipes.py::test_pipeline_records_explicit_recovery_and_failures` | **new** `test_experiment_evaluate.py::test_workflow_recovery_records`: retain requested/actual methods, attempts and skipped evaluation from workflow task records. |
| `test_benchmark_recipes.py::test_historical_report_preserves_scopes_missing_and_failures` | **new** `test_experiment_report.py::test_resource_scopes_missing_and_failures`: retain outer-join rows, scope labels, missing values and output refusal; no legacy report shim. |
| `test_preprocessing_synthetic_evaluation.py::test_smoke_run_writes_tables_and_manifest` | **new** `test_experiment_evaluate.py::test_smoke_run_writes_tables_and_manifest` with saved trial/workflow fixtures, explicit frozen thresholds and both score modes; old CSV layout assertions change to approved evaluation schema. |
| `test_preprocessing_synthetic_evaluation.py::test_shared_arms_equal_direct_recipe_runs` | **new** `test_evaluation_preprocessing.py::test_shared_arms_equal_direct_recipe_runs` on explicit bounded package recipes; preserve stage/unit distinction and radius, remove script orchestration. |
| `test_preprocessing_synthetic_evaluation.py::test_auprc_integrates_in_threshold_order_and_skips_undefined_precision` | Keep assertion under `test_evaluation_preprocessing.py` against public `auprc`; existing `test_scored_auprc_known_answer_subsets_and_undefined` pins analytic values. |
| `test_preprocessing_synthetic_evaluation.py::test_normalized_truth_follows_the_fitted_value_map` | **new** `test_evaluation_preprocessing.py::test_normalized_truth_follows_the_fitted_value_map`, exercised through public `direct_metrics`; preserve normalized-unit interpolation. |
| `test_preprocessing_synthetic_evaluation.py::test_comparison_ranges_are_across_seeds_after_channel_aggregation` | **new** `test_experiment_report.py::test_ranges_after_channel_aggregation`; preserve aggregation order and seed ranges. |
| `test_preprocessing_synthetic_evaluation.py::test_background_ablation_rows_have_before_and_after_background_error` | **new** `test_evaluation_preprocessing.py::test_background_ablation_rows_have_before_and_after_background_error` on explicit bounded package recipes; preserve stage/unit distinction and radius, remove script orchestration. |
| `test_preprocessing_calibrated_evaluation.py::test_calibrated_conditions_are_the_preset_as_returned_with_the_balanced_codebook` | **new** `test_synthetic_preprocessing.py::test_calibrated_conditions_are_the_preset_as_returned_with_the_balanced_codebook` using public synthetic configs; preserve current fixture numbers and discrete outputs. |
| `test_preprocessing_calibrated_evaluation.py::test_bright_outliers_are_four_blobs_at_four_times_the_brightness_median` | **new** `test_synthetic_preprocessing.py::test_bright_outliers_are_four_blobs_at_four_times_the_brightness_median` using public synthetic configs; preserve current fixture numbers and discrete outputs. |
| `test_preprocessing_calibrated_evaluation.py::test_saturation_scales_the_uint16_intensity_parameters_by_k` | **new** `test_synthetic_preprocessing.py::test_saturation_scales_the_uint16_intensity_parameters_by_k` using public synthetic configs; preserve current fixture numbers and discrete outputs. |
| `test_preprocessing_calibrated_evaluation.py::test_saturation_k_is_the_smallest_grid_value_reaching_the_fraction_on_development_seeds` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_clipped_fraction_counts_voxels_above_the_dtype_range` | **new** `test_synthetic_preprocessing.py::test_clipped_fraction_counts_voxels_above_the_dtype_range` using public synthetic configs; preserve current fixture numbers and discrete outputs. |
| `test_preprocessing_calibrated_evaluation.py::test_gain_strong_has_the_ln_channel_spread_and_keeps_the_geometric_mean` | **new** `test_synthetic_preprocessing.py::test_gain_strong_has_the_ln_channel_spread_and_keeps_the_geometric_mean` using public synthetic configs; preserve current fixture numbers and discrete outputs. |
| `test_preprocessing_calibrated_evaluation.py::test_multi_fov_fovs_replace_count_gains_and_scene_key` | **new** `test_synthetic_preprocessing.py::test_multi_fov_fovs_replace_count_gains_and_scene_key` using public synthetic configs; preserve current fixture numbers and discrete outputs. |
| `test_preprocessing_calibrated_evaluation.py::test_background_radius_caps_r_z_at_three_on_eight_planes` | **new** `test_evaluation_preprocessing.py::test_background_radius_caps_r_z_at_three_on_eight_planes` on explicit bounded package recipes; preserve stage/unit distinction and radius, remove script orchestration. |
| `test_preprocessing_calibrated_evaluation.py::test_plan_follows_c2_and_the_pilot_covers_every_new_path` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_threshold_mode_grids_and_fixed_points` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_operating_point_ties_go_to_the_smallest_value_and_grid_edges_are_marked` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_summarize_modes_reports_each_mode_with_its_own_selection_and_fixed_points` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_precondition_is_strict_after_rounding_and_undefined_endpoints_do_not_degrade` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_fixture_check_scope_and_fixture_gap_rows` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_both_multi_fov_checks_run_and_are_reported` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_a_failed_general_check_excludes_the_set_even_when_the_set_specific_check_passes` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_a_failed_set_specific_check_excludes_the_set_when_the_general_check_passes` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_multi_fov_spread_separates_each_fixed_point_and_the_selected_point` | **new** `test_evaluation_preprocessing.py::test_fov_spread_strata_and_fixed_conditions`; retain separate fixed-condition/FOV populations, drop development-selected point branch per W-124. |
| `test_preprocessing_calibrated_evaluation.py::test_the_w233_case_where_one_endpoint_rose_and_the_other_fell_is_low_benefit` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_benefit_bound_is_inclusive_and_clean_equal_to_the_range_holds` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_clean_harm_and_undefined_values_are_low_benefit_with_named_clauses` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_empty_eligible_targets_are_not_assessable_and_still_report_clean` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_histogram_matching_harm_test_is_reported_beside_its_flag` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_extra_roles_add_reported_and_harm_test_rows` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_calibrated_smoke_run_in_both_threshold_modes` | **new** `test_experiment_evaluate.py::test_calibrated_smoke_run_in_both_threshold_modes` with saved trial/workflow fixtures, explicit frozen thresholds and both score modes; old CSV layout assertions change to approved evaluation schema. |
| `test_preprocessing_calibrated_evaluation.py::test_pilot_projection_applies_r3_only_when_the_plan_before_it_does_not_fit` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_calibrated_evaluation.py::test_attach_time_keeps_a_recorded_pilot_projection` | Drop this old script protocol test when R4 retires: it pins development-seed/grid selection, precondition/benefit flags or pilot budget reductions of the superseded preprocessing study; these are not the frozen short-set evaluation contract and must not select new defaults. Preserve archived source/results. |
| `test_preprocessing_report.py::test_comparison_example_runs` | **new** `test_experiment_report.py::test_saved_comparison_example` uses rewritten bounded example, no deleted script import. |
| `test_preprocessing_report.py::test_revision_check_resolves_a_dirty_run_to_the_commit_that_holds_its_diff` | **new** `test_experiment_report.py::test_revision_check_resolves_a_dirty_run_to_the_commit_that_holds_its_diff` preserves the named report behavior using the approved evaluation schema; source/revision failures still refuse output, and old development-selected fields become explicit fixed-condition metadata. |
| `test_preprocessing_report.py::test_revision_check_rejects_a_commit_outside_the_history` | **new** `test_experiment_report.py::test_revision_check_rejects_a_commit_outside_the_history` preserves the named report behavior using the approved evaluation schema; source/revision failures still refuse output, and old development-selected fields become explicit fixed-condition metadata. |
| `test_preprocessing_report.py::test_sources_are_verified_against_the_manifest` | **new** `test_experiment_report.py::test_sources_are_verified_against_the_manifest` preserves the named report behavior using the approved evaluation schema; source/revision failures still refuse output, and old development-selected fields become explicit fixed-condition metadata. |
| `test_preprocessing_report.py::test_every_method_mode_and_condition_has_report_text` | **new** `test_experiment_report.py::test_every_method_mode_and_condition_has_report_text` preserves the named report behavior using the approved evaluation schema; source/revision failures still refuse output, and old development-selected fields become explicit fixed-condition metadata. |
| `test_preprocessing_report.py::test_method_figure_renders_panels_for_a_tiny_scene` | **new** `test_experiment_report.py::test_method_figure_renders_panels_for_a_tiny_scene` preserves the named report behavior using the approved evaluation schema; source/revision failures still refuse output, and old development-selected fields become explicit fixed-condition metadata. |
| `test_preprocessing_report.py::test_calibrated_cards_follow_the_page_order_and_cover_every_method` | **new** `test_experiment_report.py::test_calibrated_cards_follow_the_page_order_and_cover_every_method` preserves the named report behavior using the approved evaluation schema; source/revision failures still refuse output, and old development-selected fields become explicit fixed-condition metadata. |
| `test_preprocessing_report.py::test_calibrated_scenes_outputs_and_detections_match_the_saved_run` | **new** `test_experiment_report.py::test_calibrated_scenes_outputs_and_detections_match_the_saved_run` checks explicitly saved/hash-verified images, detections and representative selection; drop rerunning detection/generation inside renderer. |
| `test_preprocessing_report.py::test_dim_punctum_is_the_lowest_realized_peak_and_leads_the_colour_view` | **new** `test_experiment_report.py::test_dim_punctum_is_the_lowest_realized_peak_and_leads_the_colour_view` checks explicitly saved/hash-verified images, detections and representative selection; drop rerunning detection/generation inside renderer. |
| `test_preprocessing_report.py::test_collapsed_pr_curves_are_detected` | **new** `test_experiment_report.py::test_collapsed_pr_curves_are_detected` preserves the named report behavior using the approved evaluation schema; source/revision failures still refuse output, and old development-selected fields become explicit fixed-condition metadata. |
| `test_preprocessing_report.py::test_calibrated_report_puts_the_summary_first_and_refuses_unverified_inputs` | **new** `test_experiment_report.py::test_calibrated_report_puts_the_summary_first_and_refuses_unverified_inputs` preserves the named report behavior using the approved evaluation schema; source/revision failures still refuse output, and old development-selected fields become explicit fixed-condition metadata. |
| `test_preprocessing_report.py::test_full_tables_keep_every_row_and_column_within_one_capture` | **new** `test_experiment_report.py::test_full_tables_keep_every_row_and_column_within_one_capture`; retain readable/exportable full evidence tables. |
| `test_readout_scripts.py::test_four_scripts_public_diagnostics_and_saved_tensor_smoke` | **new** `test_evaluation_readout_adapters.py::test_saved_tensor_diagnostics_and_identity`: preserve saved decoding, probability normalization, candidate identity and report population; drop import-success assertions for retired scripts. |
| `test_readout_scripts.py::test_registered_stack_cache_round_trips_through_codebook_scripts` | **new** `test_evaluation_readout_adapters.py::test_registered_stack_cache_roundtrip` against extracted package/shared I/O before retiring scripts. |
| `test_readout_scripts.py::test_montage_stack_cache_reads_saved_zyxc_plane` | Keep this named test for retained montage script; migrate its dependencies from R5 scripts to shared saved-image helpers. |
| `test_readout_scripts.py::test_script_extraction_existing_small_fixture` | Keep this named test and public extraction fixture; update only imports affected by R5. |
| `test_benchmark_synthetic.py::test_tiny_registration_cli_layout_fields_and_benchmark_run` | Keep generation/layout assertions; **new** `test_benchmark_trial.py::test_generated_registration_pair_evaluate_only` replaces old API tail. All other tests in this file stay with R7 import updates only. |
| `test_registration_validation.py::test_v1_known_translation_benchmark_task` | **new** `test_registration_validation.py::test_v1_known_translation_trial` with same parameterization/translation tolerance and independent evaluate_translation on saved transform. |
| `test_evaluation.py::test_benchmark_adapter_uses_canonical_metrics` | **new** `test_experiment_evaluate.py::test_canonical_image_metrics_and_policy`; preserve NCC/SSIM and explicit MIP/volume policy without implicit detection. |
| `test_evaluation.py::test_error_only_translation_and_undefined_reporting` | Keep pure evaluate_translation assertions here; **new** `test_experiment_report.py::test_undefined_is_visible` replaces legacy printer assertion. |
| `test_spot_finding_contracts.py::test_mip_caller_keeps_two_spatial_axes` | **new** `test_spot_finding_contracts.py::test_explicit_mip_keeps_two_spatial_axes`: explicitly project/detect then evaluate two spots at distinct X; no legacy evaluator. |
| `test_registration_registry.py::test_registration_spec_validates_its_declarations` | Keep this registry-declaration test unchanged; the benchmark rejection helper is exercised by the following test. |
| `test_registration_registry.py::test_every_registration_place_accepts_a_method_registered_only_in_the_registry` | Keep all stage/workflow registry checks; replace benchmark_config acceptance and rejection-helper subchecks with private trial config resolution in same test. |
| `test_synthetic_contracts.py::test_old_generator_api_is_removed` | Keep removed-generator surface/import-independence checks. `starfinder.benchmark` still exists for private trials; add removed R1 names, do not assert module disappearance. |
| `test_synthetic_amplicon_dataset.py::test_the_public_and_the_streaming_writer_write_the_same_dataset` | Keep same test and byte comparison; migrate `_write_dataset` import under R7. Other tests in file retain public dataset coverage. |
| `test_evaluation_preprocessing.py::test_moved_metrics_and_sweep_equal_pre_move_scenes` | Retain independent `evaluation_preprocessing_oracle.py`; **new** `test_evaluation_preprocessing.py::test_moved_metrics_equal_pre_move_scenes` builds explicit package scenes/recipes, compares direct metrics exactly in one environment. Drop old sweep orchestration only after M10/AUPRC package coverage survives. |
| `test_evaluation_preprocessing.py::test_sweep_read_classification_parity_with_wrong_false_rejected_and_color_calls` | **new** `test_evaluation_preprocessing.py::test_read_classification_parity_with_wrong_false_rejected_and_color_calls` directly compares package M10 and explicitly retained color-call diagnostic to oracle. Other three tests in file stay; remove module-level script import. |

All `test_benchmark_trial.py` tests remain (D5/expansion). All pure evaluation,
registration QC, matching-oracle and other short-set tests remain (R6). R7
retains every other `test_benchmark_synthetic.py` and
`test_synthetic_amplicon_dataset.py` test. `test_benchmark_revision.py` stays
because R9/R10 still use `benchmarks/revision.py`; R4 report revision coverage
moves to the proposed report tests. R8 retains `test_workflow_cell_golden.py`,
`test_workflow_core_golden.py`, `test_workflow_orchestration_golden.py` and their
helpers; extraction adds a pure comparator test rather than deleting goldens.
R9 keeps `test_image_statistics.py`; R10 keeps all listed cellular/appearance
and integration tests and parsed check tables; R11 keeps `test_x9_arguments.py`.
G1 keeps `test_matlab_stage_comparison.py` and `test_matlab_invocation.py`.
No test in these keep/wait groups is proposed for retirement.

## Complete search inventory (part of R1–R12 table)

Line lists are content matches, not a claim that every hit is executable.
A row marked “file inventory” names a target with no self-reference; its command
is the tracked filename search below. Definitions are included as well as callers.

### R1: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e 'run_benchmark\|evaluate_benchmark\|report_benchmark\|BenchmarkCase\|BenchmarkTrialResult\|starfinder\.benchmark\|benchmark[ /](run\|evaluate\|report)\|_benchmark_(run\|evaluate\|report)\|benchmark_lifecycle\|benchmark_recipes\|registration\.benchmark' --` |
| File inventory | `git ls-files \| rg -n -e '^src/python/starfinder/benchmark/\|^docs/examples/benchmark_lifecycle.py$'` |

| File | Matching lines / inventory |
| --- | --- |
| `AGENTS.md` | 28, 141 |
| `benchmarks/README.md` | 4 |
| `benchmarks/appearance_evidence.py` | 53 |
| `benchmarks/image_statistics.py` | 42 |
| `benchmarks/preprocessing_report.py` | 64, 988 |
| `benchmarks/preprocessing_synthetic.py` | 67, 902 |
| `benchmarks/recipes.py` | 11, 23, 70, 85 |
| `benchmarks/revision.py` | 1 |
| `design/baselines/assembly.md` | 32, 35 |
| `design/baselines/orchestration.md` | 51, 130 |
| `design/baselines/segmentation.md` | 30 |
| `design/baselines/spot-finding.md` | 107 |
| `design/baselines/stitching.md` | 31, 33 |
| `design/notes/backends.md` | 16 |
| `design/notes/benchmark-recipes.md` | 69, 117, 249 |
| `design/notes/migration.md` | 27 |
| `design/notes/troubleshooting.md` | 20 |
| `design/proposals/cells-across-fovs.md` | 61, 604 |
| `design/specs/assembly-algorithms.md` | 30, 32 |
| `design/specs/assembly.md` | 86, 89 |
| `design/specs/assignment-algorithms.md` | 33 |
| `design/specs/benchmark.md` | 3, 4, 17, 18, 19, 23, 97, 138, 139, 143 |
| `design/specs/orchestration-engines.md` | 62, 715, 717 |
| `design/specs/orchestration-tiled.md` | 38 |
| `design/specs/orchestration.md` | 61, 1120 |
| `design/specs/readout-algorithms.md` | 26 |
| `design/specs/readout.md` | 83 |
| `design/specs/registration-algorithms.md` | 31 |
| `design/specs/report-algorithms.md` | 46 |
| `design/specs/report.md` | 79 |
| `design/specs/sample-output.md` | 122, 123, 124, 125, 126, 127 |
| `design/specs/sample-segmentation.md` | 124, 125 |
| `design/specs/segmentation-algorithms.md` | 32 |
| `design/specs/segmentation.md` | 56, 57 |
| `design/specs/spot-finding-algorithms.md` | 24, 331 |
| `design/specs/spot-finding.md` | 53 |
| `design/specs/stitching-algorithms.md` | 25, 26 |
| `design/specs/stitching.md` | 83, 85, 957 |
| `design/specs/synthetic/sample-scene.md` | 130 |
| `design/validation/stitching-assembly.md` | 120, 123, 126, 128, 542, 924, 925, 927, 929, 931, 932, 1124 |
| `design/validation/synthetic-real-appearance-scene.md` | 50 |
| `docs/README.md` | 59 |
| `docs/api/benchmark.rst` | 1, 8, 16, 17, 25, 26, 27 |
| `docs/benchmark-recipes.md` | 10, 33, 35 |
| `docs/benchmark.md` | 14, 27, 28, 29, 41, 42, 43, 48, 49 |
| `docs/benchmark/results.md` | 3 |
| `docs/examples/benchmark_lifecycle.py` | file inventory |
| `docs/examples/benchmark_recipes.py` | 3, 18, 67, 69, 75 |
| `docs/migration.md` | 151 |
| `src/python/pyproject.toml` | 143 |
| `src/python/starfinder/__main__.py` | 57, 60, 77, 82, 86, 90, 93 |
| `src/python/starfinder/benchmark/__init__.py` | 2, 3, 5, 6 |
| `src/python/starfinder/benchmark/_adapters.py` | file inventory |
| `src/python/starfinder/benchmark/_experiment.py` | 63 |
| `src/python/starfinder/benchmark/_legacy_evaluation.py` | file inventory |
| `src/python/starfinder/benchmark/_lifecycle.py` | 1, 17, 63, 72, 78, 83, 98, 165, 228, 251, 268, 289, 312 |
| `src/python/starfinder/benchmark/_pipeline.py` | file inventory |
| `src/python/starfinder/benchmark/_records.py` | 12, 65, 72, 78, 144 |
| `src/python/starfinder/benchmark/_reporting.py` | 1 |
| `src/python/starfinder/benchmark/_storage.py` | file inventory |
| `src/python/starfinder/benchmark/_synthetic_io.py` | file inventory |
| `src/python/starfinder/benchmark/_trial.py` | 1, 171, 193 |
| `src/python/test/data/expected/test_preprocessing_golden/README.md` | 23 |
| `src/python/test/data/expected/test_readout_golden/README.md` | 33 |
| `src/python/test/data/expected/test_readout_validation/README.md` | 27 |
| `src/python/test/data/expected/test_registration_golden/README.md` | 21 |
| `src/python/test/data/expected/test_segmentation_golden/README.md` | 23 |
| `src/python/test/data/expected/test_spot_finding_golden/README.md` | 23 |
| `src/python/test/data/expected/test_spot_finding_validation/README.md` | 21 |
| `src/python/test/data/spot_finding_metrics/README.md` | 12, 20 |
| `src/python/test/data/spot_finding_metrics/w218-remeasure-notebook-run.json` | 2 |
| `src/python/test/data/starfish_blob_parity/README.md` | 20 |
| `src/python/test/readout_scenes.py` | 4 |
| `src/python/test/test_benchmark.py` | 12, 13, 31, 39, 50, 51, 62, 69, 75, 78, 84, 104, 108, 117, 124, 138, 192, 215 |
| `src/python/test/test_benchmark_recipes.py` | 12, 13, 76, 96, 100, 104, 123, 136 |
| `src/python/test/test_benchmark_synthetic.py` | 131, 132, 148, 153, 154 |
| `src/python/test/test_benchmark_trial.py` | 13, 31, 124 |
| `src/python/test/test_evaluation.py` | 224, 230, 262 |
| `src/python/test/test_registration_registry.py` | 11 |
| `src/python/test/test_registration_validation.py` | 21, 193, 198, 199 |
| `src/python/test/test_spot_finding_contracts.py` | 148 |
| `src/python/test/test_spot_finding_weights.py` | 27 |
| `src/python/test/test_synthetic_amplicon_dataset.py` | 14 |
| `src/python/test/test_synthetic_contracts.py` | 43, 44, 45 |
| `tests/qc_benchmark.ipynb` | 33, 99, 105, 141, 142, 179, 214 |
| `tests/qc_registration.ipynb` | 29, 313, 320, 321, 322 |
| `workflow/rules/stitching.smk` | 54 |

### R2: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e '_legacy_evaluation\|_print_quality_report' --` |
| File inventory | `git ls-files \| rg -n -e '^src/python/starfinder/benchmark/(_legacy_evaluation\|_reporting).py$'` |

| File | Matching lines / inventory |
| --- | --- |
| `design/baselines/report.md` | 143, 158 |
| `design/baselines/spot-finding.md` | 58 |
| `src/python/starfinder/benchmark/_legacy_evaluation.py` | file inventory |
| `src/python/starfinder/benchmark/_reporting.py` | 7 |
| `src/python/test/test_evaluation.py` | 230, 262, 268 |
| `src/python/test/test_spot_finding_contracts.py` | 148 |

### R3: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e 'benchmarks[/"\x27 ]+(recipes\|report_saved\|configs)\|benchmark_profiles\|source-map\.json\|report_saved\|recipes\.py' --` |
| File inventory | `git ls-files \| rg -n -e '^benchmarks/(recipes.py\|report_saved.py\|configs/)'` |

| File | Matching lines / inventory |
| --- | --- |
| `.github/workflows/docs.yml` | 65 |
| `benchmarks/README.md` | 7, 10, 11 |
| `benchmarks/configs/pipeline.json` | file inventory |
| `benchmarks/configs/registration.json` | file inventory |
| `benchmarks/configs/source-map.json` | 54, 61, 68, 75, 82, 89, 96, 103, 110, 124, 131, 138, 145, 152, 180, 187, 250, 257, 278, 306, 320, 327, 334, 341, 348, 355, 362, 376, 383, 390, 404, 411, 418 |
| `benchmarks/recipes.py` | file inventory |
| `benchmarks/report_saved.py` | 14, 63 |
| `design/notes/benchmark-recipes.md` | 4, 5, 12, 53, 68, 119, 122, 249 |
| `design/notes/contributing.md` | 138 |
| `design/notes/migration.md` | 2477, 2479 |
| `design/notes/recipes.md` | 5, 21, 49, 84, 110, 253, 285 |
| `docs/_ext/benchmark_profiles.py` | 22 |
| `docs/benchmark-recipes.md` | 9, 10, 17, 24, 55 |
| `docs/conf.py` | 28 |
| `docs/examples/benchmark_recipes.py` | 3, 78 |
| `docs/recipes.md` | 7, 20, 32, 47, 60, 99, 114 |
| `src/python/test/test_benchmark_recipes.py` | 32, 41, 47, 140, 147, 155, 158 |

### R4: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e 'preprocessing_synthetic\|preprocessing_report' --` |
| File inventory | `git ls-files \| rg -n -e '^benchmarks/preprocessing_(synthetic\|report).py$'` |

| File | Matching lines / inventory |
| --- | --- |
| `benchmarks/README.md` | 12, 17, 18, 39 |
| `benchmarks/preprocessing_report.py` | 3, 29, 56, 64, 66, 893, 988, 2907 |
| `benchmarks/preprocessing_synthetic.py` | 21, 22, 23, 67, 902 |
| `benchmarks/revision.py` | 4 |
| `design/notes/benchmark-recipes.md` | 141, 147, 148, 166, 167, 189, 193, 213 |
| `design/notes/recipes.md` | 201, 223, 228 |
| `design/specs/preprocessing-algorithms.md` | 439 |
| `design/specs/synthetic/sample-scene.md` | 154 |
| `docs/examples/preprocessing_comparison.py` | 8, 27, 28 |
| `src/python/test/test_evaluation_preprocessing.py` | 17 |
| `src/python/test/test_preprocessing_calibrated_evaluation.py` | 1, 25, 26 |
| `src/python/test/test_preprocessing_report.py` | 21, 22, 46 |
| `src/python/test/test_preprocessing_synthetic_evaluation.py` | 1, 16, 17 |

### R5: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e 'run_codebook_aware_benchmark\|diagnose_codebook_aware_rescues\|qc_codebook_aware_rescues' --` |
| File inventory | `git ls-files \| rg -n -e '^src/python/scripts/(run_codebook_aware_benchmark\|diagnose_codebook_aware_rescues\|qc_codebook_aware_rescues\|_decoding_inputs).py$'` |

| File | Matching lines / inventory |
| --- | --- |
| `design/baselines/readout.md` | 150 |
| `design/notes/research-scripts.md` | 33, 86, 113 |
| `docs/research-scripts.md` | 12, 23, 25, 26 |
| `src/python/scripts/_decoding_inputs.py` | file inventory |
| `src/python/scripts/diagnose_codebook_aware_rescues.py` | file inventory |
| `src/python/scripts/generate_decoding_example_montages.py` | 13 |
| `src/python/scripts/qc_codebook_aware_rescues.py` | file inventory |
| `src/python/scripts/run_codebook_aware_benchmark.py` | file inventory |
| `src/python/test/test_readout_scripts.py` | 20, 22, 23, 30, 33, 37, 45, 50, 67, 68 |

### R6: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e 'starfinder\.evaluation\|evaluation/(matching\|readout\|spot_finding\|registration\|preprocessing)\|evaluation\.(matching\|readout\|spot_finding\|registration\|preprocessing)' --` |
| File inventory | `git ls-files \| rg -n -e '^src/python/starfinder/evaluation/'` |

| File | Matching lines / inventory |
| --- | --- |
| `benchmarks/preprocessing_synthetic.py` | 13, 46, 57, 59, 60, 726, 1834 |
| `design/baselines/readout.md` | 206, 207, 234 |
| `design/baselines/registration.md` | 170, 200 |
| `design/baselines/report.md` | 414 |
| `design/baselines/spot-finding.md` | 63, 65, 67, 177 |
| `design/notes/evaluation-api.rst` | 1, 82 |
| `design/notes/migration.md` | 34, 619, 758, 1121, 2611 |
| `design/notes/synthetic-api.rst` | 300 |
| `design/specs/benchmark.md` | 80 |
| `design/specs/readout-algorithms.md` | 391, 392 |
| `design/specs/readout.md` | 192 |
| `design/specs/registration-algorithms.md` | 217 |
| `design/specs/registration.md` | 571 |
| `design/specs/report-algorithms.md` | 277 |
| `design/specs/report.md` | 875, 1109, 1288 |
| `design/specs/spot-finding-algorithms.md` | 221 |
| `docs/api/evaluation.registration.rst` | 1, 11, 21, 34, 45, 63, 82, 100 |
| `docs/api/python.rst` | 26 |
| `docs/migration.md` | 93 |
| `src/python/pyproject.toml` | 142 |
| `src/python/scripts/run_codebook_aware_benchmark.py` | 26, 27 |
| `src/python/starfinder/benchmark/_adapters.py` | 98 |
| `src/python/starfinder/benchmark/_legacy_evaluation.py` | 13 |
| `src/python/starfinder/benchmark/_reporting.py` | 4 |
| `src/python/starfinder/dataset/fov.py` | 923 |
| `src/python/starfinder/evaluation/__init__.py` | file inventory |
| `src/python/starfinder/evaluation/_result.py` | file inventory |
| `src/python/starfinder/evaluation/matching.py` | file inventory |
| `src/python/starfinder/evaluation/preprocessing.py` | file inventory |
| `src/python/starfinder/evaluation/readout.py` | file inventory |
| `src/python/starfinder/evaluation/registration.py` | file inventory |
| `src/python/starfinder/evaluation/spot_finding.py` | file inventory |
| `src/python/starfinder/registration/_landmarks.py` | 60 |
| `src/python/test/data/spot_finding_metrics/README.md` | 4 |
| `src/python/test/evaluation_preprocessing_oracle.py` | 11 |
| `src/python/test/learned_detectors.py` | 19 |
| `src/python/test/readout_scenes.py` | 23 |
| `src/python/test/test_demons.py` | 206 |
| `src/python/test/test_e2e.py` | 21, 22, 23 |
| `src/python/test/test_evaluation.py` | 10, 11, 12, 13, 29, 30, 31, 32 |
| `src/python/test/test_evaluation_preprocessing.py` | 10, 11 |
| `src/python/test/test_evaluation_short_set.py` | 10, 11, 12 |
| `src/python/test/test_pointset.py` | 218 |
| `src/python/test/test_preprocessing_synthetic_evaluation.py` | 31 |
| `src/python/test/test_readout_deduplication.py` | 29 |
| `src/python/test/test_readout_metrics.py` | 11 |
| `src/python/test/test_readout_module_names.py` | 9, 15 |
| `src/python/test/test_readout_scoring.py` | 28 |
| `src/python/test/test_readout_validation.py` | 53, 54 |
| `src/python/test/test_registration_methods.py` | 23, 184 |
| `src/python/test/test_registration_other_rounds.py` | 20 |
| `src/python/test/test_registration_qc.py` | 6 |
| `src/python/test/test_registration_qc_memory.py` | 13 |
| `src/python/test/test_registration_validation.py` | 24 |
| `src/python/test/test_spot_finding_local_maxima.py` | 16 |
| `src/python/test/test_spot_finding_metrics.py` | 11 |
| `src/python/test/test_spot_finding_validation.py` | 40 |
| `src/python/test/test_starfish_log.py` | 26 |
| `src/python/test/test_summaries.py` | 11 |
| `src/python/test/test_workflow_tiled.py` | 170 |
| `tests/qc_registration.ipynb` | 28, 30 |

### R7: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e '_synthetic_io\|_storage\|generate_inspection_image' --` |
| File inventory | `git ls-files \| rg -n -e '^src/python/starfinder/benchmark/(_synthetic_io\|_storage\|_reporting).py$'` |

| File | Matching lines / inventory |
| --- | --- |
| `design/baselines/report.md` | 145, 159 |
| `src/python/scripts/x9_local_registration.py` | 61 |
| `src/python/starfinder/__main__.py` | 57, 60 |
| `src/python/starfinder/benchmark/_lifecycle.py` | 18 |
| `src/python/starfinder/benchmark/_pipeline.py` | 11 |
| `src/python/starfinder/benchmark/_reporting.py` | 16 |
| `src/python/starfinder/benchmark/_storage.py` | file inventory |
| `src/python/starfinder/benchmark/_synthetic_io.py` | 96, 146, 147, 168 |
| `src/python/starfinder/synthetic/__init__.py` | 11 |
| `src/python/starfinder/synthetic/_sample.py` | 35 |
| `src/python/test/test_synthetic_amplicon_dataset.py` | 14 |
| `src/python/uv.lock` | 6436, 6438 |

### R8: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e workflow_golden --` |
| File inventory | `git ls-files \| rg -n -e '^src/python/scripts/workflow_golden.py$'` |

| File | Matching lines / inventory |
| --- | --- |
| `design/baselines/orchestration.md` | 667 |
| `design/specs/orchestration.md` | 68, 145 |
| `design/validation/orchestration.md` | 35 |
| `docs/examples/prepare_workflow.py` | 5 |
| `docs/examples/workflow-full.yaml` | 3 |
| `docs/examples/workflow-minimal.yaml` | 2 |
| `docs/workflow/index.md` | 29 |
| `src/python/scripts/workflow_golden.py` | file inventory |
| `src/python/scripts/workflow_rehearsal_scene.py` | 5, 16, 125, 128 |
| `src/python/test/data/expected/test_workflow_core_golden/README.md` | 31, 50, 59 |
| `src/python/test/data/expected/test_workflow_core_golden/record_core_values.py` | 5, 22, 27, 29, 48, 58, 97, 111, 153, 203 |
| `src/python/test/data/expected/test_workflow_orchestration_golden/README.md` | 8, 10, 25, 26, 27, 28, 69 |
| `src/python/test/data/expected/test_workflow_orchestration_golden_export/README.md` | 10, 12, 31, 32, 33, 34 |
| `src/python/test/data/expected/test_workflow_plan/README.md` | 5 |
| `src/python/test/test_workflow_cell_golden.py` | 3, 10, 46 |
| `src/python/test/test_workflow_core_golden.py` | 6 |
| `src/python/test/test_workflow_orchestration_golden.py` | 4, 8, 45, 46, 198, 247 |
| `src/python/test/workflow_examples.py` | 3, 56, 57, 58 |

### R9: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e image_statistics --` |
| File inventory | `git ls-files \| rg -n -e '^benchmarks/image_statistics.py$'` |

| File | Matching lines / inventory |
| --- | --- |
| `benchmarks/README.md` | 21, 39 |
| `benchmarks/image_statistics.py` | 11, 12, 13, 15, 42, 682 |
| `benchmarks/preprocessing_report.py` | 990, 2245, 2316, 2443, 2448, 2464, 2480, 2737 |
| `benchmarks/preprocessing_synthetic.py` | 1113, 1114, 1116, 1117, 1119, 1126, 1146, 1795, 1813, 1854 |
| `benchmarks/revision.py` | 3 |
| `design/notes/benchmark-recipes.md` | 180, 225 |
| `design/notes/image-statistics.md` | 3, 21, 22, 23, 24, 35 |
| `design/specs/synthetic/sample-scene.md` | 155 |
| `src/python/test/test_image_statistics.py` | 1, 20 |
| `src/python/test/test_preprocessing_calibrated_evaluation.py` | 462 |

### R10: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e 'appearance_evidence\|cellular_evidence\|cellular_chain_evidence' --` |
| File inventory | `git ls-files \| rg -n -e '^benchmarks/(appearance_evidence\|cellular_evidence\|cellular_chain_evidence).py$'` |

| File | Matching lines / inventory |
| --- | --- |
| `benchmarks/README.md` | 24, 29, 32 |
| `benchmarks/appearance_evidence.py` | 11, 13, 92, 382 |
| `benchmarks/cellular_chain_evidence.py` | 17, 19, 99, 174 |
| `benchmarks/cellular_evidence.py` | 12, 13, 59, 225 |
| `design/notes/contributing.md` | 419, 420 |
| `design/specs/synthetic/cellular-scene.md` | 2043, 2051 |
| `design/specs/synthetic/real-appearance-scene.md` | 1164 |
| `design/validation/synthetic-cellular-scene.md` | 290, 292, 305, 389, 390, 410, 544, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 804, 806, 809, 932, 1015, 1025, 1030, 1035, 1039, 1040 |
| `design/validation/synthetic-real-appearance-scene.md` | 137, 502, 524, 527, 548, 616, 617, 618, 619, 735, 736, 740, 741, 749, 838, 842, 903, 913, 918 |
| `src/python/test/test_integration_cellular_chain.py` | 52, 113 |
| `src/python/test/test_synthetic_appearance_presets.py` | 9, 204 |
| `src/python/test/test_synthetic_cellular_checks.py` | 13, 122, 141, 142 |
| `src/python/test/test_synthetic_cellular_presets.py` | 8, 33, 171 |

### R11: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e x9_local_registration --` |
| File inventory | `git ls-files \| rg -n -e '^src/python/scripts/x9_local_registration.py$'` |

| File | Matching lines / inventory |
| --- | --- |
| `design/validation/orchestration.md` | 181 |
| `src/python/scripts/x9_local_registration.py` | file inventory |
| `src/python/test/test_x9_arguments.py` | 9 |

### R12: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e 'benchmark-recipes\|benchmark-navigation\|docs/benchmark\|benchmark/index\|benchmark/experiments\|benchmark/data\|benchmark/results\|specs/benchmark\|api/benchmark\|registration\.benchmark' --` |
| File inventory | `git ls-files \| rg -n -e '^docs/benchmark/\|^design/notes/benchmark-navigation/\|^design/notes/benchmark-recipes.md$\|^docs/(benchmark\|benchmark-recipes).md$\|^docs/api/(benchmark\|registration.benchmark).rst$'` |

| File | Matching lines / inventory |
| --- | --- |
| `benchmarks/README.md` | 5 |
| `benchmarks/configs/source-map.json` | 33, 40, 47, 117, 159, 166, 173, 264, 271, 285, 292, 299, 313, 369, 397 |
| `design/README.md` | 19 |
| `design/notes/backends.md` | 66 |
| `design/notes/benchmark-navigation/data.md` | file inventory |
| `design/notes/benchmark-navigation/dataset.md` | file inventory |
| `design/notes/benchmark-navigation/experiments.md` | file inventory |
| `design/notes/benchmark-navigation/index.md` | 12, 19, 26 |
| `design/notes/benchmark-navigation/real-data.md` | file inventory |
| `design/notes/benchmark-navigation/results.md` | file inventory |
| `design/notes/benchmark-recipes.md` | 3 |
| `design/notes/migration.md` | 453, 2612 |
| `design/specs/benchmark.md` | 7 |
| `docs/api/benchmark.rst` | 6 |
| `docs/api/evaluation.registration.rst` | 6 |
| `docs/api/registration.benchmark.rst` | file inventory |
| `docs/benchmark-recipes.md` | 5, 71, 72, 73 |
| `docs/benchmark.md` | 5, 7, 52, 64, 82 |
| `docs/benchmark/data.md` | file inventory |
| `docs/benchmark/experiments.md` | 11 |
| `docs/benchmark/index.md` | file inventory |
| `docs/benchmark/results.md` | 5 |
| `docs/check_reference.py` | 80 |
| `docs/conf.py` | 68 |
| `docs/index.md` | 79, 83, 84, 85, 98 |

### G1: exhaustive matches

| Search | Command from repository root |
| --- | --- |
| Content | `git ls-files -z \| xargs -0 rg --no-heading -n --with-filename -e 'matlab_stage_comparison\|compare_matlab_stages' --` |
| File inventory | `git ls-files \| rg -n -e '^design/validation/matlab-stage-comparison.md$\|^src/python/scripts/matlab_stage_comparison.py$\|^workflow/scripts/compare_matlab_stages.m$'` |

| File | Matching lines / inventory |
| --- | --- |
| `design/validation/matlab-stage-comparison.md` | 29, 75, 127, 137, 142, 145, 148 |
| `src/python/scripts/matlab_stage_comparison.py` | 6, 8, 9, 11, 14, 45, 67, 69, 158, 276, 279 |
| `src/python/test/data/matlab_reference/provenance.json` | 108, 113, 126, 129, 146, 148 |
| `src/python/test/test_matlab_invocation.py` | 113, 114, 128, 131, 132 |
| `src/python/test/test_matlab_stage_comparison.py` | 4, 5, 18, 19, 41, 44 |
| `workflow/scripts/compare_matlab_stages.m` | 1, 5 |

### Record-reader search

Run the following from the repository root before rewriting:

`git ls-files -z | xargs -0 rg --no-heading -n --with-filename -e 'specs/benchmark|design\(["\x27]benchmark|benchmark\.md|benchmark-recipes|benchmark-navigation' --`

No `src/python/test/` hit names the specification as a text reader; the indirect
source-map existence check described above was inspected separately.

### R12 supplement: navigation wording and responsibilities

This broad wording search includes harmless synthetic-preset names and artifact
paths as well as live page links; retain those historical/independent uses.
`CONTEXT.md` and `docs/architecture.md` need implementation-era responsibility
text updates. The synthetic quickstart/demo keeps its public synthetic imports.

| Search | Command from repository root |
| --- | --- |
| Wording | `git ls-files -z -- docs CONTEXT.md \| xargs -0 rg --no-heading -n --with-filename -i -e benchmark --` |

| File | Matching lines |
| --- | --- |
| `CONTEXT.md` | 26, 27 |
| `docs/README.md` | 33, 59 |
| `docs/_ext/benchmark_profiles.py` | 1, 12, 21, 22, 32 |
| `docs/_ext/synthetic_presets.py` | 31 |
| `docs/api/backends.md` | 33 |
| `docs/api/benchmark.rst` | 1, 5, 6, 8, 16, 17, 25, 26, 27 |
| `docs/api/evaluation.registration.rst` | 5, 6 |
| `docs/api/python.rst` | 24 |
| `docs/api/registration.benchmark.rst` | 3, 8 |
| `docs/api/registration.pointset.rst` | 8 |
| `docs/api/registration.pyramid.rst` | 8 |
| `docs/api/synthetic.rst` | 7, 23, 24 |
| `docs/architecture.md` | 12 |
| `docs/assignment-contract.md` | 28, 31 |
| `docs/benchmark-recipes.md` | 1, 4, 5, 9, 10, 17, 21, 24, 28, 33, 35, 55, 71, 72, 73 |
| `docs/benchmark.md` | 1, 3, 5, 7, 14, 27, 28, 29, 41, 42, 43, 48, 49, 52, 64, 69, 73, 82 |
| `docs/benchmark/experiments.md` | 10, 11 |
| `docs/benchmark/index.md` | 1, 4, 22, 29 |
| `docs/benchmark/results.md` | 3, 4, 5 |
| `docs/check_reference.py` | 80 |
| `docs/checkpoints.md` | 19, 27 |
| `docs/conf.py` | 28, 68 |
| `docs/contributing.md` | 142 |
| `docs/coordination.md` | 20, 28 |
| `docs/examples/benchmark_lifecycle.py` | 38, 42, 43 |
| `docs/examples/benchmark_recipes.py` | 3, 18, 24, 67, 69, 75 |
| `docs/examples/preprocessing_comparison.py` | 8, 28 |
| `docs/examples/quickstart.py` | 20, 27, 28, 29, 72 |
| `docs/get-started/single-fov-demo.md` | 42, 55, 63 |
| `docs/index.md` | 79, 81, 83, 84, 85, 98 |
| `docs/migration.md` | 69, 150, 151, 326 |
| `docs/preprocessing-contract.md` | 25, 28 |
| `docs/readout-contract.md` | 22, 25 |
| `docs/registration-contract.md` | 22, 25 |
| `docs/research-scripts.md` | 5, 26 |
| `docs/segmentation-contract.md` | 26, 29 |
| `docs/spot-finding-algorithms.md` | 15 |
| `docs/spot-finding-contract.md` | 23, 26, 84 |
| `docs/synthetic-reference.md` | 11, 13, 23 |
| `docs/synthetic-specification.md` | 20, 25, 47, 72 |
| `docs/troubleshooting.md` | 76 |
| `docs/workflow/broad-uger.md` | 37 |
