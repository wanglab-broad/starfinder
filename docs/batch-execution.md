# Batch execution

The outer Codex task starts an explicitly authorized batch and reports startup.
It then ends its turn. The deterministic runner handles waiting, validation,
recovery and terminal reports; a standing model is unnecessary.

The reusable procedure lives in the runner repository's
`prompts/launch-batch.md`; configuration version 2 is described in its
`EFFICIENCY.md` and `projects/example-v2.json`. On the development host the runner
repository is `/home/unix/jiahao/Github/linear-codex-runner`. Select a reviewed
revision/worktree explicitly rather than assuming the default checkout has the
new implementation. Shared policy lives in the
[controller operating guideline](https://linear.app/jiahaoh/document/reusable-autonomous-issue-controller-operating-guideline-be87b45b6d96).

## Launch contract

Record the ordered issue allowlist, baseline, branch/worktree, config identity,
authorization, data/environment identities, resource limits and human gates in
Linear. Assign one task label (Research, Implementation, Validation, Maintenance)
and one profile (Deep, Standard, Economy) per issue. Research, architecture and
scientific acceptance need substantive model judgment; known test commands use
no model. The runner records requested profile/model/effort and available observed
runtime identity in the issue's execution summary.

Configure a durable process supervisor and local artifact storage. Validate the
configuration and current prerequisites, start once, confirm startup and return
its execution identity plus local/Linear terminal-report destinations. The user
can retrieve the saved report later. Same-task automatic wakeup is optional and
must be verified on the host; do not replace it with model polling.

Batch 3 requires completed W-176, explicit W-174 human approval of the accepted
revision/packet, and a separately authorized batch contract. Include W-176 in
`required_done` and W-174's exact approval evidence in `human_gates`. Never include
human gates in the worker allowlist or equate software checks with human approval.
Historical batch configurations and report packets remain preserved.

## Batch 3 prerequisite sequence

The [current execution contract](https://linear.app/jiahaoh/document/chapter-ii-development-milestones-1-2-execution-plan-and-acceptance-c16c39fb9624)
records Jiahao's viewer, MATLAB and delivery decisions. Gate prerequisites by the
work they enable; final export/report implementations do not precede their specification.

W-168 selects compatible pinned SpatialData, OME-Zarr, napari, Qt,
napari-spatialdata and Fiji reader/plugin versions. Prepare an isolated environment
and qualify dependencies before dependent use, preserving the earlier environment
and recording dependency/lock changes. Checks remain locked/offline/no-sync.

Automated sessions of actual napari/SpatialData and Fiji viewers satisfy technical
acceptance. Reopen exported fixtures, exercise applicable 2D navigation, 3D
rendering, resolution levels and regional access, and verify channels, calibration,
alignment and molecule/cell/label relationships. Retain versions, commands, input
hashes, logs, screenshots and criterion results. A qualified virtual display and
software rendering may be used within the existing no-GPU/resource limits.
Import/schema/reader-only checks do not prove viewer acceptance. Required missing
capabilities block W-170; no additional interactive desktop review is required.

W-171 retains actual bounded Python/MATLAB comparisons. Its scoped execution
exception includes environment/license/toolbox preflight. Use the existing
Snakemake invocation path and unchanged MATLAB interfaces. Freeze stage criteria,
tolerances, coordinate/channel/population mappings and independent spot matching
before comparisons. Missing runtime execution blocks acceptance. Existing fixture,
CPU/thread/no-GPU/time/RSS/storage limits remain in force.

W-168 freezes the scientific evidence schema, saved-input identities, command
interface, output paths, report/offline-opening checks and per-issue ownership.
Its own specification and standalone HTML/manifest remain required; the final
W-172 scientific renderer is not a prerequisite to starting W-168. W-169 supplies
raster/fallback evidence, W-170 export/viewer evidence, W-171 storage/round-trip/
backend evidence, and W-172 the integrated scientific HTML and execution audit.
Each owning issue must implement and pass its applicable saved-evidence delivery
checks before independent acceptance. Final rendering consumes saved results.
Generic runner terminal HTML cannot replace scientific report acceptance.

At the W-168 handoff, reconcile the stopped controller configuration/guidance and
environment/data identities, then pin executable delivery hooks before dependent
continuation. Schema 2 runs configured hooks at each issue's post-commit delivery
step; specify an issue-aware command or a reviewed successor configuration.
Do not silently change a running config or treat empty hooks as qualification.
Runner telemetry collection remains fixed before the first job. These technical
handoff steps introduce no additional human approval gate.

## Validation and reuse

From `src/python`, in the prepared environment:

```bash
uv run pytest test/ -v                 # Retained default suite
uv run pytest test/ -v -m extended     # Existing full-pipeline cases
uv run pytest test/ -v -m ""           # All retained tests, when needed
```

Use focused checks during implementation. Run the default suite once per
executable-code candidate before committing. Run extended cases at batch
acceptance and whenever their pipeline interfaces are affected. Documentation
changes require the strict Sphinx build and reference check in
[contributing](contributing.md). Presentation-only repairs require the renderer's
behavioral checks, offline browser opening and report invariants; unchanged
numerical evidence may be reused.

Runner check input patterns must include all relevant code, fixtures (including
ignored TIFF bytes), configuration and lockfiles. External data and installed
environment manifests belong in `efficiency.identity_files`. Refresh environment
identity after a package/interpreter change. Reuse requires the same inputs,
command/environment and intact successful log; failures never become cached
passes. Unknown impact requires broader validation. Changed labels do not alter
an already dispatched issue's profile.

Retain independent numerical expectations, geometry/channel/coordinate contracts,
failure cases, persistence and supported compatibility. Delete obsolete behavior
and duplicate checks only after mapping their remaining coverage. A lower test
count is not a correctness metric. Report warnings, skips, commands and evidence
in Linear; run-specific deletion inventories and timings stay outside software Git.
