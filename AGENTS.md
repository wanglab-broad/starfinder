# Agent operations

Read [CONTEXT.md](CONTEXT.md) for vocabulary and package responsibilities,
[architecture](docs/architecture.md) and [contracts](docs/api/contracts.md) for
maintained behavior, and [migration](docs/migration.md) for breaking Python changes.
Python naming is snake_case/PascalCase with the explicit exception **FOV stays FOV**.
MATLAB APIs and shared MATLAB-facing keys, filenames and coordinate contracts stay unchanged.
Read the canonical [dataset catalog](docs/datasets.md) before selecting inputs;
availability is not scientific qualification. Update it and the external manifest
when delivering a new fixture or version.

## Scope and ownership

Before work, read the live issue, dependencies and acceptance criteria, and the
[current initiative workflow](https://linear.app/jiahaoh/document/thesis-and-implementation-workflow-c12f30bffe9f).
Verify repository/worktree, branch, clean starting revision, inputs, environment
and resource limits. Record session/host/revision and a short plan in Linear;
assign Jiahao and require the relevant milestone. Unknown inputs remain unverified.
Preserve unrelated edits; never reset another checkout or rewrite Git history.

Linear owns current scope, plans, decisions and validation evidence. Software Git
owns maintained code/docs/configs/tests, not diaries or benchmark run reports.
Scientific acceptance belongs to Chapter II/W-57 and W-92/W-93/W-94/W-124; software
checks do not qualify data, molecular truth or scientific claims. Scope changes
belong in the linked chapter discussion. Historical checklists do not authorize work.

For Chapter II development, read the live
[outline](https://linear.app/jiahaoh/document/chapter-ii-development-outline-and-priorities-6d2ed68bc91f),
[execution plan](https://linear.app/jiahaoh/document/chapter-ii-development-milestones-1-2-execution-plan-and-acceptance-c16c39fb9624)
and [controller guidance](https://linear.app/jiahaoh/document/reusable-autonomous-issue-controller-operating-guideline-be87b45b6d96).
The authorized batch is ordered W-153 through W-160, project
`ae82ed79-ed07-4f20-896f-2dbc79f83a7d`, on `codex/chapter-ii-batch1-20260921`
in `.worktrees/chapter-ii-batch1-20260921`. Each worker owns only its dispatched
issue. Preserve dirty main-checkout AGENTS.md and `.claude/` exactly; never edit
another worktree. Historical housekeeping branch, state and authorization do not
apply here. One outer controller owns dispatch, with fresh sessions per issue and
bounded same-issue repairs; never launch a nested controller or parallel worker.
Follow the current phase's commit instructions: implementation handoff is
uncommitted; controller review precedes a separately requested local commit.
No push, merge, deployment or publication; do not change deployment guards.
Leave the project open. Stop after W-160. W-173/W-174 are human-owned gates,
excluded from all execution allowlists; W-161 and later are not authorized.
Only Jiahao's explicit revision/packet approval and live read-back can pass a
human gate. W-160 must deliver standalone, browser-opened HTML from saved 3D
and Z=1 outputs, linked from W-160/W-173; it cannot depend on W-165.

## Implementation

Change only the authorized slice and maintained callers/tests/docs needed for
coherence. Verify diagnoses in actual code. Prefer functions and small typed
configs/results; avoid duplicated algorithms and compatibility shims for replaced
Python APIs. Keep arrays ZYX/ZYXC, explicit channel ordering and boundary conversions
as specified in the site contracts. Update notebooks' imports without executing
historical notebooks or rewriting outputs. Do not infer qualification of historical
process-dependent synthetic hash seeds. W-154/W-155 must specify independent
numerical expectations and round-trip criteria before dependent implementation.
Preserve historical source before extraction.

## Environment and validation

From `src/python`, run Python with `uv run python`. On GP099-29C use
`/home/unix/jiahao/.local/bin/uv` and
`UV_PYTHON=/home/unix/jiahao/miniforge3/bin/python3.12`; system Python 3.8 is unsuitable.
Use `PYTHONDONTWRITEBYTECODE=1 UV_LOCKED=true UV_NO_SYNC=true UV_OFFLINE=true` with the prepared
locked environment. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`, `NUMEXPR_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` and `BLIS_NUM_THREADS` to 1,
and `CUDA_VISIBLE_DEVICES=""`, `MPLBACKEND=Agg`. Keep TMPDIR on local `/tmp/starfinder-batch1-20260921`,
not the SMB artifact mount, because tests create symlinks. Use a writable uv cache
when the default cache is sandboxed. Do not silently alter dependencies or lock.
The operator prepared this worktree with
`uv sync --locked --no-default-groups --group docs --extra dev --extra local-registration`.
Do not install/sync during validation. Verify imported source paths and extras;
record a missing dependency as a blocker. Propose required format dependencies
explicitly before a separately authorized setup.

Run focused checks, then the required gates before committing (the controller may
run these after the uncommitted implementation handoff; do not duplicate them):

```bash
uv run pytest test/ -v
uv run --group docs sphinx-build -n -W --keep-going -b html ../../docs /external/new-run/html
uv run python ../../docs/check_reference.py
```

Run affected bounded examples from [contributing](docs/contributing.md). Generated
API stubs are ignored/disposable; edit authored lists/docstrings. Never weaken
assertions, hide warnings/skips or claim an unexecuted backend passed. SimpleITK
must be installed for its execution coverage; MATLAB execution is excluded.

Chapter II batch limits: CPU 0, one numerical thread, no GPU; 5400 s worker,
1800 s/check, at most two repairs; target process RSS ≤4 GiB, new artifacts ≤1 GiB.
New images ≤32×64×64 ZYX, ≤4 channels/rounds. Inventory existing test inputs and
generated legacy exceptions before reuse; see [fixture inventory](docs/datasets.md#development-fixtures-and-resource-boundaries).
Do not regenerate existing fixture TIFFs. Missing ignored fixture bytes must be
restored from a verified existing source, with checksums in the run manifest.
Medium/large historical inputs are catalog entries, not execution authorization.
No scientific sweeps, large generation, training, real-image processing or cloud jobs.
Measure with `/usr/bin/time -v`; maximum RSS is KiB. CPU affinity/thread settings
and supervisor timeouts are controls, not proof of an enforced cgroup RSS cap.
Stop for actual resource overruns rather than expanding limits.

## Artifacts and handoff

Write logs, previews, metrics, manifests and patches outside Git under
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/<issue>/<unique-run>/`.
Use full-file writes on the SMB mount. Each manifest records issue/run/date/host,
code revision and dirty patch identity, data/config/source checksums, exact commands,
environment, seeds/resources, validation/exit status, output checksums and limitations.
Owner Jiahao; retain through thesis/project handoff and associated publication;
no deletion without owner decision. Backup coverage is unverified unless confirmed.
Private paths alone do not establish public reproducibility.

Record meaningful progress and completion in Linear. Supply exact commands/results,
artifacts and limitations, commit/push/merge state and follow-ups. If uncommitted,
preserve a patch and untracked source snapshot. Mark Done only after acceptance and
controller evidence review; verify assignee/milestone/status and milestone progress.
Implementation Done is neither human nor scientific acceptance. Review history for
commit style: numbered messages for major modules, otherwise `prefix(info): message`.
Update live workflow guidance in place for agreed substantive practice changes,
with an administrative issue, dated rationale and synchronized affected instructions.
