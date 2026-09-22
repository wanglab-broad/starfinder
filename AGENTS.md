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
Current authorization, allowlist, baseline, worktree, phase instructions and
human gates belong to the live Linear batch contract, not a reusable static
checklist. Each worker owns only its dispatched issue. Preserve unrelated edits
and other worktrees. One controller owns dispatch; never launch a nested
controller or parallel worker. Version-2 workers leave source uncommitted; the
controller validates and commits it, then obtains independent acceptance.
Push, merge, deployment and publication require their own authorization.
Human approval and scientific acceptance remain separate from implementation Done.

**Outer launcher only:** follow [batch execution](docs/batch-execution.md), start
the authorized controller, confirm startup and return its execution identity and
report destination, then end the turn. Do not sleep/poll or keep a model supervising
idle execution. This rule does not stop workers from completing their own issue.
Before batch 3, require completed W-176 and explicit live W-174 human approval;
completion of either does not by itself authorize dispatch.

## Implementation

Change only the authorized slice and maintained callers/tests/docs needed for
coherence. Verify diagnoses in actual code. Prefer functions and small typed
configs/results; avoid duplicated algorithms and compatibility shims for replaced
Python APIs. Keep arrays ZYX/ZYXC, explicit channel ordering and boundary conversions
as specified in the site contracts. Update notebooks' imports without executing
historical notebooks or rewriting outputs. Do not infer qualification of historical
process-dependent synthetic hash seeds. Preserve independent numerical
expectations and round-trip criteria for current behavior.
Preserve historical source before extraction.

## Environment and validation

From `src/python`, run Python with `uv run python`. On GP099-29C use
`/home/unix/jiahao/.local/bin/uv` and
`UV_PYTHON=/home/unix/jiahao/miniforge3/bin/python3.12`; system Python 3.8 is unsuitable.
Use `PYTHONDONTWRITEBYTECODE=1 UV_LOCKED=true UV_NO_SYNC=true UV_OFFLINE=true` with the prepared
locked environment. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`, `NUMEXPR_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` and `BLIS_NUM_THREADS` to 1,
and `CUDA_VISIBLE_DEVICES=""`, `MPLBACKEND=Agg`. Keep TMPDIR on local `/tmp/<unique-run-id>`,
not the SMB artifact mount, because tests create symlinks. Use a writable uv cache
when the default cache is sandboxed. Do not silently alter dependencies or lock.
Before exporter/viewer setup, reuse the prepared locked Python 3.12 environment with
`UV_PROJECT_ENVIRONMENT=/home/unix/jiahao/Github/starfinder/.worktrees/chapter-ii-batch1-20260921/src/python/.venv`
and set `PYTHONPATH` to the **current authorized worktree** plus `/src/python`.
The operator prepared its docs/dev/local-registration/checkpoint dependencies;
verify actual availability and source paths before checks.
Do not install/sync during validation. Verify imported source paths and extras;
record a missing dependency as a blocker. Propose required format dependencies
explicitly before a separately authorized setup.
For batch 3, W-168 specifies an isolated pinned exporter/viewer environment;
preserve the batch-1 environment and record the selected environment identity
before dependent use. Follow the agreed [prerequisite sequence](docs/batch-execution.md#batch-3-prerequisite-sequence).

Run focused checks during implementation. For executable-code candidates, run
the retained default suite once before committing; documentation changes require
the strict build and reference audit. The controller may run these after handoff;
do not duplicate unchanged successful checks. At batch acceptance and for affected
end-to-end contracts, also run the extended tests. See
[validation policy](docs/batch-execution.md#validation-and-reuse) for evidence reuse.
W-176 itself requires both suites and both documentation gates before commit:

```bash
uv run pytest test/ -v
uv run pytest test/ -v -m extended
uv run --group docs sphinx-build -n -W --keep-going -b html ../../docs /external/new-run/html
uv run python ../../docs/check_reference.py
```

Run affected bounded examples from [contributing](docs/contributing.md). Generated
API stubs are ignored/disposable; edit authored lists/docstrings. Never weaken
assertions, hide warnings/skips or claim an unexecuted backend passed. SimpleITK
must be installed for its execution coverage. MATLAB execution is excluded except
for W-171's agreed bounded Python/MATLAB qualification and its environment,
license and toolbox preflight. Use the existing Snakemake invocation path,
preserve MATLAB interfaces, and retain the resource limits below. Missing actual
MATLAB execution blocks W-171; source inspection cannot substitute for it.

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
