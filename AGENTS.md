# Agent operations

Read [CONTEXT.md](CONTEXT.md) for vocabulary and package responsibilities,
[architecture](docs/architecture.md) and [contracts](docs/api/contracts.md) for
maintained behavior, and [migration](docs/migration.md) for breaking Python changes.
Python naming is snake_case/PascalCase with the explicit exception **FOV stays FOV**.
MATLAB APIs and shared MATLAB-facing keys, filenames and coordinate contracts stay unchanged.

## Scope and ownership

Before work, read the live issue, dependencies and acceptance criteria, and the
[current initiative workflow](https://linear.app/jiahaoh/document/thesis-and-implementation-workflow-c12f30bffe9f).
Verify repository/worktree, branch, clean starting revision, inputs, environment
and resource limits. Record session/host/revision and a short plan in Linear;
assign Jiahao and require the relevant milestone. Unknown inputs remain unverified.
Preserve unrelated edits, including a dirty main checkout; never reset another
checkout or rewrite Git history.

Linear owns current scope, plans, decisions and validation evidence. Software Git
owns maintained code/docs/configs/tests, not diaries or benchmark run reports.
Scientific acceptance belongs to Chapter II/W-57 and W-92/W-93/W-94/W-124; software
checks do not qualify data, molecular truth or scientific claims. Scope changes
belong in the linked chapter discussion. Historical checklists do not authorize work.

When an automated runner dispatches a task, the task's own instructions govern
commits, Linear updates, checks and handoff; they take precedence over this file
where they differ. Otherwise follow the initiative workflow linked above.

Push, merge, deployment and publication require their own authorization.
`codex/docs-autonomous` publishes the documentation site on every push; do not
change deployment guards without that authorization.

## Implementation

Change only the authorized slice and maintained callers/tests/docs needed for
coherence. Verify diagnoses in actual code. Prefer functions and small typed
configs/results; avoid duplicated algorithms and compatibility shims for replaced
Python APIs. Keep arrays ZYX/ZYXC, explicit channel ordering and boundary conversions
as specified in the site contracts. Update notebooks' imports without executing
historical notebooks or rewriting outputs. Do not qualify process-dependent
synthetic hash seeds without explicit scope. Preserve historical source before extraction.

## Environment and validation

From `src/python`, run Python with `uv run python`. On GP099-29C use
`/home/unix/jiahao/.local/bin/uv` and
`UV_PYTHON=/home/unix/jiahao/miniforge3/bin/python3.12`; system Python 3.8 is unsuitable.
Use `PYTHONDONTWRITEBYTECODE=1 UV_LOCKED=true UV_NO_SYNC=true` with the prepared
locked environment. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` and `NUMEXPR_NUM_THREADS` to 1,
and `CUDA_VISIBLE_DEVICES=""`. Keep TMPDIR on local disk, never on
the SMB artifact mount, because tests create symlinks. Use a writable uv cache
when the default cache is sandboxed. Do not silently alter dependencies or lock.
If editable metadata requires reinstalling, record
`uv sync --locked --group docs --extra local-registration` and verify extras.

Run focused checks, then the required gates before committing:

```bash
uv run pytest test/ -v
uv run --group docs sphinx-build -n -W --keep-going -b html ../../docs /external/new-run/html
uv run python ../../docs/check_reference.py
```

Run affected bounded examples from [contributing](docs/contributing.md). Generated
API stubs are ignored/disposable; edit authored lists/docstrings. Never weaken
assertions, hide warnings/skips or claim an unexecuted backend passed. SimpleITK
must be installed for its execution coverage; MATLAB execution is excluded.

Default resource limits: one CPU, one numerical thread, no GPU; 1800 s per check;
target process RSS ≤4 GiB, new artifacts ≤1 GiB.
New images ≤32×64×64, ≤4 channels/rounds. Existing exceptions: small synthetic
test dataset 16×256×256 (generated once per pytest session), pointset 16×128×128,
compression 10×128×128, tiny examples 8×128×128, benchmark allocation 256×1024
float32. Tests generate synthetic images in session; do not commit generated TIFFs.
No scientific sweeps, large generation, training, real-image processing or cloud jobs.
Measure with `/usr/bin/time -v`; maximum RSS is KiB. CPU affinity/thread settings
and timeouts are controls, not proof of an enforced cgroup RSS cap.
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
evidence review; verify assignee/milestone/status and milestone progress.
Implementation Done is neither human nor scientific acceptance. Review history for
commit style: numbered messages for major modules, otherwise `prefix(info): message`.
Update live workflow guidance in place for agreed substantive practice changes,
with an administrative issue, dated rationale and synchronized affected instructions.
