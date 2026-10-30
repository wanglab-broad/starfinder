# Agent operations

Read [CONTEXT.md](CONTEXT.md) for vocabulary and package responsibilities,
[architecture](docs/architecture.md) and [contracts](docs/api/contracts.md) for
maintained behavior, and [migration](docs/migration.md) for breaking Python changes.
Python naming is snake_case/PascalCase with the explicit exception **FOV stays FOV**.
MATLAB APIs, and the files, keys and coordinate conventions that MATLAB code reads or
writes, stay unchanged. Otherwise the Python package follows its own design: MATLAB
compatibility alone is no reason to keep a Python name, key, layout or format
(W-526, 2026-10-19).

Specifications, validation designs, baselines and proposals are design records in
[design/](design/README.md), which the documentation site does not build. `docs/` holds
the site's short guides and references, written from the implemented code once an
implementation lands; never publish a specification there (W-639, 2026-10-27).

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
Scientific acceptance belongs to the Chapter II project and its linked evidence and
benchmark issues in Linear; software checks do not qualify data, molecular truth or
scientific claims. Scope changes
belong in the linked chapter discussion. Historical checklists do not authorize work.

When an automated runner dispatches a task, the task's own instructions govern
commits, Linear updates, checks and handoff; they take precedence over this file
where they differ. Otherwise follow the initiative workflow linked above.

Push, merge, deployment and publication require their own authorization.
`codex/docs-autonomous` publishes the documentation site on every push; do not
change deployment guards without that authorization.

## Implementation

Every config dataclass requires a complete numpydoc `Parameters` section with an entry for every field; the strict documentation build enforces completeness for every class shown by `config-options`.

Change only the authorized slice and maintained callers/tests/docs needed for
coherence. Verify diagnoses in actual code. Prefer functions and small typed
configs/results; avoid duplicated algorithms and compatibility shims for replaced
Python APIs. Keep arrays ZYX/ZYXC, explicit channel ordering and boundary conversions
as specified in the site contracts. The two notebooks in `example/introduction/`,
`single_fov_demo.ipynb` and `sample_demo.ipynb`, are the Get Started demos (W-587):
executed, outputs kept (at most 1.5 MB each), within 2 minutes and 2 GiB on one CPU,
classical methods and public API only. After changing one, execute it again and run
`docs/examples/render_demos.py` to regenerate its page under `docs/get-started/`.
For other notebooks, update imports only, without executing them or rewriting
outputs. Do not qualify process-dependent synthetic hash seeds without explicit
scope. Preserve historical source before extraction.

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

Run focused checks, then the gates of each change before committing: the quick
gate Q, `not (slow or learned or extended) and (contract or not validation)`, and
the reference audit.

```bash
uv run pytest test/ -v -m "not (slow or learned or extended) and (contract or not validation)"  # Q
uv run python ../../docs/check_reference.py
```

The other selections, defined in [contributing](docs/contributing.md) ("Test
selections"), run on these occasions:

* V, `validation and not contract and not (slow or learned or extended)`: at a
  batch's end and before merge.
* R = Q ∪ V, the fast tier, `not (slow or learned or extended)`: for a change to
  a shared path (`image`, `io`, `dataset`, `conftest.py`, `pyproject.toml`,
  `uv.lock`), at a batch's canary, at a batch's end and before merge.
* Slow, `slow and not (learned or extended) and <subsystem>`: the subsystems the
  change touches, and every subsystem at a batch's end.
* Learned, `learned`: when a dependency of the learned tests changes, and at a
  batch's end if one changed.
* Extended, `extended and not learned`: at a batch's end and before merge.
* Documentation: the reference audit on every change; the strict build
  (`uv run --group docs sphinx-build -n -W --keep-going -b html ../../docs
  /external/new-run/html`) at a batch's end, before merge, and on a change to
  `docs/` or to a public interface.

R, slow, learned and extended together hold every collected test once
(`scripts/selection_coverage.py`). Every test file carries a subsystem marker;
mark new tests as [contributing](docs/contributing.md) specifies.

Run affected bounded examples from [contributing](docs/contributing.md). Generated
API stubs are ignored/disposable; edit authored lists/docstrings. Keep an
assertion exact (bit equality, or a digest of computed values) only where a
contract requires identity: file, record and export hashes; checkpoint and
old-format round trips; model and weight identity; stream descriptors;
same-environment repeatability; and integer or discrete outputs. State other
numerical expectations as explicit values with the tolerances of
[contributing](docs/contributing.md) ("Expected values and tolerances"). Never
widen a tolerance or change an expected value to make a result pass, hide
warnings/skips, or claim an unexecuted backend passed. SimpleITK
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
