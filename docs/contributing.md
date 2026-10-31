# Contributing and local preview

Build from `src/python` with Python 3.12 and the locked environment. The
[installation guide](get-started/installation.md) covers package setup.
Keep code, docstrings, public API lists and examples together when an interface changes.

## Quick start

```bash
uv run --group docs sphinx-build -n -W --keep-going -b html ../../docs /tmp/starfinder-docs-html
uv run python ../../docs/check_reference.py
```

Choose a fresh output directory outside the checkout. The strict build checks
internal references; the reference audit checks exports and navigation.
Generated API stubs in `docs/api/generated/` are disposable: edit `__all__`,
authored autosummary lists and implementation docstrings instead.
For a preview, serve the HTML directory with Python's `http.server` and open
its localhost URL. Check navigation, search and API source links.

## Documentation pages

A guide is about 1,500 words at most. Where applicable, use these sections in
order: a short lead, Quick start, What it does, Methods or Presets, What you get,
Main options, Limits, How it is checked, then a link to its specification.
Show a minimal public-API example. Explain choices in plain words; link precise
rules to the API or specification.

A reference holds lookup tables with brief introductions. Generate tables from
code where possible using the directives in `docs/_ext/`: `method-table`,
`config-options` and `synthetic-presets`. Link config classes to their API
pages for further details rather than copying parameter descriptions.

Every config dataclass and every public function, method and dataclass on the
API pages requires a complete numpydoc `Parameters` section, with an entry for
every public field or parameter (excluding `self` and `cls`): what it sets, its
unit and simple valid values, in one or two plain sentences. Variadic parameters
keep their stars. Add a `Returns` or `Yields` section when the callable returns a
value. Keep precise rules in the specification and the first docstring line
unchanged. `config-options` reads these descriptions; the strict build rejects
missing, unknown, duplicate or empty entries for its targets and API objects,
and requires a return section when an API callable's annotation denotes a value.
Unannotated callables have no automated return check. Exceptions, warnings,
enums, constants and private dataclass fields are exempt. The quick-gate contract
test checks all public config dataclasses, every `config-options` target and
every API object, including public methods defined by listed classes.

Specifications, validation designs, baselines and proposals are design records
in [design/](https://github.com/wanglab-broad/starfinder/blob/dev/design/README.md),
never site pages. An implementation batch's last issue writes the section's
guide and reference from the implemented code. Check facts against source and
run examples on small inputs. Tests read labelled tables in design records
with `docs_pages.design("path/under/design.md")`; keep their labels and row shapes.

## Test markers

Markers are registered in `src/python/pyproject.toml`.

* Set a subsystem marker on the file with `pytestmark`; use `integration` for
  behavior across three or more subsystems. Prefer an existing test module;
  name a new one `test_<subsystem>_<topic>.py`.
* Use `contract` for interfaces, dtypes, errors, registries and workflow keys;
  `golden` for pinned outputs; `validation` for checks in design tables;
  `e2e` for the full pipeline. A validation of a public interface also gets
  `contract`. Unmarked kinds are ordinary unit tests.
* Use `learned` for tests loading detector models or needing
  `STARFINDER_WEIGHTS_DIR`.
* Mark a test `slow` when it takes at least 5 seconds on one CPU and one
  numerical thread in the locked environment, excluding one-time imports.
  Include cached setup cost paid by the first selected user. Measure with
  `--durations=0`; mark individual slow parameters when appropriate.
* Put page-reading tests in docs-contract modules without slow, learned or
  extended tests, and mark them `contract`. Those modules are not imported by
  expensive tests. Find parsed tables by MyST labels with `test/docs_pages.py`,
  rather than headings. Use a shared helper without page paths when needed.
  Executable files under `docs/examples/` remain check inputs.

## Expected values and tolerances

Keep exact assertions where the contract requires identity: file, record, export
and run hashes; checkpoint and old-format round trips; models and weights;
stream descriptors; OME bytes and content-derived UUIDs; same-environment
repeatability; integer and discrete outputs. Replace historical numerical pins
with explicit expected values and tolerances. Use analytic formulas where
available; remove obsolete source locks while retaining behavioral checks.

| Quantity | Comparison |
| --- | --- |
| integers, counts, labels, integer images, strings, statuses, booleans | exact |
| floats that hold integer values (sums of integer pixels, integer-valued coordinates and peaks) | exact |
| float64 computed values (scores, thresholds, quantiles, transforms, displacement fields, random draws through `exp` or `log`) | `rtol=1e-9`, `atol=1e-12` |
| float32 computed values | `rtol=1e-6`, `atol=1e-6` |
| physical coordinates and lengths with a declared grid precision | the declared precision (1e-9 µm) |
| solver guarantees and analytic bounds that a contract states | the stated bound, unchanged |
| analytic expectations (a formula the test evaluates) | the tolerance of the computed quantity's dtype |

Use the compared result's dtype. `assert_allclose` checks
`|actual - expected| <= atol + rtol * |expected|`; give both `rel` and `abs`
with `pytest.approx`. These measure software precision, not scientific accuracy.
Do not widen tolerances or change expectations to make a test pass.

Write about 20 expected values or fewer in the test. Larger tables and arrays
go under `test/data/expected/<test module>/` as CSV or NPZ with a README naming
the generating revision and command; state per-column tolerances in the test.
Before replacing a digest, save a base-revision comparison showing the new
expectations reproduce the old digest through the old encoding. Do not take
expectations from changed code. A negative control compares changed output
with the explicit expectation and checks that it lies outside tolerance.

## Test selections

Run from `src/python` with the `dev` dependency group installed. A command-line
`-m` replaces pytest's default exclusion of `extended`.

| Selection | Marker expression | Cadence |
| --- | --- | --- |
| Q | `not (slow or learned or extended) and (contract or not validation)` | every change |
| V | `validation and not contract and not (slow or learned or extended)` | batch end and before merge |
| R = Q ∪ V | `not (slow or learned or extended)` | shared paths, batch canary, batch end and before merge |
| Slow | `slow and not (learned or extended) and <subsystem>` | affected subsystems; all at batch end |
| Learned | `learned` | changed dependencies; batch end if changed |
| Extended | `extended and not learned` | batch end and before merge |
| Documentation | reference audit and strict Sphinx build | audit every change; build for docs/public interfaces, batch end and before merge |

Shared paths are `image`, `io`, `dataset`, `conftest.py`, `pyproject.toml` and
`uv.lock`. The slow groups are benchmark; synthetic excluding benchmark;
registration; spot_finding; and other excluding those four.
`scripts/selection_coverage.py` checks that R, slow, learned and extended
cover collected tests once. `scripts/marker_drift.py` reads a durations log to
review cost markers. Review new and changed test markers before merging.

The selections are independent: they may run at the same time when each has its
own core and temporary directory.

## Limits

A local HTML build does not execute MATLAB or optional algorithms. Record
unavailable checks in the handoff. Publication is a separate authorized action;
the [release instructions](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/contributing.md#publish-and-verify-a-release)
explain the deployment workflow.
