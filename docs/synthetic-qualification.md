# Independent development qualification

The qualification example audits the saved `controlled-development-v1` package
against the [frozen specification](synthetic-specification.md) and published
[preset constants](development-presets.md). It is independent software evidence,
not calibrated D04, historical v1/v2 provenance, benchmark eligibility or
scientific acceptance. W-93/W-57 retain those responsibilities.

From `src/python`, use the prepared locked environment and an existing package:

```bash
uv run python ../../docs/examples/qualify_synthetic.py audit /external/presets /external/new-run/qualification.json
PYTHONHASHSEED=1 uv run python ../../docs/examples/qualify_synthetic.py repeat /external/presets /external/new-run/repeat-1.json
PYTHONHASHSEED=999 uv run python ../../docs/examples/qualify_synthetic.py repeat /external/presets /external/new-run/repeat-999.json
uv run pytest test/test_synthetic_qualification.py -v
```

`audit` checks the input checksum manifest and all 63 cases without importing
the production renderer, geometry, observation or preset helper. It independently
computes full-grid Gaussian values, amplitudes, transient/persistent histories,
positions and visibility. The combined background inverse uses scalar bisection
for the single-control preset, independently of production fixed-point iteration.
Noise/texture/placement use the frozen descriptor arithmetic from the independent
specification example. These checks deliberately pin v1 constants; a new preset
version requires a reviewed oracle update, not a tolerance increase.

Truth amplitudes and flags are exact. Position tolerance is 1e-12 voxel indices;
inverse residual and float64 combined-image tolerance are 2e-10. Saved float32
image error is bounded by `max(1e-6, abs(oracle)*eps32/2 + 2e-10)` with relative
tolerance zero: the second term accounts for final rounding at intensities above
the original amplitude-8 literal examples. Persisted byte/dtype/config comparisons
remain exact. `repeat` regenerates only in memory, checks every saved image,
truth table, signal tensor and configuration exactly, and records process identity.
It tests reserved split descriptors and rejects calibration/evaluation generation;
it never generates held-out data. NumPy-version portability is not implied.

The edge regression adds empty, overlapping, out-of-frame and no-support objects
in Z=1 and 3D, with dropout followed by persistent loss. Complete truth populations
remain distinct from emission, visibility, detection and accepted molecules.
Existing focused contract tests cover malformed parameters, count limits, geometry
rejection, component isolation and corruption paths; run them with the qualification
tests when qualifying a release. Saved integration tests independently exercise
registration, detection, extraction, decoding/filtering and source-trace reload.

## Offline review rendering

`qualification_report.py` reads a supplied `review-context.json`, qualification
records, two process-repeat records, and pinned preset/processing manifests. It
reuses `starfinder.reporting` and the accepted saved-example image/decoding helpers.
It performs no image processing or generation. Context supplies criterion mapping,
validated command results, parameter tables, input identities, issue dependencies,
recorded intervals and unique-session usage coverage. Missing evidence must be
explicitly unavailable; a renderer cannot establish qualification or human approval.

```bash
uv run python ../../docs/examples/qualification_report.py /external/new-run /external/new-run/review.html
```

The context defaults to `phase: "implementation"`, which visibly retains pending
validation and uncommitted wording. After validation and the authorized local
commit, create a new context with `phase: "final"`, `code.commit` set to the full
40-character revision, `code.dirty: false`, and `controller_checks` containing the
three successful pytest, strict Sphinx and reference records. Each record includes
`command` (an argument list), `exit_code`, `log`, and its `sha256`. Verify those log
hashes and committed source identities before rendering. The renderer rejects
missing/failed gates or a dirty final revision, and derives the opening, acceptance
table, check table and appendix delivery state from this context. It does not
verify Git or execute controller commands. W-174 human approval stays pending.
Update the context's source hashes, intervals (including repair/revalidation),
session coverage and other evidence to the same snapshot; preserve earlier
contexts separately. Browser-open and hash the new final report after rendering.

Use a fresh output directory for each report revision; existing report and summary
destinations are refused. Essential images/tables are embedded. Copy the single
HTML for offline reading; companion summaries and HDF5/Parquet/TIFF links require
their saved files. The [Fiji recipe](fiji-inspection.md) preserves ZYXC/ZCYX order
and removes invented physical calibration. Actual browser/Fiji execution and
screenshots belong in the external delivery evidence, not a report-generation
claim. Report/manifest hashes live outside the report to avoid self-reference.

Keep histories and prior packets; show successful and partial/failed diagnostics
separately. Count cumulative usage only once per unique session after verifying
counter semantics; unknown sessions stay unknown, and tokens do not imply cost.
Human waits are gate-open intervals, not measured human effort. Implementation,
validation and repair intervals may overlap and must not be summed blindly.
