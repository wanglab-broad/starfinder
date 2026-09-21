# Offline run summaries and checkpoint inspection

`starfinder.reporting.write_run_summary(run_path, destination)` validates saved
provenance and checkpoint components, then writes standalone HTML. It does not
rerun processing. Optional `sha256=` pins the run manifest. Existing reports are
never overwritten; missing/corrupt components fail before publication.

```python
from starfinder.reporting import write_run_summary
write_run_summary('/external/run/run.json', '/external/review/summary.html')
```

Reports include requested/effective configuration, actual events and methods,
failures/recovery, geometry/transforms, stage and artifact states, pre-QC/final
populations, source identities, environment and retention. Run success describes
the requested pipeline only. Omitted checkpoints differ from failed or partial
stages; absent counts remain unavailable. The table/figure helpers also serve the
saved synthetic review. Population tables show at most 200 rows, with full counts
and source links; readers still load whole-FOV checkpoints. Dense arrays show shape/dtype/hash instead of all values.

## Bounded examples

From `src/python`, in the prepared environment, use a fresh external directory:

```bash
uv run python ../../docs/examples/run_summaries.py prepare /external/new-run/summary-example
uv run python ../../docs/examples/run_summaries.py render /external/new-run/summary-example
uv run pytest test/test_run_summaries.py test/test_saved_synthetic_example.py -v
```

Preparation reuses saved-formed-v3: two objects, three rounds, four channels,
`(9,32,32)` / `(1,32,32)` ZYX. A deliberate TPS failure has insufficient landmarks,
failed registration, partial extraction and unavailable decoded counts. Separate
rendering verifies input hashes and writes `z9-summary.html`, `z1-summary.html`,
`failed-summary.html` and `inspection.html`. No historical inputs are used.

Open HTML directly in a browser, or copy it for offline review; figures/tables
are embedded. Raw source links require original files/mounts. Copying HTML does
not relocate checkpoint references; absolute molecular source references must
remain reachable. Private paths do not establish public reproducibility.
Owner Jiahao; retain through thesis/publication; backup unverified. Actual browser
opening is a separate per-delivery check.

## Notebook and deeper saved-output reading

Open {download}`checkpoint_inspection.ipynb <examples/checkpoint_inspection.ipynb>` with a
kernel started from `src/python`; set `STARFINDER_INSPECTION_ROOT` to the example
output directory. Cells read images, transforms, candidate traces, full truth
histories, decoded pre-QC and final filtering. The helper
[checkpoint_inspection.py](examples/checkpoint_inspection.py) reuses W-175 XY/XZ
views and explicit truth/detection correspondence. Prior W-175 deliveries are
also supported: missing molecular checkpoints are labeled unavailable and saved
comparison tables are identified as example tables.

Jupyter is optional and requires separate environment preparation. Focused tests
execute ordinary Python cells, not a live kernel. Cells write a fresh
`notebook-inspection.html`; choose a new output name for a second execution.
No historical notebook is executed or rewritten.

Coordinates remain zero-based ZYX indices; calibration remains unknown. Detector
`spot-1/spot-2` and truth `gt-A/gt-B` use explicit correspondence, never row order.
Observed colors, nucleotide decoding, gene assignment and filtering stay separate.
Development software checks do not supply calibrated truth, validated real-data
performance, scientific acceptance or human-gate approval.
