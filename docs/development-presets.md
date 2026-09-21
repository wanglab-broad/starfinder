# Controlled development presets

`synthetic.development_scene_preset(condition, size="small")` packages the existing
formed-scene generator, with no new numerical model. Version
`controlled-development-v1` is **development-only**, not calibrated evaluation
data, original RNA, cells, empirical tissue or a registration benchmark.
It follows the [frozen specification](synthetic-specification.md), revision
`f9512694a0960c10ce5236efbaaf9d6f425c1d8a`, generator version 4.

## Fixed scene and named controls

Every condition within a size uses root seed 42, scene key
`controlled-development-v1`, development split, sample `sample`, FOV `FOV_001`.
Dataset namespaces are `controlled-development-v1-{size}-{condition}`; changed
namespaces/frame IDs identify distinct artifacts and do not enter random keys.
Compare by explicit `gt-A`/`gt-B` correspondence across these namespaces.
These IDs never designate detections; this example performs no detection,
extraction, nucleotide decoding, gene assignment or filtering.

Clean positions are `(Z//2,10,10)` and `(Z//2,22,22)`, gene-A=`123` and
gene-B=`214`, A=8, axial sigma=1, lateral sigma=1.25, elongation=1, angle=0.
Rounds are `(round10,round2,round1)`, channels `(ch02,ch00,ch03,ch01)`, with
color mapping `1→1, 2→0, 3→3, 4→2`. Calibration is unknown, lengths are voxel
indices. Images are float32 ZYXC, signal truth is float64 NCR.

| Size | ZYX | Interpretation |
| --- | --- | --- |
| z1 | (1,32,32) | Same 3D kernel sampled at z=0, with axial truncation, not projection |
| small | (9,32,32) | Two-object compact 3D comparison |
| wide | (9,48,48) | Larger bounded field, same supplied molecular centers |

All 21 conditions are supported at each of these three sizes. Sizes are distinct
comparisons: random texture placement scales with extent and noise draw layout
changes; matching overlapping voxel noise across sizes is **not** promised.
At Z=1, changing axial width changes truth but not a kernel sampled at its
center plane (dz=0); this is an explicit image-invariant comparison.
No historical tiny/small fixture or TIFF is replaced or regenerated.

| Individual condition | Only changed component relative to clean |
| --- | --- |
| brightness | A=16 |
| axial_width | sigma Z=1.5 |
| lateral_width | sigma YX=2 |
| elongation | e=2, angle remains 0 |
| placement | Two clustered positions, center `(Z//2,10,10)`, spread `(1,2,2)`; Z offset is identically 0 for Z=1 |
| dropout | Middle round probability 1, other rounds 0; recovery in last round |
| weakening | Middle round probability 1, factor .25; recovery in last round |
| trend | b=.5, absolute factors (1,.5,.25) |
| loss | All objects lost persistently from round index 1; full truth retained |
| gain | Source gains .5 for all round/channel pairs |
| mixing | Identity plus M[0,1]=.25, M[2,3]=.25, destination rows/source columns |
| baseline | Destination levels (1,2,3,4), all rounds |
| gradient | Intercept 1, normalized ZYX slopes (0,0,2) |
| regions | Gaussian at `(Z//2,10,10)`, widths (2,5,5), height 3 |
| texture | Two persistent uniform blobs, widths (1,3,3), height 5 |
| dependent_noise | alpha=.25 |
| independent_noise | sigma=.5 |
| translation | Absolute vectors (0,0,0), (0,.5,-.5), (0,-1,1) |
| local | One control at `(Z//2,10,10)`, scale 8; vectors (0,0,0), (0,0,.5), (0,.25,0) |

Structured components use destination weights (1,.5,.25,0) in all rounds.
All controls retain the same requested parameters when disabled, but disabled
controls are algebraic identities. `development_preset_factors(condition)` names
exactly the enabled controls. `combined` enables weakening, trend, gain, mixing,
baseline, gradient, regions, texture, both noise terms, translation and local
geometry. Appearance/placement stay clean; dropout/loss stay disabled so signal
and motion remain inspectable. This intentionally modest combination is not a
parameter sweep or a claim that every combination is scientifically realistic.

Order: persistent appearance/placement → intended channels → survival and
transient masks → trend → source gain → mixing → shared geometry → render
molecules and advect actual analytic background through the inverse map →
destination baseline → dependent noise → independent noise → float32 cast.
Molecule widths/angle stay fixed under motion; background moves through the same
map, while instrument baseline does not. Changing noise strength preserves
standardized draws and unrelated latents; dependent amplitude still responds to
upstream total intensity. No normalization or clipping of float images occurs.

## Save and inspect

In the prepared environment, from `src/python`, choose a fresh external folder:

```bash
uv run python ../../docs/examples/development_presets.py create /external/new-run/presets
uv run python ../../docs/examples/development_presets.py inspect /external/new-run/presets
uv run pytest test/test_development_presets.py -v
```

Create saves all 63 fixed cases sequentially through shared HDF5 image checkpoint
export, Parquet full formed/history tables, NPZ intended/pre-mix/realized tensors,
and JSON full codebook/config/stream/transform/background/visibility provenance.
It checks exact reloads against in-memory values and records per-case elapsed
time, stored bytes and cumulative process peak RSS (KiB). `manifest.json` pins
source/config/output hashes. Existing destinations are refused.

Inspect runs in a separate process using saved files only: it verifies hashes,
logical image values, axes and pinned configs, then uses shared reporting helpers
to write `inspection.html` and `inspection.json`. Open HTML locally or copy it
for offline inspection. Channel-specific middle-round center slices share display
range −1 to 16; cyan crosses/IDs mark independent truth even when emission is
absent. All history rows and full transforms/config remain available. Prepared
layers keep their distinct round frame IDs; they are not registered images.
For detailed array access use `load_image_checkpoint(case / "prepared").layers`;
use each layer's `loaded.image`, metadata and channel labels. Prepared rounds
with different frames cannot be passed as an already aligned extraction stack.

The report is an inspection example, not the W-167/W-174 qualification packet.
Use the [independent qualification example](synthetic-qualification.md) to audit
saved presets, verify fresh-process repeatability and assemble a review packet.
No browser, Fiji, MATLAB or scientific accuracy claim follows from creating it.
Jiahao owns artifacts through thesis/publication; backup/public reproducibility
remain unverified. Preserve earlier packets. Full qualification and human review
remain separate from software checks.
