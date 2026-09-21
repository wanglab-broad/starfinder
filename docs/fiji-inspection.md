# Inspect the saved synthetic example in Fiji

The version 2 [saved example](contributing.md) exports three float32 ImageJ TIFF
hyperstacks per case, alongside canonical prepared/registered HDF5 checkpoints.
The original array layout and checkpoint semantics are unchanged. Round order is
`round10, round2, round1`; channel order is `ch02, ch00, ch03, ch01`.

## Open TIFF directly

Use File → Open on `z9/inspection/round10.tif` (or another saved round/case).
The stored layout is ZCYX, converted from ZYXC by transpose `(0,3,1,2)` with no
normalization or quantization. Expect width/height 32, channels 4, Z slices 9
(or 1 for `z1`) and one frame per file. Every plane label records the sequencing
round, channel and zero-based Z. ImageJ C/Z selectors are one-based.
Physical spacing is unknown; the default pixel grid is not measured calibration.

## Open canonical HDF5

With [HDF5_Vibez](https://github.com/fiji/HDF5_Vibez#load-data-sets) installed,
use File → Import → HDF5 on `z9/registered/images.h5`. Select only image datasets:

| HDF5 dataset | Sequencing round |
| --- | --- |
| `/layers/layer0000/image` | round10 |
| `/layers/layer0001/image` | round2 |
| `/layers/layer0002/image` | round1 |

Choose individual hyperstacks (custom layout) and enter `zyxc`. This directly
maps the canonical dimensions to ImageJ without resampling. Do not select
metadata or transform datasets as images. Each file's `artifact.json` remains
the authoritative checkpoint metadata; `inspection/mapping.json` is a small
human inspection map.

HDF5_Vibez 1.1.1 assumes 1×1×1 µm when `element_size_um` is absent. That assumption
is **not valid calibration for these fixtures**. The script below explicitly
replaces it with a unit pixel grid. For manual imports use Image → Properties:
set unit `pixel`, widths/heights/depths 1, and interpret coordinates as indices.
Use a display range of 0–8 for every channel. The center Z is ImageJ slice 5 for
3D and slice 1 for singleton Z. Spot-A/B centers are XY `(10,10)`/`(22,22)`.

## Reopen and verify using Fiji

Open `docs/examples/inspect_saved_synthetic_fiji.py` in Fiji's Script Editor,
select Python, and run. Choose the delivery directory and enable “Show images”
to keep all round/format windows open with explicit plane labels. The script
uses the plugin's custom-layout reader and ImageJ's TIFF reader, checks all
voxels against each other and independent Gaussian expectations, and records
versions, dimensions, labels and numeric results in `fiji-verification.json`.
It also saves the active spot-A center plane as PNG from each reopened stack.
Interactive “Show images” mode reads without writing artifacts. Headless viewer
evidence is protected; use a fresh delivery directory for another recorded run.

For a reproducible headless Fiji application check (not a Python format-only
check), with the existing Fiji installation:

```bash
/path/to/Fiji.app/ImageJ-linux64 --headless --console --mem=1024m \
  --run /checkout/docs/examples/inspect_saved_synthetic_fiji.py \
  'delivery="/external/new-run/delivery",show_images=false'
```

Record Fiji/plugin/Java versions and plugin hashes with the output. No install
or update is implicit in this command. The supplied script was exercised with
Fiji/ImageJ 2.14.0/1.54f, HDF5_Vibez 1.1.1; version-specific import behavior must
be checked again when the runtime changes. Headless reopening establishes
application reader compatibility and values; it does not test GUI interactions.
