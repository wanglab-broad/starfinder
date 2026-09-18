# Conventions

Use this checklist when moving data between Python, MATLAB and workflow YAML.
The [generated Python contracts](api/contracts.md) and
[MATLAB reference](api/matlab.md#array-and-calling-conventions) give further
operation-specific details. Corresponding APIs do not imply numerical parity.

## Axes and coordinates

| Quantity | Python | MATLAB / exported signal CSV |
| --- | --- | --- |
| Image | `(Z,Y,X,C)`; single channel `(Z,Y,X)` | MATLAB `(Y,X,Z,C)`, single channel `(Y,X,Z)` |
| Spot table | Named `z,y,x`, **0-based** voxel indices | Named `x,y,z`, **1-based**; x is column, y is row |
| Channel index | Table `channel` is 0-based | MATLAB array channel indices are 1-based |
| Color labels | Strings `1`–`4` for four channels | Strings `1`–`4`; labels are not zero-based channel indices |
| Extraction `voxel_size` | Half-widths `(dz,dy,dx)` | Half-widths `(row,column,Z)` |

For array exchange, `python_image.transpose(1,2,0,3)` produces MATLAB axis order;
`matlab_array.transpose(2,0,1,3)` restores Python order after loading into NumPy.
These are axis permutations, not coordinate-origin conversions. Confirm file
metadata first; do not use `reshape` to exchange axes. For a single channel,
the analogous permutations are `(1,2,0)` and `(2,0,1)`.

{py:meth}`~starfinder.dataset.FOV.save_signal` adds one to selected x/y/z columns
in a copy. In-memory `(z,y,x)=(5,10,10)` becomes CSV `(x,y,z)=(11,11,6)`.
To index a NumPy image from that CSV, select columns by name in ZYX order and
subtract one **once**, as in the [output recipe](recipes.md#inspect-molecule-outputs).
Do not subtract one from shift vectors, color labels or already internal
coordinates. Exports carry no physical calibration; distances are voxel indices.

Extraction `(1,2,2)` in Python corresponds to `(2,2,1)` in MATLAB, both describing
interior windows 3 Z × 5 Y × 5 X. Python zero-pads boundary windows; MATLAB clips
them. Workflow `voxel_size_xy` / `voxel_size_z` are separate physical quantities
for downstream stitching. See [configuration fields](workflow-configuration.md#top-level-fields).

## Channel order

The wavelength-sorted MATLAB default is **`ch00,ch02,ch01,ch03`**, so filename
`ch01` and `ch02` are swapped relative to natural sorting. The Python loader
uses exactly the provided pattern list; it does not infer wavelengths. Set
`channel_order` in the direct API or `seq_channel_order` in Python workflow YAML.
An empty Python list is not a request for the MATLAB default.

The tiny synthetic quickstart deliberately uses **`ch00,ch01,ch02,ch03`**.
Color `2` therefore selects ch01 there, but selects ch02 under the real-data
MATLAB ordering. Check acquisition and codebook metadata together before decoding.
A channel permutation can produce plausible intensities with incorrect genes.

Use unambiguous patterns: {py:func}`~starfinder.io.load_round` searches
`*{channel}*.tif` and `*{channel}*.tiff`, and rejects ambiguous matches. Channel
size mismatches error unless an explicit minimum crop is requested. Inspect
`ImageLoadResult.diagnostics` for `original_shapes` and `cropped`. MATLAB's loader expects a struct array for custom
channel settings; the schema's string list is not portable to that custom path.
See the [backend configuration caveat](workflow-configuration.md#top-level-fields).

## Registration signs

| Interface / stored value | Meaning and correction |
| --- | --- |
| Python {py:func}`~starfinder.registration.phase_correlate` | Detected displacement `(dz,dy,dx)`; negate it for {py:func}`~starfinder.registration.apply_shift` |
| Python {py:func}`~starfinder.registration.register_volume` and {py:meth}`~starfinder.dataset.FOV.global_registration` | Apply the negative internally; returned/stored shifts remain detected displacements |
| Python `global_shifts` / `log/gr_shifts/*.txt` | Internal ZYX vectors; log columns `row,col,z` contain detected `(dy,dx,dz)`, with no origin offset |
| MATLAB {mat:func}`DFTRegister3D` → {mat:func}`DFTApply3D` | Correction parameters in `(row,column,Z)` order; passed directly, without Python's negation |
| Python dense local displacement field | Backward sampling: `registered[p] = moving[p + field[p]]`; pass directly to its warp function |

For a moving volume displaced by `(1,-2,3)`, Python detects `(1,-2,3)` and applies
`(-1,2,-3)`. The [registration recipe](recipes.md#register-two-volumes) verifies
this with an interior spot. Positive `apply_shift` components move content
toward larger indices; wrapped edges are zeroed. Lost edge content cannot be
recovered. Dense fields have shape `(Z,Y,X,3)` with components `(dz,dy,dx)`;
they are not global translation arguments. MATLAB behavior here is source-checked,
not runtime-validated.

## Spot-finding thresholds

Always specify **`intensity_estimation` and `intensity_threshold` together** in
{py:func}`~starfinder.spotfinding.find_spots_3d`, FOV calls and workflow parameters.

| Python mode | Absolute cutoff | Example pair |
| --- | --- | --- |
| `noise` (default) | Channel median + k × MAD × 1.4826 | `noise`, `5.0` for the synthetic quickstart |
| `adaptive` | Channel maximum × fraction | `adaptive`, `0.2` as a fraction-of-max example |
| `adaptive_round` | Maximum across all channels × fraction | `adaptive_round`, `0.2` |
| `global` | Dtype maximum × fraction; uint8/uint16 only | `global`, `0.2` |

Examples explain units, not recommended parameters for every dataset.
`min_distance` is in voxel indices and also sets the excluded border width;
it does not account for physical anisotropy. All-zero channels yield no spots.

MATLAB {mat:func}`SpotFindingMax3D` supports adaptive/global, not Python's noise
or adaptive_round. The schema accepts `local`, but the Python detector rejects
it. MATLAB direct/local-subtile wrappers only forward the threshold; their mode
stays adaptive. The deep wrapper forwards the mode. Consult
[rule parameters and overrides](workflow-configuration.md#rule-resources-and-parameters)
before assuming a YAML setting reached the selected backend.
