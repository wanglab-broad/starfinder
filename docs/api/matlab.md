# MATLAB API

The reference parses help comments and signatures directly from project-owned
`src/matlab/*.m` files using `sphinxcontrib-matlabdomain`. It does not execute
MATLAB. The class coordinates a single field of view; the functions below are
its processing and output helpers. Third-party code in `src/matlab-addon/` and
workflow entry scripts are not included as public API objects.

```{toctree}
:maxdepth: 1

matlab/dataset
matlab/io
matlab/registration
matlab/reads
matlab/preprocessing
matlab/visualization
matlab/utilities
```

## Array and calling conventions

* MATLAB images have shape **(row, column, Z, channel)**, or **(Y, X, Z, C)**.
  Names such as `dimX` in older source often mean the first array dimension
  (rows), not Cartesian X. Python uses **(Z, Y, X, C)**; exchanging arrays needs
  an explicit permutation.
* Spot tables use **1-based** `x` (column), `y` (row), `z` (plane). Extraction
  indexes images as `(y, x, z, :)`. The extraction `voxel_size` is a vector of
  half-widths in **(row, column, Z)** order, not physical voxel spacing; interior
  windows have size `2 * voxel_size + 1` and are clipped at image boundaries.
  Python coordinates are zero-based internally; its CSV writer adds one.
* Default sequencing channels are wavelength-sorted:
  **ch00, ch02, ch01, ch03** (488, 546, 594, 647 nm). Color labels `1`–`4`
  refer to this ordering. Do not silently substitute filename sort order.
* `STARMapDataset` is a **value class**: retain the result of each method,
  for example `sdata = sdata.MakeProjection()`. Round names key its image and
  metadata dictionaries. Cell-array slices of these dictionaries are passed to
  helpers such as `MinMaxNorm`; these helpers do not take a bare image array.
* Read each method's option list carefully. The source mixes `addOptional`
  (positional arguments) and `addParameter` (name/value pairs), while workflow
  scripts often pass both as name/value pairs. Source parsing does not validate
  those calls. Validate a small dataset on your MATLAB installation before
  scheduling production work.

## Supported workflow entry points

MATLAB processing is invoked through Snakemake, not a standalone STARfinder
MATLAB CLI. At the repository root, start with a copy of
[`tests/tissue_2D_test.yaml`](https://github.com/wanglab-broad/starfinder/blob/dev/tests/tissue_2D_test.yaml),
edit dataset and checkout paths (including `config_path`), select a small FOV
subset, and explicitly set `backend: matlab`. Then inspect the job graph:

```bash
snakemake -s workflow/Snakefile --configfile /absolute/path/my-config.yaml -n
```

On a configured Broad UGER host, the execution form is:

```bash
snakemake -s workflow/Snakefile --configfile /absolute/path/my-config.yaml \
  --profile profile/broad-uger --workflow-profile profile/broad-uger
```

These templates require real paths and data; the documentation build does not
run them. `rsf_preparation` converts the YAML to JSON for MATLAB.
[`run_matlab_scripts`](https://github.com/wanglab-broad/starfinder/blob/dev/workflow/rules/common.smk)
adds `workflow/scripts` to MATLAB's path, sources
`/broad/software/scripts/useuse`, selects `use Matlab`, and calls MATLAB with
`-nodisplay -nosplash -nodesktop -r`. Each entry script adds `src/matlab` and
`src/matlab-addon` and receives the JSON config path plus FOV/subtile identifiers.
An installed `matlab` binary alone does not supply that Broad environment.
See also the [workflow guide](../workflows.md) and
[existing workflow examples](https://github.com/wanglab-broad/starfinder/blob/dev/example/README.md).

| Workflow mode | Rules / same-named MATLAB scripts | Processing boundary |
| --- | --- | --- |
| `direct` | `rsf_single_fov` | Load, rotate/flip, enhance, register, find spots, extract, filter, save |
| `subtile` | `gr_single_fov_subtile`, `lrsf_single_fov_subtile`, then Python `stitch_subtile` | Global registration and `.mat` subtile creation; local registration and reads per subtile; merge coordinates |
| `deep` | `deep_create_subtile`, `deep_rsf_subtile`, then Python `stitch_subtile` | Split first, then registration and reads per subtile |
| `free` | Individual rule `run` flags; also `rsf_single_fov_seq` | Configured combination; sequence-only extraction is a separate entry point |

## Stage and Python cross-references

These links identify corresponding responsibilities, **not numerical
equivalence**. Thresholding, interpolation, normalization, dtype conversion,
registration signs and filtering differ. No cross-backend numerical validation
is implied by this reference.

| Stage | MATLAB interface | Python operation |
| --- | --- | --- |
| Orchestration | {mat:class}`STARMapDataset` | {py:class}`starfinder.dataset.Dataset`, {py:class}`starfinder.dataset.FOV` |
| Load / output | {mat:func}`LoadImageStacks`, {mat:func}`LoadMultipageTiff`, {mat:func}`SaveSingleStack` | {py:func}`starfinder.io.load_round`, {py:func}`starfinder.io.load_volume`, {py:func}`starfinder.io.save_volume` |
| Enhance | {mat:func}`MinMaxNorm`, {mat:func}`MorphologicalReconstruction` | {py:func}`starfinder.preprocessing.normalize_intensity`, {py:func}`starfinder.preprocessing.reconstruct_background` |
| Histogram / background | {mat:meth}`STARMapDataset.HistEqualize`, {mat:meth}`STARMapDataset.Tophat` | {py:func}`starfinder.preprocessing.match_histogram`, {py:func}`starfinder.preprocessing.filter_tophat` |
| Global registration | {mat:func}`DFTRegister3D`, {mat:func}`DFTApply3D` | {py:func}`starfinder.registration.estimate_transform`, {py:func}`starfinder.registration.apply_transform` |
| Local registration | {mat:func}`RegisterImagesLocal` | {py:func}`starfinder.registration.estimate_transform` |
| Spot finding | {mat:func}`SpotFindingMax3D` | {py:func}`starfinder.spot_finding.find_spots` |
| Extraction | {mat:func}`ExtractFromLocation` | {py:func}`starfinder.barcode.extract_intensities` |
| Codebook / decoding | {mat:func}`LoadCodebook`, {mat:func}`EncodeBases`, {mat:func}`DecodeCS` | {py:func}`starfinder.barcode.load_codebook`, {py:func}`starfinder.barcode.encode_bases`, {py:func}`starfinder.barcode.decode_color_sequence` |
| Filtration | {mat:func}`FilterReads`, {mat:func}`FilterReadsMultiSegment` | {py:func}`starfinder.barcode.filter_reads` |
| Preview | {mat:func}`MakeProjections`, {mat:func}`MakeMontage`, {mat:func}`PlotCentroids` | [FOV output and preview methods](generated/starfinder.dataset.FOV.rst); no direct exported montage helper |

MATLAB passes the correction parameters from `DFTRegister3D` directly to
`DFTApply3D` in `(row, column, Z)` order. Python `estimate_transform` returns the correction in `(dz, dy, dx)` order; `apply_transform` applies it directly to correct alignment. Never transfer a shift vector without checking
both axis order and sign. See [Python contracts](contracts.md).

## Runtime requirements and validation limits

The repository targets **MATLAB R2023b or newer**. It uses `dictionary` and modern
string/table APIs. Image Processing Toolbox supplies operations including
`imregdemons`, `imwarp`, `imregionalmax`, `regionprops3`, `imadjustn`, morphology,
and montage rendering. Parallel Computing Toolbox and a compatible GPU are
needed only for GPU-specific use; the workflow scripts construct CPU datasets,
and setting the stored `useGPU` flag does not automatically accelerate them.
`SaveImageSingleFolder` uses the bundled `saveastiff` helper. Preview/export paths
also require MATLAB graphics support on a headless host.

Snakemake 9.x and the Broad cluster-generic executor environment are separate
from MATLAB; see `config/environment-v9.yaml` and `profile/broad-uger/` in the
checkout. This API reference neither installs nor tests those runtimes.

The documentation dependency group includes the MATLAB domain extension and its
parser dependencies. Sphinx reads `.m` files without a MATLAB license. Successful
strict builds validate parsing, rendering, and cross-references only. MATLAB
runtime examples and production Snakemake commands have **not** been executed
as part of this reference's validation. Toolbox availability, license checkout,
graphics behavior and backend agreement remain unverified. Source-visible
limitations are called out on the relevant pages; no algorithms are changed.
