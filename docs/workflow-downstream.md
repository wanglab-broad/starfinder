# Downstream segmentation and assignment

The sequencing endpoint is `signal/{fovID}_goodSpots.csv`. Producing a cell
expression matrix additionally requires aligned morphology images, segmentation
labels, codebook metadata and tile geometry. The rules below are included for
both backends. The segmentation and assignment scripts call the Starfinder package
under both backends, so the environment that runs Snakemake needs the package and the
extras named below, also for `backend: matlab`; the other downstream scripts keep
their own dependencies.

The contracts below were checked against `workflow/rules/segmentation.smk`,
`stitching.smk`, `reads-assignment.smk`, `utils.smk` and their scripts. They are
not an executed image-to-cell validation. Scientific validation on a public
image-to-molecule-to-cell example is still pending.
The [downstream examples](https://github.com/wanglab-broad/starfinder/blob/dev/example/downstream/README.md)
remain an additional entry point.

## Morphology and labels

All paths in this table are relative to OUTPUT unless marked INPUT. Each rule
uses `workflow/scripts/<rule>.py` except nuclei registration (`.m` with the
MATLAB backend, `.py` with the Python backend) and DAPI rotation (inline Python
in the backend registration rule file).

| Rule | Inputs | Outputs / behavior |
| --- | --- | --- |
| `nuclei_registration` | Config JSON and INPUT additional-round/FOV directories; script also reads reference-round `*ch04.tif` | `log/{fovID}_nr.txt`, `log/gr_shifts/{fovID}_nr.txt`, registered `images/<round>/<channel name>/{fovID}.tif`; MATLAB script, or with `backend: python` the Python script ([workflows](workflows.md#nuclei-registration)) |
| `rotate_nuclei` | INPUT/`{dapi_round}/{fovID}/*ch04.tif` | `images/DAPI/{fovID}.tif`, rotated and optionally projected by top-level `maximum_projection` |
| `create_nuclei_amplicon_overlay` | DAPI and `images/ref_merged/{fovID}.tif` (the reference round's channel-merged detection image, ZYX, or YX with top-level `maximum_projection`; see [projection views](workflows.md#projection-views-and-the-reference-merged-image)) | `images/overlay/{fovID}.tif`; `composite_nuclei_amplicon` (contrast stretch, maximum of the DAPI and amplicon images) and the optional Z projection |
| `enhance_dapi_with_flamingo` | `images/flamingo/DAPI/{fovID}.tif`, `images/flamingo/Flamingo/{fovID}.tif` | `images/flamingo/enhanced_DAPI/{fovID}.tif` (`enhance_with_flamingo`); those input folders must be supplied separately |
| `stardist_segmentation` | `images/{segmentation_input_folder}/{fovID}.tif` (default folder `overlay`) | `images/stardist_segmentation/{fovID}.tif`, `uint32` labels of the package's `stardist` method, and the run record beside it as `{fovID}.json` |

Nuclei registration's additional-round objects need `channel_order` structs
with MATLAB channel/name metadata beyond the schema-described `round_name`;
both backends read them.
It is not sufficient to add a name to `additional_round`. Reference-channel
identity, rotation and registration must match the sequencing coordinate frame.

`stardist_segmentation` runs the package's `stardist` method in the environment that
runs Snakemake, which needs the `stardist` extra (StarDist, CSBDeep and TensorFlow); it
no longer uses a conda environment under `envs_path`. The model is the folder
`<stardist_base_path>/<stardist_model_name>`, hashed before use; a known model is
resolved in the weights cache, and nothing is downloaded. The model's `n_dim` chooses 2D
or 3D. The input file is read with its stored metadata or, without one, with metadata
declared from `voxel_size_z` and `voxel_size_xy`; the translation of the thresholds,
`rescale` and the label expansion is in [configuration](workflow-configuration.md#downstream-parameter-blocks).

Behavior to check on real inputs:

- The overlay accepts YX inputs as one plane (the legacy script raised on them), so a
  top-level projection before the overlay works; inputs of different shapes raise.
- An image without objects gives an all-zero label image with outcome `empty`; the
  legacy Otsu gate, which raised on an image without foreground, is gone.
- `rotate_nuclei` raises unless exactly one file matches
  `INPUT/{dapi_round}/{fovID}/*ch04.tif`, naming the matches.
- Label expansion in segmentation and in assignment operates per XY slice on 3-D
  data. `reads_assignment` refuses a label file expanded by `stardist_segmentation`
  (`expand_labels: true` there), because no original mask exists: expand in
  `reads_assignment` instead, once.

## Tile geometry and sample aggregation

| Rule | Inputs | Outputs relative to OUTPUT |
| --- | --- | --- |
| `create_sample_maf` | `documents/sample-annotation.csv`, `documents/raw.maf` | `documents/maf/{sample}.maf`; acquisition metadata must be supplied, not inferred from molecule coordinates |
| `stitching_preparation` | Annotation, sample MAF, DAPI images for the annotation's complete FOV range | `images/fused/{sample}/blank.tif`, `grid.csv`, `grid.png` and prepared image side effects |
| `create_BigStitcher_macro` | Fused sample `grid.csv` | `images/fused/{sample}/BigStitcher_macro.ijm`; actual script is `create_BigStitcher_macro_1.py` |
| `run_BigStitcher_macro` | Macro and grid | `images/fused/{sample}/DAPI/dataset.xml`, through configured Fiji executable |
| `create_tile_config` | Sample's dataset XML and grid | `output/tile_config_{sample}.csv` and `.html` |
| `reads_assignment` | Annotation, DAPI, segmentation labels, goodSpots CSV, `documents/genes.csv`, sample tile config | `expr/{fovID}/raw.h5ad`, `expr/{fovID}/reads_assignment.csv`; additional diagnostic images/logs may be written |
| `create_sample_h5ad` | Annotation and all per-FOV H5ADs in its sample range | `expr/{sample}_raw.h5ad` |
| `create_sample_reads_assignment` | Annotation and all per-FOV assignment CSVs in its sample range | `expr/{sample}_reads_assignment.csv` |

The scripts for these rules are named after the rules except the macro script
noted above; `run_BigStitcher_macro` invokes Fiji in a shell. Fiji must include
BigStitcher. `create_tile_config.py` assumes specific XML transform names
(`Stitching Transform`, `Translation to Regular Grid`) and grid conventions;
arbitrary Fiji XML is not a guaranteed compatible input.

`reads_assignment` runs `assign_molecules` ({doc}`assignment-contract`) on the label
file, imported with the target the segmentation translation infers (`cell` for the
`overlay` folder, `nucleus` otherwise) on a grid declared from the file's shape and
`voxel_size_z`, `voxel_size_xy` (a YX label file is a plane of an `img_z` × Y × X
grid). The goodSpots CSV's one-based `x,y,z`, integer or float, become zero-based
positions sampled at `floor(c + 0.5)`; a molecule outside the label grid is
`outside_grid`. `documents/genes.csv`, a headerless gene/barcode table not copied from
INPUT by these rules, gives the matrix columns and must list the same genes as the
codebook the sequencing rules read (`INPUT/genes.csv`, or the panel with
`readout_mode: direct`); a goodSpots gene outside it raises before assignment. The
tile config supplies integer `id,x,y,z` and
`start_x_norm,end_x_norm,start_y_norm,end_y_norm` columns; the adapter applies the
global offsets and the non-overlap filter to the package result as the script did,
outside the package (§2.10).

`raw.h5ad` keeps its legacy `obs` columns (`sample`, `fov_id`, `volume`,
`fov_x/y/z`, `seg_label`, `global_x/y/z`, computed on the expanded territories when
assign expands) and gains `size_voxels`, `expanded_size_voxels`, `size_physical`,
`centroid_z/y/x`, `n_molecules`, `n_nuclei`, `correspondence`, `correspondence_flags`,
`compartments` and the record as JSON text in `uns["assignment"]`; `X` holds the
float64 whole-cell counts of the kept cells, also of cells without molecules. With
nuclei, the `nucleus` and `cytoplasm` layers hold 0.0 for a measured zero and NaN
where compartments are not available. `reads_assignment.csv` keeps its columns and
gains `spot_id`, `assignment_status`, `cell_id`, `in_expansion`, `original_cell_id`,
`nucleus_id` and `compartment`; after the rows the overlap filter keeps, it holds every
`outside_grid` molecule, with `seg_label` 0. The rule also writes `assignment.png`
(`plot_assignment` over the DAPI image) and `log.txt` beside them.

Assignment needs the `anndata` extra for `raw.h5ad`; sample H5AD aggregation uses
Scanpy, which the base package does not install. The matrix counts decoded genes
inside labels. It does not establish segmentation quality, biological identity, or
cross-FOV equivalence. Dry-running the sequencing examples does not exercise these
downstream scripts, models, acquisition metadata or coordinate checks.
