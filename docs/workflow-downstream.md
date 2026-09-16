# Downstream segmentation and assignment

The sequencing endpoint is `signal/{fovID}_goodSpots.csv`. Producing a cell
expression matrix additionally requires aligned morphology images, segmentation
labels, codebook metadata and tile geometry. The rules below are included for
both backends; choosing Python sequencing does not make all downstream stages
Python-only or supply their dependencies.

The contracts below were checked against `workflow/rules/segmentation.smk`,
`stitching.smk`, `reads-assignment.smk`, `utils.smk` and their scripts. They are
not an executed image-to-cell validation. Scientific validation belongs to
[W-92: public image-to-molecule-to-cell example](https://linear.app/jiahaoh/issue/W-92/verify-a-public-image-to-molecule-to-cell-example-for-chapter-ii).
The [downstream examples](https://github.com/wanglab-broad/starfinder/blob/dev/example/downstream/README.md)
remain an additional entry point.

## Morphology and labels

All paths in this table are relative to OUTPUT unless marked INPUT. Each rule
uses `workflow/scripts/<rule>.py` except nuclei registration (`.m`) and DAPI
rotation (inline Python in the backend registration rule file).

| Rule | Inputs | Outputs / behavior |
| --- | --- | --- |
| `nuclei_registration` | Config JSON and INPUT additional-round/FOV directories; script also reads reference-round `*ch04.tif` | `log/{fovID}_nr.txt`, `log/gr_shifts/{fovID}_nr.txt`; MATLAB even with Python backend |
| `rotate_nuclei` | INPUT/`{dapi_round}/{fovID}/*ch04.tif` | `images/DAPI/{fovID}.tif`, rotated and optionally projected by top-level `maximum_projection` |
| `create_nuclei_amplicon_overlay` | DAPI and `images/ref_merged/{fovID}.tif` | `images/overlay/{fovID}.tif`; contrast adjustment, channel maximum and optional Z projection |
| `enhance_dapi_with_flamingo` | `images/flamingo/DAPI/{fovID}.tif`, `images/flamingo/Flamingo/{fovID}.tif` | `images/flamingo/enhanced_DAPI/{fovID}.tif`; those input folders must be supplied separately |
| `stardist_segmentation` | `images/{segmentation_input_folder}/{fovID}.tif` (default folder `overlay`) | `images/stardist_segmentation/{fovID}.tif`, uint16 labels |

Nuclei registration's additional-round objects need `channel_order` structs
with MATLAB channel/name metadata beyond the schema-described `round_name`.
It is not sufficient to add a name to `additional_round`. Reference-channel
identity, rotation and registration must match the sequencing coordinate frame.

StarDist needs an existing conda environment at `envs_path/stardist`, compatible
StarDist/CSBDeep/TensorFlow and TIFF packages, and a trained model at the
configured base path/name. The script chooses StarDist2D versus StarDist3D from
input dimensionality; thresholds and rescaling/label expansion are explicit
parameters. A model name in YAML is not a bundled or downloaded model.

Source-visible limitations require checking on real inputs:

- The overlay script reduces `axis=3` after adding a channel dimension, so it
  expects 3-D input stacks. Feeding already projected 2-D images is incompatible
  with that operation. For this route, keep top-level projection off until the
  overlay's own optional projection.
- StarDist calls `areas.max()` after Otsu/connected-component preprocessing;
  an image with no foreground regions can fail before its zero-label fallback.
  Its `tifffile.imsave` import also requires a compatible TIFF library version.
- DAPI globbing must find the intended channel file. Missing/multiple matches
  are not diagnosed by the configuration schema.
- Label expansion in both segmentation and assignment operates per XY slice
  on 3-D data, not by a volumetric distance. Avoid unintentionally expanding twice.

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

`reads_assignment.py` subtracts 1 from the CSV's `x,y,z` before indexing labels
as `[z,y,x]` (or `[y,x]` for 2-D). Coordinates must be in bounds and share the
label image's frame. The tile config supplies integer `id,x,y,z` and
`start_x_norm,end_x_norm,start_y_norm,end_y_norm` columns for global offsets and
non-overlap filtering. The separate `documents/genes.csv` is a headerless
gene/barcode table, not automatically copied from INPUT by these rules.

Assignment uses `parse`, NumPy, pandas, scikit-image, TIFF, matplotlib and
AnnData; sample H5AD aggregation uses Scanpy. These downstream dependencies are
not all installed by the base Python package. The matrix counts decoded genes
inside labels; background reads and overlap handling follow the script's tile
bounds. It does not establish segmentation quality, biological identity, or
cross-FOV equivalence. Dry-running the sequencing examples does not exercise
these downstream scripts, models, acquisition metadata or coordinate checks.
