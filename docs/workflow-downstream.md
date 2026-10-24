# Downstream segmentation and assignment

The sequencing endpoint is the FOV's molecule table: `fovs/{fovID}/molecules.csv` of the
Python `rsf_single_fov`, or on the legacy route (MATLAB's `rsf_single_fov`, and
`stitch_subtile` in the subtile and deep modes) `signal/{fovID}_goodSpots.csv`
({doc}`workflows`, "Molecule tables"). Producing a cell
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
| `nuclei_registration` | Config JSON and INPUT additional-round/FOV directories; script also reads reference-round `*ch04.tif` | MATLAB: `log/{fovID}_nr.txt`, `log/gr_shifts/{fovID}_nr.txt`; Python: `fovs/{fovID}/nuclei_registration.json` (the registered rounds and their attempts); both: registered `images/<round>/<channel name>/{fovID}.tif`; MATLAB script, or with `backend: python` the Python script, which calls `FOV.prepare_morphology` and, with the Python-only `parameters.checkpoints`, also writes the saved form under `fovs/{fovID}/other_rounds/` (not declared) ([workflows](workflows.md#nuclei-registration)) |
| `rotate_nuclei` | INPUT/`{dapi_round}/{fovID}/*ch04.tif` | `images/DAPI/{fovID}.tif`, rotated and optionally projected by top-level `maximum_projection` |
| `create_nuclei_amplicon_overlay` | DAPI and the reference image: MATLAB's `images/ref_merged/{fovID}.tif`, on the Python backend the file `reference_image` names (`fovs/{fovID}/reference.tif` by default) or `reference_image_import` (the reference round's channel-merged detection image, ZYX, or YX with top-level `maximum_projection`; see [projection views](workflows.md#projection-views-and-the-reference-merged-image)) | `images/overlay/{fovID}.tif`; `composite_nuclei_amplicon` (contrast stretch, maximum of the DAPI and amplicon images) and the optional Z projection |
| `enhance_dapi_with_flamingo` | `images/flamingo/DAPI/{fovID}.tif`, `images/flamingo/Flamingo/{fovID}.tif` | `images/flamingo/enhanced_DAPI/{fovID}.tif` (`enhance_with_flamingo`); those input folders must be supplied separately |
| `stardist_segmentation` | `images/{segmentation_input_folder}/{fovID}.tif` (default folder `overlay`) | `images/stardist_segmentation/{fovID}.tif`, `uint32` labels of the package's `stardist` method, and the run record beside it as `{fovID}.json` |

Nuclei registration's additional-round objects need `channel_order` structs
with MATLAB channel/name metadata beyond the schema-described `round_name`;
both backends read them.
It is not sufficient to add a name to `additional_round`. Reference-channel
identity, rotation and registration must match the sequencing coordinate frame.

Like these rules, which read saved reference-frame images and never load a sequencing
round, a Python process can segment from saved images: `FOV.load_reference_image()`
restores the reference image (`fovs/{fovID}/reference.tif` by default) and `FOV.load_registered_round(name)` the
saved form of each prepared image (the `reference_stain` and the additional rounds),
and `FOV.segment` then runs with no sequencing round resident
({doc}`segmentation-contract`, "Coordination per FOV"). For the Python rule to write
that saved form, set
`rules.nuclei_registration.parameters.checkpoints` (for example `{}`; `null` or absent
writes none). The Python-only `segmentation` block is not wired to a rule yet (§2.13),
and `rotate_nuclei` and the legacy segmentation rules are unchanged.

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

The rules of this table other than `reads_assignment` read MATLAB's sample annotation and
run with `backend: matlab` only; a Python configuration uses the stitching and assembly
blocks instead, and its `reads_assignment` reads the FOV membership `INPUT/fovs.csv`
({doc}`workflow-configuration`, "FOV membership").

| Rule | Inputs | Outputs relative to OUTPUT |
| --- | --- | --- |
| `create_sample_maf` | `documents/sample-annotation.csv`, `documents/raw.maf` | `documents/maf/{sample}.maf`; acquisition metadata must be supplied, not inferred from molecule coordinates |
| `stitching_preparation` | Annotation, sample MAF, DAPI images for the annotation's complete FOV range | `images/fused/{sample}/blank.tif`, `grid.csv`, `grid.png` and prepared image side effects |
| `create_BigStitcher_macro` | Fused sample `grid.csv` | `images/fused/{sample}/BigStitcher_macro.ijm`; actual script is `create_BigStitcher_macro_1.py` |
| `run_BigStitcher_macro` | Macro and grid | `images/fused/{sample}/DAPI/dataset.xml`, through configured Fiji executable |
| `create_tile_config` | Sample's dataset XML and grid | `output/tile_config_{sample}.csv` and `.html` |
| `reads_assignment` | The FOV membership (`INPUT/fovs.csv` for the Python backend, the annotation for MATLAB), DAPI, segmentation labels, the molecule table, the codebook of the sequencing rules (`INPUT/genes.csv`, or the file `codebook` or `direct_panel` names), sample tile config | `expr/{fovID}/raw.h5ad`, `expr/{fovID}/reads_assignment.csv`; additional diagnostic images/logs may be written |
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
grid). The molecules of `molecules.csv` keep their identities and zero-based
positions (`load_molecules`); a legacy goodSpots CSV's one-based `x,y,z`, integer or float,
become zero-based positions and the row `i` the spot `csv:<i>`
(`import_legacy_molecules_csv`). Positions are sampled at `floor(c + 0.5)`; a molecule
outside the label grid is
`outside_grid`. The matrix columns are the genes of the codebook the sequencing rules
read, loaded as they load it, in its order: `Codebook.genes` (first appearance) or, with
`readout_mode: direct`, the panel's `DirectPanel.genes` (table order). That file is
`INPUT/genes.csv`, or the file the Python-only key `codebook` (`direct_panel`) names
({doc}`workflow-configuration`, "Reference file"); a molecule gene outside it raises
before assignment, and a codebook gene no molecule carries is a zero column. The
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
where compartments are not available. `reads_assignment.csv` holds the molecule
table's columns (on the Python route those of `molecules.csv`, joined to the assignment by
`(spot_namespace, spot_id)`; on the legacy route the goodSpots columns, joined by row, with
`spot_id`), then `global_x/y/z` and `seg_label`, and gains `assignment_status`, `cell_id`,
`in_expansion`, `original_cell_id`, `nucleus_id` and `compartment`; after the rows the
overlap filter keeps, it holds every `outside_grid` molecule, with `seg_label` 0. The rule also writes, beside them,
`assignment.png` (`plot_assignment`, the Z-maximum view over the DAPI image) for every
FOV, `assignment_single_layer.png` (the middle plane `Z // 2`) when the label image is
ZYX, not for a YX one, and `log.txt`: one line of totals, then the assignment ratio,
assigned molecules over all molecules of the table with both counts
(`assignment ratio: 61.11% (11 of 18 molecules assigned)`), or
`assignment ratio: none (no molecule)` when the FOV has none. With an `assembly` block,
on the route per FOV (below), the sample's `export_sample` rule draws the two figures
again once the sample is assembled, with the FOV's ownership region as a yellow line, the cells and molecules the
assembly excluded in grey and, as each panel's x-axis label, how many of those drawn it retained
({doc}`assignment-contract`, "Diagnostics"); `log.txt` is not rewritten, and without the
block the figures stay as `reads_assignment` draws them.

Assignment needs the `anndata` extra for `raw.h5ad`; sample H5AD aggregation uses
Scanpy, which the base package does not install. The matrix counts decoded genes
inside labels. It does not establish segmentation quality, biological identity, or
cross-FOV equivalence. Dry-running the sequencing examples does not exercise these
downstream scripts, models, acquisition metadata or coordinate checks.

## Two routes to a sample's cells

With `stitching` and `assembly` blocks, `export_sample` reaches the sample's cells by one
of two routes, set by `export.route` ({doc}`sample-segmentation-contract`):

* **Per FOV** (`per_fov`). Each FOV is segmented and assigned on its own
  (`reads_assignment`), and the export assembles the FOVs: a cell is kept by the FOV that owns its centroid. Two
  kept cells of different FOVs whose territories share at least `duplicate_threshold` of
  the smaller one are taken as two observations of one cell: the larger stays and the
  other is removed as a second observation. A cell cut by a FOV edge is not completed from
  the other FOV, and nothing is merged across FOVs.
* **After fusion** (`after_fusion`), for projection samples only. The fused Z maximum of
  the sample, at `scale_yx`, is segmented once, and the sample's molecules are assigned to
  those cells, each molecule kept by the FOV that owns its position. A cell across a FOV
  seam is one cell of one segmentation. The whole fused projection is segmented in one
  call, so it must fit the memory of the segmentation's device, and with a learned model
  on `cuda` the export rule needs a GPU node.

**The default** follows `dimensionality`: `projection` takes the route after fusion;
`stack` and `both` take the route per FOV. A stack is segmented per FOV: the route after
fusion refuses `dimensionality: stack` (with `both` and `route: after_fusion` it segments
the projection), and refuses `projection: sum` as well, because it segments the Z
maximum. A configuration with an assignment name that exports a projection must
therefore set `export.segmentation` (the runs of the route after fusion) or
`route: per_fov`; until it does, the workflow refuses it when the rules are parsed, naming
both, so that no run changes route without a key that says so.

**Which to choose.** The route per FOV for a 3D stack, for a sample that cannot be fused,
or whose fused projection cannot be segmented in one call, on the hardware at hand, and to
stay comparable with samples processed per FOV before. The route after fusion for a
projection sample whose fused projection can be segmented in one call, where cells across
FOV seams should be segmented once.

**The same outputs.** Both routes write the same files of `export/<sample>/`: the same
tables with the same columns, the same `sample.h5ad` and the same fused images, so that
what reads them need not know the route; the record names it (`route` in the `assembly`
section of `export.json`). The route after fusion adds the label image of each
segmentation run, `segmentation_<run>.ome.tif` (and `territories.ome.tif` when the
assignment expands the cells), and the `segmentation` section. Whether the two routes find
the same cells in a sample is what `compare_routes` reports: it reads an export folder of
each route for the same sample, says how their cells and molecules correspond, and decides
nothing (it is not an accuracy, and neither route is a truth). An example configuration
of each route is in {doc}`workflow-configuration` ("Stitching, assembly and export").
