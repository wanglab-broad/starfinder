# Synthetic reference

The presets and files of the four synthetic scenes. The preset tables are read from the
code when the site is built. For the functions and configuration classes, see the
{doc}`synthetic API <api/synthetic>`.

## Presets

### Amplicon scene

`benchmark_scene_preset(name)` returns a codebook and an `AmpliconSceneConfig`; all have
four rounds and four channels and write `uint16` unless asked for `uint8`. Version
`PRESET_VERSION` (`benchmark-presets-v1`).

```{synthetic-presets} amplicon
```

| Other presets | Returned by | Use |
| --- | --- | --- |
| `formed-small-v1`, `formed-z1-v1` | `amplicon_scene_preset(name)` | Tests of the amplicon model: 8 × 32 × 32 or one plane, every effect off |
| `z1-clean`, `z1-combined`, `small-clean`, `small-combined` | `development_scene_preset(condition, size=...)` | Small scenes with two amplicons for exact tests |
| `calibrated-development-v1` | `calibrated_scene_preset(condition, ...)` | One 8 × 64 × 64 FOV under each of the conditions in `CALIBRATED_CONDITIONS`, for preprocessing evaluation |
| Registration pairs | `generate_registration_pair(preset, deformation=...)` | A reference and a moved round of a benchmark preset; the deformations are in `DEFORMATION_PRESETS` |

### Sample scene

`sample_scene_preset(name)`; version `SAMPLE_PRESET_VERSION` (`sample-presets-v1`).

```{synthetic-presets} sample
```

### Cellular scene

`cellular_scene_preset(name)`; version `CELLULAR_PRESET_VERSION`
(`context-presets-v1`). Counts are typical values; they vary with the seed.

```{synthetic-presets} cellular
```

### Real-appearance scene

`appearance_scene_preset(name)`; version `APPEARANCE_PRESET_VERSION`
(`appearance-presets-v1`). The slow presets need their real bank.

```{synthetic-presets} appearance
```

## Files

Each save function writes into a new or empty directory and refuses to overwrite.
Positions in tables are in µm in the sample frame unless the column says voxels;
CSV files have one header row.

### Amplicon scene and dataset

| File | Contents |
| --- | --- |
| `images/<round>.ome.tif` | `save_amplicon_scene`: the ZYXC image of one round |
| `formed.csv` | One row per amplicon: ID, gene, colour sequence, position, brightness and shape |
| `round_truth.csv` | One row per amplicon and round: moved position, and whether it was dropped, weakened, lost or emitting |
| `signals.csv` | The intended, pre-mixing and realized signal of each amplicon, channel and round |
| `provenance.json` | Requested and effective configuration, seed, codebook, image hashes, clipping counts and transforms |
| `<round>/<FOV>/<channel>.tif` | `save_amplicon_dataset`: one ZYX image per round, FOV and channel, the layout that `Dataset` reads |
| `codebook.csv`, `scene_truth.csv`, `ground_truth.json`, `generation.json` | The dataset's codebook, one row per amplicon and round with gene and channel, the historical truth record, and how it was generated |

The command line, `starfinder synthetic generate`, writes the same truth in a
FOV-first layout, `<FOV>/<round>/<channel>.tif`, with a `manifest.json`.

### Sample scene

| File | Contents |
| --- | --- |
| `sample.json` | Versions, configuration, seed, the sample grid, the sample's extent, the codebook, and the SHA-256 of every file |
| `positions.csv` | Each FOV's nominal position, as a microscope would report it |
| `placement_truth.csv` | Each FOV's grid place, nominal and true origin, true correction and random offset in voxels |
| `nuclei.csv`, `cells.csv` | Each nucleus and cell, its position, size, and whether it lies in the sample |
| `molecules.csv` | Each molecule: gene, position, `true_cell_id`, `true_nucleus_id`, `compartment`, and the FOVs that see it |
| `cell_fovs.csv`, `molecule_fovs.csv` | Which FOVs see each cell and molecule, and the labels at the molecule in each FOV (`label_agrees`) |
| `labels/sample_nuclei.tif`, `labels/sample_cells.tif` | Nucleus and cell labels on the sample grid |
| `fovs/<FOV>/` | Each FOV's amplicon-scene files, `stain.ome.tif`, `labels/nuclei.tif`, `labels/cells.tif`, and `molecules.csv` with one-based `x, y, z, gene` |

### Cellular scene

`save_cellular_scene` writes the sample scene's files, without `stain.ome.tif`, plus:

| File | Contents |
| --- | --- |
| `lobes.csv`, `harmonics.csv` | The extra lobes and shape terms of each nucleus |
| `cell_context.csv`, `regions.csv` | Each cell's type and region, and each region's main type |
| `type_abundances.csv`, `cell_gene_counts.csv` | Gene weights per type, and the count of each gene in each cell, in total and in the nucleus |
| `marker_pairs.csv` | The named marker pairs with their cell counts and the expected answer |
| `stain_channels.csv`, `stain_objects.csv`, `bright_regions.csv`, `unevenness.csv` | Each stain's source, round, channel and scale, the brightness of each object, and its bright spots and unevenness |
| `morphology_truth.csv` | The morphology round's true translation per FOV |
| `fovs/<FOV>/stains/<name>.ome.tif`, `fovs/<FOV>/morphology.ome.tif` | The stain images and the morphology round |
| `context.json` | Versions, configuration, how far packing got, seed, and the SHA-256 of every file |

`save_cellular_dataset` writes a raw dataset instead: `<round>/<FOV>/<channel>.tif`,
`codebook.csv` and `dataset.json`.

### Real-appearance scene

`save_appearance_scene` writes the cellular scene's files, plus:

| File | Contents |
| --- | --- |
| `placements.csv` | Each placed object: its bank entry and source, position, rotation, flips, scale, gain and overlap |
| `background_patches.csv` | The background patch used at each lattice point (empty for a bank drawn in memory) |
| `appearance.json` | Versions, configuration, the bank's name and hashes, how the layout ended, and the SHA-256 of every file |

## Reproducibility

Every random draw comes from its own stream, keyed by the seed, the scene and the part
it belongs to (component, object, round and channel). Adding or removing a part of the
configuration therefore leaves the draws of the other parts unchanged. A scene is
identical, byte for byte, for the same configuration, NumPy build and CPU; each record
keeps the SHA-256 of what it wrote.
