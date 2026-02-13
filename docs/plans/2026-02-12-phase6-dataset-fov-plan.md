# Plan: Phase 6 — STARMapDataset + FOV Implementation

## Context

Phases 1-5 implemented all low-level modules (I/O, registration, spotfinding, barcode, preprocessing). Phase 6 creates the orchestration layer: `STARMapDataset` (sample-level config) and `FOV` (per-FOV processor) classes that wrap Phase 1-5 functions into a fluent pipeline API. The design doc (`docs/main_python_object_design.md` v3) is the authoritative specification.

**Outcome**: A `starfinder.dataset` subpackage that enables:
```python
dataset = STARMapDataset.from_config(config)
fov = dataset.fov("Position001")
fov.load_raw_images().enhance_contrast().morph_recon()
fov.global_registration()
fov.spot_finding().reads_extraction()
dataset.load_codebook(path)
fov.reads_filtration()
fov.save_signal()
```

---

## File Plan

### New files (in `src/python/starfinder/dataset/`)

| File | Contents | ~Lines |
|------|----------|--------|
| `__init__.py` | Public API exports | ~25 |
| `types.py` | Type aliases + dataclasses (LayerState, Codebook, CropWindow, SubtileConfig) | ~150 |
| `logging.py` | `log_step` decorator | ~25 |
| `paths.py` | `FOVPaths` frozen dataclass | ~35 |
| `dataset.py` | `STARMapDataset` class | ~80 |
| `fov.py` | `FOV` class (all methods) | ~350 |

### New test files (in `src/python/test/`)

| File | Contents | ~Lines |
|------|----------|--------|
| `test_types.py` | LayerState, Codebook, CropWindow, SubtileConfig | ~100 |
| `test_dataset.py` | STARMapDataset from_config, fov factory, load_codebook | ~80 |
| `test_fov.py` | FOV pipeline on mini synthetic dataset | ~150 |

### Modified files

| File | Change |
|------|--------|
| `starfinder/__init__.py` | Add `from starfinder.dataset import ...` exports |

---

## Implementation Steps

### Step 1: `dataset/types.py` — Type definitions

Type aliases and dataclasses from design doc §4:

```python
Shift3D: TypeAlias = tuple[float, float, float]       # (dz, dy, dx)
ImageArray: TypeAlias = np.ndarray                      # (Z, Y, X, C)
ChannelOrder: TypeAlias = list[str]                     # ["ch00", "ch01", ...]
```

**LayerState**: `seq`, `other`, `ref` fields; `all_layers` property (NOT `.all`); `to_register` property; `validate()` method.

**Codebook**: `gene_to_seq`, `seq_to_gene` dicts; `from_csv()` factory delegates to `starfinder.barcode.load_codebook()`.

**CropWindow** (frozen): `y_start, y_end, x_start, x_end` (0-based, exclusive end); `to_slice()` returns `(slice_y, slice_x)`.

**SubtileConfig**: `sqrt_pieces`, `overlap_ratio`, `windows: list[CropWindow]`; `n_subtiles` property; `compute_windows(height, width)` implements MATLAB `MakeSubtileTable` tiling logic:
- `tile_size = height // sqrt_pieces` (uses height since tiles are square and MATLAB uses dims(1))
- Overlap half = `floor(tile_size * overlap_ratio)`
- Edge compensation: no overlap on outer edges; clamp right/bottom to image boundary
- IMPORTANT: coordinates are 0-based internally (Python convention); convert to 1-based only in CSV output

### Step 2: `dataset/logging.py` — log_step decorator

Exactly as in design doc §7. Uses `logging.getLogger("starfinder")`.

### Step 3: `dataset/paths.py` — FOVPaths

Frozen dataclass from design doc. Properties: `ref_merged_tif`, `subtile_dir`, `signal_csv(slot)`, `signal_png(slot)`, `shift_log(suffix)`, `score_log(suffix)`.

### Step 4: `dataset/dataset.py` — STARMapDataset

Non-frozen `@dataclass`. Fields: `input_root`, `output_root`, `dataset_id`, `sample_id`, `output_id`, `layers`, `channel_order`, `codebook`, `subtile`, `rotate_angle`, `maximum_projection`, `fov_pattern`.

**`from_config(config)`**: Builds `LayerState(seq=[...], ref=config["ref_round"])` and populates `channel_order` from config.

**`fov(fov_id)`**: Returns `FOV(dataset=self, fov_id=fov_id)`.

**`fov_ids(n_fovs, start=0)`**: Returns list from pattern.

**`load_codebook(path, split_index, do_reverse)`**: Sets `self.codebook = Codebook.from_csv(...)`.

### Step 5: `dataset/fov.py` — FOV class

The main workhorse. Each method delegates to a Phase 1-5 function.

**Attributes**: `dataset`, `fov_id`, `images: dict`, `metadata: dict`, `global_shifts: dict[str, Shift3D]`, `local_registered: set[str]`, `all_spots`, `good_spots`.

**Delegated properties**: `layers` → `self.dataset.layers`, `codebook` → `self.dataset.codebook`.

#### Image loading
- **`load_raw_images(rounds, channel_order, *, convert_uint8, subdir, layer_slot)`**
  - Defaults: `rounds` from `self.layers.seq` or `.other`; `channel_order` from `self.dataset.channel_order`
  - Loop: `load_image_stacks(self.input_dir(round_name), channel_order, subdir, convert_uint8)` per round
  - Stores in `self.images[round_name]`, `self.metadata[round_name]`

#### Preprocessing (5 methods, all follow same pattern)
- Loop over `layers` (default `self.layers.all_layers`), apply function, store back
- **`enhance_contrast`** → `min_max_normalize()`
- **`hist_equalize`** → `histogram_match(volume, reference)` where `reference = self.images[self.layers.ref][:, :, :, ref_channel]`
- **`morph_recon`** → `morphological_reconstruction()`
- **`tophat`** → `tophat_filter()`
- **`make_projection`** → `utils.make_projection()` — applies to ALL images (no layer filter)

#### Registration
- **`global_registration(*, layers_to_register, ref_img, mov_img, ref_channel, save_shifts)`**
  - Uses `self.layers.ref` for reference round
  - Create ref/mov 3D volumes: `"merged"` = sum across channels; `"single-channel"` = pick `ref_channel`
  - For each round: `registered, shifts = register_volume(images, ref_3d, mov_3d)`
  - Store: `self.images[round] = registered`, `self.global_shifts[round] = shifts`
  - If `save_shifts`: write CSV to `self.paths.shift_log()` with columns `fov_id,round,row,col,z` (row=dy, col=dx, z=dz)

- **`local_registration(*, ref_channel, layers_to_register, iterations, smoothing_sigma, method, pyramid_mode)`**
  - Uses `self.layers.ref`, extracts single channel for ref/mov: `[:, :, :, ref_channel]`
  - For each round: `registered, _ = register_volume_local(images, ref_3d, mov_3d, ...)`
  - Store: `self.images[round] = registered`, `self.local_registered.add(round)`

#### Spot finding & barcode
- **`spot_finding(*, intensity_estimation, intensity_threshold)`**
  - `self.all_spots = find_spots_3d(self.images[self.layers.ref], ...)`

- **`reads_extraction(voxel_size, layers)`**
  - Default layers: `self.layers.seq`
  - Per round: `color, score = extract_from_location(self.images[round], self.all_spots, voxel_size)`
  - Add columns: `self.all_spots[f'{round}_color'] = color`, `self.all_spots[f'{round}_score'] = score`
  - Concatenate: `self.all_spots['color_seq'] = concat of {round}_color strings`

- **`reads_filtration(*, end_bases, start_base)`**
  - `good, stats = filter_reads(self.all_spots, self.codebook.seq_to_gene, end_bases, start_base)`
  - `self.good_spots = good`

#### Output
- **`save_ref_merged()`**
  - Get ref image; if `dataset.maximum_projection`: apply `make_projection()`
  - `save_stack(image, self.paths.ref_merged_tif)`

- **`save_signal(slot, columns)`**
  - Select `all_spots` or `good_spots`
  - Coordinate conversion: add 1 to z/y/x, reorder to (x, y, z, gene)
  - `df.to_csv(self.paths.signal_csv(slot), index=False)`

#### Subtile operations
- **`create_subtiles(*, out_dir)`**
  - Uses `self.dataset.subtile.windows` for crop regions
  - For each window: extract `images[:, y_slice, x_slice, :]`, save as NPZ
  - Build and save `subtile_coords.csv` with 1-based coordinates for stitch_subtile.py
  - Return coords DataFrame

- **`from_subtile(subtile_path, dataset, fov_id)`** (classmethod)
  - Load NPZ, populate `images` dict from `images_{round}` keys

### Step 6: `dataset/__init__.py` + update `starfinder/__init__.py`

Export: `STARMapDataset`, `FOV`, `FOVPaths`, `LayerState`, `Codebook`, `CropWindow`, `SubtileConfig`, `Shift3D`, `ImageArray`, `ChannelOrder`, `log_step`.

### Step 7: Tests

**`test_types.py`**:
- `TestLayerState`: validate, all_layers, to_register, ref not in layers error
- `TestCodebook`: from_csv with mini dataset codebook, genes property, n_genes
- `TestCropWindow`: to_slice correctness
- `TestSubtileConfig`: compute_windows on small image, edge compensation, n_subtiles

**`test_dataset.py`**:
- `TestSTARMapDataset`: from_config with tissue_2D_test.yaml structure, fov factory, fov_ids, load_codebook sets attribute

**`test_fov.py`** (uses mini_dataset fixture):
- `TestFOVPipeline`: load_raw_images → enhance_contrast → global_registration → spot_finding → reads_extraction → reads_filtration → save_signal
- Verify: images dict populated, global_shifts populated, all_spots columns correct, good_spots has gene column, saved CSV has 1-based coords with (x,y,z,gene) columns

---

## Key Implementation Details

### Subtile coordinate contract (critical for stitch_subtile.py)
- CropWindow stores 0-based coords internally (Python slicing)
- `subtile_coords.csv` must use 1-based coords (MATLAB convention)
- Convert: `scoords_x_csv = window.x_start + 1`, `ecoords_x_csv = window.x_end` (exclusive→inclusive + 1-based)
- `stitch_subtile.py` formula: `global_1based = local_1based + scoords_1based - 1`

### Shift log CSV format
- Python stores `(dz, dy, dx)` in `global_shifts`
- CSV columns: `fov_id,round,row,col,z` where `row=dy, col=dx, z=dz`

### Reference image creation for registration
- `"merged"`: `ref_3d = np.sum(self.images[ref_round], axis=-1)` (sum channels → 3D)
- `"single-channel"`: `ref_3d = self.images[ref_round][:, :, :, ref_channel]`

### color_seq concatenation in reads_extraction
- Each round produces a 1-char string ("1"-"4", "M", "N") per spot
- Concatenate across rounds: `"1" + "3" + "2" + "4"` → `"1324"`

---

## Verification

1. `cd src/python && uv run pytest test/test_types.py test/test_dataset.py test/test_fov.py -v`
2. Verify CSV output has correct 1-based coordinates
3. Verify shift log CSV matches expected format
4. Run full test suite: `uv run pytest test/ -v` to check no regressions
