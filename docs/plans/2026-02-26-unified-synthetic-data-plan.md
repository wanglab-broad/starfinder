# Plan: Unify Synthetic Data Generation & Coordinate-Level Deformation

## Context

The project has two independent synthetic data generators:
- `starfinder.testdata` — multi-round, multi-channel FOV datasets for E2E pipeline testing
- `starfinder.benchmark.data` — single-channel ref/moving pairs for registration benchmarking

These share concepts (spot rendering, shifting) but have no code reuse. The deformation path in `benchmark.data` uses `scipy.ndimage.map_coordinates` on rendered images, causing interpolation blur and unnatural blank/noise regions.

**Goals:**
1. Unify into `starfinder.benchmark` with consistent presets (tiny, small, medium, large, tissue, thick_medium)
2. Redesign so both global shifts and local deformations are coordinate-level operations applied to spot positions *before* rendering — images always contain clean Gaussians
3. Add per-round spot rendering randomness (slightly varying intensity and sigma across rounds for the same spot)
4. Delete `starfinder.testdata` package entirely — update all consumers (including network-mount scripts) to new import paths

## Key Design Decisions

### Coordinate-First Rendering

**Old approach:** render spots → shift/warp image (creates artifacts)
**New approach:** transform spot coordinates → render at transformed positions (always clean)

Both global shift and local deformation become coordinate transforms:
- **Global shift:** `new_pos = pos + (dz, dy, dx)` — spots that move out of bounds are dropped
- **Local deformation:** `new_pos = pos + field[z, y, x]` — sample displacement field at spot position, add to coords
- **Combined:** shift first, then deform

### Per-Round Spot Variation

Real microscopy data shows slight intensity and PSF size variation for the same molecule across sequencing rounds. To simulate this, each spot gets per-round jitter applied at render time:
- **Intensity jitter:** `intensity * (1 + rng.normal(0, 0.1))` — ~10% variation
- **Sigma jitter:** `sigma * (1 + rng.normal(0, 0.05))` — ~5% variation
- Deterministic per (spot_id, round_idx, channel) via seeded RNG

This is implemented inside `generate_synthetic_dataset()` when building the `channel_spots` list for each round — each spot's intensity and sigma are jittered before passing to `create_test_image_stack()`.

`create_test_image_stack()` signature changes to accept per-spot sigma:
```python
spots: list[tuple[int, int, int, int, float]]  # (z, y, x, intensity, sigma)
```

---

## Step 1: Clean up `starfinder/benchmark/presets.py`

Remove `xlarge` and `thick_large` from `SIZE_PRESETS`, `SPOT_COUNTS`, and `SHIFT_RANGES`.

After change — 6 canonical presets:
```
tiny:         (8,   128,  128)    10 spots
small:        (16,  256,  256)    50 spots
medium:       (32,  512,  512)   400 spots
large:        (30, 1024, 1024)  1500 spots  → use generate_codebook(64)
tissue:       (30, 3072, 3072) 14000 spots  → use generate_codebook(64)
thick_medium: (100, 1024, 1024) 5200 spots  → use generate_codebook(64)
```

**File:** `src/python/starfinder/benchmark/presets.py`

---

## Step 2: Create `starfinder/benchmark/synthetic.py` (the unified generator)

This new file absorbs `testdata/synthetic.py` and replaces the data generation parts of `benchmark/data.py`.

### What moves here:

**From `testdata/synthetic.py`:**
- `TEST_CODEBOOK`, `generate_codebook()`, `encode_barcode_to_colors()` — constants and codebook helpers
- `SyntheticConfig` — unified config dataclass (add `tiny` preset, add deformation field)
- `create_test_image_stack()` — the authoritative Gaussian spot renderer
- `get_preset_config()` — preset lookup (updated for 6 presets, pull shape/spots/shifts from `presets.py`)
- `generate_synthetic_dataset()` — main multi-round orchestrator (refactored for coordinate transforms)
- `_generate_annotated_visualization()` — annotation PNGs
- `create_test_volume()` — simplified single-channel convenience wrapper

**From `benchmark/data.py`:**
- `DEFORMATION_CONFIGS`, `scale_deformation_config()` — deformation presets
- `create_deformation_field()` — displacement field generation (polynomial, gaussian, multi_point, linear)

### Modified functions:

**`create_test_image_stack()` — now accepts per-spot sigma:**
```python
def create_test_image_stack(
    shape: tuple[int, int, int],
    spots: list[tuple[int, int, int, int, float]],  # (z, y, x, intensity, sigma)
    background: int = 20,
    noise_std: int = 10,
    seed: int | None = None,
    add_noise: bool = True,
    dtype: Literal["uint8", "uint16"] = "uint8",
) -> np.ndarray:
```
Each spot carries its own sigma, enabling per-round PSF variation. The old `spot_sigma` parameter is removed — sigma is per-spot.

### New functions:

**`apply_shift_to_spots(spots, shift, shape)`**
```python
def apply_shift_to_spots(
    spots: list[tuple[int, int, int, int, float]],  # (z, y, x, intensity, sigma)
    shift: tuple[int, int, int],                     # (dz, dy, dx)
    shape: tuple[int, int, int],                     # (Z, Y, X) volume bounds
) -> list[tuple[int, int, int, int, float]]:
    """Shift spot coordinates. Drop spots that move out of bounds."""
```

**`apply_deformation_to_spots(spots, field, shape)`**
```python
def apply_deformation_to_spots(
    spots: list[tuple[int, int, int, int, float]],  # (z, y, x, intensity, sigma)
    field: np.ndarray,                               # (Z, Y, X, 3) → (dz, dy, dx)
    shape: tuple[int, int, int],                     # (Z, Y, X) volume bounds
) -> list[tuple[int, int, int, int, float]]:
    """Move spots by sampling displacement field at their positions. Drop out-of-bounds."""
```

**`generate_registration_benchmark()`** — thin wrapper for single-channel ref/mov pair generation (replacing `generate_synthetic_benchmark()` from data.py).

### Removed (no longer needed):

- `create_shifted_stack()` — replaced by `apply_shift_to_spots()` + re-render
- `apply_global_shift()` (from data.py) — same replacement
- `apply_deformation_field()` (from data.py) — replaced by `apply_deformation_to_spots()` + re-render
- `create_benchmark_volume()` (from data.py) — use `create_test_volume()` instead

### Refactored `generate_synthetic_dataset()`:

The per-round rendering pipeline changes to:
```python
# 1. Generate base spot info with reference positions (once per FOV)
# 2. For each round:
#    a. Apply per-round intensity/sigma jitter
#    b. Apply shift (coordinate transform)
#    c. Apply deformation if configured (coordinate transform)
#    d. Render clean Gaussians at transformed positions
for round_idx in range(1, config.n_rounds + 1):
    shift = tuple(shifts[round_id])
    for ch in range(config.n_channels):
        # Collect spots for this channel, apply per-round jitter
        channel_spots = []
        for spot in spots_info:
            if COLOR_TO_CHANNEL[spot["color_seq"][round_idx - 1]] == ch:
                jitter_rng = np.random.default_rng(config.seed + spot["id"] * 100 + round_idx)
                jittered_intensity = int(spot["intensity"] * (1 + jitter_rng.normal(0, 0.1)))
                jittered_sigma = config.spot_sigma * (1 + jitter_rng.normal(0, 0.05))
                z, y, x = spot["position"]
                channel_spots.append((z, y, x, jittered_intensity, jittered_sigma))

        # Apply coordinate transforms (shift, then deformation if active)
        if round_idx > 1:
            channel_spots = apply_shift_to_spots(channel_spots, shift, shape)
            if config.deformation and deformation_field is not None:
                channel_spots = apply_deformation_to_spots(channel_spots, deformation_field, shape)

        # Render clean Gaussians
        image = create_test_image_stack(shape, channel_spots, ...)
```

### SyntheticConfig additions:

```python
@dataclass
class SyntheticConfig:
    # ... existing fields ...

    # Deformation (optional, for rounds 2+)
    deformation: str | None = None  # None, or a key from DEFORMATION_CONFIGS
```

### Ground truth changes:

For rounds with deformation, `ground_truth.json` stores:
```json
{
  "fovs": {
    "FOV_001": {
      "shifts": {"round1": [0,0,0], "round2": [3, -2, 5], ...},
      "deformations": {
        "round2": {
          "type": "polynomial_small",
          "field_file": "FOV_001/round2/field.npy",
          "max_displacement": 7.7
        }
      },
      "spots": [...]
    }
  }
}
```

### Registration benchmark output structure (unchanged):

```
output_dir/synthetic/{preset}/
├── ref.tif
├── mov_shift.tif
├── mov_deform_{name}.tif
├── field_{name}.npy
├── inspection_*.png
└── ground_truth.json
```

**File:** `src/python/starfinder/benchmark/synthetic.py` (new)

---

## Step 3: Move `validation.py` → `starfinder/benchmark/validation.py`

Move `testdata/validation.py` to `benchmark/validation.py` with no functional changes.

Functions: `compare_shifts()`, `compare_spots()`, `compare_genes()`, `e2e_summary()`

**File:** `src/python/starfinder/benchmark/validation.py` (new, moved from testdata)

---

## Step 4: Slim down `starfinder/benchmark/data.py`

After absorbing its generation code into `synthetic.py`, `data.py` retains only:
- `generate_inspection_image()` — 4-panel green-magenta PNG
- `generate_overview_grid()` — grid of inspection PNGs
- `extract_real_benchmark_data()` / `_load_round_mip()` — real data extraction
- `REAL_DATASETS` — real dataset configs

Remove from `data.py`:
- `BenchmarkDataConfig` — replaced by `SyntheticConfig`
- `create_benchmark_volume()` — replaced by `create_test_volume()`
- `apply_global_shift()` — replaced by `apply_shift_to_spots()`
- `create_deformation_field()` — moved to synthetic.py
- `apply_deformation_field()` — replaced by `apply_deformation_to_spots()`
- `generate_synthetic_benchmark()` — replaced by `generate_registration_benchmark()`
- `DEFORMATION_CONFIGS` / `scale_deformation_config()` — moved to synthetic.py

**File:** `src/python/starfinder/benchmark/data.py`

---

## Step 5: Update `starfinder/benchmark/__init__.py`

Add new exports from `synthetic.py` and `validation.py`. Remove deleted exports from `data.py`.

New exports:
```python
from starfinder.benchmark.synthetic import (
    SyntheticConfig, TEST_CODEBOOK, DEFORMATION_CONFIGS,
    generate_codebook, encode_barcode_to_colors, get_preset_config,
    generate_synthetic_dataset, generate_registration_benchmark,
    create_test_image_stack, create_test_volume,
    create_deformation_field, apply_shift_to_spots, apply_deformation_to_spots,
    scale_deformation_config,
)
from starfinder.benchmark.validation import (
    compare_shifts, compare_spots, compare_genes, e2e_summary,
)
```

Remove from `data.py` imports: `create_benchmark_volume`, `apply_global_shift`, `apply_deformation_field`, `generate_synthetic_benchmark`, `DEFORMATION_CONFIGS`.

**File:** `src/python/starfinder/benchmark/__init__.py`

---

## Step 6: Delete `starfinder/testdata/` and update all consumers

Delete the entire `testdata/` package (no backward-compat shims). Update all import paths.

### In-repo files (5 files):

1. `test/test_encoding.py:6` → `from starfinder.benchmark.synthetic import encode_barcode_to_colors`
2. `test/test_barcode.py:16` → `from starfinder.benchmark.synthetic import TEST_CODEBOOK`
3. `test/test_e2e.py:14` → `from starfinder.benchmark.validation import compare_genes, compare_shifts, compare_spots`
4. `starfinder/registration/benchmark.py:37` (lazy) → `from starfinder.benchmark.synthetic import create_test_volume`
5. `test/conftest.py` — update skip message to say `python -m starfinder.benchmark`

### Network-mount scripts (5 files):

6. `starfinder_benchmark/e2e/results/large/run_e2e_large.py:72`
   `from starfinder.testdata.validation import ...` → `from starfinder.benchmark.validation import compare_shifts, compare_spots, compare_genes, e2e_summary`

7. `starfinder_benchmark/e2e/results/tissue/run_e2e_tissue.py:73`
   Same change as above.

8. `starfinder_benchmark/e2e/results/thick_medium/run_e2e_thick_medium.py:73`
   Same change as above.

9. `starfinder_benchmark/e2e_LR/results/run_e2e_LR.py:514`
   `from starfinder.testdata.validation import ...` → `from starfinder.benchmark.validation import compare_shifts, compare_spots, compare_genes`

10. `starfinder_benchmark/e2e_LR/data/generate_data.py:23-24`
    `from starfinder.testdata import generate_synthetic_dataset, get_preset_config` → `from starfinder.benchmark.synthetic import generate_synthetic_dataset, get_preset_config`
    `from starfinder.benchmark.data import create_deformation_field, apply_deformation_field` → `from starfinder.benchmark.synthetic import create_deformation_field, apply_deformation_to_spots`

### Delete:

- `src/python/starfinder/testdata/__init__.py`
- `src/python/starfinder/testdata/__main__.py`
- `src/python/starfinder/testdata/synthetic.py`
- `src/python/starfinder/testdata/validation.py`

---

## Step 7: Create `starfinder/benchmark/__main__.py` (new CLI entry point)

New CLI replacing `python -m starfinder.testdata`:
```bash
# E2E multi-round dataset (primary use, default mode)
uv run python -m starfinder.benchmark --preset small --output tests/fixtures/synthetic/small

# Registration benchmark pairs
uv run python -m starfinder.benchmark --mode registration --output benchmark_data/
```

**File:** `src/python/starfinder/benchmark/__main__.py` (new)

---

## Step 8: Update `starfinder/benchmark/runner.py`

`runner.py` lazy-imports `DEFORMATION_CONFIGS` from `starfinder.benchmark.data`. Update to import from `starfinder.benchmark.synthetic`.

**File:** `src/python/starfinder/benchmark/runner.py` (2 lazy import sites)

---

## Step 9: Tests

### Remove old tests:

- `test/test_synthetic.py` — tests pre-generated fixture files (small_dataset, small_ground_truth, etc.). These fixtures will need regeneration after the format change (per-spot sigma in ground truth). Remove this file; the E2E tests in `test_e2e.py` provide better coverage.

### New/updated tests in `test/test_benchmark_synthetic.py`:

```python
class TestApplyShiftToSpots:
    def test_spots_shift_correctly(self):
        # Uniform shift → verify positions
    def test_boundary_spots_dropped(self):
        # Shift that pushes spots outside → dropped
    def test_zero_shift_preserves_spots(self):
        # Identity case

class TestApplyDeformationToSpots:
    def test_uniform_field(self):
        # Constant displacement field → all spots shift equally
    def test_boundary_spots_dropped(self):
        # Field pushes boundary spots outside → dropped
    def test_zero_field_preserves_spots(self):
        # Zero field → no change

class TestPerRoundVariation:
    def test_intensity_varies_across_rounds(self):
        # Same spot, different rounds → different intensity
    def test_sigma_varies_across_rounds(self):
        # Same spot, different rounds → different sigma
    def test_variation_is_deterministic(self):
        # Same seed → same jitter

class TestGenerateSyntheticDataset:
    def test_tiny_preset(self, tmp_path):
        # Generate with tiny preset, verify output structure
```

### Update existing tests:

- `test_benchmark.py::TestPresets` — add assertion that `xlarge`/`thick_large` are not in `SIZE_PRESETS`

---

## Verification

1. **Unit tests:** `uv run pytest test/ -v` — all tests pass
2. **Generate small dataset:** `uv run python -m starfinder.benchmark --preset small --output /tmp/test_small` — verify output structure
3. **Generate tiny dataset:** `--preset tiny` — quick sanity check
4. **Registration benchmark:** `--mode registration --preset tiny --output /tmp/test_reg` — verify ref/mov pairs
5. **Visual inspection:** check that deformed images have clean Gaussian spots (no interpolation blur)

---

## Critical Files Summary

| File | Action |
|------|--------|
| `src/python/starfinder/benchmark/presets.py` | Edit: remove xlarge/thick_large |
| `src/python/starfinder/benchmark/synthetic.py` | **New**: unified generator |
| `src/python/starfinder/benchmark/validation.py` | **New**: moved from testdata |
| `src/python/starfinder/benchmark/data.py` | Edit: slim down (keep inspection/real data only) |
| `src/python/starfinder/benchmark/__init__.py` | Edit: update exports |
| `src/python/starfinder/benchmark/__main__.py` | **New**: CLI entry point |
| `src/python/starfinder/benchmark/runner.py` | Edit: update lazy imports |
| `src/python/starfinder/testdata/` | **Delete**: entire package |
| `test/test_encoding.py` | Edit: update import |
| `test/test_barcode.py` | Edit: update import |
| `test/test_e2e.py` | Edit: update import |
| `test/test_synthetic.py` | **Delete**: replaced by test_benchmark_synthetic.py |
| `test/test_benchmark_synthetic.py` | **New**: tests for coordinate transforms + presets |
| `test/test_benchmark.py` | Edit: update preset assertions |
| `starfinder/registration/benchmark.py` | Edit: update lazy import |
| `test/conftest.py` | Edit: update skip message |
| Network-mount scripts (5 files) | Edit: update import paths |
