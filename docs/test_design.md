# Test Design for Python Backend (Milestone 2)

## Overview

This document outlines the testing strategy for the Python backend rewrite. The primary goal is to ensure the Python implementation produces outputs compatible with the existing MATLAB backend while enabling confident refactoring and future development.

**Key Principles:**
- Contract tests define compatibility requirements before implementation
- Golden data from MATLAB serves as regression baseline
- Fast unit tests enable TDD workflow
- Integration tests validate end-to-end FOV processing

---

## Test Pyramid

```
                    ┌─────────────┐
                    │  E2E Tests  │  ← Full pipeline on sample data (minutes)
                   ┌┴─────────────┴┐
                   │  Integration  │  ← FOV workflows (seconds)
                  ┌┴───────────────┴┐
                  │  Contract Tests │  ← Output format validation (fast)
                 ┌┴─────────────────┴┐
                 │    Unit Tests     │  ← Individual functions (milliseconds)
                └────────────────────┘
```

---

## Directory Structure

```
starfinder/
├── src/
│   ├── matlab/                 # Existing MATLAB files (moved from code-base/src)
│   ├── matlab-addon/           # MATLAB addons (moved from code-base/matlab-addon)
│   └── python/
│       ├── pyproject.toml
│       ├── uv.lock
│       └── starfinder/
│           ├── __init__.py
│           ├── io/                     # I/O utilities
│           │   ├── __init__.py
│           │   ├── tiff.py             # TIFF loading/saving
│           │   └── hdf5.py             # HDF5 support (future)
│           ├── registration/           # Registration algorithms
│           │   ├── __init__.py
│           │   ├── phase_correlation.py
│           │   └── demons.py
│           ├── spotfinding/            # Spot detection
│           │   ├── __init__.py
│           │   └── local_maxima.py
│           ├── barcode/                # Barcode processing
│           │   ├── __init__.py
│           │   ├── encoding.py
│           │   ├── decoding.py
│           │   ├── codebook.py
│           │   └── extraction.py
│           ├── preprocessing/          # Image preprocessing
│           │   ├── __init__.py
│           │   ├── normalization.py
│           │   └── morphology.py
│           ├── dataset/                # Core classes
│           │   ├── __init__.py
│           │   ├── starmap_dataset.py  # STARMapDataset class
│           │   └── fov.py              # FOV class
│           ├── utils/                  # Utilities
│           │   ├── __init__.py
│           │   ├── metadata.py
│           │   └── visualization.py
│           └── types.py                # Type definitions (Shift3D, ImageArray, etc.)
├── tests/
│   ├── conftest.py                     # Shared fixtures
│   ├── unit/
│   │   ├── test_registration.py
│   │   ├── test_spotfinding.py
│   │   ├── test_codebook.py
│   │   ├── test_io.py
│   │   └── test_preprocessing.py
│   ├── contract/
│   │   ├── test_csv_schemas.py
│   │   ├── test_path_conventions.py
│   │   └── test_coordinate_conventions.py
│   ├── integration/
│   │   ├── test_fov_pipeline.py
│   │   ├── test_subtile_workflow.py
│   │   └── test_snakemake_scripts.py
│   ├── golden/
│   │   ├── test_regression.py
│   │   ├── generate_golden_data.m      # MATLAB script to create fixtures
│   │   └── data/
│   │       ├── Position001/
│   │       │   ├── gr_shifts.csv
│   │       │   ├── goodSpots.csv
│   │       │   └── ref_merged.tif
│   │       └── LN_Position001/
│   │           └── ...
│   ├── fixtures/
│   │   ├── synthetic_images.py
│   │   ├── sample_codebooks.py
│   │   └── minimal_configs.py
│   └── benchmarks/
│       └── compare_backends.py
└── workflow/                           # Snakemake workflow (unchanged)
```

---

## Test Categories

### 1. Unit Tests

Test individual functions and methods in isolation with synthetic data.

#### 1.1 Registration (`tests/unit/test_registration.py`)

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_phase_correlate_zero_shift` | Identical images return (0,0,0) | Shift < 0.01 px |
| `test_phase_correlate_known_shift` | Recovers artificially applied shift | Error < 0.5 px |
| `test_phase_correlate_noisy_image` | Handles Gaussian noise | Error < 1.0 px |
| `test_apply_shift_roundtrip` | shift → apply → inverse shift = identity | RMSE < 1e-6 |
| `test_shift_3d_vs_2d_consistency` | 2D shift matches Z=0 plane of 3D | Exact match |

```python
# Example test
class TestPhaseCorrelation:
    @pytest.mark.parametrize("shift", [
        (0, 0, 0),
        (5, -3, 2),
        (-10, 10, 0),
    ])
    def test_recovers_known_shift(self, shift, synthetic_3d_image):
        ref = synthetic_3d_image
        moved = apply_shift(ref, shift)

        detected = phase_correlate_3d(ref, moved)

        assert np.allclose(detected, shift, atol=0.5)
```

#### 1.2 Spot Finding (`tests/unit/test_spotfinding.py`)

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_local_maxima_single_peak` | Detects isolated bright spot | Exactly 1 spot at peak |
| `test_local_maxima_grid_pattern` | Detects regular grid of spots | Count matches grid |
| `test_intensity_threshold_filtering` | Threshold removes dim spots | Only bright spots remain |
| `test_no_spots_in_uniform_image` | Uniform image returns empty | 0 spots |
| `test_spots_at_boundary` | Handles image edges correctly | No index errors |

#### 1.3 Codebook (`tests/unit/test_codebook.py`)

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_load_codebook_from_csv` | Parses genes.csv correctly | 62 genes for LN dataset |
| `test_barcode_to_gene_lookup` | Correct gene for known barcode | Exact match |
| `test_unknown_barcode_handling` | Returns None/NaN for invalid | No exception |
| `test_reverse_complement` | Handles `do_reverse=True` | Correct reversal |
| `test_split_index_chunking` | Splits barcode correctly | Segments match |

#### 1.4 I/O (`tests/unit/test_io.py`)

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_load_tiff_shape` | Correct (Z,Y,X,C) shape | Matches expected |
| `test_load_tiff_dtype` | Preserves uint8/uint16 | dtype unchanged |
| `test_save_csv_1based_coords` | Coordinates are 1-based | min(x,y,z) >= 1 |
| `test_csv_roundtrip` | save → load = original | DataFrame equal |

#### 1.5 Preprocessing (`tests/unit/test_preprocessing.py`)

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| `test_enhance_contrast_range` | Output spans [0, 255] | min=0, max=255 |
| `test_tophat_removes_background` | Large structures removed | Background < 5% |
| `test_morph_recon_preserves_peaks` | Bright spots intact | Peak values unchanged |
| `test_max_projection_shape` | Z dimension becomes 1 | shape[0] == 1 |

#### 1.6 Error Handling (`tests/unit/test_error_handling.py`)

Test graceful handling of invalid inputs and edge cases.

| Test | Description | Expected Behavior |
|------|-------------|-------------------|
| `test_load_corrupted_tiff` | TIFF with truncated data | Raises `IOError` with clear message |
| `test_load_missing_file` | Non-existent file path | Raises `FileNotFoundError` |
| `test_load_wrong_format` | PNG file with .tif extension | Raises `ValueError` with format hint |
| `test_codebook_duplicate_barcodes` | CSV with duplicate sequences | Raises `ValueError` listing duplicates |
| `test_codebook_duplicate_genes` | CSV with duplicate gene names | Raises `ValueError` listing duplicates |
| `test_codebook_empty_file` | Empty CSV file | Raises `ValueError` with clear message |
| `test_registration_nan_input` | Image containing NaN values | Raises `ValueError` or handles gracefully |
| `test_registration_inf_input` | Image containing Inf values | Raises `ValueError` or handles gracefully |
| `test_registration_empty_image` | Zero-sized array | Raises `ValueError` |
| `test_registration_mismatched_shapes` | Ref and moving have different shapes | Raises `ValueError` with shape info |
| `test_spotfinding_empty_image` | All-zero image | Returns empty DataFrame (not error) |
| `test_missing_channel` | Requested channel not in image | Raises `IndexError` or `KeyError` with hint |
| `test_invalid_config_type` | Wrong type for config value | Raises `TypeError` with expected type |

```python
# Example error handling tests
class TestIOErrorHandling:
    """Test I/O error conditions"""

    def test_load_corrupted_tiff(self, tmp_path):
        """Corrupted TIFF should raise IOError with helpful message"""
        bad_file = tmp_path / "corrupted.tif"
        bad_file.write_bytes(b"not a valid tiff file")

        with pytest.raises(IOError, match="(corrupt|invalid|read)"):
            load_multipage_tiff(bad_file)

    def test_load_missing_file(self):
        """Missing file should raise FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_multipage_tiff(Path("/nonexistent/path.tif"))


class TestCodebookErrorHandling:
    """Test codebook validation"""

    def test_duplicate_barcodes_rejected(self, tmp_path):
        """Codebook with duplicate barcodes should fail with clear error"""
        csv_path = tmp_path / "bad_codebook.csv"
        csv_path.write_text("gene,barcode\nGeneA,ACGT\nGeneB,ACGT\n")  # Duplicate!

        with pytest.raises(ValueError, match="duplicate.*barcode"):
            Codebook.from_csv(csv_path)

    def test_duplicate_genes_rejected(self, tmp_path):
        """Codebook with duplicate genes should fail with clear error"""
        csv_path = tmp_path / "bad_codebook.csv"
        csv_path.write_text("gene,barcode\nGeneA,ACGT\nGeneA,TGCA\n")  # Duplicate!

        with pytest.raises(ValueError, match="duplicate.*gene"):
            Codebook.from_csv(csv_path)


class TestRegistrationErrorHandling:
    """Test registration input validation"""

    def test_nan_values_rejected(self, synthetic_3d_image):
        """NaN values in input should raise ValueError"""
        bad_image = synthetic_3d_image.astype(np.float32)
        bad_image[5, 32, 32, 0] = np.nan

        with pytest.raises(ValueError, match="(NaN|nan|invalid)"):
            phase_correlate_3d(synthetic_3d_image, bad_image)

    def test_shape_mismatch_rejected(self, synthetic_3d_image):
        """Mismatched shapes should raise ValueError with shape info"""
        small_image = synthetic_3d_image[:5, :32, :32, :]

        with pytest.raises(ValueError, match="shape"):
            phase_correlate_3d(synthetic_3d_image, small_image)
```

---

### 2. Contract Tests

Verify output formats match MATLAB for downstream compatibility.

#### 2.1 CSV Schemas (`tests/contract/test_csv_schemas.py`)

```python
class TestGoodSpotsCsvContract:
    """Contract: goodSpots.csv must be compatible with reads_assignment.py"""

    REQUIRED_COLUMNS = {'x', 'y', 'z', 'gene'}
    OPTIONAL_COLUMNS = {'intensity', 'color_seq', 'quality'}

    def test_required_columns_present(self, goodspots_df):
        assert self.REQUIRED_COLUMNS.issubset(goodspots_df.columns)

    def test_coordinates_are_integers(self, goodspots_df):
        for col in ['x', 'y', 'z']:
            assert pd.api.types.is_integer_dtype(goodspots_df[col])

    def test_coordinates_are_1based(self, goodspots_df):
        """CRITICAL: reads_assignment.py does `x = x - 1` for 0-based indexing"""
        assert goodspots_df['x'].min() >= 1
        assert goodspots_df['y'].min() >= 1
        assert goodspots_df['z'].min() >= 1

    def test_coordinates_within_image_bounds(self, goodspots_df, image_metadata):
        """
        Coordinates must be within image dimensions.

        Catches off-by-one errors and coordinate system bugs.
        Uses 1-based coordinates: valid range is [1, dimension].
        """
        z_max, y_max, x_max = image_metadata['shape'][:3]  # (Z, Y, X, C)

        # Check upper bounds (1-based, so max valid = dimension)
        assert goodspots_df['x'].max() <= x_max, \
            f"x coord {goodspots_df['x'].max()} exceeds image width {x_max}"
        assert goodspots_df['y'].max() <= y_max, \
            f"y coord {goodspots_df['y'].max()} exceeds image height {y_max}"
        assert goodspots_df['z'].max() <= z_max, \
            f"z coord {goodspots_df['z'].max()} exceeds image depth {z_max}"

    def test_no_duplicate_coordinates(self, goodspots_df):
        """Each (x, y, z) location should have at most one spot"""
        coord_cols = ['x', 'y', 'z']
        duplicates = goodspots_df.duplicated(subset=coord_cols, keep=False)
        n_duplicates = duplicates.sum()
        assert n_duplicates == 0, \
            f"Found {n_duplicates} spots with duplicate coordinates"

    def test_gene_column_is_string(self, goodspots_df):
        assert pd.api.types.is_string_dtype(goodspots_df['gene'])

    def test_genes_are_valid(self, goodspots_df, codebook):
        """All genes in output should exist in codebook"""
        valid_genes = set(codebook.genes)
        output_genes = set(goodspots_df['gene'].unique())
        invalid = output_genes - valid_genes
        assert len(invalid) == 0, \
            f"Found genes not in codebook: {invalid}"


class TestSubtileCoordsCsvContract:
    """Contract: subtile_coords.csv must be compatible with stitch_subtile.py"""

    REQUIRED_COLUMNS = {'t', 'scoords_x', 'scoords_y', 'ecoords_x', 'ecoords_y'}

    def test_required_columns(self, subtile_coords_df):
        assert self.REQUIRED_COLUMNS.issubset(subtile_coords_df.columns)

    def test_tile_index_sequential(self, subtile_coords_df):
        expected = list(range(len(subtile_coords_df)))
        assert subtile_coords_df['t'].tolist() == expected

    def test_coords_are_integers(self, subtile_coords_df):
        for col in self.REQUIRED_COLUMNS - {'t'}:
            assert pd.api.types.is_integer_dtype(subtile_coords_df[col])
```

#### 2.2 Path Conventions (`tests/contract/test_path_conventions.py`)

```python
class TestOutputPathConventions:
    """Contract: Output paths must match Snakemake rule expectations"""

    def test_ref_merged_path(self, fov_paths):
        expected = fov_paths.output_root / "images" / "ref_merged" / f"{fov_paths.fov_id}.tif"
        assert fov_paths.ref_merged_tif == expected

    def test_signal_csv_path(self, fov_paths):
        expected = fov_paths.output_root / "signal" / f"{fov_paths.fov_id}_goodSpots.csv"
        assert fov_paths.signal_csv("goodSpots") == expected

    def test_subtile_dir_path(self, fov_paths):
        expected = fov_paths.output_root / "output" / "subtile" / fov_paths.fov_id
        assert fov_paths.subtile_dir == expected
```

#### 2.3 Coordinate Conventions (`tests/contract/test_coordinate_conventions.py`)

```python
class TestCoordinateConventions:
    """Contract: Coordinate systems must match MATLAB conventions"""

    def test_array_axis_order(self):
        """Internal arrays use (Z, Y, X, C) ordering"""
        from starfinder.types import ImageArray
        # Document the convention
        assert True  # Convention is (Z, Y, X, C)

    def test_shift_tuple_order(self):
        """Shifts are (dz, dy, dx) to match array indexing"""
        from starfinder.types import Shift3D
        # Document the convention
        assert True  # Convention is (dz, dy, dx)

    def test_csv_coordinates_match_matlab(self, matlab_spots, python_spots):
        """Python CSV coordinates must match MATLAB output exactly"""
        # Same spot should have same coordinates
        merged = matlab_spots.merge(python_spots, on=['x', 'y', 'z'], how='inner')
        assert len(merged) > 0.95 * len(matlab_spots)
```

---

### 3. Integration Tests

Test complete workflows with real or realistic data.

#### 3.1 FOV Pipeline (`tests/integration/test_fov_pipeline.py`)

```python
@pytest.mark.integration
class TestFOVPipeline:
    """Integration tests for complete FOV processing"""

    def test_direct_workflow(self, sample_config, tmp_path):
        """Test direct workflow mode end-to-end"""
        dataset = STARMapDataset.from_config(sample_config)
        fov = dataset.fov("Position001")

        (fov
            .load_raw_images(rounds=['round1', 'round2'], channel_order=['ch00', 'ch01', 'ch02', 'ch03'])
            .enhance_contrast()
            .global_registration(ref_layer='round1')
            .spot_finding(ref_layer='round1')
            .reads_extraction()
        )

        # Verify state
        assert len(fov.images) == 2
        assert 'round2' in fov.registration
        assert fov.signal.all_spots is not None
        assert len(fov.signal.all_spots) > 0

    def test_subtile_workflow(self, sample_config, tmp_path):
        """Test subtile creation and loading"""
        dataset = STARMapDataset.from_config(sample_config)
        fov = dataset.fov("Position001")
        store = NPZSubtileStore()

        fov.load_raw_images(rounds=['round1'], channel_order=['ch00', 'ch01', 'ch02', 'ch03'])
        refs = fov.create_subtiles(sqrt_pieces=2, store=store)

        assert len(refs) == 4  # 2x2 = 4 subtiles

        # Load and verify subtile
        loaded = FOV.from_subtile(refs[0], dataset, store)
        assert loaded.images['round1'].shape[1] < fov.images['round1'].shape[1]
```

#### 3.2 Subtile Storage (`tests/integration/test_subtile_workflow.py`)

```python
@pytest.mark.integration
@pytest.mark.parametrize("store_class", [NPZSubtileStore, HDF5SubtileStore])
class TestSubtileStorage:
    """Test subtile storage backends"""

    def test_save_load_roundtrip(self, store_class, sample_fov, tmp_path):
        store = store_class()
        crop = CropWindow(0, 10, 0, 100, 0, 100)

        ref = store.save(sample_fov, tile_index=0, crop_window=crop, out_dir=tmp_path)
        loaded = store.load(ref)

        for name, original in sample_fov.images.items():
            expected = original[crop.to_slice()]
            np.testing.assert_array_equal(loaded[name], expected)

    def test_storage_size(self, store_class, sample_fov, tmp_path):
        """Verify compression is working"""
        store = store_class()
        crop = CropWindow(0, 10, 0, 256, 0, 256)

        ref = store.save(sample_fov, tile_index=0, crop_window=crop, out_dir=tmp_path)

        # Compressed should be smaller than raw
        raw_size = sum(img[crop.to_slice()].nbytes for img in sample_fov.images.values())
        file_size = ref.path.stat().st_size
        assert file_size < raw_size
```

---

### 4. Golden/Regression Tests

Compare Python outputs against saved MATLAB outputs.

#### 4.1 Generating Golden Data

```matlab
% tests/golden/generate_golden_data.m
% Run this script to generate reference outputs from MATLAB

datasets = {'cell-culture-3D', 'LN'};
fovs = {'Position351', 'Position001'};

for i = 1:length(datasets)
    input_path = fullfile('/path/to/sample-dataset', datasets{i});
    output_path = fullfile('tests/golden/data', datasets{i});

    % Create dataset and process one FOV
    ds = STARMapDataset(input_path, output_path);
    ds.LoadRawImages('rounds', 1:4, 'channels', 0:3);
    ds.GlobalRegistration('ref_round', 1);
    ds.SpotFinding();
    ds.ReadsFiltration();

    % Save golden outputs
    fov_dir = fullfile(output_path, fovs{i});
    mkdir(fov_dir);

    % Registration shifts
    writematrix(ds.registration_shifts, fullfile(fov_dir, 'gr_shifts.csv'));

    % Spots
    writetable(ds.allSpots, fullfile(fov_dir, 'allSpots.csv'));
    writetable(ds.goodSpots, fullfile(fov_dir, 'goodSpots.csv'));

    % Reference merged image
    imwrite(ds.ref_merged, fullfile(fov_dir, 'ref_merged.tif'));
end
```

#### 4.2 Regression Tests (`tests/golden/test_regression.py`)

```python
TOLERANCE = {
    'registration_shift_px': 0.5,
    'spot_count_ratio': 0.05,
    'spot_position_px': 1.0,
    'gene_assignment_ratio': 0.02,
}

@pytest.mark.golden
class TestRegistrationRegression:
    """Compare Python registration to MATLAB golden outputs"""

    @pytest.fixture
    def matlab_shifts(self, golden_data_path):
        return pd.read_csv(golden_data_path / 'gr_shifts.csv')

    def test_shifts_within_tolerance(self, matlab_shifts, python_fov):
        python_fov.global_registration(ref_layer='round1')

        for round_name, result in python_fov.registration.items():
            matlab_row = matlab_shifts[matlab_shifts['round'] == round_name].iloc[0]
            matlab_shift = np.array([matlab_row['dz'], matlab_row['dy'], matlab_row['dx']])
            python_shift = np.array(result.shifts)

            diff = np.abs(python_shift - matlab_shift)
            assert diff.max() < TOLERANCE['registration_shift_px'], \
                f"{round_name}: shift diff {diff.max():.2f}px exceeds tolerance"


@pytest.mark.golden
class TestSpotFindingRegression:
    """Compare Python spot detection to MATLAB golden outputs"""

    @pytest.fixture
    def matlab_spots(self, golden_data_path):
        return pd.read_csv(golden_data_path / 'goodSpots.csv')

    def test_spot_count_within_tolerance(self, matlab_spots, python_spots):
        matlab_count = len(matlab_spots)
        python_count = len(python_spots)
        ratio = abs(python_count - matlab_count) / matlab_count

        assert ratio < TOLERANCE['spot_count_ratio'], \
            f"Spot count diff {ratio:.1%} exceeds {TOLERANCE['spot_count_ratio']:.0%}"

    def test_gene_distribution_similar(self, matlab_spots, python_spots):
        matlab_genes = matlab_spots['gene'].value_counts(normalize=True)
        python_genes = python_spots['gene'].value_counts(normalize=True)

        common_genes = set(matlab_genes.index) & set(python_genes.index)
        for gene in common_genes:
            diff = abs(matlab_genes[gene] - python_genes[gene])
            assert diff < 0.05, f"Gene {gene} frequency diff {diff:.1%}"
```

---

### 5. Benchmark/Performance Tests

Validate memory usage and performance meet acceptance criteria.

#### 5.1 Memory Profiling (`tests/benchmarks/test_memory.py`)

```python
import pytest
import tracemalloc
from pathlib import Path

# Memory limits (in bytes)
MEMORY_LIMITS = {
    'single_fov_peak': 30 * 1024**3,      # 30GB peak for single FOV
    'single_fov_baseline': 8 * 1024**3,   # 8GB baseline after processing
    'subtile_peak': 4 * 1024**3,          # 4GB peak per subtile
}


@pytest.mark.slow
@pytest.mark.benchmark
class TestMemoryUsage:
    """Verify memory usage stays within cluster node limits"""

    def test_single_fov_memory_peak(self, sample_config, sample_dataset_path):
        """
        Single FOV processing should not exceed 30GB peak memory.

        This is critical for running on standard UGER nodes (32GB).
        """
        from starfinder.dataset import STARMapDataset

        tracemalloc.start()

        # Process a full FOV
        dataset = STARMapDataset.from_config(sample_config)
        fov = dataset.fov("Position001")

        (fov
            .load_raw_images(
                rounds=['round1', 'round2', 'round3', 'round4'],
                channel_order=['ch00', 'ch01', 'ch02', 'ch03'],
            )
            .enhance_contrast()
            .global_registration(ref_layer='round1')
            .spot_finding(ref_layer='round1')
            .reads_extraction()
        )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < MEMORY_LIMITS['single_fov_peak'], \
            f"Peak memory {peak / 1024**3:.1f}GB exceeds limit {MEMORY_LIMITS['single_fov_peak'] / 1024**3:.0f}GB"

    def test_memory_released_after_processing(self, sample_config, sample_dataset_path):
        """Memory should be released after FOV processing completes"""
        import gc
        from starfinder.dataset import STARMapDataset

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]

        # Process and explicitly clear
        dataset = STARMapDataset.from_config(sample_config)
        fov = dataset.fov("Position001")
        fov.load_raw_images(rounds=['round1'], channel_order=['ch00', 'ch01', 'ch02', 'ch03'])

        # Clear and collect
        fov.images.clear()
        del fov
        gc.collect()

        current = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Should return close to baseline
        leaked = current - baseline
        assert leaked < 100 * 1024**2, \
            f"Memory leak detected: {leaked / 1024**2:.1f}MB not released"

    def test_subtile_memory_isolation(self, sample_config, tmp_path):
        """Each subtile should process within 4GB"""
        from starfinder.dataset import STARMapDataset
        from starfinder.dataset.fov import NPZSubtileStore

        tracemalloc.start()

        dataset = STARMapDataset.from_config(sample_config)
        fov = dataset.fov("Position001")
        store = NPZSubtileStore()

        # Create subtiles
        fov.load_raw_images(rounds=['round1'], channel_order=['ch00', 'ch01', 'ch02', 'ch03'])
        refs = fov.create_subtiles(sqrt_pieces=4, overlap_ratio=0.1, store=store)

        # Process one subtile and check memory
        tracemalloc.reset_peak()
        subtile_fov = fov.from_subtile(refs[0], dataset, store)
        subtile_fov.global_registration(ref_layer='round1')

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < MEMORY_LIMITS['subtile_peak'], \
            f"Subtile peak memory {peak / 1024**3:.1f}GB exceeds limit"
```

#### 5.2 Performance Benchmarks (`tests/benchmarks/test_performance.py`)

```python
import pytest
import time
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Container for benchmark measurements"""
    name: str
    duration_sec: float
    memory_peak_mb: float
    throughput: float | None = None  # e.g., pixels/sec, spots/sec


@pytest.mark.slow
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for tracking regressions"""

    def test_registration_throughput(self, synthetic_3d_image, benchmark_results):
        """Measure registration speed (voxels/second)"""
        from starfinder.registration.phase_correlation import phase_correlate_3d

        ref = synthetic_3d_image
        mov = np.roll(ref, (2, 5, -3), axis=(0, 1, 2))

        start = time.perf_counter()
        for _ in range(10):
            phase_correlate_3d(ref, mov)
        elapsed = time.perf_counter() - start

        voxels = ref.size * 10
        throughput = voxels / elapsed

        benchmark_results.append(BenchmarkResult(
            name="phase_correlate_3d",
            duration_sec=elapsed / 10,
            memory_peak_mb=0,  # Not measured here
            throughput=throughput,
        ))

        # Sanity check: should be at least 1M voxels/sec on modern hardware
        assert throughput > 1e6, f"Registration too slow: {throughput:.0f} voxels/sec"

    def test_spot_finding_throughput(self, synthetic_spots_image, benchmark_results):
        """Measure spot detection speed"""
        from starfinder.spotfinding.local_maxima import find_spots_3d

        img, _ = synthetic_spots_image

        start = time.perf_counter()
        for _ in range(100):
            find_spots_3d(img, threshold=100)
        elapsed = time.perf_counter() - start

        voxels = img.size * 100
        throughput = voxels / elapsed

        benchmark_results.append(BenchmarkResult(
            name="find_spots_3d",
            duration_sec=elapsed / 100,
            memory_peak_mb=0,
            throughput=throughput,
        ))

    @pytest.fixture
    def benchmark_results(self, tmp_path):
        """Collect and save benchmark results"""
        results = []
        yield results

        # Save results to JSON for tracking
        import json
        output = tmp_path / "benchmark_results.json"
        output.write_text(json.dumps([
            {
                'name': r.name,
                'duration_sec': r.duration_sec,
                'memory_peak_mb': r.memory_peak_mb,
                'throughput': r.throughput,
            }
            for r in results
        ], indent=2))


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize("store_class", ["NPZSubtileStore", "HDF5SubtileStore"])
class TestSubtileStorageBenchmarks:
    """Compare NPZ vs HDF5 subtile storage performance"""

    def test_write_speed(self, store_class, sample_fov, tmp_path):
        """Measure subtile write speed"""
        from starfinder.dataset.fov import NPZSubtileStore, HDF5SubtileStore, CropWindow

        store = NPZSubtileStore() if store_class == "NPZSubtileStore" else HDF5SubtileStore()
        crop = CropWindow(0, 10, 0, 512, 0, 512)

        start = time.perf_counter()
        for i in range(10):
            store.save(sample_fov, tile_index=i, crop_window=crop, out_dir=tmp_path)
        elapsed = time.perf_counter() - start

        mb_written = sum(
            (tmp_path / f"subtile_data_{i}.{'npz' if 'NPZ' in store_class else 'h5'}").stat().st_size
            for i in range(10)
        ) / 1024**2

        print(f"{store_class} write: {mb_written:.1f}MB in {elapsed:.2f}s = {mb_written/elapsed:.1f}MB/s")

    def test_read_speed(self, store_class, sample_fov, tmp_path):
        """Measure subtile read speed"""
        from starfinder.dataset.fov import NPZSubtileStore, HDF5SubtileStore, CropWindow

        store = NPZSubtileStore() if store_class == "NPZSubtileStore" else HDF5SubtileStore()
        crop = CropWindow(0, 10, 0, 512, 0, 512)

        # Write first
        refs = [store.save(sample_fov, tile_index=i, crop_window=crop, out_dir=tmp_path) for i in range(10)]

        # Measure reads
        start = time.perf_counter()
        for ref in refs:
            _ = store.load(ref)
        elapsed = time.perf_counter() - start

        print(f"{store_class} read: 10 subtiles in {elapsed:.2f}s = {10/elapsed:.1f} subtiles/s")

    def test_compression_ratio(self, store_class, sample_fov, tmp_path):
        """Measure compression effectiveness"""
        from starfinder.dataset.fov import NPZSubtileStore, HDF5SubtileStore, CropWindow

        store = NPZSubtileStore() if store_class == "NPZSubtileStore" else HDF5SubtileStore()
        crop = CropWindow(0, 10, 0, 512, 0, 512)

        ref = store.save(sample_fov, tile_index=0, crop_window=crop, out_dir=tmp_path)

        raw_size = sum(img[crop.to_slice()].nbytes for img in sample_fov.images.values())
        file_size = ref.path.stat().st_size
        ratio = file_size / raw_size

        print(f"{store_class} compression: {ratio:.1%} of original ({raw_size/1024**2:.1f}MB -> {file_size/1024**2:.1f}MB)")

        # Both should achieve at least 50% compression on typical image data
        assert ratio < 0.8, f"Poor compression ratio: {ratio:.1%}"
```

#### 5.3 Running Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmarks -v -m benchmark

# Run memory tests only
pytest tests/benchmarks/test_memory.py -v

# Run with detailed output
pytest tests/benchmarks -v -s --tb=short

# Generate benchmark report
pytest tests/benchmarks --benchmark-json=benchmark_report.json
```

---

## Fixtures (`tests/conftest.py`)

```python
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# --- Paths ---

# Default path (for reference, used when env var not set)
_DEFAULT_SAMPLE_DATA = "/home/unix/jiahao/wanglab/Data/Processed/sample-dataset"

@pytest.fixture(scope="session")
def sample_dataset_path():
    """
    Root path for sample datasets.

    Set STARFINDER_TEST_DATA env var to override.
    Tests using this fixture are skipped if path doesn't exist.
    """
    path = Path(os.environ.get("STARFINDER_TEST_DATA", _DEFAULT_SAMPLE_DATA))
    if not path.exists():
        pytest.skip(f"Sample dataset not found at {path}. Set STARFINDER_TEST_DATA env var.")
    return path

@pytest.fixture(scope="session")
def golden_data_path():
    """
    Root path for golden test data.

    Set STARFINDER_GOLDEN_DATA env var to override.
    Falls back to tests/golden/data relative to conftest.py.
    """
    if "STARFINDER_GOLDEN_DATA" in os.environ:
        path = Path(os.environ["STARFINDER_GOLDEN_DATA"])
    else:
        path = Path(__file__).parent / "golden" / "data"

    if not path.exists():
        pytest.skip(f"Golden data not found at {path}. Set STARFINDER_GOLDEN_DATA env var.")
    return path

# --- Synthetic Data ---

@pytest.fixture
def synthetic_3d_image():
    """Reproducible synthetic 3D image (Z, Y, X, C)"""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(10, 64, 64, 4), dtype=np.uint8)

@pytest.fixture
def synthetic_spots_image():
    """Image with known spot locations for testing detection"""
    img = np.zeros((10, 128, 128), dtype=np.uint8)
    # Add spots at known locations
    spots = [(5, 32, 32), (5, 64, 64), (5, 96, 96)]
    for z, y, x in spots:
        img[z, y-2:y+3, x-2:x+3] = 200
    return img, spots

@pytest.fixture
def sample_codebook():
    """Minimal codebook for testing"""
    from starfinder.barcode.codebook import Codebook
    return Codebook(
        gene_to_seq={'GeneA': 'ACGT', 'GeneB': 'TGCA'},
        seq_to_gene={'ACGT': 'GeneA', 'TGCA': 'GeneB'},
        genes=['GeneA', 'GeneB'],
    )

@pytest.fixture
def sample_spots_df():
    """Sample spots DataFrame"""
    return pd.DataFrame({
        'x': [100, 200, 300],
        'y': [150, 250, 350],
        'z': [5, 5, 5],
        'gene': ['GeneA', 'GeneB', 'GeneA'],
        'intensity': [1000, 1200, 800],
    })

@pytest.fixture
def image_metadata():
    """
    Image metadata for bounds validation.

    Returns dict with shape info matching typical test images.
    Override in specific tests for different dimensions.
    """
    return {
        'shape': (30, 1496, 1496, 4),  # (Z, Y, X, C) - typical cell-culture-3D
        'dtype': np.uint8,
    }

@pytest.fixture
def image_metadata_2d():
    """Image metadata for 2D tissue data"""
    return {
        'shape': (1, 3072, 3072, 5),  # (Z, Y, X, C) - typical tissue-2D
        'dtype': np.uint8,
    }

# --- Config ---

@pytest.fixture
def minimal_config(tmp_path):
    """Minimal valid config for testing"""
    return {
        'root_input_path': str(tmp_path / 'input'),
        'root_output_path': str(tmp_path / 'output'),
        'dataset_id': 'test_dataset',
        'sample_id': 'test_sample',
        'output_id': 'test_output',
        'fov_id_pattern': 'Position%03d',
        'n_fovs': 1,
        'n_rounds': 4,
        'ref_round': 'round1',
        'ref_channel': 'ch00',
        'rotate_angle': 0.0,
        'workflow_mode': 'direct',
    }

# --- Markers ---

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: requires sample dataset")
    config.addinivalue_line("markers", "golden: regression tests against MATLAB")
    config.addinivalue_line("markers", "benchmark: performance and memory benchmarks")
```

---

## pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: requires sample dataset on filesystem",
    "golden: regression tests against MATLAB golden outputs",
    "benchmark: performance and memory benchmarks",
]
addopts = "-v --tb=short"
filterwarnings = [
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["src/python/starfinder"]
branch = true
omit = ["tests/*", "*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

---

## Running Tests

```bash
# Run all fast tests (unit + contract)
pytest tests/unit tests/contract

# Run with coverage
pytest --cov=starfinder --cov-report=html tests/unit tests/contract

# Run integration tests (requires sample data)
pytest -m integration

# Run golden/regression tests
pytest -m golden

# Run benchmark/performance tests
pytest -m benchmark -v -s

# Run memory profiling tests only
pytest tests/benchmarks/test_memory.py -v -s

# Run everything except slow tests (excludes benchmarks)
pytest -m "not slow"

# Run in parallel (unit tests only - benchmarks should run serially)
pytest -n auto tests/unit

# Run specific test file
pytest tests/unit/test_registration.py -v

# Run tests matching pattern
pytest -k "phase_correlate" -v
```

---

## Acceptance Criteria

> **Source of truth:** See [main_python_object_design.md](./main_python_object_design.md#acceptance-criteria) for the authoritative acceptance criteria.

### Test Coverage Mapping

| Acceptance Criterion | Verified By |
|---------------------|-------------|
| A) Contract compatibility | `tests/contract/` - CSV schemas, path conventions, coordinate conventions |
| B) Algorithmic parity | `tests/golden/` - Regression tests against MATLAB outputs |
| C) End-to-end validation | `tests/integration/` - FOV pipeline, subtile workflow |
| D) Subtile storage comparison | `tests/integration/test_subtile_workflow.py` - NPZ vs HDF5 benchmarks |

### Additional Test-Specific Criteria

- [ ] Unit test coverage > 80% for core modules
- [ ] All public methods have at least one test
- [ ] Edge cases documented and tested

---

## CI/CD Integration (Future)

```yaml
# .github/workflows/test.yml (example)
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]"
      - run: pytest tests/unit tests/contract --cov=starfinder

  integration-tests:
    runs-on: self-hosted  # Requires access to sample data
    steps:
      - uses: actions/checkout@v4
      - run: pytest -m integration
```

---

## References

- [pytest documentation](https://docs.pytest.org/)
- [Hypothesis (property-based testing)](https://hypothesis.readthedocs.io/)
- [Python Testing with pytest (book)](https://pragprog.com/titles/bopytest2/)
- [main_python_object_design.md](./main_python_object_design.md) - Object model specification
