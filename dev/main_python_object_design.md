# Python Dataset/FOV Object Design (Milestone 2)

## Context + Objectives

STARfinder is currently orchestrated by MATLAB scripts that instantiate `STARMapDataset` per FOV (e.g., `workflow/scripts/rsf_single_fov.m`). Milestone 2 rewrites the backend in Python while keeping Snakemake workflows runnable end-to-end on cluster.

**Design goals:**
- One sample per run: `STARMapDataset` represents a single `{dataset_id}/{sample_id}/{output_id}` context.
- One FOV per job: `FOV` is the main object; each Snakemake rule operates on exactly one `FOV` (or one subtile).
- Preserve I/O contracts so existing downstream steps keep working (notably `stitch_subtile.py` and `reads_assignment.py`).
- Make subtile intermediate storage pluggable (NPZ vs HDF5) to evaluate cluster performance.
- Type-safe, testable, and memory-conscious design.

**Non-goals (for initial cut):**
- No new workflow modes beyond existing `direct/subtile/deep/free`.
- No breaking changes to output filenames/paths unless explicitly versioned.
- No GPU acceleration (future enhancement).

---

## Pipeline Contracts To Preserve

### Required outputs (current workflow expectations)
| Output | Path Pattern |
|--------|--------------|
| Reference merged image | `{output_root}/images/ref_merged/{fov_id}.tif` |
| Final reads CSV | `{output_root}/signal/{fov_id}_goodSpots.csv` |
| Final reads preview | `{output_root}/signal/{fov_id}_goodSpots.png` |
| Subtile coordinates | `{output_root}/output/subtile/{fov_id}/subtile_coords.csv` |
| Subtile reads | `{output_root}/output/subtile/{fov_id}/subtile_goodSpots_{t}.csv` |

### CSV schema expectations

**`goodSpots.csv`** columns (minimum):
```
x, y, z, gene
```

**Coordinate indexing convention:**
- Internal (Python): 0-based indexing for NumPy arrays
- CSV outputs: **1-based integer** coordinates for MATLAB compatibility
  - Downstream `reads_assignment.py` does `reads_df['x'] = reads_df['x'] - 1`

**`subtile_coords.csv`** columns (required by `stitch_subtile.py`):
```
t, scoords_x, scoords_y, ecoords_x, ecoords_y
```

---

## Core Conventions

### Array axis ordering

**Decision: Use `(Z, Y, X, C)` ordering** (volumetric-first, channel-last)

| Axis | Index | Description |
|------|-------|-------------|
| Z | 0 | Depth/slice dimension (1 for 2D data) |
| Y | 1 | Row dimension (height) |
| X | 2 | Column dimension (width) |
| C | 3 | Channel dimension |

**Rationale:**
- Matches ITK/SimpleITK conventions (common in medical imaging)
- Compatible with scikit-image functions that expect `(Z, Y, X)` for 3D
- Channel-last is memory-efficient for per-channel operations
- For 2D data, use `Z=1` (shape `(1, Y, X, C)`) for consistent API

**MATLAB to Python coordinate mapping:**
```
MATLAB: image(row, col, z, channel)  →  row=Y, col=X
Python: image[z, y, x, c]
```

### Registration shift convention

**Decision: Store shifts as `(dz, dy, dx)` tuple**

- Matches array indexing order `(Z, Y, X)`
- Positive shift = image moves in positive direction
- All functions return/accept `Shift3D = tuple[float, float, float]`

### Data types

| Data | Type | Notes |
|------|------|-------|
| Raw images | `uint16` (preserve) | Convert to `float32` only for computation |
| Processed images | `uint8` or `uint16` | Explicit conversion with warnings |
| Spot coordinates | `int32` | Integer voxel positions |
| Registration shifts | `float64` | Sub-pixel precision |

---

## Type Definitions

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias
from pathlib import Path
import numpy as np
import pandas as pd

# Type aliases
Shift3D: TypeAlias = tuple[float, float, float]  # (dz, dy, dx)
ImageArray: TypeAlias = np.ndarray  # Shape: (Z, Y, X, C), dtype: uint8|uint16|float32
ChannelOrder: TypeAlias = list[str]  # e.g., ["ch00", "ch01", "ch02", "ch03"]

@dataclass(frozen=True)
class ImageMeta:
    """Immutable metadata for a single image stack."""
    shape: tuple[int, int, int, int]  # (Z, Y, X, C)
    dtype: np.dtype
    path: Path
    round_name: str

    @property
    def n_slices(self) -> int: return self.shape[0]
    @property
    def height(self) -> int: return self.shape[1]
    @property
    def width(self) -> int: return self.shape[2]
    @property
    def n_channels(self) -> int: return self.shape[3]
    @property
    def is_2d(self) -> bool: return self.n_slices == 1

@dataclass
class LayerState:
    """Tracks which rounds belong to which layer category."""
    seq: list[str] = field(default_factory=list)      # Sequencing rounds
    other: list[str] = field(default_factory=list)    # Non-sequencing (protein, organelle)
    ref: str | None = None                             # Reference round name

    @property
    def all(self) -> list[str]:
        """All registered layers in order."""
        return self.seq + self.other

@dataclass(frozen=True)
class RegistrationResult:
    """Immutable result of registering one round to reference."""
    round_name: str
    shifts: Shift3D
    diffphase: float
    method: Literal["global", "local"]
    ref_round: str

@dataclass
class SignalState:
    """Mutable container for detected spots and reads."""
    all_spots: pd.DataFrame | None = None   # Raw detected spots
    good_spots: pd.DataFrame | None = None  # Filtered spots with gene assignments
    scores: list[str] = field(default_factory=list)  # QC score logs
    codebook: Codebook | None = None

@dataclass(frozen=True)
class CropWindow:
    """Immutable crop region for subtile extraction."""
    z_start: int
    z_end: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int

    def to_slice(self) -> tuple[slice, slice, slice]:
        """Return slices for array indexing: arr[z, y, x]."""
        return (
            slice(self.z_start, self.z_end),
            slice(self.y_start, self.y_end),
            slice(self.x_start, self.x_end),
        )

@dataclass
class SubtileState:
    """State for subtile-based processing."""
    index: int | None = None                    # Current subtile index (0-based)
    coords_df: pd.DataFrame | None = None       # All subtile coordinates
    crop_window: CropWindow | None = None       # Current crop window

@dataclass(frozen=True)
class SubtileRef:
    """Reference to a saved subtile for later loading."""
    fov_id: str
    tile_index: int
    path: Path
    format: Literal["npz", "hdf5"]

@dataclass
class Codebook:
    """Barcode-to-gene mapping."""
    gene_to_seq: dict[str, str]  # gene_name -> color_sequence
    seq_to_gene: dict[str, str]  # color_sequence -> gene_name
    genes: list[str]             # Ordered gene list

    @classmethod
    def from_csv(cls, path: Path, split_index: list[int] | None = None,
                 do_reverse: bool = True) -> Codebook:
        """Load codebook from CSV file."""
        ...
```

---

## Object Model

### `STARMapDataset` (sample-level factory)

Lightweight, immutable configuration holder that creates FOV instances.

```python
@dataclass(frozen=True)
class STARMapDataset:
    """
    Sample-level configuration and FOV factory.

    Immutable after creation. Thread-safe for parallel FOV processing.
    """
    # Paths
    input_root: Path      # {root_input_path}/{dataset_id}/{sample_id}
    output_root: Path     # {root_output_path}/{dataset_id}/{output_id}

    # Sample metadata
    dataset_id: str
    sample_id: str
    output_id: str

    # Round configuration
    seq_rounds: tuple[str, ...]   # Immutable sequence
    ref_round: str
    ref_channel: str

    # Processing parameters
    rotate_angle: float = 0.0
    maximum_projection: bool = False
    fov_pattern: str = "Position%03d"  # or "tile_%d"

    # Cached resources (lazy-loaded)
    _codebook_cache: Codebook | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config: dict) -> STARMapDataset:
        """
        Create dataset from validated config dict.

        Args:
            config: Snakemake config dict (already validated by schema)

        Returns:
            Immutable STARMapDataset instance
        """
        return cls(
            input_root=Path(config["root_input_path"]) / config["dataset_id"] / config["sample_id"],
            output_root=Path(config["root_output_path"]) / config["dataset_id"] / config["output_id"],
            dataset_id=config["dataset_id"],
            sample_id=config["sample_id"],
            output_id=config["output_id"],
            seq_rounds=tuple(f"round{i}" for i in range(1, config["n_rounds"] + 1)),
            ref_round=config["ref_round"],
            ref_channel=config.get("ref_channel", "ch00"),
            rotate_angle=config.get("rotate_angle", 0.0),
            maximum_projection=config.get("maximum_projection", False),
            fov_pattern=config["fov_id_pattern"],
        )

    def fov(self, fov_id: str) -> FOV:
        """Create a new FOV instance for processing."""
        return FOV(dataset=self, fov_id=fov_id)

    def fov_ids(self, n_fovs: int, start: int = 0) -> list[str]:
        """Generate FOV ID list based on pattern."""
        return [self.fov_pattern % i for i in range(start, start + n_fovs)]

    def load_codebook(self, path: Path, split_index: list[int] | None = None,
                      do_reverse: bool = True) -> Codebook:
        """Load and cache codebook (thread-safe singleton pattern)."""
        if self._codebook_cache is None:
            object.__setattr__(self, '_codebook_cache',
                              Codebook.from_csv(path, split_index, do_reverse))
        return self._codebook_cache
```

### `FOV` (per-FOV stateful processor)

Main workhorse class. Mutable state, fluent API for chaining.

```python
@dataclass
class FOV:
    """
    Per-FOV processing state and methods.

    Mutable. NOT thread-safe. One instance per Snakemake job.
    """
    dataset: STARMapDataset
    fov_id: str

    # Mutable state
    images: dict[str, ImageArray] = field(default_factory=dict)
    metadata: dict[str, ImageMeta] = field(default_factory=dict)
    layers: LayerState = field(default_factory=LayerState)
    registration: dict[str, RegistrationResult] = field(default_factory=dict)
    signal: SignalState = field(default_factory=SignalState)
    subtile: SubtileState = field(default_factory=SubtileState)

    # --- Path helpers ---

    @property
    def paths(self) -> FOVPaths:
        """Lazy path helper."""
        return FOVPaths(self.dataset.output_root, self.fov_id)

    def input_dir(self, round_name: str) -> Path:
        """Input directory for a specific round."""
        return self.dataset.input_root / round_name / self.fov_id

    # --- Image loading ---

    def load_raw_images(
        self,
        rounds: list[str],
        channel_order: ChannelOrder,
        *,
        layer_slot: Literal["seq", "other"] = "seq",
        z_range: tuple[int, int] | None = None,
        convert_uint8: bool = False,
        rotate_angle: float | None = None,
        flip: Literal["horizontal", "vertical"] | None = None,
    ) -> FOV:
        """
        Load raw TIFF stacks for specified rounds.

        Returns self for fluent chaining.
        """
        ...
        return self

    # --- Preprocessing ---

    def enhance_contrast(
        self,
        method: Literal["min-max", "clahe"] = "min-max",
        layers: list[str] | None = None,
    ) -> FOV:
        """Apply contrast enhancement. Mutates images in-place."""
        ...
        return self

    def hist_equalize(
        self,
        ref_layer: str,
        ref_channel: int = 0,
        nbins: int = 64,
        layers: list[str] | None = None,
    ) -> FOV:
        """Histogram matching to reference. Mutates images in-place."""
        ...
        return self

    def morph_recon(self, radius: int = 3, layers: list[str] | None = None) -> FOV:
        """Morphological reconstruction for background subtraction."""
        ...
        return self

    def tophat(self, radius: int = 3, layers: list[str] | None = None) -> FOV:
        """White tophat filtering."""
        ...
        return self

    def make_projection(self, method: Literal["max", "sum"] = "max") -> FOV:
        """Apply Z-projection to all images. Converts (Z,Y,X,C) -> (1,Y,X,C)."""
        ...
        return self

    # --- Registration ---

    def global_register(
        self,
        ref_layer: str,
        *,
        layers_to_register: list[str] | None = None,
        ref_img: Literal["merged", "single-channel"] = "merged",
        mov_img: Literal["merged", "single-channel"] = "merged",
        ref_channel: int = 0,
        scale: float = 1.0,
        save_shifts: bool = True,
        log_suffix: str = "",
    ) -> FOV:
        """
        Global (rigid) registration using phase correlation.

        Mutates images and populates self.registration.
        """
        ...
        return self

    def local_register(
        self,
        ref_layer: str,
        *,
        layers_to_register: list[str] | None = None,
        iterations: int = 10,
        smoothing: float = 1.0,
    ) -> FOV:
        """
        Local (non-rigid) registration using demons algorithm.

        Mutates images. Does NOT populate self.registration (no simple shift).
        """
        ...
        return self

    # --- Spot finding ---

    def spot_find(
        self,
        ref_layer: str,
        *,
        method: Literal["max3d"] = "max3d",
        intensity_estimation: Literal["adaptive", "global"] = "adaptive",
        intensity_threshold: float = 0.2,
    ) -> FOV:
        """
        Detect spots using 3D local maxima.

        Populates self.signal.all_spots.
        """
        ...
        return self

    def reads_extract(
        self,
        voxel_size: tuple[int, int, int] = (1, 2, 2),
        layers: list[str] | None = None,
    ) -> FOV:
        """
        Extract color sequences from spot locations.

        Adds 'color_seq' column to self.signal.all_spots.
        """
        ...
        return self

    def reads_filter(
        self,
        codebook: Codebook,
        *,
        end_base: str = "G",
        n_segments: int = 1,
        split_index: list[int] | None = None,
        save_scores: bool = True,
    ) -> FOV:
        """
        Filter reads against codebook.

        Populates self.signal.good_spots.
        """
        ...
        return self

    # --- Output ---

    def save_ref_merged(self) -> Path:
        """
        Save reference merged image as TIFF.

        Honors dataset.maximum_projection setting.
        Returns path to saved file.
        """
        ...

    def save_signal(
        self,
        slot: Literal["allSpots", "goodSpots"] = "goodSpots",
        columns: list[str] | None = None,
    ) -> Path:
        """
        Save spots to CSV with 1-based coordinates.

        Returns path to saved file.
        """
        ...

    # --- Subtile operations ---

    def create_subtiles(
        self,
        sqrt_pieces: int,
        *,
        overlap_ratio: float = 0.1,
        store: SubtileStore,
    ) -> list[SubtileRef]:
        """
        Partition FOV into overlapping subtiles and save.

        Populates self.subtile.coords_df.
        Returns list of SubtileRef for downstream rules.
        """
        ...

    @classmethod
    def from_subtile(cls, ref: SubtileRef, dataset: STARMapDataset,
                     store: SubtileStore) -> FOV:
        """Load FOV state from saved subtile."""
        ...


@dataclass(frozen=True)
class FOVPaths:
    """Immutable path helper for consistent output locations."""
    output_root: Path
    fov_id: str

    @property
    def ref_merged_tif(self) -> Path:
        return self.output_root / "images" / "ref_merged" / f"{self.fov_id}.tif"

    @property
    def subtile_dir(self) -> Path:
        return self.output_root / "output" / "subtile" / self.fov_id

    def signal_csv(self, slot: str) -> Path:
        return self.output_root / "signal" / f"{self.fov_id}_{slot}.csv"

    def signal_png(self, slot: str) -> Path:
        return self.output_root / "signal" / f"{self.fov_id}_{slot}.png"

    def shift_log(self, suffix: str = "") -> Path:
        name = f"{self.fov_id}{suffix}.txt" if suffix else f"{self.fov_id}.txt"
        return self.output_root / "log" / "gr_shifts" / name

    def score_log(self, suffix: str = "") -> Path:
        name = f"{self.fov_id}{suffix}.txt" if suffix else f"{self.fov_id}.txt"
        return self.output_root / "log" / "sf_scores" / name
```

---

## Subtile Storage (Protocol-based)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class SubtileStore(Protocol):
    """Protocol for subtile storage backends."""

    def save(
        self,
        fov: FOV,
        tile_index: int,
        crop_window: CropWindow,
        out_dir: Path,
    ) -> SubtileRef:
        """Save subtile data and return reference."""
        ...

    def load(self, ref: SubtileRef) -> dict[str, ImageArray]:
        """Load subtile images by reference."""
        ...


class NPZSubtileStore:
    """NumPy compressed archive storage."""

    compression: bool = True

    def save(self, fov: FOV, tile_index: int, crop_window: CropWindow,
             out_dir: Path) -> SubtileRef:
        path = out_dir / f"subtile_data_{tile_index}.npz"
        slices = crop_window.to_slice()

        data = {name: img[slices] for name, img in fov.images.items()}

        if self.compression:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)

        return SubtileRef(fov.fov_id, tile_index, path, "npz")

    def load(self, ref: SubtileRef) -> dict[str, ImageArray]:
        with np.load(ref.path) as data:
            return {k: data[k] for k in data.files}


class HDF5SubtileStore:
    """HDF5 chunked storage with optional compression."""

    compression: str = "gzip"
    compression_opts: int = 4
    chunk_shape: tuple[int, ...] | None = None  # Auto if None

    def save(self, fov: FOV, tile_index: int, crop_window: CropWindow,
             out_dir: Path) -> SubtileRef:
        import h5py

        path = out_dir / f"subtile_data_{tile_index}.h5"
        slices = crop_window.to_slice()

        with h5py.File(path, "w") as f:
            for name, img in fov.images.items():
                cropped = img[slices]
                chunks = self.chunk_shape or self._auto_chunks(cropped.shape)
                f.create_dataset(
                    name, data=cropped, chunks=chunks,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

        return SubtileRef(fov.fov_id, tile_index, path, "hdf5")

    def load(self, ref: SubtileRef) -> dict[str, ImageArray]:
        import h5py

        with h5py.File(ref.path, "r") as f:
            return {k: f[k][:] for k in f.keys()}

    def _auto_chunks(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute reasonable chunk shape for given array."""
        # Target ~1MB chunks
        return tuple(min(s, 64) for s in shape)


def get_subtile_store(config: dict) -> SubtileStore:
    """Factory for subtile storage based on config."""
    storage_type = config.get("subtile_storage", "npz")

    if storage_type == "npz":
        return NPZSubtileStore()
    elif storage_type == "hdf5":
        return HDF5SubtileStore(
            compression=config.get("hdf5_compression", "gzip"),
            compression_opts=config.get("hdf5_compression_opts", 4),
        )
    else:
        raise ValueError(f"Unknown subtile_storage: {storage_type}")
```

---

## Memory Management

### Strategy for large volumes

```python
from contextlib import contextmanager
from typing import Iterator

class FOV:
    # ... existing methods ...

    def load_raw_images_lazy(
        self,
        rounds: list[str],
        channel_order: ChannelOrder,
        **kwargs,
    ) -> FOV:
        """
        Load images with memory-mapped arrays (read-only).

        Use for inspection/analysis without full memory load.
        """
        for round_name in rounds:
            path = self._find_image_path(round_name)
            # Use tifffile's memory-mapped mode
            self.images[round_name] = tifffile.memmap(path, mode='r')
            self.metadata[round_name] = self._extract_metadata(path)
        return self

    @contextmanager
    def processing_context(self, max_memory_gb: float = 8.0) -> Iterator[FOV]:
        """
        Context manager for memory-conscious processing.

        Clears intermediate data on exit.
        """
        try:
            yield self
        finally:
            # Clear large arrays to free memory
            self.images.clear()
            import gc
            gc.collect()

    def process_in_chunks(
        self,
        chunk_size: int = 10,
        operation: Callable[[ImageArray], ImageArray],
    ) -> None:
        """
        Apply operation to images in Z-chunks to limit memory.

        Useful for preprocessing large 3D stacks.
        """
        for name, img in self.images.items():
            n_slices = img.shape[0]
            result_chunks = []

            for z_start in range(0, n_slices, chunk_size):
                z_end = min(z_start + chunk_size, n_slices)
                chunk = img[z_start:z_end]
                result_chunks.append(operation(chunk))

            self.images[name] = np.concatenate(result_chunks, axis=0)
```

---

## Logging Strategy

```python
import logging
from functools import wraps
from time import perf_counter

# Configure module logger
logger = logging.getLogger("starfinder")

def log_step(func):
    """Decorator to log FOV processing steps."""
    @wraps(func)
    def wrapper(self: FOV, *args, **kwargs):
        step_name = func.__name__
        logger.info(f"[{self.fov_id}] Starting {step_name}")
        start = perf_counter()

        try:
            result = func(self, *args, **kwargs)
            elapsed = perf_counter() - start
            logger.info(f"[{self.fov_id}] Completed {step_name} in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"[{self.fov_id}] Failed {step_name}: {e}")
            raise

    return wrapper

class FOV:
    @log_step
    def global_register(self, ref_layer: str, **kwargs) -> FOV:
        ...

    @log_step
    def spot_find(self, ref_layer: str, **kwargs) -> FOV:
        ...
```

---

## Snakemake Integration Pattern

### Rule script template

```python
#!/usr/bin/env python
"""
workflow/scripts/py_rsf_single_fov.py

Snakemake rule script for single FOV processing (Python backend).
"""
import json
import logging
from pathlib import Path

from starfinder import STARMapDataset, get_subtile_store

# Configure logging for cluster jobs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

def main():
    # 1. Load config (from Snakemake or JSON file)
    config_path = Path(snakemake.input.config)
    with open(config_path) as f:
        config = json.load(f)

    # 2. Create dataset and FOV
    dataset = STARMapDataset.from_config(config)
    fov = dataset.fov(snakemake.wildcards.fovID)

    # 3. Get rule-specific parameters
    rule_config = config["rules"]["rsf_single_fov"]["parameters"]
    gr_config = rule_config["global_registration"]
    sf_config = rule_config["spot_finding"]

    # 4. Execute pipeline (fluent API)
    (fov
        .load_raw_images(
            rounds=list(dataset.seq_rounds),
            channel_order=config["channel_order"],
            rotate_angle=dataset.rotate_angle,
        )
        .enhance_contrast(method="min-max")
        .morph_recon(radius=3)
    )

    # Conditional global registration
    if gr_config.get("run", True):
        fov.global_register(
            ref_layer=gr_config["ref_round"],
            ref_img=gr_config.get("ref_img", "merged"),
            mov_img=gr_config.get("mov_img", "merged"),
        )

    # Save reference merged
    fov.save_ref_merged()

    # Spot finding and filtering
    (fov
        .spot_find(
            ref_layer=dataset.ref_round,
            intensity_threshold=sf_config.get("intensity_threshold", 0.2),
        )
        .reads_extract(voxel_size=tuple(sf_config.get("voxel_size", [1, 2, 2])))
    )

    # Load codebook and filter
    codebook = dataset.load_codebook(
        path=Path(config["codebook_path"]),
        split_index=config.get("split_index"),
    )
    fov.reads_filter(codebook, end_base=config.get("end_base", "G"))

    # Save outputs
    fov.save_signal(slot="goodSpots")

    # Write completion marker
    Path(snakemake.output.log).write_text(f"Completed {fov.fov_id}\n")


if __name__ == "__main__":
    main()
```

### Backend switching in common.smk

```python
# workflow/rules/common.smk

def get_backend(rule_name: str) -> str:
    """Get backend for a specific rule."""
    global_backend = config.get("backend", "matlab")
    return get_rule_config(rule_name, "backend", global_backend)

def get_rule_script(rule_name: str) -> str:
    """Get script path based on backend."""
    backend = get_backend(rule_name)
    if backend == "python":
        return f"workflow/scripts/py_{rule_name}.py"
    else:
        return f"workflow/scripts/{rule_name}.m"
```

---

## MATLAB → Python Method Mapping

| MATLAB method | Python method | Side effects | Outputs |
|---------------|---------------|--------------|---------|
| `STARMapDataset(in, out)` | `STARMapDataset.from_config(cfg).fov(id)` | Init state | None |
| `LoadRawImages(...)` | `fov.load_raw_images(...)` | Populate `images`, `metadata`, `layers` | None |
| `EnhanceContrast(...)` | `fov.enhance_contrast(...)` | Mutate `images` | None |
| `HistEqualize(...)` | `fov.hist_equalize(...)` | Mutate `images` | None |
| `MorphRecon(...)` | `fov.morph_recon(...)` | Mutate `images` | None |
| `Tophat(...)` | `fov.tophat(...)` | Mutate `images` | None |
| `GlobalRegistration(...)` | `fov.global_register(...)` | Mutate `images`, fill `registration` | Optional shift log |
| `LocalRegistration(...)` | `fov.local_register(...)` | Mutate `images` | None |
| `MakeProjection(...)` | `fov.make_projection(...)` | Mutate `images` | None |
| `SaveImages(...)` | `fov.save_ref_merged()` | None | `images/ref_merged/{fov}.tif` |
| `SpotFinding(...)` | `fov.spot_find(...)` | Set `signal.all_spots` | None |
| `ReadsExtraction(...)` | `fov.reads_extract(...)` | Add `color_seq` column | None |
| `LoadCodebook(...)` | `dataset.load_codebook(...)` | Cache codebook | None |
| `ReadsFiltration(...)` | `fov.reads_filter(...)` | Set `signal.good_spots` | Score log |
| `SaveSignal(...)` | `fov.save_signal(...)` | None | `signal/{fov}_{slot}.csv` |
| `CreateSubtiles(...)` | `fov.create_subtiles(...)` | Set `subtile.coords_df` | coords CSV + data files |

---

## Acceptance Criteria

### A) Contract compatibility
- [ ] Python backend produces files at same paths as MATLAB
- [ ] `goodSpots.csv` has columns `x,y,z,gene` with 1-based integer coordinates
- [ ] Existing `stitch_subtile.py` works without modification
- [ ] Existing `reads_assignment.py` works without modification

### B) Algorithmic parity
- [ ] Global registration shifts match MATLAB within ±0.5 pixels
- [ ] Spot counts within ±5% of MATLAB for same parameters
- [ ] Gene assignments match MATLAB for identical spots

### C) End-to-end validation
- [ ] Complete one sample on UGER with Python backend
- [ ] Snakemake `benchmark:` files for all rules
- [ ] No OOM errors on standard node (32GB)

### D) Subtile storage comparison
- [ ] Both NPZ and HDF5 complete subtile workflow
- [ ] Benchmark: read/write time, storage size, concurrent access

---

## Testing Plan

### Unit tests
- `test_registration.py` - Phase correlation accuracy, shift application
- `test_spotfinding.py` - Local maxima detection, thresholding
- `test_barcode.py` - Encoding, decoding, codebook matching
- `test_io.py` - TIFF loading, CSV writing with coordinate conversion

### Contract tests
- `test_output_contracts.py` - CSV schemas, path patterns, coordinate conventions

### Integration tests
- `test_fov_pipeline.py` - Single FOV end-to-end
- `test_subtile_workflow.py` - Create, save, load, process subtiles

### Benchmarks
- Snakemake `benchmark:` directive on all rules
- `benchmarks/compare_backends.py` - Parse and summarize benchmark files
