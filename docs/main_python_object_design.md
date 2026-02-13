# Python Dataset/FOV Object Design (Milestone 2)

> **Revised 2026-02-12 (v3)** — Updated after Phases 0-5 implementation to reflect
> actual APIs, benchmark findings, and architectural decisions for Phase 6.

## Context + Objectives

STARfinder is currently orchestrated by MATLAB scripts that instantiate `STARMapDataset` per FOV (e.g., `workflow/scripts/rsf_single_fov.m`). Milestone 2 rewrites the backend in Python while keeping Snakemake workflows runnable end-to-end on cluster.

**Design goals:**
- One sample per run: `STARMapDataset` represents a single `{dataset_id}/{sample_id}/{output_id}` context.
- One FOV per job: `FOV` is the main object; each Snakemake rule operates on exactly one `FOV` (or one subtile).
- Preserve I/O contracts so existing downstream steps keep working (notably `stitch_subtile.py` and `reads_assignment.py`).
- Type-safe, testable, and minimal design — no speculative abstractions.

**Non-goals (for initial cut):**
- No new workflow modes beyond existing `direct/subtile/deep/free`.
- No breaking changes to output filenames/paths unless explicitly versioned.
- No GPU acceleration (future enhancement).
- No pluggable subtile storage — use NPZ only; refactor if HDF5 proves beneficial later.
- No memory management framework — address if OOM occurs in production.

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

### `all_spots` internal schema

Spot detection through filtration follows a progressive column schema:

| Stage | Columns | Coordinate basis |
|-------|---------|-----------------|
| After `spot_finding()` | `z, y, x, intensity, channel` | 0-based |
| After `reads_extraction()` | + `{round}_color`, `{round}_score` per seq round, + `color_seq` | 0-based |
| After `reads_filtration()` → `good_spots` | + `gene` | 0-based |
| After `save_signal()` → CSV | `x, y, z, gene` (reordered, subset) | **1-based** |

### Shift log CSV format

Global registration shifts are saved to `{output_root}/log/gr_shifts/{fov_id}.txt`:

```
fov_id,round,row,col,z
Position001,round2,12.5,-3.0,1.0
Position001,round3,8.0,0.5,0.0
```

Column mapping from Python's `(dz, dy, dx)` shift convention:
- `row` = dy, `col` = dx, `z` = dz (MATLAB column ordering)

### NPZ subtile file schema

Each subtile is saved as a compressed NPZ with these keys:

| Key | Shape / Type | Description |
|-----|-------------|-------------|
| `images_{round}` | `(Z, Y, X, C)` uint8 | Registered image stack per round |
| `fov_id` | str | Parent FOV identifier |
| `subtile_id` | int | Subtile index (0-based) |
| `layers_seq` | list[str] | Sequencing round names |
| `layers_ref` | str | Reference round name |

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
- Validated across all 5 implemented phases

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
| Raw images | `uint8` (preserve, if not, convert to `uint8`) | Convert to `float32` only for computation if necessary|
| Processed images | `uint8` or `uint16` | Explicit conversion with warnings |
| Spot coordinates | `int32` | Integer voxel positions |
| Registration shifts | `float64` | Sub-pixel precision |

---

## Type Definitions

The dataclass hierarchy is kept minimal. Only types that carry semantic meaning
beyond a plain dict or tuple get their own class.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, TypeAlias
from pathlib import Path
import numpy as np
import pandas as pd

# Type aliases
Shift3D: TypeAlias = tuple[float, float, float]  # (dz, dy, dx)
ImageArray: TypeAlias = np.ndarray  # Shape: (Z, Y, X, C), dtype: uint8|uint16|float32
ChannelOrder: TypeAlias = list[str]  # e.g., ["ch00", "ch01", "ch02", "ch03"]


@dataclass
class LayerState:
    """
    Tracks which rounds belong to which layer category.

    Invariants:
    - `seq` contains sequencing rounds used for barcode decoding
    - `other` contains non-sequencing rounds (protein markers, organelle stains, etc.)
    - `ref` is the reference round for registration; all other rounds are aligned to this
    - `ref` can be any round from `seq` OR `other` (not restricted to sequencing rounds)
    - A round cannot be in both `seq` and `other`

    Example:
        layers = LayerState(
            seq=['round1', 'round2', 'round3', 'round4'],
            other=['protein', 'organelle'],
            ref='round1',
        )
    """
    seq: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    ref: str | None = None

    @property
    def all_layers(self) -> list[str]:
        """All loaded layers in order (seq first, then other)."""
        return self.seq + self.other

    @property
    def to_register(self) -> list[str]:
        """Layers that need registration (all except ref)."""
        return [r for r in self.all_layers if r != self.ref]

    def validate(self) -> None:
        """Check invariants. Raises ValueError if violated."""
        if self.ref is not None and self.ref not in self.all_layers:
            raise ValueError(f"ref '{self.ref}' not found in seq or other")
        overlap = set(self.seq) & set(self.other)
        if overlap:
            raise ValueError(f"Rounds in both seq and other: {overlap}")


@dataclass
class Codebook:
    """
    Barcode-to-gene mapping.

    Wraps the two dicts returned by `starfinder.barcode.load_codebook()` into
    a single object with named access and a convenient factory method.
    """
    gene_to_seq: dict[str, str]  # gene_name -> color_sequence
    seq_to_gene: dict[str, str]  # color_sequence -> gene_name

    @property
    def genes(self) -> list[str]:
        """Ordered gene list."""
        return sorted(self.gene_to_seq.keys())

    @property
    def n_genes(self) -> int:
        return len(self.gene_to_seq)

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        do_reverse: bool = True,
        split_index: int | None = None,
    ) -> Codebook:
        """Load codebook from CSV file.

        Delegates to starfinder.barcode.load_codebook() and wraps the result.
        """
        from starfinder.barcode import load_codebook
        gene_to_seq, seq_to_gene = load_codebook(path, do_reverse=do_reverse,
                                                  split_index=split_index)
        return cls(gene_to_seq=gene_to_seq, seq_to_gene=seq_to_gene)


@dataclass(frozen=True)
class CropWindow:
    """Immutable crop region for subtile extraction (Y/X only; Z is kept whole)."""
    y_start: int
    y_end: int
    x_start: int
    x_end: int

    def to_slice(self) -> tuple[slice, slice]:
        """Return slices for Y/X indexing: arr[:, y, x] or arr[:, y, x, :]."""
        return (
            slice(self.y_start, self.y_end),
            slice(self.x_start, self.x_end),
        )


@dataclass
class SubtileConfig:
    """
    Dataset-level subtile partitioning configuration.

    Computes overlapping 2D windows that tile the Y/X plane.
    Stored on STARMapDataset and shared across all FOVs.
    """
    sqrt_pieces: int                        # Grid is sqrt_pieces × sqrt_pieces
    overlap_ratio: float = 0.1              # Fractional overlap between adjacent tiles
    windows: list[CropWindow] = field(default_factory=list)

    @property
    def n_subtiles(self) -> int:
        """Total number of subtiles."""
        return len(self.windows)

    def compute_windows(self, height: int, width: int) -> None:
        """
        Populate self.windows for a given (Y, X) image size.

        Tiles are sqrt_pieces × sqrt_pieces with overlap_ratio overlap.
        Matches MATLAB CreateSubtiles tiling logic.
        """
        ...
```

---

## Object Model

### `STARMapDataset` (sample-level configuration)

Mutable configuration holder that creates FOV instances. Owns dataset-level
state (layers, codebook, subtile config, channel order) that FOVs inherit
via properties.

```python
@dataclass
class STARMapDataset:
    """
    Sample-level configuration and FOV factory.

    Non-frozen: allows lazy loading of codebook and subtile config after
    construction. FOVs access dataset-level state via delegation.
    """
    # Paths
    input_root: Path      # {root_input_path}/{dataset_id}/{sample_id}
    output_root: Path     # {root_output_path}/{dataset_id}/{output_id}

    # Sample metadata
    dataset_id: str
    sample_id: str
    output_id: str

    # Dataset-level state (shared across FOVs)
    layers: LayerState = field(default_factory=LayerState)
    channel_order: ChannelOrder = field(default_factory=list)
    codebook: Codebook | None = None
    subtile: SubtileConfig | None = None

    # Processing parameters
    rotate_angle: float = 0.0
    maximum_projection: bool = False
    fov_pattern: str = "Position%03d"  # or "tile_%d"

    @classmethod
    def from_config(cls, config: dict) -> STARMapDataset:
        """Create dataset from validated Snakemake config dict.

        Builds LayerState from config round information and populates
        channel_order from config.
        """
        layers = LayerState(
            seq=[f"round{i}" for i in range(1, config["n_rounds"] + 1)],
            ref=config["ref_round"],
        )
        return cls(
            input_root=Path(config["root_input_path"]) / config["dataset_id"] / config["sample_id"],
            output_root=Path(config["root_output_path"]) / config["dataset_id"] / config["output_id"],
            dataset_id=config["dataset_id"],
            sample_id=config["sample_id"],
            output_id=config["output_id"],
            layers=layers,
            channel_order=config.get("channel_order", []),
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

    def load_codebook(
        self,
        path: Path | str,
        split_index: int | None = None,
        do_reverse: bool = True,
    ) -> None:
        """Load codebook from CSV and store on self.codebook."""
        self.codebook = Codebook.from_csv(path, do_reverse=do_reverse,
                                          split_index=split_index)
```

### `FOV` (per-FOV stateful processor)

Main workhorse class. Mutable state, fluent API for chaining.

```python
@dataclass
class FOV:
    """
    Per-FOV processing state and methods.

    Mutable. NOT thread-safe. One instance per Snakemake job.
    Delegates to dataset for layers, codebook, and channel_order.
    """
    dataset: STARMapDataset
    fov_id: str

    # Mutable state — flat attributes, no wrapper dataclasses
    images: dict[str, ImageArray] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)  # from load_image_stacks
    global_shifts: dict[str, Shift3D] = field(default_factory=dict)  # round -> (dz, dy, dx)
    local_registered: set[str] = field(default_factory=set)  # rounds that had demons applied
    all_spots: pd.DataFrame | None = None      # Raw detected spots
    good_spots: pd.DataFrame | None = None     # Filtered spots with gene assignments

    # --- Delegated properties ---

    @property
    def layers(self) -> LayerState:
        """Layer configuration (delegated to dataset)."""
        return self.dataset.layers

    @property
    def codebook(self) -> Codebook | None:
        """Codebook (delegated to dataset)."""
        return self.dataset.codebook

    # --- Path helpers ---

    @property
    def paths(self) -> FOVPaths:
        """Lazy path helper."""
        return FOVPaths(self.dataset.output_root, self.fov_id)

    def input_dir(self, round_name: str) -> Path:
        """Input directory for a specific round."""
        return self.dataset.input_root / round_name / self.fov_id

    # --- Image loading ---

    @log_step
    def load_raw_images(
        self,
        rounds: list[str] | None = None,
        channel_order: ChannelOrder | None = None,
        *,
        convert_uint8: bool = True,
        subdir: str = "",
        layer_slot: Literal["seq", "other"] = "seq",
    ) -> FOV:
        """
        Load raw TIFF stacks for specified rounds.

        Delegates to starfinder.io.load_image_stacks() per round.
        Populates self.images and self.metadata.

        Args:
            rounds: Round names to load. Defaults to self.layers.seq or
                self.layers.other based on layer_slot.
            channel_order: Channel file ordering. Defaults to
                self.dataset.channel_order.
            convert_uint8: Convert to uint8 on load (default True).
            subdir: Subdirectory within each round/fov dir.
            layer_slot: Which layer list to default rounds from.

        Per-round loop:
            for round_name in rounds:
                img, meta = load_image_stacks(
                    self.input_dir(round_name),
                    channel_order=channel_order,
                    subdir=subdir,
                    convert_uint8=convert_uint8,
                )
                self.images[round_name] = img
                self.metadata[round_name] = meta
        """
        ...
        return self

    # --- Preprocessing ---
    # These methods delegate to starfinder.preprocessing functions,
    # applying them to all layers (or a specified subset).

    @log_step
    def enhance_contrast(self, layers: list[str] | None = None) -> FOV:
        """Per-channel min-max normalization. Delegates to min_max_normalize()."""
        ...
        return self

    @log_step
    def hist_equalize(
        self,
        ref_channel: int = 0,
        nbins: int = 64,
        layers: list[str] | None = None,
    ) -> FOV:
        """Histogram matching to reference. Delegates to histogram_match().

        Uses self.layers.ref as the reference layer. Extracts single channel
        from reference image as the histogram template:
            reference = self.images[self.layers.ref][:, :, :, ref_channel]
        """
        ...
        return self

    @log_step
    def morph_recon(self, radius: int = 3, layers: list[str] | None = None) -> FOV:
        """Background removal. Delegates to morphological_reconstruction()."""
        ...
        return self

    @log_step
    def tophat(self, radius: int = 3, layers: list[str] | None = None) -> FOV:
        """White tophat filtering. Delegates to tophat_filter()."""
        ...
        return self

    @log_step
    def make_projection(self, method: Literal["max", "sum"] = "max") -> FOV:
        """Apply Z-projection to all images. Delegates to utils.make_projection()."""
        ...
        return self

    # --- Registration ---

    @log_step
    def global_registration(
        self,
        *,
        layers_to_register: list[str] | None = None,
        ref_img: Literal["merged", "single-channel"] = "merged",
        mov_img: Literal["merged", "single-channel"] = "merged",
        ref_channel: int = 0,
        save_shifts: bool = True,
    ) -> FOV:
        """
        Global (rigid) registration using phase correlation.

        Uses self.layers.ref as the reference round.
        Delegates to starfinder.registration.register_volume().
        Mutates images and stores shifts in self.global_shifts[round_name].

        If save_shifts is True, writes shift log to self.paths.shift_log().
        """
        ...
        return self

    @log_step
    def local_registration(
        self,
        *,
        ref_channel: int = 0,
        layers_to_register: list[str] | None = None,
        iterations: list[int] | None = None,
        smoothing_sigma: float = 1.0,
        method: str = "demons",
        pyramid_mode: str = "antialias",
    ) -> FOV:
        """
        Local (non-rigid) registration using demons algorithm.

        Uses self.layers.ref as the reference round.
        Delegates to starfinder.registration.register_volume_local().
        Mutates images in-place. Displacement fields are ephemeral (not stored).
        Adds round names to self.local_registered set.

        Default parameters match MATLAB imregdemons quality:
        - iterations=[100,50,25]: 3-level pyramid
        - smoothing_sigma=1.0: matches MATLAB AccumulatedFieldSmoothing
        - method="demons": Thirion demons (1.6x faster than diffeomorphic)
        - pyramid_mode="antialias": Butterworth-filtered downsampling
        """
        ...
        return self

    # --- Spot finding ---

    @log_step
    def spot_finding(
        self,
        *,
        intensity_estimation: Literal["adaptive", "global"] = "adaptive",
        intensity_threshold: float = 0.2,
    ) -> FOV:
        """
        Detect spots on the reference round using 3D local maxima.

        Uses self.layers.ref to select the reference image.
        Delegates to starfinder.spotfinding.find_spots_3d().
        Populates self.all_spots with columns [z, y, x, intensity, channel]
        (0-based coordinates).
        """
        ...
        return self

    @log_step
    def reads_extraction(
        self,
        voxel_size: tuple[int, int, int] = (1, 2, 2),  # (dz, dy, dx)
        layers: list[str] | None = None,
    ) -> FOV:
        """
        Extract color sequences from spot locations across sequencing rounds.

        Loops over each sequencing round and calls
        starfinder.barcode.extract_from_location() to get per-round color
        and score. Concatenates per-round colors into a single `color_seq`
        string per spot.

        Args:
            voxel_size: Half-widths for voxel neighborhood, ordered (dz, dy, dx).
            layers: Rounds to extract from. Defaults to self.layers.seq.

        Per-round loop:
            for round_name in layers:
                color, score = extract_from_location(
                    self.images[round_name], self.all_spots, voxel_size
                )
                self.all_spots[f'{round_name}_color'] = color
                self.all_spots[f'{round_name}_score'] = score

            # Concatenate per-round colors into color_seq
            self.all_spots['color_seq'] = (
                sum of {round}_color columns as concatenated string
            )
        """
        ...
        return self

    @log_step
    def reads_filtration(
        self,
        *,
        end_bases: str | None = None,
        start_base: str = "C",
    ) -> FOV:
        """
        Filter reads against codebook.

        Uses self.dataset.codebook (must be loaded first via
        dataset.load_codebook()).
        Delegates to starfinder.barcode.filter_reads().
        Populates self.good_spots.
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

        Coordinate conversion: internal 0-based (z, y, x) → CSV 1-based
        (x, y, z). Adds 1 to each coordinate and reorders columns.

        Returns path to saved file.
        """
        ...

    # --- Subtile operations ---

    def create_subtiles(
        self,
        *,
        out_dir: Path | None = None,
    ) -> pd.DataFrame:
        """
        Partition FOV into overlapping subtiles and save as NPZ files.

        Uses self.dataset.subtile for window configuration (must be set).
        Returns subtile coordinates DataFrame (t, scoords_x, scoords_y,
        ecoords_x, ecoords_y) matching stitch_subtile.py expectations.
        """
        ...

    @classmethod
    def from_subtile(
        cls,
        subtile_path: Path,
        dataset: STARMapDataset,
        fov_id: str,
    ) -> FOV:
        """Load FOV state from a saved NPZ subtile."""
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

## Registration: Multi-Step Design

Production workflows use sequential global → local registration:

```
Direct mode:    rsf_single_fov  (global + local in one script)
Subtile mode:   gr_single_fov_subtile (global, full FOV) → lrsf_single_fov_subtile (local, per subtile)
```

### What gets stored

| Step | Stored where | Data | Persistence |
|------|-------------|------|-------------|
| Global shifts | `FOV.global_shifts[round]` | `Shift3D` `(dz, dy, dx)` | Saved to shift log on disk |
| Local refinement | `FOV.local_registered` set | Round name | In-memory only |
| Displacement field | Not stored | `(Z, Y, X, 3)` array | Ephemeral — applied then discarded |
| Warped images | `FOV.images[round_name]` | Mutated in-place | Saved as subtile NPZ / ref merged TIFF |

### Multi-step flow in FOV

```python
# Direct workflow: global → local in same FOV instance
fov.global_registration()               # stores global_shifts["round2"] = (dz, dy, dx)
fov.local_registration()                # adds "round2" to local_registered set

# Subtile workflow: global at FOV level, local at subtile level
fov.global_registration()
fov.create_subtiles()                   # saves globally-registered images as NPZ

# Later, in a separate Snakemake job:
subtile_fov = FOV.from_subtile(path, dataset, fov_id)
subtile_fov.local_registration()        # adds round to local_registered;
                                        # global_shifts remains empty (subtile-level)
```

### Why global shifts matter most

- Global shifts are compact, interpretable, and saved to disk for QC
- Displacement fields from demons are large (same size as the volume × 3) and not routinely useful after warping
- MATLAB's `RegisterImagesLocal` also discards the displacement field after applying it
- If displacement fields are needed later (e.g., for debugging), `demons_register()` can be called directly outside the FOV class

---

## Logging Strategy

```python
import logging
from functools import wraps
from time import perf_counter

logger = logging.getLogger("starfinder")

def log_step(func):
    """Decorator to log FOV processing steps with timing."""
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
```

### Decorated methods

The `@log_step` decorator is applied to FOV processing methods that perform
meaningful computation. Excluded: path helpers, properties, `save_*` methods.

| Method | Description |
|--------|-------------|
| `load_raw_images` | TIFF loading |
| `enhance_contrast` | Min-max normalization |
| `hist_equalize` | Histogram matching |
| `morph_recon` | Morphological reconstruction |
| `tophat` | Tophat filtering |
| `make_projection` | Z-projection |
| `global_registration` | Phase correlation |
| `local_registration` | Demons refinement |
| `spot_finding` | Spot detection |
| `reads_extraction` | Color extraction |
| `reads_filtration` | Codebook filtering |

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

from starfinder import STARMapDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def main():
    # 1. Load config
    config_path = Path(snakemake.input.config)
    with open(config_path) as f:
        config = json.load(f)

    # 2. Create dataset and FOV
    dataset = STARMapDataset.from_config(config)
    fov = dataset.fov(snakemake.wildcards.fovID)

    # 3. Get rule-specific parameters
    rule_config = config["rules"]["rsf_single_fov"]["parameters"]
    gr_config = rule_config["global_registration"]
    lr_config = rule_config.get("local_registration", {})
    sf_config = rule_config["spot_finding"]

    # 4. Execute pipeline (fluent API)
    (fov
        .load_raw_images()   # defaults from dataset.layers and dataset.channel_order
        .enhance_contrast()
        .morph_recon(radius=3)
    )

    # 5. Multi-step registration
    if gr_config.get("run", True):
        fov.global_registration(
            ref_img=gr_config.get("ref_img", "merged"),
            mov_img=gr_config.get("mov_img", "merged"),
        )

    if lr_config.get("run", False):
        fov.local_registration(
            iterations=lr_config.get("iterations"),        # defaults to [100,50,25]
            smoothing_sigma=lr_config.get("smoothing", 1.0),
        )

    fov.save_ref_merged()

    # 6. Spot finding and barcode processing
    (fov
        .spot_finding(
            intensity_threshold=sf_config.get("intensity_threshold", 0.2),
        )
        .reads_extraction(voxel_size=tuple(sf_config.get("voxel_size", [1, 2, 2])))
    )

    dataset.load_codebook(
        path=Path(config["codebook_path"]),
        split_index=config.get("split_index"),
    )
    fov.reads_filtration(end_bases=config.get("end_bases"))

    fov.save_signal(slot="goodSpots")
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

| MATLAB method | Python method | Delegates to | Side effects |
|---------------|---------------|--------------|--------------|
| `STARMapDataset(in, out)` | `STARMapDataset.from_config(cfg).fov(id)` | — | Init state |
| `LoadRawImages(...)` | `fov.load_raw_images(...)` | `io.load_image_stacks()` | Populate images, metadata |
| `EnhanceContrast(...)` | `fov.enhance_contrast()` | `preprocessing.min_max_normalize()` | Mutate images |
| `HistEqualize(...)` | `fov.hist_equalize(...)` | `preprocessing.histogram_match()` | Mutate images |
| `MorphRecon(...)` | `fov.morph_recon(...)` | `preprocessing.morphological_reconstruction()` | Mutate images |
| `Tophat(...)` | `fov.tophat(...)` | `preprocessing.tophat_filter()` | Mutate images |
| `MakeProjection(...)` | `fov.make_projection(...)` | `utils.make_projection()` | Mutate images |
| `GlobalRegistration(...)` | `fov.global_registration(...)` | `registration.register_volume()` | Mutate images, fill global_shifts |
| `LocalRegistration(...)` | `fov.local_registration(...)` | `registration.register_volume_local()` | Mutate images, update local_registered |
| `SaveImages(...)` | `fov.save_ref_merged()` | `io.save_stack()` | Write TIFF |
| `SpotFinding(...)` | `fov.spot_finding(...)` | `spotfinding.find_spots_3d()` | Set all_spots |
| `ReadsExtraction(...)` | `fov.reads_extraction(...)` | `barcode.extract_from_location()` | Add per-round color columns, color_seq |
| `LoadCodebook(...)` | `dataset.load_codebook(...)` | `Codebook.from_csv()` | Set dataset.codebook |
| `ReadsFiltration(...)` | `fov.reads_filtration(...)` | `barcode.filter_reads()` | Set good_spots |
| `SaveSignal(...)` | `fov.save_signal(...)` | `pd.DataFrame.to_csv()` | Write CSV |
| `CreateSubtiles(...)` | `fov.create_subtiles(...)` | `np.savez_compressed()` | Write NPZ + coords CSV |

---

## Acceptance Criteria

### A) Contract compatibility
- [ ] Python backend produces files at same paths as MATLAB
- [ ] `goodSpots.csv` has columns `x,y,z,gene` with 1-based integer coordinates
- [ ] Existing `stitch_subtile.py` works without modification
- [ ] Existing `reads_assignment.py` works without modification

### B) Algorithmic parity
- [x] Global registration shifts match MATLAB within ±0.5 pixels (Phase 2)
- [x] Local registration quality matches or exceeds MATLAB (Phase 2 + antialias)
- [ ] Spot counts within ±5% of MATLAB for same parameters
- [ ] Gene assignments match MATLAB for identical spots

### C) End-to-end validation
- [ ] Complete one sample on UGER with Python backend
- [ ] Snakemake `benchmark:` files for all rules
- [ ] No OOM errors on standard node (32GB)

---

## Testing Plan

### Unit tests (existing, Phases 1-5)
- `test_io.py` — TIFF loading, CSV writing
- `test_registration.py` — Phase correlation, shift application
- `test_demons.py` — Demons registration, pyramid utilities
- `test_spotfinding.py` — Local maxima detection, thresholding
- `test_barcode.py` — Encoding, decoding, codebook, filtering
- `test_preprocessing.py` — Normalization, morphology
- `test_utils.py` — Projections

### New tests (Phase 6)
- `test_dataset.py` — STARMapDataset creation, config parsing, FOV factory
- `test_fov.py` — FOV fluent API, multi-step registration, save/load contracts
- `test_codebook_class.py` — Codebook.from_csv, gene lookup
- `test_output_contracts.py` — CSV schemas, path patterns, coordinate conventions

### Integration tests
- `test_fov_pipeline.py` — Single FOV end-to-end on mini synthetic dataset
- `test_subtile_workflow.py` — Create subtiles, load, process, stitch

---

## Revision History

| Date | Changes |
|------|---------|
| 2025-01-29 | Original design document |
| 2026-02-12 (v2) | **Major revision**: Simplified dataclass hierarchy (removed ImageMeta, SignalState, SubtileState, SubtileRef, SubtileStore protocol). Updated RegistrationResult for multi-step registration (global shifts + local_applied flag). Updated local_registration signature to match actual demons API ([100,50,25] pyramid, method, pyramid_mode). Kept Codebook class with from_csv factory. Removed speculative memory management section. Removed CLAHE from enhance_contrast. Removed HDF5SubtileStore. Updated method mapping table to show delegation to Phase 1-5 functions. |
| 2026-02-12 (v3) | **Pre-Phase 6 revision**: Removed `RegistrationResult` dataclass (name collision with `benchmark.runner`); replaced with `FOV.global_shifts: dict` + `FOV.local_registered: set`. Renamed `LayerState.all` → `all_layers` (shadowed Python builtin). Simplified `CropWindow` to Y/X only (subtiles are 2D partitions). Added `SubtileConfig` dataclass (dataset-level). Made `STARMapDataset` non-frozen with `layers`, `codebook`, `subtile`, `channel_order` fields. FOV now delegates `layers`/`codebook` to dataset via properties. Fixed FOV method signatures to match actual Phase 1-5 function APIs (`convert_uint8`, `subdir`, `ref_channel`, removed explicit `ref_layer` params). Added contract specs: `all_spots` schema, shift log format, NPZ subtile schema. Added `@log_step` decorated method list. Removed "Removed dataclasses" historical table. |
