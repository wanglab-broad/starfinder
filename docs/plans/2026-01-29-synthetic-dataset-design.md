# Synthetic Test Dataset Design

**Created:** 2026-01-29
**Purpose:** Reusable synthetic datasets for quick testing of STARfinder Python backend

---

## 1. Directory Structure

```
tests/fixtures/synthetic/
├── mini/                          # Fast unit tests (~1 FOV)
│   ├── FOV_001/
│   │   ├── round1/
│   │   │   ├── ch00.tif          # 256×256×5, 3D stack
│   │   │   ├── ch01.tif
│   │   │   ├── ch02.tif
│   │   │   └── ch03.tif
│   │   ├── round2/
│   │   ├── round3/
│   │   └── round4/
│   ├── codebook.csv
│   └── ground_truth.json          # Spot positions, expected barcodes
│
└── standard/                      # Integration tests (~4 FOVs)
    ├── FOV_001/
    ├── FOV_002/
    ├── FOV_003/
    ├── FOV_004/
    ├── codebook.csv
    └── ground_truth.json
```

**Presets:**
| Preset | FOVs | Dimensions | Spots/FOV | Use Case |
|--------|------|------------|-----------|----------|
| mini | 1 | 256×256×5 | 20 | Unit tests, CI |
| standard | 4 | 512×512×10 | 100 | Integration, validation |

---

## 2. Image Generation Strategy

Each synthetic image stack is generated with:

1. **Background**: Gaussian noise with mean=20, std=5
2. **Spots**: 3D Gaussian blobs (sigma=1.5) at known positions
3. **Additional noise**: Poisson-like noise (std=10) for realism
4. **Inter-round shifts**: Small random translations (±5 px XY, ±2 px Z) to test registration

**Spot placement:**
- Positions chosen randomly but deterministically (seed=42)
- Each spot assigned a gene from codebook
- Spot appears in correct channel based on color sequence for that round
- Intensity varies within range [200, 255]

**Channel mapping (color → channel):**
| Color | Channel | Meaning |
|-------|---------|---------|
| 1 | ch00 | Same base (AA/CC/GG/TT) |
| 2 | ch01 | A↔C, G↔T transition |
| 3 | ch02 | A↔G, C↔T transition |
| 4 | ch03 | A↔T, C↔G transition |

---

## 3. Codebook Design

Uses two-base color-space encoding matching MATLAB implementation:
- Consecutive base pairs → color (1-4)
- Barcode is **reversed first**, then encoded to color sequence

**Encoding table:**
| Base Pair | Color |
|-----------|-------|
| AA, CC, GG, TT | 1 |
| AC, CA, GT, TG | 2 |
| AG, CT, GA, TC | 3 |
| AT, CG, GC, TA | 4 |

**Test codebook (8 genes):**
| Gene | Barcode | Reversed | Color Sequence | Channels (R1-R4) |
|------|---------|----------|----------------|------------------|
| GeneA | CACGC | CGCAC | 4422 | ch03, ch03, ch01, ch01 |
| GeneB | CATGC | CGTAC | 4242 | ch03, ch01, ch03, ch01 |
| GeneC | CGAAC | CAAGC | 2134 | ch01, ch00, ch02, ch03 |
| GeneD | CGTAC | CATGC | 2424 | ch01, ch03, ch01, ch03 |
| GeneE | CTGAC | CAGTC | 2323 | ch01, ch02, ch01, ch02 |
| GeneF | CTAGC | CGATC | 4343 | ch03, ch02, ch03, ch02 |
| GeneG | CCATC | CTACC | 3421 | ch02, ch03, ch01, ch00 |
| GeneH | CGCTC | CTCGC | 3344 | ch02, ch02, ch03, ch03 |

**codebook.csv format:**
```csv
gene,barcode
GeneA,CACGC
GeneB,CATGC
GeneC,CGAAC
GeneD,CGTAC
GeneE,CTGAC
GeneF,CTAGC
GeneG,CCATC
GeneH,CGCTC
```

---

## 4. Metadata & Ground Truth

**ground_truth.json structure:**
```json
{
  "version": "1.0",
  "preset": "mini",
  "seed": 42,
  "image_shape": [5, 256, 256],
  "n_rounds": 4,
  "n_channels": 4,
  "fovs": {
    "FOV_001": {
      "shifts": {
        "round1": [0, 0, 0],
        "round2": [1, -2, 3],
        "round3": [-1, 3, -2],
        "round4": [2, 1, -1]
      },
      "spots": [
        {
          "id": 0,
          "gene": "GeneA",
          "barcode": "CACGC",
          "color_seq": "4422",
          "position": [2, 128, 64],
          "intensity": 230
        }
      ]
    }
  }
}
```

---

## 5. Generator Script API

```python
# src/python/starfinder/testing/synthetic.py

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np

@dataclass
class SyntheticConfig:
    """Configuration for synthetic dataset generation."""
    # Image dimensions
    height: int = 256
    width: int = 256
    n_z: int = 10

    # Dataset structure
    n_fovs: int = 2
    n_rounds: int = 4
    n_channels: int = 4

    # Spot generation
    n_spots_per_fov: int = 50
    spot_sigma: float = 1.5
    spot_intensity: tuple[int, int] = (200, 255)

    # Noise and background
    background_mean: int = 20
    background_std: int = 5
    noise_std: int = 10

    # Registration shifts (for testing registration)
    max_shift_xy: int = 5
    max_shift_z: int = 2

    # Random seed for reproducibility
    seed: int = 42


def generate_synthetic_dataset(
    output_dir: Path,
    config: SyntheticConfig | None = None,
    preset: Literal["mini", "standard"] = "mini",
) -> dict:
    """
    Generate a complete synthetic dataset with ground truth.

    Parameters
    ----------
    output_dir : Path
        Directory to write generated files
    config : SyntheticConfig, optional
        Custom configuration. If None, uses preset defaults.
    preset : {"mini", "standard"}
        Preset configuration:
        - "mini": 1 FOV, 4 rounds, 256x256x5, 20 spots (fast unit tests)
        - "standard": 4 FOVs, 4 rounds, 512x512x10, 100 spots (integration)

    Returns
    -------
    dict
        Ground truth metadata including spot positions and expected barcodes
    """
    ...


def get_preset_config(preset: Literal["mini", "standard"]) -> SyntheticConfig:
    """Get predefined configuration for a preset."""
    presets = {
        "mini": SyntheticConfig(
            height=256, width=256, n_z=5,
            n_fovs=1, n_spots_per_fov=20,
            seed=42
        ),
        "standard": SyntheticConfig(
            height=512, width=512, n_z=10,
            n_fovs=4, n_spots_per_fov=100,
            seed=42
        ),
    }
    return presets[preset]


# Convenience functions for tests
def create_test_image_stack(
    shape: tuple[int, int, int],
    spots: list[tuple[int, int, int, int]],  # (z, y, x, intensity)
    background: int = 20,
    noise_std: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """Create a single 3D image stack with spots at specified locations."""
    ...


def create_shifted_stack(
    base_stack: np.ndarray,
    shift: tuple[int, int, int],  # (dz, dy, dx)
) -> np.ndarray:
    """Create a shifted version of a stack (for registration testing)."""
    ...
```

---

## 6. Usage Examples & pytest Integration

### pytest fixtures (tests/conftest.py)

```python
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "synthetic"


@pytest.fixture(scope="session")
def mini_dataset() -> Path:
    """Path to pre-generated mini synthetic dataset (1 FOV, fast tests)."""
    path = FIXTURES_DIR / "mini"
    if not path.exists():
        pytest.skip("Mini synthetic dataset not found. Run: python -m starfinder.testing.generate")
    return path


@pytest.fixture(scope="session")
def standard_dataset() -> Path:
    """Path to pre-generated standard synthetic dataset (4 FOVs)."""
    path = FIXTURES_DIR / "standard"
    if not path.exists():
        pytest.skip("Standard synthetic dataset not found. Run: python -m starfinder.testing.generate")
    return path


@pytest.fixture(scope="session")
def mini_ground_truth(mini_dataset: Path) -> dict:
    """Load ground truth metadata for mini dataset."""
    import json
    with open(mini_dataset / "ground_truth.json") as f:
        return json.load(f)
```

### Example test

```python
# tests/test_spotfinding/test_local_maxima.py

def test_find_spots_recovers_ground_truth(mini_dataset, mini_ground_truth):
    """Verify spot finding recovers injected spots within tolerance."""
    from starfinder.io.tiff import load_image_stack
    from starfinder.spotfinding.local_maxima import find_spots_3d

    # Load reference round image
    ref_path = mini_dataset / "FOV_001" / "round1" / "ch00.tif"
    stack = load_image_stack(ref_path)

    # Find spots
    detected = find_spots_3d(stack, threshold=50)

    # Compare with ground truth
    gt_spots = mini_ground_truth["fovs"]["FOV_001"]["spots"]

    # Allow 2-pixel tolerance for spot center detection
    matched = match_spots(detected, gt_spots, tolerance=2)
    recall = len(matched) / len(gt_spots)

    assert recall >= 0.9, f"Expected >=90% recall, got {recall:.1%}"
```

### CLI for regeneration

```bash
# Regenerate fixtures (run from repo root)
uv run python -m starfinder.testing.generate --preset mini --output tests/fixtures/synthetic/mini
uv run python -m starfinder.testing.generate --preset standard --output tests/fixtures/synthetic/standard
```

---

## 7. Implementation Checklist

- [ ] Create `src/python/starfinder/testing/` module
- [ ] Implement `synthetic.py` with generator functions
- [ ] Implement `__main__.py` for CLI entry point
- [ ] Generate mini preset and commit to `tests/fixtures/synthetic/mini/`
- [ ] Generate standard preset and commit to `tests/fixtures/synthetic/standard/`
- [ ] Add pytest fixtures to `tests/conftest.py`
- [ ] Write validation tests that use synthetic data
