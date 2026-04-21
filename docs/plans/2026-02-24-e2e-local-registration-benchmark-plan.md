# E2E Local Registration Benchmark Plan

## Context

We recently optimized the local registration algorithm (anti-aliased pyramid, demons registration). This benchmark validates the full end-to-end pipeline **with local registration enabled**, measuring its impact on gene accuracy, runtime, and memory. Everything is stored at `starfinder_benchmark/e2e_LR/`.

**Key question**: Does adding local registration after global registration improve gene decoding accuracy on data with local deformations?

---

## Part 1: Add `"linear"` Deformation Type

**File**: `src/python/starfinder/benchmark/data.py`

Add a new deformation type `"linear"` to `create_deformation_field()`:
```python
elif deform_type == "linear":
    # No cross terms: d = c1*x + c2*y + c3*z per axis
    coeffs = rng.uniform(-1, 1, size=(3, 3))  # 3 coeffs per displacement axis
    for axis in range(3):
        c = coeffs[axis]
        displacement = c[0] * xx + c[1] * yy + c[2] * zz
        displacement = displacement / np.abs(displacement).max() * max_displacement
        field[..., axis] = displacement
```

Add to `DEFORMATION_CONFIGS`:
```python
"linear_small": {"type": "linear", "max_displacement_pct": 0.5, "cap_px": 5.0},
```

**Why linear with no cross terms**: The simplest non-rigid deformation — a linearly varying displacement gradient. On coordinates normalized to [-1,1], each term has zero mean, so global registration cannot absorb it. 9 random coefficients total (3 axes × 3 terms).

---

## Part 2: Data Generation Script

**File**: `starfinder_benchmark/e2e_LR/data/generate_data.py`

**Strategy**: Two-step generation — avoids modifying existing `testdata/synthetic.py`:
1. Call `generate_synthetic_dataset(output_dir, config)` → creates standard dataset with global shifts
2. Post-process non-ref rounds: load TIFFs → apply deformation → overwrite TIFFs → update ground_truth.json

**3 synthetic presets** (standardized shifts across all presets):
| Preset | Shape | Spots/FOV | Genes | Max Shift (yx/z) | Deformation |
|--------|-------|-----------|-------|-------------------|-------------|
| large | 1024²×30 | 2000 | 64 | 50/5 px | linear, 5px |
| tissue | 3072²×30 | 14000 | 64 | 50/7 px | linear, 5px |
| thick_medium | 1024²×100 | 5200 | 64 | 50/10 px | linear, 5px |

**Deformation field**: One field per round per FOV (3 non-ref rounds × 2 FOVs = 6 fields per preset). Deterministic seed: `base_seed + fov_idx * 100 + round_idx * 25`.

**Ground truth update**: Add `deformations` key per FOV with per-round metadata.

**Output layout**:
```
e2e_LR/data/
├── generate_data.py
├── large/
│   ├── ground_truth.json
│   ├── codebook.csv
│   ├── FOV_001/{round1..round4}/{ch00..ch03}.tif
│   └── FOV_002/...
├── tissue/
└── thick_medium/
```

---

## Part 3: Benchmark Script

**File**: `starfinder_benchmark/e2e_LR/results/run_e2e_LR.py`

Single parametrized script. Usage:
```bash
uv run python run_e2e_LR.py <dataset> [--mode global_only|global_local|both]
```

**Datasets**: 6 total (3 synthetic + 3 real), same config dict pattern as `run_streaming_benchmark.py`.

### Pipeline (two modes)

**Mode: `global_only`** (baseline for synthetic):
```
load_raw_images → enhance_contrast(snr_threshold=5.0) → global_registration
→ spot_finding → reads_extraction → reads_filtration
```

**Mode: `global_local`**:
```
load_raw_images → enhance_contrast(snr_threshold=5.0) → global_registration
→ local_registration(method="demons", pyramid_mode="antialias")
→ spot_finding → reads_extraction → reads_filtration
```

### Output layout
```
e2e_LR/results/{preset}/
├── global_only/
│   ├── signal/FOV_*_goodSpots.csv
│   └── log/ (QC CSV, inspection PNGs)
├── global_local/
│   ├── signal/FOV_*_goodSpots.csv
│   └── log/ (QC CSV, inspection PNGs)
├── comparison.csv
└── e2e_LR_results.json
```

### QC CSV columns (29 columns, superset of existing 27)
Same 27 columns as existing e2e QC CSV, plus:
- `time_local_reg_s` — local registration time (0 for global_only mode)
- `rss_after_local_reg_mb` — RSS after local registration

---

## Part 4: Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/python/starfinder/benchmark/data.py` | **Edit** | Add `"linear"` deform type + `"linear_small"` config |
| `starfinder_benchmark/e2e_LR/data/generate_data.py` | **Create** | Data generation (3 synthetic presets) |
| `starfinder_benchmark/e2e_LR/results/run_e2e_LR.py` | **Create** | Main benchmark script (6 datasets, 2 modes) |

---

## Part 5: Verification

1. **Generate data**: `cd src/python && uv run python .../generate_data.py` — verify ground_truth.json has deformation metadata
2. **Run benchmark**: `uv run python run_e2e_LR.py large --mode both` — verify both modes run, comparison.csv generated
3. **Check gene accuracy**: global_only should have degraded accuracy; global_local should recover close to 100%
4. **Run real data**: `uv run python run_e2e_LR.py tissue_2D --mode global_local`
5. **Review comparison.csv**: confirm local_reg provides measurable benefit

**Expected outcomes**:
- Synthetic: global_only gene accuracy < 1.0 (deformation degrades extraction), global_local ≈ 1.0
- Real data: spot counts comparable or slightly improved with local registration
- Runtime: local_reg adds ~1-3x overhead to registration step
- Memory: modest increase from displacement field allocation
