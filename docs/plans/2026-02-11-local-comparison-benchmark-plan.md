# Plan: Python vs MATLAB Local Registration Comparison Benchmark

## Context

The new anti-aliased pyramid demons (`python_matlab` config) showed dramatic improvements on the synthetic large dataset benchmark. Now we need a head-to-head comparison against MATLAB's `imregdemons` with **matched settings** to quantify the remaining gap and test two Python method variants.

**Key results from initial benchmark** (large synthetic, polynomial_small):
- Old Python (single-level): NCC 0.124, Match Rate 0.234
- New Python (antialias): NCC 0.386, Match Rate 0.503
- We need to see how this compares to MATLAB on the same data

---

## Benchmark Design

### Datasets (3 total)

| Dataset | Shape | Voxels | Category |
|---------|-------|--------|----------|
| large | 30×1024×1024 | 31M | synthetic (5 deformation types) |
| cell_culture_3D | 30×1496×1496 | 67M | real |
| LN | 50×1496×1496 | 112M | real |

### Configs (3 total, all matched)

| Config | Backend | Method | Iterations | Sigma/AFS | Pyramid |
|--------|---------|--------|------------|-----------|---------|
| `py_demons` | Python | demons (Thirion) | [100,50,25] | 1.0 | antialias |
| `py_diffeo` | Python | diffeomorphic | [100,50,25] | 1.0 | antialias |
| `matlab` | MATLAB | imregdemons | [100,50,25] | AFS=1.0 | built-in (3-level) |

**Total runs**: 5 deformations × 3 configs + 2 real × 3 configs = **21 runs** (~40-60 min)

### Output Directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/results/registration/local_comparison/`

---

## Scripts

All at `.../starfinder_benchmark/results/registration/scripts/`:

| File | Description |
|------|-------------|
| `benchmark_local_comparison_single.py` | Python worker: runs demons/diffeomorphic on one dataset |
| `benchmark_local_comparison_matlab.m` | MATLAB worker: runs imregdemons with matched settings |
| `run_local_comparison_v2.py` | Orchestrator: loops all 21 combos with `/usr/bin/time -v` |
| `generate_local_comparison_v2.py` | Report: merges quality + timing into comparison table |

## Verification

1. Dry run: test Python worker on `large`/`polynomial_small` with `py_demons` (~55s)
2. Dry run: test MATLAB worker on `large`/`polynomial_small` (~80s)
3. Full run via orchestrator (~40-60 min)
4. Phase 2 evaluation via `evaluate_directory()`
5. Generate comparison report
