# Real Dataset E2E Benchmark Plan

**Goal:** Validate the Python e2e pipeline on all three real datasets (tissue-2D, LN, cell-culture-3D) and compare with existing MATLAB outputs.

**Architecture:** Self-contained benchmark scripts per dataset (mirroring `run_e2e_large.py`), with a shared QC/comparison module. No ground truth available — validation uses internal consistency metrics plus MATLAB output comparison. Start with tissue-2D (closest to synthetic), then LN, then cell-culture-3D (most complex: 6 rounds, 998 genes).

**Tech Stack:** Python 3.10+, uv, starfinder (io, registration, spotfinding, barcode, dataset, preprocessing), matplotlib (headless), pandas

---

## Context: Real Dataset Summary

| Dataset | FOVs | Seq Rounds | Image Size | Genes | Barcode Len | Ch Files | Start FOV | Pattern | MATLAB Outputs |
|---------|------|-----------|------------|-------|-------------|----------|-----------|---------|----------------|
| tissue-2D | 56 tiles | 4 + protein | 3072×3072×30 | 64 | 5-char (CNNNNC) | 5 (ch00-ch04) | 1 | `tile_%d` | tile_1: 46,745 good spots |
| LN | 64 FOVs | 4 + flamingo | 1496×1496×50 | 62 | 5-char (mixed start) | 4 (ch00-ch03) | 1 | `Position%03d` | No signal outputs |
| cell-culture-3D | 70 FOVs | 6 + organelle | 1496×1496×30 | 998 | 7-char (CNNNNNNC) | 5 (ch00-ch04) | 351 | `Position%03d` | Position351-355+: ~33K spots each |

**Data locations:**
- Raw: `/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/{sample_id}/`
- MATLAB outputs: `/home/unix/jiahao/wanglab/Data/Analyzed/sample-dataset/{output_id}/`
- Benchmark results: `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/e2e/`

## Known Issues to Address

1. **Codebook headers**: Real `genes.csv` files lack the `gene,barcode` header that `load_codebook()` expects via `csv.DictReader`. LN also has UTF-8 BOM (`\xef\xbb\xbf`) and `\r\n` line endings.
2. **Channel order**: All datasets have 4-5 channel files per FOV/tile, but only 4 are sequencing channels. Need to determine which channels map to which color digits for each dataset.
3. **tissue-2D max projection**: MIP is for visualization and downstream segmentation only, NOT for the decoding pipeline. The e2e benchmark runs spot finding on the full 3D volume (3072×3072×30).
4. **cell-culture-3D has 6 rounds**: 7-char barcodes. `encode_bases()` handles arbitrary length (sliding window), so color sequences will be 6 digits instead of 4.
5. **Rotation**: All datasets use `rotate_angle: -90`. The Python pipeline needs to apply this rotation after loading, before registration. Add a `rotate()` step to FOV using `scipy.ndimage.rotate()`.
6. **No ground truth**: Cannot compute recall/precision/gene accuracy. Need internal QC metrics + MATLAB comparison.

---

### Task 1: Fix `load_codebook` to handle headerless CSV files

The real codebook files (genes.csv) don't have a `gene,barcode` header row. Our `load_codebook()` uses `csv.DictReader` which treats the first data row as the header — this silently corrupts the lookup tables.

**Files:**
- Modify: `src/python/starfinder/barcode/codebook.py`
- Test: `src/python/test/test_barcode.py`

**Step 1: Write a failing test for headerless codebook loading**

```python
def test_load_codebook_without_header(tmp_path):
    """load_codebook should handle CSV files without gene,barcode header."""
    codebook_file = tmp_path / "genes.csv"
    codebook_file.write_text("GeneA,CACGC\nGeneB,CATGC\n")
    gene_to_seq, seq_to_gene = load_codebook(codebook_file)
    assert "GeneA" in gene_to_seq
    assert len(gene_to_seq) == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /home/unix/jiahao/Github/starfinder/src/python && uv run pytest test/test_barcode.py::test_load_codebook_without_header -v`
Expected: FAIL (KeyError on `row["gene"]` because first data row becomes header)

**Step 3: Write a failing test for BOM handling**

```python
def test_load_codebook_with_bom(tmp_path):
    """load_codebook should handle UTF-8 BOM in codebook files."""
    codebook_file = tmp_path / "genes.csv"
    codebook_file.write_bytes(b"\xef\xbb\xbfGeneA,CACGC\r\nGeneB,CATGC\r\n")
    gene_to_seq, seq_to_gene = load_codebook(codebook_file)
    assert "GeneA" in gene_to_seq
    assert len(gene_to_seq) == 2
```

**Step 4: Implement fix in `load_codebook()`**

Strategy: Open with `encoding='utf-8-sig'` (handles BOM). Peek at first row — if it contains `gene` and `barcode` keys, use `DictReader` normally; otherwise, use `fieldnames=["gene", "barcode"]` and re-read from the start.

```python
def load_codebook(path, do_reverse=True, split_index=None):
    path = Path(path)
    gene_to_seq = {}
    seq_to_gene = {}

    with open(path, newline="", encoding="utf-8-sig") as f:
        # Peek at first line to detect header
        first_line = f.readline().strip()
        f.seek(0)

        fields = first_line.split(",")
        if fields[0].strip().lower() == "gene":
            reader = csv.DictReader(f)
        else:
            reader = csv.DictReader(f, fieldnames=["gene", "barcode"])

        for row in reader:
            gene = row["gene"].strip()
            barcode = row["barcode"].strip()

            if do_reverse:
                barcode = barcode[::-1]

            color_seq = encode_bases(barcode)

            if split_index is not None:
                color_seq = color_seq[:split_index] + color_seq[split_index + 1:]
                front = color_seq[:split_index]
                back = color_seq[split_index:]
                color_seq = back + front

            gene_to_seq[gene] = color_seq
            seq_to_gene[color_seq] = gene

    return gene_to_seq, seq_to_gene
```

**Step 5: Run all codebook tests to verify**

Run: `cd /home/unix/jiahao/Github/starfinder/src/python && uv run pytest test/test_barcode.py -v -k codebook`
Expected: All PASS (new tests + existing ones still pass)

**Step 6: Commit**

```bash
git add src/python/starfinder/barcode/codebook.py src/python/test/test_barcode.py
git commit -m "fix: handle headerless and BOM codebook CSV files in load_codebook"
```

---

### Task 1b: Add `FOV.rotate()` method

The Python FOV pipeline currently has no rotation step. All three real datasets require `rotate_angle: -90` (applied in MATLAB's `rsf_single_fov.m` before registration). We need to add this to the Python pipeline for shift estimates to match MATLAB.

**Files:**
- Modify: `src/python/starfinder/dataset/fov.py`
- Test: `src/python/test/test_fov.py`

**Step 1: Add `rotate()` method to FOV class**

Insert in the preprocessing section of the FOV class (before `enhance_contrast()`):

```python
@log_step
def rotate(self, *, angle: float) -> FOV:
    """Rotate all loaded volumes by angle degrees in the YX plane.

    Applied after loading, before any other processing. Uses bilinear
    interpolation with reshape=False (output same shape as input).
    """
    from scipy.ndimage import rotate as ndimage_rotate

    for round_name in list(self.images.keys()):
        vol = self.images[round_name]
        # (Z, Y, X, C) or (Z, Y, X): rotate in YX plane
        if vol.ndim == 4:
            yx_axes = (1, 2)
        elif vol.ndim == 3:
            yx_axes = (0, 1)  # (Y, X, C) or (Z, Y, X) — rotate first two spatial
        else:
            yx_axes = (0, 1)
        self.images[round_name] = ndimage_rotate(
            vol, angle, axes=yx_axes, reshape=False, order=1
        ).astype(vol.dtype)
    return self
```

**Step 2: Write a minimal test**

```python
def test_fov_rotate(mini_dataset):
    """FOV.rotate() should rotate all volumes in the YX plane."""
    ds, _ = mini_dataset
    fov = ds.fov("FOV_001")
    fov.load_raw_images()
    original_shape = fov.images["round1"].shape
    fov.rotate(angle=-90)
    assert fov.images["round1"].shape == original_shape  # reshape=False
```

**Step 3: Run tests**

Run: `cd /home/unix/jiahao/Github/starfinder/src/python && uv run pytest test/test_fov.py -v -k rotate`
Expected: PASS

**Step 4: Commit**

```bash
git add src/python/starfinder/dataset/fov.py src/python/test/test_fov.py
git commit -m "feat: add FOV.rotate() method for real dataset -90° rotation"
```

---

### Task 2: Verify channel-to-color mapping for each real dataset

The channel file selection is confirmed: ch00-ch03 are the 4 sequencing channels for all datasets (ch04 is DAPI nuclei staining, excluded from decoding). However, we still need to verify which channel file (ch00-ch03) maps to which color digit (1-4) in the encoding scheme.

**Channel order (confirmed):**
- tissue-2D: `["ch00", "ch01", "ch02", "ch03"]` — ch04 = DAPI (excluded)
- cell-culture-3D: `["ch00", "ch01", "ch02", "ch03"]` — ch04 = DAPI (excluded)
- LN: `["ch00", "ch01", "ch02", "ch03"]` (only 4 files, all sequencing)

**Files:**
- Read: MATLAB config files, `src/matlab/STARMapDataset.m`
- Output: Document verified channel-to-color mapping

**Step 1: Check MATLAB STARMapDataset.m for channel-to-color mapping**

Read `src/matlab/STARMapDataset.m` and look for how `img_col` parameter maps channel indices to color digits. The question is: does ch00→color1, ch01→color2, etc. (identity mapping), or is there a permutation?

**Step 2: Validate with MATLAB shift comparison**

Run Python pipeline on tissue-2D tile_1 with default `["ch00", "ch01", "ch02", "ch03"]` ordering. Compare shift estimates against MATLAB shifts at `log/gr_shifts/tile_1.txt` (round2=84,-48,-1; round3=108,-73,1; round4=108,-92,2). If shifts match, the mapping is correct. If not, try permutations.

**Step 3: Spot-check codebook match rate**

If the channel-to-color mapping is wrong, spots will still be detected (spot finding is per-channel) but the assembled color sequences won't match the codebook. A low codebook match rate (<1%) indicates incorrect channel ordering.

---

### Task 3: Create the real-data e2e benchmark runner for tissue-2D

A self-contained script that runs the Python pipeline on tissue-2D FOVs and generates QC outputs. Modeled after `run_e2e_large.py` but without ground-truth comparison, and with MATLAB output comparison instead.

**Files:**
- Create: `starfinder_benchmark/e2e/results/tissue_2D/run_e2e_tissue2D.py`
- Read: `starfinder_benchmark/e2e/results/large/run_e2e_large.py` (template)

**Step 1: Write the benchmark script skeleton**

The script needs:
1. Dataset config (paths, channel_order, FOV list)
2. Per-FOV pipeline execution with timing
3. QC metrics collection (internal consistency)
4. MATLAB comparison (shift log + good_spots)
5. Output: signal CSVs, inspection PNGs, QC CSVs, summary JSON

```python
#!/usr/bin/env python3
"""E2E benchmark: tissue-2D real dataset."""
import os
os.environ["MPLBACKEND"] = "Agg"

import json
import resource
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Config ---
BENCHMARK_ROOT = Path("/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark")
DATA_ROOT = Path("/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/tissue-2D")
MATLAB_ROOT = Path("/home/unix/jiahao/wanglab/Data/Analyzed/sample-dataset/tissue-2D-test-v9")
RESULTS_DIR = BENCHMARK_ROOT / "e2e" / "results" / "tissue_2D"

# Dataset parameters
N_ROUNDS = 4
REF_ROUND = "round1"
CHANNEL_ORDER = ["ch00", "ch01", "ch02", "ch03"]  # first 4 = sequencing; ch04 = DAPI (excluded)
FOV_PATTERN = "tile_%d"
FOV_IDS = [1]  # Start with single FOV, expand later

# Pipeline parameters
ROTATE_ANGLE = -90
SNR_THRESHOLD = 5.0
INTENSITY_ESTIMATION = "noise"
INTENSITY_THRESHOLD = 5.0
VOXEL_SIZE = (1, 2, 2)
END_BASES = "CC"  # tissue-2D uses CNNNNC barcodes → end_bases="CC"
```

**Step 2: Implement per-FOV pipeline function**

```python
def run_fov(fov_id_num: int) -> dict:
    """Run full e2e pipeline on a single tissue-2D FOV."""
    from starfinder.dataset import STARMapDataset, LayerState

    fov_id = FOV_PATTERN % fov_id_num
    print(f"\n{'='*60}")
    print(f"Processing {fov_id}")
    print(f"{'='*60}")

    timings = {}
    memory = {}

    def timed_step(name, func):
        t0 = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
        timings[name] = elapsed
        memory[f"rss_after_{name}"] = rss
        print(f"  {name}: {elapsed:.2f}s (RSS: {rss:.0f} MB)")
        return result

    # Create dataset
    ds = STARMapDataset(
        input_root=DATA_ROOT,
        output_root=RESULTS_DIR,
        dataset_id="sample-dataset",
        sample_id="tissue-2D",
        output_id="tissue-2D-python-benchmark",
        layers=LayerState(
            seq=[f"round{i}" for i in range(1, N_ROUNDS + 1)],
            ref=REF_ROUND,
        ),
        channel_order=CHANNEL_ORDER,
    )
    ds.load_codebook(DATA_ROOT / "genes.csv")

    fov = ds.fov(fov_id)

    # Pipeline (full 3D — MIP is for visualization only, not decoding)
    timed_step("load", lambda: fov.load_raw_images())
    timed_step("rotate", lambda: fov.rotate(angle=ROTATE_ANGLE))
    timed_step("enhance", lambda: fov.enhance_contrast(snr_threshold=SNR_THRESHOLD))
    timed_step("registration", lambda: fov.global_registration())
    timed_step("spot_finding", lambda: fov.spot_finding(
        intensity_estimation=INTENSITY_ESTIMATION,
        intensity_threshold=INTENSITY_THRESHOLD,
    ))
    timed_step("extraction", lambda: fov.reads_extraction(voxel_size=VOXEL_SIZE))
    timed_step("filtration", lambda: fov.reads_filtration(end_bases=END_BASES))

    # Save outputs
    fov.save_signal(slot="goodSpots")
    fov.save_signal(slot="allSpots")

    # Collect QC metrics
    qc = collect_qc_metrics(fov, timings, memory)

    # Compare with MATLAB
    matlab_comparison = compare_with_matlab(fov, fov_id)
    qc.update(matlab_comparison)

    # Generate inspection images
    generate_inspection_images(fov, fov_id)

    # Save QC CSV
    save_qc_csv(qc, fov_id)

    return {fov_id: qc}
```

**Step 3: Implement QC metrics collection (no ground truth)**

```python
def collect_qc_metrics(fov, timings, memory) -> dict:
    """Collect internal consistency QC metrics."""
    qc = {}

    # Spot detection
    qc["n_all_spots"] = len(fov.all_spots)
    qc["n_good_spots"] = len(fov.good_spots)
    qc["codebook_match_rate"] = len(fov.good_spots) / len(fov.all_spots) if len(fov.all_spots) > 0 else 0

    # Gene diversity
    n_unique_genes = fov.good_spots["gene"].nunique()
    qc["n_unique_genes"] = n_unique_genes
    qc["gene_coverage"] = n_unique_genes / ds.codebook.n_genes  # fraction of codebook genes detected

    # Spatial distribution: coefficient of variation of spot density in quadrants
    if len(fov.good_spots) > 0:
        ys = fov.good_spots["y"].values
        xs = fov.good_spots["x"].values
        y_mid, x_mid = np.median(ys), np.median(xs)
        quadrant_counts = [
            ((ys < y_mid) & (xs < x_mid)).sum(),
            ((ys < y_mid) & (xs >= x_mid)).sum(),
            ((ys >= y_mid) & (xs < x_mid)).sum(),
            ((ys >= y_mid) & (xs >= x_mid)).sum(),
        ]
        cv = np.std(quadrant_counts) / np.mean(quadrant_counts) if np.mean(quadrant_counts) > 0 else 0
        qc["spatial_cv_quadrant"] = cv

    # Registration quality
    for round_name, (dz, dy, dx) in fov.global_shifts.items():
        qc[f"shift_{round_name}_dz"] = dz
        qc[f"shift_{round_name}_dy"] = dy
        qc[f"shift_{round_name}_dx"] = dx

    # Color score quality (mean score across all spots)
    score_cols = [c for c in fov.all_spots.columns if c.endswith("_score")]
    if score_cols:
        qc["mean_color_score"] = fov.all_spots[score_cols].mean().mean()
        qc["min_color_score"] = fov.all_spots[score_cols].min().min()

    # Timing and memory
    qc.update({f"time_{k}_s": v for k, v in timings.items()})
    qc["time_total_s"] = sum(timings.values())
    qc.update(memory)
    qc["rss_peak_mb"] = max(memory.values()) if memory else 0

    return qc
```

**Step 4: Implement MATLAB comparison**

```python
def compare_with_matlab(fov, fov_id) -> dict:
    """Compare Python results with MATLAB outputs."""
    comparison = {}

    # --- Shift comparison ---
    matlab_shift_path = MATLAB_ROOT / "log" / "gr_shifts" / f"{fov_id}.txt"
    if matlab_shift_path.exists():
        matlab_shifts = pd.read_csv(matlab_shift_path)
        for _, row in matlab_shifts.iterrows():
            round_name = row["round"]
            if round_name in fov.global_shifts:
                py_dz, py_dy, py_dx = fov.global_shifts[round_name]
                ml_dy, ml_dx, ml_dz = row["row"], row["col"], row["z"]
                comparison[f"matlab_shift_diff_{round_name}_dy"] = abs(py_dy - ml_dy)
                comparison[f"matlab_shift_diff_{round_name}_dx"] = abs(py_dx - ml_dx)
                comparison[f"matlab_shift_diff_{round_name}_dz"] = abs(py_dz - ml_dz)

    # --- Spot count comparison ---
    matlab_spots_path = MATLAB_ROOT / "signal" / f"{fov_id}_goodSpots.csv"
    if matlab_spots_path.exists():
        matlab_spots = pd.read_csv(matlab_spots_path)
        comparison["matlab_n_good_spots"] = len(matlab_spots)
        comparison["python_n_good_spots"] = len(fov.good_spots)
        comparison["spot_count_ratio"] = len(fov.good_spots) / len(matlab_spots) if len(matlab_spots) > 0 else 0

        # Gene overlap: what fraction of MATLAB genes does Python also find?
        ml_genes = set(matlab_spots["gene"].unique())
        py_genes = set(fov.good_spots["gene"].unique())
        comparison["matlab_n_unique_genes"] = len(ml_genes)
        comparison["python_n_unique_genes"] = len(py_genes)
        comparison["gene_overlap"] = len(ml_genes & py_genes) / len(ml_genes) if len(ml_genes) > 0 else 0

        # Top gene comparison: are the most abundant genes the same?
        ml_top10 = matlab_spots["gene"].value_counts().head(10).index.tolist()
        py_top10 = fov.good_spots["gene"].value_counts().head(10).index.tolist()
        comparison["top10_gene_overlap"] = len(set(ml_top10) & set(py_top10))

    return comparison
```

**Step 5: Implement inspection image generation**

```python
def generate_inspection_images(fov, fov_id):
    """Generate visual QC images."""
    from starfinder.utils import make_projection

    signal_dir = RESULTS_DIR / "signal"
    log_dir = RESULTS_DIR / "log"
    signal_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Good spots overlay on reference MIP
    ref_img = fov.images[fov.layers.ref]
    if ref_img.ndim == 4:
        mip = np.max(np.sum(ref_img, axis=-1), axis=0)
    elif ref_img.ndim == 3:
        mip = np.max(ref_img, axis=-1) if ref_img.shape[-1] <= 4 else np.max(ref_img, axis=0)
    else:
        mip = ref_img

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(mip, cmap="gray", vmax=np.percentile(mip, 99.5))
    if len(fov.good_spots) > 0:
        ax.scatter(fov.good_spots["x"], fov.good_spots["y"], s=1, c="red", alpha=0.5)
    ax.set_title(f"{fov_id}: {len(fov.good_spots)} good spots")
    ax.axis("off")
    fig.savefig(signal_dir / f"{fov_id}_goodSpots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Registration inspection: green-magenta overlays per round
    ref_3d = np.sum(fov.images[fov.layers.ref], axis=-1)
    if ref_3d.ndim == 3:
        ref_mip = np.max(ref_3d, axis=0)
    else:
        ref_mip = ref_3d

    rounds_to_show = fov.layers.to_register
    n_rounds = len(rounds_to_show)
    fig, axes = plt.subplots(1, n_rounds, figsize=(5 * n_rounds, 5))
    if n_rounds == 1:
        axes = [axes]

    for ax, round_name in zip(axes, rounds_to_show):
        mov_img = np.sum(fov.images[round_name], axis=-1)
        if mov_img.ndim == 3:
            mov_mip = np.max(mov_img, axis=0)
        else:
            mov_mip = mov_img

        # Green-magenta composite
        ref_norm = ref_mip.astype(float) / max(ref_mip.max(), 1)
        mov_norm = mov_mip.astype(float) / max(mov_mip.max(), 1)
        composite = np.stack([mov_norm, ref_norm, mov_norm], axis=-1)

        ax.imshow(np.clip(composite, 0, 1))
        shift = fov.global_shifts.get(round_name, (0, 0, 0))
        ax.set_title(f"{round_name}\nshift=({shift[0]:.0f},{shift[1]:.0f},{shift[2]:.0f})")
        ax.axis("off")

    fig.suptitle(f"{fov_id} — Registration (green=ref, magenta=registered)")
    fig.savefig(log_dir / f"{fov_id}_inspection_registration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
```

**Step 6: Implement main function and QC CSV output**

```python
def save_qc_csv(qc, fov_id):
    """Save per-FOV QC metrics to CSV."""
    log_dir = RESULTS_DIR / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([qc])
    df.to_csv(log_dir / f"{fov_id}_qc.csv", index=False)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t0 = time.perf_counter()

    for fov_num in FOV_IDS:
        result = run_fov(fov_num)
        all_results.update(result)

    total_time = time.perf_counter() - t0

    # Summary
    summary = {
        "dataset": "tissue-2D",
        "n_fovs": len(FOV_IDS),
        "total_time_s": total_time,
        "fovs": all_results,
    }

    with open(RESULTS_DIR / "e2e_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: tissue-2D ({len(FOV_IDS)} FOVs, {total_time:.1f}s total)")
    print(f"{'='*60}")
    for fov_id, qc in all_results.items():
        print(f"\n{fov_id}:")
        print(f"  All spots:    {qc.get('n_all_spots', '?')}")
        print(f"  Good spots:   {qc.get('n_good_spots', '?')} (match rate: {qc.get('codebook_match_rate', 0):.1%})")
        print(f"  Unique genes: {qc.get('n_unique_genes', '?')}")
        if "matlab_n_good_spots" in qc:
            print(f"  MATLAB spots: {qc['matlab_n_good_spots']} (ratio: {qc.get('spot_count_ratio', 0):.2f})")
            print(f"  Gene overlap: {qc.get('gene_overlap', 0):.1%}")
        print(f"  Time:         {qc.get('time_total_s', 0):.1f}s")
        print(f"  Peak RSS:     {qc.get('rss_peak_mb', 0):.0f} MB")


if __name__ == "__main__":
    main()
```

**Step 7: Run the benchmark on tile_1**

Run: `cd /home/unix/jiahao/Github/starfinder/src/python && uv run python /path/to/run_e2e_tissue2D.py`

Evaluate output:
- Do shifts match MATLAB (`log/gr_shifts/tile_1.txt`: round2=84,-48,-1; round3=108,-73,1; round4=108,-92,2)?
- Is codebook match rate reasonable (>5%)?
- Are spots spatially distributed across the tile?
- Are inspection images showing proper registration?

**Step 8: Iterate on parameters if needed**

If shifts don't match MATLAB, revisit channel_order (Task 2). If codebook match rate is extremely low (<1%), check if the channel-to-color mapping is wrong.

**Step 9: Commit**

```bash
git add starfinder_benchmark/e2e/results/tissue_2D/run_e2e_tissue2D.py
git commit -m "bench: add tissue-2D real data e2e benchmark runner"
```

---

### Task 4: Scale tissue-2D to multiple FOVs

Once single-FOV results look correct, expand to a representative subset.

**Files:**
- Modify: `starfinder_benchmark/e2e/results/tissue_2D/run_e2e_tissue2D.py`

**Step 1: Update FOV_IDS to include a representative subset**

Pick 5-8 tiles spread across the grid (e.g., corners + center): `FOV_IDS = [1, 4, 25, 28, 53, 56]` (corners of the 7×8 grid + center-ish tiles). Rationale: different tiles may have different tissue content and SNR.

**Step 2: Run the expanded benchmark**

Run: `cd /home/unix/jiahao/Github/starfinder/src/python && uv run python /path/to/run_e2e_tissue2D.py`

**Step 3: Analyze cross-FOV consistency**

Add a cross-FOV summary at the end of `main()`:
```python
# Cross-FOV consistency
spot_counts = [qc["n_good_spots"] for qc in all_results.values()]
print(f"\nCross-FOV stats:")
print(f"  Good spots: {np.mean(spot_counts):.0f} ± {np.std(spot_counts):.0f}")
print(f"  Min/Max: {np.min(spot_counts)}/{np.max(spot_counts)}")
```

Check: Are spot counts consistent across tiles? Do registration shifts look reasonable for all tiles?

**Step 4: Commit**

```bash
git commit -am "bench: tissue-2D multi-FOV e2e results"
```

---

### Task 5: Create LN benchmark runner

LN is similar to tissue-2D (4 rounds, 5-char barcodes) but with important differences: thicker Z (50 slices), only 4 channel files, no MATLAB comparison (only `documents/` in analyzed output).

**Files:**
- Create: `starfinder_benchmark/e2e/results/LN/run_e2e_LN.py`

**Step 1: Adapt the tissue-2D script for LN**

Key differences from tissue-2D:
- `DATA_ROOT`: `.../LN/`
- `CHANNEL_ORDER`: `["ch00", "ch01", "ch02", "ch03"]` (only 4 files, all sequencing)
- `FOV_PATTERN`: `"Position%03d"`
- `FOV_IDS`: Start with `[1]` (Position001)
- `ROTATE_ANGLE`: `-90` (same as tissue-2D)
- `END_BASES`: `"AC"` (LN barcodes use AC end-base pattern, not CC)
- `MATLAB_ROOT`: No signal outputs available — skip MATLAB comparison
- Image size: 1496×1496×50 (~112 MB per channel, ~450 MB per round)
- Pipeline: `load → rotate(-90) → enhance → registration → spot_finding → extraction → filtration(end_bases="AC")`

**Step 2: Handle LN-specific codebook conventions**

LN barcodes don't always start with C (e.g., `Acta1,ACTTC`). The pipeline's `filter_reads()` uses codebook membership as the primary filter. For LN, use `end_bases="AC"` and `start_base="A"` in `reads_filtration()` to match the LN barcode convention (barcodes end with varied bases but the decoded color sequences should be validated with `"AC"` end-base pattern).

**Step 3: Run on Position001**

Run: `cd /home/unix/jiahao/Github/starfinder/src/python && uv run python /path/to/run_e2e_LN.py`

Evaluate: Do spots look reasonable? Is codebook match rate >1%? Check memory usage (50 Z-slices = more memory than tissue-2D).

**Step 4: Commit**

```bash
git add starfinder_benchmark/e2e/results/LN/run_e2e_LN.py
git commit -m "bench: add LN real data e2e benchmark runner"
```

---

### Task 6: Create cell-culture-3D benchmark runner

The most complex dataset: 6 sequencing rounds, 996 genes with 7-char barcodes. This tests the pipeline's ability to handle arbitrary round counts and large codebooks.

**Files:**
- Create: `starfinder_benchmark/e2e/results/cell_culture_3D/run_e2e_cc3D.py`

**Step 1: Adapt for 6-round pipeline**

Key differences:
- `DATA_ROOT`: `.../cell-culture-3D/`
- `N_ROUNDS`: 6
- `CHANNEL_ORDER`: `["ch00", "ch01", "ch02", "ch03"]` (first 4 = sequencing; ch04 = DAPI, excluded)
- `FOV_PATTERN`: `"Position%03d"`
- `FOV_IDS`: Start with `[351]` (first FOV, has MATLAB output)
- `ROTATE_ANGLE`: `-90` (same as all datasets)
- `END_BASES`: `"CC"` (CNNNNNNC barcodes, both ends are C)
- `MATLAB_ROOT`: `.../cell-culture-3D/`
- Pipeline: `load → rotate(-90) → enhance → registration → spot_finding → extraction → filtration(end_bases="CC")`

**Step 2: Verify 6-round color sequence handling**

With 6 rounds, `reads_extraction()` produces 6-digit color sequences. The codebook has 7-char barcodes (e.g., `CCTACCC`) which after `encode_bases(barcode[::-1])` = `encode_bases("CCCCATCC"[::-1])` → a 6-digit color sequence. Verify this manually for a few genes before running.

```python
from starfinder.barcode.encoding import encode_bases
# cell-culture-3D example: METTL14 has barcode CCTACCC
barcode = "CCTACCC"
reversed_barcode = barcode[::-1]  # "CCCATCC"
color_seq = encode_bases(reversed_barcode)  # sliding window on 7 chars → 6 digits
print(f"{barcode} -> reversed={reversed_barcode} -> color_seq={color_seq}")
```

**Step 3: Run on Position351 with MATLAB comparison**

MATLAB has ~33,463 good spots for Position351. Compare:
- Shift estimates
- Spot counts (Python may differ due to different thresholding)
- Gene overlap
- Top-gene agreement

**Step 4: Commit**

```bash
git add starfinder_benchmark/e2e/results/cell_culture_3D/run_e2e_cc3D.py
git commit -m "bench: add cell-culture-3D real data e2e benchmark runner"
```

---

### Task 7: Cross-dataset summary and analysis

Create a summary comparing all three datasets' results.

**Files:**
- Create: `starfinder_benchmark/e2e/results/cross_dataset_summary.py`

**Step 1: Write summary script**

Reads `e2e_results.json` from each dataset, produces a combined CSV and summary table.

Key comparisons:
- **Pipeline timing**: Per-step and total, normalized per voxel
- **Memory**: Peak RSS, scaling with image size
- **Spot density**: Spots per FOV, spots per mm² (using voxel_size)
- **Codebook match rate**: Across datasets
- **MATLAB agreement**: Shift accuracy, spot count ratio, gene overlap (where available)

**Step 2: Generate comparison figures**

- Bar chart: time per step across datasets
- Scatter: Python vs MATLAB spot counts (for tissue-2D and cell-culture-3D)
- Table: Key metrics summary

**Step 3: Commit**

```bash
git add starfinder_benchmark/e2e/results/cross_dataset_summary.py
git commit -m "bench: add cross-dataset e2e summary analysis"
```

---

## Appendix: Rotation Implementation

All three real datasets use `rotate_angle: -90`. In the MATLAB pipeline, rotation is handled by `rsf_single_fov.m` and applied before registration. The Python pipeline must match this behavior for shift estimates to agree with MATLAB.

**Implementation**: Add a `FOV.rotate(angle)` method that applies `scipy.ndimage.rotate()` with `axes=(1, 2)` (YX plane) and `reshape=False` to each round's volume. The rotation step goes **after loading but before any processing** (enhancement, registration, spot finding).

```python
@log_step
def rotate(self, *, angle: float) -> FOV:
    """Rotate all loaded volumes by angle degrees in the YX plane."""
    from scipy.ndimage import rotate as ndimage_rotate

    for round_name in list(self.images.keys()):
        vol = self.images[round_name]
        # Rotate in YX plane; axes depend on ndim
        # (Z, Y, X, C) → axes=(1, 2); (Y, X, C) → axes=(0, 1)
        yx_axes = (1, 2) if vol.ndim == 4 else (0, 1) if vol.ndim == 3 and vol.shape[-1] <= 4 else (0, 1)
        self.images[round_name] = ndimage_rotate(
            vol, angle, axes=yx_axes, reshape=False, order=1
        ).astype(vol.dtype)
    return self
```

**Pipeline order** (all datasets): `load → rotate(-90) → enhance → registration → spot_finding → extraction → filtration`

**Per-dataset end_bases**:
- tissue-2D: `end_bases="CC"` (CNNNNC barcodes, both ends are C)
- LN: `end_bases="AC"` (mixed-start barcodes, e.g. `ACTTC`)
- cell-culture-3D: `end_bases="CC"` (CNNNNNNC barcodes, both ends are C)

## Appendix: Expected Performance Estimates

Based on the synthetic large benchmark (1024×1024×30, ~52s/FOV):

| Dataset | Size (per FOV) | Est. Time/FOV | Est. Peak RSS |
|---------|---------------|---------------|---------------|
| tissue-2D | 3072×3072×30 (3D) | ~60-120s | ~6-10 GB |
| LN | 1496×1496×50 | ~60-120s | ~3-5 GB |
| cell-culture-3D | 1496×1496×30 | ~40-80s | ~2-4 GB |

Note: Registration dominates timing. Loading large TIFFs from network mount may also be slow.
