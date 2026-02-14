# Plan: SNR-Gated Normalization + Round-Max Spot Finding Threshold

## Problem

The current `min_max_normalize` rescales each channel independently to [0, 255]. Channels with no real spots (low SNR) have their noise inflated to fill the full uint8 range. Downstream spot finding then detects thousands of noise peaks in those channels — e.g., channel 0 in the mini dataset produces 7,846 false positives out of 9,044 total detections (20 ground truth spots).

## Root Cause

Two compounding issues:

1. **Per-channel normalization inflates noise**: A channel with max=74 (pure noise) gets stretched to max=255, making noise indistinguishable from real spots in other channels.

2. **Per-channel adaptive threshold is too lenient**: `threshold = channel_max * 0.2` gives dim noise-only channels a very low bar. After normalization, all channels have max=255, so the threshold is always `51` — unable to distinguish signal from inflated noise.

## Solution

Two complementary changes that work together:

1. **SNR-gated normalization**: Skip per-channel normalization for channels where `max/mean < snr_threshold`. These channels contain no spots and should retain their raw (low) intensity values.

2. **Round-max adaptive threshold**: In spot finding, compute the threshold from `max(all_channels)` instead of per-channel max. Channels that were skipped by SNR gating retain low raw max values, so the round-max threshold (driven by signal channels at max=255) effectively suppresses their noise.

### Empirical validation (mini dataset, round1)

| Config | Spots | ch0 | ch1-3 | Recall | Precision |
|--------|-------|-----|-------|--------|-----------|
| Current (full norm + channel_max) | 9,044 | 7,846 | 1,198 | 1.000 | 0.002 |
| SNR-gated + round_max | 1,616 | 418 | 1,198 | 1.000 | 0.012 |
| SNR-gated + round_max + md=2 | 480 | 125 | 355 | 0.700 | 0.029 |

---

## File Plan

### Modified files

| File | Change |
|------|--------|
| `starfinder/preprocessing/normalization.py` | Add `snr_threshold` parameter to `min_max_normalize` |
| `starfinder/spotfinding/local_maxima.py` | Add `"adaptive_round"` intensity estimation mode |
| `starfinder/dataset/fov.py` | Expose `snr_threshold` in `enhance_contrast()`, `intensity_estimation` default in `spot_finding()` |
| `test/test_preprocessing.py` | Add tests for SNR-gated normalization |
| `test/test_spotfinding.py` | Add test for `adaptive_round` mode |
| `test/conftest.py` | Update `e2e_result` fixture to use new defaults |
| `test/test_e2e.py` | Recalibrate thresholds, add precision assertion |

---

## Implementation Steps

### Step 1: SNR-gated `min_max_normalize` (`normalization.py`)

Add `snr_threshold` parameter. When set, compute per-channel `max/mean` before normalizing. Skip normalization for channels below the threshold (keep raw values, cast to uint8).

```python
def min_max_normalize(
    volume: np.ndarray,
    snr_threshold: float | None = None,
) -> np.ndarray:
    """Per-channel min-max normalization to uint8.

    Parameters
    ----------
    snr_threshold : float or None
        If set, channels with max/mean < snr_threshold are not
        normalized (raw values kept, cast to uint8). This prevents
        noise inflation in channels with no real signal.
        Recommended value: 5.0.
    """
```

Logic per channel:
- Compute `snr = channel.max() / channel.mean()` (guard against zero mean)
- If `snr_threshold` is set and `snr < snr_threshold`: cast to uint8 without rescaling
- Otherwise: rescale `[min, max] → [0, 255]` as before

Default `snr_threshold=None` preserves backward compatibility.

### Step 2: `adaptive_round` mode in `find_spots_3d` (`local_maxima.py`)

Add a new intensity estimation mode that uses the max across all channels instead of per-channel max.

```python
# In the function, before the per-channel loop:
if intensity_estimation == "adaptive_round":
    round_max = float(max(image[:, :, :, c].max() for c in range(n_channels)))

# In the per-channel loop:
if intensity_estimation == "adaptive":
    threshold_abs = float(channel.max()) * intensity_threshold
elif intensity_estimation == "adaptive_round":
    threshold_abs = round_max * intensity_threshold
elif intensity_estimation == "global":
    # ... existing code
```

### Step 3: Expose in FOV layer (`fov.py`)

Update `enhance_contrast()` to accept `snr_threshold`:

```python
def enhance_contrast(
    self,
    layers: list[str] | None = None,
    snr_threshold: float | None = None,
) -> FOV:
    """Per-channel min-max normalization."""
    from starfinder.preprocessing import min_max_normalize
    self._apply_to_layers(
        lambda v: min_max_normalize(v, snr_threshold=snr_threshold), layers
    )
    return self
```

Update `spot_finding()` default `intensity_estimation` to remain `"adaptive"` (backward compat). Users opt into `"adaptive_round"` explicitly.

### Step 4: Tests

**`test_preprocessing.py`** — Add 2 tests:

- `test_snr_gating_skips_low_snr_channel`: Create 4D volume where ch0 has low max (pure noise) and ch1 has bright spots. With `snr_threshold=5.0`, ch0 should NOT be rescaled to [0, 255] while ch1 should be.
- `test_snr_gating_none_is_backward_compat`: With `snr_threshold=None`, behavior is identical to current (all channels normalized).

**`test_spotfinding.py`** — Add 1 test:

- `test_adaptive_round_threshold`: Create 4D volume where ch0 is dim (max=50) and ch1 is bright (max=255). With `"adaptive_round"`, ch0 should use threshold `255 * 0.2 = 51`, detecting zero spots (max < threshold). With `"adaptive"`, ch0 would use `50 * 0.2 = 10`, detecting many.

### Step 5: Update e2e fixture and recalibrate

Update `conftest.py` `e2e_result` fixture:

```python
fov.load_raw_images()
    .enhance_contrast(snr_threshold=5.0)
    .global_registration()
    .spot_finding(intensity_estimation="adaptive_round")
    .reads_extraction()
    .reads_filtration()
```

Remove `min_distance=2` from `spot_finding()` call — no longer needed as the primary false-positive fix.

Recalibrate test thresholds in `test_e2e.py` based on observed metrics with the new settings.

---

## Key Design Decisions

### SNR metric: `max/mean`

Simple, robust, and provides clear separation between noise-only channels (SNR ≈ 3-4) and signal channels (SNR ≈ 12+). Works because fluorescent spots are ~10-50x brighter than background.

### Default `snr_threshold=None`

Preserves backward compatibility. Existing code, tests, and MATLAB-parity are unaffected unless the user opts in. The FOV layer and e2e tests opt in explicitly.

### `"adaptive_round"` as separate mode

Keeps `"adaptive"` (per-channel) for MATLAB compatibility. `"adaptive_round"` is a Python-side improvement. Users and the FOV layer choose which to use.

### `min_distance` kept as parameter

`min_distance` addresses a different problem (clustered noise peaks within signal channels). It remains available as a tunable parameter but is not needed as the primary fix. The e2e fixture reverts to `min_distance=1` (default).

---

## Verification

```bash
# Unit tests for new functionality
cd src/python && uv run pytest test/test_preprocessing.py test/test_spotfinding.py -v

# E2E tests with new defaults
cd src/python && uv run pytest test/test_e2e.py -v -s

# Full regression
cd src/python && uv run pytest test/ -v
```

---

## Expected Outcome

With `snr_threshold=5.0` + `"adaptive_round"` + `min_distance=1`:
- ~1,616 spots (down from 9,044)
- 100% recall (all 20 GT spots found)
- ~0.012 precision (6x improvement)
- No recall loss (unlike `min_distance=2` which drops to 0.7-0.9)
