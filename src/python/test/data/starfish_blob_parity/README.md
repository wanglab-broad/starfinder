# starfish BlobDetector parity tables

Expected outputs of starfish `BlobDetector` (starfish `1fb00cbc`, reported version
`0.4.0+38.g1fb00cb`; `blob.py` is identical at 0.4.0, `d9a305f`) for the native LoG method
that §2.7 adds (decision D1; docs/spot-finding-algorithms.md). No test reads them yet; the
LoG implementation issue adds the parity test.

* `fixtures.py` regenerates the input arrays (NumPy only, float32 RCZYX in [0, 1]) and
  defines the parity cases and settings. Each case's array SHA-256 is in
  `starfish-tables.json`.
* `<case>_r<round>_c<channel>.csv` is starfish's table for one (round, channel), columns
  and dtypes as listed in `starfish-tables.json`; `empty-isotropic_r0_c0.csv` is starfish's
  typed empty frame (`x, y, z, radius, intensity, spot_id`).
* Settings: `BlobDetector(min_sigma, max_sigma, num_sigma, threshold, overlap=0.5,
  exclude_border=False, is_volume=True, detector_method="blob_log")`, no reference image,
  through `ImageStack.from_numpy`. starfish ran with NumPy 2.2.6, scikit-image 0.26.0 and
  pandas 3.0.0.

Copied unchanged (the tables and `fixtures.py`) from the W-266 run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-266/20260930T225252Z-967e52bd/parity/`,
where W-266 showed that its prototype (`blob_log` plus the four post-processing steps)
equals every table with `pd.testing.assert_frame_equal(check_exact=True)`
(`parity-check.json`). `starfish-tables.json` is that directory's file without the
host-specific `starfish_file` path and the per-table prototype flags.
