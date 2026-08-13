# Fix `SubtileConfig` construction in `STARMapDataset.from_config`

**Date:** 2026-08-13
**Status:** FINISHED

## Problem

`STARMapDataset.from_config` calls `SubtileConfig.compute_windows` on the class
and passes `sqrt_pieces`, but `compute_windows` is an instance method. This
raises `TypeError` whenever the subtile configuration is enabled, blocking both
the `gr_single_fov_subtile` and `deep_create_subtile` Python workflows.

## Plan

1. Add a regression test for each rule entry point that builds a dataset from a
   subtile-enabled configuration and verifies the generated windows.
2. Construct `SubtileConfig` with `sqrt_pieces`, then call
   `compute_windows(height, width)` on that instance.
3. Run the focused dataset tests followed by the complete Python test suite.
