# Fix 4D reference-image preview in `stitch_subtile`

**Date:** 2026-08-14
**Status:** FINISHED

## Problem

The Python `stitch_subtile` workflow passes a saved `(Z, Y, X, C)` reference
image directly to Matplotlib when `maximum_projection` is disabled. Matplotlib
cannot display that 4D shape, so the final subtile stitching job fails after
writing its reads CSV.

## Plan

1. Add a script-level regression test that runs `stitch_subtile.py` with a 4D
   reference TIFF and verifies both declared outputs are produced.
2. Collapse 4D reference images over their Z and channel axes before plotting,
   while preserving the existing 3D handling.
3. Run the focused regression test and the complete Python test suite.
