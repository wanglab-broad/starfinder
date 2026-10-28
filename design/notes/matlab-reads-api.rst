Spot finding, extraction and decoding
=====================================

The spot table's coordinates are 1-based X/Y/Z, while image arrays are
row/column/Z/channel. Color labels are 1-based channel numbers. MATLAB spot
finding accepts adaptive/global threshold modes; Python's noise mode is not
implemented here. Always choose intensity mode and threshold together.
See `starfinder.spot_finding <../../docs/api/spot_finding.rst>`_ and `starfinder.readout <../../docs/api/readout.rst>`_ for Python interfaces.

The workflow class discards N/M calls before filtering. The filtering helpers
use color-codebook membership to select reads; terminal-base statistics and
the class option ``q_score_thershold`` do not impose additional selection.
Do not interpret the Python and MATLAB score/filter options as equivalent.

