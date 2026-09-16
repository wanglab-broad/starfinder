Spot finding, extraction and decoding
=====================================

The spot table's coordinates are 1-based X/Y/Z, while image arrays are
row/column/Z/channel. Color labels are 1-based channel numbers. MATLAB spot
finding accepts adaptive/global threshold modes; Python's noise mode is not
implemented here. Always choose intensity mode and threshold together.
See :doc:`../spotfinding` and :doc:`../barcode` for Python interfaces.

The workflow class discards N/M calls before filtering. The filtering helpers
use color-codebook membership to select reads; terminal-base statistics and
the class option ``q_score_thershold`` do not impose additional selection.
Do not interpret the Python and MATLAB score/filter options as equivalent.

.. mat:currentmodule:: .

.. mat:autofunction:: SpotFindingMax3D

.. mat:autofunction:: ExtractFromLocation

.. mat:autofunction:: EncodeBases

.. mat:autofunction:: DecodeCS

.. mat:autofunction:: Str2Colorseq

.. mat:autofunction:: LoadCodebook

.. mat:autofunction:: FilterReads

.. mat:autofunction:: FilterReadsMultiSegment
