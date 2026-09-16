TIFF input and output
=====================

Loaders return numeric arrays in MATLAB row/column/Z/channel order. Helpers
that save multiple channels receive a cell-wrapped image, as used by the
dataset dictionary. See :doc:`../io` for corresponding Python operations.
Output channel filenames need not match acquisition channel identifiers.

.. mat:currentmodule:: .

.. mat:autofunction:: LoadMultipageTiff

.. mat:autofunction:: LoadImageStacks

.. mat:autofunction:: AdjustSizeAcrossRound

.. mat:autofunction:: SaveSingleStack

.. mat:autofunction:: SaveImageNestedFolder

.. mat:autofunction:: SaveImageSingleFolder
