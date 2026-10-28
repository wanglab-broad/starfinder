TIFF input and output
=====================

Loaders return numeric arrays in MATLAB row/column/Z/channel order. Helpers
that save multiple channels receive a cell-wrapped image, as used by the
dataset dictionary. See `starfinder.io <../../docs/api/io.rst>`_ for corresponding Python operations.
Output channel filenames need not match acquisition channel identifiers.

