Preprocessing
=============

These helpers operate on cell arrays containing 4-D images. The dataset's
``HistEqualize`` and ``Tophat`` methods supply additional processing operations.
See :doc:`../preprocessing` for Python counterparts, whose normalization and
dtype contracts differ. Disk morphology is applied to each XY plane, not as
a spherical 3-D neighborhood.

.. mat:currentmodule:: .

.. mat:autofunction:: MinMaxNorm

.. mat:autofunction:: MorphologicalReconstruction
