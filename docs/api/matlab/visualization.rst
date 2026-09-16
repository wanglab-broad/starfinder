Projection and spot previews
============================

The workflow uses maximum projections. Montage and centroid helpers return
MATLAB figure handles, despite their ``output_img`` return names. Plotting and
export require graphics support even when MATLAB runs without a desktop.
Python preview/output methods are documented under :doc:`../dataset`.

.. mat:currentmodule:: .

.. mat:autofunction:: MakeProjections

.. mat:autofunction:: MakeMontage

.. mat:autofunction:: PlotCentroids
