starfinder.io
=============

TIFF readers preserve dtype and require explicit selection for ambiguous axes.
See :doc:`contracts` for conversion, geometry and channel ordering.
The :doc:`output guide <../recipes>` shows loading and saving examples.

.. currentmodule:: starfinder.io

Loading
-------

.. autosummary::
   :toctree: generated

   ImageLoadConfig
   ImageLoadResult
   load_round
   load_volume
   load_volume_zyxc

Conversion and saving
---------------------

.. autosummary::
   :toctree: generated

   convert_image
   ImageConversionConfig
   save_volume
