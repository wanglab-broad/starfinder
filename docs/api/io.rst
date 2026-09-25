starfinder.io
=============

TIFF readers preserve dtype and require explicit selection for ambiguous axes.
See :doc:`contracts` for conversion, geometry and channel ordering, and
:doc:`../checkpoints` for the per-FOV checkpoint layout.

.. currentmodule:: starfinder.io

.. autosummary::
   :toctree: generated

   convert_image
   export_spots
   ImageConversionConfig
   ImageLoadConfig
   ImageLoadResult
   load_round
   load_volume
   load_volume_zyxc
   read_checkpoint
   save_volume
