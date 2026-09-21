starfinder.io
=============

TIFF readers preserve dtype and require explicit selection for ambiguous axes.
See :doc:`contracts` for conversion, geometry and channel ordering.
Opt-in per-FOV HDF5 storage and reload are described in :doc:`../image-checkpoints`.

.. currentmodule:: starfinder.io

.. autosummary::
   :toctree: generated

   convert_image
   export_spots
   ImageCheckpoint
   ImageConversionConfig
   ImageLayer
   ImageLoadConfig
   ImageLoadResult
   ImageProcessingState
   load_image_checkpoint
   load_round
   load_volume
   save_image_checkpoint
   save_volume
