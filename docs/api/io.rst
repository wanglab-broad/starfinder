starfinder.io
=============

TIFF readers preserve dtype and require explicit selection for ambiguous axes.
See :doc:`contracts` for conversion, geometry and channel ordering.
Opt-in per-FOV HDF5 storage and reload are described in :doc:`../image-checkpoints`.
Pre-rejection Parquet persistence is described in :doc:`../candidate-checkpoints`.

.. currentmodule:: starfinder.io

.. autosummary::
   :toctree: generated

   CandidateCheckpoint
   CandidateSaveResult
   convert_image
   export_spots
   ImageCheckpoint
   ImageConversionConfig
   ImageLayer
   ImageLoadConfig
   ImageLoadResult
   ImageProcessingState
   load_candidate_checkpoint
   load_image_checkpoint
   load_round
   load_volume
   save_candidate_checkpoint
   save_image_checkpoint
   save_volume
