starfinder.io
=============

TIFF readers preserve dtype and require explicit selection for ambiguous axes.
See :doc:`contracts` for conversion, geometry and channel ordering.
Opt-in per-FOV HDF5 storage and reload are described in :doc:`../image-checkpoints`.
Pre-rejection Parquet persistence is described in :doc:`../candidate-checkpoints`.
Decoded/final and sample access are described in :doc:`../molecular-checkpoints`.

.. currentmodule:: starfinder.io

.. autosummary::
   :toctree: generated

   ArtifactReference
   CandidateCheckpoint
   CandidateSaveResult
   checkpoint_reference
   convert_image
   DecodedCheckpoint
   export_spots
   FinalCheckpoint
   ImageCheckpoint
   ImageConversionConfig
   ImageLayer
   ImageLoadConfig
   ImageLoadResult
   ImageProcessingState
   load_candidate_checkpoint
   load_decoded_checkpoint
   load_final_checkpoint
   load_image_checkpoint
   load_molecule_index
   load_round
   load_volume
   MoleculeBatch
   MoleculeIndex
   save_candidate_checkpoint
   save_decoded_checkpoint
   save_final_checkpoint
   save_image_checkpoint
   save_molecule_index
   save_volume
