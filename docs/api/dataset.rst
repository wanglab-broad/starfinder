starfinder.dataset
==================

Dataset and FOV coordination with typed pipeline settings. The
:doc:`coordination guide <../coordination>` shows how to run a pipeline; the
:doc:`coordination reference <../coordination-reference>` lists configurations and results.

.. currentmodule:: starfinder.dataset

Dataset, FOV and rounds
-----------------------

.. autosummary::
   :toctree: generated

   ChannelInfo
   Dataset
   FOV
   RoundState

Pipeline and execution
----------------------

.. autosummary::
   :toctree: generated

   ExecutionConfig
   MorphologyConfig
   PipelineConfig

Workflow configuration
----------------------

.. autosummary::
   :toctree: generated

   from_workflow_config
   WorkflowConfig

Checkpoints
-----------

.. autosummary::
   :toctree: generated

   CheckpointConfig
   load_checkpoint

Tiled execution
---------------

A FOV that does not fit one job runs as a global task, one task per tile and a merge, with the outputs of direct
execution (``processing.mode: tiled`` of ``starfinder workflow``). The chunked checkpoint the tasks share is
:class:`starfinder.io.ChunkedCheckpointWriter`, read by window with :func:`starfinder.io.read_window`.

.. currentmodule:: starfinder.dataset.tiled

.. autosummary::
   :toctree: generated

   detector_order
   largest_tile_size
   merge_tiles
   predicted_tile_memory
   run_tile
   run_tile_global
   Tile
   tile_footprints
   tile_grid
   unbounded_operations
