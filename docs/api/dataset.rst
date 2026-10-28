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

   CropWindow
   ExecutionConfig
   MorphologyConfig
   PipelineConfig
   SubtileConfig

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
