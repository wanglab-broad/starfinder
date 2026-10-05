starfinder.dataset
==================

Dataset/FOV coordination with typed scientific stages and separate residency.
FOV remains the public acronym. See :doc:`contracts` and :doc:`../coordination`.

Evaluating a ``Dataset`` or ``FOV`` shows a plain-text summary of names and
structure, never array values or table rows. ``FOV.results`` is a read-only
mapping of the stages that have run (``registration``, ``spot_finding``,
``extraction``, ``decoding``, ``filtering``) to the stored result objects;
the existing attributes remain. ``Codebook``, the spot finding, extraction,
decoding, filtering, registration and evaluation results have one-line
summaries. See :ref:`inspecting-results`.

``PipelineConfig.registration`` is a ``RegistrationRecipe`` (or ``None``): global steps,
at most one local step, one signal and one final resampling per moving round; see
:doc:`../registration-contract`. ``FOV.register_rounds`` registers other rounds to a
reference round or an ``ExternalReference`` through a shared stain.

Each channel is a ``ChannelInfo`` (file pattern, content name, wavelength in nm).
``Dataset.channel_info(round)`` returns a round's channels (``reference_stain``: the
reference stains), and ``Dataset.channel_index(round, key)`` is the one lookup by
index, pattern or name; see :doc:`../coordination`, "Channels".

.. currentmodule:: starfinder.dataset

.. autosummary::
   :toctree: generated

   ChannelInfo
   CheckpointConfig
   CropWindow
   Dataset
   ExecutionConfig
   ExternalReference
   FOV
   from_workflow_config
   PipelineConfig
   RecoveryConfig
   RegistrationRecipe
   RegistrationStep
   RoundState
   SubtileConfig
   WorkflowConfig
