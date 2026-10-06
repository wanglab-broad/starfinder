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

``FOV.run`` and ``FOV.register`` handle the reference round and the sequencing rounds
only. ``FOV.prepare_morphology(MorphologyConfig(...), checkpoints=...)``, an entry
beside ``FOV.run``, loads and rotates the reference stain (the image
``reference_stain``) and loads, rotates and registers each other round, and can save
each prepared image; ``FOV.load_registered_round`` restores one. ``FOV.load_images``
takes ``rotation_degrees``. See :doc:`../coordination`, "Sequencing rounds and other
rounds".

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
   MorphologyConfig
   PipelineConfig
   RecoveryConfig
   RegistrationRecipe
   RegistrationStep
   RoundState
   SubtileConfig
   WorkflowConfig
