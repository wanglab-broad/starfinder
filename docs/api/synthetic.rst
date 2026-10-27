starfinder.synthetic
====================

Synthetic scenes with their truth, for testing the pipeline against known answers. The
:doc:`synthetic overview <../wiki/synthetic>` explains the four scenes and when to use
each; the :doc:`synthetic reference <../synthetic-reference>` lists their presets and
files. Importing this package does not import benchmark orchestration.

.. currentmodule:: starfinder.synthetic

Amplicon scene
--------------

See the :doc:`amplicon scene guide <../synthetic-specification>`.

.. autosummary::
   :toctree: generated

   amplicon_scene_preset
   AmpliconScene
   AmpliconSceneConfig
   BackgroundConfig
   BENCHMARK_PRESETS
   benchmark_scene_preset
   CALIBRATED_CONDITIONS
   calibrated_scene_preset
   deformation_geometry
   DEFORMATION_PRESETS
   development_codebook
   DEVELOPMENT_FACTORS
   DEVELOPMENT_FIXTURES
   development_preset_factors
   development_scene_preset
   DEVELOPMENT_SIZES
   forward_displacement
   generate_amplicon_dataset
   generate_amplicon_scene
   generate_codebook
   generate_registration_pair
   GeometryConfig
   NoiseConfig
   PRESET_VERSION
   ReadoutEffectsConfig
   registration_scene_preset
   save_amplicon_dataset
   save_amplicon_scene
   ScalarDistribution
   SCENE_PRESETS
   SyntheticDataset
   TextureConfig

Sample scene
------------

See the :doc:`sample scene guide <../synthetic-sample-scene>`.

.. autosummary::
   :toctree: generated

   AcquisitionConfig
   CellsConfig
   generate_sample_scene
   SAMPLE_PRESET_VERSION
   SAMPLE_PRESETS
   sample_scene_preset
   SAMPLE_TRUTH_VERSION
   SampleScene
   SampleSceneConfig
   save_sample_scene
   StainConfig

Cellular scene
--------------

See the :doc:`cellular scene guide <../synthetic-cellular-scene>`.

.. autosummary::
   :toctree: generated

   BrightRegionConfig
   CellTypeConfig
   CELLULAR_PRESET_VERSION
   CELLULAR_PRESETS
   cellular_scene_preset
   CELLULAR_TRUTH_VERSION
   CellularScene
   CellularSceneConfig
   generate_cellular_scene
   MoleculeConfig
   MorphologyRoundConfig
   NucleusShapeConfig
   OutputScaleConfig
   PackingConfig
   save_cellular_dataset
   save_cellular_scene
   StainChannelConfig
   TerritoryConfig
   UnevennessConfig

Real-appearance scene
---------------------

See the :doc:`real-appearance scene guide <../synthetic-real-appearance>`.

.. autosummary::
   :toctree: generated

   APPEARANCE_PRESET_VERSION
   APPEARANCE_PRESETS
   appearance_scene_preset
   APPEARANCE_TRUTH_VERSION
   AppearanceScene
   AppearanceSceneConfig
   AugmentationConfig
   BankReference
   build_appearance_bank
   CompositionConfig
   generate_appearance_scene
   generate_closed_form_bank
   LayoutConfig
   load_appearance_bank
   save_appearance_scene

Examples
--------

Three runnable scripts that check the amplicon scene's options against exact values.
Run one from ``src/python`` with ``uv run python ../../docs/examples/<name>.py``.

.. dropdown:: A small amplicon scene in 3D and in one plane, extracted and decoded

   .. literalinclude:: ../examples/amplicon_scene.py
      :language: python

.. dropdown:: Loss, dropout, weakening, trend, gain and crosstalk, each with its exact effect

   .. literalinclude:: ../examples/readout_effects.py
      :language: python

.. dropdown:: Background shapes and noise, each switched on alone

   .. literalinclude:: ../examples/background_noise.py
      :language: python
