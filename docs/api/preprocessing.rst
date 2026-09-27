starfinder.preprocessing
========================

Intensity normalization and background removal, and the step/recipe contract that
orders them. See :doc:`contracts` for shape and dtype rules and
:doc:`../preprocessing-contract` for steps, recipes and the enforcement wrapper.
``FOV.run`` executes a :py:class:`~starfinder.preprocessing.PreprocessingRecipe` given as
``PipelineConfig.preprocessing``; each step is looked up in :py:data:`~starfinder.preprocessing.STEPS` by
its exact config type and run through :py:func:`~starfinder.preprocessing.run_step`.

Snapshots and sources
---------------------

A :py:class:`~starfinder.preprocessing.RecipeStep` with ``save_as`` keeps a named snapshot of
its output. The recipe's ``extraction_source`` names the snapshot that extraction reads,
for example the background-corrected image before normalization (recipe 2), and
``registration_source`` names the snapshot from which registration signals are built;
both default to the detection image, the last step's output. ``FOV.run`` applies each
registration step's transform to every snapshot of a moving round, so the snapshots stay
aligned with the detection image, and records per round and snapshot the transforms
applied. The workflow key ``preprocessing`` declares the same recipe in YAML; see
:doc:`../workflow-configuration`.

Background subtraction
----------------------

:py:class:`~starfinder.preprocessing.ScalarBackgroundConfig` (step ``scalar_background``)
subtracts one level per channel and round, the inverted-CDF ``percentile`` of the
channel's voxels, and clips at zero.
:py:class:`~starfinder.preprocessing.Background3DConfig` (step ``background_3d``) subtracts a
grey opening with an anisotropic ellipsoidal footprint, a volumetric white top-hat.
Both keep the input dtype and record the shared per-channel diagnostics (zero fraction,
median, MAD, noise threshold and ``mad_zero``). The XY methods
(:py:class:`~starfinder.preprocessing.ReconstructionConfig`,
:py:class:`~starfinder.preprocessing.TophatConfig`) are unchanged. Choosing the 3D radius
relative to puncta size is described in :doc:`../recipes`.

Sample-level statistics
-----------------------

Steps with ``fit="supplied"`` (percentile normalization, scalar background and the
histogram-matching reference) read their values from the recipe's ``supplied_statistics`` file, schema
``starfinder.preprocessing.supplied/1``, as specified in
:doc:`../preprocessing-algorithms`. A summary pass runs the steps before the fitted
step (:py:func:`~starfinder.preprocessing.summary_stage`), records the integer
histograms of each FOV at that step's input
(:py:func:`~starfinder.preprocessing.summarize_histograms`), sums them
(:py:func:`~starfinder.preprocessing.merge_histograms`) and fits the step's section
(:py:func:`~starfinder.preprocessing.supplied_section`). The application pass runs the
full recipe; ``FOV.run`` validates the file against the recipe, the dataset
``channel_order`` and the rounds before any step runs. The manual two-pass example
below summarizes three FOVs after a white top-hat, merges them, writes the file and
applies it; from ``src/python``, run
``uv run python ../../docs/examples/percentile_two_pass.py <new output directory>``.
When only scalar background steps lie between two supplied steps,
:py:func:`~starfinder.preprocessing.scalar_background_histograms` derives the later
summary exactly from the earlier one, per FOV before merging (``fit="fov"``) or from the
merged counts (``fit="supplied"``), without another pass over the images.

.. literalinclude:: ../examples/percentile_two_pass.py
   :language: python

API
---

.. currentmodule:: starfinder.preprocessing

.. autosummary::
   :toctree: generated

   Background3DConfig
   filter_tophat
   histogram_percentile
   HistogramMatchingConfig
   HistogramSummary
   match_histogram
   merge_histograms
   MinMaxNormalizationConfig
   normalize_intensity
   normalize_percentile
   PercentileNormalizationConfig
   PreprocessingRecipe
   project_image
   ProjectionConfig
   read_histograms
   read_supplied_statistics
   RecipeStep
   reconstruct_background
   ReconstructionConfig
   run_step
   scalar_background_histograms
   ScalarBackgroundConfig
   step_config_type
   step_spec
   StepContext
   StepResult
   STEPS
   StepSpec
   subtract_background_3d
   subtract_scalar_background
   summarize_histograms
   summary_stage
   supplied_section
   supplied_statistics
   TophatConfig
   write_histograms
   write_supplied_statistics
