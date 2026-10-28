starfinder.preprocessing
========================

Intensity normalization, background removal and preprocessing recipes. The
:doc:`preprocessing guide <../preprocessing-contract>` shows how to run the steps; the
:doc:`preprocessing reference <../preprocessing-algorithms>` lists methods and options.

.. currentmodule:: starfinder.preprocessing

Steps and recipes
-----------------

.. autosummary::
   :toctree: generated

   PREPROCESSING_METHODS
   PreprocessingRecipe
   PreprocessingSpec
   PreprocessingStep
   run_step
   step_config_type
   step_spec
   StepContext
   StepResult

Normalization and histogram matching
------------------------------------

.. autosummary::
   :toctree: generated

   HistogramMatchingConfig
   match_histogram
   MinMaxNormalizationConfig
   normalize_min_max
   normalize_percentile
   PercentileNormalizationConfig

Background removal
------------------

.. autosummary::
   :toctree: generated

   Background3DConfig
   correct_background
   filter_tophat
   ReconstructionConfig
   ScalarBackgroundConfig
   subtract_background_3d
   subtract_scalar_background
   TophatConfig

Projection
----------

.. autosummary::
   :toctree: generated

   project_image
   ProjectionConfig

Sample-level statistics
-----------------------

.. autosummary::
   :toctree: generated

   fit_supplied_section
   histogram_percentile
   HistogramSummary
   load_histograms
   load_supplied_statistics
   merge_histograms
   save_histograms
   save_supplied_statistics
   scalar_background_histograms
   summarize_histograms
   summary_recipe
   supplied_statistics

Examples
--------

Run from ``src/python`` with
``uv run python ../../docs/examples/percentile_two_pass.py OUTPUT_DIR``.

.. dropdown:: Fit and apply sample-level statistics in two passes

   .. literalinclude:: ../examples/percentile_two_pass.py
      :language: python
