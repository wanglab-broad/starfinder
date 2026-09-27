starfinder.preprocessing
========================

Intensity normalization and background removal, and the step/recipe contract that
orders them. See :doc:`contracts` for shape and dtype rules and
:doc:`../preprocessing-contract` for steps, recipes and the enforcement wrapper.
``FOV.run`` executes a :py:class:`~starfinder.preprocessing.PreprocessingRecipe` given as
``PipelineConfig.preprocessing``; each step is looked up in :py:data:`~starfinder.preprocessing.STEPS` by
its exact config type and run through :py:func:`~starfinder.preprocessing.run_step`.

Sample-level statistics
-----------------------

Steps with ``fit="supplied"`` (percentile normalization and the histogram-matching
reference) read their values from the recipe's ``supplied_statistics`` file, schema
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

.. literalinclude:: ../examples/percentile_two_pass.py
   :language: python

API
---

.. currentmodule:: starfinder.preprocessing

.. autosummary::
   :toctree: generated

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
   step_config_type
   step_spec
   StepContext
   StepResult
   STEPS
   StepSpec
   summarize_histograms
   summary_stage
   supplied_section
   supplied_statistics
   TophatConfig
   write_histograms
   write_supplied_statistics
