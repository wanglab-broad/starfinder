starfinder.preprocessing
========================

Intensity normalization and background removal, and the step/recipe contract that
orders them. See :doc:`contracts` for shape and dtype rules and
:doc:`../preprocessing-contract` for steps, recipes and the enforcement wrapper.
``FOV.run`` executes a :py:class:`~starfinder.preprocessing.PreprocessingRecipe` given as
``PipelineConfig.preprocessing``; each step is looked up in :py:data:`~starfinder.preprocessing.STEPS` by
its exact config type and run through :py:func:`~starfinder.preprocessing.run_step`.

.. currentmodule:: starfinder.preprocessing

.. autosummary::
   :toctree: generated

   filter_tophat
   HistogramMatchingConfig
   match_histogram
   MinMaxNormalizationConfig
   normalize_intensity
   PreprocessingRecipe
   project_image
   ProjectionConfig
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
   TophatConfig
