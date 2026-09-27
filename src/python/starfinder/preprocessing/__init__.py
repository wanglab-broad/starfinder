"""Finite ZYX/ZYXC image operations, their typed configuration and the step/recipe contract."""
from starfinder.preprocessing.morphology import ReconstructionConfig, TophatConfig, reconstruct_background, filter_tophat
from starfinder.preprocessing.normalization import HistogramMatchingConfig, MinMaxNormalizationConfig, match_histogram, normalize_intensity
from starfinder.preprocessing.projection import ProjectionConfig, project_image
from starfinder.preprocessing.steps import (STEPS as _STEPS, PreprocessingRecipe, RecipeStep, StepContext, StepResult,
    StepSpec, run_step, step_config_type, step_spec)

#: The single explicit step registry, mapping each exact frozen config type to its
#: StepSpec. The lookup from step name to config type (step_config_type) is derived from it.
STEPS = _STEPS

__all__ = ["HistogramMatchingConfig", "MinMaxNormalizationConfig", "PreprocessingRecipe", "ProjectionConfig", "RecipeStep", "ReconstructionConfig", "STEPS", "StepContext", "StepResult", "StepSpec", "TophatConfig", "filter_tophat", "match_histogram", "normalize_intensity", "project_image", "reconstruct_background", "run_step", "step_config_type", "step_spec"]
