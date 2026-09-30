"""Finite ZYX/ZYXC image operations, their typed configuration and the step/recipe contract."""
from starfinder.preprocessing.background import (Background3DConfig, ScalarBackgroundConfig, scalar_background_histograms,
    subtract_background_3d, subtract_scalar_background)
from starfinder.preprocessing.histograms import (HistogramSummary, histogram_percentile, merge_histograms, read_histograms,
    summarize_histograms, write_histograms)
from starfinder.preprocessing.morphology import ReconstructionConfig, TophatConfig, reconstruct_background, filter_tophat
from starfinder.preprocessing.normalization import (HistogramMatchingConfig, MinMaxNormalizationConfig,
    PercentileNormalizationConfig, match_histogram, normalize_intensity, normalize_percentile)
from starfinder.preprocessing.projection import ProjectionConfig, project_image
from starfinder.preprocessing.steps import (PREPROCESSING_METHODS as _PREPROCESSING_METHODS, PreprocessingRecipe,
    PreprocessingSpec, PreprocessingStep, StepContext, StepResult, run_step, step_config_type, step_spec)
from starfinder.preprocessing.supplied import (read_supplied_statistics, summary_stage, supplied_section,
    supplied_statistics, write_supplied_statistics)

#: The single explicit step registry, mapping each exact frozen config type to its
#: PreprocessingSpec. The lookup from step name to config type (step_config_type) is derived from it.
PREPROCESSING_METHODS = _PREPROCESSING_METHODS

__all__ = ["Background3DConfig", "HistogramMatchingConfig", "HistogramSummary", "MinMaxNormalizationConfig", "PercentileNormalizationConfig", "PREPROCESSING_METHODS", "PreprocessingRecipe", "PreprocessingSpec", "PreprocessingStep", "ProjectionConfig", "ReconstructionConfig", "ScalarBackgroundConfig", "StepContext", "StepResult", "TophatConfig", "filter_tophat", "histogram_percentile", "match_histogram", "merge_histograms", "normalize_intensity", "normalize_percentile", "project_image", "read_histograms", "read_supplied_statistics", "reconstruct_background", "run_step", "scalar_background_histograms", "step_config_type", "step_spec", "subtract_background_3d", "subtract_scalar_background", "summarize_histograms", "summary_stage", "supplied_section", "supplied_statistics", "write_histograms", "write_supplied_statistics"]
