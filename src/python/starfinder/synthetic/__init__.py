"""Synthetic generation: amplicon, sample, cellular and appearance scenes, and benchmark datasets."""
from ._common import ScalarDistribution
from ._development import (DEVELOPMENT_FACTORS as _DEVELOPMENT_FACTORS,
                           DEVELOPMENT_FIXTURES as _DEVELOPMENT_FIXTURES,
                           DEVELOPMENT_SIZES as _DEVELOPMENT_SIZES,
                           development_preset_factors, development_scene_preset)
from ._formed import (AmpliconScene, AmpliconSceneConfig, ReadoutEffectsConfig,
                      amplicon_scene_preset, generate_amplicon_scene)
from ._geometry import GeometryConfig
from ._observation import BackgroundConfig, NoiseConfig, TextureConfig
from ._storage import save_amplicon_dataset, save_amplicon_scene
from ._presets import (BENCHMARK_PRESETS as _BENCHMARK_PRESETS,
                       DEFORMATION_PRESETS as _DEFORMATION_PRESETS, PRESET_VERSION,
                       _registry, benchmark_scene_preset, deformation_geometry,
                       generate_codebook, registration_scene_preset)
from ._datasets import (SyntheticDataset, forward_displacement, generate_amplicon_dataset,
                        generate_registration_pair)
from ._calibrated import (CALIBRATED_CONDITIONS as _CALIBRATED_CONDITIONS, calibrated_scene_preset,
                          development_codebook)
from ._sample import (SAMPLE_PRESET_VERSION, SAMPLE_PRESETS as _SAMPLE_PRESETS, SAMPLE_TRUTH_VERSION,
                      AcquisitionConfig, CellsConfig, SampleScene, SampleSceneConfig, StainConfig,
                      generate_sample_scene, sample_scene_preset, save_sample_scene)
from ._stains import (BrightRegionConfig, MorphologyRoundConfig, OutputScaleConfig, StainChannelConfig,
                      UnevennessConfig)
from ._context import (CELLULAR_PRESET_VERSION, CELLULAR_TRUTH_VERSION, CellTypeConfig, CellularScene,
                       CellularSceneConfig, MoleculeConfig, NucleusShapeConfig, PackingConfig, TerritoryConfig,
                       generate_cellular_scene, save_cellular_dataset, save_cellular_scene)
from ._context_presets import CELLULAR_PRESETS as _CONTEXT_PRESETS, cellular_scene_preset
from ._appearance_bank import (BankReference, build_appearance_bank, generate_closed_form_bank,
                               load_appearance_bank)
from ._appearance_layout import AugmentationConfig, LayoutConfig
from ._appearance import (APPEARANCE_TRUTH_VERSION, AppearanceScene, AppearanceSceneConfig, CompositionConfig,
                          generate_appearance_scene, save_appearance_scene)
from ._appearance_presets import (APPEARANCE_PRESET_VERSION, APPEARANCE_PRESETS as _APPEARANCE_PRESETS,
                                  appearance_scene_preset)

#: Named single-factor controls available on demand from development_scene_preset.
DEVELOPMENT_FACTORS = _DEVELOPMENT_FACTORS
#: Packaged fixture names mapped to (condition, size); checked by an independent oracle.
DEVELOPMENT_FIXTURES = _DEVELOPMENT_FIXTURES
#: Development size names mapped to ZYX voxel shapes.
DEVELOPMENT_SIZES = _DEVELOPMENT_SIZES
#: Calibrated (calibrated-development-v1) condition names mapped to the factors added to clean.
CALIBRATED_CONDITIONS = _CALIBRATED_CONDITIONS

#: Benchmark preset names mapped to historical shape, count, seed, FOVs, shifts and genes.
BENCHMARK_PRESETS = _BENCHMARK_PRESETS
#: Historical deformation names mapped to percent/cap parameters (see deformation_geometry).
DEFORMATION_PRESETS = _DEFORMATION_PRESETS
#: The single scene preset registry, with fixture, development_fixture and benchmark tiers.
SCENE_PRESETS = _registry()
#: Multi-FOV sample scene presets (sample-presets-v1), separate from SCENE_PRESETS.
SAMPLE_PRESETS = _SAMPLE_PRESETS
#: Cellular scene presets (context-presets-v1), separate from SAMPLE_PRESETS.
CELLULAR_PRESETS = _CONTEXT_PRESETS
#: Real-appearance scene presets (appearance-presets-v1), separate from CELLULAR_PRESETS.
APPEARANCE_PRESETS = _APPEARANCE_PRESETS

__all__ = ['BackgroundConfig', 'CALIBRATED_CONDITIONS', 'calibrated_scene_preset', 'DEVELOPMENT_FACTORS',
           'DEVELOPMENT_FIXTURES', 'DEVELOPMENT_SIZES', 'development_codebook', 'development_preset_factors',
           'development_scene_preset',
           'amplicon_scene_preset', 'AmpliconScene', 'AmpliconSceneConfig', 'generate_amplicon_scene',
           'GeometryConfig', 'NoiseConfig', 'ReadoutEffectsConfig', 'save_amplicon_scene',
           'ScalarDistribution', 'TextureConfig', 'BENCHMARK_PRESETS', 'DEFORMATION_PRESETS',
           'PRESET_VERSION', 'SCENE_PRESETS', 'SyntheticDataset', 'benchmark_scene_preset',
           'deformation_geometry', 'forward_displacement', 'generate_codebook', 'generate_amplicon_dataset',
           'generate_registration_pair', 'registration_scene_preset', 'save_amplicon_dataset',
           'AcquisitionConfig', 'CellsConfig', 'generate_sample_scene', 'SAMPLE_PRESET_VERSION',
           'SAMPLE_PRESETS', 'SAMPLE_TRUTH_VERSION', 'SampleScene', 'SampleSceneConfig', 'sample_scene_preset',
           'save_sample_scene', 'StainConfig', 'CellTypeConfig', 'CELLULAR_TRUTH_VERSION', 'CellularScene',
           'CellularSceneConfig', 'generate_cellular_scene', 'MoleculeConfig', 'NucleusShapeConfig',
           'PackingConfig',
           'save_cellular_scene', 'TerritoryConfig', 'BrightRegionConfig', 'MorphologyRoundConfig',
           'OutputScaleConfig', 'save_cellular_dataset', 'StainChannelConfig', 'UnevennessConfig',
           'CELLULAR_PRESET_VERSION', 'CELLULAR_PRESETS', 'cellular_scene_preset', 'BankReference',
           'build_appearance_bank', 'generate_closed_form_bank', 'load_appearance_bank',
           'APPEARANCE_TRUTH_VERSION',
           'AppearanceScene', 'AppearanceSceneConfig', 'AugmentationConfig', 'CompositionConfig', 'LayoutConfig',
           'generate_appearance_scene', 'save_appearance_scene', 'APPEARANCE_PRESET_VERSION', 'APPEARANCE_PRESETS',
           'appearance_scene_preset']
