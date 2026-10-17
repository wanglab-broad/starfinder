"""Formed-scene synthetic generation: oracle fixtures and benchmark datasets."""
from ._common import ScalarDistribution
from ._development import (DEVELOPMENT_FACTORS as _DEVELOPMENT_FACTORS,
                           DEVELOPMENT_FIXTURES as _DEVELOPMENT_FIXTURES,
                           DEVELOPMENT_SIZES as _DEVELOPMENT_SIZES,
                           development_preset_factors, development_scene_preset)
from ._formed import (FormedScene, FormedSceneConfig, ReadoutEffectsConfig,
                      formed_scene_preset, generate_formed_scene)
from ._geometry import GeometryConfig
from ._observation import BackgroundConfig, NoiseConfig, TextureConfig
from ._storage import save_formed_scene
from ._presets import (BENCHMARK_PRESETS as _BENCHMARK_PRESETS,
                       DEFORMATION_PRESETS as _DEFORMATION_PRESETS, PRESET_VERSION,
                       _registry, benchmark_scene_preset, deformation_geometry,
                       generate_codebook, registration_scene_preset)
from ._datasets import (SyntheticDataset, forward_displacement, generate_dataset,
                        generate_registration_pair)
from ._calibrated import (CALIBRATED_CONDITIONS as _CALIBRATED_CONDITIONS, calibrated_scene_preset,
                          development_codebook)
from ._sample import (SAMPLE_PRESET_VERSION, SAMPLE_PRESETS as _SAMPLE_PRESETS, SAMPLE_TRUTH_VERSION,
                      AcquisitionConfig, CellularSceneConfig, SampleScene, SampleSceneConfig, StainConfig,
                      generate_sample_scene, sample_scene_preset, save_sample_scene)
from ._stains import (BrightRegionConfig, MorphologyRoundConfig, OutputScaleConfig, StainChannelConfig,
                      UnevennessConfig)
from ._context import (CONTEXT_PRESET_VERSION, CONTEXT_TRUTH_VERSION, CellTypeConfig, ContextScene, ContextSceneConfig, MoleculeConfig,
                       NucleusShapeConfig, PackingConfig, TerritoryConfig, generate_context_scene,
                       save_context_dataset, save_context_scene)
from ._context_presets import CONTEXT_PRESETS as _CONTEXT_PRESETS, context_scene_preset
from ._appearance_bank import BankReference, build_appearance_bank, closed_form_bank, load_appearance_bank

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
#: Cellular-scene context presets (context-presets-v1), separate from SAMPLE_PRESETS.
CONTEXT_PRESETS = _CONTEXT_PRESETS

__all__ = ['BackgroundConfig', 'CALIBRATED_CONDITIONS', 'calibrated_scene_preset', 'DEVELOPMENT_FACTORS',
           'DEVELOPMENT_FIXTURES', 'DEVELOPMENT_SIZES', 'development_codebook', 'development_preset_factors',
           'development_scene_preset',
           'formed_scene_preset', 'FormedScene', 'FormedSceneConfig', 'generate_formed_scene',
           'GeometryConfig', 'NoiseConfig', 'ReadoutEffectsConfig', 'save_formed_scene',
           'ScalarDistribution', 'TextureConfig', 'BENCHMARK_PRESETS', 'DEFORMATION_PRESETS',
           'PRESET_VERSION', 'SCENE_PRESETS', 'SyntheticDataset', 'benchmark_scene_preset',
           'deformation_geometry', 'forward_displacement', 'generate_codebook', 'generate_dataset',
           'generate_registration_pair', 'registration_scene_preset',
           'AcquisitionConfig', 'CellularSceneConfig', 'generate_sample_scene', 'SAMPLE_PRESET_VERSION',
           'SAMPLE_PRESETS', 'SAMPLE_TRUTH_VERSION', 'SampleScene', 'SampleSceneConfig', 'sample_scene_preset',
           'save_sample_scene', 'StainConfig', 'CellTypeConfig', 'CONTEXT_TRUTH_VERSION', 'ContextScene',
           'ContextSceneConfig', 'generate_context_scene', 'MoleculeConfig', 'NucleusShapeConfig', 'PackingConfig',
           'save_context_scene', 'TerritoryConfig', 'BrightRegionConfig', 'MorphologyRoundConfig',
           'OutputScaleConfig', 'save_context_dataset', 'StainChannelConfig', 'UnevennessConfig',
           'CONTEXT_PRESET_VERSION', 'CONTEXT_PRESETS', 'context_scene_preset', 'BankReference',
           'build_appearance_bank', 'closed_form_bank', 'load_appearance_bank']
