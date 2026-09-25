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

#: Named single-factor controls available on demand from development_scene_preset.
DEVELOPMENT_FACTORS = _DEVELOPMENT_FACTORS
#: Packaged fixture names mapped to (condition, size); checked by an independent oracle.
DEVELOPMENT_FIXTURES = _DEVELOPMENT_FIXTURES
#: Development size names mapped to ZYX voxel shapes.
DEVELOPMENT_SIZES = _DEVELOPMENT_SIZES

#: Benchmark preset names mapped to historical shape, count, seed, FOVs, shifts and genes.
BENCHMARK_PRESETS = _BENCHMARK_PRESETS
#: Historical deformation names mapped to percent/cap parameters (see deformation_geometry).
DEFORMATION_PRESETS = _DEFORMATION_PRESETS
#: The single scene preset registry, with fixture, development_fixture and benchmark tiers.
SCENE_PRESETS = _registry()

__all__ = ['BackgroundConfig', 'DEVELOPMENT_FACTORS', 'DEVELOPMENT_FIXTURES',
           'DEVELOPMENT_SIZES', 'development_preset_factors', 'development_scene_preset',
           'formed_scene_preset', 'FormedScene', 'FormedSceneConfig', 'generate_formed_scene',
           'GeometryConfig', 'NoiseConfig', 'ReadoutEffectsConfig', 'save_formed_scene',
           'ScalarDistribution', 'TextureConfig', 'BENCHMARK_PRESETS', 'DEFORMATION_PRESETS',
           'PRESET_VERSION', 'SCENE_PRESETS', 'SyntheticDataset', 'benchmark_scene_preset',
           'deformation_geometry', 'forward_displacement', 'generate_codebook', 'generate_dataset',
           'generate_registration_pair', 'registration_scene_preset']
