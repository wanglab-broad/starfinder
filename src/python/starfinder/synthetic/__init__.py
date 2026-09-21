"""Independent processed-image scene generation, with explicit truth limitations."""
from ._config import SyntheticConfig
from ._truth import SyntheticDataset
from ._presets import generate_codebook, get_preset_config
from ._fields import generate_displacement_field
from ._rendering import render_spots, generate_volume
from ._generation import generate_dataset, generate_registration_pairs

__all__ = ['SyntheticConfig', 'SyntheticDataset', 'generate_codebook',
           'get_preset_config', 'generate_displacement_field', 'render_spots',
           'generate_volume', 'generate_dataset', 'generate_registration_pairs']

from ._formed import (BackgroundConfig, NoiseConfig, TextureConfig,
                      FormedScene, FormedSceneConfig, ReadoutEffectsConfig, ScalarDistribution,
                      formed_scene_preset, generate_formed_scene)

__all__ += ['BackgroundConfig', 'NoiseConfig', 'TextureConfig',
            'FormedScene', 'FormedSceneConfig', 'ReadoutEffectsConfig', 'ScalarDistribution',
            'formed_scene_preset', 'generate_formed_scene']

from ._geometry import GeometryConfig

__all__ += ["GeometryConfig"]

from ._development import (DEVELOPMENT_FACTORS as _DEVELOPMENT_FACTORS,
                           DEVELOPMENT_SIZES as _DEVELOPMENT_SIZES,
                           development_preset_factors, development_scene_preset)

#: Named individual controls in the controlled-development-v1 package.
DEVELOPMENT_FACTORS = _DEVELOPMENT_FACTORS
#: Development size names mapped to bounded ZYX voxel shapes.
DEVELOPMENT_SIZES = _DEVELOPMENT_SIZES

__all__ += ["DEVELOPMENT_FACTORS", "DEVELOPMENT_SIZES",
            "development_preset_factors", "development_scene_preset"]
