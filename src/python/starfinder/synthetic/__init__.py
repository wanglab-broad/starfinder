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

#: Named single-factor controls available on demand from development_scene_preset.
DEVELOPMENT_FACTORS = _DEVELOPMENT_FACTORS
#: Packaged fixture names mapped to (condition, size); checked by an independent oracle.
DEVELOPMENT_FIXTURES = _DEVELOPMENT_FIXTURES
#: Development size names mapped to ZYX voxel shapes.
DEVELOPMENT_SIZES = _DEVELOPMENT_SIZES

__all__ += ['BackgroundConfig', 'DEVELOPMENT_FACTORS', 'DEVELOPMENT_FIXTURES',
            'DEVELOPMENT_SIZES', 'development_preset_factors', 'development_scene_preset',
            'formed_scene_preset', 'FormedScene', 'FormedSceneConfig', 'generate_formed_scene',
            'GeometryConfig', 'NoiseConfig', 'ReadoutEffectsConfig', 'save_formed_scene',
            'ScalarDistribution', 'TextureConfig']
