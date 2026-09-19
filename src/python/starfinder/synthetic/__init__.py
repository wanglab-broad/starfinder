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
