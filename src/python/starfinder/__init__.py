"""STARfinder: Spatial transcriptomics data processing pipeline."""

from starfinder import registration
from starfinder.io import load_image_stacks, load_multipage_tiff, save_stack
from starfinder.registration import (
    apply_shift,
    phase_correlate,
    phase_correlate_skimage,
    register_volume,
)

__version__ = "0.1.0"

__all__ = [
    # I/O functions
    "load_multipage_tiff",
    "load_image_stacks",
    "save_stack",
    # Registration module and functions
    "registration",
    "phase_correlate",
    "apply_shift",
    "register_volume",
    "phase_correlate_skimage",
    # Package metadata
    "__version__",
]
