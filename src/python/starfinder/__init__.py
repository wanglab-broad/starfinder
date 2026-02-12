"""STARfinder: Spatial transcriptomics data processing pipeline."""

from starfinder import barcode, preprocessing, registration, spotfinding
from starfinder.barcode import extract_from_location, filter_reads, load_codebook
from starfinder.io import load_image_stacks, load_multipage_tiff, save_stack
from starfinder.preprocessing import (
    histogram_match,
    min_max_normalize,
    morphological_reconstruction,
    tophat_filter,
)
from starfinder.registration import (
    apply_shift,
    phase_correlate,
    phase_correlate_skimage,
    register_volume,
)
from starfinder.spotfinding import find_spots_3d
from starfinder.utils import make_projection

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
    # Spot finding
    "spotfinding",
    "find_spots_3d",
    # Barcode processing
    "barcode",
    "extract_from_location",
    "load_codebook",
    "filter_reads",
    # Preprocessing
    "preprocessing",
    "min_max_normalize",
    "histogram_match",
    "morphological_reconstruction",
    "tophat_filter",
    # Utilities
    "make_projection",
    # Package metadata
    "__version__",
]
