"""STARfinder: Spatial transcriptomics data processing pipeline."""

from starfinder.image import ImageMetadata
from starfinder import barcode, preprocessing, registration, spot_finding
from starfinder.barcode import extract_from_location, filter_reads, load_codebook
from starfinder.dataset import FOV, STARMapDataset
from starfinder.io import load_round, load_volume, save_volume
from starfinder.preprocessing import (
    match_histogram,
    normalize_intensity,
    reconstruct_background,
    filter_tophat,
)
from starfinder.registration import (
    apply_shift,
    phase_correlate,
    phase_correlate_skimage,
    register_volume,
)
from starfinder.spot_finding import find_spots
from starfinder.preprocessing import project_image

__version__ = "0.1.0"

__all__ = [
    "ImageMetadata",
    # Dataset/FOV orchestration
    "STARMapDataset",
    "FOV",
    # I/O functions
    "load_volume",
    "load_round",
    "save_volume",
    # Registration module and functions
    "registration",
    "phase_correlate",
    "apply_shift",
    "register_volume",
    "phase_correlate_skimage",
    # Spot finding
    "spot_finding",
    "find_spots",
    # Barcode processing
    "barcode",
    "extract_from_location",
    "load_codebook",
    "filter_reads",
    # Preprocessing
    "preprocessing",
    "normalize_intensity",
    "match_histogram",
    "reconstruct_background",
    "filter_tophat",
    # Utilities
    "project_image",
    # Package metadata
    "__version__",
]
