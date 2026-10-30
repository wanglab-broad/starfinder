"""STARfinder: Spatial transcriptomics data processing pipeline."""

from importlib.metadata import PackageNotFoundError, version as _package_version

from starfinder.image import ImageMetadata
from starfinder import readout, preprocessing, registration, spot_finding
from starfinder.readout import decode_reads, extract_intensities, filter_reads, load_codebook
from starfinder.dataset import FOV, Dataset
from starfinder.io import load_round, load_volume, save_volume
from starfinder.preprocessing import (
    match_histogram,
    normalize_min_max,
    correct_background,
    filter_tophat,
)
from starfinder.registration import estimate_transform, apply_transform
from starfinder.spot_finding import find_spots
from starfinder.preprocessing import project_image

try:
    __version__ = _package_version("starfinder")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "ImageMetadata",
    # Dataset/FOV orchestration
    "Dataset",
    "FOV",
    # I/O functions
    "load_volume",
    "load_round",
    "save_volume",
    # Registration module and functions
    "registration",
    "estimate_transform",
    "apply_transform",
    # Spot finding
    "spot_finding",
    "find_spots",
    # Readout
    "readout",
    "extract_intensities",
    "decode_reads",
    "load_codebook",
    "filter_reads",
    # Preprocessing
    "preprocessing",
    "normalize_min_max",
    "match_histogram",
    "correct_background",
    "filter_tophat",
    # Utilities
    "project_image",
    # Package metadata
    "__version__",
]
