"""STARfinder: Spatial transcriptomics data processing pipeline."""

from starfinder.io import load_multipage_tiff, load_image_stacks, save_stack

__version__ = "0.1.0"

__all__ = [
    "load_multipage_tiff",
    "load_image_stacks",
    "save_stack",
    "__version__",
]
