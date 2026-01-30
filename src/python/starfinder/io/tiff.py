"""TIFF image I/O using bioio backend."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def load_multipage_tiff(
    path: Path | str,
    convert_uint8: bool = True,
) -> np.ndarray:
    """
    Load a multi-page TIFF file.

    Automatically detects OME-TIFF and ImageJ hyperstacks and uses their
    dimension metadata for correct interpretation. Plain TIFFs are read
    with raw array shape.

    Args:
        path: Path to TIFF file.
        convert_uint8: If True (default), convert to uint8.

    Returns:
        Array with shape (Z, Y, X).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TIFF file not found: {path}")

    # Check for dimension metadata (OME-TIFF or ImageJ hyperstack)
    with tifffile.TiffFile(path) as tif:
        has_metadata = tif.is_ome or tif.is_imagej

        if has_metadata:
            # Use bioio for metadata-aware loading
            from bioio import BioImage
            from bioio_tifffile import Reader as TiffReader

            img = BioImage(path, reader=TiffReader)
            data = img.get_image_data("ZYX", T=0, C=0)
            logger.debug(f"Loaded {path} with bioio (OME={tif.is_ome}, ImageJ={tif.is_imagej})")
        else:
            # Plain TIFF - read raw array
            data = tif.asarray()
            logger.debug(f"Loaded {path} with tifffile (plain TIFF)")

    # Ensure 3D array (Z, Y, X)
    if data.ndim == 2:
        # Single slice: add Z dimension
        data = data[np.newaxis, ...]

    if convert_uint8:
        data = _to_uint8(data)

    return data


def _to_uint8(data: np.ndarray) -> np.ndarray:
    """Convert array to uint8 with proper scaling."""
    if data.dtype == np.uint8:
        return data

    # Get min/max for scaling
    data_min = float(data.min())
    data_max = float(data.max())

    if data_max > data_min:
        # Scale to 0-255 range
        scaled = (data.astype(np.float32) - data_min) / (data_max - data_min) * 255.0
        return scaled.astype(np.uint8)
    else:
        # Constant image
        return np.zeros(data.shape, dtype=np.uint8)


def load_image_stacks(
    round_dir: Path | str,
    channel_order: list[str],
    subdir: str = "",
    convert_uint8: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Load multiple channel TIFFs from a directory.

    Args:
        round_dir: Directory containing channel TIFF files.
        channel_order: List of channel patterns, e.g., ["ch00", "ch01"].
        subdir: Optional subdirectory within round_dir.
        convert_uint8: If True (default), convert to uint8.

    Returns:
        Tuple of (image array with shape (Z, Y, X, C), metadata dict).

    Raises:
        FileNotFoundError: If directory does not exist.
        ValueError: If no files match a channel pattern.

    Notes:
        If channels have different sizes, crops to minimum and logs warning.
        Metadata includes: shape, dtype, original_shapes, cropped (bool).
    """
    round_dir = Path(round_dir)
    search_dir = round_dir / subdir if subdir else round_dir

    if not search_dir.exists():
        raise FileNotFoundError(f"Directory not found: {search_dir}")

    # Find and load each channel
    channel_images: list[np.ndarray] = []
    original_shapes: list[tuple[int, ...]] = []

    for channel in channel_order:
        # Find file matching channel pattern
        matches = list(search_dir.glob(f"*{channel}*.tif"))
        if not matches:
            raise ValueError(f"No TIFF file found matching channel pattern: {channel}")
        if len(matches) > 1:
            logger.warning(f"Multiple files match '{channel}', using first: {matches[0]}")

        # Load the channel (without uint8 conversion yet)
        img = load_multipage_tiff(matches[0], convert_uint8=False)
        channel_images.append(img)
        original_shapes.append(img.shape)

    # Find minimum dimensions
    min_z = min(img.shape[0] for img in channel_images)
    min_y = min(img.shape[1] for img in channel_images)
    min_x = min(img.shape[2] for img in channel_images)

    # Check for size mismatch
    cropped = False
    for i, (img, shape) in enumerate(zip(channel_images, original_shapes)):
        if shape != (min_z, min_y, min_x):
            cropped = True
            break

    if cropped:
        warnings.warn(
            f"Channel size mismatch detected. Cropping to minimum dimensions "
            f"({min_z}, {min_y}, {min_x}). Original shapes: {original_shapes}",
            UserWarning,
        )

    # Crop all channels to minimum size and stack
    cropped_images = [img[:min_z, :min_y, :min_x] for img in channel_images]
    stacked = np.stack(cropped_images, axis=-1)  # (Z, Y, X, C)

    # Convert to uint8 if requested
    if convert_uint8:
        stacked = _to_uint8(stacked)

    metadata = {
        "shape": stacked.shape,
        "dtype": str(stacked.dtype),
        "original_shapes": original_shapes,
        "cropped": cropped,
    }

    return stacked, metadata


def save_stack(
    image: np.ndarray,
    path: Path | str,
    compress: bool = False,
) -> None:
    """
    Save a 3D or 4D array as a multi-page TIFF.

    Args:
        image: Array with shape (Z, Y, X) or (Z, Y, X, C).
        path: Output path.
        compress: If True, use compression.

    Notes:
        Overwrites existing file if present.
    """
    path = Path(path)

    # Remove existing file if present
    if path.exists():
        path.unlink()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    compression = "zlib" if compress else None
    tifffile.imwrite(path, image, compression=compression)
