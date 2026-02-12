"""Morphological preprocessing for STARfinder.

Ports MATLAB MorphologicalReconstruction.m and STARMapDataset.Tophat.
"""

import numpy as np
from skimage.morphology import (
    black_tophat,
    disk,
    erosion,
    reconstruction,
    white_tophat,
)


def tophat_filter(volume: np.ndarray, radius: int = 3) -> np.ndarray:
    """White tophat filtering per Z-slice.

    Removes large-scale background by subtracting the morphological opening
    from the original. Matches MATLAB ``imtophat(slice, strel('disk', r))``.

    Parameters
    ----------
    volume : np.ndarray
        Input volume with shape (Z, Y, X) or (Z, Y, X, C).
    radius : int
        Radius of the disk structuring element (default 3).

    Returns
    -------
    np.ndarray
        Filtered volume, dtype uint8, same shape as input.
    """
    is_3d = volume.ndim == 3
    if is_3d:
        volume = volume[..., np.newaxis]

    se = disk(radius)
    result = np.empty_like(volume, dtype=np.uint8)

    for c in range(volume.shape[3]):
        for z in range(volume.shape[0]):
            result[z, :, :, c] = white_tophat(volume[z, :, :, c], se)

    if is_3d:
        result = result[..., 0]
    return result


def morphological_reconstruction(volume: np.ndarray, radius: int = 3) -> np.ndarray:
    """Background removal via morphological reconstruction.

    Per Z-slice algorithm (matching MATLAB MorphologicalReconstruction.m):
      1. marker = erosion(slice, disk(radius))
      2. obr = reconstruction(marker, slice, method='dilation')
      3. subtracted = slice - obr
      4. result = subtracted + white_tophat(subtracted, se) - black_tophat(subtracted, se)

    Parameters
    ----------
    volume : np.ndarray
        Input volume with shape (Z, Y, X) or (Z, Y, X, C).
    radius : int
        Radius of the disk structuring element (default 3).

    Returns
    -------
    np.ndarray
        Reconstructed volume, dtype uint8, same shape as input.
    """
    is_3d = volume.ndim == 3
    if is_3d:
        volume = volume[..., np.newaxis]

    se = disk(radius)
    result = np.empty_like(volume, dtype=np.uint8)

    for c in range(volume.shape[3]):
        for z in range(volume.shape[0]):
            slc = volume[z, :, :, c]
            marker = erosion(slc, se)
            obr = reconstruction(marker, slc, method="dilation")
            subtracted = slc.astype(np.int16) - obr.astype(np.int16)
            subtracted = np.clip(subtracted, 0, 255).astype(np.uint8)
            # Enhance: add tophat, subtract bothat (use int16 to avoid overflow)
            top = white_tophat(subtracted, se).astype(np.int16)
            bot = black_tophat(subtracted, se).astype(np.int16)
            enhanced = subtracted.astype(np.int16) + top - bot
            result[z, :, :, c] = np.clip(enhanced, 0, 255).astype(np.uint8)

    if is_3d:
        result = result[..., 0]
    return result
