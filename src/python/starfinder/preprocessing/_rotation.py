"""Rotation of a loaded ZYX(C) volume in YX, the first per-FOV preprocessing of a round."""
from dataclasses import asdict

import numpy as np

from starfinder.image import _validate_image


def _rotated(volume, source, angle):
    """(image, metadata, diagnostics) of a ZYX(C) volume rotated by angle degrees in YX.

    Exact 90-degree multiples use np.rot90 and a contiguous copy; other angles
    scipy.ndimage.rotate with bilinear interpolation, keeping the shape and dtype.
    """
    vol = _validate_image(volume)
    metadata = source.rotated(vol.shape[:3], angle, frame_id=f"{source.frame_id}/rotate:{angle}")
    k_90 = round(angle / 90)
    if abs(angle - k_90 * 90) < 1e-6:
        # np.rot90 and imrotate share sign convention: k=1 is CCW, k=-1 is CW
        image = np.ascontiguousarray(np.rot90(vol, k=k_90, axes=(1, 2)))
    else:
        from scipy.ndimage import rotate as ndimage_rotate
        image = ndimage_rotate(vol, angle, axes=(1, 2), reshape=False, order=1).astype(vol.dtype)
    return image, metadata, {
        "source_metadata": asdict(source), "source_shape_zyx": vol.shape[:3],
        "output_shape_zyx": image.shape[:3], "angle_degrees": angle,
        "mapping": "source = R @ (output - output_center) + source_center",
    }
