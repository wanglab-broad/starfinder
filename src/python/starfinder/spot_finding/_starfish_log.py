"""The native Starfish LoG: starfish BlobDetector (blob_log, is_volume=True, no reference image) per channel.

Follows starfish/core/spots/FindSpots/blob.py at 1fb00cbc (identical at 0.4.0):
blob_log on the dtype-scaled image, then starfish's four post-processing steps.
docs/spot-finding-algorithms.md ("Starfish LoG") is the specification.
"""
import numpy as np
import pandas as pd
from skimage import img_as_float32
from skimage.feature import blob_log

from starfinder.image import IncompatibleGeometryError

# W-266 measured a peak minus pre-inference RSS of 10.0 to 10.4 bytes per voxel and scale
# (float32 input, about 2.5 copies of the float32 scale-space cube); the estimate uses 10.4.
SCALE_SPACE_BYTES_PER_VOXEL_AND_SIGMA = 10.4

_COLUMNS = {'z': 'float64', 'y': 'float64', 'x': 'float64', 'channel': 'int64', 'peak_intensity': 'float64',
            'radius': 'float64'}


def check_range(channel):
    """Raise ValueError unless the dtype scaling puts the channel in [0, 1] (floats in [0, 1], no negative integers)."""
    low = channel.min()
    if low < 0 or (channel.dtype.kind == 'f' and channel.max() > 1):
        raise ValueError("starfish_log needs intensities in [0, 1]: integer images are scaled by their dtype "
                         f"maximum, and a float image must already lie in [0, 1] (this channel spans "
                         f"[{float(low)!r}, {float(channel.max())!r}])")


def scaled(channel):
    """The channel as starfish's ImageStack holds it: float32 in [0, 1].

    Integer images are scaled by their dtype maximum (img_as_float32, which
    multiplies by the float32 reciprocal of the maximum); float images are
    cast to float32 and must already lie in [0, 1].
    """
    check_range(channel)
    return img_as_float32(channel)


def _blobs(image, config):
    """blob_log, then starfish's steps: truncated integer coordinates and radius round(sigma * sqrt(ndim))."""
    blobs = blob_log(image, min_sigma=config.min_sigma, max_sigma=config.max_sigma, num_sigma=config.num_sigma,
                     threshold=config.threshold, overlap=config.overlap, exclude_border=config.exclude_border)
    ndim = image.ndim
    coords = blobs[:, :ndim].astype(int)
    sigmas = blobs[:, ndim:]
    sigma = sigmas[:, 0] if sigmas.shape[1] == 1 else sigmas.mean(axis=1)
    return coords, np.round(sigma * np.sqrt(ndim))


def starfish_log(image, config, context):
    """Blobs of each channel in context.channels; rows by channel, in blob_log's order.

    A single Z plane is squeezed to 2D (z=0). Coordinates are blob_log's,
    truncated to integers; peak_intensity is the original pixel value there and
    radius is round(sigma * sqrt(ndim)), with the mean of per-axis sigmas. A
    constant channel yields no rows without running blob_log. Returns the
    table and the thresholds and the scale-space memory estimate
    (10.4 bytes x num_sigma x voxels of one channel, recorded, not enforced).
    """
    plane = image.shape[0] == 1
    if plane and (isinstance(config.min_sigma, tuple) or isinstance(config.max_sigma, tuple)):
        raise IncompatibleGeometryError("starfish_log detects a Z=1 image as a YX plane, so min_sigma and max_sigma "
                                        "must be numbers, not ZYX 3-tuples")
    voxels = int(np.prod(image.shape[:3]))
    channels = [image[..., c] if image.ndim == 4 else image for c in context.channels]
    for channel in channels:
        check_range(channel)
    frames = []
    for c, channel in zip(context.channels, channels):
        if channel.min() == channel.max():
            continue
        values = scaled(channel)
        coords, radius = _blobs(values[0] if plane else values, config)
        if plane:
            z, (y, x) = np.zeros(len(coords), dtype=int), coords.T
        else:
            z, y, x = coords.T
        frames.append(pd.DataFrame({'z': z, 'y': y, 'x': x, 'channel': c,
                                    'peak_intensity': channel[z, y, x], 'radius': radius}))
    table = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(_COLUMNS))).astype(_COLUMNS)
    geometry = {'voxels': voxels, 'num_sigma': config.num_sigma,
                'bytes_per_voxel_and_sigma': SCALE_SPACE_BYTES_PER_VOXEL_AND_SIGMA,
                'scale_space_bytes_estimate': SCALE_SPACE_BYTES_PER_VOXEL_AND_SIGMA * config.num_sigma * voxels}
    return table, {'thresholds': tuple(float(config.threshold) for _ in context.channels), 'geometry': geometry,
                   'measurements': {'peak_intensity': 'original pixel intensity at the truncated blob position',
                                    'radius': 'starfish blob radius round(sigma * sqrt(ndim)) in voxels'}}


def starfish_view(spots, channel):
    """One channel's rows of a starfish_log table in starfish's columns, for the parity test.

    spots holds the rows of one channel; channel is that channel's ZYX image.
    intensity is the scaled pixel value (float32), z, y, x are int64, spot_id
    is the row index; an empty table is starfish's typed empty frame.
    """
    if not len(spots):
        return pd.DataFrame(np.array([], dtype=[('x', int), ('y', int), ('z', int), ('radius', float),
                                                ('intensity', object), ('spot_id', object)]))
    z, y, x = (spots[a].to_numpy().astype(int) for a in 'zyx')
    frame = pd.DataFrame({'intensity': scaled(channel)[z, y, x], 'z': z, 'y': y, 'x': x,
                          'radius': spots['radius'].to_numpy()})
    frame['spot_id'] = np.arange(len(frame))
    return frame
