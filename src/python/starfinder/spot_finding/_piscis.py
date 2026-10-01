"""Piscis: plane mode for Z=1, stack mode for Z>1 (docs/spot-finding-algorithms.md, "Piscis").

Piscis 1.1.0 is loaded through the hash-checked absolute-path wrapper:
Piscis(model_name=<verified folder>/<model>), which the library resolves to
<model>.pt in that folder (MODELS_DIR / '<absolute path>.pt' is the absolute
path), so it never reaches ~/.piscis/models, its JAX conversion or a download.
"""
import copy
from dataclasses import replace

import numpy as np

from ._learned import absolute, cached_model, channels_of, frame, is_constant, model_record, table, verified_folder
from ._weights import known_weights

COLUMNS = ('z', 'y', 'x', 'channel', 'peak_intensity')
# Piscis tiles axes longer than the tile with this overlap fraction (piscis/core.py, _Piscis._predict).
_OVERLAP = 0.1


def tiling(shape_yx, tile_size):
    """The tile size, overlap (pixels) and keep-boundaries per lateral axis of a Piscis run.

    Piscis cuts each overlap at one keep-boundary b - 0.5 (deeptile's border
    indices, stitch_coords); a prediction is kept by the tile whose kept range
    holds it, and seam repeats are not merged.
    """
    from deeptile.core.utils import calculate_indices_1d, calculate_overlap_size
    record = {'tile_size': list(tile_size), 'overlap': [], 'keep_boundaries': {}}
    for axis, n, size in zip('yx', shape_yx, tile_size):
        overlap = _OVERLAP if n > size else 0
        _, _, borders = calculate_indices_1d(n, size, overlap)
        record['overlap'].append(int(calculate_overlap_size(size, overlap)))
        record['keep_boundaries'][axis] = [float(b) - 0.5 for b in borders[1:-1]]
    return record


def piscis(image, config, context):
    """Spots of each channel in context.channels with the named model; rows by channel, in Piscis's order.

    Z=1 runs plane mode (z=0); Z>1 runs stack mode, whose z is an integer.
    Returns the table (z, y, x, channel, peak_intensity) and the thresholds,
    the effective configs (input_size resolved), the model record and the
    tiling (tile size, overlap and keep-boundaries per axis).
    """
    entry = known_weights("piscis", config.model)
    folder = verified_folder("piscis", config.model)
    from piscis import Piscis
    from piscis.models.spots import round_input_size
    path = folder / entry.files[0].path
    loaded = cached_model("piscis", config.model,
                          lambda: Piscis(model_name=absolute(path.with_suffix(""), "Piscis"), device=context.device))
    detector = copy.copy(loaded)   # input_size is a per-call setting; the network is shared
    if config.input_size is not None:
        detector.input_size = round_input_size((config.input_size, config.input_size))
    plane = image.shape[0] == 1
    tile_size = tuple(int(round(n / config.scale)) for n in detector.input_size)
    effective = replace(config, input_size=int(detector.input_size[0]))
    frames = []
    for c, channel in channels_of(image, context):
        if is_constant(channel):
            continue
        coords = np.asarray(detector.predict(channel[0] if plane else channel, stack=not plane, scale=config.scale,
                                             threshold=config.threshold, min_distance=config.min_distance),
                            dtype=np.float64).reshape(-1, 2 if plane else 3)
        zyx = np.column_stack((np.zeros(len(coords)), coords)) if plane else coords
        frames.append(frame(channel, c, zyx, COLUMNS))
    return table(frames, COLUMNS), {
        'thresholds': tuple(float(config.threshold) for _ in context.channels),
        'effective': tuple(effective for _ in context.channels),
        'model': model_record("piscis", config.model, folder),
        'geometry': {'mode': 'plane' if plane else 'stack', **tiling(image.shape[1:3], tile_size)},
        'measurements': {'peak_intensity': 'original pixel intensity at the coordinate rounded half to even'}}
