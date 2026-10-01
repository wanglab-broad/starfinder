"""Spotiflow: native 3D with a 3D model, a YX plane with a 2D model (docs/spot-finding-algorithms.md, "Spotiflow").

Spotiflow 0.6.5 is loaded with Spotiflow.from_folder(<verified folder>,
map_location="cpu") from the Starfinder weights cache, never through
from_pretrained or ~/.spotiflow.
"""
from dataclasses import replace

from starfinder.image import IncompatibleGeometryError

from ._learned import absolute, cached_model, channels_of, frame, is_constant, model_record, table, verified_folder
from ._weights import KNOWN_WEIGHTS, known_weights

COLUMNS = ('z', 'y', 'x', 'channel', 'peak_intensity', 'probability')


def _models(dimensionality):
    return ", ".join(m for (method, m), e in KNOWN_WEIGHTS.items() if method == "spotiflow"
                     and e.dimensionality == dimensionality)


def check_geometry(entry, shape):
    """Raise IncompatibleGeometryError where Spotiflow would return nothing or fail (W-266 probes).

    A 3D model needs Z>1 and its minimum shape (Z >= 7, Y and X >= 8); a 2D
    model needs Z=1 and Y and X of at least its minimum (6).
    """
    model, minimum = entry.model, entry.min_shape_zyx
    if entry.dimensionality == "3D":
        if shape[0] == 1:
            raise IncompatibleGeometryError(
                f"spotiflow model {model!r} is a 3D model and returns nothing on a Z=1 image; a Z=1 image needs a "
                f"2D model ({_models('2D')})")
        if any(n < m for n, m in zip(shape, minimum)):
            raise IncompatibleGeometryError(
                f"spotiflow model {model!r} needs a ZYX shape of at least {minimum}, not {tuple(shape)}; Spotiflow "
                "would return nothing")
    else:
        if shape[0] > 1:
            raise IncompatibleGeometryError(
                f"spotiflow model {model!r} is a 2D model and detects Z=1 images only; a volume needs a 3D model "
                f"({_models('3D')})")
        if any(n < m for n, m in zip(shape[1:], minimum[1:])):
            raise IncompatibleGeometryError(
                f"spotiflow model {model!r} needs Y and X of at least {minimum[1:]}, not {tuple(shape[1:])}; "
                "Spotiflow would return nothing")


def spotiflow(image, config, context):
    """Spots of each channel in context.channels with the named model; rows by channel, in Spotiflow's order.

    Returns the table (z, y, x, channel, peak_intensity, probability) and the
    thresholds, the effective configs (prob_thresh, subpix and n_tiles
    resolved), the model record and the tiling (n_tiles).
    """
    entry = known_weights("spotiflow", config.model)
    check_geometry(entry, image.shape[:3])
    folder = verified_folder("spotiflow", config.model)
    import torch
    from spotiflow.model import Spotiflow
    from spotiflow.model.spotiflow import infer_n_tiles
    model = cached_model("spotiflow", config.model,
                         lambda: Spotiflow.from_folder(absolute(folder, "Spotiflow.from_folder"),
                                                       map_location=context.device))
    plane = image.shape[0] == 1
    n_tiles = config.n_tiles or tuple(int(n) for n in infer_n_tiles(
        image.shape[1:3] if plane else image.shape[:3], None, device=torch.device(context.device)))
    effective = replace(
        config, prob_thresh=float(model._prob_thresh[0]) if config.prob_thresh is None else config.prob_thresh,
        subpix=bool(model.config.compute_flow) if config.subpix is None else config.subpix, n_tiles=n_tiles)
    frames = []
    for c, channel in channels_of(image, context):
        if is_constant(channel):
            continue
        points, details = model.predict(
            channel[0] if plane else channel, prob_thresh=config.prob_thresh, n_tiles=n_tiles,
            min_distance=config.min_distance, exclude_border=config.exclude_border, scale=config.scale,
            subpix=config.subpix, peak_mode="fast", normalizer="auto", verbose=False, device=context.device)
        zyx = points if not plane else [(0.0, y, x) for y, x in points]
        frames.append(frame(channel, c, zyx, COLUMNS, probability=details.prob))
    return table(frames, COLUMNS), {
        'thresholds': tuple(effective.prob_thresh for _ in context.channels),
        'effective': tuple(effective for _ in context.channels),
        'model': model_record("spotiflow", config.model, folder),
        'geometry': {'n_tiles': n_tiles},
        'measurements': {'peak_intensity': 'original pixel intensity at the coordinate rounded half to even',
                         'probability': "Spotiflow's spot-wise heatmap probability (details.prob)"}}
