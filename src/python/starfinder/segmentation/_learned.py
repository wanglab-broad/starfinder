"""The learned segmentation methods: StarDist 2D/3D and Cellpose, with their configs.

Both backends are imported inside the method functions only, after the stage
wrapper has imported the dependencies (check 5) and resolved and hashed the model
files (check 6); nothing here downloads. See docs/segmentation-contract.md
("Registered methods", "Block-wise prediction", "Models", "Device and environments")
and docs/segmentation-algorithms.md ("stardist", "cellpose").
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np

from ._inputs import normalize_percentiles
from ._models import KNOWN_MODELS

BLOCK_FIELDS = ("block_size", "min_overlap", "context")


def _finite(value, name, *, low=None, high=None, low_open=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    if low is not None and (value < low or (low_open and value == low)):
        raise ValueError(f"{name} must be {'>' if low_open else '>='} {low}; got {value!r}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be <= {high}; got {value!r}")
    return float(value)


def _integer(value, name, low):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}; got {value!r}")
    return int(value)


def _check_model(config):
    """Exactly one of model and model_path; a known model name; model_sha256 only with a path."""
    if (config.model is None) == (config.model_path is None):
        raise ValueError(f"{config.method} needs exactly one of model (a known name) and model_path")
    if config.model is not None:
        known = sorted(m for k, m in KNOWN_MODELS if k == config.method)
        if config.model not in known:
            raise ValueError(f"unknown {config.method} model {config.model!r}; known models: {known} "
                             "(a user-trained model is given by model_path)")
        if config.model_sha256 is not None:
            raise ValueError("model_sha256 applies to model_path; a known model is checked against KNOWN_MODELS")
    elif not isinstance(config.model_path, (str, os.PathLike)):
        raise TypeError("model_path must be a path")
    if config.model_sha256 is not None and not isinstance(config.model_sha256, Mapping):
        raise TypeError("model_sha256 must map file names to SHA-256 values")


def _check_percentiles(value):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"normalize_percentiles must be a (low, high) tuple; got {value!r}")
    low, high = (_finite(v, "normalize_percentiles", low=0.0, high=100.0) for v in value)
    if not low < high:
        raise ValueError(f"normalize_percentiles must have low < high; got {value!r}")


def _per_axis(value, n):
    """A scalar for every axis, or a tuple with one value per axis (n of them)."""
    if isinstance(value, tuple):
        if len(value) != n:
            raise ValueError(f"a {n}D model needs {n} values per block field; got {value!r}")
        return tuple(int(v) for v in value)
    return (int(value),) * n


def _check_blocks(config):
    """The block fields: all None, or all given with min_overlap + 2 × context < block_size on every axis."""
    values = [getattr(config, name) for name in BLOCK_FIELDS]
    given = [v is not None for v in values]
    if not any(given):
        return
    if not all(given):
        raise ValueError("block_size, min_overlap and context are given together (block-wise prediction) or not at "
                         f"all; got {dict(zip(BLOCK_FIELDS, values))}")
    lengths = set()
    for name, value in zip(BLOCK_FIELDS, values):
        items = value if isinstance(value, tuple) else (value,)
        if isinstance(value, tuple):
            if len(value) not in (2, 3):
                raise ValueError(f"{name} must be one integer or one per axis (Y, X or Z, Y, X); got {value!r}")
            lengths.add(len(value))
        for item in items:
            _integer(item, name, 1 if name == "block_size" else 0)
    if len(lengths) > 1:
        raise ValueError("the tuple-valued block fields must have the same number of axes")
    n = lengths.pop() if lengths else 1
    block, overlap, context = (_per_axis(v, n) for v in values)
    for axis, (b, o, c) in enumerate(zip(block, overlap, context)):
        if not o + 2 * c < b:
            raise ValueError(f"block-wise prediction needs min_overlap + 2 × context < block_size on every axis; "
                             f"axis {axis}: {o} + 2 × {c} >= {b}")
    if any(s != 1 for s in _scale_zyx(config.scale)):
        raise ValueError("block fields together with a scale other than 1 are not supported (W-309): nothing has "
                         "tested that combination")


def _scale_zyx(scale) -> tuple[float, float, float]:
    """A number scales Y and X (Z factor 1); a 3-tuple is per axis ZYX."""
    if isinstance(scale, tuple):
        return tuple(float(s) for s in scale)
    return (1.0, float(scale), float(scale))


@dataclass(frozen=True, kw_only=True)
class StarDistConfig:
    """StarDist 2D or 3D on one ``nuclear`` or ``composite`` channel (docs/segmentation-algorithms.md, "stardist").

    model names a known pretrained model (``2D_versatile_fluo``) in the weights
    cache, model_path the folder of a user-trained model (``config.json``,
    ``thresholds.json``, ``weights_best.h5``, such as ``3D_spleen``); exactly
    one is given, and model_sha256 optionally maps a path's file names to their
    expected SHA-256. The model's ``n_dim`` decides 2D (Z=1) or 3D (Z>1).

    scale is required (no default; it follows the model's training data): a
    number scales Y and X with Z factor 1, a 3-tuple is per axis ZYX; it is
    passed to ``predict_instances(scale=…)`` and the labels come back on the
    input grid. prob_thresh and nms_thresh are None for the model's stored
    ``thresholds.json`` (``threshold_source`` ``stored``); a number is a
    sensitivity override (``override``). normalize_percentiles is csbdeep's
    percentile normalization over the whole image. n_tiles is None for the
    current script's tiling (1×4×4 for a volume, 2×2 for a plane) or a tuple
    passed as given (ZYX; Y, X for a plane).

    block_size, min_overlap and context (voxels of the input grid, one integer
    for every axis or one per axis) are all None for whole-image prediction, or
    all given for block-wise prediction with ``predict_instances_big``; on every
    axis min_overlap + 2 × context < block_size, and scale must be 1. The
    library rounds each value up to a multiple of the model's grid; the record
    holds both. Objects must be smaller than min_overlap on each axis. There is
    no default block size, and the method never switches to blocks on its own.

    Raises
    ------
    ValueError
        Neither or both of model and model_path, an unknown model, a scale,
        threshold, percentile or tile count out of range, block fields given
        partly, violating the overlap condition or combined with a scale other
        than 1.
    """

    scale: float | tuple[float, float, float]
    model: str | None = None
    model_path: str | None = None
    model_sha256: Mapping[str, str] | None = None
    prob_thresh: float | None = None
    nms_thresh: float | None = None
    normalize_percentiles: tuple[float, float] = (1.0, 99.8)
    n_tiles: tuple[int, ...] | None = None
    block_size: int | tuple[int, ...] | None = None
    min_overlap: int | tuple[int, ...] | None = None
    context: int | tuple[int, ...] | None = None
    method: str = field(default="stardist", init=False)

    def __post_init__(self):
        _check_model(self)
        if isinstance(self.scale, tuple):
            if len(self.scale) != 3:
                raise ValueError(f"scale must be a number (Y and X) or a ZYX 3-tuple; got {self.scale!r}")
            for s in self.scale:
                _finite(s, "scale", low=0.0, low_open=True)
        else:
            _finite(self.scale, "scale", low=0.0, low_open=True)
        for name in ("prob_thresh", "nms_thresh"):
            if getattr(self, name) is not None:
                _finite(getattr(self, name), name, low=0.0, high=1.0)
        _check_percentiles(self.normalize_percentiles)
        if self.n_tiles is not None:
            if not isinstance(self.n_tiles, tuple) or len(self.n_tiles) not in (2, 3):
                raise ValueError(f"n_tiles must be None or a tuple of 2 (Y, X) or 3 (Z, Y, X) integers; "
                                 f"got {self.n_tiles!r}")
            for n in self.n_tiles:
                _integer(n, "n_tiles", 1)
        _check_blocks(self)


@dataclass(frozen=True, kw_only=True)
class CellposeConfig:
    """Cellpose on a ``cytoplasm`` or ``nuclear`` channel, optionally with ``nuclear`` beside ``cytoplasm``.

    model names a known model (``cpsam_v2``) in the weights cache, model_path a
    model file; exactly one is given, and model_sha256 optionally maps the
    file's name to its expected SHA-256. Cellpose itself would silently load
    (and download) its default model for a missing path; resolution raises
    first.

    diameter (pixels) is required and has no default: the image is rescaled
    by 30/diameter; ``None`` must be given explicitly, means no rescale and is
    refused with ``do_3d=True``. do_3d runs 3D mode on Z>1 (a Z>1 input with
    False is rejected, as is Z=1 with True) and needs anisotropy, the Z spacing
    over the Y, X spacing. flow_threshold (0.4), cellprob_threshold (0.0),
    tile_overlap (0.1, within the [0.05, 0.5] Cellpose keeps), bfloat16
    (True) and min_size (15 pixels) are the library defaults, recorded.
    normalize_percentiles is passed per channel as a new mapping on every
    call, so Cellpose's module-level default is never written.

    Raises
    ------
    ValueError
        Neither or both of model and model_path, an unknown model, a diameter,
        anisotropy, threshold, overlap, size or percentile out of range, a
        missing diameter or anisotropy in 3D mode, or anisotropy without it.
    TypeError
        diameter is not given.
    """

    diameter: float | None
    model: str | None = None
    model_path: str | None = None
    model_sha256: Mapping[str, str] | None = None
    do_3d: bool = False
    anisotropy: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    tile_overlap: float = 0.1
    bfloat16: bool = True
    min_size: int = 15
    normalize_percentiles: tuple[float, float] = (1.0, 99.0)
    method: str = field(default="cellpose", init=False)

    def __post_init__(self):
        _check_model(self)
        if self.diameter is not None:
            _finite(self.diameter, "diameter", low=0.0, low_open=True)
        if not isinstance(self.do_3d, bool) or not isinstance(self.bfloat16, bool):
            raise ValueError("do_3d and bfloat16 must be Boolean")
        if self.do_3d:
            if self.diameter is None:
                raise ValueError("diameter None (no rescale) is refused with do_3d=True (W-306)")
            if self.anisotropy is None:
                raise ValueError("do_3d=True needs anisotropy (Z spacing over Y, X spacing)")
        elif self.anisotropy is not None:
            raise ValueError("anisotropy applies only with do_3d=True")
        if self.anisotropy is not None:
            _finite(self.anisotropy, "anisotropy", low=0.0, low_open=True)
        _finite(self.flow_threshold, "flow_threshold", low=0.0)
        _finite(self.cellprob_threshold, "cellprob_threshold")
        _finite(self.tile_overlap, "tile_overlap", low=0.05, high=0.5)
        _integer(self.min_size, "min_size", 0)
        _check_percentiles(self.normalize_percentiles)


# --- Execution entries -------------------------------------------------------------------------------

def _driver_version():
    """The NVIDIA driver version from nvidia-smi, or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.splitlines()[0].strip() if out.strip() else None


def _tensorflow_device(tf, device):
    """Enable memory growth on "cuda" before the first model call; return the TensorFlow framework entry."""
    gpus = tf.config.list_physical_devices("GPU")
    entry = {"name": "tensorflow", "version": str(tf.__version__),
             "cuda": tf.sysconfig.get_build_info().get("cuda_version"),
             "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
             "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
             "memory_growth": None, "gpu": None, "driver": None}
    if device == "cuda":
        if not gpus:
            raise ValueError("device 'cuda' was requested, but TensorFlow sees no GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:  # already initialized in this process; the recorded value says what holds
                pass
        entry.update(memory_growth=all(tf.config.experimental.get_memory_growth(g) for g in gpus),
                     gpu=tf.config.experimental.get_device_details(gpus[0]).get("device_name"),
                     driver=_driver_version())
    return entry


def _torch_device(torch, device):
    """The torch device of a call and its framework entry; "cuda" needs a visible GPU."""
    entry = {"name": "torch", "version": str(torch.__version__), "cuda": torch.version.cuda,
             "intra_op_threads": torch.get_num_threads(), "inter_op_threads": torch.get_num_interop_threads(),
             "gpu": None, "driver": None}
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device 'cuda' was requested, but torch sees no GPU")
        entry.update(gpu=torch.cuda.get_device_name(0), driver=_driver_version())
    return torch.device(device), entry


@contextmanager
def _offline():
    """HF_HUB_OFFLINE=1 for the duration of a Cellpose call, as the contract requires; restored afterwards."""
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


# --- The method functions ----------------------------------------------------------------------------

def _stardist_blocks(config, n_dim, grid, shape):
    """Requested and effective (rounded up to the model's grid, as the library rounds) block values."""
    requested = {name: list(_per_axis(getattr(config, name), n_dim)) for name in BLOCK_FIELDS}
    effective = {name: [math.ceil(v / g) * g for v, g in zip(values, grid)] for name, values in requested.items()}
    for axis, (b, o, c, n) in enumerate(zip(effective["block_size"], effective["min_overlap"], effective["context"],
                                            shape)):
        if not o + 2 * c < b:
            raise ValueError(f"after rounding to the model's grid {tuple(grid)}, axis {axis} has min_overlap {o} + "
                             f"2 × context {c} >= block_size {b}")
        if b > n:
            raise ValueError(f"block_size {b} (after rounding to the model's grid) exceeds the input's size {n} on "
                             f"axis {axis}")
    return {"requested": requested, "effective": effective, "grid": list(grid)}


def _stardist(image, config: StarDistConfig, context) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize, then ``predict_instances`` or block-wise ``predict_instances_big`` (docs/segmentation-algorithms.md).

    The one channel is normalized by csbdeep's percentiles over the whole image
    (normalize_percentiles, float32, unclipped); a Z=1 input to a 2D model is
    squeezed to YX (the wrapper restores the axis). The model folder is loaded
    with ``StarDist2D`` or ``StarDist3D(None, name=…, basedir=…)``, which reads
    the folder and downloads nothing. A scale of 1 on every axis is passed as
    None, as the current script calls the library.
    """
    if image.shape[-1] != 1:
        raise ValueError(f"stardist segments one channel (nuclear or composite); the input has roles {context.roles}")
    folder = context.model
    model_config = json.loads((folder / "config.json").read_text())
    n_dim, grid = int(model_config["n_dim"]), tuple(int(g) for g in model_config["grid"])
    scale_zyx = _scale_zyx(config.scale)
    if n_dim == 2 and scale_zyx[0] != 1:
        raise ValueError(f"a 2D model scales Y and X only; the Z factor of scale {config.scale!r} must be 1")
    library_scale = None if all(s == 1 for s in scale_zyx) else (scale_zyx if n_dim == 3 else scale_zyx[1:])
    n_tiles = config.n_tiles if config.n_tiles is not None else ((1, 4, 4) if n_dim == 3 else (2, 2))
    if len(n_tiles) != n_dim:
        raise ValueError(f"n_tiles {n_tiles} needs {n_dim} values for a {n_dim}D model")
    shape = image.shape[:3] if n_dim == 3 else image.shape[1:3]
    blocks = None if config.block_size is None else _stardist_blocks(config, n_dim, grid, shape)
    p_low, p_high = config.normalize_percentiles
    normalized, norm_record = normalize_percentiles(image[..., 0], p_low=p_low, p_high=p_high)
    x, axes = (normalized, "ZYX") if n_dim == 3 else (normalized[0], "YX")

    import tensorflow as tf
    from stardist.models import StarDist2D, StarDist3D

    framework = _tensorflow_device(tf, context.device)
    with tf.device("/GPU:0" if context.device == "cuda" else "/CPU:0"):
        model = (StarDist3D if n_dim == 3 else StarDist2D)(None, name=folder.name, basedir=str(folder.parent))
        stored = {"prob": float(model.thresholds.prob), "nms": float(model.thresholds.nms)}
        prob = stored["prob"] if config.prob_thresh is None else float(config.prob_thresh)
        nms = stored["nms"] if config.nms_thresh is None else float(config.nms_thresh)
        options = dict(n_tiles=tuple(n_tiles), prob_thresh=prob, nms_thresh=nms, show_tile_progress=False)
        if blocks is None:
            labels, polys = model.predict_instances(x, axes=axes, scale=library_scale, **options)
        else:
            effective = blocks["effective"]
            labels, polys = model.predict_instances_big(
                x, axes=axes, block_size=tuple(effective["block_size"]), min_overlap=tuple(effective["min_overlap"]),
                context=tuple(effective["context"]), show_progress=False, **options)
    source = "stored" if config.prob_thresh is None and config.nms_thresh is None else "override"
    details = {
        "effective": {"scale_zyx": list(scale_zyx), "library_scale": None if library_scale is None
                      else list(library_scale), "prob_thresh": prob, "nms_thresh": nms, "threshold_source": source,
                      "stored_thresholds": stored, "normalize_percentiles": [p_low, p_high],
                      "percentiles": norm_record["percentiles"], "n_tiles": list(n_tiles), "n_dim": n_dim,
                      "grid": list(grid), "prediction": "whole_image" if blocks is None else "block_wise",
                      "blocks": blocks},
        "library_dtype": str(labels.dtype), "counts": {"objects": int(len(polys["prob"]))},
        "framework": framework,
    }
    return labels, details


def _cellpose(image, config: CellposeConfig, context) -> tuple[np.ndarray, dict[str, Any]]:
    """``CellposeModel(pretrained_model=<checked path>).eval`` with the recorded parameters.

    See docs/segmentation-algorithms.md, "cellpose".

    The channels are passed in role order, ``cytoplasm`` then ``nuclear``; a
    Z=1 input is passed as one plane (YX, or YXC with ``channel_axis``) and the
    wrapper restores the Z axis Cellpose drops; 3D mode passes the ZYXC stack
    with ``z_axis=0``, ``do_3D=True`` and the anisotropy. The normalization is
    a new mapping on every call; HF_HUB_OFFLINE=1 is set during the call.
    """
    channels = [role for role in ("cytoplasm", "nuclear") if role in context.roles]
    x = image[..., [context.roles.index(role) for role in channels]]
    kwargs = dict(diameter=config.diameter, flow_threshold=config.flow_threshold,
                  cellprob_threshold=config.cellprob_threshold, min_size=config.min_size,
                  tile_overlap=config.tile_overlap,
                  normalize={"normalize": True, "percentile": list(config.normalize_percentiles)})
    if image.shape[0] == 1:
        x = x[0]
        if x.shape[-1] == 1:
            x = x[..., 0]
        else:
            kwargs["channel_axis"] = -1
    else:
        kwargs.update(channel_axis=-1, z_axis=0, do_3D=True, anisotropy=config.anisotropy)

    import torch
    from cellpose import models

    device, framework = _torch_device(torch, context.device)
    with _offline():
        model = models.CellposeModel(pretrained_model=str(context.model), device=device,
                                     use_bfloat16=config.bfloat16)
        masks = model.eval(x, **kwargs)[0]
    masks = np.asarray(masks)
    defaults = config.flow_threshold == 0.4 and config.cellprob_threshold == 0.0
    details = {
        "effective": {"diameter": config.diameter,
                      "rescale": None if config.diameter is None else 30.0 / config.diameter,
                      "flow_threshold": config.flow_threshold, "cellprob_threshold": config.cellprob_threshold,
                      "threshold_source": "library_default" if defaults else "override",
                      "tile_overlap": config.tile_overlap, "bfloat16": config.bfloat16, "min_size": config.min_size,
                      "normalize_percentiles": list(config.normalize_percentiles), "do_3d": config.do_3d,
                      "anisotropy": config.anisotropy, "channels": channels},
        "library_dtype": str(masks.dtype), "framework": framework,
    }
    return masks, details
