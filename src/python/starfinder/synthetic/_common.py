"""Validation, scalar laws and placement shared by formed scenes and backgrounds.

This module imports no other formed-scene module, so the formed, geometry and
observation modules depend on it without an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import unicodedata

import numpy as np


def _label(value):
    if (not isinstance(value, str) or not value
            or unicodedata.normalize("NFC", value) != value):
        raise ValueError("labels and IDs must be nonempty NFC strings")
    return value


def _integer(value, name, low, high=None):
    """Require a non-Boolean int in [low, high]; high None means unbounded."""
    if type(value) is not int or value < low or (high is not None and value > high):
        bound = "∞)" if high is None else f"{high}]"
        raise ValueError(f"{name} must be an integer in [{low}, {bound}")


def _array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return raw.astype(np.float64)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ScalarDistribution:
    """Persistent per-ID scalar law in voxel/intensity units.

    Modes and parameters: constant (value,), uniform (low, high), lognormal
    (log_median, log_sd), folded_lognormal (log_median, log_sd), or supplied
    (no parameters, values keyed by amplicon ID). Folded lognormal is only for
    elongation; angle allows constant or supplied values in [0, pi), or uniform
    (0, pi). See the specification for domains. Supplied values consume no
    random draws. Instances are not hashable (supplied values are a dict).
    """

    __hash__ = None

    mode: str = "constant"
    parameters: tuple[float, ...] = (1.0,)
    values: dict[str, float] | None = None


def _distribution(spec, name, ids):
    if not isinstance(spec, ScalarDistribution):
        raise TypeError(f"{name} must be ScalarDistribution")
    modes = {"constant": 1, "uniform": 2, "lognormal": 2,
             "folded_lognormal": 2, "supplied": 0}
    if spec.mode not in modes:
        raise ValueError(f"unknown distribution: {spec.mode}")
    p = _array(spec.parameters, name)
    if p.shape != (modes[spec.mode],):
        raise ValueError(f"wrong parameters for {name}/{spec.mode}")
    if spec.mode == "supplied":
        if not isinstance(spec.values, dict) or set(spec.values) != set(ids):
            raise ValueError(f"{name} supplied values must match amplicon IDs exactly")
        values = _array(list(spec.values.values()), name)
        if values.ndim != 1:
            raise ValueError("supplied properties must be scalars")
    else:
        if spec.values is not None:
            raise ValueError("values require supplied mode")
        values = p
    if name == "angle" and spec.mode not in ("constant", "uniform", "supplied"):
        raise ValueError("angle supports only constant, uniform or supplied")
    if spec.mode == "uniform" and p[0] > p[1]:
        raise ValueError("uniform low must not exceed high")
    if name == "angle" and spec.mode == "uniform" and tuple(p) != (0, np.pi):
        raise ValueError("angle uniform parameters must be (0, pi)")
    if spec.mode in ("lognormal", "folded_lognormal"):
        if p[1] < 0:
            raise ValueError("log standard deviation must be nonnegative")
        if (spec.mode == "folded_lognormal") != (name == "elongation"):
            raise ValueError("elongation requires folded_lognormal; other properties do not")
        if name == "elongation" and p[0] < 0:
            raise ValueError("folded log median must be nonnegative")
    elif ((name == "brightness" and (values < 0).any())
          or (name.startswith("width.") and (values <= 0).any())
          or (name == "elongation" and (values < 1).any())
          or (name == "angle" and spec.mode != "uniform"
              and ((values < 0) | (values >= np.pi)).any())):
        raise ValueError(f"invalid {name} domain")


def _draw(spec, name, identity, stream):
    p = spec.parameters
    if spec.mode == "supplied":
        return float(spec.values[identity])
    if spec.mode == "constant":
        return float(p[0])
    rng = stream(name, identity)
    if spec.mode == "uniform":
        return float(rng.uniform(*p))
    normal = p[1] * rng.standard_normal()
    with np.errstate(over="raise", invalid="raise"):
        try:
            return float(np.exp(p[0] + (abs(normal) if spec.mode == "folded_lognormal" else normal)))
        except FloatingPointError as exc:
            raise ValueError(f"nonfinite generated {name}") from exc


def _probabilities(value, size, name, *, positive=False):
    weights = _array(value, name)
    if (weights.shape != (size,) or (weights < 0).any()
            or (positive and (weights <= 0).any()) or not np.any(weights > 0)):
        raise ValueError(f"invalid {name}")
    # Scale first to avoid overflow of an otherwise valid finite weight sum.
    weights = weights / weights.max()
    return weights / weights.sum()


def _placement(config, shape):
    mode = config.placement
    if mode not in ("uniform", "weighted", "clustered"):
        raise ValueError("unknown placement mode")
    if config.coordinates is not None and mode != "uniform":
        raise ValueError("explicit coordinates cannot also request placement")
    if (mode != "weighted" and config.spatial_weights is not None
            or mode != "clustered" and (config.cluster_centers is not None or config.cluster_weights is not None)):
        raise ValueError("placement parameters do not match mode")
    spread = _array(config.spread_zyx, "spread_zyx")
    if spread.shape != (3,) or (spread <= 0).any():
        raise ValueError("cluster spread must be a positive ZYX triple")
    if mode == "weighted":
        weights = _array(config.spatial_weights, "spatial_weights")
        if weights.shape != tuple(shape):
            raise ValueError("spatial weights must match ZYX shape")
        return _probabilities(weights.ravel(), weights.size, "spatial_weights"), None, spread
    if mode == "clustered":
        centers = _array(config.cluster_centers, "cluster_centers")
        if (centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0
                or (centers < 0).any() or (centers > shape - 1).any()):
            raise ValueError("cluster centers must be nonempty in-bounds K by 3")
        weights = _probabilities(config.cluster_weights, len(centers), "cluster_weights", positive=True)
        return weights, centers, spread
    return None, None, spread


def _position(config, shape, rng, weights, centers, spread):
    if config.placement == "uniform":
        return rng.uniform(0, shape - 1, 3)
    if config.placement == "weighted":
        voxel = np.array(np.unravel_index(rng.choice(weights.size, p=weights), tuple(shape)))
        return rng.uniform(np.maximum(0, voxel - .5), np.minimum(shape - 1, voxel + .5))
    center = centers[rng.choice(len(centers), p=weights)]
    for _ in range(10000):
        offset = rng.normal(0, spread, 3)
        offset[shape == 1] = 0
        point = center + offset
        if ((point >= 0) & (point <= shape - 1)).all():
            return point
    raise ValueError("cluster placement exhausted 10000 proposals for one amplicon")
