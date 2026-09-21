"""Bounded formed-amplicon model, starfinder.synthetic/1.

Historical rendering has different support, precision and observation semantics;
this implementation deliberately does not alter those historical fixtures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import unicodedata

import numpy as np
import pandas as pd

from starfinder.barcode import Codebook
from starfinder.image import ImageMetadata

CONTRACT = "starfinder.synthetic/1"
SPEC_REVISION = "f9512694a0960c10ce5236efbaaf9d6f425c1d8a"
_COMPONENTS = ("count", "placement", "identity", "brightness", "width.axial",
               "width.lateral", "elongation", "angle")


def _label(value):
    if (not isinstance(value, str) or not value
            or unicodedata.normalize("NFC", value) != value):
        raise ValueError("labels and IDs must be nonempty NFC strings")
    return value


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


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
    elongation; angle allows constant or uniform (0, pi). See the specification
    for domains. Supplied values consume no random draws.
    """

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
          or (name == "elongation" and (values < 1).any())):
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


@dataclass(frozen=True)
class ReadoutEffectsConfig:
    """Optional effective readout controls; every enable flag defaults false.

    Probabilities/factors are length-R sequences (None selects zero probability
    or unit weakening). Trend is b**r, b in [0,1]. Loss is one draw per ID,
    starting at zero-based loss_start (None means 1 for R>1, else 0).
    Gains are R by C; mixing is R by C by C with destination rows/source
    columns. None selects ones/identity. No broadcasting or normalization.
    All requested values are validated even when disabled; disabled values
    become algebraic identities and consume no random draws.
    """

    dropout_enabled: bool = False
    dropout_probability: tuple[float, ...] | None = None
    weakening_enabled: bool = False
    weakening_probability: tuple[float, ...] | None = None
    weak_factor: tuple[float, ...] | None = None
    trend_enabled: bool = False
    trend_base: float = 1.0
    loss_enabled: bool = False
    loss_probability: float = 0.0
    loss_start: int | None = None
    gain_enabled: bool = False
    gains: object = None
    mixing_enabled: bool = False
    mixing: object = None


def _readout_effects(config, intended, ids, labels, stream):
    if not isinstance(config, ReadoutEffectsConfig):
        raise TypeError("readout must be ReadoutEffectsConfig")
    n, c, rounds = intended.shape
    flags = {name: getattr(config, name + "_enabled") for name in
             ("dropout", "weakening", "trend", "loss", "gain", "mixing")}
    if any(type(value) is not bool for value in flags.values()):
        raise ValueError("readout enable flags must be Boolean")

    def parameter(value, name, shape, default, upper=None):
        result = np.full(shape, default, dtype=np.float64) if value is None else _array(value, name)
        if (result.shape != shape or (result < 0).any()
                or (upper is not None and (result > upper).any())):
            raise ValueError(f"invalid {name}: expected shape {shape} and values in [0, {upper}]")
        return result

    drop = parameter(config.dropout_probability, "dropout_probability", (rounds,), 0, 1)
    weak = parameter(config.weakening_probability, "weakening_probability", (rounds,), 0, 1)
    factor = parameter(config.weak_factor, "weak_factor", (rounds,), 1, 1)
    base = parameter(_array(config.trend_base, "trend_base"), "trend_base", (), 1, 1)
    loss = parameter(_array(config.loss_probability, "loss_probability"), "loss_probability", (), 0, 1)
    start = min(1, rounds - 1) if config.loss_start is None else config.loss_start
    _integer(start, "loss_start", 0, rounds - 1)
    gains = parameter(config.gains, "gains", (rounds, c), 1)
    mixing = parameter(np.repeat(np.eye(c)[None], rounds, axis=0)
                       if config.mixing is None else config.mixing,
                       "mixing", (rounds, c, c), 0)
    if not flags["dropout"]:
        drop = np.zeros(rounds)
    if not flags["weakening"]:
        weak, factor = np.zeros(rounds), np.ones(rounds)
    if not flags["trend"]:
        base = 1.0
    if not flags["loss"]:
        loss = 0.0
    if not flags["gain"]:
        gains = np.ones((rounds, c))
    if not flags["mixing"]:
        mixing = np.repeat(np.eye(c)[None], rounds, axis=0)
    dropped, weakened, lost = (np.zeros((n, rounds), dtype=bool) for _ in range(3))
    first_loss = [pd.NA] * n
    for i, identity in enumerate(ids):
        if flags["loss"] and stream("round.loss", identity).random() < loss:
            first_loss[i] = start
            lost[i, start:] = True
        for r, label in enumerate(labels):
            if flags["dropout"]:
                dropped[i, r] = stream("round.dropout", identity, label).random() < drop[r]
            if flags["weakening"]:
                weakened[i, r] = stream("round.weakening", identity, label).random() < weak[r]
    trend = np.full((n, rounds), base) ** np.arange(rounds)
    multiplier = np.where(weakened, factor[None, :], 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        pre_mix = (intended * (~lost & ~dropped)[:, None, :]
                   * multiplier[:, None, :] * trend[:, None, :] * gains.T[None, :, :])
        realized = np.einsum("rds,nsr->ndr", mixing, pre_mix)
    if not np.isfinite(pre_mix).all() or not np.isfinite(realized).all():
        raise ValueError("nonfinite readout amplitudes")
    state = dict(dropped=dropped, weakened=weakened, lost=lost,
                 trend_multiplier=trend, weak_multiplier=multiplier)
    effective = dict(type="ReadoutEffectsConfig", **{k + "_enabled": v for k, v in flags.items()},
                     dropout_probability=drop.tolist(), weakening_probability=weak.tolist(),
                     weak_factor=factor.tolist(), trend_base=float(base), loss_probability=float(loss),
                     loss_start=start, gains=gains.tolist(), mixing=mixing.tolist())
    return pre_mix, realized, state, first_loss, effective


@dataclass(frozen=True)
class TextureConfig:
    """Persistent Gaussian blobs, independent of formed molecules.

    Count/density/coordinates and placement follow FormedSceneConfig. Default
    enabled count is four. Width distributions are axial/lateral (YX shared),
    in voxel indices; brightness is peak intensity. Supplied values use blob-N
    IDs. Enable this component with BackgroundConfig.texture_enabled.
    """

    count: int | None = None
    density: float | None = None
    coordinates: object = None
    max_count: int = 1024
    placement: str = "uniform"
    spatial_weights: object = None
    cluster_centers: object = None
    cluster_weights: object = None
    spread_zyx: tuple[float, float, float] = (1, 2, 2)
    axial_width: ScalarDistribution = field(default_factory=ScalarDistribution)
    lateral_width: ScalarDistribution = field(default_factory=lambda: ScalarDistribution(parameters=(3,)))
    brightness: ScalarDistribution = field(default_factory=lambda: ScalarDistribution(parameters=(5,)))


@dataclass(frozen=True)
class BackgroundConfig:
    """Analytic reference background and destination-frame R×C baseline.

    All flags default false. Gradient is max(0, intercept + slopes dot
    normalized ZYX). Regions are supplied K×3 centers, positive sigma_zyx
    triples and nonnegative heights (None gives (2,8,8), 10 per region).
    Tissue weights and baselines are nonnegative R×C arrays, default zero.
    Disabled parameters are validated and retained, but contribute zero.
    """

    baseline_enabled: bool = False
    baseline: object = None
    gradient_enabled: bool = False
    gradient_intercept: float = 0.0
    gradient_slopes_zyx: tuple[float, float, float] = (0, 0, 0)
    regions_enabled: bool = False
    region_centers: object = ()
    region_sigma_zyx: object = None
    region_heights: object = None
    texture_enabled: bool = False
    texture: TextureConfig = field(default_factory=TextureConfig)
    tissue_weights: object = None


@dataclass(frozen=True)
class NoiseConfig:
    """Separate residual normals: sqrt(alpha*J)*Z_dep, then sigma*Z_ind.

    J includes signal, weighted tissue and baseline. Both flags default false;
    alpha/sigma are finite nonnegative scalars (zero by default). Round/channel
    keyed standardized draws persist when strengths change. No photon claim.
    """

    dependent_enabled: bool = False
    alpha: float = 0.0
    independent_enabled: bool = False
    sigma: float = 0.0


@dataclass(frozen=True)
class FormedSceneConfig:
    """Inputs for one bounded development FOV; readout defaults clean/disabled.

    Select coordinates, count or density (all None defaults to count=8).
    Explicit coordinates are N by 3 in the order of amplicon_ids; default IDs
    are amplicon-0, amplicon-1, etc. Gene/property overrides are ID-keyed.
    Placement is uniform, weighted (ZYX spatial_weights), or clustered (centers,
    positive cluster_weights, spread_zyx). Density is Poisson amplicons/voxel.
    Bounds are ZYX <= (32,64,64), max_count <= 1024, and <=4 ordered rounds.
    """

    dataset_version: str = "formed-small-v1"
    sample_id: str = "sample"
    FOV_id: str = "FOV_001"
    scene_key: str = "formed-v1"
    seed: int = 42
    split: str = "development"
    shape_zyx: tuple[int, int, int] = (8, 32, 32)
    dtype: str = "float32"
    count: int | None = None
    density: float | None = None
    coordinates: tuple[tuple[float, float, float], ...] | None = None
    max_count: int = 1024
    amplicon_ids: tuple[str, ...] | None = None
    gene_ids: dict[str, str] | None = None
    abundances: tuple[float, ...] | None = None
    placement: str = "uniform"
    spatial_weights: object = None
    cluster_centers: tuple[tuple[float, float, float], ...] | None = None
    cluster_weights: tuple[float, ...] | None = None
    spread_zyx: tuple[float, float, float] = (1, 2, 2)
    brightness: ScalarDistribution = field(default_factory=lambda: ScalarDistribution(parameters=(100,)))
    axial_width: ScalarDistribution = field(default_factory=ScalarDistribution)
    lateral_width: ScalarDistribution = field(default_factory=ScalarDistribution)
    elongation: ScalarDistribution = field(default_factory=ScalarDistribution)
    angle: ScalarDistribution = field(default_factory=lambda: ScalarDistribution(parameters=(0,)))
    readout: ReadoutEffectsConfig = field(default_factory=ReadoutEffectsConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)


@dataclass
class FormedScene:
    """Complete formed and per-round truth, images and reproducibility payload.

    rounds maps acquisition labels to ZYXC arrays in codebook order. Signal
    tensors are float64 NCR, indexed by the explicit amplicon_ids/channel_labels/
    round_labels axes. Tables join on (namespace, amplicon_id), never row number.
    No eligibility or biological RNA truth is inferred. provenance contains an
    artifacts/1 synthetic source extension; generation performs no file writes.
    """

    rounds: dict[str, np.ndarray]
    metadata: ImageMetadata
    formed: pd.DataFrame
    round_truth: pd.DataFrame
    intended: np.ndarray
    pre_mix: np.ndarray
    realized: np.ndarray
    amplicon_ids: tuple[str, ...]
    round_labels: tuple[str, ...]
    channel_labels: tuple[str, ...]
    codebook: Codebook
    provenance: dict


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


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


def _kernel(shape, point, sz, sl, elongation, angle):
    c, s = np.cos(angle), np.sin(angle)
    with np.errstate(over="ignore", invalid="ignore"):
        extent = 4 * np.array([sz, sl * np.hypot(elongation * c, s),
                              sl * np.hypot(elongation * s, c)])
    if not np.isfinite(extent).all():
        raise ValueError("nonfinite kernel extent")
    truncated = bool(((point - extent < 0) | (point + extent > shape - 1)).any())
    # Clip in floating point before integer conversion (large finite centers).
    low = np.ceil(np.clip(point - extent, 0, shape)).astype(int)
    high = np.floor(np.clip(point + extent, -1, shape - 1)).astype(int) + 1
    slices = tuple(slice(a, b) for a, b in zip(low, high))
    if (high <= low).any():
        return slices, None, truncated
    z, y, x = np.ogrid[tuple(slice(a, b) for a, b in zip(low, high))]
    dz, dy, dx = z - point[0], y - point[1], x - point[2]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        radius = (dz / sz)**2 + ((c * dy + s * dx) / (sl * elongation))**2 + ((-s * dy + c * dx) / sl)**2
    support = radius <= 16
    kernel = np.where(support, np.exp(-.5 * radius), 0.)
    if not np.isfinite(kernel).all():
        raise ValueError("nonfinite generated kernel")
    return slices, kernel, truncated


def generate_formed_scene(codebook: Codebook, *, config: FormedSceneConfig = FormedSceneConfig(),
                          metadata: ImageMetadata | None = None) -> FormedScene:
    """Generate an in-memory formed population and controlled readout images.

    Implements A1–A6/A8 and truth/provenance of A9 in synthetic/1. Geometry is
    identity; optional background/noise default disabled with no hidden defaults.
    Invalid input or nonfinite generation raises ValueError (wrong typed objects
    raise TypeError). Density overflow/max_count errors never cap or retry N.
    """
    if not isinstance(config, FormedSceneConfig) or not isinstance(codebook, Codebook):
        raise TypeError("expected FormedSceneConfig and Codebook")
    # Revalidate/copy mutable payloads instead of trusting an old constructor.
    codebook = Codebook(codebook.table, codebook.round_labels, codebook.channel_labels,
                        codebook.color_to_channel, codebook.encoding)
    if not 1 <= len(codebook.round_labels) <= 4 or codebook.n_genes == 0:
        raise ValueError("require 1–4 rounds and a nonempty codebook")
    for value in (config.dataset_version, config.sample_id, config.FOV_id, config.scene_key,
                  *codebook.round_labels, *codebook.channel_labels, *codebook.genes):
        _label(value)
    if config.split != "development":
        raise ValueError("only development scene generation is supported")
    _integer(config.seed, "seed", 0, 2**64 - 1)
    if len(config.shape_zyx) != 3:
        raise ValueError("shape must be ZYX")
    for n, limit in zip(config.shape_zyx, (32, 64, 64)):
        _integer(n, "shape dimension", 1, limit)
    shape = np.array(config.shape_zyx)
    if config.dtype not in ("float32", "float64", "uint8", "uint16"):
        raise ValueError("unsupported output dtype")
    _integer(config.max_count, "max_count", 0, 1024)
    namespace = _json([config.dataset_version, config.sample_id, config.FOV_id, "formed"])
    if metadata is None:
        metadata = ImageMetadata(namespace + "/reference")
    if not isinstance(metadata, ImageMetadata):
        raise TypeError("metadata must be ImageMetadata")
    metadata = ImageMetadata(**asdict(metadata))
    streams = {}

    def stream(component, entity=None, round_label=None, channel_label=None):
        descriptor = [CONTRACT, config.split, config.seed, config.scene_key,
                      component, entity, round_label, channel_label]
        encoded = _json(descriptor)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        streams[encoded] = dict(descriptor=descriptor, sha256=digest, bit_generator="PCG64", numpy_version=np.__version__)
        return np.random.Generator(np.random.PCG64(int(digest, 16)))

    if sum(x is not None for x in (config.count, config.density, config.coordinates)) > 1:
        raise ValueError("select exactly one of coordinates/count/density")
    weights, centers, spread = _placement(config, shape)
    coordinates = None
    if config.coordinates is not None:
        coordinates = _array(config.coordinates, "coordinates")
        if coordinates.shape == (0,):
            coordinates = coordinates.reshape(0, 3)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must be N by 3")
        n = len(coordinates)
        if shape[0] == 1 and np.any(coordinates[:, 0] != 0):
            raise ValueError("Z=1 requires formed z=0")
    elif config.density is not None:
        density = _array(config.density, "density")
        if density.ndim != 0 or density < 0:
            raise ValueError("density must be a nonnegative scalar")
        mean = float(density) * int(shape.prod())
        if not np.isfinite(mean):
            raise ValueError("nonfinite expected count")
        n = int(stream("count").poisson(mean))
    else:
        n = 8 if config.count is None else config.count
        _integer(n, "count", 0, config.max_count)
    if n > config.max_count:
        raise ValueError("formed count exceeds max_count")
    ids = tuple(f"amplicon-{i}" for i in range(n)) if config.amplicon_ids is None else config.amplicon_ids
    if not isinstance(ids, tuple) or len(ids) != n or len(set(ids)) != n:
        raise ValueError("amplicon_ids must be a unique tuple of length N")
    for identity in ids:
        _label(identity)
    if config.gene_ids is not None:
        if (not isinstance(config.gene_ids, dict) or set(config.gene_ids) != set(ids)
                or any(g not in codebook.genes for g in config.gene_ids.values())):
            raise ValueError("gene_ids must map every amplicon ID to a codebook gene")
        if config.abundances is not None:
            raise ValueError("supplied genes and abundance draws are mutually exclusive")
    abundances = _probabilities(config.abundances if config.abundances is not None else np.ones(codebook.n_genes),
                               codebook.n_genes, "abundances")
    specs = (config.brightness, config.axial_width, config.lateral_width, config.elongation, config.angle)
    properties = _COMPONENTS[3:]
    for name, spec in zip(properties, specs):
        _distribution(spec, name, ids)
    points = np.empty((n, 3), dtype=np.float64)
    values = np.empty((n, 5), dtype=np.float64)
    genes = []
    for i, identity in enumerate(ids):
        points[i] = (coordinates[i] if coordinates is not None else
                     _position(config, shape, stream("placement", identity), weights, centers, spread))
        genes.append(config.gene_ids[identity] if config.gene_ids is not None else
                     codebook.genes[stream("identity", identity).choice(codebook.n_genes, p=abundances)])
        values[i] = [_draw(spec, name, identity, stream) for name, spec in zip(properties, specs)]
    if not np.isfinite(points).all() or not np.isfinite(values).all() or (values[:, 1:3] <= 0).any():
        raise ValueError("nonfinite or out-of-domain generated properties")
    sequences = [codebook.gene_to_seq[g] for g in genes]
    formed = pd.DataFrame(dict(namespace=[namespace]*n, amplicon_id=ids, gene_id=genes, codeword=sequences,
                               frame_id=[metadata.frame_id]*n, formed_index=np.arange(n, dtype=np.int64),
                               **{k: points[:, j] for j, k in enumerate(("z", "y", "x"))},
                               **{k: values[:, j] for j, k in enumerate(("A", "sz", "sl", "e", "theta"))}))
    formed = formed.astype({k: "string" for k in ("namespace", "amplicon_id", "gene_id", "codeword", "frame_id")})
    intended = np.zeros((n, 4, len(codebook.round_labels)), dtype=np.float64)
    for i, sequence in enumerate(sequences):
        for r, color in enumerate(sequence):
            intended[i, codebook.color_to_channel[color], r] = values[i, 0]
    pre_mix, realized, effects, first_loss, effective_readout = _readout_effects(
        config.readout, intended, ids, codebook.round_labels, stream)
    images = {label: np.zeros((*shape, 4), dtype=np.float64) for label in codebook.round_labels}
    intersects = np.zeros(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    # Sorting IDs fixes accumulation order independently of caller scheduling.
    for i in sorted(range(n), key=lambda i: ids[i]):
        _, sz, sl, e, theta = values[i]
        slices, kernel, truncated[i] = _kernel(shape, points[i], sz, sl, e, theta)
        intersects[i] = kernel is not None and bool(np.any(kernel > 0))
        if intersects[i]:
            with np.errstate(over="ignore", invalid="ignore"):
                for r, image in enumerate(images.values()):
                    image[slices] += kernel[..., None] * realized[i, :, r]
    round_tables = []
    transforms = {}
    for r, label in enumerate(codebook.round_labels):
        transform_id = _json([namespace, label, "identity"])
        transforms[transform_id] = dict(kind="identity", direction="reference_to_round", units="voxel_index",
                                        source_frame=metadata.frame_id, destination_frame=metadata.frame_id,
                                        translation_zyx=[0., 0., 0.])
        table = formed[["namespace", "amplicon_id", "z", "y", "x"]].copy()
        table["round_label"] = pd.Series([label]*n, dtype="string")
        table["round_index"] = np.full(n, r, dtype=np.int64)
        table["transform_id"] = pd.Series([transform_id]*n, dtype="string")
        for key in ("dropped", "weakened", "lost"):
            table[key] = effects[key][:, r]
        table["emitting"] = np.any(realized[:, :, r] > 0, axis=1)
        table["center_in_bounds"] = ((points >= 0) & (points <= shape - 1)).all(axis=1)
        table["support_intersects"] = intersects
        table["support_truncated"] = truncated
        table["first_loss_round"] = pd.Series(first_loss, dtype="Int64")
        for key in ("trend_multiplier", "weak_multiplier"):
            table[key] = effects[key][:, r]
        round_tables.append(table)
    from ._observation import _prepare_background, evaluate_background, _observe
    background, effective_background = _prepare_background(config.background, shape, len(images), stream)
    for component in background:
        component['frame_id'] = metadata.frame_id
    tissue = evaluate_background(background, np.moveaxis(np.indices(shape, dtype=np.float64), 0, -1))
    observation, effective_noise = _observe(images, tissue, effective_background, config.noise,
                                           codebook.channel_labels, stream)
    clipping = {}
    for label, image in images.items():
        if not np.isfinite(image).all():
            raise ValueError("nonfinite rendered image")
        low_count = high_count = 0
        if config.dtype.startswith("uint"):
            rounded = np.rint(image)
            limit = np.iinfo(config.dtype).max
            low_count, high_count = int((rounded < 0).sum()), int((rounded > limit).sum())
            image = np.clip(rounded, 0, limit)
        with np.errstate(over="ignore"):
            images[label] = image.astype(config.dtype)
        if not np.isfinite(images[label]).all():
            raise ValueError("output dtype overflow")
        clipping[label] = dict(below=low_count, above=high_count)
    config_payload = _plain(asdict(config))
    config_payload["type"] = "FormedSceneConfig"
    config_payload["readout"]["type"] = "ReadoutEffectsConfig"
    config_payload["background"]["type"] = "BackgroundConfig"
    config_payload["background"]["texture"]["type"] = "TextureConfig"
    config_payload["noise"]["type"] = "NoiseConfig"
    for key in ("axial_width", "lateral_width", "brightness"):
        config_payload["background"]["texture"][key]["type"] = "ScalarDistribution"
    for key in ("brightness", "axial_width", "lateral_width", "elongation", "angle"):
        config_payload[key]["type"] = "ScalarDistribution"
    codebook_payload = dict(rows=codebook.table.to_dict("records"), round_labels=list(codebook.round_labels),
                            channel_labels=list(codebook.channel_labels), color_to_channel=codebook.color_to_channel,
                            encoding=asdict(codebook.encoding))
    effective = dict(config_payload, count=n, placement="explicit" if coordinates is not None else config.placement,
                     abundance_probabilities=abundances.tolist(), readout=effective_readout,
                     background=effective_background, noise=effective_noise,
                     effects_enabled=any(v for group in (effective_readout, effective_background, effective_noise)
                                         for k, v in group.items() if k.endswith("_enabled")))
    payload = dict(contract_id=CONTRACT, contract_revision=SPEC_REVISION,
                   generator="starfinder.synthetic.generate_formed_scene", generator_version="3",
                   truth_namespace=namespace,
                   requested_config=config_payload, effective_config=effective, codebook=codebook_payload,
                   geometry=asdict(metadata), singleton_z_sampling=bool(shape[0] == 1),
                   amplicon_ids=list(ids), signal_axes="NCR", round_labels=list(codebook.round_labels),
                   channel_labels=list(codebook.channel_labels), streams=list(streams.values()),
                   transforms=transforms, background_components=background, observation=dict(
                       **observation,
                       shape_zyxc=[*config.shape_zyx, 4], dtype=config.dtype, clipping_counts=clipping,
                       image_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in images.items()}),
                   truth_components=["formed", "round_truth", "intended", "pre_mix", "realized"],
                   order=["formed", "persistent_properties", "intended", "survival", "dropout",
                          "weakening", "trend", "source_gain", "mixing", "render", "tissue",
                          "baseline", "noise.dependent", "noise.independent", "cast", "truth"])
    payload["config_sha256"] = hashlib.sha256(_json(dict(config=config_payload, codebook=codebook_payload,
                                                        metadata=asdict(metadata))).encode()).hexdigest()
    provenance = dict(contract_id="starfinder.artifacts/1", source_kind="synthetic",
                      source_id="formed:" + payload["config_sha256"],
                      dataset_version=config.dataset_version,
                      catalog=("docs/datasets.md#structured-background-development-v1" if
                               any(v for group in (effective_background, effective_noise)
                                   for k, v in group.items() if k.endswith("_enabled")) else
                               "docs/datasets.md#controlled-readout-development-v1" if effective["effects_enabled"]
                               else "docs/datasets.md#formed-amplicon-clean-development-v1"),
                      uri=None, sha256=None,
                      unverified_reason="In-memory source; serialized artifact identities must be recorded by the publisher.",
                      selection=dict(axes="ZYXC", round_labels=list(codebook.round_labels),
                                     channel_labels=list(codebook.channel_labels),
                                     geometry=asdict(metadata), conversion=None),
                      extensions={"starfinder.synthetic": payload},
                      limitations=["Development processed-image model, not calibrated D04 or biological RNA truth.",
                                   "Geometry is identity; no eligibility inferred.",
                                   "Analytic background and residual noise are not measured tissue or detector physics.",
                                   "Readout probabilities and gains are effective controls, not calibrated chemistry.",
                                   "Bitwise repeatability is limited to the pinned NumPy environment."])
    return FormedScene(images, metadata, formed, pd.concat(round_tables, ignore_index=True),
                       intended, pre_mix, realized, ids, codebook.round_labels,
                       codebook.channel_labels, codebook, provenance)


def formed_scene_preset(name: str = "formed-small-v1") -> tuple[Codebook, FormedSceneConfig]:
    """Return the exact clean 3D or singleton-Z synthetic/1 development preset."""
    if name not in ("formed-small-v1", "formed-z1-v1"):
        raise ValueError("unknown formed scene preset")
    codebook = Codebook(pd.DataFrame(dict(gene_id=["gene-A", "gene-B"], color_sequence=["123", "214"])),
                        ("round10", "round2", "round1"), ("ch02", "ch00", "ch03", "ch01"),
                        {"1": 1, "2": 0, "3": 3, "4": 2})
    return codebook, FormedSceneConfig(dataset_version=name, shape_zyx=(1 if name == "formed-z1-v1" else 8, 32, 32))
