"""Formed-amplicon scene model; see docs/synthetic-specification.md.

Historical rendering has different support, precision and observation semantics;
this implementation deliberately does not alter those historical fixtures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib

import numpy as np
import pandas as pd

from starfinder.barcode import Codebook
from starfinder.image import ImageMetadata
from ._common import (ScalarDistribution, _array, _distribution, _draw, _integer,
                      _json, _label, _placement, _position, _probabilities)
from ._geometry import GeometryConfig, _forward, _inverse, _kind, _prepare_geometry
from ._observation import (BackgroundConfig, NoiseConfig, _observe,
                           _prepare_background, evaluate_background)

# First element of every stream descriptor; changing it would redraw every scene.
_STREAM_NAMESPACE = "starfinder.synthetic/1"
_STREAM_KEY = ("namespace", "split", "seed", "scene_key", "component", "entity",
               "round_label", "channel_label")
_GENERATOR_VERSION = "5"
_COMPONENTS = ("count", "placement", "identity", "brightness", "width.axial",
               "width.lateral", "elongation", "angle")


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
    Instances are not hashable (see FormedSceneConfig).
    """

    __hash__ = None

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
class FormedSceneConfig:
    """Inputs for one formed-amplicon FOV; every effect defaults disabled.

    Select coordinates, count or density (all None defaults to count=8).
    Explicit coordinates are N by 3 in the order of amplicon_ids; default IDs
    are amplicon-0, amplicon-1, etc. Gene/property overrides are ID-keyed.
    Placement is uniform, weighted (ZYX spatial_weights), or clustered (centers,
    positive cluster_weights, spread_zyx). Density is Poisson amplicons/voxel.
    Shape dimensions are positive integers; the codebook sets the round count.
    max_count is a user-settable allocation guard: a larger N fails, never
    truncates. split and scene_key are stream-key labels (no reserved values).

    Configs are frozen but deliberately not hashable: several fields accept
    dicts and arrays. Use provenance["requested_config"] as a stable identity.
    """

    __hash__ = None

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
    geometry: GeometryConfig = field(default_factory=GeometryConfig)


@dataclass
class FormedScene:
    """Complete formed and per-round truth, images and reproducibility payload.

    rounds maps acquisition labels to ZYXC arrays in codebook order. Signal
    tensors are float64 NCR, indexed by the explicit amplicon_ids/channel_labels/
    round_labels axes. Tables join on (namespace, amplicon_id), never row number.
    No eligibility or biological RNA truth is inferred. provenance records the
    generator version, requested/effective config, seed and stream scheme,
    codebook, per-round image SHA-256, clipping counts and transforms.
    Generation performs no file writes; see save_formed_scene.
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

    @property
    def round_metadata(self) -> dict[str, ImageMetadata]:
        """Output grid metadata per round, with explicit destination frame IDs.

        ``metadata`` remains the reference grid. Physical grid fields are shared;
        deformation moves content, not the voxel grid or its calibration.
        """
        transforms = self.provenance['transforms']
        return {t['round_label']: ImageMetadata(**dict(asdict(self.metadata),
                                                      frame_id=t['destination_frame']))
                for t in transforms.values()}


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

    Stages and effect order follow docs/synthetic-specification.md. Readout,
    background, noise and geometry are all optional and default disabled, so
    the default geometry is identity; nothing is enabled implicitly. Invalid
    input or nonfinite generation raises ValueError (wrong typed objects raise
    TypeError). Density overflow/max_count errors never cap or retry N. Shape,
    round count and max_count have no library upper bound: callers own memory
    and time limits (images are R arrays of ZYX×4 float64 before the cast).
    """
    if not isinstance(config, FormedSceneConfig) or not isinstance(codebook, Codebook):
        raise TypeError("expected FormedSceneConfig and Codebook")
    # Revalidate/copy mutable payloads instead of trusting an old constructor.
    codebook = Codebook(codebook.table, codebook.round_labels, codebook.channel_labels,
                        codebook.color_to_channel, codebook.encoding)
    if codebook.n_genes == 0:
        raise ValueError("require a nonempty codebook")
    for value in (config.dataset_version, config.sample_id, config.FOV_id, config.scene_key,
                  config.split, *codebook.round_labels, *codebook.channel_labels, *codebook.genes):
        _label(value)
    _integer(config.seed, "seed", 0, 2**64 - 1)
    if not isinstance(config.shape_zyx, (tuple, list)) or len(config.shape_zyx) != 3:
        raise ValueError("shape must be ZYX")
    for n in config.shape_zyx:
        _integer(n, "shape dimension", 1)
    shape = np.array(config.shape_zyx)
    if config.dtype not in ("float32", "float64", "uint8", "uint16"):
        raise ValueError("unsupported output dtype")
    _integer(config.max_count, "max_count", 0)
    namespace = _json([config.dataset_version, config.sample_id, config.FOV_id, "formed"])
    if metadata is None:
        metadata = ImageMetadata(namespace + "/reference")
    if not isinstance(metadata, ImageMetadata):
        raise TypeError("metadata must be ImageMetadata")
    metadata = ImageMetadata(**asdict(metadata))
    streams = {}

    def stream(component, entity=None, round_label=None, channel_label=None):
        descriptor = [_STREAM_NAMESPACE, config.split, config.seed, config.scene_key,
                      component, entity, round_label, channel_label]
        encoded = _json(descriptor)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        streams[encoded] = descriptor
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
        _integer(n, "count", 0)
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
    maps, effective_geometry = _prepare_geometry(config.geometry, shape, codebook.round_labels, stream)
    background, effective_background = _prepare_background(config.background, shape, len(images), stream)
    for component in background:
        component['frame_id'] = metadata.frame_id
    grid = np.moveaxis(np.indices(shape, dtype=np.float64), 0, -1)
    tissue = {}
    round_tables = []
    transforms = {}
    for r, label in enumerate(codebook.round_labels):
        mapping = maps[r]
        moved = _forward(points, mapping)
        reference_grid, inverse_diagnostics = _inverse(grid, mapping)
        tissue[label] = evaluate_background(background, reference_grid)
        intersects = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        # Sorting IDs fixes accumulation order independently of caller scheduling.
        for i in sorted(range(n), key=lambda i: ids[i]):
            _, sz, sl, e, theta = values[i]
            slices, kernel, truncated[i] = _kernel(shape, moved[i], sz, sl, e, theta)
            intersects[i] = kernel is not None and bool(np.any(kernel > 0))
            if intersects[i]:
                with np.errstate(over="ignore", invalid="ignore"):
                    images[label][slices] += kernel[..., None] * realized[i, :, r]
        kind = _kind(mapping)
        transform_id = _json([namespace, label, kind])
        destination = metadata.frame_id if kind == "identity" else _json([metadata.frame_id, label, "round"])
        transforms[transform_id] = dict(kind=kind, direction="reference_to_round", units="voxel_index",
                                        source_frame=metadata.frame_id, destination_frame=destination,
                                        round_label=label, **mapping, inverse=inverse_diagnostics,
                                        composition="q + d(q) + t", interpolation="analytic; no image interpolation",
                                        molecule_shape="fixed widths/angle; four-sigma support",
                                        background_sampling="analytic B(F_inverse(p)); untruncated")
        table = formed[["namespace", "amplicon_id", "z", "y", "x"]].copy()
        table[["z", "y", "x"]] = moved
        table["frame_id"] = pd.Series([destination]*n, dtype="string")
        table["round_label"] = pd.Series([label]*n, dtype="string")
        table["round_index"] = np.full(n, r, dtype=np.int64)
        table["transform_id"] = pd.Series([transform_id]*n, dtype="string")
        for key in ("dropped", "weakened", "lost"):
            table[key] = effects[key][:, r]
        table["emitting"] = np.any(realized[:, :, r] > 0, axis=1)
        table["center_in_bounds"] = ((moved >= 0) & (moved <= shape - 1)).all(axis=1)
        table["support_intersects"] = intersects
        table["support_truncated"] = truncated
        table["first_loss_round"] = pd.Series(first_loss, dtype="Int64")
        for key in ("trend_multiplier", "weak_multiplier"):
            table[key] = effects[key][:, r]
        round_tables.append(table)
    effective_noise = _observe(images, tissue, effective_background, config.noise,
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
    config_payload["geometry"]["type"] = "GeometryConfig"
    for key in ("axial_width", "lateral_width", "brightness"):
        config_payload["background"]["texture"][key]["type"] = "ScalarDistribution"
    for key in ("brightness", "axial_width", "lateral_width", "elongation", "angle"):
        config_payload[key]["type"] = "ScalarDistribution"
    codebook_payload = dict(rows=codebook.table.to_dict("records"), round_labels=list(codebook.round_labels),
                            channel_labels=list(codebook.channel_labels), color_to_channel=codebook.color_to_channel,
                            encoding=asdict(codebook.encoding))
    effective = dict(config_payload, count=n, placement="explicit" if coordinates is not None else config.placement,
                     abundance_probabilities=abundances.tolist(), readout=effective_readout,
                     background=effective_background, noise=effective_noise, geometry=effective_geometry,
                     effects_enabled=any(v for group in (effective_readout, effective_background, effective_noise, effective_geometry)
                                         for k, v in group.items() if k.endswith("_enabled")))
    provenance = dict(
        generator="starfinder.synthetic.generate_formed_scene", generator_version=_GENERATOR_VERSION,
        requested_config=config_payload, effective_config=effective, seed=config.seed,
        stream_scheme=dict(key=list(_STREAM_KEY), namespace=_STREAM_NAMESPACE,
                           encoding="compact UTF-8 JSON array", digest="SHA-256, big-endian integer",
                           bit_generator="PCG64", numpy_version=np.__version__,
                           streams=list(streams.values())),
        codebook=codebook_payload,
        image_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in images.items()},
        clipping_counts=clipping, transforms=transforms)
    return FormedScene(images, metadata, formed, pd.concat(round_tables, ignore_index=True),
                       intended, pre_mix, realized, ids, codebook.round_labels,
                       codebook.channel_labels, codebook, provenance)


def formed_scene_preset(name: str = "formed-small-v1") -> tuple[Codebook, FormedSceneConfig]:
    """Return the clean 3D (formed-small-v1) or singleton-Z (formed-z1-v1) preset.

    Both use N=8 uniform amplicons, seed 42 and every effect disabled.
    """
    if name not in ("formed-small-v1", "formed-z1-v1"):
        raise ValueError("unknown formed scene preset")
    codebook = Codebook(pd.DataFrame(dict(gene_id=["gene-A", "gene-B"], color_sequence=["123", "214"])),
                        ("round10", "round2", "round1"), ("ch02", "ch00", "ch03", "ch01"),
                        {"1": 1, "2": 0, "3": 3, "4": 2})
    return codebook, FormedSceneConfig(dataset_version=name, shape_zyx=(1 if name == "formed-z1-v1" else 8, 32, 32))
