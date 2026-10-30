"""Multi-FOV sample scene: one continuous scene, FOVs at true offsets, cellular truth.

See design/specs/synthetic/sample-scene.md. Every FOV is rendered by the unchanged formed
generator from a per-FOV AmpliconSceneConfig composed from public fields (molecules
at their FOV index positions, background components moved into the FOV frame).
Labels and the nuclear stain are closed-form functions of the sample position,
evaluated at voxel centres; no rendered image is interpolated.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import erfc

from starfinder._records import file_sha256
from starfinder.readout import Codebook
from starfinder.image import ImageMetadata
from starfinder.io import save_volume

from ._cellular import _labels
from ._common import ScalarDistribution, _distribution, _draw, _integer, _json, _label, _probabilities
from ._formed import (_GENERATOR_VERSION, _STREAM_KEY, _STREAM_NAMESPACE, AmpliconScene,
                      AmpliconSceneConfig, _plain, generate_amplicon_scene)
from ._geometry import _TERM_AXES, GeometryConfig, _prepare_geometry
from ._observation import (BackgroundConfig, NoiseConfig, TextureConfig, _noise, _observe,
                           _prepare_background, evaluate_background_translated)
from ._storage import save_amplicon_scene

#: Version of the sample presets; changing any preset value bumps it.
SAMPLE_PRESET_VERSION = "sample-presets-v1"
#: Version of the sample-level truth files; any change to a column or rule bumps it.
SAMPLE_TRUTH_VERSION = "1"

_ORDERS = ("column_major_down_right",)
_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
# Label points per power-diagram block.
_BLOCK = 1 << 18
_PROPERTIES = ("brightness", "width.axial", "width.lateral", "elongation", "angle")
# The new stream components: entity and draws of each (design/specs/synthetic/sample-scene.md).
_SAMPLE_COMPONENTS = {
    "sample.jitter": ("FOV ID", "u_z, u_y, u_x"),
    "sample.nucleus.axial": ("nucleus-<n>", "one value"),
    "sample.nucleus.lateral": ("nucleus-<n>", "one value"),
    "sample.nucleus.elongation": ("nucleus-<n>", "one value"),
    "sample.nucleus.angle": ("nucleus-<n>", "one value"),
    "sample.cell.placement": ("cell-<k>", "z, y, x per proposal; then phi for a binucleate seed"),
    "sample.cell.reach": ("cell-<k>", "one value"),
    "sample.molecule.count": ("cell-<k> or ambient", "one Poisson draw"),
    "sample.molecule.placement": ("molecule ID", "z, y, x per proposal"),
    **{f"sample.{name}": ("molecule ID", "one value") for name in ("identity", *_PROPERTIES)},
    **{f"sample.background.{name}": ("as background.{name}", "as background.{name}")
       for name in ("count", "placement", "width", "brightness")},
    "sample.stain.brightness": ("nucleus-<n>", "one value"),
    **{f"sample.stain.background.{name}": ("as background.{name}", "as background.{name}")
       for name in ("count", "placement", "width", "brightness")},
    **{f"stain.noise.{name}": ("none; scene key [scene_key, fov_id]", "as noise.{name}, stain channel")
       for name in ("dependent", "independent", "correlated")},
}


def _triple(value, name, *, integer=False):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        raw = np.asarray(None)
    if raw.shape != (3,) or raw.dtype.kind not in ("iu" if integer else "iuf"):
        raise ValueError(f"{name} must be a ZYX triple of {'integers' if integer else 'real numbers'}")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _real(value, name, low=0.0, *, strict=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < low or (strict and value == low):
        raise ValueError(f"{name} must be finite and {'>' if strict else '≥'} {low}")
    return value


@dataclass(frozen=True)
class AcquisitionConfig:
    """A nominal grid of FOVs acquired from one sample scene at true offsets.

    Parameters
    ----------
    columns : int
        Positive number of FOV grid columns.
    rows : int
        Positive number of FOV grid rows.
    order : str
        FOV traversal order, ``column_major_down_right``.
    fov_shape_zyx : tuple[int, int, int]
        FOV shape as positive voxel counts in ZYX order.
    spacing_zyx : tuple[float, float, float]
        Positive voxel spacing in µm in ZYX order.
    overlap : float or tuple[float, float]
        Fractional overlap between neighbouring FOVs, as a common value or a ``(y, x)`` pair.
    jitter : str
        Origin-jitter mode: ``uniform`` for random offsets or ``supplied`` for explicit offsets.
    jitter_max_zyx : tuple[float, float, float]
        Nonnegative absolute origin-jitter bounds in ZYX voxels.
    jitter_zyx : Mapping[str, tuple[float, float, float]] or None
        Explicit origin-jitter offsets in ZYX voxels keyed by FOV ID for ``supplied`` mode. ``None``
        leaves jitter to the ``uniform`` mode.
    fov_ids : tuple[str, ...] or None
        FOV identifiers in grid traversal order. ``None`` generates ``FOV_001``, ``FOV_002`` and
        subsequent numbered identifiers.

    Notes
    -----
    FOV ``k`` (IDs ``FOV_001``, ``FOV_002``, ... unless ``fov_ids`` is given) sits
    at column ``k // rows``, row ``k % rows``, with nominal origin
    ``(0, row · step_y, col · step_x)`` µm and ``step = (1 − overlap) · n · s``.
    The true origin is the nominal origin plus the jitter (voxels) times the
    spacing: ``"uniform"`` draws ``J · (2u − 1)`` per axis (J1), ``"supplied"``
    takes ``jitter_zyx`` (one triple per FOV ID, within the bound). With two or
    more columns (rows), ``overlap · n ≥ 2 · jitter_max + 1`` voxels in X (Y);
    ``jitter_max_z < n_z / 2``, and 0 when ``n_z`` is 1. Violations raise
    ValueError naming the axis or field. Instances are not hashable.
    """

    __hash__ = None

    columns: int = 2
    rows: int = 2
    order: str = "column_major_down_right"
    fov_shape_zyx: tuple[int, int, int] = (5, 36, 36)
    spacing_zyx: tuple[float, float, float] = (1.0, 0.5, 0.5)
    overlap: float | tuple[float, float] = 1 / 3
    jitter: str = "uniform"
    jitter_max_zyx: tuple[float, float, float] = (0.4, 1.5, 1.5)
    jitter_zyx: Mapping[str, tuple[float, float, float]] | None = None
    fov_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        _integer(self.columns, "columns", 1)
        _integer(self.rows, "rows", 1)
        if self.order not in _ORDERS:
            raise ValueError(f"order must be one of {list(_ORDERS)}; got {self.order!r}")
        shape = _triple(self.fov_shape_zyx, "fov_shape_zyx", integer=True)
        if (shape < 1).any():
            raise ValueError(f"fov_shape_zyx must hold positive integers; got {tuple(self.fov_shape_zyx)}")
        spacing = _triple(self.spacing_zyx, "spacing_zyx")
        if (spacing <= 0).any():
            raise ValueError(f"spacing_zyx must be finite and > 0; got {tuple(self.spacing_zyx)}")
        overlap = self._overlap()
        bound = _triple(self.jitter_max_zyx, "jitter_max_zyx")
        if (bound < 0).any():
            raise ValueError(f"jitter_max_zyx must be finite and ≥ 0; got {tuple(self.jitter_max_zyx)}")
        if shape[0] == 1 and bound[0] != 0:
            raise ValueError("z: jitter_max_zyx must be 0 in z when fov_shape_zyx has one plane (n_z = 1)")
        if not bound[0] < shape[0] / 2:
            raise ValueError(f"z: jitter_max_z {bound[0]} must be below n_z / 2 = {shape[0] / 2}")
        for axis, count, share, n, j in (("y", self.rows, overlap[0], shape[1], bound[1]),
                                         ("x", self.columns, overlap[1], shape[2], bound[2])):
            if count >= 2 and not share * n >= 2 * j + 1:
                raise ValueError(f"{axis}: the overlap rule overlap · n ≥ 2 · jitter_max + 1 fails "
                                 f"({share * n} < {2 * j + 1} voxels)")
        ids = self.ids()
        if self.jitter not in ("uniform", "supplied"):
            raise ValueError(f"jitter must be 'uniform' or 'supplied'; got {self.jitter!r}")
        if self.jitter == "uniform":
            if self.jitter_zyx is not None:
                raise ValueError("jitter_zyx requires jitter='supplied'")
            return
        if not isinstance(self.jitter_zyx, Mapping):
            raise ValueError("jitter_zyx must map every FOV ID to a ZYX triple with jitter='supplied'")
        unknown = sorted(set(self.jitter_zyx) - set(ids))
        if unknown:
            raise ValueError(f"jitter_zyx names FOVs that are not in the grid: {unknown}")
        for fov in ids:
            if fov not in self.jitter_zyx:
                raise ValueError(f"jitter_zyx misses FOV {fov}")
            values = _triple(self.jitter_zyx[fov], f"jitter_zyx[{fov!r}]")
            for axis, value, limit in zip("zyx", values, bound):
                if abs(value) > limit:
                    raise ValueError(f"jitter_zyx[{fov!r}] exceeds the {axis} bound: |{value}| > {limit}")

    def _overlap(self):
        values = self.overlap if isinstance(self.overlap, (tuple, list)) else (self.overlap, self.overlap)
        if len(values) != 2:
            raise ValueError("overlap must be one value or a (y, x) pair")
        result = []
        for axis, value in zip("yx", values):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"overlap ({axis}) must be a real number")
            if not (math.isfinite(value) and 0 <= value < 1):
                raise ValueError(f"overlap ({axis}) must satisfy 0 ≤ overlap < 1; got {value}")
            result.append(float(value))
        return tuple(result)

    def ids(self) -> tuple[str, ...]:
        """The FOV IDs in tile position table order.

        Returns
        -------
        tuple of str
            FOV identifiers in tile position table order.
        """
        count = self.columns * self.rows
        if self.fov_ids is None:
            return tuple(f"FOV_{k + 1:03d}" for k in range(count))
        ids = self.fov_ids
        if not isinstance(ids, tuple) or len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"fov_ids must be a tuple of {count} unique labels")
        for fov in ids:
            try:
                _label(fov)
            except ValueError:
                raise ValueError(f"fov_ids holds an invalid label: {fov!r}") from None
        return ids


def _lognormal(median, sd):
    return ScalarDistribution("lognormal", (math.log(median), sd))


@dataclass(frozen=True)
class CellsConfig:
    """Seeds, nucleus laws and territories of the minimal cellular scene (µm, radians).

    Parameters
    ----------
    nuclei : int
        Nonnegative count of ordinary cells with one nucleus.
    binucleate : int
        Nonnegative count of cells with two nuclei.
    empty : int
        Nonnegative count of cells without nuclei.
    nucleus_axial_radius : ScalarDistribution
        Distribution of the positive nucleus axial semi-axis in µm.
    nucleus_lateral_radius : ScalarDistribution
        Distribution of the positive nucleus lateral semi-axis in µm before elongation.
    nucleus_elongation : ScalarDistribution
        Distribution of the dimensionless lateral elongation ratio of a nucleus.
    nucleus_angle : ScalarDistribution
        Distribution of the nucleus orientation angle in the Y-X plane, in radians.
    empty_radius : float
        Positive virtual-site radius in µm for an empty cell.
    gap : float
        Positive minimum gap in µm between seed site spheres.
    reach : ScalarDistribution
        Distribution of the positive territory reach beyond the site radius, in µm.
    max_proposals : int
        Positive proposal budget per seed and per molecule.

    Notes
    -----
    Seeds are ordinary cells (one nucleus), binucleate cells (two) and empty
    cells (a virtual site of ``empty_radius``), placed in that order by random
    sequential addition with a minimum ``gap`` between site spheres. Cell
    territories are power-diagram cells limited by the site radius plus the
    cell's ``reach``. ``max_proposals`` bounds the proposals per seed and per
    molecule; exhausting it raises ValueError. Instances are not hashable.
    """

    __hash__ = None

    nuclei: int = 7
    binucleate: int = 1
    empty: int = 0
    nucleus_axial_radius: ScalarDistribution = field(default_factory=lambda: _lognormal(1.5, 0.1))
    nucleus_lateral_radius: ScalarDistribution = field(default_factory=lambda: _lognormal(2.5, 0.1))
    nucleus_elongation: ScalarDistribution = field(
        default_factory=lambda: ScalarDistribution("folded_lognormal", (0.0, 0.1)))
    nucleus_angle: ScalarDistribution = field(default_factory=lambda: ScalarDistribution("uniform", (0.0, math.pi)))
    empty_radius: float = 2.5
    gap: float = 1.0
    reach: ScalarDistribution = field(default_factory=lambda: _lognormal(3.0, 0.15))
    max_proposals: int = 10_000

    def __post_init__(self):
        for name in ("nuclei", "binucleate", "empty"):
            _integer(getattr(self, name), name, 0)
        _integer(self.max_proposals, "max_proposals", 1)
        _real(self.empty_radius, "empty_radius", strict=True)
        _real(self.gap, "gap", strict=True)
        nuclei = self.nucleus_ids()
        cells = tuple(f"cell-{k}" for k in range(1, self.nuclei + self.binucleate + self.empty + 1))
        for name, domain, ids in (("nucleus_axial_radius", "width.axial", nuclei),
                                  ("nucleus_lateral_radius", "width.lateral", nuclei),
                                  ("nucleus_elongation", "elongation", nuclei),
                                  ("nucleus_angle", "angle", nuclei), ("reach", "width.lateral", cells)):
            _checked_distribution(getattr(self, name), domain, ids, name)

    def nucleus_ids(self) -> tuple[str, ...]:
        """Stream entities of the nuclei, ``nucleus-1`` ... in seed order.

        Returns
        -------
        tuple of str
            Nucleus stream identifiers in seed order.
        """
        return tuple(f"nucleus-{n}" for n in range(1, self.nuclei + 2 * self.binucleate + 1))


def _checked_distribution(spec, domain, ids, name):
    try:
        _distribution(spec, domain, ids)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{name}: {exc}") from None


@dataclass(frozen=True)
class StainConfig:
    """Nuclear stain: a soft-edged ellipsoid per nucleus, then the observation stages.

    Parameters
    ----------
    channel_label : str
        Nuclear-stain channel label.
    brightness : ScalarDistribution
        Distribution of plateau intensity per nucleus.
    edge_width : float
        Positive Gaussian edge width in µm.
    background : BackgroundConfig
        Single-round, single-channel stain background settings in sample-grid index units.
    noise : NoiseConfig
        Residual stain-noise model drawn per FOV.

    Notes
    -----
    ``brightness`` is the plateau intensity per nucleus, ``edge_width`` the
    Gaussian edge ``w`` in µm. ``background`` is a one-round, one-channel
    BackgroundConfig in the sample grid's index units and ``noise`` the residual
    model, drawn per FOV from ``stain.noise.*`` streams. Instances are not hashable.
    """

    __hash__ = None

    channel_label: str = "DAPI"
    brightness: ScalarDistribution = field(default_factory=lambda: _lognormal(200, 0.2))
    edge_width: float = 0.5
    background: BackgroundConfig = field(
        default_factory=lambda: BackgroundConfig(baseline_enabled=True, baseline=((100.0,),)))
    noise: NoiseConfig = field(default_factory=lambda: NoiseConfig(
        dependent_enabled=True, model="poisson", alpha=1.0, independent_enabled=True, sigma=3.0))

    def __post_init__(self):
        try:
            _label(self.channel_label)
        except ValueError:
            raise ValueError("channel_label must be a nonempty NFC string") from None
        _real(self.edge_width, "edge_width", strict=True)
        spec = self.brightness
        ids = tuple(spec.values) if isinstance(spec, ScalarDistribution) and isinstance(spec.values, dict) else ()
        _checked_distribution(spec, "brightness", ids, "stain brightness")
        if not isinstance(self.background, BackgroundConfig):
            raise TypeError("stain background must be BackgroundConfig")
        _noise(self.noise)


def _template():
    return AmpliconSceneConfig(dataset_version="sample-scene-v1", sample_id="sample", scene_key="sample-scene-v1",
                               shape_zyx=(5, 36, 36), dtype="float32", accumulation="float64")


_POPULATION_FIELDS = ("count", "density", "coordinates", "amplicon_ids", "gene_ids", "spatial_weights",
                      "cluster_centers", "cluster_weights")


@dataclass(frozen=True)
class SampleSceneConfig:
    """One sample scene: a FOV template, the acquisition, cells, stain and molecules.

    Parameters
    ----------
    fov : AmpliconSceneConfig
        Amplicon-scene template for rendering and random-stream settings.
    acquisition : AcquisitionConfig
        FOV grid, voxel spacing and origin-jitter settings.
    cells : CellsConfig or None
        Cell seeds, nucleus laws and territory settings. ``None`` gives an ambient-molecule scene
        without cell labels.
    stain : StainConfig or None
        Nuclear-stain settings. ``None`` disables the stain image.
    molecules_per_cell : float
        Nonnegative Poisson mean of the number of molecules per cell.
    ambient_density : float
        Nonnegative ambient molecule density per µm³ of the scene domain.
    max_count : int
        Nonnegative upper guard on the total number of generated molecules.

    Notes
    -----
    ``fov`` supplies dtype, accumulation, property laws, abundances, readout,
    background (in the sample grid's index units), noise, per-round geometry,
    seed, split, scene key and sample ID; its ``shape_zyx`` must equal
    ``acquisition.fov_shape_zyx``, its population fields must stay at their
    defaults and its stochastic readout effects (loss, dropout, weakening)
    must be disabled. ``cells=None`` gives only ambient molecules and no
    labels; ``stain=None`` no stain image. ``molecules_per_cell`` is a Poisson
    mean per cell, ``ambient_density`` molecules per µm³ of the scene domain,
    ``max_count`` a guard on the molecules generated. Instances are not hashable.
    """

    __hash__ = None

    fov: AmpliconSceneConfig = field(default_factory=_template)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    cells: CellsConfig | None = field(default_factory=CellsConfig)
    stain: StainConfig | None = field(default_factory=StainConfig)
    molecules_per_cell: float = 8.0
    ambient_density: float = 0.002
    max_count: int = 100_000

    def __post_init__(self):
        template, acquisition = self.fov, self.acquisition
        if not isinstance(template, AmpliconSceneConfig):
            raise TypeError("fov must be AmpliconSceneConfig")
        if not isinstance(acquisition, AcquisitionConfig):
            raise TypeError("acquisition must be AcquisitionConfig")
        if self.cells is not None and not isinstance(self.cells, CellsConfig):
            raise TypeError("cells must be CellsConfig or None")
        if self.stain is not None and not isinstance(self.stain, StainConfig):
            raise TypeError("stain must be StainConfig or None")
        if tuple(template.shape_zyx) != tuple(acquisition.fov_shape_zyx):
            raise ValueError(f"fov.shape_zyx {tuple(template.shape_zyx)} must equal "
                             f"acquisition.fov_shape_zyx {tuple(acquisition.fov_shape_zyx)}")
        for name in _POPULATION_FIELDS:
            if getattr(template, name) is not None:
                raise ValueError(f"fov.{name} must be None: the sample scene owns the population")
        if template.placement != "uniform":
            raise ValueError("fov.placement must be 'uniform': the sample scene owns the population")
        for name in ("loss", "dropout", "weakening"):
            if getattr(template.readout, f"{name}_enabled", False):
                raise ValueError(f"fov.readout.{name}_enabled must be False: stochastic readout effects "
                                 "would be drawn per FOV")
        _real(self.molecules_per_cell, "molecules_per_cell")
        _real(self.ambient_density, "ambient_density")
        _integer(self.max_count, "max_count", 0)


@dataclass
class SampleScene:
    """A generated sample scene: per-FOV amplicon scenes, stains, labels and sample truth.

    Parameters
    ----------
    sample : str
        Sample identifier.
    fovs : dict of str to AmpliconScene
        Retained amplicon scenes keyed by FOV ID; empty with an ``on_fov`` callback.
    stains : dict of str to numpy.ndarray
        Retained ZYX nuclear stain images keyed by FOV ID; empty with an ``on_fov`` callback.
    fov_metadata : dict of str to ImageMetadata
        Reference-grid metadata keyed by FOV ID, with spacing and origin in micrometres.
    placement : pandas.DataFrame
        FOV placement truth, with sample positions in micrometres and jitter in voxels.
    positions : pandas.DataFrame
        Nominal FOV positions in sample coordinates, in micrometres.
    nuclei : pandas.DataFrame
        Nucleus geometry and ownership truth, with sample coordinates and radii in micrometres.
    cells : pandas.DataFrame
        Cell geometry, membership and population truth, with sample coordinates and reach in micrometres.
    molecules : pandas.DataFrame
        Molecule identities and cellular ownership, with sample coordinates in micrometres.
    cell_fovs : pandas.DataFrame
        Cell visibility and label coverage by FOV.
    molecule_fovs : pandas.DataFrame
        Molecule rendering and footprint membership by FOV, with ZYX coordinates in FOV voxels.
    sample_grid : ImageMetadata
        Sample label-grid metadata, with spacing and origin in micrometres.
    sample_labels : dict of str to numpy.ndarray
        ZYX sample-grid label images keyed by ``nucleus`` and ``cell``; empty without cells.
    fov_labels : dict of str to dict of str to numpy.ndarray
        Retained ZYX label images keyed by FOV ID, then ``nucleus`` and ``cell``; empty with an ``on_fov`` callback.
    record : dict
        Generation provenance, configuration and truth record.

    Notes
    -----
    ``fovs``, ``stains`` and ``fov_labels`` are keyed by FOV ID in tile position
    table order and are empty when the FOVs were handed to ``on_fov``. The
    tables are described in design/specs/synthetic/sample-scene.md ("Placement truth",
    "Truth schema"); ``record`` holds versions, the requested and effective
    configuration, the stream scheme and descriptors, the sample grid, the
    bounds of the scene domain and covered region, and per-FOV digests.
    """

    sample: str
    fovs: dict[str, AmpliconScene]
    stains: dict[str, np.ndarray]
    fov_metadata: dict[str, ImageMetadata]
    placement: pd.DataFrame
    positions: pd.DataFrame
    nuclei: pd.DataFrame
    cells: pd.DataFrame
    molecules: pd.DataFrame
    cell_fovs: pd.DataFrame
    molecule_fovs: pd.DataFrame
    sample_grid: ImageMetadata
    sample_labels: dict[str, np.ndarray]
    fov_labels: dict[str, dict[str, np.ndarray]]
    record: dict


def _layout(acquisition, grow=None):
    """Nominal grid, scene domain Ω, covered region K and the sample grid.

    ``grow`` (voxels per axis, the cellular scene's morphology bound T) widens Ω by ``T ⊙ s`` on each side and
    leaves K as it is; None is the sample scene's layout.
    """
    n = np.array(acquisition.fov_shape_zyx, dtype=np.int64)
    s = np.array(acquisition.spacing_zyx, dtype=np.float64)
    bound = np.array(acquisition.jitter_max_zyx, dtype=np.float64)
    share_y, share_x = acquisition._overlap()
    step_y = (1 - share_y) * acquisition.fov_shape_zyx[1] * float(s[1])
    step_x = (1 - share_x) * acquisition.fov_shape_zyx[2] * float(s[2])
    ids = acquisition.ids()
    grid = [(k // acquisition.rows, k % acquisition.rows) for k in range(len(ids))]
    nominal = np.array([(0.0, row * step_y, column * step_x) for column, row in grid], dtype=np.float64)
    top = nominal.max(axis=0)
    margin = bound if grow is None else bound + np.asarray(grow, dtype=np.float64)
    omega = (-s / 2 - margin * s, top + (n - .5) * s + margin * s)
    covered = (-s / 2 + bound * s, top + (n - .5) * s - bound * s)
    origin = omega[0] + s / 2
    shape = np.ceil((omega[1] - omega[0]) / s - 1e-9).astype(np.int64)
    return SimpleNamespace(n=n, s=s, bound=bound, ids=ids, grid=grid, step=(step_y, step_x), nominal=nominal,
                           omega=omega,
                           covered=covered, origin=origin, shape=shape, flat=bool(n[0] == 1))


def _streams(split, seed, scene_key, log):
    def stream(component, entity=None, round_label=None, channel_label=None):
        descriptor = [_STREAM_NAMESPACE, split, seed, scene_key, component, entity, round_label, channel_label]
        encoded = _json(descriptor)
        if log is not None:
            log[encoded] = descriptor
        return np.random.Generator(np.random.PCG64(int(hashlib.sha256(encoded.encode("utf-8")).hexdigest(), 16)))
    return stream


def _inside(points, box):
    return np.all((points >= box[0]) & (points < box[1]), axis=-1)


def _rho(points, centre, axial, lateral, elongation, angle):
    """Normalized ellipsoid radius of the nucleus model (amplicon-kernel convention)."""
    d = points - centre
    c, s = np.cos(angle), np.sin(angle)
    u = c * d[..., 1] + s * d[..., 2]
    v = -s * d[..., 1] + c * d[..., 2]
    return np.sqrt((d[..., 0] / axial) ** 2 + (u / (lateral * elongation)) ** 2 + (v / lateral) ** 2)


def _label_functions(points, sites, omega):
    """Cell label C and nucleus label N at (..., 3) sample positions in µm (rule L1).

    Sites of the cellular scene (design/specs/synthetic/cellular-scene.md) carry their nucleus shapes and territory
    rule and are labelled by that page's rule; sites of the real-appearance scene
    (design/specs/synthetic/real-appearance.md) carry their own label function, the placed objects in placement order; every
    other call is the rule L1 below.
    """
    appearance = getattr(sites, "appearance", None)
    if appearance is not None:
        return appearance(points)
    if getattr(sites, "shapes", None) is not None:
        return _labels(points, sites, omega, _BLOCK)
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    cells = np.zeros(len(flat), dtype=np.uint32)
    nuclei = np.zeros(len(flat), dtype=np.uint32)
    if sites is None or not len(sites.centres):
        return cells.reshape(shape), nuclei.reshape(shape)
    for start in range(0, len(flat), _BLOCK):
        block = flat[start:start + _BLOCK]
        distance = None
        for a in range(3):
            delta = block[:, a, None] - sites.centres[None, :, a]
            distance = delta * delta if distance is None else distance + delta * delta
        power = distance - sites.radii * sites.radii
        nearest = np.argmin(power, axis=1)
        rows = np.arange(len(block))
        hit = (_inside(block, omega)
               & (np.sqrt(distance[rows, nearest]) <= sites.radii[nearest] + sites.reach[nearest]))
        cells[start:start + len(block)] = np.where(hit, sites.cell[nearest], 0)
        nucleus = np.where(hit, sites.nucleus[nearest], 0)
        candidates = np.flatnonzero(nucleus)
        if len(candidates):
            index = nucleus[candidates].astype(np.int64) - 1
            rho = _rho(block[candidates], sites.nucleus_centre[index], sites.axial[index],
                       sites.lateral[index], sites.elongation[index], sites.angle[index])
            nucleus[candidates[rho > 1]] = 0
        nuclei[start:start + len(block)] = nucleus
    return cells.reshape(shape), nuclei.reshape(shape)


def _clear(centre, radius, accepted, gap):
    return all(np.linalg.norm(centre - other) >= radius + other_radius + gap for other, other_radius in accepted)


def _cellular_scene(config, layout, stream):
    """Draw nucleus sizes, place seeds by random sequential addition, draw reaches."""
    seeds = ["ordinary"] * config.nuclei + ["binucleate"] * config.binucleate + ["empty"] * config.empty
    nuclei_ids = config.nucleus_ids()
    laws = (("sample.nucleus.axial", config.nucleus_axial_radius),
            ("sample.nucleus.lateral", config.nucleus_lateral_radius),
            ("sample.nucleus.elongation", config.nucleus_elongation),
            ("sample.nucleus.angle", config.nucleus_angle))
    sizes = []
    for identity in nuclei_ids:
        values = [_draw(spec, component.split(".", 1)[1], identity,
                        lambda _name, entity, component=component: stream(component, entity))
                  for component, spec in laws]
        if not np.isfinite(values).all() or min(values[:2]) <= 0 or values[2] < 1:
            raise ValueError(f"{identity}: nonfinite or out-of-domain nucleus size")
        sizes.append(values)
    reaches = [_draw(config.reach, "reach", f"cell-{k}", lambda _name, entity: stream("sample.cell.reach", entity))
               for k in range(1, len(seeds) + 1)]
    if not np.isfinite(reaches).all() or min(reaches, default=1) <= 0:
        raise ValueError("nonfinite or nonpositive cell reach")
    lo, hi = layout.omega
    accepted, nuclei, cells = [], [], []
    next_nucleus = 0
    for k, kind in enumerate(seeds, start=1):
        own = [next_nucleus, next_nucleus + 1] if kind == "binucleate" else [next_nucleus] if kind == "ordinary" else []
        next_nucleus += len(own)
        radii = [max(sizes[i][0], sizes[i][1] * sizes[i][2]) for i in own] or [float(config.empty_radius)]
        rng = stream("sample.cell.placement", f"cell-{k}")
        for _ in range(config.max_proposals):
            centre = rng.uniform(lo, hi)
            if layout.flat:
                centre[0] = 0.0
            if kind == "binucleate":
                phi = rng.uniform(0.0, math.pi)
                separation = radii[0] + radii[1] + config.gap
                distance = separation
                while True:
                    half = (distance / 2) * np.array([0.0, math.sin(phi), math.cos(phi)])
                    centres = [centre + half, centre - half]
                    # Rounding of c ± h must not bring the pair below the gap rule.
                    if np.linalg.norm(centres[0] - centres[1]) >= radii[0] + radii[1] + config.gap:
                        break
                    distance = float(np.nextafter(distance, np.inf))
            else:
                centres = [centre]
            if all(_clear(c, r, accepted, config.gap) for c, r in zip(centres, radii)):
                break
        else:
            raise ValueError(f"cell-{k}: no proposal accepted after max_proposals={config.max_proposals} "
                             "(the scene domain is too crowded for the seeds)")
        accepted.extend(zip(centres, radii))
        for i, c in zip(own, centres):
            axial, lateral, elongation, angle = sizes[i]
            nuclei.append(dict(nucleus_id=i + 1, cell_id=k, centre=c, axial=axial, lateral=lateral,
                               elongation=elongation, angle=angle, site_radius=max(axial, lateral * elongation)))
        cells.append(dict(cell_id=k, kind=kind, anchor=centres[0], n_nuclei=len(own), reach=reaches[k - 1],
                          sites=list(zip(centres, radii))))
    site_cells = [cell["cell_id"] for cell in cells for _ in cell["sites"]]
    sites = SimpleNamespace(
        centres=np.array([c for cell in cells for c, _ in cell["sites"]], dtype=np.float64).reshape(-1, 3),
        radii=np.array([r for cell in cells for _, r in cell["sites"]], dtype=np.float64),
        reach=np.array([cells[k - 1]["reach"] for k in site_cells], dtype=np.float64),
        cell=np.array(site_cells, dtype=np.uint32),
        nucleus=np.array([n["nucleus_id"] for n in nuclei] + [0] * config.empty, dtype=np.uint32),
        nucleus_centre=np.array([n["centre"] for n in nuclei], dtype=np.float64).reshape(-1, 3),
        axial=np.array([n["axial"] for n in nuclei]), lateral=np.array([n["lateral"] for n in nuclei]),
        elongation=np.array([n["elongation"] for n in nuclei]), angle=np.array([n["angle"] for n in nuclei]))
    return nuclei, cells, sites


def _molecules(config, layout, codebook, cells, sites, stream):
    """Cell and ambient populations, positions in µm, genes and properties."""
    lo, hi = layout.omega
    plan = []
    if cells is not None:
        for cell in cells:
            count = int(stream("sample.molecule.count", f"cell-{cell['cell_id']}").poisson(config.molecules_per_cell))
            plan.append((cell, count))
    volume = float(np.prod(hi - lo))
    ambient = int(stream("sample.molecule.count", "ambient").poisson(config.ambient_density * volume))
    total = sum(count for _, count in plan) + ambient
    if total > config.max_count:
        raise ValueError(f"max_count: {total} molecules would be generated in the scene domain, "
                         f"above max_count={config.max_count}")
    ids, population, positions = [], [], []
    limit = config.cells.max_proposals if config.cells is not None else 1
    for cell, count in plan:
        k = cell["cell_id"]
        box_lo = np.maximum(np.min([c - r - cell["reach"] for c, r in cell["sites"]], axis=0), lo)
        box_hi = np.minimum(np.max([c + r + cell["reach"] for c, r in cell["sites"]], axis=0), hi)
        for j in range(count):
            identity = f"c{k}-m{j}"
            rng = stream("sample.molecule.placement", identity)
            for _ in range(limit):
                point = rng.uniform(box_lo, box_hi)
                if layout.flat:
                    point[0] = 0.0
                if _label_functions(point[None], sites, layout.omega)[0][0] == k:
                    break
            else:
                raise ValueError(f"{identity}: no proposal accepted in cell {k} after max_proposals={limit}")
            ids.append(identity)
            population.append("cell")
            positions.append(point)
    for j in range(ambient):
        identity = f"a-m{j}"
        point = stream("sample.molecule.placement", identity).uniform(lo, hi)
        if layout.flat:
            point[0] = 0.0
        ids.append(identity)
        population.append("ambient")
        positions.append(point)
    template = config.fov
    specs = (template.brightness, template.axial_width, template.lateral_width, template.elongation, template.angle)
    for name, spec in zip(_PROPERTIES, specs):
        _checked_distribution(spec, name, tuple(ids), f"fov.{name}")
    # The codebook lookups are built once here, not per molecule.
    book_genes, n_genes = codebook.genes, codebook.n_genes
    abundances = _probabilities(template.abundances if template.abundances is not None
                                else np.ones(n_genes), n_genes, "abundances")
    genes, values = [], np.empty((len(ids), 5), dtype=np.float64)
    for i, identity in enumerate(ids):
        genes.append(book_genes[stream("sample.identity", identity).choice(n_genes, p=abundances)])
        values[i] = [_draw(spec, name, identity, lambda component, entity: stream("sample." + component, entity))
                     for name, spec in zip(_PROPERTIES, specs)]
    if not np.isfinite(values).all() or (values[:, 1:3] <= 0).any():
        raise ValueError("nonfinite or out-of-domain molecule properties")
    return (tuple(ids), population, np.array(positions, dtype=np.float64).reshape(-1, 3), genes, values,
            abundances)


def _kernel_extent(values):
    """Largest axis-aligned one-sigma kernel extent (voxels) over the molecules."""
    if not len(values):
        return np.zeros(3)
    _, sz, sl, e, theta = values.T
    c, s = np.cos(theta), np.sin(theta)
    return np.array([sz.max(), (sl * np.hypot(e * c, s)).max(), (sl * np.hypot(e * s, c)).max()])


def _displacement_bound(maps, shape, base):
    """Per-axis bound (voxels) of the per-round displacement within the rendering margin.

    Translation and local terms are bounded everywhere by their coefficients.
    Affine and polynomial terms grow away from the grid, so their bound is
    evaluated on the grid box grown by the margin itself, iterated to a fixed point.
    """
    fixed = np.zeros(3)
    for mapping in maps:
        vectors = np.abs(np.asarray(mapping["vectors_zyx"], dtype=np.float64)).reshape(-1, 3)
        fixed = np.maximum(fixed, np.abs(mapping["translation_zyx"]) + vectors.sum(axis=0))
    linear = [m for m in maps if "affine_zyx" in m]
    if not linear:
        return fixed
    extent = (np.asarray(shape, dtype=np.float64) - 1) / 2
    half = max(float(extent.max()), 1.0)
    bound = fixed
    for _ in range(200):
        reach = extent + .5 + base + bound
        monomials = np.array([reach[a] * reach[b] for a, b in _TERM_AXES]) / half ** 2
        extra = np.zeros(3)
        for mapping in linear:
            extra = np.maximum(extra, np.abs(np.asarray(mapping["affine_zyx"])) @ reach
                               + np.abs(np.asarray(mapping["polynomial_zyx"])) @ monomials)
        update = fixed + extra
        if np.all(update - bound <= 1e-9 * (1 + bound)):
            return update * (1 + 1e-6) + 1e-6
        bound = update
    raise ValueError("the per-round displacement bound of this geometry does not converge; "
                     "use a translation or local geometry")


def _fov_background(background, components, delta, sample_shape, fov_shape, fov):
    """The template background moved into one FOV's frame (route R1)."""
    changes = {}
    texture = [c for c in components if c["kind"] == "texture"]
    regions = [c for c in components if c["kind"] == "region"]
    for component in components:
        if component["kind"] != "gradient":
            continue
        slopes = np.asarray(component["slopes_zyx"], dtype=np.float64)
        intercept, fov_slopes = component["intercept"], np.zeros(3)
        for a in range(3):
            if sample_shape[a] > 1:
                intercept = intercept + slopes[a] * delta[a] / (sample_shape[a] - 1)
                if fov_shape[a] > 1:
                    fov_slopes[a] = slopes[a] * (fov_shape[a] - 1) / (sample_shape[a] - 1)
        if not intercept >= 0:
            raise ValueError(f"{fov}: the background gradient intercept re-expressed for this FOV is "
                             f"{intercept}, below 0; the sample scene needs a nonnegative FOV intercept")
        changes.update(gradient_intercept=float(intercept), gradient_slopes_zyx=tuple(fov_slopes.tolist()))
    if regions:
        changes.update(region_centers=tuple(tuple((np.asarray(c["center_zyx"]) - delta).tolist()) for c in regions),
                       region_sigma_zyx=tuple(tuple(c["sigma_zyx"]) for c in regions),
                       region_heights=tuple(c["height"] for c in regions))
    if background.texture_enabled:
        # Always supplied, even when empty, so no FOV draws a texture of its own.
        supplied = lambda key, index: ScalarDistribution(  # noqa: E731
            "supplied", (), {c["component_id"]: (c[key] if index is None else c[key][index]) for c in texture})
        changes.update(texture=TextureConfig(
            coordinates=tuple(tuple((np.asarray(c["center_zyx"]) - delta).tolist()) for c in texture),
            max_count=max(background.texture.max_count, len(texture)),
            axial_width=supplied("sigma_zyx", 0), lateral_width=supplied("sigma_zyx", 1),
            brightness=supplied("height", None)))
    return replace(background, **changes)


def _fov_labels(points, sites, omega):
    cell, nucleus = _label_functions(points, sites, omega)
    return {"nucleus": nucleus, "cell": cell}


def _cell_rows(labels, cells, fov, true, spacing, shape):
    """cell_fovs rows of one FOV: label voxels, anchor in footprint, edge flags."""
    rows = []
    for cell in cells:
        mask = labels == cell["cell_id"]
        count = int(np.count_nonzero(mask))
        anchor = (cell["anchor"] - true) / spacing
        in_footprint = bool(np.all((anchor >= -.5) & (anchor < shape - .5)))
        if count or in_footprint:
            edge = bool(mask[:, [0, -1], :].any() or mask[:, :, [0, -1]].any())
            z_edge = bool(shape[0] > 1 and mask[[0, -1]].any())
            rows.append((cell["cell_id"], fov, in_footprint, count, edge, z_edge))
    return rows


def _stain_signal(nuclei, brightness, width, true, spacing, shape):
    """Pre-noise stain S at the FOV voxel centres true + p ⊙ s (option St1), float64."""
    plane = np.zeros(tuple(shape), dtype=np.float64)
    for nucleus, amplitude in zip(nuclei, brightness):
        semi = np.array([nucleus["axial"], nucleus["lateral"] * nucleus["elongation"], nucleus["lateral"]])
        cutoff = 1 + 4 * width / semi.min()
        # The cut ellipsoid lies in the sphere of radius cutoff · max semi-axis; one voxel of slack.
        centre = (nucleus["centre"] - true) / spacing
        reach = cutoff * semi.max() / spacing + 1
        low = np.clip(np.floor(centre - reach), 0, shape).astype(np.int64)
        high = np.clip(np.ceil(centre + reach) + 1, 0, shape).astype(np.int64)
        if (high <= low).any():
            continue
        index = np.moveaxis(np.indices(tuple(high - low), dtype=np.float64), 0, -1) + low
        rho = _rho(true + index * spacing, nucleus["centre"], nucleus["axial"], nucleus["lateral"],
                   nucleus["elongation"], nucleus["angle"])
        mean = float(np.prod(semi) ** (1 / 3))
        value = amplitude * .5 * erfc((rho - 1) * mean / (math.sqrt(2) * width))
        plane[tuple(slice(a, b) for a, b in zip(low, high))] += np.where(rho <= cutoff, value, 0.0)
    return plane


def _cast(plane, dtype):
    if dtype.startswith("uint"):
        np.rint(plane, out=plane)
        np.clip(plane, 0, np.iinfo(dtype).max, out=plane)
    with np.errstate(over="ignore"):
        image = plane.astype(dtype)
    if not np.isfinite(image).all():
        raise ValueError("stain output dtype overflow")
    return image


def _config_payload(config):
    payload = _plain(asdict(config))
    payload["type"] = "SampleSceneConfig"
    payload["fov"]["type"] = "AmpliconSceneConfig"
    payload["acquisition"]["type"] = "AcquisitionConfig"
    if config.acquisition.jitter_zyx is not None:
        payload["acquisition"]["jitter_zyx"] = {k: [float(v) for v in value]
                                               for k, value in config.acquisition.jitter_zyx.items()}
    for key in ("cells", "stain"):
        if payload[key] is not None:
            payload[key]["type"] = type(getattr(config, key)).__name__
    return payload


def _codebook_payload(codebook):
    return dict(rows=_plain(codebook.table.to_dict("records")), round_labels=list(codebook.round_labels),
                channel_labels=list(codebook.channel_labels), color_to_channel=codebook.color_to_channel,
                encoding=asdict(codebook.encoding))


def generate_sample_scene(codebook: Codebook, *, config: SampleSceneConfig = SampleSceneConfig(),
                          on_fov: Callable | None = None) -> SampleScene:
    """Generate one sample scene and acquire it as a grid of FOVs at true offsets.

    Parameters
    ----------
    codebook : Codebook
        Gene identities, color sequences and acquisition round and channel labels.
    config : SampleSceneConfig
        Sample acquisition, cellular population and rendering settings.
    on_fov : callable or None
        Callback receiving ``(fov_id, scene, stain, labels)`` as a sample FOV completes.
        None retains the sample FOV scenes, stains and labels in the result.

    Returns
    -------
    SampleScene
        Generated scene and its truth records.

    Notes
    -----
    The scene (seeds, nuclei, cell territories, molecules, background) is
    generated once in continuous sample coordinates over the scene domain Ω;
    each FOV is rendered by ``generate_amplicon_scene`` at its true origin, with
    its own per-round geometry and noise streams (scene key ``[scene_key,
    fov_id]``), and receives every molecule within the rendering margin. Labels
    and the stain are evaluated at voxel centres. With ``on_fov(fov_id, scene,
    stain, labels)`` each FOV is handed over when complete and not retained:
    ``fovs``, ``stains`` and ``fov_labels`` are then empty and the tables are
    complete. Invalid input, crowding beyond ``max_proposals`` and counts above
    ``max_count`` raise ValueError; nothing is written to disk.
    """
    if not isinstance(config, SampleSceneConfig):
        raise TypeError("config must be SampleSceneConfig")
    return _generate(codebook, config, on_fov)


def _generate(codebook, config, on_fov, cellular=_cellular_scene, molecules=_molecules, grow=None):
    """The body of generate_sample_scene; the cellular scene passes its own placement and molecule draws and, with a
    morphology round, its translation bound ``grow`` (voxels), by which the scene domain grows (``_layout``)."""
    if not isinstance(codebook, Codebook):
        raise TypeError("codebook must be Codebook")
    if on_fov is not None and not callable(on_fov):
        raise TypeError("on_fov must be callable or None")
    template, acquisition = config.fov, config.acquisition
    for value in (template.split, template.scene_key, template.sample_id, template.dataset_version):
        _label(value)
    _integer(template.seed, "seed", 0, 2**64 - 1)
    layout = _layout(acquisition, grow)
    sample = template.sample_id
    log = {}
    stream = _streams(template.split, template.seed, template.scene_key, log)
    n, s = layout.n, layout.s

    jitter = np.empty((len(layout.ids), 3), dtype=np.float64)
    for k, fov in enumerate(layout.ids):
        if acquisition.jitter == "uniform":
            jitter[k] = layout.bound * (2 * stream("sample.jitter", fov).random(3) - 1) + 0.0
        else:
            jitter[k] = np.asarray(acquisition.jitter_zyx[fov], dtype=np.float64) + 0.0
    true = layout.nominal + jitter * s
    delta = (true - layout.origin) / s

    nuclei = cells = sites = None
    if config.cells is not None:
        nuclei, cells, sites = cellular(config.cells, layout, stream)
    ids, population, sigma, genes, values, abundances = molecules(config, layout, codebook, cells, sites, stream)
    cell_of, nucleus_of = _label_functions(sigma, sites, layout.omega)
    in_sample = _inside(sigma, layout.covered)

    rounds, channels = len(codebook.round_labels), len(codebook.channel_labels)
    components, background = _prepare_background(
        template.background, layout.shape, rounds,
        lambda component, *keys: stream("sample." + component, *keys), channels)
    stain = config.stain
    stain_brightness, stain_components, stain_background, stain_noise = [], [], None, None
    if stain is not None:
        stain_components, stain_background = _prepare_background(
            stain.background, layout.shape, 1, lambda component, *keys: stream("sample.stain." + component, *keys), 1)
        stain_noise = _noise(stain.noise)
        if nuclei:
            identities = tuple(f"nucleus-{x['nucleus_id']}" for x in nuclei)
            _checked_distribution(stain.brightness, "brightness", identities, "stain brightness")
            stain_brightness = [_draw(stain.brightness, "brightness", identity,
                                      lambda _name, entity: stream("sample.stain.brightness", entity))
                                for identity in identities]

    reference = template.geometry.reference_round or codebook.round_labels[0]
    geometry = replace(template.geometry, reference_round=reference)
    base = 4 * _kernel_extent(values) + 1
    plans = []
    for k, fov in enumerate(layout.ids):
        maps, _ = _prepare_geometry(geometry, n, codebook.round_labels,
                                    _streams(template.split, template.seed, _json([template.scene_key, fov]), None))
        bound = _displacement_bound(maps, n, base)
        fov_background = _fov_background(template.background, components, delta[k], layout.shape, n, fov)
        plans.append((bound, base + bound, fov_background))

    sample_points = layout.origin + np.moveaxis(np.indices(tuple(layout.shape), dtype=np.float64), 0, -1) * s
    sample_labels = {}
    if cells is not None:
        sample_cells, sample_nuclei = _label_functions(sample_points, sites, layout.omega)
        sample_labels = {"nucleus": sample_nuclei, "cell": sample_cells}
    del sample_points

    fov_points = np.moveaxis(np.indices(tuple(n), dtype=np.float64), 0, -1)
    fovs, stains, fov_labels, fov_metadata, fov_records = {}, {}, {}, {}, {}
    molecule_rows, cell_rows = [], []
    for k, fov in enumerate(layout.ids):
        bound, margin, fov_background = plans[k]
        p = (sigma - true[k]) / s
        chosen = np.flatnonzero(np.all((p >= -.5 - margin) & (p <= n - .5 + margin), axis=1))
        chosen_ids = tuple(ids[i] for i in chosen)
        supplied = lambda j: ScalarDistribution(  # noqa: E731
            "supplied", (), {ids[i]: float(values[i, j]) for i in chosen})
        fov_config = replace(
            template, FOV_id=fov, scene_key=_json([template.scene_key, fov]),
            coordinates=tuple(tuple(row) for row in p[chosen].tolist()), amplicon_ids=chosen_ids,
            gene_ids={ids[i]: genes[i] for i in chosen}, abundances=None,
            brightness=supplied(0), axial_width=supplied(1), lateral_width=supplied(2),
            elongation=supplied(3), angle=supplied(4), max_count=max(template.max_count, len(chosen)),
            background=fov_background, geometry=geometry)
        namespace = _json([template.dataset_version, sample, fov, "formed"])
        metadata = ImageMetadata(namespace + "/reference", tuple(s.tolist()), (0.0, 0.0, 0.0), _IDENTITY, "µm")
        fov_metadata[fov] = metadata
        scene = generate_amplicon_scene(codebook, config=fov_config, metadata=metadata)

        labels = {} if cells is None else _fov_labels(true[k] + fov_points * s, sites, layout.omega)
        image = None
        if stain is not None:
            plane = _stain_signal(nuclei or [], stain_brightness, stain.edge_width, true[k], s, n)
            tissue = (evaluate_background_translated(stain_components, n, -delta[k], np.float64)[0]
                      if stain_components else None)
            fov_stream = _streams(template.split, template.seed, _json([template.scene_key, fov]), log)
            _observe(plane, tissue, stain_background, 0, 0, stain_noise, None, stain.channel_label,
                     lambda component, entity=None, round_label=None, channel_label=None:
                     fov_stream("stain." + component, entity, None, channel_label))
            if not np.isfinite(plane).all():
                raise ValueError("nonfinite rendered stain")
            image = _cast(plane, template.dtype)
            del plane, tissue

        footprint = np.all((p[chosen] >= -.5) & (p[chosen] < n - .5), axis=1)
        voxel = np.clip(np.floor(p[chosen] + .5).astype(np.int64), 0, n - 1)
        for row, i in enumerate(chosen):
            cell_label = nucleus_label = agrees = None
            if cells is not None and footprint[row]:
                at = tuple(voxel[row])
                cell_label, nucleus_label = int(labels["cell"][at]), int(labels["nucleus"][at])
                agrees = cell_label == int(cell_of[i])
            molecule_rows.append((ids[i], fov, bool(in_sample[i]), *p[i], bool(footprint[row]),
                                  cell_label, nucleus_label, agrees, row))
        if cells is not None:
            cell_rows.extend(_cell_rows(labels["cell"], cells, fov, true[k], s, n))
        fov_records[fov] = dict(
            grid_column=layout.grid[k][0], grid_row=layout.grid[k][1], delta_zyx=delta[k].tolist(),
            displacement_bound_zyx=bound.tolist(), margin_zyx=margin.tolist(), rendered=len(chosen),
            image_sha256=dict(scene.provenance["image_sha256"]),
            stain_sha256=None if image is None else hashlib.sha256(image.tobytes()).hexdigest(),
            label_sha256={t: hashlib.sha256(a.tobytes()).hexdigest() for t, a in labels.items()})
        if on_fov is None:
            fovs[fov] = scene
            if image is not None:
                stains[fov] = image
            if labels:
                fov_labels[fov] = labels
        else:
            on_fov(fov, scene, image, labels)
        # Nothing of a handed-over FOV stays referenced once the next FOV starts.
        del scene, image, labels

    return _assemble(config, codebook, layout, sample, jitter, true, ids, population, sigma, genes, values,
                     abundances, in_sample, cell_of, nucleus_of, nuclei, cells, stain_brightness, sample_labels,
                     molecule_rows, cell_rows, fovs, stains, fov_labels, fov_metadata, fov_records, components,
                     background, stain_components, stain_background, stain_noise, base, log)


def _assemble(config, codebook, layout, sample, jitter, true, ids, population, sigma, genes, values, abundances,
              in_sample, cell_of, nucleus_of, nuclei, cells, stain_brightness, sample_labels, molecule_rows,
              cell_rows, fovs, stains, fov_labels, fov_metadata, fov_records, components, background,
              stain_components, stain_background, stain_noise, base, log):
    template = config.fov
    n, s = layout.n, layout.s
    nominal = layout.nominal
    correction = true - nominal
    g1 = nominal + correction - correction.mean(axis=0)
    positions = pd.DataFrame(dict(fov_id=list(layout.ids), nominal_z=nominal[:, 0], nominal_y=nominal[:, 1],
                                  nominal_x=nominal[:, 2]))
    placement = pd.DataFrame(dict(
        fov_id=list(layout.ids),
        grid_column=np.array([c for c, _ in layout.grid], dtype=np.int64),
        grid_row=np.array([r for _, r in layout.grid], dtype=np.int64),
        **{f"nominal_{a}": nominal[:, i] for i, a in enumerate("zyx")},
        **{f"true_origin_{a}": true[:, i] for i, a in enumerate("zyx")},
        **{f"true_correction_{a}": correction[:, i] for i, a in enumerate("zyx")},
        **{f"jitter_{a}_vox": jitter[:, i] for i, a in enumerate("zyx")},
        **{f"true_origin_g1_{a}": g1[:, i] for i, a in enumerate("zyx")},
        **{f"shape_{a}": np.full(len(layout.ids), n[i], dtype=np.int64) for i, a in enumerate("zyx")},
        **{f"spacing_{a}": np.full(len(layout.ids), s[i]) for i, a in enumerate("zyx")}))

    molecule_fovs = pd.DataFrame(molecule_rows, columns=[
        "amplicon_id", "fov_id", "in_sample", "z", "y", "x", "in_footprint", "fov_cell_label",
        "fov_nucleus_label", "label_agrees", "csv_row"])
    molecule_fovs = molecule_fovs.astype(dict(
        amplicon_id=object, fov_id=object, in_sample=bool, z=np.float64, y=np.float64, x=np.float64,
        in_footprint=bool, fov_cell_label="UInt32", fov_nucleus_label="UInt32", label_agrees="boolean",
        csv_row=np.int64))
    seen = molecule_fovs[molecule_fovs.in_footprint]
    seen_by = seen.groupby("amplicon_id", sort=False).fov_id.agg(list).to_dict()
    has_cells = cells is not None
    true_cell = pd.array([int(c) if c else None for c in cell_of] if has_cells else [None] * len(ids),
                         dtype="UInt32")
    true_nucleus = pd.array([int(v) if v else None for v in nucleus_of] if has_cells else [None] * len(ids),
                            dtype="UInt32")
    cell_in_sample = ({cell["cell_id"]: bool(_inside(cell["anchor"], layout.covered)) for cell in cells}
                      if has_cells else {})
    compartment = (["nucleus" if v else "cytoplasm" if c else "outside" for c, v in zip(cell_of, nucleus_of)]
                   if has_cells else [None] * len(ids))
    # Built once, not per molecule; without molecules it is never read (it raises for repeated genes).
    gene_to_seq = codebook.gene_to_seq if len(genes) else {}
    molecules = pd.DataFrame(dict(
        amplicon_id=list(ids), population=population, in_sample=in_sample.astype(bool), gene_id=genes,
        color_sequence=[gene_to_seq[g] for g in genes],
        sample_z=sigma[:, 0], sample_y=sigma[:, 1], sample_x=sigma[:, 2],
        **{key: values[:, j] for j, key in enumerate(("A", "sz", "sl", "e", "theta"))},
        true_cell_id=true_cell,
        true_cell_in_sample=pd.array([cell_in_sample[int(c)] if c else None for c in cell_of]
                                     if has_cells else [None] * len(ids), dtype="boolean"),
        true_nucleus_id=true_nucleus, compartment=compartment,
        fov_ids=[";".join(seen_by.get(i, [])) for i in ids],
        n_fovs=np.array([len(seen_by.get(i, [])) for i in ids], dtype=np.int64)))

    cell_fovs = pd.DataFrame([(key, fov, cell_in_sample[key], anchor, count, edge, z_edge)
                              for key, fov, anchor, count, edge, z_edge in cell_rows],
                             columns=["cell_id", "fov_id", "in_sample", "anchor_in_footprint", "n_voxels",
                                      "fov_edge", "z_edge"])
    cell_fovs = cell_fovs.astype(dict(cell_id=np.uint32, fov_id=object, in_sample=bool, anchor_in_footprint=bool,
                                      n_voxels=np.int64, fov_edge=bool, z_edge=bool))
    order = {fov: k for k, fov in enumerate(layout.ids)}
    cell_fovs = cell_fovs.sort_values(["cell_id", "fov_id"], key=lambda c: c.map(order) if c.name == "fov_id" else c,
                                      kind="stable").reset_index(drop=True)

    nucleus_columns = ["nucleus_id", "cell_id", "in_sample", "centre_z", "centre_y", "centre_x", "axial_radius",
                       "lateral_radius", "elongation", "angle", "site_radius", "stain_brightness"]
    cell_columns = ["cell_id", "kind", "in_sample", "anchor_z", "anchor_y", "anchor_x", "n_nuclei", "reach",
                    "centroid_z", "centroid_y", "centroid_x", "n_voxels", "n_molecules", "fov_ids", "n_fovs",
                    "label_fov_ids"]
    if has_cells:
        brightness = stain_brightness or [np.nan] * len(nuclei)
        nuclei_table = pd.DataFrame([(x["nucleus_id"], x["cell_id"], cell_in_sample[x["cell_id"]], *x["centre"],
                                      x["axial"], x["lateral"], x["elongation"], x["angle"], x["site_radius"], b)
                                     for x, b in zip(nuclei, brightness)], columns=nucleus_columns)
        labels = sample_labels["cell"].reshape(-1).astype(np.int64)
        size = len(cells) + 1
        voxels = np.bincount(labels, minlength=size)
        centre = np.moveaxis(np.indices(tuple(layout.shape), dtype=np.float64), 0, -1).reshape(-1, 3)
        sums = np.stack([np.bincount(labels, weights=layout.origin[a] + centre[:, a] * s[a], minlength=size)
                         for a in range(3)], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            centroids = sums / voxels[:, None]
        counts = pd.Series(cell_of[cell_of > 0]).value_counts().to_dict()
        anchored = cell_fovs[cell_fovs.anchor_in_footprint].groupby("cell_id").fov_id.agg(list).to_dict()
        labelled = cell_fovs[cell_fovs.n_voxels > 0].groupby("cell_id").fov_id.agg(list).to_dict()
        cells_table = pd.DataFrame([
            (cell["cell_id"], cell["kind"], cell_in_sample[cell["cell_id"]], *cell["anchor"], cell["n_nuclei"],
             cell["reach"], *(centroids[cell["cell_id"]] if voxels[cell["cell_id"]] else (np.nan,) * 3),
             int(voxels[cell["cell_id"]]), int(counts.get(cell["cell_id"], 0)),
             ";".join(anchored.get(cell["cell_id"], [])), len(anchored.get(cell["cell_id"], [])),
             ";".join(labelled.get(cell["cell_id"], [])))
            for cell in cells], columns=cell_columns)
    else:
        nuclei_table = pd.DataFrame(columns=nucleus_columns)
        cells_table = pd.DataFrame(columns=cell_columns)
    nuclei_table = nuclei_table.astype(dict(nucleus_id=np.uint32, cell_id=np.uint32, in_sample=bool,
                                            **{c: np.float64 for c in nucleus_columns[3:]}))
    cells_table = cells_table.astype(dict(
        cell_id=np.uint32, kind=object, in_sample=bool, n_nuclei=np.int64, n_voxels=np.int64,
        n_molecules=np.int64, fov_ids=object, n_fovs=np.int64, label_fov_ids=object,
        **{c: np.float64 for c in ("anchor_z", "anchor_y", "anchor_x", "reach", "centroid_z", "centroid_y",
                                   "centroid_x")}))

    sample_grid = ImageMetadata(f"sample:{sample}", tuple(s.tolist()), tuple(layout.origin.tolist()), _IDENTITY, "µm")
    flat = lambda array: [float(v) for v in array]  # noqa: E731
    effective = dict(
        type="SampleSceneConfig", sample=sample, fov_ids=list(layout.ids),
        reference_round=template.geometry.reference_round or codebook.round_labels[0],
        nominal_step_yx=flat(layout.step),
        molecules=dict(generated=len(ids), cell=population.count("cell"), ambient=population.count("ambient"),
                       sample=int(in_sample.sum()), abundance_probabilities=flat(abundances)),
        cells=None if not has_cells else dict(seeds=len(cells), nuclei=len(nuclei),
                                              sample=int(sum(cell_in_sample.values()))),
        kernel_extent_zyx=flat((base - 1) / 4), background=background,
        stain=None if config.stain is None else dict(background=stain_background, noise=stain_noise))
    record = dict(
        generator="starfinder.synthetic.generate_sample_scene", sample_truth_version=SAMPLE_TRUTH_VERSION,
        sample_preset_version=SAMPLE_PRESET_VERSION, formed_generator_version=_GENERATOR_VERSION,
        requested_config=_config_payload(config), effective_config=_plain(effective), seed=template.seed,
        stream_scheme=dict(key=list(_STREAM_KEY), namespace=_STREAM_NAMESPACE, encoding="compact UTF-8 JSON array",
                           digest="SHA-256, big-endian integer", bit_generator="PCG64",
                           numpy_version=np.__version__,
                           components={k: dict(entity=e, draws=d) for k, (e, d) in _SAMPLE_COMPONENTS.items()},
                           streams=list(log.values())),
        sample_grid=dict(frame_id=sample_grid.frame_id, spacing_zyx=flat(s), origin_zyx=flat(layout.origin),
                         direction_zyx=[list(r) for r in _IDENTITY], spatial_unit="µm",
                         shape_zyx=[int(v) for v in layout.shape]),
        domain=dict(omega=dict(lo_zyx=flat(layout.omega[0]), hi_zyx=flat(layout.omega[1])),
                    covered=dict(lo_zyx=flat(layout.covered[0]), hi_zyx=flat(layout.covered[1])),
                    units="µm", interval="half-open [lo, hi) per axis"),
        codebook=_codebook_payload(codebook), fovs=fov_records,
        sample_label_sha256={t: hashlib.sha256(a.tobytes()).hexdigest() for t, a in sample_labels.items()})
    return SampleScene(sample, fovs, stains, fov_metadata, placement, positions, nuclei_table, cells_table,
                       molecules, cell_fovs, molecule_fovs, sample_grid, sample_labels, fov_labels, record)


def save_sample_scene(scene: SampleScene, directory: Path | str) -> Path:
    """Write a sample scene to a new or empty directory and return its path.

    Parameters
    ----------
    scene : SampleScene
        Sample scene to save.
    directory : pathlib.Path or str
        Destination directory.

    Returns
    -------
    pathlib.Path
        Path of the written directory.

    Notes
    -----
    Layout (design/specs/synthetic/sample-scene.md, "Truth schema"): ``sample.json``
    (the record plus the SHA-256 of every other file), ``positions.csv``,
    ``placement_truth.csv``, ``nuclei.csv`` (with a cellular scene),
    ``cells.csv``, ``molecules.csv``, ``cell_fovs.csv``, ``molecule_fovs.csv``,
    ``labels/sample_nuclei.tif`` and ``labels/sample_cells.tif``, and per FOV the
    ``save_amplicon_scene`` layout under ``fovs/<fov_id>/`` with ``stain.ome.tif``,
    ``labels/nuclei.tif``, ``labels/cells.tif`` and the legacy one-based
    ``molecules.csv`` (x, y, z, gene). A scene generated with ``on_fov`` holds no
    FOV images and raises ValueError; an existing nonempty directory raises
    FileExistsError, and nothing is overwritten.
    """
    if not isinstance(scene, SampleScene):
        raise TypeError("scene must be SampleScene")
    fov_ids = list(scene.positions.fov_id)
    if set(scene.fovs) != set(fov_ids):
        raise ValueError("the scene holds no FOV images (generated with on_fov); save each FOV in the callback")
    for fov in fov_ids:
        if fov in (".", "..") or Path(fov).name != fov or "\\" in fov:
            raise ValueError(f"FOV ID is not a plain file name: {fov!r}")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"sample scene directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    scene.positions.to_csv(directory / "positions.csv", index=False)
    scene.placement.to_csv(directory / "placement_truth.csv", index=False)
    cellular = bool(scene.sample_labels)
    if cellular:
        scene.nuclei.to_csv(directory / "nuclei.csv", index=False)
    scene.cells.to_csv(directory / "cells.csv", index=False)
    scene.molecules.to_csv(directory / "molecules.csv", index=False)
    scene.cell_fovs.to_csv(directory / "cell_fovs.csv", index=False)
    scene.molecule_fovs.to_csv(directory / "molecule_fovs.csv", index=False)
    for target, name in (("nucleus", "sample_nuclei.tif"), ("cell", "sample_cells.tif")):
        if target in scene.sample_labels:
            save_volume(scene.sample_labels[target], directory / "labels" / name, metadata=scene.sample_grid)
    genes = dict(zip(scene.molecules.amplicon_id, scene.molecules.gene_id))
    for fov in fov_ids:
        root = save_amplicon_scene(scene.fovs[fov], directory / "fovs" / fov)
        metadata = scene.fov_metadata[fov]
        if fov in scene.stains:
            # One channel of a ZYXC OME-TIFF keeps the grid metadata on a .ome.tif name.
            save_volume(scene.stains[fov][..., None], root / "stain.ome.tif", metadata=metadata)
        for target, name in (("nucleus", "nuclei.tif"), ("cell", "cells.tif")):
            if target in scene.fov_labels.get(fov, {}):
                save_volume(scene.fov_labels[fov][target], root / "labels" / name, metadata=metadata)
        rows = scene.molecule_fovs[scene.molecule_fovs.fov_id == fov].sort_values("csv_row")
        pd.DataFrame(dict(x=rows.x.to_numpy() + 1, y=rows.y.to_numpy() + 1, z=rows.z.to_numpy() + 1,
                          gene=[genes[i] for i in rows.amplicon_id])).to_csv(root / "molecules.csv", index=False)
    files = {path.relative_to(directory).as_posix(): file_sha256(path)
             for path in sorted(directory.rglob("*")) if path.is_file()}
    record = dict(scene.record, files=files)
    (directory / "sample.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return directory


_PRESET_SPECS = {
    "sample-tiny-v1": dict(
        tier="routine", columns=2, rows=2, fov_shape_zyx=(5, 36, 36), spacing_zyx=(1.0, 0.5, 0.5), overlap=1 / 3,
        jitter_max_zyx=(0.4, 1.5, 1.5), translation_max_zyx=(0.0, 1.0, 1.0), sample_grid_zyx=(6, 63, 63),
        seeds=dict(ordinary=7, binucleate=1, empty=1), nucleus_radius_median=dict(axial=1.5, lateral=2.5),
        reach_median=3.0, molecules_per_cell=8.0, ambient_density=0.002,
        appearance="constant A 100, sz = sl = 1, no readout effect or background, Gaussian read noise sigma 1",
        dtype="float32", accumulation="float64"),
    "sample-small-v1": dict(
        tier="slow", columns=3, rows=3, fov_shape_zyx=(14, 96, 96), spacing_zyx=(0.346255, 0.194398, 0.194398),
        overlap=0.25, jitter_max_zyx=(0.8, 4.5, 4.5), translation_max_zyx=(1.0, 2.0, 2.0),
        sample_grid_zyx=(16, 249, 249), seeds=dict(ordinary=14, binucleate=2, empty=2),
        nucleus_radius_median=dict(axial=2.0, lateral=3.0), reach_median=2.5, molecules_per_cell=30.0,
        ambient_density=0.002, appearance="benchmark appearance of benchmark-presets-v1 on the sample grid",
        dtype="float32", accumulation="float32"),
}

#: Sample presets (sample-presets-v1), separate from SCENE_PRESETS.
SAMPLE_PRESETS = {name: dict(version=SAMPLE_PRESET_VERSION, seed=42, split="development", scene_key=name,
                             codebook="development_codebook('balanced')", **spec)
                  for name, spec in _PRESET_SPECS.items()}


def sample_scene_preset(name: str) -> tuple[Codebook, SampleSceneConfig]:
    """Return (codebook, config) for a SAMPLE_PRESETS name.

    Parameters
    ----------
    name : str
        Preset name from ``SAMPLE_PRESETS``.

    Returns
    -------
    codebook : Codebook
        Sequencing codebook for the preset.
    config : SampleSceneConfig
        Scene settings for the preset.

    Notes
    -----
    Both presets use ``development_codebook("balanced")``, seed 42, split
    ``development``, float32 output, the scene key of their name, a
    translation-only per-round geometry about ``round1`` and the default
    ``StainConfig``. ``sample-tiny-v1`` (2 × 2 FOVs of 5×36×36, routine tier)
    renders constant amplicons with float64 accumulation for the oracle
    identities; ``sample-small-v1`` (3 × 3 FOVs of 14×96×96 at the LN
    calibration, slow tier) uses the benchmark appearance placed on the sample
    grid. Unknown names raise ValueError.
    """
    from ._calibrated import development_codebook
    from ._presets import _appearance
    if name not in _PRESET_SPECS:
        raise ValueError(f"unknown sample preset: {name}; choose from {list(_PRESET_SPECS)}")
    spec = _PRESET_SPECS[name]
    codebook = development_codebook("balanced")
    acquisition = AcquisitionConfig(columns=spec["columns"], rows=spec["rows"], fov_shape_zyx=spec["fov_shape_zyx"],
                                    spacing_zyx=spec["spacing_zyx"], overlap=spec["overlap"],
                                    jitter_max_zyx=spec["jitter_max_zyx"])
    geometry = GeometryConfig(translation_enabled=True, translation_max_zyx=spec["translation_max_zyx"],
                              reference_round=codebook.round_labels[0])
    base = dict(dataset_version=name, sample_id="sample", scene_key=name, seed=42, split="development",
                shape_zyx=spec["fov_shape_zyx"], dtype=spec["dtype"], accumulation=spec["accumulation"],
                geometry=geometry)
    if name == "sample-tiny-v1":
        template = AmpliconSceneConfig(**base, noise=NoiseConfig(independent_enabled=True, sigma=1.0))
    else:
        shape = tuple(int(v) for v in _layout(acquisition).shape)
        template = AmpliconSceneConfig(**base, **{k: v for k, v in _appearance(shape, len(codebook.round_labels),
                                                                               "float32").items() if k != "dtype"})
    seeds, radius = spec["seeds"], spec["nucleus_radius_median"]
    cells = CellsConfig(nuclei=seeds["ordinary"], binucleate=seeds["binucleate"], empty=seeds["empty"],
                        nucleus_axial_radius=_lognormal(radius["axial"], 0.1),
                        nucleus_lateral_radius=_lognormal(radius["lateral"], 0.1),
                        reach=_lognormal(spec["reach_median"], 0.15))
    return codebook, SampleSceneConfig(fov=template, acquisition=acquisition, cells=cells, stain=StainConfig(),
                                       molecules_per_cell=spec["molecules_per_cell"],
                                       ambient_density=spec["ambient_density"])
