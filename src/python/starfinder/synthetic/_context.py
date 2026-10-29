"""The cellular scene: lobed and touching nuclei, territories, cell types and molecules by compartment.

See design/specs/synthetic/cellular-scene.md. ``generate_cellular_scene`` runs the sample scene's
generation (design/specs/synthetic/sample-scene.md) with that page's placement, label and molecule
rules in place of the minimal ones; with every new part at its default it is
``generate_sample_scene`` itself (identity O1). Every new draw comes from a ``context.``
stream, and the new truth is written beside the unchanged sample layout with its own version.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from starfinder._records import file_sha256
from starfinder.readout import Codebook
from starfinder.io import save_volume

from ._cellular import _contact, _extent, _harmonic_bound, _harmonic_terms, _labels, _lobe_one_extent, _shapes
from ._common import ScalarDistribution, _draw, _integer, _json, _label, _probabilities
from ._formed import _STREAM_KEY, _STREAM_NAMESPACE, _plain
from ._observation import _noise, _observe, _prepare_background, evaluate_background_translated
from ._sample import (_BLOCK, SampleScene, SampleSceneConfig, _cast, _checked_distribution, _config_payload, _generate,
                      _inside, _label_functions, _layout, _real, _streams, save_sample_scene)
from ._stains import (_ROUNDS, MorphologyRoundConfig, StainChannelConfig, _bright_regions, _bumps, _median, _model,
                      _plateaus, _Points)

#: Version of the cellular-scene truth files (context.json and its tables); any change to a column or rule bumps it.
CELLULAR_TRUTH_VERSION = "1"
#: Version of the cellular presets (CELLULAR_PRESETS); any change to a preset value bumps it.
CELLULAR_PRESET_VERSION = "context-presets-v1"

_KINDS = ("ordinary", "binucleate", "empty")
_RULES = ("reach", "grown", "power")
# nextafter steps of a contact proposal or a binucleate pair before the proposal is given up.
_STEPS = 10_000
# Largest batch of proposals drawn ahead and tested in drawing order (seeds and molecules); the draws per proposal
# are those of one call at a time.
_CHUNK = 256
# The new stream components of this delivery: entity and draws of each (design/specs/synthetic/cellular-scene.md).
_CONTEXT_COMPONENTS = {
    "context.cell.kind": ("cell-<k>", "one draw of the kind (density mode)"),
    "context.nucleus.lobed": ("nucleus-<n>", "one uniform (lobed or not)"),
    "context.nucleus.lobes": ("nucleus-<n>", "one integer, the lobes beyond the first"),
    "context.nucleus.lobe.scale": ("nucleus-<n>-lobe-<l>", "one value"),
    "context.nucleus.lobe.offset": ("nucleus-<n>-lobe-<l>", "one value"),
    "context.nucleus.lobe.direction": ("nucleus-<n>-lobe-<l>", "two uniforms (one in a plane or a monolayer)"),
    "context.nucleus.lobe.angle": ("nucleus-<n>-lobe-<l>", "one value"),
    "context.nucleus.harmonics": ("nucleus-<n>", "one standard normal per coefficient, by degree then order"),
    "context.cell.contact": ("cell-<k>", "per proposal one uniform; for a contact proposal the partner index "
                                         "and up to 16 directions"),
    "context.cell.growth": ("cell-<k>", "one value (T2)"),
    "context.region.placement": ("region-<r>", "z, y, x"),
    "context.region.type": ("region-<r>", "one type"),
    "context.cell.type": ("cell-<k>", "one uniform (purity), then one type draw when it fails"),
    "context.molecule.mean": ("cell-<k>", "one standard normal of the lognormal mean (M2)"),
    "context.molecule.compartment": ("molecule ID", "one uniform (NF1)"),
    "context.identity": ("molecule ID", "one gene draw with the type's abundance"),
    "context.ambient.identity": ("molecule ID", "one gene draw with the ambient abundance"),
    "context.stain.brightness": ("nucleus-<n> or cell-<k>", "one value; channel_label the stain name"),
    "context.texture.count": ("nucleus-<n>", "one Poisson value; channel_label the stain name"),
    "context.texture.placement": ("nucleus-<n>-region-<b>", "z, y, x per proposal; channel_label the stain name"),
    "context.texture.size": ("nucleus-<n>-region-<b>", "one value; channel_label the stain name"),
    "context.texture.height": ("nucleus-<n>-region-<b>", "one value; channel_label the stain name"),
    "context.unevenness.count": ("field", "one Poisson value; channel_label the stain name"),
    "context.unevenness.placement": ("bump-<q>", "z, y, x; channel_label the stain name"),
    "context.unevenness.amplitude": ("bump-<q>", "one standard normal z_q; channel_label the stain name"),
    **{f"context.stain.background.{name}": ("as background.{name}", "as background.{name}; channel_label the stain "
                                            "name") for name in ("count", "placement", "width", "brightness")},
    "context.morphology.translation": ("FOV ID", "u_z, u_y, u_x (MT1)"),
}
# The truth tables of the stains and the morphology round, with their columns ("Truth schema").
_STAIN_TABLES = {
    "stain_channels": ["stain", "source", "round", "channel", "dtype", "scale", "m_A"],
    "stain_objects": ["stain", "object", "object_id", "plateau"],
    "bright_regions": ["region_id", "nucleus_id", "stain", "centre_z", "centre_y", "centre_x", "size", "height"],
    "unevenness": ["stain", "bump_id", "centre_z", "centre_y", "centre_x", "z", "a", "length"],
    "morphology_truth": ["fov_id", "round", "translation_z_um", "translation_y_um", "translation_x_um",
                         "translation_z_vox", "translation_y_vox", "translation_x_vox", "pull_displacement_z_vox",
                         "pull_displacement_y_vox", "pull_displacement_x_vox"],
}
_STAIN_TYPES = {
    "stain_channels": dict(stain=object, source=object, round=object, channel=object, dtype=object, scale=np.float64,
                           m_A=np.float64),
    "stain_objects": dict(stain=object, object=object, object_id=np.uint32, plateau=np.float64),
    "bright_regions": dict(region_id=np.int64, nucleus_id=np.uint32, stain=object, centre_z=np.float64,
                           centre_y=np.float64, centre_x=np.float64, size=np.float64, height=np.float64),
    "unevenness": dict(stain=object, bump_id=np.int64, centre_z=np.float64, centre_y=np.float64, centre_x=np.float64,
                       z=np.float64, a=np.float64, length=np.float64),
    "morphology_truth": dict(fov_id=object, round=object,
                             **{c: np.float64 for c in _STAIN_TABLES["morphology_truth"][2:]}),
}


def _stain_table(name, rows):
    return pd.DataFrame(rows, columns=_STAIN_TABLES[name]).astype(_STAIN_TYPES[name])


def _g1(types=4, genes=16, marker=0.15, shared=0.0875):
    """G1: two marker genes per type (genes 2t − 1 and 2t), the genes after the markers shared by every type."""
    rows = []
    for t in range(types):
        row = [0.0] * genes
        row[2 * t] = row[2 * t + 1] = marker
        for g in range(2 * types, genes):
            row[g] = shared
        rows.append(tuple(row))
    return tuple(rows)


def _unit(value, name):
    value = _real(value, name)
    if value > 1:
        raise ValueError(f"{name} must satisfy 0 ≤ {name} ≤ 1; got {value}")
    return value


def _weights(value, size, name):
    if not isinstance(value, tuple) or (size is not None and len(value) != size):
        raise ValueError(f"{name} must be a tuple" + ("" if size is None else f" of {size} values"))
    for v in value:
        _real(v, name)
    if not sum(value) > 0:
        raise ValueError(f"{name} must hold a positive value")


@dataclass(frozen=True)
class NucleusShapeConfig:
    """Lobed nuclei (N1) and the optional harmonic perturbation of lobe 1 (N2), µm and radians.

    A nucleus has more than one lobe with probability ``lobed_share``; then an integer
    drawn uniformly in ``extra_lobes`` gives its further lobes. Lobe ``l ≥ 2`` has semi-axes
    ``lobe_scale`` times lobe 1's, lobe 1's elongation, an angle of the nucleus angle law and
    its centre at ``D_l û_l`` from the nucleus centre, ``D_l = min(λ_l a_l, 0.9 (e_1 + ε_l))``
    with ``λ_l`` from ``lobe_offset`` (LB1). ``harmonics`` is the largest degree ``L`` of N2
    (0 off); the coefficients are scaled so that ``Σ abs(a_lm) = harmonic_amplitude``, and
    ``harmonic_amplitude · M_Y < 1`` is required. Instances are not hashable.
    """

    __hash__ = None

    lobed_share: float = 0.0
    extra_lobes: tuple[int, int] = (1, 2)
    lobe_scale: ScalarDistribution = field(default_factory=lambda: ScalarDistribution("uniform", (0.5, 0.8)))
    lobe_offset: ScalarDistribution = field(default_factory=lambda: ScalarDistribution("uniform", (0.5, 0.9)))
    harmonics: int = 0
    harmonic_amplitude: float = 0.0

    def __post_init__(self):
        _unit(self.lobed_share, "lobed_share")
        lobes = self.extra_lobes
        if (not isinstance(lobes, tuple) or len(lobes) != 2 or any(type(v) is not int for v in lobes)
                or not 1 <= lobes[0] <= lobes[1]):
            raise ValueError(f"extra_lobes must be integers (low, high) with 1 ≤ low ≤ high; got {lobes!r}")
        for name in ("lobe_scale", "lobe_offset"):
            _checked_distribution(getattr(self, name), "width.lateral", (), name)
        _integer(self.harmonics, "harmonics", 0)
        if self.harmonics == 1:
            raise ValueError("harmonics must be 0 (off) or a largest degree L ≥ 2")
        _real(self.harmonic_amplitude, "harmonic_amplitude")


@dataclass(frozen=True)
class PackingConfig:
    """Placement with touching nuclei (K1, P1), a target density and the crowding rule (F2), µm.

    ``density`` is the target nominal nucleus volume fraction of the scene domain (the area
    fraction of its Y-X box with one plane or a monolayer), ``0 < density < 1``; None keeps the
    seed counts of ``CellsConfig``. ``kind_shares`` (ordinary, binucleate, empty) gives
    the kind probabilities in density mode (None: proportional to the counts). A proposal is
    placed against an accepted site with probability ``contact``; ``overlap`` is ``κ``,
    ``0 ≤ κ < 0.5``, and ``gap`` the gap ``g ≥ 0``. ``monolayer`` puts every centre in the
    stack's mid-plane. ``on_crowded`` is ``"raise"`` or ``"stop"``; with ``"stop"`` a reached
    share below ``floor`` of the target still raises. Instances are not hashable.
    """

    __hash__ = None

    density: float | None = None
    kind_shares: tuple[float, float, float] | None = None
    contact: float = 0.0
    overlap: float = 0.0
    gap: float = 0.0
    monolayer: bool = False
    on_crowded: str = "raise"
    floor: float = 0.9

    def __post_init__(self):
        if self.density is not None:
            density = _real(self.density, "density")
            if not 0 < density < 1:
                raise ValueError(f"density must satisfy 0 < density < 1; got {density}")
        if self.kind_shares is not None:
            _weights(self.kind_shares, 3, "kind_shares")
            if self.density is not None and not self.kind_shares[0] + self.kind_shares[1] > 0:
                raise ValueError("kind_shares: the ordinary and binucleate shares are both 0 in density mode, "
                                 "so placement would not reach the density")
        _unit(self.contact, "contact")
        overlap = _real(self.overlap, "overlap")
        if not overlap < 0.5:
            raise ValueError(f"overlap must satisfy 0 ≤ overlap < 0.5; got {overlap}")
        _real(self.gap, "gap")
        if not isinstance(self.monolayer, bool):
            raise ValueError("monolayer must be a bool")
        if self.on_crowded not in ("raise", "stop"):
            raise ValueError(f"on_crowded must be 'raise' or 'stop'; got {self.on_crowded!r}")
        floor = _real(self.floor, "floor", strict=True)
        if floor > 1:
            raise ValueError(f"floor must satisfy 0 < floor ≤ 1; got {floor}")


@dataclass(frozen=True)
class TerritoryConfig:
    """The territory rule: ``"reach"`` (T1, the site sphere plus the cell's reach), ``"grown"``
    (T2, the nucleus with every semi-axis grown by ``δ`` µm, one ``growth`` draw per cell) or
    ``"power"`` (T3, the whole power cell). Instances are not hashable."""

    __hash__ = None

    rule: str = "reach"
    growth: ScalarDistribution = field(default_factory=lambda: ScalarDistribution("lognormal", (math.log(0.8), 0.15)))

    def __post_init__(self):
        if self.rule not in _RULES:
            raise ValueError(f"rule must be one of {list(_RULES)}; got {self.rule!r}")
        _checked_distribution(self.growth, "width.lateral", (), "growth")


@dataclass(frozen=True)
class CellTypeConfig:
    """Cell types with a gene abundance per type (G1) and their placement by region (Ty2).

    ``abundances`` holds one row per type over the codebook's genes (rows are normalized);
    ``proportions`` is ``π``. ``regions`` Voronoi regions (0: Ty1, types at random) each have a
    main type, which a cell takes with probability ``purity`` from the region of its anchor.
    The per-type values replace the scene's molecules per cell and nuclear fraction and scale
    the nucleus size laws. ``marker_pairs`` are (gene_a, gene_b, relation) with relation
    ``exclusive`` or ``coexpressed``. Instances are not hashable.
    """

    __hash__ = None

    types: tuple[str, ...] = ("type-1", "type-2", "type-3", "type-4")
    abundances: tuple[tuple[float, ...], ...] = field(default_factory=_g1)
    proportions: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    regions: int = 4
    purity: float = 0.9
    molecules_per_cell: tuple[float, ...] | None = None
    nuclear_fraction: tuple[float, ...] | None = None
    size_scale: tuple[float, ...] | None = None
    marker_pairs: tuple[tuple[str, str, str], ...] = (("balanced-01", "balanced-03", "exclusive"),
                                                      ("balanced-01", "balanced-02", "coexpressed"))

    def __post_init__(self):
        names = self.types
        if not isinstance(names, tuple) or not names or len(set(names)) != len(names):
            raise ValueError("types must be a nonempty tuple of unique names")
        for name in names:
            try:
                _label(name)
            except ValueError:
                raise ValueError(f"types holds an invalid name: {name!r}") from None
        if "ambient" in names:
            raise ValueError("types: 'ambient' names the ambient abundance row and is not a type name")
        count = len(names)
        if not isinstance(self.abundances, tuple) or len(self.abundances) != count:
            raise ValueError(f"abundances must hold one row per type ({count})")
        widths = {len(row) if isinstance(row, tuple) else -1 for row in self.abundances}
        if len(widths) != 1 or widths == {-1}:
            raise ValueError("abundances must be tuples of one length (the codebook's genes)")
        for t, row in enumerate(self.abundances):
            _weights(row, None, f"abundances[{t}]")
        _weights(self.proportions, count, "proportions")
        _integer(self.regions, "regions", 0)
        _unit(self.purity, "purity")
        for name, check in (("molecules_per_cell", _real), ("nuclear_fraction", _unit),
                            ("size_scale", lambda v, n: _real(v, n, strict=True))):
            values = getattr(self, name)
            if values is None:
                continue
            if not isinstance(values, tuple) or len(values) != count:
                raise ValueError(f"{name} must be None or a tuple of one value per type ({count})")
            for v in values:
                check(v, name)
        if not isinstance(self.marker_pairs, tuple):
            raise ValueError("marker_pairs must be a tuple of (gene_a, gene_b, relation)")
        for pair in self.marker_pairs:
            if (not isinstance(pair, tuple) or len(pair) != 3 or pair[2] not in ("exclusive", "coexpressed")
                    or not all(isinstance(g, str) for g in pair[:2])):
                raise ValueError(f"marker_pairs holds an invalid pair: {pair!r}; relation is exclusive or coexpressed")


@dataclass(frozen=True)
class MoleculeConfig:
    """Molecules per cell (M2), the nuclear fraction of reads (NF1) and ambient genes (A2).

    The count of cell ``k`` is Poisson with mean ``μ_k ~ lognormal(log m, s)``
    (``per_cell_median`` m, ``per_cell_spread`` s; s = 0 is M1). Each cell molecule lies in
    its cell's nucleus with probability ``nuclear_fraction`` and in the rest of its territory
    otherwise (None: uniform in the territory). ``ambient_density`` is per µm³ of the scene
    domain (None: the sample scene's); ``ambient_abundances`` (None: the shared genes,
    uniform) gives the ambient genes. Instances are not hashable.
    """

    __hash__ = None

    per_cell_median: float = 8.0
    per_cell_spread: float = 0.0
    nuclear_fraction: float | None = None
    ambient_density: float | None = None
    ambient_abundances: tuple[float, ...] | None = None

    def __post_init__(self):
        _real(self.per_cell_median, "per_cell_median")
        _real(self.per_cell_spread, "per_cell_spread")
        if self.nuclear_fraction is not None:
            _unit(self.nuclear_fraction, "nuclear_fraction")
        if self.ambient_density is not None:
            _real(self.ambient_density, "ambient_density")
        if self.ambient_abundances is not None:
            _weights(self.ambient_abundances, None, "ambient_abundances")


@dataclass(frozen=True)
class CellularSceneConfig:
    """A sample scene and the parts of the cellular scene; each part None (or empty) is off.

    ``sample`` is an unchanged ``SampleSceneConfig`` (template, acquisition, seeds, size laws,
    molecules, stain). ``nuclei``, ``packing``, ``territories``, ``types`` and ``molecules``
    are described by their classes and need ``sample.cells``. ``stains`` is a tuple of
    ``StainChannelConfig`` (empty: the sample scene's stain), which requires ``sample.stain``
    None; ``morphology`` is the round that holds the stains of round ``morphology``. Invalid
    combinations raise ValueError naming the field. Instances are not hashable.
    """

    __hash__ = None

    sample: SampleSceneConfig = field(default_factory=SampleSceneConfig)
    nuclei: NucleusShapeConfig | None = None
    packing: PackingConfig | None = None
    territories: TerritoryConfig | None = None
    types: CellTypeConfig | None = None
    molecules: MoleculeConfig | None = None
    stains: tuple[StainChannelConfig, ...] = ()
    morphology: MorphologyRoundConfig | None = None

    def __post_init__(self):
        if not isinstance(self.sample, SampleSceneConfig):
            raise TypeError("sample must be SampleSceneConfig")
        for name, kind in (("nuclei", NucleusShapeConfig), ("packing", PackingConfig),
                           ("territories", TerritoryConfig), ("types", CellTypeConfig),
                           ("molecules", MoleculeConfig), ("morphology", MorphologyRoundConfig)):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name} must be {kind.__name__} or None")
        if not isinstance(self.stains, tuple):
            raise TypeError("stains must be a tuple")
        if self.stains and self.sample.stain is not None:
            raise ValueError("stains: sample.stain must be None when stains is given (the context's first "
                             "nucleus stain of the reference round takes its place)")
        if not all(isinstance(stain, StainChannelConfig) for stain in self.stains):
            raise TypeError("stains must hold StainChannelConfig")
        self._check_stains()
        parts = [name for name in ("nuclei", "packing", "territories", "types", "molecules")
                 if getattr(self, name) is not None]
        cells = self.sample.cells
        if parts and cells is None:
            raise ValueError(f"sample.cells must be CellsConfig when {', '.join(parts)} is given")
        layout = _layout(self.sample.acquisition)
        packing = self.packing
        monolayer = packing is not None and packing.monolayer
        if monolayer:
            if layout.flat:
                raise ValueError("monolayer: the scene must have more than one plane (one plane is the Z=1 case)")
            z_mid = (layout.n[0] - 1) / 2 * layout.s[0]
            lo, hi = layout.covered[0][0], layout.covered[1][0]
            if not lo <= z_mid < hi:
                raise ValueError(f"monolayer: z_mid = {z_mid} µm must lie in the covered region's Z interval "
                                 f"[{lo}, {hi}) µm")
        nuclei = self.nuclei
        if nuclei is not None and nuclei.harmonics >= 2:
            bound = _harmonic_bound(nuclei.harmonics, layout.flat or monolayer)
            if not nuclei.harmonic_amplitude * bound < 1:
                raise ValueError(f"harmonic_amplitude · M_Y must be below 1 so that r > 0; got "
                                 f"{nuclei.harmonic_amplitude} · {bound} = {nuclei.harmonic_amplitude * bound}")
        if packing is not None and packing.density is not None:
            shares = packing.kind_shares or (cells.nuclei, cells.binucleate, cells.empty)
            if not shares[0] + shares[1] > 0:
                raise ValueError("kind_shares: the ordinary and binucleate shares are both 0 in density mode "
                                 "(from the counts of sample.cells), so placement would not reach the density")

    def _check_stains(self):
        """Names and channels unique per round, one configuration per name, the morphology round and its bound."""
        stains, morphology = self.stains, self.morphology
        for round_ in _ROUNDS:
            own = [stain for stain in stains if stain.round == round_]
            for key in ("name", "channel"):
                values = [getattr(stain, key) for stain in own]
                repeated = sorted({v for v in values if values.count(v) > 1})
                if repeated:
                    raise ValueError(f"stains: the {key} {repeated[0]!r} occurs more than once in the {round_} round")
        first = {}
        for stain in stains:
            plain = replace(stain, round="reference", channel="ch04")
            if stain.name in first and first[stain.name] != plain:
                raise ValueError(f"stains: the stains named {stain.name!r} differ apart from round and channel; "
                                 "stains that share a name share their draws and must share their configuration")
            first.setdefault(stain.name, plain)
        held = [stain.name for stain in stains if stain.round == "morphology"]
        if held and morphology is None:
            raise ValueError(f"morphology: the stains {held} of round 'morphology' need a MorphologyRoundConfig")
        if morphology is None:
            return
        if not held:
            raise ValueError("morphology: the morphology round holds no stain; give stains with round='morphology'")
        acquisition = self.sample.acquisition
        if acquisition.fov_shape_zyx[0] == 1 and morphology.translation_max_zyx[0] != 0:
            raise ValueError("morphology: translation_max_zyx must be 0 in z when fov_shape_zyx has one plane "
                             "(n_z = 1)")
        if morphology.translation == "supplied":
            ids = acquisition.ids()
            unknown = sorted(set(morphology.translation_zyx) - set(ids))
            missing = [fov for fov in ids if fov not in morphology.translation_zyx]
            if unknown or missing:
                raise ValueError(f"morphology: translation_zyx must name exactly the FOVs of the grid; unknown "
                                 f"{unknown}, missing {missing}")


@dataclass
class CellularScene:
    """A generated cellular scene: the sample scene and the new truth tables.

    ``sample`` is the ``SampleScene`` (its tables, labels and FOVs made with the page's
    rules). ``stains`` maps each FOV ID to its reference-round stain images by stain name
    (ZYX, each in its output dtype) and ``morphology`` to the morphology round (ZYXC, its
    stains in configuration order); both are filled also when the FOVs were handed to
    ``on_fov``. The tables are those of design/specs/synthetic/cellular-scene.md ("Truth schema").
    ``record`` holds versions, the requested and effective configuration, the placement
    outcome and the context streams drawn (the stains' noise streams included).
    """

    sample: SampleScene
    stains: dict[str, dict[str, np.ndarray]]
    morphology: dict[str, np.ndarray]
    lobes: pd.DataFrame
    harmonics: pd.DataFrame
    cell_context: pd.DataFrame
    regions: pd.DataFrame
    type_abundances: pd.DataFrame
    cell_gene_counts: pd.DataFrame
    marker_pairs: pd.DataFrame
    stain_channels: pd.DataFrame
    stain_objects: pd.DataFrame
    bright_regions: pd.DataFrame
    unevenness: pd.DataFrame
    morphology_truth: pd.DataFrame
    record: dict


def _direction(rng, planar):
    """A uniform unit vector: on the Y-X circle (one uniform) or on the sphere (two uniforms)."""
    if planar:
        phi = 2 * math.pi * rng.random()
        return np.array([0.0, math.sin(phi), math.cos(phi)])
    u = rng.random(2)
    z = 2 * u[0] - 1
    s = math.sqrt(max(0.0, 1 - z * z))
    phi = 2 * math.pi * u[1]
    return np.array([z, s * math.sin(phi), s * math.cos(phi)])


def _nominal(lobes, planar):
    """Nominal size of a nucleus: Σ over lobes of 4/3 π a_z a_l² e (π a_l² e in a plane or a monolayer)."""
    if planar:
        return sum(math.pi * lateral * lateral * elongation for _, _, lateral, elongation, _ in lobes)
    return sum(4 / 3 * math.pi * axial * lateral * lateral * elongation for _, axial, lateral, elongation, _ in lobes)


def _sites(cells, nuclei, rule, empty_radius, terms, planar):
    """The label sites of the cellular scene, in cell order with each cell's sites in nucleus order."""
    rows = [(cell, centre, radius) for cell in cells for centre, radius in cell["sites"]]
    own = {}
    for x in nuclei:
        own.setdefault(x["cell_id"], []).append(x["nucleus_id"])
    site_nucleus = []
    for cell in cells:
        site_nucleus.extend(own.get(cell["cell_id"], [0] * len(cell["sites"])))
    nuclei = sorted(nuclei, key=lambda x: x["nucleus_id"])
    return SimpleNamespace(
        centres=np.array([c for _, c, _ in rows], dtype=np.float64).reshape(-1, 3),
        radii=np.array([r for _, _, r in rows], dtype=np.float64),
        reach=np.array([cell["reach"] for cell, _, _ in rows], dtype=np.float64),
        growth=np.array([np.nan if cell.get("growth") is None else cell["growth"] for cell, _, _ in rows],
                        dtype=np.float64),
        cell=np.array([cell["cell_id"] for cell, _, _ in rows], dtype=np.uint32),
        nucleus=np.array(site_nucleus, dtype=np.uint32),
        rule=rule, empty_radius=float(empty_radius),
        shapes=_shapes([x["centre"] for x in nuclei], [x["lobes"] for x in nuclei],
                       [x["coefficients"] for x in nuclei], terms, planar))


class _Context:
    """The draws of one cellular scene, passed to the sample scene's generation as its two hooks."""

    def __init__(self, config, codebook, stream):
        self.config, self.codebook, self.cstream = config, codebook, stream
        self.types = {}
        self.shapes, self.cells = {}, {}
        self.regions = []
        self.means = {}
        self.packing = None
        self.planar = False
        self.z_mid = None
        self.rule = config.territories.rule if config.territories is not None else "reach"
        nuclei = config.nuclei
        self.degree = nuclei.harmonics if nuclei is not None else 0

    # ------------------------------------------------------------------ cell types

    def _region(self, point):
        if not self.regions:
            return None
        centres = np.array([r["centre"] for r in self.regions])
        distance = None
        for a in range(3):
            delta = point[a] - centres[:, a]
            distance = delta * delta if distance is None else distance + delta * delta
        return int(np.argmin(distance))

    def _draw_regions(self, layout):
        types = self.config.types
        lo, hi = layout.omega
        probabilities = _probabilities(types.proportions, len(types.types), "proportions")
        for r in range(1, types.regions + 1):
            centre = self.cstream("context.region.placement", f"region-{r}").uniform(lo, hi)
            if layout.flat:
                centre[0] = 0.0
            main = int(self.cstream("context.region.type", f"region-{r}").choice(len(types.types), p=probabilities))
            self.regions.append(dict(region_id=r, centre=centre, main_type=main))

    def _type_at(self, k, point):
        """The type of cell k with its anchor at ``point`` (Ty2; Ty1 without regions)."""
        types = self.config.types
        draws = self.types.setdefault(k, {})
        if "rng" not in draws:
            draws["rng"] = self.cstream("context.cell.type", f"cell-{k}")
            draws["purity"] = draws["rng"].random() if self.regions else None
        region = self._region(point)
        if region is not None and draws["purity"] < types.purity:
            return self.regions[region]["main_type"], region
        if "fallback" not in draws:
            probabilities = _probabilities(types.proportions, len(types.types), "proportions")
            draws["fallback"] = int(draws["rng"].choice(len(types.types), p=probabilities))
        return draws["fallback"], region

    # ------------------------------------------------------------------ nucleus shapes

    def _sizes(self, cells_config, n, stream):
        laws = (("sample.nucleus.axial", cells_config.nucleus_axial_radius),
                ("sample.nucleus.lateral", cells_config.nucleus_lateral_radius),
                ("sample.nucleus.elongation", cells_config.nucleus_elongation),
                ("sample.nucleus.angle", cells_config.nucleus_angle))
        identity = f"nucleus-{n}"
        values = [_draw(spec, component.split(".", 1)[1], identity,
                        lambda _name, entity, component=component: stream(component, entity))
                  for component, spec in laws]
        if not np.isfinite(values).all() or min(values[:2]) <= 0 or values[2] < 1:
            raise ValueError(f"{identity}: nonfinite or out-of-domain nucleus size")
        return values

    def _shape_draws(self, cells_config, n):
        """The drawn values of nucleus n's lobes and harmonics, independent of its size."""
        nuclei = self.config.nuclei
        identity = f"nucleus-{n}"
        draws = SimpleNamespace(lobes=[], coefficients=None)
        if nuclei is None:
            return draws
        terms = _harmonic_terms(self.degree, self.planar)
        if terms:
            raw = self.cstream("context.nucleus.harmonics", identity).standard_normal(len(terms))
            total = float(np.abs(raw).sum())
            draws.coefficients = raw * (nuclei.harmonic_amplitude / total) if total > 0 else np.zeros(len(terms))
        if not self.cstream("context.nucleus.lobed", identity).random() < nuclei.lobed_share:
            return draws
        low, high = nuclei.extra_lobes
        extra = int(self.cstream("context.nucleus.lobes", identity).integers(low, high + 1))
        for l in range(2, extra + 2):
            entity = f"{identity}-lobe-{l}"
            value = lambda spec, component: _draw(  # noqa: E731
                spec, component, entity, lambda _name, e, component=component: self.cstream(component, e))
            scale = value(nuclei.lobe_scale, "context.nucleus.lobe.scale")
            offset = value(nuclei.lobe_offset, "context.nucleus.lobe.offset")
            direction = _direction(self.cstream("context.nucleus.lobe.direction", entity), self.planar)
            angle = value(cells_config.nucleus_angle, "context.nucleus.lobe.angle")
            if not (math.isfinite(scale) and scale > 0 and math.isfinite(offset) and offset > 0):
                raise ValueError(f"{entity}: nonfinite or nonpositive lobe scale or offset")
            draws.lobes.append((scale, offset, direction, angle))
        return draws

    def _shape(self, sizes, scale, draws):
        """Lobes (offset_zyx, axial, lateral, elongation, angle), coefficients, site radius, nominal size and the
        stacked shape of one nucleus (its centre at the origin) for its extents."""
        axial, lateral, elongation, angle = sizes
        axial, lateral = axial * scale, lateral * scale
        terms = _harmonic_terms(self.degree, self.planar)
        coefficients = None if not terms else np.asarray(draws.coefficients, dtype=np.float64)
        origin = (0.0, 0.0, 0.0)
        lobes = [(origin, axial, lateral, elongation, angle)]
        first = _shapes([origin], [lobes], [coefficients] if terms else None, terms, self.planar)
        radius = max(axial, lateral * elongation)
        if terms:
            radius = radius * (1 + self.config.nuclei.harmonic_amplitude * _harmonic_bound(self.degree, self.planar))
        start = np.zeros(1, dtype=np.int64)
        for k, offset, direction, lobe_angle in draws.lobes:
            lobe = (origin, k * axial, k * lateral, elongation, lobe_angle)
            own = _shapes([origin], [[lobe]], None, [], self.planar)
            # LB1: D_l = min(λ_l a_l, 0.9 (e_{n,1}(û_l) + ε_l(−û_l))).
            e1 = float(_lobe_one_extent(first, start, direction[None])[0])
            eps = float(_lobe_one_extent(own, start, -direction[None])[0])
            distance = min(offset * lateral, 0.9 * (e1 + eps))
            o = tuple((distance * direction + 0.0).tolist())
            lobes.append((o, k * axial, k * lateral, elongation, lobe_angle))
            radius = max(radius, math.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2)
                         + max(k * axial, k * lateral * elongation))
        shape = _shapes([origin], [lobes], [coefficients] if terms else None, terms, self.planar)
        return dict(lobes=lobes, coefficients=coefficients, radius=radius, nominal=_nominal(lobes, self.planar),
                    shape=shape)

    # ------------------------------------------------------------------ placement

    def _accepted(self, entries):
        """Stacked accepted sites: centres, radii, the shape index of each (−1 for an empty site)."""
        terms = _harmonic_terms(self.degree, self.planar)
        shaped = [e for e in entries if e["lobes"] is not None]
        index, j = [], 0
        for e in entries:
            index.append(j if e["lobes"] is not None else -1)
            j += e["lobes"] is not None
        shapes = (_shapes([e["centre"] for e in shaped], [e["lobes"] for e in shaped],
                          [e["coefficients"] for e in shaped] if terms else None, terms, self.planar)
                  if shaped else None)
        return SimpleNamespace(centres=np.array([e["centre"] for e in entries], dtype=np.float64).reshape(-1, 3),
                               radii=np.array([e["radius"] for e in entries], dtype=np.float64),
                               index=np.array(index, dtype=np.int64), shapes=shapes)

    @staticmethod
    def _subset(accepted, sites):
        return SimpleNamespace(centres=accepted.centres[sites], radii=accepted.radii[sites],
                               index=accepted.index[sites], shapes=accepted.shapes)

    def _extents(self, accepted, sites, directions):
        """e_i(û) of accepted sites (an empty site's virtual sphere: empty_radius in every direction)."""
        index = accepted.index[sites]
        result = np.full(len(sites), self.empty_radius)
        shaped = index >= 0
        if shaped.any():
            result[shaped] = _extent(accepted.shapes, index[shaped], directions[shaped])
        return result

    def _seed_extent(self, entry, directions):
        if entry["shape"] is None:
            return np.full(len(directions), self.empty_radius)
        return _extent(entry["shape"], np.zeros(len(directions), dtype=np.int64), directions)

    def _passes(self, points, entry, accepted):
        """K1 of proposals (B, 3) of one seed site against every accepted site: the two power tests with the label
        function's arithmetic and ‖c − c_i‖ ≥ (1 − κ) t_ik(û) + g for each site within reach; (B,) booleans."""
        packing = self.config.packing
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        ok = np.ones(len(points), dtype=bool)
        if not len(accepted.radii):
            return ok
        centres, radii, radius = accepted.centres, accepted.radii, entry["radius"]
        distance = None
        for a in range(3):
            delta = points[:, a, None] - centres[None, :, a]
            distance = delta * delta if distance is None else distance + delta * delta
        # pow_k(c) < pow_i(c) and pow_i(c_i) < pow_k(c_i); a centre's distance to itself is 0.
        ok &= np.all(0.0 - radius * radius < distance - radii * radii, axis=1)
        ok &= np.all(0.0 - radii * radii < distance - radius * radius, axis=1)
        d = np.sqrt(distance)
        b, s = np.nonzero((d < radii + radius + packing.gap) & ok[:, None])
        if len(b):
            u = (points[b] - centres[s]) / d[b, s, None]
            t = _contact(self._extents(accepted, s, u), self._seed_extent(entry, -u),
                         radii[s] * radii[s] - radius * radius)
            ok[b[d[b, s] < (1 - packing.overlap) * t + packing.gap]] = False
        return ok

    def _clear(self, points, radius, accepted):
        """Today's rule without packing: every site sphere keeps the gap to every accepted sphere."""
        if not len(accepted.radii):
            return np.ones(len(points), dtype=bool)
        d = np.linalg.norm(points[:, None, :] - accepted.centres[None, :, :], axis=-1)
        return np.all(d >= radius + accepted.radii + self.cells_config.gap, axis=1)

    def _contact_centre(self, rng, j, entry, accepted):
        """P1's contact proposal against site j: a direction with t > 0 (at most 16 draws), then c = c_j + D û,
        D increased by nextafter steps until the K1 tests against j hold. Returns (centre or None, found)."""
        packing = self.config.packing
        site = np.array([j])
        partner = self._subset(accepted, site)
        delta = accepted.radii[j] ** 2 - entry["radius"] ** 2
        for _ in range(16):
            u = _direction(rng, self.planar)
            t = float(_contact(self._extents(accepted, site, u[None]), self._seed_extent(entry, -u[None]), delta)[0])
            if t > 0:
                break
        else:
            return None, False
        distance = max((1 - packing.overlap) * t + packing.gap, math.sqrt(abs(delta)))
        for _ in range(_STEPS):
            centre = accepted.centres[j] + distance * u
            if self._passes(centre, entry, partner)[0]:
                return centre, True
            distance = float(np.nextafter(distance, np.inf))
        return None, True

    def _pairs(self, centres, phis, first, second):
        """The two nuclei of binucleate proposals at c ± (D_12 / 2) û_φ, D_12 the distance of P1 along û_φ (today's
        R_1 + R_2 + gap without packing), with nextafter steps until the tests between the two hold; NaN rows
        where they do not within the step limit."""
        packing = self.config.packing
        w = np.stack([np.zeros(len(phis)), np.sin(phis), np.cos(phis)], axis=1)
        r1, r2 = first["radius"], second["radius"]
        if packing is None:
            distance = np.full(len(phis), r1 + r2 + self.cells_config.gap)
        else:
            delta = r1 * r1 - r2 * r2
            t = _contact(self._seed_extent(first, -w), self._seed_extent(second, w), delta)
            distance = np.maximum((1 - packing.overlap) * t + packing.gap, math.sqrt(abs(delta)))
        one, two = np.full((len(phis), 3), np.nan), np.full((len(phis), 3), np.nan)
        todo = np.arange(len(phis))
        for _ in range(_STEPS):
            half = (distance[todo] / 2)[:, None] * w[todo]
            c1, c2 = centres[todo] + half, centres[todo] - half
            if packing is None:
                ok = np.linalg.norm(c1 - c2, axis=1) >= r1 + r2 + self.cells_config.gap
            else:
                ok = self._pair_ok(c1, c2, first, second)
            one[todo[ok]], two[todo[ok]] = c1[ok], c2[ok]
            todo = todo[~ok]
            if not len(todo):
                break
            distance[todo] = np.nextafter(distance[todo], np.inf)
        return one, two

    def _pair_ok(self, c1, c2, first, second):
        """K1 of the second nucleus at c2 against the first at c1, row by row (as ``_passes`` with one site)."""
        packing = self.config.packing
        r1, r2 = first["radius"], second["radius"]
        distance = None
        for a in range(3):
            delta = c2[:, a] - c1[:, a]
            distance = delta * delta if distance is None else distance + delta * delta
        ok = (0.0 - r2 * r2 < distance - r1 * r1) & (0.0 - r1 * r1 < distance - r2 * r2)
        d = np.sqrt(distance)
        near = np.flatnonzero(ok & (d < r1 + r2 + packing.gap))
        if len(near):
            u = (c2[near] - c1[near]) / d[near, None]
            t = _contact(self._seed_extent(first, u), self._seed_extent(second, -u), r1 * r1 - r2 * r2)
            ok[near[d[near] < (1 - packing.overlap) * t + packing.gap]] = False
        return ok

    def cellular(self, cells_config, layout, stream):
        """The cellular-scene hook: shapes, placement, territories and types."""
        config = self.config
        self.cells_config = cells_config
        self.empty_radius = float(cells_config.empty_radius)
        packing = config.packing
        self.planar = layout.flat or (packing is not None and packing.monolayer)
        self.z_mid = (layout.n[0] - 1) / 2 * layout.s[0] if packing is not None and packing.monolayer else None
        self.layout = layout
        if config.types is not None:
            types = config.types
            if len(types.abundances[0]) != self.codebook.n_genes:
                raise ValueError(f"abundances: rows have {len(types.abundances[0])} values; the codebook has "
                                 f"{self.codebook.n_genes} genes")
            genes = set(self.codebook.genes)
            for pair in types.marker_pairs:
                missing = [g for g in pair[:2] if g not in genes]
                if missing:
                    raise ValueError(f"marker_pairs names genes not in the codebook: {missing}")
            self._draw_regions(layout)
        # Every scene with a part set is placed and labelled by the cellular rules; a part left None takes the rule
        # the page gives it (one lobe, today's gap rule, T1), so no part falls back to the sample scene's code.
        nuclei, cells, sites = self._place(cells_config, layout, stream)
        if config.types is not None:
            for cell in cells:
                cell["type"], cell["region"] = self._type_at(cell["cell_id"], cell["anchor"])
        self.shapes = {int(x["nucleus_id"]): (x["lobes"], x["coefficients"]) for x in nuclei}
        self.cells = {int(cell["cell_id"]): dict(type=cell.get("type"), region=cell.get("region"),
                                                 growth=cell.get("growth")) for cell in cells}
        return nuclei, cells, sites

    def _first(self, batch, kind, k, accepted, scaled):
        """The first proposal of a batch, in drawing order, that passes the tests; (centres, seed) or None."""
        packing = self.config.packing
        rows = [i for i, proposal in enumerate(batch) if proposal["valid"]]
        if not rows:
            return None
        candidates = {}
        if kind == "binucleate":
            for group in ([rows] if not scaled else [[i] for i in rows]):
                one, two = self._pairs(np.array([batch[i]["centre"] for i in group]),
                                       np.array([batch[i]["phi"] for i in group]), *batch[group[0]]["seed"])
                for i, c1, c2 in zip(group, one, two):
                    if np.isfinite(c1).all():
                        candidates[i] = [c1, c2]
        else:
            candidates = {i: [batch[i]["centre"]] for i in rows}
        if scaled:
            candidates = {i: c for i, c in candidates.items()
                          if self._type_at(k, c[0])[0] == batch[i]["type_index"]}
        order = sorted(candidates)
        if not order:
            return None
        passing = np.ones(len(order), dtype=bool)
        for group in ([order] if not scaled else [[i] for i in order]):
            mask = np.isin(order, group)
            for slot, entry in enumerate(batch[group[0]]["seed"]):
                points = np.array([candidates[i][slot] for i in group])
                if packing is None:
                    passing[mask] &= self._clear(points, entry["radius"], accepted)
                else:
                    passing[mask] &= self._passes(points, entry, accepted)
        hits = np.flatnonzero(passing)
        if not len(hits):
            return None
        i = order[hits[0]]
        return candidates[i], batch[i]["seed"]

    def _place(self, cells_config, layout, stream):
        """Seeds in count or density mode, placed by P1 and K1 (today's gap rule without packing), F2's stop."""
        config, packing = self.config, self.config.packing
        lo, hi = layout.omega
        extent = hi - lo
        volume = float(extent[1] * extent[2]) if self.planar else float(np.prod(extent))
        density = None if packing is None else packing.density
        scaled = config.types is not None and config.types.size_scale is not None
        stop = packing is not None and packing.on_crowded == "stop"
        counts = (cells_config.nuclei, cells_config.binucleate, cells_config.empty)
        if density is None:
            plan = iter([kind for kind, count in zip(_KINDS, counts) for _ in range(count)])
            requested = sum(counts)
        else:
            shares = _probabilities(packing.kind_shares or counts, 3, "kind_shares")
            requested = None
        entries, nuclei, cells = [], [], []
        accepted = self._accepted([])
        total, k, next_nucleus, stopped = 0.0, 0, 1, None
        while True:
            if density is None:
                kind = next(plan, None)
                if kind is None:
                    break
            else:
                if total >= density * volume:
                    break
                kind = _KINDS[int(self.cstream("context.cell.kind", f"cell-{k + 1}").choice(3, p=shares))]
            k += 1
            own = list(range(next_nucleus, next_nucleus + {"ordinary": 1, "binucleate": 2, "empty": 0}[kind]))
            sizes = [self._sizes(cells_config, n, stream) for n in own]
            draws = [self._shape_draws(cells_config, n) for n in own]
            reach = _draw(cells_config.reach, "reach", f"cell-{k}",
                          lambda _name, entity: stream("sample.cell.reach", entity))
            if not (math.isfinite(reach) and reach > 0):
                raise ValueError("nonfinite or nonpositive cell reach")
            growth = None
            if self.rule == "grown":
                growth = _draw(config.territories.growth, "growth", f"cell-{k}",
                               lambda _name, entity: self.cstream("context.cell.growth", entity))
                if not (math.isfinite(growth) and growth > 0):
                    raise ValueError(f"cell-{k}: nonfinite or nonpositive growth")
            built = {}

            def seed_at(point, k=k, own=own, sizes=sizes, draws=draws, built=built):
                """The seed's sites for the type of its anchor at ``point`` (the type matters with a size scale)."""
                t = self._type_at(k, point)[0] if scaled else 0
                if t not in built:
                    scale = config.types.size_scale[t] if scaled else 1.0
                    built[t] = [dict(self._shape(size, scale, draw), nucleus=n)
                                for n, size, draw in zip(own, sizes, draws)] or [
                        dict(nucleus=None, lobes=None, coefficients=None, radius=self.empty_radius, nominal=0.0,
                             shape=None)]
                return t, built[t]

            rng = stream("sample.cell.placement", f"cell-{k}")
            contact_rng = self.cstream("context.cell.contact", f"cell-{k}") if (
                packing is not None and packing.contact > 0) else None

            def propose(rng=rng, contact_rng=contact_rng, kind=kind, accepted=accepted, seed_at=seed_at):
                """One proposal's draws in today's order: (contact draws,) the uniform centre, φ of a binucleate."""
                contact = (contact_rng is not None and len(accepted.radii) > 0
                           and contact_rng.random() < packing.contact)
                centre = None
                if contact:
                    partner = int(contact_rng.integers(len(accepted.radii)))
                    # With a size scale by type, the partner's centre stands in for the seed's region.
                    type_index, seed = seed_at(accepted.centres[partner])
                    centre, contact = self._contact_centre(contact_rng, partner, seed[0], accepted)
                if not contact:
                    # No contact draw, or no direction after 16 draws: uniformly in Ω as today.
                    centre = rng.uniform(lo, hi)
                    if layout.flat:
                        centre[0] = 0.0
                    elif self.z_mid is not None:
                        centre[0] = self.z_mid
                    type_index, seed = seed_at(centre)
                phi = rng.uniform(0.0, math.pi) if kind == "binucleate" else None
                valid = centre is not None and (not contact or bool(_inside(centre, layout.omega)))
                return dict(centre=centre, phi=phi, type_index=type_index, seed=seed, valid=valid)

            # Proposals are drawn ahead in growing batches and tested in drawing order; each stream belongs to this
            # seed alone, so draws past the accepted proposal change nothing.
            placed, drawn, size = None, 0, 1
            while placed is None and drawn < cells_config.max_proposals:
                size = min(size, cells_config.max_proposals - drawn)
                placed = self._first([propose() for _ in range(size)], kind, k, accepted, scaled)
                drawn += size
                size = min(2 * size, _CHUNK)
            if placed is None:
                message = (f"cell-{k}: no proposal accepted after max_proposals={cells_config.max_proposals} "
                           "(the scene domain is too crowded for the seeds)")
                if not stop:
                    raise ValueError(message)
                stopped = f"cell-{k}"
                break
            centres, seed = placed
            for c, e in zip(centres, seed):
                entries.append(dict(centre=c, radius=e["radius"], lobes=e["lobes"], coefficients=e["coefficients"]))
                if e["nucleus"] is not None:
                    axial, lateral, elongation, angle = e["lobes"][0][1:]
                    nuclei.append(dict(nucleus_id=e["nucleus"], cell_id=k, centre=c, axial=axial, lateral=lateral,
                                       elongation=elongation, angle=angle, site_radius=e["radius"],
                                       lobes=e["lobes"], coefficients=e["coefficients"]))
                    total += e["nominal"]
            cells.append(dict(cell_id=k, kind=kind, anchor=centres[0], n_nuclei=len(own), reach=reach,
                              growth=growth, sites=[(c, e["radius"]) for c, e in zip(centres, seed)]))
            next_nucleus += len(own)
            accepted = self._accepted(entries)
        reached = total / volume
        if stopped is not None:
            if density is not None and reached < packing.floor * density:
                raise ValueError(f"{stopped}: placement stopped at a reached density of {reached} below "
                                 f"floor · density = {packing.floor} · {density} = {packing.floor * density}")
            if density is None and len(cells) < packing.floor * requested:
                raise ValueError(f"{stopped}: placement stopped after {len(cells)} placed seeds, below "
                                 f"floor · requested = {packing.floor} · {requested} = {packing.floor * requested}")
        self.packing = dict(mode="count" if density is None else "density", density=density,
                            reached_density=reached, nominal_size=total, domain_size=volume,
                            domain_unit="µm²" if self.planar else "µm³", placed_seeds=len(cells),
                            requested_seeds=requested, stopped=stopped is not None, stopped_at=stopped,
                            placed_nuclei=len(nuclei))
        terms = _harmonic_terms(self.degree, self.planar)
        return nuclei, cells, _sites(cells, nuclei, self.rule, self.empty_radius, terms, self.planar)

    # ------------------------------------------------------------------ molecules

    def _abundances(self):
        """Normalized type rows, the ambient row (None: the sample scene's draw) and each gene's role.

        Roles follow G1's layout, not the rows ("Cell types"): with K types, genes 1 to 2K are the markers (type t
        has genes 2t − 1 and 2t) and the genes after them are shared. A2's default ambient abundance is the shared
        genes, uniform ("Molecules"), so a change of one type's row changes no other type's genes and no ambient gene.
        """
        types, molecules = self.config.types, self.config.molecules
        n_genes = self.codebook.n_genes
        rows = roles = None
        if types is not None:
            rows = [_probabilities(row, n_genes, f"abundances[{t}]") for t, row in enumerate(types.abundances)]
            markers = 2 * len(types.types)
            roles = ["marker" if g < markers else "shared" for g in range(n_genes)]
        ambient = None
        if molecules is not None and molecules.ambient_abundances is not None:
            if len(molecules.ambient_abundances) != n_genes:
                raise ValueError(f"ambient_abundances has {len(molecules.ambient_abundances)} values; the codebook "
                                 f"has {n_genes} genes")
            ambient = _probabilities(molecules.ambient_abundances, n_genes, "ambient_abundances")
        elif types is not None:
            shared = np.array([role == "shared" for role in roles], dtype=np.float64)
            if not shared.any():
                raise ValueError(f"ambient_abundances: the codebook's {n_genes} genes are all markers of the "
                                 f"{len(types.types)} types (two per type, G1), so A2 has no shared gene; give "
                                 "ambient_abundances")
            ambient = shared / shared.sum()
        return rows, ambient, roles

    def molecules(self, config, layout, codebook, cells, sites, stream):
        """The molecule hook: counts (M2), compartments (NF1) and genes by type (G1, A2)."""
        context = self.config
        types, molecules = context.types, context.molecules
        lo, hi = layout.omega
        rows, ambient_row, _ = self._abundances()
        spread = molecules.per_cell_spread if molecules is not None else 0.0
        plan = []
        for cell in cells or []:
            k = cell["cell_id"]
            t = cell.get("type")
            median = (types.molecules_per_cell[t] if types is not None and types.molecules_per_cell is not None
                      else molecules.per_cell_median if molecules is not None else config.molecules_per_cell)
            fraction = (types.nuclear_fraction[t] if types is not None and types.nuclear_fraction is not None
                        else molecules.nuclear_fraction if molecules is not None else None)
            mean = float(median)
            if spread > 0 and median > 0:
                normal = self.cstream("context.molecule.mean", f"cell-{k}").standard_normal()
                with np.errstate(over="raise"):
                    try:
                        mean = float(np.exp(math.log(median) + spread * normal))
                    except FloatingPointError:
                        raise ValueError(f"cell-{k}: nonfinite molecule mean") from None
            self.means[k] = mean
            count = int(stream("sample.molecule.count", f"cell-{k}").poisson(mean))
            plan.append((cell, count, fraction))
        density = (molecules.ambient_density if molecules is not None and molecules.ambient_density is not None
                   else config.ambient_density)
        ambient = int(stream("sample.molecule.count", "ambient").poisson(density * float(np.prod(hi - lo))))
        total = sum(count for _, count, _ in plan) + ambient
        if total > config.max_count:
            raise ValueError(f"max_count: {total} molecules would be generated in the scene domain, "
                             f"above max_count={config.max_count}")
        limit = config.cells.max_proposals if config.cells is not None else 1
        ids, population, positions, cell_type = [], [], [], []
        for cell, count, fraction in plan:
            k = cell["cell_id"]
            nucleus_sites = [(c, r) for (c, r), n in zip(cell["sites"], self._site_nuclei(cell, sites)) if n]
            if self.rule == "power":
                territory = (lo, hi)
            else:
                # The site sphere grown by the reach (T1) or by δ (T2; an empty site's radius is empty_radius) bounds
                # the territory; summed as the sample scene sums it, so T1 proposes exactly today's points.
                margin = cell["reach"] if self.rule == "reach" else cell["growth"]
                territory = (np.maximum(np.min([c - r - margin for c, r in cell["sites"]], axis=0), lo),
                             np.minimum(np.max([c + r + margin for c, r in cell["sites"]], axis=0), hi))
            nucleus_box = None
            if nucleus_sites:
                nucleus_box = (np.maximum(np.min([c - r for c, r in nucleus_sites], axis=0), lo),
                               np.minimum(np.max([c + r for c, r in nucleus_sites], axis=0), hi))
            for j in range(count):
                identity = f"c{k}-m{j}"
                where = "territory"
                if fraction is not None and nucleus_sites:
                    u = self.cstream("context.molecule.compartment", identity).random()
                    where = "nucleus" if u < fraction else "cytoplasm"
                box = nucleus_box if where == "nucleus" else territory
                point = self._place_molecule(stream("sample.molecule.placement", identity), box, k, where, sites,
                                             layout, limit, identity)
                ids.append(identity)
                population.append("cell")
                positions.append(point)
                cell_type.append(cell.get("type"))
        for j in range(ambient):
            identity = f"a-m{j}"
            point = stream("sample.molecule.placement", identity).uniform(lo, hi)
            if layout.flat:
                point[0] = 0.0
            ids.append(identity)
            population.append("ambient")
            positions.append(point)
            cell_type.append(None)
        template = config.fov
        specs = (template.brightness, template.axial_width, template.lateral_width, template.elongation,
                 template.angle)
        names = ("brightness", "width.axial", "width.lateral", "elongation", "angle")
        for name, spec in zip(names, specs):
            _checked_distribution(spec, name, tuple(ids), f"fov.{name}")
        names_of_genes, n_genes = list(codebook.genes), codebook.n_genes
        abundances = _probabilities(template.abundances if template.abundances is not None
                                    else np.ones(n_genes), n_genes, "abundances")
        genes, values = [], np.empty((len(ids), 5), dtype=np.float64)
        for i, identity in enumerate(ids):
            if population[i] == "cell" and rows is not None:
                gene = self.cstream("context.identity", identity).choice(n_genes, p=rows[cell_type[i]])
            elif population[i] == "ambient" and ambient_row is not None:
                gene = self.cstream("context.ambient.identity", identity).choice(n_genes, p=ambient_row)
            else:
                gene = stream("sample.identity", identity).choice(n_genes, p=abundances)
            genes.append(names_of_genes[gene])
            values[i] = [_draw(spec, name, identity, lambda component, entity: stream("sample." + component, entity))
                         for name, spec in zip(names, specs)]
        if not np.isfinite(values).all() or (values[:, 1:3] <= 0).any():
            raise ValueError("nonfinite or out-of-domain molecule properties")
        return (tuple(ids), population, np.array(positions, dtype=np.float64).reshape(-1, 3), genes, values,
                abundances)

    @staticmethod
    def _site_nuclei(cell, sites):
        rows = np.flatnonzero(sites.cell == cell["cell_id"])
        return [bool(sites.nucleus[r]) for r in rows]

    @staticmethod
    def _place_molecule(rng, box, k, where, sites, layout, limit, identity):
        """The first of up to ``limit`` uniform proposals in ``box`` that lies in the compartment of cell k."""
        drawn, size = 0, 4
        while drawn < limit:
            size = min(size, limit - drawn)
            points = rng.uniform(box[0], box[1], size=(size, 3))
            if layout.flat:
                points[:, 0] = 0.0
            cell, nucleus = _label_functions(points, sites, layout.omega)
            hit = cell == k
            if where == "nucleus":
                hit &= nucleus > 0
            elif where == "cytoplasm":
                hit &= nucleus == 0
            found = np.flatnonzero(hit)
            if len(found):
                return points[found[0]]
            drawn += size
            size = min(2 * size, _CHUNK)
        compartment = {"territory": "", "nucleus": " nucleus", "cytoplasm": " cytoplasm"}[where]
        raise ValueError(f"{identity}: no proposal accepted in the{compartment} of cell {k} after "
                         f"max_proposals={limit}" + (" (no room outside its nucleus)" if where == "cytoplasm" else ""))


def _payload(config):
    payload = dict(type="CellularSceneConfig", sample=_config_payload(config.sample))
    for name in ("nuclei", "packing", "territories", "types", "molecules"):
        part = getattr(config, name)
        payload[name] = None if part is None else dict(_plain(asdict(part)), type=type(part).__name__)
    payload["stains"] = [dict(_plain(asdict(stain)), type="StainChannelConfig") for stain in config.stains]
    morphology = config.morphology
    payload["morphology"] = None if morphology is None else dict(_plain(asdict(morphology)),
                                                                 type="MorphologyRoundConfig")
    return payload


def _tables(config, codebook, scene, state):
    """The cellular-scene tables of this delivery from the sample tables and the context draws."""
    nuclei, cells, molecules = scene.nuclei, scene.cells, scene.molecules
    types = config.types
    lobe_rows, harmonic_rows = [], []
    shapes = state.shapes if state is not None else {}
    terms = _harmonic_terms(state.degree, state.planar) if state is not None else []
    for x in nuclei.itertuples():
        lobes, coefficients = shapes.get(int(x.nucleus_id), (None, None))
        if lobes is None:
            lobes = [((0.0, 0.0, 0.0), x.axial_radius, x.lateral_radius, x.elongation, x.angle)]
        for l, (o, axial, lateral, elongation, angle) in enumerate(lobes, start=1):
            lobe_rows.append((x.nucleus_id, l, *o, axial, lateral, elongation, angle))
        if coefficients is not None:
            for (degree, order), value in zip(terms, coefficients):
                harmonic_rows.append((x.nucleus_id, degree, order, float(value)))
    lobes = pd.DataFrame(lobe_rows, columns=["nucleus_id", "lobe", "offset_z", "offset_y", "offset_x",
                                             "axial_radius", "lateral_radius", "elongation", "angle"])
    lobes = lobes.astype(dict(nucleus_id=np.uint32, lobe=np.int64, **{c: np.float64 for c in lobes.columns[2:]}))
    harmonics = pd.DataFrame(harmonic_rows, columns=["nucleus_id", "degree", "order", "coefficient"])
    harmonics = harmonics.astype(dict(nucleus_id=np.uint32, degree=np.int64, order=object, coefficient=np.float64))

    info = state.cells if state is not None else {}
    names = types.types if types is not None else ()
    context_rows = []
    for cell in cells.itertuples():
        own = info.get(int(cell.cell_id), {})
        t, region = own.get("type"), own.get("region")
        mean = state.means.get(int(cell.cell_id)) if state is not None else None
        context_rows.append((cell.cell_id, cell.kind, None if t is None else names[t],
                             None if region is None else region + 1,
                             config.sample.molecules_per_cell if mean is None else mean,
                             np.nan if own.get("growth") is None else own["growth"]))
    cell_context = pd.DataFrame(context_rows, columns=["cell_id", "kind", "type", "region", "molecule_mean",
                                                       "growth"])
    cell_context = cell_context.astype(dict(cell_id=np.uint32, kind=object, type=object, region="UInt32",
                                            molecule_mean=np.float64, growth=np.float64))

    region_rows = [] if state is None else [(r["region_id"], *r["centre"], names[r["main_type"]])
                                             for r in state.regions]
    regions = pd.DataFrame(region_rows, columns=["region_id", "centre_z", "centre_y", "centre_x", "main_type"])
    regions = regions.astype(dict(region_id=np.uint32, centre_z=np.float64, centre_y=np.float64,
                                  centre_x=np.float64, main_type=object))

    abundance_rows = []
    if state is not None:
        rows, ambient, roles = state._abundances()
        genes = list(codebook.genes)
        for t, row in enumerate(rows or []):
            abundance_rows.extend((names[t], g, float(w), roles[i]) for i, (g, w) in enumerate(zip(genes, row)))
        if ambient is not None:
            abundance_rows.extend(("ambient", g, float(w), None if roles is None else roles[i])
                                  for i, (g, w) in enumerate(zip(genes, ambient)))
    type_abundances = pd.DataFrame(abundance_rows, columns=["type", "gene_id", "weight", "role"])
    type_abundances = type_abundances.astype(dict(type=object, gene_id=object, weight=np.float64, role=object))

    order = {g: i for i, g in enumerate(codebook.genes)}
    owned = molecules[molecules.true_cell_id.notna()]
    count_rows = []
    for (cell_id, gene), group in owned.groupby([owned.true_cell_id.astype(np.int64), owned.gene_id], sort=False):
        count_rows.append((cell_id, gene, len(group), int((group.compartment == "nucleus").sum())))
    count_rows.sort(key=lambda row: (row[0], order[row[1]]))
    cell_gene_counts = pd.DataFrame(count_rows, columns=["cell_id", "gene_id", "count", "nuclear_count"])
    cell_gene_counts = cell_gene_counts.astype(dict(cell_id=np.uint32, gene_id=object, count=np.int64,
                                                    nuclear_count=np.int64))

    pair_rows = []
    if types is not None:
        sample_cells = cells.cell_id[cells.in_sample].astype(np.int64).tolist()
        counts = {(int(r.cell_id), r.gene_id): int(r.count) for r in cell_gene_counts.itertuples()}
        for gene_a, gene_b, relation in types.marker_pairs:
            a = np.array([counts.get((c, gene_a), 0) for c in sample_cells]) >= 1
            b = np.array([counts.get((c, gene_b), 0) for c in sample_cells]) >= 1
            both, either = int((a & b).sum()), int((a | b).sum())
            pair_rows.append((gene_a, gene_b, relation, int(a.sum()), int(b.sum()), both, either,
                              both / either if either else np.nan))
    marker_pairs = pd.DataFrame(pair_rows, columns=["gene_a", "gene_b", "relation", "cells_a", "cells_b", "both",
                                                    "either", "mecr"])
    marker_pairs = marker_pairs.astype(dict(gene_a=object, gene_b=object, relation=object, cells_a=np.int64,
                                            cells_b=np.int64, both=np.int64, either=np.int64, mecr=np.float64))
    return dict(lobes=lobes, harmonics=harmonics, cell_context=cell_context, regions=regions,
                type_abundances=type_abundances, cell_gene_counts=cell_gene_counts, marker_pairs=marker_pairs)


def generate_cellular_scene(codebook: Codebook, *, config: CellularSceneConfig = CellularSceneConfig(),
                            on_fov: Callable | None = None) -> CellularScene:
    """Generate one cellular scene and acquire it as a grid of FOVs at true offsets.

    The sample scene of ``config.sample`` is generated by the sample scene's own code
    (``generate_sample_scene``), with the placement, label and molecule rules of the parts
    given (design/specs/synthetic/cellular-scene.md): lobed nuclei, packing with touching nuclei,
    territories, cell types and molecules by compartment. With every part at its default the
    result's ``sample`` equals ``generate_sample_scene(codebook, config=config.sample)``.
    The stains are then rendered per FOV from the truth (the reference round's at the FOV's
    voxel centres, the morphology round's at the centres moved by its translation); a
    morphology round grows the scene domain by its translation bound on each side.
    ``on_fov`` is passed on as there; the stains stay in the result. Invalid input, crowding
    (with ``on_crowded="raise"`` or below the floor) and counts above ``max_count`` raise
    ValueError; nothing is written.
    """
    if not isinstance(config, CellularSceneConfig):
        raise TypeError("config must be CellularSceneConfig")
    if not isinstance(codebook, Codebook):
        raise TypeError("codebook must be Codebook")
    template = config.sample.fov
    log = {}
    stream = _streams(template.split, template.seed, template.scene_key, log)
    parts = any(getattr(config, name) is not None for name in ("nuclei", "packing", "territories", "types",
                                                                 "molecules"))
    _check_stain_names(config, codebook)
    grow = None if config.morphology is None else np.asarray(config.morphology.translation_max_zyx, dtype=np.float64)
    state = None
    if parts:
        state = _Context(config, codebook, stream)
        # The molecules are placed over the territories of the selected rule and drawn by the page's laws, whichever
        # parts are set; only a scene with every part off is generate_sample_scene's by construction (O1).
        scene = _generate(codebook, config.sample, on_fov, state.cellular, state.molecules, grow=grow)
    else:
        scene = _generate(codebook, config.sample, on_fov, grow=grow)
    tables = _tables(config, codebook, scene, state)
    effective = dict(
        type="CellularSceneConfig", territory_rule=state.rule if state is not None else "reach",
        planar=bool(state.planar) if state is not None else bool(_layout(config.sample.acquisition).flat),
        z_mid=None if state is None or state.z_mid is None else float(state.z_mid),
        harmonic_terms=[list(t) for t in (_harmonic_terms(state.degree, state.planar) if state is not None else [])],
        harmonic_bound=(None if state is None or state.degree < 2
                        else _harmonic_bound(state.degree, state.planar)),
        empty_radius=None if config.sample.cells is None else float(config.sample.cells.empty_radius),
        packing=None if state is None else state.packing,
        types=None if config.types is None else dict(
            names=list(config.types.types),
            proportions=_probabilities(config.types.proportions, len(config.types.types), "proportions").tolist()))
    truth = SimpleNamespace(sample=scene, record=dict(effective_config=_plain(effective)), **tables)
    stains = _stain_part(config, truth, stream, log)
    effective.update(stains=stains.pop("effective"), morphology=stains.pop("morphology_effective"))
    record = dict(
        generator="starfinder.synthetic.generate_cellular_scene", context_truth_version=CELLULAR_TRUTH_VERSION,
        context_preset_version=CELLULAR_PRESET_VERSION, requested_config=_payload(config),
        effective_config=_plain(effective), seed=template.seed,
        stream_scheme=dict(key=list(_STREAM_KEY), namespace=_STREAM_NAMESPACE, encoding="compact UTF-8 JSON array",
                           digest="SHA-256, big-endian integer", bit_generator="PCG64", numpy_version=np.__version__,
                           components={k: dict(entity=e, draws=d) for k, (e, d) in _CONTEXT_COMPONENTS.items()},
                           streams=list(log.values())))
    return CellularScene(scene, stains.pop("images"), stains.pop("morphology"), **tables, **stains, record=record)


def _check_stain_names(config, codebook):
    """The stains' files beside the amplicon channels and rounds: a reference stain is not an amplicon channel, and
    the morphology round is not an amplicon round."""
    shared = sorted({stain.channel for stain in config.stains if stain.round == "reference"}
                    & set(codebook.channel_labels))
    if shared:
        raise ValueError(f"stains: the reference-round channels {shared} are amplicon channels of the codebook; a "
                         "reference stain is a file of its own")
    morphology = config.morphology
    if morphology is not None and morphology.name in (*codebook.round_labels, "reference_stain"):
        raise ValueError(f"morphology: the round name {morphology.name!r} is an amplicon round or reserved")


def _stain_part(config, truth, stream, log):
    """The stains and the morphology round: draws, truth tables and per-FOV images ("Stains", "The morphology round").

    Stains that share a name share every draw; each stain's noise is drawn per FOV from ``stain.noise.*`` with scene
    key ``[scene_key, fov_id]``, the stain name as channel label and, in the morphology round, its name as round label.
    """
    stains, morphology = config.stains, config.morphology
    template = config.sample.fov
    sample = truth.sample
    record = sample.record
    grid = record["sample_grid"]
    origin = np.array(grid["origin_zyx"], dtype=np.float64)
    s = np.array(grid["spacing_zyx"], dtype=np.float64)
    shape = np.array(grid["shape_zyx"], dtype=np.int64)
    domain = record["domain"]["omega"]
    omega = (np.array(domain["lo_zyx"], dtype=np.float64), np.array(domain["hi_zyx"], dtype=np.float64))
    n = np.array(config.sample.acquisition.fov_shape_zyx, dtype=np.int64)
    flat = bool(n[0] == 1)
    fov_ids = list(sample.placement.fov_id)
    true = sample.placement[["true_origin_z", "true_origin_y", "true_origin_x"]].to_numpy(dtype=np.float64)
    sites, _ = _scene_sites(truth)
    nuclei = sorted(int(v) for v in sample.nuclei.nucleus_id)
    cells = sorted(int(v) for v in sample.cells.cell_id)
    limit = config.sample.cells.max_proposals if config.sample.cells is not None else 1
    first, parts = {}, {}
    rows = {name: [] for name in _STAIN_TABLES}
    for stain in stains:
        if stain.name in first:
            continue
        first[stain.name] = stain
        nucleus = stain.source == "nucleus"
        ids = nuclei if nucleus else cells
        values = _plateaus(stain, [f"{'nucleus' if nucleus else 'cell'}-{i}" for i in ids], stream)
        rows["stain_objects"].extend((stain.name, "nucleus" if nucleus else "cell", i, v) for i, v in zip(ids, values))
        if stain.texture is not None and ids:
            rows["bright_regions"].extend((b, i, stain.name, *rest) for b, i, *rest in _bright_regions(
                stain, ids, sites, omega, flat, limit, stream))
        if stain.unevenness is not None:
            bumps, length = _bumps(stain, omega, (n[1] * s[1], n[2] * s[2]), flat, stream)
            rows["unevenness"].extend((stain.name, q, cz, cy, cx, z, float(stain.unevenness.amplitude), length)
                                      for q, cz, cy, cx, z in bumps)
        components, background = _prepare_background(
            stain.background, shape, 1,
            lambda component, entity=None, name=stain.name: stream("context.stain." + component, entity, None, name), 1)
        parts[stain.name] = (components, background, _noise(stain.noise), values)
    effective = []
    for stain in stains:
        components, background, noise, values = parts[stain.name]
        median = _median(stain.brightness, values)
        scale = 1.0
        if stain.output is not None:
            if not median > 0:
                raise ValueError(f"stains[{stain.name!r}].output: the median brightness m_A must be positive; got "
                                 f"{median}")
            scale = float(stain.output.plateau) / median
        dtype = stain.output.dtype if stain.output is not None else template.dtype
        rows["stain_channels"].append((stain.name, stain.source, stain.round, stain.channel, dtype, scale, median))
        effective.append(dict(name=stain.name, source=stain.source, round=stain.round, channel=stain.channel,
                              dtype=dtype, scale=scale, m_A=median, background=background, noise=noise))
    shift = {}
    if morphology is not None:
        bound = np.asarray(morphology.translation_max_zyx, dtype=np.float64)
        for fov in fov_ids:
            if morphology.translation == "uniform":
                vox = bound * (2 * stream("context.morphology.translation", fov).random(3) - 1) + 0.0
            else:
                vox = np.asarray(morphology.translation_zyx[fov], dtype=np.float64) + 0.0
            shift[fov] = vox * s
            rows["morphology_truth"].append((fov, morphology.name, *shift[fov].tolist(), *vox.tolist(),
                                             *(-vox + 0.0).tolist()))
    tables = {name: _stain_table(name, value) for name, value in rows.items()}
    model = _model(sites, [_plain(asdict(stain)) for stain in stains], tables["stain_objects"],
                   tables["bright_regions"], tables["unevenness"])
    images, rounds = {}, {}
    for k, fov in enumerate(fov_ids):
        fov_stream = _streams(template.split, template.seed, _json([template.scene_key, fov]), log)
        images[fov], planes = {}, []
        for stain, channel in zip(stains, effective):
            components, background, noise, _ = parts[stain.name]
            start = true[k] if stain.round == "reference" else true[k] + shift[fov]
            label = None if stain.round == "reference" else morphology.name
            plane = model.signal(stain.name, _Points(origin=start, spacing=s, shape=n))
            tissue = (evaluate_background_translated(components, n, -((start - origin) / s), np.float64)[0]
                      if components else None)
            _observe(plane, tissue, background, 0, 0, noise, label, stain.name,
                     lambda component, entity=None, round_label=None, channel_label=None:
                     fov_stream("stain." + component, entity, round_label, channel_label))
            if not np.isfinite(plane).all():
                raise ValueError(f"nonfinite rendered stain {stain.name!r}")
            if stain.output is not None:
                plane *= channel["scale"]
            image = _cast(plane, channel["dtype"])
            if stain.round == "reference":
                images[fov][stain.name] = image
            else:
                planes.append(image)
        if planes:
            rounds[fov] = np.stack(planes, axis=-1)
    morphology_effective = None
    if morphology is not None:
        morphology_effective = dict(name=morphology.name, translation=morphology.translation,
                                    translation_max_zyx=[float(v) for v in morphology.translation_max_zyx],
                                    domain_growth_um=(np.asarray(morphology.translation_max_zyx) * s).tolist(),
                                    stains=[stain.name for stain in stains if stain.round == "morphology"])
    return dict(images={fov: own for fov, own in images.items() if own}, morphology=rounds, effective=effective,
                morphology_effective=morphology_effective, **tables)


def _scene_sites(scene):
    """(sites, Ω) of a cellular scene rebuilt from its truth tables (``nuclei``, ``lobes``, ``harmonics``, ``cells``,
    ``cell_context``) and records, as the generator's label function takes them; sites None without a cell."""
    sample = scene.sample
    effective = scene.record["effective_config"]
    domain = sample.record["domain"]["omega"]
    omega = (np.array(domain["lo_zyx"], dtype=np.float64), np.array(domain["hi_zyx"], dtype=np.float64))
    if not len(sample.cells):
        return None, omega
    planar = bool(effective["planar"])
    terms = [tuple(t) for t in effective["harmonic_terms"]]
    by_nucleus = {}
    for row in scene.lobes.itertuples():
        by_nucleus.setdefault(int(row.nucleus_id), []).append(
            ((row.offset_z, row.offset_y, row.offset_x), row.axial_radius, row.lateral_radius, row.elongation,
             row.angle))
    coefficients = {}
    index = {term: j for j, term in enumerate(terms)}
    for row in scene.harmonics.itertuples():
        coefficients.setdefault(int(row.nucleus_id), np.zeros(len(terms)))[index[row.degree, row.order]] = \
            row.coefficient
    growth = dict(zip(scene.cell_context.cell_id.astype(np.int64), scene.cell_context.growth))
    nuclei = [dict(nucleus_id=int(x.nucleus_id), cell_id=int(x.cell_id),
                   centre=np.array([x.centre_z, x.centre_y, x.centre_x]), site_radius=x.site_radius,
                   lobes=by_nucleus[int(x.nucleus_id)],
                   coefficients=coefficients.get(int(x.nucleus_id), np.zeros(len(terms))) if terms else None)
              for x in sample.nuclei.itertuples()]
    own = {}
    for x in sorted(nuclei, key=lambda x: x["nucleus_id"]):
        own.setdefault(x["cell_id"], []).append((x["centre"], x["site_radius"]))
    empty_radius = effective["empty_radius"]
    cells = []
    for cell in sample.cells.itertuples():
        k = int(cell.cell_id)
        sites = own.get(k) or [(np.array([cell.anchor_z, cell.anchor_y, cell.anchor_x]), empty_radius)]
        value = growth.get(k, np.nan)
        cells.append(dict(cell_id=k, reach=cell.reach, growth=None if np.isnan(value) else value, sites=sites))
    return _sites(cells, nuclei, effective["territory_rule"], empty_radius, terms, planar), omega


def _context_labels(scene, points):
    """The point evaluator: C and N of a cellular scene at (..., 3) sample positions (µm), rebuilt from its truth
    tables (``nuclei``, ``lobes``, ``harmonics``, ``cells``, ``cell_context``) and records, evaluated by the
    generator's own label function."""
    sites, omega = _scene_sites(scene)
    points = np.asarray(points, dtype=np.float64)
    if sites is None:
        return _labels(points, SimpleNamespace(centres=np.zeros((0, 3))), omega, _BLOCK)
    return _labels(points, sites, omega, _BLOCK)


def _context_model(scene):
    """The stain functions of a cellular scene rebuilt from its truth tables (``stain_objects``, ``bright_regions``,
    ``unevenness``), its label sites and the requested stain configurations."""
    sites, _ = _scene_sites(scene)
    return _model(sites, scene.record["requested_config"]["stains"], scene.stain_objects, scene.bright_regions,
                  scene.unevenness)


def _context_stain(scene, name, points):
    """The point evaluator of the stains: the pre-noise signal of stain ``name`` (without its background) at (..., 3)
    sample positions (µm), by the functions the generator renders with."""
    points = np.asarray(points, dtype=np.float64)
    return _context_model(scene).signal(name, _Points(points.reshape(-1, 3))).reshape(points.shape[:-1])


def _context_band(scene, name, cell_id, points):
    """(d_k, β_k) of cell ``cell_id`` for the boundary stain ``name`` at (..., 3) sample positions; NaN beyond the
    cell's box (where its band and territory are 0)."""
    points = np.asarray(points, dtype=np.float64)
    d, beta = _context_model(scene).band(name, int(cell_id), _Points(points.reshape(-1, 3)))
    return d.reshape(points.shape[:-1]), beta.reshape(points.shape[:-1])


def _context_field(scene, name, points):
    """U of stain ``name`` at (..., 3) sample positions (ones without a field)."""
    points = np.asarray(points, dtype=np.float64)
    return _context_model(scene).field(name, _Points(points.reshape(-1, 3))).reshape(points.shape[:-1])


_CONTEXT_TABLES = ("lobes", "harmonics", "cell_context", "regions", "type_abundances", "cell_gene_counts",
                   "marker_pairs", *_STAIN_TABLES)


def save_cellular_scene(scene: CellularScene, directory: Path | str) -> Path:
    """Write a cellular scene to a new or empty directory and return its path.

    Layout (design/specs/synthetic/cellular-scene.md, "Truth schema"): the ``save_sample_scene``
    layout of ``scene.sample``, unchanged, plus ``lobes.csv``, ``harmonics.csv``,
    ``cell_context.csv``, ``regions.csv``, ``type_abundances.csv``, ``cell_gene_counts.csv``,
    ``marker_pairs.csv``, ``stain_channels.csv``, ``stain_objects.csv``,
    ``bright_regions.csv``, ``unevenness.csv``, ``morphology_truth.csv``, per FOV
    ``fovs/<fov_id>/stains/<name>.ome.tif`` (each reference-round stain, ZYX as one channel
    of an OME-TIFF in its own dtype) and ``fovs/<fov_id>/morphology.ome.tif`` (the morphology
    round, ZYXC), each with the FOV's metadata, and ``context.json`` (the record with
    ``context_truth_version`` and the SHA-256 of every other file). An empty population gives
    a header and no rows. A scene generated with ``on_fov`` raises ValueError; an existing
    nonempty directory raises FileExistsError, and nothing is overwritten.
    """
    if not isinstance(scene, CellularScene):
        raise TypeError("scene must be CellularScene")
    directory = save_sample_scene(scene.sample, directory)
    for name in _CONTEXT_TABLES:
        getattr(scene, name).to_csv(directory / f"{name}.csv", index=False)
    for fov in scene.sample.positions.fov_id:
        metadata = scene.sample.fov_metadata[fov]
        for name, image in scene.stains.get(fov, {}).items():
            save_volume(image[..., None], directory / "fovs" / fov / "stains" / f"{name}.ome.tif", metadata=metadata)
        if fov in scene.morphology:
            save_volume(scene.morphology[fov], directory / "fovs" / fov / "morphology.ome.tif", metadata=metadata)
    files = {path.relative_to(directory).as_posix(): file_sha256(path)
             for path in sorted(directory.rglob("*")) if path.is_file()}
    record = dict(scene.record, files=files)
    (directory / "context.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return directory


def save_cellular_dataset(scene: CellularScene, directory: Path | str) -> Path:
    """Write a cellular scene as a raw dataset that ``Dataset`` and ``FOV`` read; return its path.

    Layout (design/specs/synthetic/cellular-scene.md, "Files and layout"): ``<round>/<fov_id>/<channel>.tif``
    for each amplicon round (one ZYX file per codebook channel), the reference-round stains
    as files of the reference round's folder (``<reference round>/<fov_id>/<channel>.tif``),
    the morphology round's stains in ``<morphology round>/<fov_id>/<channel>.tif``, every file
    with the FOV's reference metadata; ``codebook.csv`` (``entry_id``, ``gene_id``,
    ``color_sequence``); and ``dataset.json`` with the ``Dataset`` fields (rounds,
    ``channel_order``, ``other_channel_order`` and ``reference_stains`` with pattern and name,
    the voxel sizes, the codebook's labels and FOV IDs) and the SHA-256 of every other file.
    Each stain's channel name is its stain name, so ``FOV.prepare_morphology`` without a
    recipe registers the morphology round on the one reference stain that shares a name
    with one of its channels (MR1). A scene generated with ``on_fov`` raises ValueError, as
    does a channel pattern that selects more than one file of its folder; an existing
    nonempty directory raises FileExistsError, and nothing is overwritten.
    """
    if not isinstance(scene, CellularScene):
        raise TypeError("scene must be CellularScene")
    sample = scene.sample
    fov_ids = list(sample.positions.fov_id)
    if set(sample.fovs) != set(fov_ids):
        raise ValueError("the scene holds no FOV images (generated with on_fov); save each FOV in the callback")
    record = sample.record
    codebook = record["codebook"]
    reference = record["effective_config"]["reference_round"]
    stains = scene.record["effective_config"]["stains"]
    morphology = scene.record["effective_config"]["morphology"]
    folders = {label: list(codebook["channel_labels"]) for label in codebook["round_labels"]}
    folders[reference] = folders[reference] + [s["channel"] for s in stains if s["round"] == "reference"]
    if morphology is not None:
        folders[morphology["name"]] = [s["channel"] for s in stains if s["round"] == "morphology"]
    for folder, channels in folders.items():
        for channel in channels:
            matches = [other for other in channels if channel in other]
            if len(matches) != 1:
                raise ValueError(f"the channel pattern {channel!r} selects {len(matches)} files of round {folder!r} "
                                 f"({matches}); patterns must select one file each")
    for fov in fov_ids:
        for name in (fov, *folders):
            if name in (".", "..") or Path(name).name != name or "\\" in name:
                raise ValueError(f"FOV ID or round is not a plain file name: {name!r}")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"dataset directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    for fov in fov_ids:
        metadata = sample.fov_metadata[fov]
        for label in codebook["round_labels"]:
            image = sample.fovs[fov].rounds[label]
            for c, channel in enumerate(codebook["channel_labels"]):
                save_volume(np.ascontiguousarray(image[..., c]), directory / label / fov / f"{channel}.tif",
                            metadata=metadata)
        held = 0
        for stain in stains:
            if stain["round"] == "reference":
                image, folder = scene.stains[fov][stain["name"]], reference
            else:
                image, folder = scene.morphology[fov][..., held], morphology["name"]
                held += 1
            save_volume(np.ascontiguousarray(image), directory / folder / fov / f"{stain['channel']}.tif",
                        metadata=metadata)
    columns = [c for c in ("entry_id", "gene_id", "color_sequence") if c in pd.DataFrame(codebook["rows"]).columns]
    pd.DataFrame(codebook["rows"])[columns].to_csv(directory / "codebook.csv", index=False)
    spacing = record["sample_grid"]["spacing_zyx"]
    channel = lambda s: dict(channel=s["channel"], name=s["name"])  # noqa: E731
    files = {path.relative_to(directory).as_posix(): file_sha256(path)
             for path in sorted(directory.rglob("*")) if path.is_file()}
    dataset = dict(
        format="starfinder.synthetic.save_context_dataset", context_truth_version=CELLULAR_TRUTH_VERSION,
        dataset_id=sample.record["requested_config"]["fov"]["dataset_version"], sample_id=sample.sample,
        rounds=dict(sequencing=list(codebook["round_labels"]), reference=reference,
                    other=[] if morphology is None else [morphology["name"]]),
        channel_order=list(codebook["channel_labels"]),
        other_channel_order={} if morphology is None else {
            morphology["name"]: [channel(s) for s in stains if s["round"] == "morphology"]},
        reference_stains=[channel(s) for s in stains if s["round"] == "reference"],
        voxel_size_xy=spacing[1] if spacing[1] == spacing[2] else None, voxel_size_z=spacing[0],
        spacing_zyx=list(spacing), codebook=dict(path="codebook.csv", round_labels=list(codebook["round_labels"]),
                                                 channel_labels=list(codebook["channel_labels"]),
                                                 color_to_channel=codebook["color_to_channel"],
                                                 encoding=codebook["encoding"]),
        fov_ids=fov_ids, files=files)
    (directory / "dataset.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return directory
