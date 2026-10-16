"""Fixtures of the cellular scene's generator checks (docs/synthetic-cellular-scene-validation.md, "Fixtures").

Each fixture is built in the session from the values of the validation page on ``sample-tiny-v1`` (seed 42) and
cached by name, so the test files that share one generate it once. Nothing is read from disk.
"""
from dataclasses import replace
from functools import cache
import math

from starfinder.synthetic import (BrightRegionConfig, CellTypeConfig, ContextSceneConfig, MoleculeConfig,
                                  MorphologyRoundConfig, NoiseConfig, NucleusShapeConfig, OutputScaleConfig,
                                  PackingConfig, ScalarDistribution, StainChannelConfig, TerritoryConfig,
                                  UnevennessConfig, generate_context_scene, sample_scene_preset)

from .cellular_oracle import harmonic_bound

RULES = ("reach", "grown", "power")
# ctx_shapes: every nucleus lobed with the preset lobe laws (the defaults of NucleusShapeConfig); touching nuclei,
# binucleate cells with unequal radii and empty cells.
SHAPES = NucleusShapeConfig(lobed_share=1.0)
PACKING = PackingConfig(contact=1.0, overlap=0.2, gap=0.0, kind_shares=(0.6, 0.2, 0.2), on_crowded="stop")
# ctx_types: G1 with the two marker pairs of C8, M2 and NF1, A2 ambient molecules.
TYPES = CellTypeConfig()
MOLECULES = MoleculeConfig(per_cell_median=200, per_cell_spread=0.3, nuclear_fraction=0.5)


# ctx_stains (W-484): the culture preset's bright-region and field laws, the presets' largest slopes; a field length of
# one FOV width (the default); the whole-voxel morphology translation (0, 2, −3) voxels for every FOV.
TEXTURE = BrightRegionConfig(per_nucleus=2, size=ScalarDistribution("lognormal", (math.log(0.17), 0.64)),
                             height=ScalarDistribution("lognormal", (math.log(2.0), 0.69)))
FIELD = UnevennessConfig(centres_per_area=4, amplitude=0.74)
WHOLE = (0.0, 2.0, -3.0)
FOVS = ("FOV_001", "FOV_002", "FOV_003", "FOV_004")
# The worked example of identity 2 of docs/synthetic-sample-scene.md moved by T = (0.5, 6, 6): FOV_001 at δ = (1, 8, 6)
# and FOV_002 at δ = (1, 31, 8) on the sample grid, so the two differ by whole voxels and both are aligned crops.
PAIR = {"FOV_001": (0.1, 0.5, -1.5), "FOV_002": (0.1, -0.5, 0.5), "FOV_003": (0.2, 1.0, -1.0),
        "FOV_004": (-0.3, 0.0, 0.0)}


def stains(noise=NoiseConfig(), output=None, unevenness=FIELD, mr1=False):
    """ctx_stains' channels: DAPI (nucleus, bright regions, the 2D tissue presets' edge of 0.25 µm, so that its
    plateau of argument ≤ −4w holds voxels), Flamingo (territory, nuclear ratio 0.3) and membrane at
    band widths 0.5 µm (``membrane``) and 0.01 µm (``membrane-thin``) in the reference round, and DAPI and Flamingo in
    the morphology round ``organelle``; ``mr1`` keeps the culture presets' layout (DAPI the one reference stain)."""
    common = dict(unevenness=unevenness, noise=noise, output=output)
    dapi = StainChannelConfig("DAPI", edge_width=0.25, texture=TEXTURE, **common)
    flamingo = StainChannelConfig("Flamingo", source="territory", channel="ch05", nuclear_ratio=0.3, **common)
    reference = (dapi,) if mr1 else (
        dapi, flamingo, StainChannelConfig("membrane", source="boundary", channel="ch06", band_width=0.5, **common),
        StainChannelConfig("membrane-thin", source="boundary", channel="ch07", band_width=0.01, **common))
    return reference + (replace(dapi, round="morphology", channel="ch00"),
                        replace(flamingo, round="morphology", channel="ch02"))


def morphology(mode, flat=False):
    bound = (0.0, 6.0, 6.0) if flat else (0.5, 6.0, 6.0)
    if mode == "drawn":
        return MorphologyRoundConfig(translation_max_zyx=bound)
    return MorphologyRoundConfig(translation="supplied", translation_max_zyx=bound,
                                 translation_zyx={fov: WHOLE for fov in FOVS})


def float64(sample):
    """The template dtype float64 and no sample-scene stain (the context's stains take its place)."""
    return replace(sample, stain=None, fov=replace(sample.fov, dtype="float64"))


def tiny():
    return sample_scene_preset("sample-tiny-v1")


def harmonic(planar):
    """N2 on: largest degree 3, the amplitude so that harmonic_amplitude · M_Y is 0.5."""
    return replace(SHAPES, harmonics=3, harmonic_amplitude=0.5 / harmonic_bound(3, planar))


def plane(config):
    """The single-plane configuration of docs/stitching-assembly-validation.md (``scene_plane``): FOVs of
    1 × 36 × 36 and a jitter bound of (0, 1.5, 1.5) voxels."""
    return replace(config, acquisition=replace(config.acquisition, fov_shape_zyx=(1, 36, 36),
                                               jitter_max_zyx=(0.0, 1.5, 1.5)),
                   fov=replace(config.fov, shape_zyx=(1, 36, 36)))


def config_of(name):
    """The ContextSceneConfig of a fixture name."""
    book, config = tiny()
    shapes = {f"shapes-{rule}": dict(nuclei=SHAPES, packing=PACKING, territories=TerritoryConfig(rule))
              for rule in RULES}
    shapes.update({f"harmonic-{rule}": dict(nuclei=harmonic(False), packing=PACKING, territories=TerritoryConfig(rule))
                   for rule in RULES})
    if name in shapes:
        return ContextSceneConfig(sample=config, **shapes[name])
    if name in ("monolayer", "monolayer-harmonic"):
        return ContextSceneConfig(sample=config, nuclei=harmonic(True) if name.endswith("harmonic") else SHAPES,
                                  packing=replace(PACKING, monolayer=True), territories=TerritoryConfig("grown"))
    if name == "plane":
        # ctx_plane: ctx_shapes on the single-plane configuration, N2 in its Fourier form, T2.
        return ContextSceneConfig(sample=plane(config), nuclei=harmonic(True), packing=PACKING,
                                  territories=TerritoryConfig("grown"))
    if name == "types":
        return ContextSceneConfig(sample=config, types=TYPES, molecules=MOLECULES)
    if name == "types-abundance":
        rows = list(TYPES.abundances)
        # Type 2's row: its markers lighter, the shared genes heavier.
        rows[1] = tuple(0.05 if w == 0.15 else 0.1125 if w == 0.0875 else w for w in rows[1])
        return ContextSceneConfig(sample=config, types=replace(TYPES, abundances=tuple(rows)), molecules=MOLECULES)
    if name == "types-support":
        # The review's case: type 2's weight of balanced-09 moves to balanced-10, a change of the row's support.
        rows = list(TYPES.abundances)
        row = list(rows[1])
        row[9], row[8] = row[9] + row[8], 0.0
        rows[1] = tuple(row)
        return ContextSceneConfig(sample=config, types=replace(TYPES, abundances=tuple(rows)), molecules=MOLECULES)
    if name == "types-count":
        return ContextSceneConfig(sample=config, types=replace(TYPES, molecules_per_cell=(200.0, 200.0, 50.0, 200.0)),
                                  molecules=MOLECULES)
    if name == "spread-zero":
        molecules = MoleculeConfig(per_cell_median=20, per_cell_spread=0.0, nuclear_fraction=0.5)
        return ContextSceneConfig(sample=config, types=TYPES, molecules=molecules)
    if name.startswith("crowded-"):
        # ctx_crowded: density 0.6, more than the domain holds, with the floors 0.9 and 0.1; "crowded-below" is the
        # variant whose target is beyond reach by more than its floor (see the worker notes, finding 1).
        density, floor = (0.9, 0.9) if name == "crowded-below" else (0.6, float(name[len("crowded-"):]))
        return ContextSceneConfig(sample=config, packing=PackingConfig(density=density, on_crowded="stop",
                                                                       floor=floor))
    if name == "density":
        # Density mode that reaches its target, lobed nuclei and contact proposals.
        return ContextSceneConfig(sample=config, nuclei=replace(SHAPES, lobed_share=0.5),
                                  packing=PackingConfig(density=0.15, contact=0.5, overlap=0.1, on_crowded="stop"))
    if name.startswith("territory-"):
        # The review's case: one spherical nucleus of radius 0.5 µm, reach 0.1 µm, a mean of 100 molecules, no ambient
        # molecule; "territory-<rule>[-molecules]" with T3 or T2 (δ = 2 µm), and optionally MoleculeConfig(100).
        rule = name.split("-")[1]
        cells = replace(config.cells, nuclei=1, binucleate=0, empty=0,
                        nucleus_axial_radius=ScalarDistribution("constant", (0.5,)),
                        nucleus_lateral_radius=ScalarDistribution("constant", (0.5,)),
                        nucleus_elongation=ScalarDistribution("constant", (1.0,)),
                        reach=ScalarDistribution("constant", (0.1,)))
        sample = replace(config, cells=cells, molecules_per_cell=100.0, ambient_density=0.0)
        territory = (TerritoryConfig("power") if rule == "power"
                     else TerritoryConfig("grown", growth=ScalarDistribution("constant", (2.0,))))
        molecules = MoleculeConfig(per_cell_median=100) if name.endswith("-molecules") else None
        return ContextSceneConfig(sample=sample, territories=territory, molecules=molecules)
    if name.startswith(("stains-", "uint8", "plane-")) or name == "mr1":
        return stain_config(name)
    raise KeyError(name)


# ctx_uint8: the stain noise of StainChannelConfig's default (Poisson with α 1 and read noise σ 3).
NOISE = NoiseConfig(dependent_enabled=True, model="poisson", alpha=1.0, independent_enabled=True, sigma=3.0)


def stain_config(name):
    """The fixtures of the stains and the morphology round (docs/synthetic-cellular-scene-validation.md, "Fixtures"):

    * ``stains-<rule>-<whole|drawn>``: ctx_stains, ctx_shapes of the rule with noise off, the template dtype float64,
      no output scale, and the morphology translation (0, 2, −3) voxels or MT1's drawn one;
    * ``stains-pair`` (supplied jitter differing by whole voxels between FOV_001 and FOV_002), ``stains-empty`` (no
      cell), ``stains-flat-field`` (no centre of the field), ``stains-extra`` (one more stain, first) and
      ``stains-off`` (no morphology round): copies of ``stains-reach-whole``;
    * ``uint8``: ctx_uint8, noise on and OutputScaleConfig(uint8, 50) on every stain but ``membrane-thin`` (kept in
      the float32 template dtype); ``uint8-float``: its float64 rendering with the same streams; ``uint8-empty``: no
      cell, and ``uint8-empty-float`` its float64 rendering;
    * ``plane-<whole|drawn>``, ``plane-uint8``, ``plane-uint8-float``: ctx_plane, the same at Z=1 (T_z = 0);
    * ``mr1``: the culture presets' layout (DAPI the one reference stain) with MT1's drawn translation.
    """
    book, config = tiny()
    shapes = dict(nuclei=SHAPES, packing=PACKING)
    if name.startswith("plane-"):
        mode = "drawn" if name == "plane-drawn" else "whole"
        sample = plane(config)
        parts = dict(nuclei=harmonic(True), packing=PACKING, territories=TerritoryConfig("grown"),
                     morphology=morphology(mode, flat=True))
        if name == "plane-uint8":
            return ContextSceneConfig(sample=replace(sample, stain=None), stains=uint8(), **parts)
        if name == "plane-uint8-float":
            return ContextSceneConfig(sample=float64(sample), stains=stains(noise=NOISE), **parts)
        return ContextSceneConfig(sample=float64(sample), stains=stains(), **parts)
    if name.startswith("stains-") and name.split("-")[1] in RULES:
        _, rule, mode = name.split("-")
        return ContextSceneConfig(sample=float64(config), territories=TerritoryConfig(rule), stains=stains(),
                                  morphology=morphology(mode), **shapes)
    base = ContextSceneConfig(sample=float64(config), territories=TerritoryConfig("reach"), stains=stains(),
                              morphology=morphology("whole"), **shapes)
    empty = replace(config.cells, nuclei=0, binucleate=0, empty=0)
    if name == "stains-pair":
        acquisition = replace(config.acquisition, jitter="supplied", jitter_zyx=PAIR)
        return replace(base, sample=replace(base.sample, acquisition=acquisition))
    if name == "stains-empty":
        return replace(base, sample=replace(base.sample, cells=empty))
    if name == "stains-flat-field":
        return replace(base, stains=stains(unevenness=replace(FIELD, centres_per_area=0)))
    if name == "stains-extra":
        extra = StainChannelConfig("PI", channel="ch08", texture=TEXTURE, unevenness=FIELD, noise=NoiseConfig())
        return replace(base, stains=(extra,) + base.stains)
    if name == "stains-off":
        return replace(base, stains=tuple(s for s in base.stains if s.round == "reference"), morphology=None)
    if name == "uint8":
        return replace(base, sample=replace(config, stain=None), stains=uint8())
    if name == "uint8-float":
        return replace(base, stains=stains(noise=NOISE))
    if name == "uint8-empty":
        return replace(base, sample=replace(config, stain=None, cells=empty), stains=uint8())
    if name == "uint8-empty-float":
        return replace(base, sample=replace(base.sample, cells=empty), stains=stains(noise=NOISE))
    if name == "mr1":
        return replace(base, stains=stains(mr1=True), morphology=morphology("drawn"))
    raise KeyError(name)


def uint8():
    """ctx_uint8's channels: noise on, 8-bit output at plateau 50, ``membrane-thin`` in the template dtype."""
    return tuple(replace(s, output=None) if s.name == "membrane-thin" else s
                 for s in stains(noise=NOISE, output=OutputScaleConfig("uint8", 50.0)))


@cache
def scene(name):
    """(codebook, config, ContextScene) of a fixture name, generated once per session."""
    book, _ = tiny()
    config = config_of(name)
    return book, config, generate_context_scene(book, config=config)
