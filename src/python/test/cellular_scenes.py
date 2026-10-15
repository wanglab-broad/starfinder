"""Fixtures of the cellular scene's generator checks (docs/synthetic-cellular-scene-validation.md, "Fixtures").

Each fixture is built in the session from the values of the validation page on ``sample-tiny-v1`` (seed 42) and
cached by name, so the test files that share one generate it once. Nothing is read from disk.
"""
from dataclasses import replace
from functools import cache

from starfinder.synthetic import (CellTypeConfig, ContextSceneConfig, MoleculeConfig, NucleusShapeConfig,
                                  PackingConfig, ScalarDistribution, TerritoryConfig, generate_context_scene,
                                  sample_scene_preset)

from .cellular_oracle import harmonic_bound

RULES = ("reach", "grown", "power")
# ctx_shapes: every nucleus lobed with the preset lobe laws (the defaults of NucleusShapeConfig); touching nuclei,
# binucleate cells with unequal radii and empty cells.
SHAPES = NucleusShapeConfig(lobed_share=1.0)
PACKING = PackingConfig(contact=1.0, overlap=0.2, gap=0.0, kind_shares=(0.6, 0.2, 0.2), on_crowded="stop")
# ctx_types: G1 with the two marker pairs of C8, M2 and NF1, A2 ambient molecules.
TYPES = CellTypeConfig()
MOLECULES = MoleculeConfig(per_cell_median=200, per_cell_spread=0.3, nuclear_fraction=0.5)


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
    raise KeyError(name)


@cache
def scene(name):
    """(codebook, config, ContextScene) of a fixture name, generated once per session."""
    book, _ = tiny()
    config = config_of(name)
    return book, config, generate_context_scene(book, config=config)
