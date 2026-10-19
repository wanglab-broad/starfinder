"""The six context presets of the cellular scene (docs/synthetic-cellular-scene.md, "Presets", option PS1).

One routine and one ``slow`` preset per context of D2: dense 3D tissue, 2D tissue (projection samples) and the
sparse culture. Each value is the page's table of its context; the routine and ``slow`` presets of a context differ
only where that table says so. The expected numbers of nuclei and molecules are outcomes of audit A (W-394, re-run by
W-481 for the two ``slow`` presets whose pixel is twice the acquired one), recorded beside the values, not parameters.
"""
from __future__ import annotations

from starfinder.barcode import Codebook

from ._common import ScalarDistribution
from ._context import (CONTEXT_PRESET_VERSION, CellTypeConfig, ContextSceneConfig, MoleculeConfig, NucleusShapeConfig,
                       PackingConfig, TerritoryConfig)
from ._formed import FormedSceneConfig
from ._geometry import GeometryConfig
from ._observation import BackgroundConfig, NoiseConfig
from ._sample import AcquisitionConfig, CellularSceneConfig, SampleSceneConfig, _layout, _lognormal
from ._stains import BrightRegionConfig, MorphologyRoundConfig, OutputScaleConfig, StainChannelConfig, UnevennessConfig

# The background region's σ in Y and X: 128 LN pixels of 0.194398 µm (24.9 µm), given in sample-grid voxels.
_REGION_SIGMA_UM = 128 * 0.194398
# Parameters common to the six presets ("Presets", the first table).
_COMMON = dict(kind_shares=(0.9, 0.05, 0.05), empty_radius=2.5, max_proposals=10_000, on_crowded="stop", floor=0.9,
               overlap=0.1, gap=0.0, ambient_density=0.002, plateau_median=1000.0, alpha=1.0,
               centres_per_area=4.0, translation_max_zyx=(1.0, 2.0, 2.0))
# Values by context, shared by its routine and slow presets.
_CONTEXTS = {
    "dense3d": dict(
        spacing_zyx=(0.346255, 0.194398, 0.194398), axial=(2.1, 0.14), lateral=(2.7, 0.10), elongation=(0.089, 0.151),
        lobed_share=0.07, contact=0.3, monolayer=False, territory="grown", growth=(0.78, 0.15), reach=(2.5, 0.15),
        nuclear_fraction=0.5, stain="DAPI", plateau_spread=0.23, unevenness=None, edge_width=0.93,
        texture=(3.0, (0.41, 0.37), (1.35, 0.27)), baseline=190.0, region_height=600.0, sigma=16.0, grey=42.0,
        morphology=None),
    "tissue2d": dict(
        spacing_zyx=(0.346255, 0.37854, 0.37854), axial=(2.1, 0.14), lateral=(3.6, 0.34), elongation=(0.052, 0.301),
        lobed_share=0.04, contact=0.65, monolayer=True, territory="grown", growth=(0.5, 0.15), reach=(2.5, 0.15),
        nuclear_fraction=0.75, stain="PI", plateau_spread=0.20, unevenness=0.22, edge_width=0.25,
        texture=(2.0, (0.21, 0.36), (1.0, 0.23)), baseline=25.0, region_height=None, sigma=10.0, grey=78.0,
        morphology=None),
    "culture": dict(
        spacing_zyx=(0.346255, 0.777592, 0.777592), axial=(2.1, 0.14), lateral=(7.5, 0.30), elongation=(0.117, 0.408),
        lobed_share=0.24, contact=0.4, monolayer=True, territory="reach", growth=None, reach=(5.3, 0.3),
        nuclear_fraction=0.26, stain="DAPI", plateau_spread=0.53, unevenness=0.74, edge_width=0.38,
        texture=(2.0, (0.17, 0.64), (2.0, 0.69)), baseline=12.0, region_height=350.0, sigma=25.0, grey=20.0,
        morphology=dict(name="organelle", translation="uniform", translation_max_zyx=(0.5, 6.0, 6.0),
                        cytoplasm=dict(name="Flamingo", channel="ch02", brightness=(1000.0, 0.3), nuclear_ratio=0.3,
                                       edge_width=0.5),
                        nuclear_channel="ch00")),
}
# Values by preset: what differs between the routine and slow presets of a context, and the outcomes of audit A.
_PRESET_SPECS = {
    "context-dense3d-v1": dict(
        context="dense3d", tier="routine", columns=2, rows=2, fov_shape_zyx=(16, 64, 64), overlap=0.25,
        jitter_max_zyx=(0.8, 3.0, 3.0), density=0.16, per_cell_median=8.0, per_cell_spread=0.45,
        expected_nuclei=7, expected_molecules=70),
    "context-dense3d-slow-v1": dict(
        context="dense3d", tier="slow", columns=3, rows=3, fov_shape_zyx=(16, 256, 256), overlap=0.25,
        jitter_max_zyx=(0.8, 4.5, 4.5), density=0.13, per_cell_median=8.0, per_cell_spread=0.45,
        expected_nuclei=150, expected_molecules=1500),
    "context-tissue2d-v1": dict(
        context="tissue2d", tier="routine", columns=2, rows=2, fov_shape_zyx=(8, 64, 64), overlap=0.25,
        jitter_max_zyx=(0.8, 3.0, 3.0), density=0.36, per_cell_median=85.0, per_cell_spread=0.30,
        expected_nuclei=11, expected_molecules=1000),
    "context-tissue2d-slow-v1": dict(
        context="tissue2d", tier="slow", columns=3, rows=3, fov_shape_zyx=(16, 256, 256), overlap=0.25,
        jitter_max_zyx=(0.8, 4.5, 4.5), density=0.36, per_cell_median=194.0, per_cell_spread=0.30,
        spacing_zyx=(0.346255, 0.18927, 0.18927), expected_nuclei=76, expected_molecules=15_800),
    "context-culture-v1": dict(
        context="culture", tier="routine", columns=2, rows=2, fov_shape_zyx=(16, 64, 64), overlap=0.25,
        jitter_max_zyx=(0.8, 3.0, 3.0), density=0.26, per_cell_median=120.0, per_cell_spread=0.40,
        expected_nuclei=8, expected_molecules=1200),
    "context-culture-slow-v1": dict(
        context="culture", tier="slow", columns=3, rows=3, fov_shape_zyx=(16, 256, 256), overlap=0.25,
        jitter_max_zyx=(0.8, 4.5, 4.5), density=0.26, per_cell_median=982.0, per_cell_spread=0.40,
        spacing_zyx=(0.346255, 0.388796, 0.388796), expected_nuclei=42, expected_molecules=44_500),
}


def _spec(name):
    spec = dict(_CONTEXTS[_PRESET_SPECS[name]["context"]])
    spec.update(_PRESET_SPECS[name])
    return spec


#: Context presets (context-presets-v1), separate from SAMPLE_PRESETS and SCENE_PRESETS.
CONTEXT_PRESETS = {name: dict(version=CONTEXT_PRESET_VERSION, seed=42, split="development", scene_key=name,
                              codebook="development_codebook('balanced')",
                              **{k: v for k, v in _spec(name).items() if k not in ("morphology", "texture")},
                              morphology=None if _spec(name)["morphology"] is None
                              else _spec(name)["morphology"]["name"])
                   for name in _PRESET_SPECS}


def _stains(spec, background):
    """The stain channels of a preset: the nuclear stain of the reference round, and in the culture the morphology
    round's nuclear and cytoplasm stains."""
    count, size, height = spec["texture"]
    noise = NoiseConfig(dependent_enabled=True, model="poisson", alpha=_COMMON["alpha"], independent_enabled=True,
                        sigma=spec["sigma"])
    unevenness = None if spec["unevenness"] is None else UnevennessConfig(
        centres_per_area=_COMMON["centres_per_area"], amplitude=spec["unevenness"], length=None)
    output = OutputScaleConfig(dtype="uint8", plateau=spec["grey"])
    nuclear = StainChannelConfig(
        spec["stain"], source="nucleus", round="reference", channel="ch04",
        brightness=_lognormal(_COMMON["plateau_median"], spec["plateau_spread"]), edge_width=spec["edge_width"],
        texture=BrightRegionConfig(per_nucleus=count, size=_lognormal(*size), height=_lognormal(*height)),
        unevenness=unevenness, background=background, noise=noise, output=output)
    morphology = spec["morphology"]
    if morphology is None:
        return (nuclear,), None
    cytoplasm = morphology["cytoplasm"]
    stains = (nuclear,
              StainChannelConfig(nuclear.name, source="nucleus", round="morphology",
                                 channel=morphology["nuclear_channel"], brightness=nuclear.brightness,
                                 edge_width=nuclear.edge_width, texture=nuclear.texture, unevenness=unevenness,
                                 background=background, noise=noise, output=output),
              StainChannelConfig(cytoplasm["name"], source="territory", round="morphology",
                                 channel=cytoplasm["channel"], brightness=_lognormal(*cytoplasm["brightness"]),
                                 edge_width=cytoplasm["edge_width"], nuclear_ratio=cytoplasm["nuclear_ratio"],
                                 unevenness=unevenness, background=background, noise=noise, output=output))
    return stains, MorphologyRoundConfig(name=morphology["name"], translation=morphology["translation"],
                                         translation_max_zyx=morphology["translation_max_zyx"])


def context_scene_preset(name: str) -> tuple[Codebook, ContextSceneConfig]:
    """Return (codebook, config) for a CONTEXT_PRESETS name: a ``ContextSceneConfig``.

    The six presets of docs/synthetic-cellular-scene.md ("Presets", option PS1):
    ``context-dense3d-v1`` and ``context-dense3d-slow-v1`` (dense 3D tissue, the LN grid),
    ``context-tissue2d-v1`` and ``context-tissue2d-slow-v1`` (2D tissue, a monolayer for
    projection samples; the ``slow`` preset at 0.18927 µm in Y and X) and
    ``context-culture-v1`` and ``context-culture-slow-v1`` (the sparse culture, a monolayer
    with the morphology round ``organelle``; the ``slow`` preset at 0.388796 µm). Every
    preset uses ``development_codebook("balanced")``, seed 42, split ``development``, the
    scene key of its name, the amplicon appearance of ``sample-small-v1`` with float32
    output, the cell types of G1 and an 8-bit nuclear stain ``ch04`` of the reference round;
    no preset uses the ``boundary`` source. Unknown names raise ValueError.
    """
    from ._calibrated import development_codebook
    from ._presets import _appearance
    if name not in _PRESET_SPECS:
        raise ValueError(f"unknown context preset: {name}; choose from {list(_PRESET_SPECS)}")
    spec = _spec(name)
    codebook = development_codebook("balanced")
    acquisition = AcquisitionConfig(columns=spec["columns"], rows=spec["rows"], fov_shape_zyx=spec["fov_shape_zyx"],
                                    spacing_zyx=spec["spacing_zyx"], overlap=spec["overlap"],
                                    jitter_max_zyx=spec["jitter_max_zyx"])
    morphology = spec["morphology"]
    grow = None if morphology is None else morphology["translation_max_zyx"]
    layout = _layout(acquisition, grow)
    shape = tuple(int(v) for v in layout.shape)
    geometry = GeometryConfig(translation_enabled=True, translation_max_zyx=_COMMON["translation_max_zyx"],
                              reference_round=codebook.round_labels[0])
    template = FormedSceneConfig(dataset_version=name, sample_id="sample", scene_key=name, seed=42,
                                 split="development", shape_zyx=spec["fov_shape_zyx"], dtype="float32",
                                 accumulation="float32", geometry=geometry,
                                 **{k: v for k, v in _appearance(shape, len(codebook.round_labels), "float32").items()
                                    if k != "dtype"})
    cells = CellularSceneConfig(nucleus_axial_radius=_lognormal(*spec["axial"]),
                                nucleus_lateral_radius=_lognormal(*spec["lateral"]),
                                nucleus_elongation=ScalarDistribution("folded_lognormal", spec["elongation"]),
                                empty_radius=_COMMON["empty_radius"], reach=_lognormal(*spec["reach"]),
                                max_proposals=_COMMON["max_proposals"])
    if spec["region_height"] is None:
        background = BackgroundConfig(baseline_enabled=True, baseline=((spec["baseline"],),))
    else:
        # One region at the sample grid's centre, σ n_z planes in Z and 24.9 µm in Y and X, in sample-grid voxels.
        s = acquisition.spacing_zyx
        sigma = (float(spec["fov_shape_zyx"][0]), _REGION_SIGMA_UM / s[1], _REGION_SIGMA_UM / s[2])
        background = BackgroundConfig(baseline_enabled=True, baseline=((spec["baseline"],),), regions_enabled=True,
                                      region_centers=(tuple((n - 1) / 2 for n in shape),), region_sigma_zyx=(sigma,),
                                      region_heights=(spec["region_height"],), tissue_weights=((1.0,),))
    stains, morphology_config = _stains(spec, background)
    sample = SampleSceneConfig(fov=template, acquisition=acquisition, cells=cells, stain=None,
                               molecules_per_cell=spec["per_cell_median"],
                               ambient_density=_COMMON["ambient_density"])
    growth = (TerritoryConfig(rule=spec["territory"]) if spec["growth"] is None
              else TerritoryConfig(rule=spec["territory"], growth=_lognormal(*spec["growth"])))
    config = ContextSceneConfig(
        sample=sample, nuclei=NucleusShapeConfig(lobed_share=spec["lobed_share"]),
        packing=PackingConfig(density=spec["density"], kind_shares=_COMMON["kind_shares"], contact=spec["contact"],
                              overlap=_COMMON["overlap"], gap=_COMMON["gap"], monolayer=spec["monolayer"],
                              on_crowded=_COMMON["on_crowded"], floor=_COMMON["floor"]),
        territories=growth, types=CellTypeConfig(),
        molecules=MoleculeConfig(per_cell_median=spec["per_cell_median"], per_cell_spread=spec["per_cell_spread"],
                                 nuclear_fraction=spec["nuclear_fraction"],
                                 ambient_density=_COMMON["ambient_density"]),
        stains=stains, morphology=morphology_config)
    return codebook, config
