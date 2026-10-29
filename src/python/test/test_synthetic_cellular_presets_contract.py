"""The six context presets against the page's tables (design/validation/synthetic-cellular-scene.md, GN2, preset part).

Every value of ``cellular_scene_preset`` is compared with the PS1 tables of design/specs/synthetic/cellular-scene.md
("Presets": the common parameters and the table of each context, with the ``slow`` pixels of the decision of
2026-10-15), written here from the page, not from the module; no preset uses the ``boundary`` source (O2).
Floats are compared within the project's absolute 1e-9 (the page states no other bound for a configuration value).
"""
from dataclasses import fields
import math

import numpy as np
import pytest

import starfinder.synthetic as synthetic
from starfinder.synthetic import (CELLULAR_PRESET_VERSION, CELLULAR_PRESETS, SAMPLE_PRESETS, BackgroundConfig,
                                  CellTypeConfig, CellularSceneConfig, NoiseConfig, OutputScaleConfig,
                                  ScalarDistribution, cellular_scene_preset, development_codebook, sample_scene_preset)
from starfinder.synthetic._presets import _appearance
from starfinder.synthetic._sample import _layout

pytestmark = [pytest.mark.synthetic, pytest.mark.contract]

NAMES = ["context-dense3d-v1", "context-dense3d-slow-v1", "context-tissue2d-v1", "context-tissue2d-slow-v1",
         "context-culture-v1", "context-culture-slow-v1"]
LN = (0.346255, 0.194398, 0.194398)
# The page's tables: per context the values its routine and slow presets share, per preset what differs.
CONTEXT = {
    "dense3d": dict(axial=(2.1, 0.14), lateral=(2.7, 0.10), elongation=(0.089, 0.151), lobed=0.07, contact=0.3,
                    monolayer=False, rule="grown", growth=(0.78, 0.15), nuclear_fraction=0.5, stain="DAPI",
                    spread=0.23, field=None, edge=0.93, texture=(3, 0.41, 0.37, 1.35, 0.27), baseline=190,
                    region=600, sigma=16, grey=42),
    "tissue2d": dict(axial=(2.1, 0.14), lateral=(3.6, 0.34), elongation=(0.052, 0.301), lobed=0.04, contact=0.65,
                     monolayer=True, rule="grown", growth=(0.5, 0.15), nuclear_fraction=0.75, stain="PI",
                     spread=0.20, field=0.22, edge=0.25, texture=(2, 0.21, 0.36, 1.0, 0.23), baseline=25,
                     region=None, sigma=10, grey=78),
    "culture": dict(axial=(2.1, 0.14), lateral=(7.5, 0.30), elongation=(0.117, 0.408), lobed=0.24, contact=0.4,
                    monolayer=True, rule="reach", reach=(5.3, 0.3), nuclear_fraction=0.26, stain="DAPI",
                    spread=0.53, field=0.74, edge=0.38, texture=(2, 0.17, 0.64, 2.0, 0.69), baseline=12,
                    region=350, sigma=25, grey=20),
}
PRESET = {
    "context-dense3d-v1": dict(context="dense3d", grid=(2, 2), fov=(16, 64, 64), spacing=LN, jitter=(0.8, 3.0, 3.0),
                               density=0.16, per_cell=(8, 0.45)),
    "context-dense3d-slow-v1": dict(context="dense3d", grid=(3, 3), fov=(16, 256, 256), spacing=LN,
                                    jitter=(0.8, 4.5, 4.5), density=0.13, per_cell=(8, 0.45)),
    "context-tissue2d-v1": dict(context="tissue2d", grid=(2, 2), fov=(8, 64, 64), spacing=(0.346255, 0.37854, 0.37854),
                                jitter=(0.8, 3.0, 3.0), density=0.36, per_cell=(85, 0.30)),
    "context-tissue2d-slow-v1": dict(context="tissue2d", grid=(3, 3), fov=(16, 256, 256),
                                     spacing=(0.346255, 0.18927, 0.18927), jitter=(0.8, 4.5, 4.5), density=0.36,
                                     per_cell=(194, 0.30)),
    "context-culture-v1": dict(context="culture", grid=(2, 2), fov=(16, 64, 64),
                               spacing=(0.346255, 0.777592, 0.777592), jitter=(0.8, 3.0, 3.0), density=0.26,
                               per_cell=(120, 0.40)),
    "context-culture-slow-v1": dict(context="culture", grid=(3, 3), fov=(16, 256, 256),
                                    spacing=(0.346255, 0.388796, 0.388796), jitter=(0.8, 4.5, 4.5), density=0.26,
                                    per_cell=(982, 0.40)),
}


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=0, atol=tol)


def lognormal(dist, median, sd):
    return dist.mode == "lognormal" and close(dist.parameters, (math.log(median), sd))


def test_registry_names_and_version():
    assert CELLULAR_PRESET_VERSION == "context-presets-v1"
    assert list(CELLULAR_PRESETS) == NAMES
    assert {"CELLULAR_PRESETS", "CELLULAR_PRESET_VERSION", "cellular_scene_preset"} <= set(synthetic.__all__)
    for name, entry in CELLULAR_PRESETS.items():
        assert entry["version"] == "context-presets-v1" and entry["seed"] == 42 and entry["scene_key"] == name
        assert entry["tier"] == ("slow" if "-slow-" in name else "routine")
    # SAMPLE_PRESETS keeps its two keys, and the sample presets are unchanged by the new registry.
    assert list(SAMPLE_PRESETS) == ["sample-tiny-v1", "sample-small-v1"]
    with pytest.raises(ValueError, match="unknown context preset"):
        cellular_scene_preset("context-dense3d-v2")


@pytest.mark.parametrize("name", NAMES)
def test_gn2_every_value_equals_the_page_table(name):
    p = PRESET[name]
    c = CONTEXT[p["context"]]
    book, config = cellular_scene_preset(name)
    assert isinstance(config, CellularSceneConfig)
    # The common parameters.
    reference = development_codebook("balanced")
    assert book.genes == reference.genes and book.round_labels == reference.round_labels and book.n_genes == 16
    assert len(book.round_labels) == 4 and len(book.channel_labels) == 4
    sample, template, acquisition = config.sample, config.sample.fov, config.sample.acquisition
    assert (template.seed, template.split, template.scene_key, template.dataset_version) == (
        42, "development", name, name)
    geometry = template.geometry
    assert geometry.translation_enabled and tuple(geometry.translation_max_zyx) == (1, 2, 2)
    assert geometry.reference_round == "round1"
    assert not (geometry.local_enabled or geometry.affine_enabled or geometry.polynomial_enabled)
    assert template.dtype == "float32"
    # The amplicon appearance of sample-small-v1: _appearance of benchmark-presets-v1 on this sample grid.
    morphology_bound = (0.5, 6.0, 6.0) if p["context"] == "culture" else None
    shape = tuple(int(v) for v in _layout(acquisition, morphology_bound).shape)
    appearance = _appearance(shape, 4, "float32")
    small = sample_scene_preset("sample-small-v1")[1].fov
    assert template.accumulation == small.accumulation == "float32"
    for key in ("brightness", "axial_width", "lateral_width", "elongation", "angle"):
        assert getattr(template, key) == appearance[key] == getattr(small, key), key
    for key in ("readout", "background", "noise"):
        assert repr(getattr(template, key)) == repr(appearance[key]), key
    readout = template.readout
    assert not (readout.loss_enabled or readout.dropout_enabled or readout.weakening_enabled)    # effects off
    cells = sample.cells
    assert config.packing.kind_shares == (0.9, 0.05, 0.05) and cells.empty_radius == 2.5
    assert cells.max_proposals == 10_000 and config.packing.on_crowded == "stop" and config.packing.floor == 0.9
    nuclei = config.nuclei
    assert nuclei.extra_lobes == (1, 2) and nuclei.lobe_scale == ScalarDistribution("uniform", (0.5, 0.8))
    assert nuclei.lobe_offset == ScalarDistribution("uniform", (0.5, 0.9)) and nuclei.harmonics == 0
    assert config.packing.overlap == 0.1 and config.packing.gap == 0.0
    assert repr(config.types) == repr(CellTypeConfig())    # G1, proportions 0.25, 4 regions, purity 0.9, C8's pairs
    molecules = config.molecules
    assert molecules.ambient_density == 0.002 and molecules.ambient_abundances is None and sample.ambient_density == 0.002
    # The context's table.
    assert (acquisition.columns, acquisition.rows) == p["grid"]
    assert tuple(acquisition.fov_shape_zyx) == p["fov"] == tuple(template.shape_zyx)
    assert close(acquisition.spacing_zyx, p["spacing"]) and acquisition.overlap == 0.25
    assert close(acquisition.jitter_max_zyx, p["jitter"]) and acquisition.jitter == "uniform"
    assert lognormal(cells.nucleus_axial_radius, *c["axial"]) and lognormal(cells.nucleus_lateral_radius, *c["lateral"])
    assert cells.nucleus_elongation.mode == "folded_lognormal" and close(cells.nucleus_elongation.parameters,
                                                                         c["elongation"])
    assert close(nuclei.lobed_share, c["lobed"]) and close(config.packing.density, p["density"])
    assert close(config.packing.contact, c["contact"]) and config.packing.monolayer is c["monolayer"]
    assert config.territories.rule == c["rule"]
    if c["rule"] == "grown":
        assert lognormal(config.territories.growth, *c["growth"])
        assert lognormal(cells.reach, 2.5, 0.15)       # drawn but unused by T2; the reach law SD 0.15
    else:
        assert lognormal(cells.reach, *c["reach"])
    assert close((molecules.per_cell_median, molecules.per_cell_spread), p["per_cell"])
    assert close(molecules.nuclear_fraction, c["nuclear_fraction"])
    # The stains: the nuclear stain of the reference round; in the culture the morphology round's two stains.
    assert sample.stain is None
    assert all(stain.source != "boundary" for stain in config.stains)     # O2
    first = config.stains[0]
    assert (first.name, first.source, first.round, first.channel) == (c["stain"], "nucleus", "reference", "ch04")
    nuclear = [s for s in config.stains if s.source == "nucleus"]
    for stain in config.stains:
        assert lognormal(stain.brightness, 1000, c["spread"] if stain.source == "nucleus" else 0.3)
        assert stain.output == OutputScaleConfig(dtype="uint8", plateau=c["grey"])
        assert stain.noise == NoiseConfig(dependent_enabled=True, model="poisson", alpha=1.0,
                                          independent_enabled=True, sigma=c["sigma"])
        if c["field"] is None:
            assert stain.unevenness is None
        else:
            u = stain.unevenness
            assert (u.centres_per_area, u.length) == (4.0, None) and close(u.amplitude, c["field"])
        background = stain.background
        assert background.baseline_enabled and close(background.baseline, ((c["baseline"],),))
        assert not (background.gradient_enabled or background.texture_enabled)
        if c["region"] is None:
            assert not background.regions_enabled
        else:
            # One region at the sample grid's centre: height, σ 24.9 µm (128 LN pixels) in Y and X given in this
            # grid's voxels, n_z planes in Z, tissue weight 1.
            s = acquisition.spacing_zyx
            assert background.regions_enabled and close(background.region_heights, (c["region"],))
            assert close(background.region_centers, ((np.array(shape) - 1) / 2,))
            assert close(background.region_sigma_zyx, ((p["fov"][0], 128 * LN[1] / s[1], 128 * LN[2] / s[2]),))
            assert close(np.array(background.region_sigma_zyx)[0, 1:] * np.array(s[1:]), (24.9, 24.9), tol=0.02)
            assert close(background.tissue_weights, ((1.0,),))
    for stain in nuclear:
        assert close(stain.edge_width, c["edge"])
        count, size, size_sd, height, height_sd = c["texture"]
        texture = stain.texture
        assert texture.per_nucleus == count and lognormal(texture.size, size, size_sd)
        assert lognormal(texture.height, height, height_sd)
    if p["context"] == "culture":
        assert [(s.name, s.source, s.round, s.channel) for s in config.stains] == [
            ("DAPI", "nucleus", "reference", "ch04"), ("DAPI", "nucleus", "morphology", "ch00"),
            ("Flamingo", "territory", "morphology", "ch02")]
        flamingo = config.stains[2]
        assert (flamingo.nuclear_ratio, flamingo.edge_width, flamingo.texture) == (0.3, 0.5, None)
        for field in ("unevenness", "background", "noise", "output"):
            assert repr(getattr(flamingo, field)) == repr(getattr(config.stains[0], field)), field
        # The morphology DAPI is the reference DAPI's configuration in another round and channel.
        for field in fields(config.stains[0]):
            if field.name not in ("round", "channel"):
                assert repr(getattr(config.stains[1], field.name)) == repr(getattr(config.stains[0], field.name))
        m = config.morphology
        assert (m.name, m.translation, tuple(m.translation_max_zyx), m.translation_zyx) == (
            "organelle", "uniform", (0.5, 6.0, 6.0), None)
    else:
        assert len(config.stains) == 1 and config.morphology is None
    assert isinstance(first.background, BackgroundConfig)
    print(f"GN2 {name}: every value equals the PS1 table of {p['context']} and the common parameters; stains "
          f"{[(s.name, s.source, s.round, s.channel) for s in config.stains]}; no boundary source")
