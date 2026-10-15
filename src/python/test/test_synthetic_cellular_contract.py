"""Configuration, refusals and persistence of the cellular scene (docs/synthetic-cellular-scene-validation.md).

GN2 (the fields of the existing configurations as at 943bb6b, the new names and fields), GN3 (the
refusals of the geometry and molecule parts) and GN22 (the truth schema and ``save_context_scene``)
for the parts of W-483. The stains, the morphology round and the presets follow with their issues.
"""
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

import starfinder.synthetic as synthetic
from starfinder.synthetic import (CONTEXT_TRUTH_VERSION, SAMPLE_PRESETS, CellTypeConfig, CellularSceneConfig,
                                  ContextScene, ContextSceneConfig, MoleculeConfig, NucleusShapeConfig, PackingConfig,
                                  SampleSceneConfig, ScalarDistribution, StainConfig, TerritoryConfig,
                                  generate_context_scene, save_context_scene, save_sample_scene)

from .cellular_oracle import harmonic_bound
from .cellular_scenes import SHAPES, config_of, plane, scene, tiny

pytestmark = [pytest.mark.synthetic, pytest.mark.contract]

# The columns of "Truth schema" of docs/synthetic-cellular-scene.md, in order.
TABLES = {
    "lobes": ["nucleus_id", "lobe", "offset_z", "offset_y", "offset_x", "axial_radius", "lateral_radius",
              "elongation", "angle"],
    "harmonics": ["nucleus_id", "degree", "order", "coefficient"],
    "cell_context": ["cell_id", "kind", "type", "region", "molecule_mean", "growth"],
    "regions": ["region_id", "centre_z", "centre_y", "centre_x", "main_type"],
    "type_abundances": ["type", "gene_id", "weight", "role"],
    "cell_gene_counts": ["cell_id", "gene_id", "count", "nuclear_count"],
    "marker_pairs": ["gene_a", "gene_b", "relation", "cells_a", "cells_b", "both", "either", "mecr"],
}


def names(kind):
    return [f.name for f in fields(kind)]


# ---------------------------------------------------------------- GN2: names, fields and the existing registry

def test_gn2_existing_fields_and_presets_as_at_the_base():
    assert names(SampleSceneConfig) == ["fov", "acquisition", "cells", "stain", "molecules_per_cell",
                                        "ambient_density", "max_count"]
    assert names(CellularSceneConfig) == ["nuclei", "binucleate", "empty", "nucleus_axial_radius",
                                          "nucleus_lateral_radius", "nucleus_elongation", "nucleus_angle",
                                          "empty_radius", "gap", "reach", "max_proposals"]
    assert names(StainConfig) == ["channel_label", "brightness", "edge_width", "background", "noise"]
    assert list(SAMPLE_PRESETS) == ["sample-tiny-v1", "sample-small-v1"]


def test_gn2_new_names_and_fields():
    new = {"generate_context_scene", "ContextSceneConfig", "ContextScene", "NucleusShapeConfig", "PackingConfig",
           "TerritoryConfig", "CellTypeConfig", "MoleculeConfig", "save_context_scene", "CONTEXT_TRUTH_VERSION"}
    assert new <= set(synthetic.__all__) and all(hasattr(synthetic, name) for name in new)
    assert CONTEXT_TRUTH_VERSION == "1"
    config = ContextSceneConfig()
    assert names(ContextSceneConfig) == ["sample", "nuclei", "packing", "territories", "types", "molecules",
                                         "stains", "morphology"]
    assert (config.nuclei, config.packing, config.territories, config.types, config.molecules, config.stains,
            config.morphology) == (None, None, None, None, None, (), None)
    assert names(ContextScene) == ["sample", "stains", "morphology", "lobes", "harmonics", "cell_context", "regions",
                                   "type_abundances", "cell_gene_counts", "marker_pairs", "stain_channels",
                                   "stain_objects", "bright_regions", "unevenness", "morphology_truth", "record"]
    shape = NucleusShapeConfig()
    assert names(NucleusShapeConfig) == ["lobed_share", "extra_lobes", "lobe_scale", "lobe_offset", "harmonics",
                                         "harmonic_amplitude"]
    assert (shape.lobed_share, shape.extra_lobes, shape.harmonics, shape.harmonic_amplitude) == (0.0, (1, 2), 0, 0.0)
    assert (shape.lobe_scale, shape.lobe_offset) == (ScalarDistribution("uniform", (0.5, 0.8)),
                                                     ScalarDistribution("uniform", (0.5, 0.9)))
    packing = PackingConfig()
    assert names(PackingConfig) == ["density", "kind_shares", "contact", "overlap", "gap", "monolayer", "on_crowded",
                                    "floor"]
    assert (packing.density, packing.kind_shares, packing.contact, packing.overlap, packing.gap, packing.monolayer,
            packing.on_crowded, packing.floor) == (None, None, 0.0, 0.0, 0.0, False, "raise", 0.9)
    territory = TerritoryConfig()
    assert names(TerritoryConfig) == ["rule", "growth"]
    assert territory.rule == "reach" and territory.growth == ScalarDistribution("lognormal", (math.log(0.8), 0.15))
    types = CellTypeConfig()
    assert names(CellTypeConfig) == ["types", "abundances", "proportions", "regions", "purity", "molecules_per_cell",
                                     "nuclear_fraction", "size_scale", "marker_pairs"]
    # The per-type fields are exactly these three ("Cell types").
    assert {name for name in names(CellTypeConfig)} - {"types", "abundances", "proportions", "regions", "purity",
                                                       "marker_pairs"} == {"molecules_per_cell", "nuclear_fraction",
                                                                           "size_scale"}
    assert types.types == ("type-1", "type-2", "type-3", "type-4") and types.proportions == (0.25,) * 4
    assert (types.regions, types.purity) == (4, 0.9)
    assert types.marker_pairs == (("balanced-01", "balanced-03", "exclusive"),
                                  ("balanced-01", "balanced-02", "coexpressed"))
    g1 = [[0.15 if g in (2 * t, 2 * t + 1) else 0.0875 if g >= 8 else 0.0 for g in range(16)] for t in range(4)]
    assert np.allclose(types.abundances, g1, rtol=0, atol=1e-12)
    molecules = MoleculeConfig()
    assert names(MoleculeConfig) == ["per_cell_median", "per_cell_spread", "nuclear_fraction", "ambient_density",
                                     "ambient_abundances"]
    assert (molecules.per_cell_median, molecules.per_cell_spread, molecules.nuclear_fraction,
            molecules.ambient_density, molecules.ambient_abundances) == (8.0, 0.0, None, None, None)
    for value in (config, shape, packing, territory, types, molecules):
        with pytest.raises(TypeError):
            hash(value)
        with pytest.raises(FrozenInstanceError):
            setattr(value, names(type(value))[0], None)


def test_stains_and_morphology_refuse_a_value_until_their_issue():
    """Placeholder of W-483: the stains and the morphology round are added by W-484, which may change this test."""
    _, config = tiny()
    with pytest.raises(NotImplementedError, match="W-484"):
        ContextSceneConfig(sample=replace(config, stain=None), stains=(object(),))
    with pytest.raises(NotImplementedError, match="W-484"):
        ContextSceneConfig(sample=config, morphology=object())


# ---------------------------------------------------------------- GN3: refusals

def _refusals():
    _, config = tiny()
    one = plane(config)
    empty_only = replace(config, cells=replace(config.cells, nuclei=0, binucleate=0, empty=2))
    return [
        ("stains with sample.stain", lambda: ContextSceneConfig(sample=config, stains=(object(),)), "sample.stain"),
        ("harmonic bound, sphere", lambda: ContextSceneConfig(sample=config, nuclei=replace(
            SHAPES, harmonics=3, harmonic_amplitude=1 / harmonic_bound(3, False))), "harmonic_amplitude · M_Y"),
        ("harmonic bound, plane", lambda: ContextSceneConfig(sample=one, nuclei=replace(
            SHAPES, harmonics=2, harmonic_amplitude=1.0)), "harmonic_amplitude · M_Y"),
        ("overlap 0.5", lambda: PackingConfig(overlap=0.5), "overlap"),
        ("overlap < 0", lambda: PackingConfig(overlap=-0.1), "overlap"),
        ("gap < 0", lambda: PackingConfig(gap=-1e-3), "gap"),
        ("contact > 1", lambda: PackingConfig(contact=1.5), "contact"),
        ("contact < 0", lambda: PackingConfig(contact=-0.1), "contact"),
        ("density 0", lambda: PackingConfig(density=0.0), "density"),
        ("density 1", lambda: PackingConfig(density=1.0), "density"),
        ("shares both 0", lambda: PackingConfig(density=0.2, kind_shares=(0.0, 0.0, 1.0)), "kind_shares"),
        ("counts both 0", lambda: ContextSceneConfig(sample=empty_only, packing=PackingConfig(density=0.2)),
         "kind_shares"),
        ("floor 0", lambda: PackingConfig(floor=0.0), "floor"),
        ("floor > 1", lambda: PackingConfig(floor=1.5), "floor"),
        ("monolayer, one plane", lambda: ContextSceneConfig(sample=one, packing=PackingConfig(monolayer=True)),
         "monolayer"),
        ("lobed_share > 1", lambda: NucleusShapeConfig(lobed_share=1.5), "lobed_share"),
        ("extra_lobes", lambda: NucleusShapeConfig(extra_lobes=(0, 2)), "extra_lobes"),
        ("harmonics 1", lambda: NucleusShapeConfig(harmonics=1), "harmonics"),
        ("territory rule", lambda: TerritoryConfig(rule="nearest"), "rule"),
        ("per-type length", lambda: CellTypeConfig(nuclear_fraction=(0.5, 0.5)), "nuclear_fraction"),
        ("purity > 1", lambda: CellTypeConfig(purity=1.2), "purity"),
        ("marker relation", lambda: CellTypeConfig(marker_pairs=(("balanced-01", "balanced-02", "x"),)),
         "marker_pairs"),
        ("nuclear_fraction > 1", lambda: MoleculeConfig(nuclear_fraction=1.5), "nuclear_fraction"),
        ("parts without cells", lambda: ContextSceneConfig(sample=replace(config, cells=None), nuclei=SHAPES),
         "sample.cells"),
    ]


@pytest.mark.parametrize("case", _refusals(), ids=lambda case: case[0])
def test_gn3_configuration_refusals(case):
    _, build, field_name = case
    with pytest.raises(ValueError, match=field_name.replace("·", ".")):
        build()


def test_gn3_generation_refusals():
    book, config = tiny()
    with pytest.raises(ValueError, match="abundances"):
        generate_context_scene(book, config=ContextSceneConfig(sample=config, types=CellTypeConfig(
            abundances=tuple(row[:8] for row in CellTypeConfig().abundances))))
    with pytest.raises(ValueError, match="marker_pairs"):
        generate_context_scene(book, config=ContextSceneConfig(sample=config, types=CellTypeConfig(
            marker_pairs=(("balanced-01", "gene-x", "exclusive"),))))
    # A cytoplasm molecule with no room: the territory is the nucleus grown by 1e-9 µm and every molecule goes to the
    # cytoplasm.
    cramped = ContextSceneConfig(
        sample=replace(config, cells=replace(config.cells, max_proposals=200)),
        territories=TerritoryConfig(rule="grown", growth=ScalarDistribution("constant", (1e-9,))),
        molecules=MoleculeConfig(per_cell_median=50, nuclear_fraction=0.0))
    with pytest.raises(ValueError, match=r"cytoplasm of cell 1 after max_proposals=200"):
        generate_context_scene(book, config=cramped)


def test_gn3_a_crowded_scene_below_its_floor():
    """F2's floor: placement stops at the same seed as ctx_crowded (the same draws), below floor · density = 0.81;
    the error names the seed and both densities (ctx_crowded at floor 0.9 reaches its floor: finding 1)."""
    book, _ = tiny()
    _, _, reference = scene("crowded-0.1")
    record = reference.record["effective_config"]["packing"]
    with pytest.raises(ValueError) as error:
        generate_context_scene(book, config=config_of("crowded-below"))
    message = str(error.value)
    assert message.startswith(f"{record['stopped_at']}:")
    assert f"reached density of {record['reached_density']}" in message and "0.9 · 0.9 = 0.81" in message


# ---------------------------------------------------------------- GN22: truth schema and persistence

def _read(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=True)


def _equal_table(saved, table):
    assert list(saved.columns) == list(table.columns)
    assert len(saved) == len(table)
    for column in table.columns:
        expected = table[column].tolist()
        got = saved[column].tolist()
        for a, b in zip(got, expected):
            if pd.isna(b):
                assert pd.isna(a), column
            elif isinstance(b, (float, np.floating)):
                assert float(a) == float(b), column  # round-trip precision
            else:
                assert str(a) == str(b), column


# ctx_types alone takes about 5 s to generate and save twice, so its case is `slow`.
@pytest.mark.parametrize("name", [pytest.param("types", marks=pytest.mark.slow), "harmonic-grown", "plane"])
def test_gn22_saved_layout_tables_and_record(name, tmp_path):
    _, config, generated = scene(name)
    root = save_context_scene(generated, tmp_path / "context")
    sample_root = save_sample_scene(generated.sample, tmp_path / "sample")
    sample_files = sorted(p.relative_to(sample_root).as_posix() for p in sample_root.rglob("*") if p.is_file())
    for file in sample_files:
        assert (root / file).read_bytes() == (sample_root / file).read_bytes(), file
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert sorted(set(files) - set(sample_files)) == sorted([f"{t}.csv" for t in TABLES] + ["context.json"])
    for table, columns in TABLES.items():
        assert list(getattr(generated, table).columns) == columns
        _equal_table(_read(root / f"{table}.csv"), getattr(generated, table))
    record = json.loads((root / "context.json").read_text(encoding="utf-8"))
    assert record["context_truth_version"] == CONTEXT_TRUTH_VERSION == "1"
    assert record["generator"] == "starfinder.synthetic.generate_context_scene"
    assert set(record) >= {"context_preset_version", "requested_config", "effective_config", "stream_scheme", "seed",
                           "files"}
    assert record["requested_config"]["type"] == "ContextSceneConfig"
    assert record["requested_config"]["sample"] == generated.sample.record["requested_config"]
    assert {k: v for k, v in record.items() if k != "files"} == json.loads(json.dumps(generated.record))
    assert sorted(record["files"]) == sorted(f for f in files if f != "context.json")
    for file, digest in record["files"].items():
        assert hashlib.sha256((root / file).read_bytes()).hexdigest() == digest, file
    sample_record = json.loads((root / "sample.json").read_text(encoding="utf-8"))
    assert sample_record["sample_truth_version"] == "1"
    with pytest.raises(FileExistsError):
        save_context_scene(generated, root)


def test_gn22_empty_populations(tmp_path):
    book, config = tiny()
    # A fifth type with no cell (proportion 0, so no region's main type and no fallback draw), no N2, no regions.
    types = CellTypeConfig(types=("type-1", "type-2", "type-3", "type-4", "type-5"),
                           abundances=CellTypeConfig().abundances + ((0.0,) * 8 + (0.125,) * 8,),
                           proportions=(0.25, 0.25, 0.25, 0.25, 0.0), regions=0)
    generated = generate_context_scene(book, config=ContextSceneConfig(sample=config, types=types))
    assert "type-5" not in set(generated.cell_context.type)
    assert (generated.type_abundances.type == "type-5").sum() == 16
    assert generated.cell_context.region.isna().all()
    root = save_context_scene(generated, tmp_path / "types")
    for table in ("harmonics", "regions"):
        saved = _read(root / f"{table}.csv")
        assert list(saved.columns) == TABLES[table] and saved.empty
    off = generate_context_scene(book, config=ContextSceneConfig(sample=config))
    root = save_context_scene(off, tmp_path / "off")
    for table in ("harmonics", "regions", "type_abundances", "marker_pairs"):
        saved = _read(root / f"{table}.csv")
        assert list(saved.columns) == TABLES[table] and saved.empty
    saved = _read(root / "lobes.csv")
    assert len(saved) == len(off.sample.nuclei) and (saved.lobe == 1).all()
    context = _read(root / "cell_context.csv")
    assert context.type.isna().all() and context.growth.isna().all() and (context.molecule_mean == 8.0).all()
