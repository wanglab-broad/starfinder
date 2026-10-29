"""Configuration, refusals and persistence of the cellular scene's stains and morphology round.

design/validation/synthetic-cellular-scene.md: GN2 (the names and fields of "Names" this issue adds), GN3 (the refusals
of the stains and the morphology round) and GN22 (the truth tables, the stain images of ``save_cellular_scene``, and
the dataset of ``save_cellular_dataset`` read by ``Dataset``, ``FOV.load_images`` and ``FOV.prepare_morphology``).
"""
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

import starfinder.synthetic as synthetic
from starfinder.readout import load_codebook
from starfinder.dataset import ChannelInfo, Dataset, MorphologyConfig, RoundState
from starfinder.registration import RegistrationRecipe, RegistrationStep
from starfinder.io import load_volume, load_volume_zyxc
from starfinder.registration import RegistrationSignalConfig, TranslationConfig
from starfinder.synthetic import (BackgroundConfig, BrightRegionConfig, CellularSceneConfig, MorphologyRoundConfig,
                                  NoiseConfig, OutputScaleConfig, ScalarDistribution, StainChannelConfig,
                                  UnevennessConfig, generate_cellular_scene, save_cellular_dataset, save_cellular_scene)

from .cellular_scenes import FOVS, WHOLE, plane, scene, stain_config, tiny

pytestmark = [pytest.mark.synthetic, pytest.mark.contract]

# The columns of the five tables of this issue ("Truth schema"), in order.
STAIN_TABLES = {
    "stain_channels": ["stain", "source", "round", "channel", "dtype", "scale", "m_A"],
    "stain_objects": ["stain", "object", "object_id", "plateau"],
    "bright_regions": ["region_id", "nucleus_id", "stain", "centre_z", "centre_y", "centre_x", "size", "height"],
    "unevenness": ["stain", "bump_id", "centre_z", "centre_y", "centre_x", "z", "a", "length"],
    "morphology_truth": ["fov_id", "round", "translation_z_um", "translation_y_um", "translation_x_um",
                         "translation_z_vox", "translation_y_vox", "translation_x_vox", "pull_displacement_z_vox",
                         "pull_displacement_y_vox", "pull_displacement_x_vox"],
}


def names(kind):
    return [f.name for f in fields(kind)]


# ---------------------------------------------------------------- GN2: the names and fields of this issue

def test_gn2_stain_names_and_fields():
    new = {"StainChannelConfig", "BrightRegionConfig", "UnevennessConfig", "OutputScaleConfig",
           "MorphologyRoundConfig", "save_cellular_dataset"}
    assert new <= set(synthetic.__all__) and all(hasattr(synthetic, name) for name in new)
    assert names(StainChannelConfig) == ["name", "source", "round", "channel", "brightness", "edge_width",
                                         "band_width", "nuclear_ratio", "texture", "unevenness", "background", "noise",
                                         "output"]
    stain = StainChannelConfig("DAPI")
    assert (stain.source, stain.round, stain.channel, stain.edge_width, stain.band_width, stain.nuclear_ratio,
            stain.texture, stain.unevenness, stain.output) == ("nucleus", "reference", "ch04", 0.5, 0.5, 0.3, None,
                                                               None, None)
    assert stain.brightness == ScalarDistribution("lognormal", (math.log(1000), 0.2))
    assert stain.background == BackgroundConfig(baseline_enabled=True, baseline=((100.0,),))
    assert stain.noise == NoiseConfig(dependent_enabled=True, model="poisson", alpha=1.0, independent_enabled=True,
                                      sigma=3.0)
    regions = BrightRegionConfig()
    assert names(BrightRegionConfig) == ["per_nucleus", "size", "height"]
    assert (regions.per_nucleus, regions.size, regions.height) == (
        2.0, ScalarDistribution("lognormal", (math.log(0.3), 0.4)), ScalarDistribution("lognormal", (0.0, 0.3)))
    field = UnevennessConfig()
    assert names(UnevennessConfig) == ["centres_per_area", "amplitude", "length"]
    assert (field.centres_per_area, field.amplitude, field.length) == (4.0, 0.2, None)
    output = OutputScaleConfig()
    assert names(OutputScaleConfig) == ["dtype", "plateau"] and (output.dtype, output.plateau) == ("uint8", 50.0)
    morphology = MorphologyRoundConfig()
    assert names(MorphologyRoundConfig) == ["name", "translation", "translation_max_zyx", "translation_zyx"]
    assert (morphology.name, morphology.translation, morphology.translation_max_zyx, morphology.translation_zyx) == (
        "organelle", "uniform", (0.5, 6.0, 6.0), None)
    for value in (stain, regions, field, output, morphology):
        with pytest.raises(TypeError):
            hash(value)
        with pytest.raises(FrozenInstanceError):
            setattr(value, names(type(value))[0], None)


# ---------------------------------------------------------------- GN3: refusals

def _refusals():
    _, config = tiny()
    free = replace(config, stain=None)
    one = replace(plane(config), stain=None)
    dapi = StainChannelConfig("DAPI")
    held = replace(dapi, round="morphology", channel="ch00")
    return [
        ("source", lambda: StainChannelConfig("DAPI", source="cytoplasm"), "source"),
        ("round", lambda: StainChannelConfig("DAPI", round="organelle"), "round"),
        ("name with a slash", lambda: StainChannelConfig("a/b"), "name"),
        ("edge_width 0", lambda: StainChannelConfig("DAPI", edge_width=0.0), "edge_width"),
        ("band_width < 0", lambda: StainChannelConfig("DAPI", band_width=-0.1), "band_width"),
        ("texture on a territory stain", lambda: StainChannelConfig("F", source="territory",
                                                                    texture=BrightRegionConfig()), "texture"),
        ("negative per_nucleus", lambda: BrightRegionConfig(per_nucleus=-1.0), "per_nucleus"),
        ("negative amplitude", lambda: UnevennessConfig(amplitude=-0.1), "amplitude"),
        ("length 0", lambda: UnevennessConfig(length=0.0), "length"),
        ("output dtype", lambda: OutputScaleConfig(dtype="float32"), "dtype"),
        ("plateau 255", lambda: OutputScaleConfig(plateau=255.0), "plateau"),
        ("translation", lambda: MorphologyRoundConfig(translation="drift"), "translation"),
        ("supplied beyond the bound", lambda: MorphologyRoundConfig(
            translation="supplied", translation_zyx={"FOV_001": (0.0, 7.0, 0.0)}), "translation_zyx"),
        ("translation_zyx with uniform", lambda: MorphologyRoundConfig(translation_zyx={"FOV_001": WHOLE}),
         "translation_zyx"),
        ("a name twice in a round",
         lambda: CellularSceneConfig(sample=free, stains=(dapi, replace(dapi, channel="ch05"))),
         "name"),
        ("a channel twice in a round", lambda: CellularSceneConfig(sample=free, stains=(
            dapi, StainChannelConfig("PI"))), "channel"),
        ("a shared name, another configuration", lambda: CellularSceneConfig(
            sample=free, stains=(dapi, replace(held, edge_width=0.3)), morphology=MorphologyRoundConfig()), "differ"),
        ("morphology stains without the round", lambda: CellularSceneConfig(sample=free, stains=(dapi, held)),
         "morphology"),
        ("a morphology round without stains", lambda: CellularSceneConfig(sample=free, stains=(dapi,),
                                                                          morphology=MorphologyRoundConfig()),
         "morphology"),
        ("T_z at Z=1", lambda: CellularSceneConfig(sample=one, stains=(dapi, held), morphology=MorphologyRoundConfig()),
         "translation_max_zyx"),
        ("supplied FOVs", lambda: CellularSceneConfig(sample=free, stains=(dapi, held), morphology=MorphologyRoundConfig(
            translation="supplied", translation_zyx={"FOV_001": WHOLE})), "translation_zyx"),
    ]


@pytest.mark.parametrize("case", _refusals(), ids=lambda case: case[0])
def test_gn3_stain_refusals(case):
    _, build, field_name = case
    with pytest.raises(ValueError, match=field_name):
        build()


def test_gn3_stain_generation_refusals():
    book, _ = tiny()
    config = stain_config("stains-reach-whole")
    amplicon = replace(config, stains=(replace(config.stains[0], channel="ch01"),) + config.stains[1:])
    with pytest.raises(ValueError, match="amplicon channels"):
        generate_cellular_scene(book, config=amplicon)
    with pytest.raises(ValueError, match="round name 'round2'"):
        generate_cellular_scene(book, config=replace(config, morphology=replace(config.morphology, name="round2")))
    with pytest.raises(TypeError, match="StainChannelConfig"):
        CellularSceneConfig(sample=config.sample, stains=("DAPI",))
    # A supplied translation for every FOV within the bound is accepted.
    supplied = MorphologyRoundConfig(translation="supplied", translation_zyx={fov: WHOLE for fov in FOVS})
    assert CellularSceneConfig(sample=config.sample, stains=config.stains, morphology=supplied).morphology == supplied


# ---------------------------------------------------------------- GN22: truth schema and persistence

def _read(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=True)


def _equal_table(saved, table):
    assert list(saved.columns) == list(table.columns) and len(saved) == len(table)
    for column in table.columns:
        for a, b in zip(saved[column].tolist(), table[column].tolist()):
            if pd.isna(b):
                assert pd.isna(a), column
            elif isinstance(b, (float, np.floating)):
                assert float(a) == float(b), column  # round-trip precision
            else:
                assert str(a) == str(b), column


@pytest.mark.parametrize("name", ["stains-grown-drawn", "plane-whole", "stains-empty", "stains-off"])
def test_gn22_saved_stain_tables_and_images(name, tmp_path):
    _, config, generated = scene(name)
    root = save_cellular_scene(generated, tmp_path / "context")
    record = json.loads((root / "context.json").read_text(encoding="utf-8"))
    for table, columns in STAIN_TABLES.items():
        assert list(getattr(generated, table).columns) == columns
        _equal_table(_read(root / f"{table}.csv"), getattr(generated, table))
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    stain_files = sorted(f for f in files if "/stains/" in f or f.endswith("morphology.ome.tif"))
    reference = [s.name for s in config.stains if s.round == "reference"]
    expected = sorted([f"fovs/{fov}/stains/{n}.ome.tif" for fov in FOVS for n in reference]
                      + ([f"fovs/{fov}/morphology.ome.tif" for fov in FOVS] if config.morphology else []))
    assert stain_files == expected
    for file, digest in record["files"].items():
        assert hashlib.sha256((root / file).read_bytes()).hexdigest() == digest, file
    for fov in FOVS:
        for stain_name in reference:
            loaded = load_volume_zyxc(root / "fovs" / fov / "stains" / f"{stain_name}.ome.tif")
            assert np.array_equal(loaded.image[..., 0], generated.stains[fov][stain_name])
            assert loaded.image.dtype == generated.stains[fov][stain_name].dtype
        if config.morphology:
            loaded = load_volume_zyxc(root / "fovs" / fov / "morphology.ome.tif")
            assert np.array_equal(loaded.image, generated.morphology[fov])
    assert record["requested_config"]["stains"] == json.loads(json.dumps(generated.record["requested_config"]
                                                                         ["stains"]))
    if name == "stains-empty":
        for table in ("stain_objects", "bright_regions"):
            saved = _read(root / f"{table}.csv")
            assert list(saved.columns) == STAIN_TABLES[table] and saved.empty
    if name == "stains-off":
        saved = _read(root / "morphology_truth.csv")
        assert list(saved.columns) == STAIN_TABLES["morphology_truth"] and saved.empty
        assert not generated.morphology


def dataset_from(root, tmp_path):
    """A Dataset from the fields dataset.json records, with the codebook loaded from codebook.csv."""
    record = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    rounds = record["rounds"]
    dataset = Dataset(root, tmp_path / "out", record["dataset_id"], record["sample_id"], "out",
                      RoundState(rounds["sequencing"], rounds["other"], rounds["reference"]),
                      tuple(record["channel_order"]),
                      other_channel_order={name: tuple(ChannelInfo(c["channel"], c["name"]) for c in channels)
                                           for name, channels in record["other_channel_order"].items()},
                      reference_stains=tuple(ChannelInfo(c["channel"], c["name"]) for c in record["reference_stains"]),
                      voxel_size_xy=record["voxel_size_xy"], voxel_size_z=record["voxel_size_z"])
    labels = record["codebook"]
    dataset.codebook = load_codebook(root / labels["path"], round_labels=tuple(labels["round_labels"]),
                                     channel_labels=tuple(labels["channel_labels"]))
    return record, dataset


@pytest.mark.parametrize("name", ["mr1", "stains-reach-whole", "plane-whole"])
def test_gn22_dataset_read_and_morphology_registered(name, tmp_path):
    """save_cellular_dataset's layout read by Dataset, FOV.load_images and FOV.prepare_morphology, which registers
    the morphology round by one translation on DAPI: without a recipe on the culture layout (MR1, one reference
    stain), with a recipe naming the DAPI stain where the reference round holds several stains."""
    _, config, generated = scene(name)
    root = save_cellular_dataset(generated, tmp_path / "raw")
    record, dataset = dataset_from(root, tmp_path)
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    reference = [s for s in config.stains if s.round == "reference"]
    held = [s for s in config.stains if s.round == "morphology"]
    rounds = ("round1", "round2", "round3", "round4")
    expected = (["codebook.csv", "dataset.json"]
                + [f"{r}/{fov}/ch0{c}.tif" for r in rounds for fov in FOVS for c in range(4)]
                + [f"round1/{fov}/{s.channel}.tif" for fov in FOVS for s in reference]
                + [f"organelle/{fov}/{s.channel}.tif" for fov in FOVS for s in held])
    assert files == sorted(expected)
    assert sorted(record["files"]) == sorted(f for f in files if f != "dataset.json")
    for file, digest in record["files"].items():
        assert hashlib.sha256((root / file).read_bytes()).hexdigest() == digest, file
    book, _ = tiny()
    pd.testing.assert_frame_equal(dataset.codebook.table[["gene_id", "color_sequence"]],
                                  book.table[["gene_id", "color_sequence"]], check_exact=True)
    pull = generated.morphology_truth.set_index("fov_id")[
        ["pull_displacement_z_vox", "pull_displacement_y_vox", "pull_displacement_x_vox"]]
    recovered, exact = {}, 0
    for fov in FOVS:
        loaded = dataset.fov(fov).load_images()
        for r in rounds:
            assert np.array_equal(loaded.images[r], generated.sample.fovs[fov].rounds[r]), (fov, r)
            assert loaded.metadata[r] == generated.sample.fov_metadata[fov]
        recipe = None
        if len(reference) > 1:
            recipe = RegistrationRecipe((RegistrationStep(TranslationConfig()),), signal=RegistrationSignalConfig(
                "channel", reference_channel="ch04", moving_channel="ch00"))
        prepared = dataset.fov(fov).prepare_morphology(MorphologyConfig(recipe=recipe))
        stains = prepared.images["reference_stain"]
        assert stains.shape[-1] == len(reference)
        for c, stain in enumerate(reference):
            assert np.array_equal(stains[..., c], generated.stains[fov][stain.name])
        assert prepared.images["organelle"].shape == stains.shape[:3] + (len(held),)
        assert prepared.registration_record["rounds"]["reference_stain"]["relation"]
        assert prepared.registration_record["rounds"]["organelle"]["reference"] == "round1:ch04"
        transforms = prepared.registration_chains["organelle"].transforms
        assert len(transforms) == 1
        displacement = np.array(transforms[0].displacement_zyx)
        dapi = stains[..., 0]
        recovered[fov] = dict(displacement=displacement.tolist(),
                              lit_voxels=int((dapi > dapi.min() + 0.01 * (dapi.max() - dapi.min() + 1)).sum()))
        truth = pull.loc[fov].to_numpy()
        exact += int(np.array_equal(displacement, np.round(truth)))
        if config.morphology.translation == "supplied":
            # The files hold what registration recovers when it is exact: the moving DAPI at p + d is the reference
            # DAPI at p, d the pull displacement of morphology_truth (here (0, −2, 3) voxels).
            moving = load_volume(root / "organelle" / fov / "ch00.tif").image.reshape(stains.shape[:3])
            fixed = load_volume(root / "round1" / fov / "ch04.tif").image.reshape(stains.shape[:3])
            assert np.all(np.abs(truth - np.array((0.0, -2.0, 3.0))) <= 1e-9)  # truth columns within 1e-9 voxels
            ny, nx = fixed.shape[1:]
            peak = float(np.abs(fixed).max())
            assert np.all(np.abs(moving[:, :ny - 2, 3:] - fixed[:, 2:, :nx - 3]) <= 1e-10 * peak)
    print(f"GN22 {name}: the dataset holds {len(files)} files; prepare_morphology registered organelle by one "
          f"translation per FOV, {recovered}, against the truth pull displacements "
          f"{pull.round(3).to_numpy().tolist()}: the integer peak equals the rounded truth in {exact} of {len(FOVS)} "
          "FOVs (a FOV whose reference DAPI holds no nucleus, lit_voxels 0, has nothing to register; registration "
          "accuracy on these 36 × 36 FOVs is not asserted, it belongs to the chain check CU0)")
    if config.morphology.translation == "supplied":
        # The convention check: a sign or axis error could not match the truth on any FOV.
        assert exact >= 1


def test_gn22_dataset_refusals(tmp_path):
    _, _, generated = scene("mr1")
    root = save_cellular_dataset(generated, tmp_path / "first")
    with pytest.raises(FileExistsError):
        save_cellular_dataset(generated, root)
    with pytest.raises(TypeError):
        save_cellular_dataset(generated.sample, tmp_path / "second")
