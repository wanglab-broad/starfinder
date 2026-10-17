"""Contract checks of the appearance scene's truth schema, persistence and refusals
(docs/synthetic-real-appearance-validation.md, GA21 with the dataset of GA16, and the refusals and names of GA2 and
GA3 for the new entry).

``save_appearance_scene`` writes the AT2 layout of docs/synthetic-real-appearance.md ("Truth schema"): the
``save_context_scene`` layout with the shape columns null and the closed-form tables empty, ``placements.csv``,
``background_patches.csv`` and ``appearance.json`` with its versions and the SHA-256 of every file. The tables reload
equal, and the labels rebuilt by the test oracle from the saved files and the bank alone equal the saved label images.
``save_context_dataset`` of ``scene.context`` writes a dataset that ``Dataset``, ``FOV.load_images`` and
``FOV.prepare_morphology`` read, registering the morphology round. Every bank is a closed-form bank or is built in
the session from hand-made arrays; nothing is read from a run directory or a real image.
"""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import starfinder.synthetic as synthetic
from starfinder.barcode import load_codebook
from starfinder.dataset import ChannelInfo, Dataset, MorphologyConfig, RoundState
from starfinder.io import load_volume, load_volume_zyxc
from starfinder.synthetic import (APPEARANCE_TRUTH_VERSION, AppearanceScene, AppearanceSceneConfig, BankReference,
                                  CompositionConfig, LayoutConfig, StainChannelConfig, build_appearance_bank,
                                  generate_appearance_scene, load_appearance_bank, save_appearance_scene,
                                  save_context_dataset)
from starfinder.synthetic._context import _payload

from . import cellular_appearance_oracle as oracle
from .cellular_appearance_scenes import (APP_BANKS, app_context, app_layout, app_scene, spheres_case, write_case)
from .cellular_scenes import tiny

pytestmark = [pytest.mark.synthetic, pytest.mark.contract]

slow = pytest.mark.slow
NUCLEUS_COLUMNS = ["nucleus_id", "cell_id", "in_sample", "centre_z", "centre_y", "centre_x", "axial_radius",
                   "lateral_radius", "elongation", "angle", "site_radius", "stain_brightness"]
CELL_COLUMNS = ["cell_id", "kind", "in_sample", "anchor_z", "anchor_y", "anchor_x", "n_nuclei", "reach", "centroid_z",
                "centroid_y", "centroid_x", "n_voxels", "n_molecules", "fov_ids", "n_fovs", "label_fov_ids"]
SHAPE_COLUMNS = ["axial_radius", "lateral_radius", "elongation", "angle"]
EMPTY = {"lobes": ["nucleus_id", "lobe", "offset_z", "offset_y", "offset_x", "axial_radius", "lateral_radius",
                   "elongation", "angle"],
         "harmonics": ["nucleus_id", "degree", "order", "coefficient"],
         "stain_objects": ["stain", "object", "object_id", "plateau"],
         "bright_regions": ["region_id", "nucleus_id", "stain", "centre_z", "centre_y", "centre_x", "size", "height"],
         "unevenness": ["stain", "bump_id", "centre_z", "centre_y", "centre_x", "z", "a", "length"]}
PLACEMENT_COLUMNS = ["object_id", "kind", "entry_id", "source_image", "source_label", "source_plateau", "centre_z",
                     "centre_y", "centre_x", "anchor_z", "anchor_y", "anchor_x", "theta", "flip_z", "flip_y",
                     "scale_z", "scale_yx", "gain", "proposal", "overlap_share", "volume"]
PATCH_COLUMNS = ["stain", "lattice_z", "lattice_y", "lattice_x", "patch_id", "source_image"]
APPEARANCE_KEYS = {"generator", "appearance_truth_version", "appearance_preset_version", "requested_config",
                   "effective_config", "bank", "density", "placed", "proposals", "kappa", "plateau", "background",
                   "stream_scheme", "files"}


def generated(name):
    book, config, scene = app_scene(name)
    assert not isinstance(scene, Exception), scene
    return book, config, scene, load_appearance_bank(config.bank)


def stored_bank(directory):
    """app_cases' two-sphere bank built from hand-made arrays and loaded by name and SHA-256 (a stored bank)."""
    config = write_case(spheres_case(), directory / "source")
    path = directory / "bank" / f"{config['bank']}.npz"
    build_appearance_bank(config, path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    reference = BankReference(name=config["bank"], sha256=sha256, directory=str(path.parent))
    return reference, load_appearance_bank(reference)


def read(path):
    """A saved table, its floats parsed back exactly (pandas writes the shortest round-trip repr)."""
    return pd.read_csv(path, float_precision="round_trip")


# ---------------------------------------------------------------- GA21: the AT2 layout

@pytest.mark.parametrize("name", ["tissue2d", "tissue2d-plane", "tissue2d-bg1", "culture-morph-supplied",
                                  "culture", pytest.param("dense3d", marks=slow),
                                  pytest.param("dense3d-plane", marks=slow)])
def test_ga21_layout_versions_nulls_tables_and_hashes(name, tmp_path):
    """The directory save_appearance_scene writes: the save_context_scene layout, nuclei.csv and cells.csv with their
    columns, the shape columns null, the centre and anchor from placements.csv, site_radius the largest distance from
    c_k to a sample-grid voxel of the placed mask; the closed-form tables empty; appearance.json with its versions, the
    bank's name, SHA-256 and digest, κ, g and the SHA-256 of every file; the tables reloaded equal; each stain file
    under its StainChannelConfig's name, round and file pattern; context.json's requested_config as _payload writes
    it; the labels rebuilt from the saved files and the bank alone equal to the saved label images."""
    _, config, scene, bank = generated(name)
    root = save_appearance_scene(scene, tmp_path / "scene")
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    for required in ("sample.json", "context.json", "appearance.json", "placements.csv", "background_patches.csv",
                     "nuclei.csv", "cells.csv", "cell_context.csv", "stain_channels.csv", "morphology_truth.csv",
                     "labels/sample_nuclei.tif", "labels/sample_cells.tif", *[f"{t}.csv" for t in EMPTY]):
        assert required in files, required
    record = json.loads((root / "appearance.json").read_text(encoding="utf-8"))
    assert set(record) == APPEARANCE_KEYS
    assert record["appearance_truth_version"] == APPEARANCE_TRUTH_VERSION == "1"
    assert record["appearance_preset_version"] == "appearance-presets-v1"
    assert record["bank"]["name"] == bank.name and record["bank"]["sha256"] == bank.sha256
    assert record["bank"]["digest"] == bank.digest == bank.record["digest"]
    assert record["plateau"] == bank.record["plateau_median"]
    assert set(record["kappa"]) == {s.name for s in config.context.stains}
    assert record["kappa"] == scene.record["kappa"] and all(v >= 0 for v in record["kappa"].values())
    assert sorted(record["files"]) == [f for f in files if f != "appearance.json"]
    for file, digest in record["files"].items():
        assert hashlib.sha256((root / file).read_bytes()).hexdigest() == digest, file
    context = json.loads((root / "context.json").read_text(encoding="utf-8"))
    assert context["context_truth_version"] == "1"
    assert context["requested_config"] == json.loads(json.dumps(_payload(config.context)))
    # nuclei.csv and cells.csv: their columns, the shape columns null, centre and anchor from the placement.
    placements = read(root / "placements.csv")
    assert list(placements.columns) == PLACEMENT_COLUMNS
    nuclei, cells = read(root / "nuclei.csv"), read(root / "cells.csv")
    assert list(nuclei.columns) == NUCLEUS_COLUMNS and list(cells.columns) == CELL_COLUMNS
    assert nuclei[SHAPE_COLUMNS].isna().all().all() and cells.reach.isna().all()
    assert nuclei.nucleus_id.tolist() == nuclei.cell_id.tolist() == placements.object_id.tolist()
    for a in "zyx":
        assert nuclei[f"centre_{a}"].tolist() == placements[f"centre_{a}"].tolist()
    flat = config.context.sample.acquisition.fov_shape_zyx[0] == 1
    for a in "yx" if flat else "zyx":
        assert cells[f"anchor_{a}"].tolist() == placements[f"anchor_{a}"].tolist()
    if flat:
        assert (cells.anchor_z == 0).all()  # at Z=1 a cell's anchor is in the plane (membership by Y and X)
    assert (cells.kind == "ordinary").all() and (cells.n_nuclei == 1).all()
    # site_radius: the largest distance from c_k to a voxel of the placed mask on the sample grid.
    grid = scene.context.sample.record["sample_grid"]
    points = oracle.grid_points(grid["origin_zyx"], grid["spacing_zyx"], grid["shape_zyx"]).reshape(-1, 3)
    culture = oracle.kind_of(bank) == "cell"
    radius_error = 0.0
    for frame, row in zip(oracle.frames(bank, placements), nuclei.itertuples()):
        held = frame.d(points) <= 0
        if culture:
            held[held] &= frame.nd(points[held]) <= 0
        expected = float(np.sqrt(((points[held] - frame.c) ** 2).sum(axis=1).max()))
        radius_error = max(radius_error, abs(row.site_radius - expected))
    assert radius_error <= 1e-9
    # The closed-form tables are empty, with their columns.
    for table, columns in EMPTY.items():
        saved = read(root / f"{table}.csv")
        assert list(saved.columns) == columns and saved.empty, table
    # placements.csv and background_patches.csv reload equal.
    reloaded = placements.astype(scene.placements.dtypes.to_dict())
    pd.testing.assert_frame_equal(reloaded, scene.placements, check_exact=True)
    patches = read(root / "background_patches.csv")
    assert list(patches.columns) == PATCH_COLUMNS
    if len(scene.background_patches):
        pd.testing.assert_frame_equal(patches.astype(scene.background_patches.dtypes.to_dict()),
                                      scene.background_patches, check_exact=True)
    else:
        assert patches.empty and scene.record["background"]["mode"] == "closed-form"
    # Each stain keeps its StainChannelConfig's name, round and file pattern.
    channels = read(root / "stain_channels.csv")
    assert list(zip(channels.stain, channels["round"], channels.channel)) == [(s.name, s.round, s.channel)
                                                                             for s in config.context.stains]
    held_names = [s for s in config.context.stains if s.round == "morphology"]
    for fov in scene.context.sample.positions.fov_id:
        for stain in config.context.stains:
            if stain.round == "reference":
                loaded = load_volume_zyxc(root / "fovs" / fov / "stains" / f"{stain.name}.ome.tif")
                assert np.array_equal(loaded.image[..., 0], scene.context.stains[fov][stain.name])
        if held_names:
            loaded = load_volume_zyxc(root / "fovs" / fov / "morphology.ome.tif")
            assert np.array_equal(loaded.image, scene.context.morphology[fov])
    # The labels rebuilt from the saved files and the bank alone.
    growth = None
    if not culture:
        order = read(root / "cell_context.csv").set_index("cell_id").growth
        growth = [float(order[int(k)]) for k in placements.object_id]
    frames = oracle.frames(bank, placements)
    truth = read(root / "placement_truth.csv").set_index("fov_id")
    sample = json.loads((root / "sample.json").read_text(encoding="utf-8"))
    spacing = sample["sample_grid"]["spacing_zyx"]
    boundary, compared = 0, 0
    for fov in truth.index:
        row = truth.loc[fov]
        nuclei_image = load_volume(root / "fovs" / fov / "labels" / "nuclei.tif").image
        cells_image = load_volume(root / "fovs" / fov / "labels" / "cells.tif").image
        fov_points = oracle.grid_points([row.true_origin_z, row.true_origin_y, row.true_origin_x], spacing,
                                        nuclei_image.shape)
        N, C, near, _ = oracle.labels(fov_points.reshape(-1, 3), frames, oracle.kind_of(bank), growth)
        near = near.reshape(nuclei_image.shape)
        assert np.all((nuclei_image == N.reshape(nuclei_image.shape)) | near)
        assert np.all((cells_image == C.reshape(cells_image.shape)) | near)
        boundary += int(near.sum())
        compared += near.size
    print(f"GA21 {name}: {len(files)} files with their SHA-256 in appearance.json; nuclei.csv and cells.csv with "
          f"null shape columns, site_radius within {radius_error:.1e} µm; the empty tables {sorted(EMPTY)}; "
          f"placements.csv and background_patches.csv ({len(patches)} rows) reload equal; labels rebuilt from the "
          f"saved files and the bank on {compared} voxels, boundary voxels {boundary} (expected 0)")
    assert boundary == 0


# ---------------------------------------------------------------- GA16 and GA21: the dataset and the morphology round

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


@pytest.mark.parametrize("name", ["tissue2d-morph-supplied", "culture-morph-supplied"])
def test_ga21_dataset_registers_the_morphology_round(name, tmp_path):
    """save_context_dataset of scene.context read by Dataset, FOV.load_images and FOV.prepare_morphology, which
    registers the morphology round by one translation on the nuclear stain (MR1: the one reference stain shares its
    name with a channel of the round); the files hold the moving stain at p + d equal to the reference stain at p,
    d = (−1, 0, 0) voxels the pull displacement of morphology_truth, within 1e-9 of the maximum."""
    _, config, scene, _ = generated(name)
    root = save_context_dataset(scene.context, tmp_path / "raw")
    record, dataset = dataset_from(root, tmp_path)
    reference = [s for s in config.context.stains if s.round == "reference"]
    held = [s for s in config.context.stains if s.round == "morphology"]
    assert record["rounds"]["other"] == [config.context.morphology.name]
    assert [c["channel"] for c in record["other_channel_order"][config.context.morphology.name]] == \
        [s.channel for s in held]
    for file, digest in record["files"].items():
        assert hashlib.sha256((root / file).read_bytes()).hexdigest() == digest, file
    pull = scene.context.morphology_truth.set_index("fov_id")[
        ["pull_displacement_z_vox", "pull_displacement_y_vox", "pull_displacement_x_vox"]]
    recovered = {}
    for fov in scene.context.sample.positions.fov_id:
        loaded = dataset.fov(fov).load_images()
        for r, image in scene.context.sample.fovs[fov].rounds.items():
            assert np.array_equal(loaded.images[r], image)
        prepared = dataset.fov(fov).prepare_morphology(MorphologyConfig())
        stains = prepared.images["reference_stain"]
        assert stains.shape[-1] == len(reference)
        for c, stain in enumerate(reference):
            assert np.array_equal(stains[..., c], scene.context.stains[fov][stain.name])
        assert prepared.images[config.context.morphology.name].shape == stains.shape[:3] + (len(held),)
        assert prepared.registration_record["rounds"][config.context.morphology.name]["reference"] == "round1:ch04"
        transforms = prepared.registration_chains[config.context.morphology.name].transforms
        assert len(transforms) == 1
        recovered[fov] = np.array(transforms[0].displacement_zyx).tolist()
        truth = pull.loc[fov].to_numpy()
        assert np.all(np.abs(truth - np.array((-1.0, 0.0, 0.0))) <= 1e-9)
        nuclear = held[0]
        moving = load_volume(root / config.context.morphology.name / fov / f"{nuclear.channel}.tif").image
        fixed = load_volume(root / "round1" / fov / f"{reference[0].channel}.tif").image
        moving, fixed = moving.reshape(stains.shape[:3]), fixed.reshape(stains.shape[:3])
        peak = float(np.abs(fixed).max())
        assert np.all(np.abs(moving[:-1] - fixed[1:]) <= 1e-9 * peak)
    print(f"GA21 {name}: the dataset holds {len(record['files']) + 1} files; prepare_morphology registered "
          f"{config.context.morphology.name} by one translation per FOV {recovered} against the truth pull "
          f"displacements {pull.to_numpy().tolist()} (registration accuracy is not asserted; the files hold the "
          "moving stain shifted by the truth)")


# ---------------------------------------------------------------- names and refusals

def test_new_names_are_exported():
    for name in ("generate_appearance_scene", "save_appearance_scene", "AppearanceScene", "AppearanceSceneConfig",
                 "AugmentationConfig", "LayoutConfig", "CompositionConfig", "APPEARANCE_TRUTH_VERSION"):
        assert name in synthetic.__all__ and hasattr(synthetic, name), name
    assert synthetic.APPEARANCE_TRUTH_VERSION == "1"


def test_configuration_refusals():
    context = app_context("tissue2d")
    bank = BankReference(preset=APP_BANKS["tissue2d"])
    for kwargs, field in ((dict(light="soft"), "light"), (dict(background="flat"), "background"),
                          (dict(plateau=0.0), "plateau"), (dict(light_sigma_um=(1.0, 1.0)), "light_sigma_um"),
                          (dict(light_sigma_um=(0.0, 1.0, 1.0)), "light_sigma_um")):
        with pytest.raises(ValueError, match=field):
            CompositionConfig(**kwargs)
    with pytest.raises(TypeError, match="context"):
        AppearanceSceneConfig(object(), bank)
    with pytest.raises(TypeError, match="bank"):
        AppearanceSceneConfig(context, "tissue2d-v1")
    with pytest.raises(ValueError, match="sample.cells"):
        AppearanceSceneConfig(replace(context, sample=replace(context.sample, cells=None), territories=None), bank)
    with pytest.raises(TypeError, match="config"):
        generate_appearance_scene(tiny()[0], config=context)


def test_generation_refusals_and_a_stored_bank(tmp_path):
    """A stain the bank does not hold, BG2 or DN1 without its target on a stored bank raise ValueError naming them; a
    stored bank takes BG1 by default and records its SHA-256 and record path; save_appearance_scene refuses a
    non-scene, a nonempty directory and a scene generated with on_fov."""
    book, _ = tiny()
    reference, bank = stored_bank(tmp_path)
    context = app_context("dense3d")
    with pytest.raises(ValueError, match="'PI'"):
        generate_appearance_scene(book, config=AppearanceSceneConfig(
            replace(context, stains=(StainChannelConfig("PI"),)), reference, layout=app_layout(2)))
    with pytest.raises(ValueError, match="closed-form"):
        generate_appearance_scene(book, config=AppearanceSceneConfig(
            context, reference, layout=app_layout(2), composition=CompositionConfig(background="closed-form")))
    with pytest.raises(ValueError, match="layout.target and layout.threshold"):
        generate_appearance_scene(book, config=AppearanceSceneConfig(
            context, reference, layout=LayoutConfig(density="visible", contact=1.0)))
    seen = []
    scene = generate_appearance_scene(book, config=AppearanceSceneConfig(context, reference, layout=app_layout(2)),
                                      on_fov=lambda fov, *rest: seen.append(fov))
    assert isinstance(scene, AppearanceScene) and scene.record["background"]["mode"] == "patches"
    assert scene.record["bank"]["sha256"] == reference.sha256
    assert scene.record["bank"]["record_path"] == str(tmp_path / "bank" / f"{reference.name}.record.json")
    assert seen == list(scene.context.sample.positions.fov_id) and set(scene.context.stains) == set(seen)
    with pytest.raises(ValueError, match="on_fov"):
        save_appearance_scene(scene, tmp_path / "saved")
    with pytest.raises(TypeError, match="AppearanceScene"):
        save_appearance_scene(scene.context, tmp_path / "other")
    _, _, small, _ = generated("tissue2d-plane")
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file").write_text("x")
    with pytest.raises(FileExistsError):
        save_appearance_scene(small, tmp_path / "busy")
    print(f"refusals: a stain the bank lacks, BG2 and DN1 without a target on a stored bank; the stored bank "
          f"{reference.name} took BG1 ({len(scene.background_patches)} patch rows)")
