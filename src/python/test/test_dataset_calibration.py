"""W-764: declared raw calibration, preserved metadata and useful dataset refusals."""
from dataclasses import replace

import numpy as np
import pytest

from starfinder.dataset import Dataset
from starfinder.dataset.types import RoundState
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import ImageLoadConfig, load_round, save_volume

pytestmark = [pytest.mark.dataset, pytest.mark.contract]


def dataset(tmp_path, xy=0.5, z=1.5):
    return Dataset(input_root=tmp_path / "input", output_root=tmp_path / "output", dataset_id="test",
                   sample_id="sample", output_id="run", channel_order=("ch00",),
                   rounds=RoundState(sequencing_rounds=["round1"], reference_round="round1"),
                   voxel_size_xy=xy, voxel_size_z=z)


def write_round(ds, metadata=None):
    folder = ds.fov("FOV_001").input_dir("round1")
    folder.mkdir(parents=True)
    image = np.arange(2 * 6 * 8, dtype=np.uint16).reshape(2, 6, 8)
    save_volume(image, folder / "ch00.tif", metadata=metadata)
    return folder


@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("xy,z", [(0.5, 1.5), (None, 1.5), (0.5, None), (None, None)])
def test_raw_calibration_only_fills_unknown_spacing(tmp_path, stored, xy, z):
    ds = dataset(tmp_path, xy, z)
    # Non-default geometry verifies that filling spacing changes no other field.
    metadata = ImageMetadata("source", origin_zyx=(2.0, 3.0, 4.0),
                             direction_zyx=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))) if stored else None
    folder = write_round(ds, metadata)
    loaded = load_round(folder, config=ImageLoadConfig(channel_labels=("ch00",)))
    fov = ds.fov("FOV_001").load_images()
    expected = loaded.metadata
    if xy is not None and z is not None:
        expected = replace(expected, spacing_zyx=(z, xy, xy), spatial_unit="micrometer")
    assert fov.metadata["round1"] == expected
    np.testing.assert_array_equal(fov.images["round1"], loaded.image)


def test_stored_calibration_wins_over_different_declared_sizes(tmp_path):
    ds = dataset(tmp_path)
    known = ImageMetadata("known", spacing_zyx=(2., 3., 4.), spatial_unit="um", origin_zyx=(1., 2., 3.))
    write_round(ds, known)
    assert ds.fov("FOV_001").load_images().metadata["round1"] == known


def test_rotation_keeps_declared_spacing(tmp_path):
    ds = dataset(tmp_path)
    write_round(ds)
    fov = ds.fov("FOV_001").load_images(rotation_degrees=-90)
    assert fov.metadata["round1"].spacing_zyx == pytest.approx((1.5, 0.5, 0.5), rel=0, abs=1e-9)
    assert fov.metadata["round1"].spatial_unit == "micrometer"
    assert fov.images["round1"].shape == (2, 8, 6, 1)


@pytest.mark.parametrize("xy,z", [(None, 1.5), (0.5, None), (None, None)])
@pytest.mark.parametrize("explicit_spacing", [False, True])
def test_stitch_refuses_unknown_spacing_naming_both_voxel_keys(tmp_path, xy, z, explicit_spacing):
    from starfinder.dataset._stitching import read_inputs, table_from_positions
    from starfinder.dataset._declared import reference_settings
    from starfinder.stitching._placement import check_grids
    ds = dataset(tmp_path, xy, z)
    write_round(ds)
    ds.fov("FOV_001").load_images().save_reference_image()
    with pytest.raises(IncompatibleGeometryError, match="voxel_size_xy.*voxel_size_z"):
        inputs = read_inputs(ds.output_root, ["FOV_001"], "reference_merged", reference=reference_settings(ds))
        grids = {name: item.grid for name, item in inputs.items()}
        positions = {"source": "regular_grid", "regular_grid": {
            "columns": 1, "rows": 1, "order": "column_major_down_right", "overlap": 0}}
        if explicit_spacing:
            positions["spacing_yx"] = (0.5, 0.5)
        table_from_positions(positions, ["FOV_001"], grids, fov_pattern=ds.fov_pattern)
        check_grids(grids)


@pytest.mark.parametrize("xy,z", [(None, 1.5), (0.5, None), (None, None)])
def test_dataset_watershed_refusal_names_both_voxel_keys(tmp_path, xy, z):
    from starfinder.segmentation import (InputChannel, LabelImportConfig, SeededWatershedConfig,
                                        SegmentationPlan, SegmentationRun)
    ds = dataset(tmp_path, xy, z)
    write_round(ds)
    labels = tmp_path / "nuclei.tif"
    save_volume(np.ones((2, 6, 8), np.uint32), labels)
    plan = SegmentationPlan((
        SegmentationRun("nucleus", "nucleus", (), LabelImportConfig(labels, target="nucleus")),
        SegmentationRun("cell", "cell", (InputChannel("composite", reference_merged=True),),
                        SeededWatershedConfig(), seeds="nucleus")))
    with pytest.raises(ValueError, match="voxel_size_xy.*voxel_size_z"):
        ds.fov("FOV_001").load_images().segment(plan)
