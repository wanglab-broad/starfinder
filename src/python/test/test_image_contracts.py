from starfinder.preprocessing import MinMaxNormalizationConfig
"""Bounded image geometry, numeric-policy and persistence regressions."""
from dataclasses import asdict
import importlib

import numpy as np
import pytest
import tifffile

from starfinder.image import ImageMetadata
from starfinder.io import ImageConversionConfig, ImageLoadConfig, convert_image, load_round, load_volume, save_volume
from starfinder.preprocessing import (HistogramMatchingConfig, MinMaxNormalizationConfig, ProjectionConfig, ReconstructionConfig, TophatConfig, filter_tophat, match_histogram, normalize_intensity, project_image, reconstruct_background)


def geometry():
    return ImageMetadata("source", (2, 3, 3), (10, 20, 30), ((0, 0, 1), (0, 1, 0), (-1, 0, 0)), "um")


@pytest.mark.parametrize("spacing", [(0, 1, 1), (-1, 1, 1), (1, np.nan, 1), (1, np.inf, 1), (1, 1)])
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        ImageMetadata("frame", spacing_zyx=spacing)


def test_geometry_conversion_and_unknown():
    meta = geometry()
    indices = np.array([[0, 0, 0], [1.5, 2, 3]])
    np.testing.assert_allclose(meta.index_to_world(indices), [[10, 20, 30], [19, 26, 27]])
    np.testing.assert_allclose(meta.world_to_index(meta.index_to_world(indices)), indices)
    for key in ("spacing_zyx", "origin_zyx", "direction_zyx", "spatial_unit"):
        fields = asdict(meta)
        fields[key] = None
        with pytest.raises(ValueError, match="complete geometry"):
            ImageMetadata(**fields).index_to_world(indices)
    assert ImageMetadata("unknown").spacing_zyx is None
    with pytest.raises(ValueError, match="orthonormal"):
        ImageMetadata("bad", direction_zyx=np.ones((3, 3)))


@pytest.mark.parametrize("angle", [90, -90, 180, 270, 360])
def test_rot90_geometry_matches_array(angle):
    meta = ImageMetadata("frame", (2, 3, 4), (10, 20, 30), np.eye(3), "um")
    image = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    rotated = np.rot90(image, k=angle // 90, axes=(1, 2))
    output = meta.rotated(image.shape, angle, frame_id="rotated")
    for index in np.ndindex(rotated.shape):
        source_index = np.argwhere(image == rotated[index])[0]
        np.testing.assert_allclose(output.index_to_world(index), meta.index_to_world(source_index), atol=1e-12)


def test_arbitrary_rotation_crop_and_projection_geometry():
    meta = geometry()
    crop = meta.cropped((1, 2, 3), frame_id="crop")
    np.testing.assert_allclose(crop.index_to_world((0, 0, 0)), meta.index_to_world((1, 2, 3)))
    rotated = meta.rotated((3, 5, 7), 30, frame_id="rotated")
    np.testing.assert_allclose(rotated.index_to_world((1, 2, 3)), meta.index_to_world((1, 2, 3)))
    assert ImageMetadata("unknown").cropped((1, 2, 3), frame_id="crop").origin_zyx is None
    projected = meta.projected(method="sum")
    assert projected.frame_id == "source/projection:sum"
    assert projected.spacing_zyx is None
    with pytest.raises(ValueError, match="complete geometry"):
        projected.index_to_world((0, 2, 3))
    with pytest.raises(ValueError, match="shear"):
        ImageMetadata("frame", (1, 2, 3)).rotated((2, 3, 4), 30, frame_id="rotated")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_io_roundtrip_geometry_dtype_and_immutable(tmp_path, dtype):
    image = np.arange(24).reshape(2, 3, 4).astype(dtype)
    before = image.copy()
    save_volume(image, tmp_path / "ch00.tif", metadata=geometry())
    loaded = load_volume(tmp_path / "ch00.tif")
    np.testing.assert_array_equal(loaded.image, image)
    assert loaded.image.dtype == dtype and loaded.metadata == geometry()
    round_result = load_round(tmp_path, config=ImageLoadConfig(channel_labels=("ch00",)))
    assert round_result.image.shape == (2, 3, 4, 1)
    np.testing.assert_array_equal(image, before)


def test_io_explicit_channels_time_ambiguity_and_crop(tmp_path):
    path = tmp_path / "stack.tif"
    stack = np.arange(48, dtype=np.uint16).reshape(2, 3, 4, 2)
    save_volume(stack, path)
    with pytest.raises(ValueError, match="ambiguous C"):
        load_volume(path)
    np.testing.assert_array_equal(load_volume(path, config=ImageLoadConfig(channel_index=1)).image, stack[..., 1])
    save_volume(stack[..., 0], tmp_path / "ch00.tif")
    save_volume(stack[:1, ..., 1], tmp_path / "ch01.tif")
    config = ImageLoadConfig(channel_labels=("ch00", "ch01"))
    with pytest.raises(ValueError, match="size mismatch"):
        load_round(tmp_path, config=config)
    with pytest.warns(UserWarning):
        result = load_round(tmp_path, config=ImageLoadConfig(channel_labels=("ch00", "ch01"), crop_policy="minimum"))
    assert result.image.shape == (1, 3, 4, 2)
    assert result.diagnostics["crop_start_zyx"] == (0, 0, 0)
    save_volume(stack[..., 0], tmp_path / "other_ch00.tif")
    with pytest.raises(ValueError, match="found 2"):
        load_round(tmp_path, config=config)
    time_path = tmp_path / "time.ome.tif"
    tifffile.imwrite(time_path, np.zeros((2, 2, 3, 4), np.uint8), ome=True, photometric="minisblack", metadata={"axes": "TZYX"})
    with pytest.raises(ValueError, match="ambiguous T"):
        load_volume(time_path)
    assert load_volume(time_path, config=ImageLoadConfig(time_index=1)).image.shape == (2, 3, 4)


def test_conversion_range_scope_rounding_constants():
    image = np.array([0, 100, 10, 200], dtype=np.uint16).reshape(1, 1, 2, 2)
    before = image.copy()
    global_ = convert_image(image, config=ImageConversionConfig("uint8", "rescale", output_range=(0, 255), range_policy="data", rounding="truncate"))
    per_channel = convert_image(image, config=ImageConversionConfig("uint8", "rescale", output_range=(0, 255), range_policy="data", scope="per_channel"))
    np.testing.assert_array_equal(global_.ravel(), [0, 127, 12, 255])
    np.testing.assert_array_equal(per_channel.ravel(), [0, 0, 255, 255])
    np.testing.assert_array_equal(image, before)
    high = np.full((1, 2, 2), 60000, np.uint16)
    with pytest.raises(ValueError, match="representable"):
        convert_image(high, config=ImageConversionConfig("uint8", "cast"))
    assert convert_image(high, config=ImageConversionConfig("uint8", "clip", output_range=(0, 255))).max() == 255
    assert convert_image(high, config=ImageConversionConfig("uint8", "rescale", range_policy="data", output_range=(5, 255))).max() == 5
    with pytest.raises(ValueError, match="declared input_range"):
        convert_image(high, config=ImageConversionConfig("uint8", "rescale", input_range=(0, 255), output_range=(0, 255)))
    values = np.array([0.5, 1.5, 2.5]).reshape(1, 1, 3)
    np.testing.assert_array_equal(convert_image(values, config=ImageConversionConfig("uint8", "cast")).ravel(), [0, 2, 2])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_processing_source_dtype_and_constants(dtype):
    image = np.full((2, 7, 7, 2), 10, dtype=dtype)
    original = image.copy()
    for operation in (filter_tophat, reconstruct_background):
        result = operation(image)
        assert result.dtype == dtype
        assert not result.any()
    normalized = normalize_intensity(image, config=MinMaxNormalizationConfig("float32", (-1, 1)))
    np.testing.assert_array_equal(normalized, -np.ones(image.shape))
    matched = match_histogram(image, image[..., 0])
    assert matched.dtype == dtype
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("lower", [0, 7])
@pytest.mark.parametrize("scope", ["global", "per_channel"])
def test_constant_lower_endpoint_precedes_snr_gate(dtype, lower, scope):
    shape = (1, 2, 3) if scope == "global" else (1, 2, 3, 2)
    image = np.full(shape, 50, dtype=dtype)
    if scope == "per_channel":
        image[..., 1] = 100
    before = image.copy()
    result = normalize_intensity(image, config=MinMaxNormalizationConfig(
        output_dtype=np.dtype(dtype).name, output_range=(lower, 255),
        scope=scope, snr_threshold=5.0,
    ))
    np.testing.assert_array_equal(result, np.full(shape, lower, dtype=dtype))
    assert result.dtype == dtype
    np.testing.assert_array_equal(image, before)


def test_uint16_overflow_and_no_forced_uint8():
    image = np.full((1, 9, 9), 40000, dtype=np.uint16)
    image[0, 4, 4] = 60000
    top = filter_tophat(image, config=TophatConfig(radius_yx=2))
    reconstructed = reconstruct_background(image, config=ReconstructionConfig(radius_yx=2))
    assert top.dtype == reconstructed.dtype == np.uint16
    assert top[0, 4, 4] == 20000
    assert reconstructed[0, 4, 4] == 40000
    assert reconstructed[0, 0, 0] == 0
    assert image[0, 0, 0] == 40000


def test_projection_signed_float_and_overflow():
    image = np.array([-1.25, 2.5]).reshape(2, 1, 1)
    assert project_image(image, config=ProjectionConfig("sum"))[0, 0, 0] == 1.25
    image = np.full((2, 3, 4, 2), 60000, np.uint16)
    summed = project_image(image, config=ProjectionConfig("sum"))
    assert summed.dtype == np.uint64 and summed.shape == (1, 3, 4, 2)
    assert summed.max() == 120000
    with pytest.raises(ValueError, match="representable"):
        project_image(image, config=ProjectionConfig("sum", output_dtype="uint16"))
    clipped = project_image(image, config=ProjectionConfig("sum", conversion=ImageConversionConfig("uint16", "clip", output_range=(0, 65535))))
    assert clipped.max() == 65535


@pytest.mark.parametrize("image", [np.zeros((2, 2)), np.zeros((0, 2, 2)), np.full((1, 2, 2), np.nan), np.full((1, 2, 2), np.inf), np.ones((1, 2, 2), complex)])
def test_processing_invalid_images(image):
    operations = [filter_tophat, reconstruct_background, project_image,
        lambda x: normalize_intensity(x, config=MinMaxNormalizationConfig("uint8", (0, 255))),
        lambda x: match_histogram(x, np.zeros((1, 2, 2)))]
    for op in operations:
        with pytest.raises(ValueError):
            op(image)


def test_fov_subtile_rotation_and_projection_metadata(tmp_path):
    from starfinder.dataset import Dataset, RoundState, SubtileConfig, FOV
    dataset = Dataset(input_root=tmp_path, output_root=tmp_path, dataset_id="test", sample_id="small", output_id="output", rounds=RoundState(sequencing_rounds=["round1"], reference_round="round1"), channel_order=("ch00",))
    dataset.subtile = SubtileConfig(2, overlap_ratio=0)
    dataset.subtile.compute_windows(8, 8)
    save_volume(np.ones((2, 8, 8), np.uint16), tmp_path / "round1/fov/ch00.tif", metadata=geometry())
    loaded_fov = FOV(dataset, "fov").load_images(channel_order=["ch00"])
    assert loaded_fov.metadata["round1"] == geometry()
    assert loaded_fov.images["round1"].dtype == np.uint16
    fov = FOV(dataset, "fov", images={"round1": np.arange(128, dtype=np.uint16).reshape(2, 8, 8, 1)}, metadata={"round1": geometry()})
    fov.rotate(angle=90)
    rotated_meta = fov.metadata["round1"]
    fov.create_subtiles(out_dir=tmp_path / "tiles")
    child = FOV.from_subtile(tmp_path / "tiles/subtile_data_4.npz", dataset, "fov")
    assert child.images["round1"].shape == (2, 4, 4, 1)
    np.testing.assert_allclose(child.metadata["round1"].index_to_world((0, 0, 0)), rotated_meta.index_to_world((0, 4, 4)))
    child.project_image()
    assert child.images["round1"].shape == (1, 4, 4, 1)
    assert child.metadata["round1"].spacing_zyx is None
    assert "projection_source_metadata" in child.load_diagnostics["round1"]


def test_removed_interfaces_are_absent():
    import starfinder
    import starfinder.io as io
    import starfinder.preprocessing as preprocessing
    for module, names in [(io, ("load_multipage_tiff", "load_image_stacks", "save_stack")), (preprocessing, ("min_max_normalize", "histogram_match", "morphological_reconstruction", "tophat_filter"))]:
        for name in names:
            assert not hasattr(module, name)
            assert not hasattr(starfinder, name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("starfinder.utils")


def test_conversion_extrema_and_effective_range_record(tmp_path):
    maximum = np.full((1, 1, 1), np.iinfo(np.uint64).max, np.uint64)
    np.testing.assert_array_equal(convert_image(maximum, config=ImageConversionConfig("uint64", "cast")), maximum)
    with pytest.raises(ValueError, match="representable"):
        convert_image(np.full((1, 1, 1), 2.0 ** 64), config=ImageConversionConfig("uint64", "cast"))
    image = np.array([1000, 2000], np.uint16).reshape(1, 1, 2)
    save_volume(image, tmp_path / "ch00.tif", metadata=ImageMetadata("shared"))
    loaded = load_round(tmp_path, config=ImageLoadConfig(channel_labels=("ch00",), conversion=ImageConversionConfig("uint8", "rescale", range_policy="data", output_range=(0, 255))))
    assert loaded.metadata.frame_id == "shared"
    assert loaded.diagnostics["conversion"]["effective_input_ranges"] == [(1000.0, 2000.0)]
    np.testing.assert_array_equal(loaded.image.ravel(), [0, 255])


def test_invalid_config_and_io_boundaries(tmp_path):
    for factory in (lambda: TophatConfig(-1), lambda: ReconstructionConfig(1.5),
                    lambda: MinMaxNormalizationConfig("uint8", (255, 0)),
                    lambda: HistogramMatchingConfig(rounding="unknown"),
                    lambda: ImageLoadConfig(channel_labels=("ch00", "ch00")),
                    lambda: ImageConversionConfig("uint8", "rescale")):
        with pytest.raises(ValueError):
            factory()
    with pytest.raises(TypeError):
        HistogramMatchingConfig(nbins=64)
    with pytest.raises(ValueError, match="finite"):
        save_volume(np.full((1, 2, 2), np.nan), tmp_path / "bad.tif")
    with pytest.raises(ValueError, match="representable"):
        match_histogram(np.zeros((1, 2, 2), np.uint8), np.full((1, 2, 2), 500, np.uint16))
