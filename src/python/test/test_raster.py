"""Independent literal geometry/count oracles for sample-export-contract-v1."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import ImageLoadResult
from starfinder.raster import RasterConfig, RasterPreparationError, prepare_rasters, remap_labels

SPEC = json.loads((Path(__file__).resolve().parents[3] / 'docs/examples/sample_export_cases.json').read_text())
CASES = {c['id']: c for c in SPEC['cases']}
META = ImageMetadata('literal-sample', (2, 1, 1), (10, 20, 30), tuple(map(tuple, np.eye(3))), 'micrometer')


def inputs(case='E1', dtype=np.float32, metadata=META):
    labels = np.array(CASES[case]['labels'], dtype=np.uint32)
    z, y, x = np.indices(labels.shape)
    a = np.stack([100*z + 10*y + x, 100*z + 10*y + x + 1000], axis=-1).astype(dtype)
    loaded = ImageLoadResult(a, metadata, ('reference', 'stain'), (Path('literal.npy'),),
                             {'transform_history': [{'source_to_target': 'already applied upstream'}]})
    return {'selected': loaded}, labels


def prepare(images, labels, factors=(1, 2, 2), **kwargs):
    return prepare_rasters(images, labels, metadata=images['selected'].metadata,
        declared_ids=[int(v) for v in np.unique(labels) if v], config=RasterConfig(factors, **kwargs))


@pytest.mark.parametrize('case', ['E1', 'E2', 'E3', 'E4', 'E5', 'E9'])
def test_literal_reduction_fallback_and_immutability(case):
    c = CASES[case]
    images, labels = inputs(case)
    image_before, mask_before = images['selected'].image.copy(), labels.copy()
    result = prepare(images, labels, tuple(c['requested']))
    level = result.levels[0]
    assert level.factors_zyx == tuple(c['achieved'])
    np.testing.assert_array_equal(level.labels, c['expected_labels'])
    if 'expected_channel_zero' in c:
        np.testing.assert_array_equal(level.images['selected'].image[..., 0], c['expected_channel_zero'])
        np.testing.assert_array_equal(level.images['selected'].image[..., 1], np.array(c['expected_channel_zero']) + 1000)
        np.testing.assert_array_equal(level.metadata.origin_zyx, c['expected_origin'])
    if 'trial_missing' in c:
        assert result.diagnostics['trials'][0]['missing_labels'] == c['trial_missing']
        assert result.diagnostics['fallback']
        np.testing.assert_array_equal(level.images['selected'].image, image_before)
    np.testing.assert_array_equal(images['selected'].image, image_before)
    np.testing.assert_array_equal(labels, mask_before)
    assert level.images['selected'].source_paths == images['selected'].source_paths
    assert level.images['selected'].channel_labels == ('reference', 'stain')
    assert level.images['selected'].diagnostics == images['selected'].diagnostics
    level.images['selected'].diagnostics['transform_history'].clear()
    assert images['selected'].diagnostics['transform_history']
    assert not np.shares_memory(level.labels, labels)
    assert not np.shares_memory(level.images['selected'].image, images['selected'].image)


def test_collision_and_background_ties():
    images, labels = inputs('E3')
    trial = prepare(images, labels).diagnostics['trials'][0]
    assert trial['mixed_positive_blocks'] == trial['tied_blocks'] == 1
    assert trial['cells'][0]['output_voxels'] == 1  # smallest positive wins
    labels[:] = [[[0, 1], [0, 1]]]
    trial = prepare(images, labels).diagnostics['trials'][0]
    assert trial['tied_blocks'] == 1
    assert trial['missing_labels'] == [1]  # background wins a tie


def test_geometry_forward_inverse_extent_and_fidelity():
    images, labels = inputs('E4')
    level = prepare(images, labels, (2, 2, 2)).levels[0]
    expected = np.array([[2, 0, 0, .5], [0, 2, 0, .5], [0, 0, 2, .5], [0, 0, 0, 1.]])
    np.testing.assert_array_equal(level.index_to_source_zyx, expected)
    np.testing.assert_allclose(np.linalg.inv(expected) @ [0.5, 0.5, 2.5, 1], [0, 0, 1, 1], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(level.metadata.index_to_world([0, 0, 1]), [11, 20.5, 32.5])
    np.testing.assert_array_equal(level.metadata.index_to_world([-.5, -.5, -.5]), [9, 19.5, 29.5])
    np.testing.assert_array_equal(level.metadata.index_to_world([.5, .5, 1.5]), [13, 21.5, 33.5])
    for cell in level.diagnostics['cells']:
        assert cell['native_voxels'] == 8 and cell['output_voxels'] == 1
        assert cell['centroid_displacement'] == 0
        assert cell['hausdorff_distance'] == pytest.approx(np.sqrt(3)/2)
        assert cell['min_support_fraction'] == 1


def test_fidelity_failure_even_when_ids_survive():
    labels = np.array([[[1,1,1,0,1,0,1,0], [1,1,0,0,0,0,0,0]]], dtype=np.uint32)
    source = ImageLoadResult(np.zeros((*labels.shape, 1), dtype=np.float32), META, ('stain',), ())
    result = prepare({'selected': source}, labels)
    trial = result.diagnostics['trials'][0]
    assert trial['missing_labels'] == []
    # Native x centroid = (0+1+2+4+6+0+1)/7 = 2; y shift = 3/14.
    assert trial['cells'][0]['centroid_displacement'] == pytest.approx(np.hypot(1.5, 3/14))
    assert trial['cells'][0]['hausdorff_distance'] == pytest.approx(np.hypot(5.5, .5))
    assert trial['failed_checks'] == ['centroid', 'hausdorff']
    assert result.levels[0].factors_zyx == (1, 1, 1)


def test_pyramid_direct_native_means_and_stop():
    images, labels = inputs()
    result = prepare(images, labels, (1, 1, 1), coarser_levels=4)
    assert [l.factors_zyx for l in result.levels] == [(1,1,1), (1,2,2)]
    assert result.diagnostics['pyramid_stop_reason'] == 'coarser_level_not_preservable'
    assert result.diagnostics['trials'][-1]['failed_checks'] == ['indivisible_grid']
    images, labels = inputs('E3')
    result = prepare(images, labels, (1,1,1), coarser_levels=2)
    assert len(result.levels) == 1
    assert result.diagnostics['trials'][-1]['missing_labels'] == [2]


@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float32, np.float64, np.dtype('>f8')])
def test_dtype_native_and_independent_mean_oracle(dtype):
    images, labels = inputs(dtype=dtype)
    native = prepare(images, labels, (1,1,1)).levels[0]
    np.testing.assert_array_equal(native.images['selected'].image, images['selected'].image)
    assert native.images['selected'].image.dtype == dtype
    reduced = prepare(images, labels).levels[0].images['selected'].image
    expected = np.empty((1,1,2,2), dtype=np.float64)
    a = images['selected'].image
    for x in range(2):
        for c in range(2):
            expected[0,0,x,c] = sum(float(a[0,y,xx,c]) for y in range(2) for xx in range(2*x,2*x+2))/4
    is_float64 = np.dtype(dtype).kind == 'f' and np.dtype(dtype).itemsize == 8
    assert reduced.dtype == (np.float64 if is_float64 else np.float32)
    np.testing.assert_allclose(reduced, expected, rtol=1e-12 if is_float64 else 1e-6, atol=1e-12 if is_float64 else 1e-6)


def test_multiple_images_same_achieved_grid_and_channel_order():
    images, labels = inputs('E2')
    images['reference'] = replace(images['selected'], image=images['selected'].image[..., [1,0]], channel_labels=('stain','reference'))
    level = prepare(images, labels).levels[0]
    assert level.images['selected'].metadata == level.images['reference'].metadata == level.metadata
    assert level.images['reference'].channel_labels == ('stain','reference')
    np.testing.assert_array_equal(level.images['reference'].image, images['selected'].image[..., [1,0]])


def test_unknown_calibration_and_empty_labels():
    images, labels = inputs('E9', metadata=ImageMetadata('explicit-common-index'))
    result = prepare(images, labels, coordinate_space='index')
    assert result.levels[0].metadata == ImageMetadata('explicit-common-index')
    assert result.diagnostics['calibration'] == 'unknown'
    np.testing.assert_array_equal(result.levels[0].index_to_source_zyx[:3,3], [0,.5,.5])
    assert result.levels[0].diagnostics['cells'] == []


def test_namespace_mapping_and_rejections():
    a = np.array([[[1,1,0,0],[1,1,0,0]]], dtype=np.uint16)
    b = np.array([[[0,0,1,1],[0,0,1,1]]], dtype=np.uint16)
    keys = {('A',1): ('cells','A1'), ('B',1): ('cells','B1')}
    out, rows = remap_labels({'B':b, 'A':a}, cell_keys=keys)
    assert out.dtype == np.uint32
    np.testing.assert_array_equal(out, CASES['E1']['labels'])
    assert [(r['mask_namespace'],r['local_label'],r['instance_id']) for r in rows] == [('A',1,1),('B',1,2)]
    with pytest.raises(RasterPreparationError, match='overlapping'):
        remap_labels({'A':a,'B':a}, cell_keys=keys)
    with pytest.raises(RasterPreparationError, match='global cell'):
        remap_labels({'A':a,'B':b}, cell_keys={('A',1):('cells','same'),('B',1):('cells','same')})


def test_invalid_native_budget_and_grid_requests():
    images, labels = inputs()
    with pytest.raises(RasterPreparationError) as error:
        prepare_rasters(images, labels, metadata=META, declared_ids=[1,2,3], config=RasterConfig())
    assert error.value.diagnostics['missing_native_labels'] == [3]
    for invalid in [labels.astype(float), labels.astype(bool), -labels.astype(int), labels[0]]:
        with pytest.raises(RasterPreparationError):
            prepare(images, invalid)
    with pytest.raises(RasterPreparationError, match='singleton'):
        prepare(images, labels, (2,2,2))
    for f in [(1,3,2), (True,2,2), (1,0,2), (1,2)]:
        with pytest.raises(ValueError):
            RasterConfig(f)
    images, labels = inputs('E2')
    with pytest.raises(RasterPreparationError) as error:
        prepare(images, labels, max_output_bytes=30)
    assert error.value.diagnostics['required_level_bytes'] == 96
    assert error.value.diagnostics['required_shape_zyx'] == [1,2,4]
    result = prepare(images, labels, (1,8,4))
    assert [t['factors_zyx'] for t in result.diagnostics['trials']] == [[1,8,4],[1,4,2],[1,2,1],[1,1,1]]


def test_unsupported_frame_and_mismatch_fail_without_resampling():
    images, labels = inputs()
    with pytest.raises(IncompatibleGeometryError, match='index frame'):
        prepare(images, labels, coordinate_space='index')
    for meta in [replace(META, origin_zyx=(0,0,0)), replace(META, frame_id='other')]:
        bad = {'selected': replace(images['selected'], metadata=meta)}
        with pytest.raises(IncompatibleGeometryError, match='aligned'):
            prepare_rasters(bad, labels, metadata=META, declared_ids=[1,2], config=RasterConfig())
    rotated = replace(META, direction_zyx=((1,0,0),(0,0,-1),(0,1,0)))
    with pytest.raises(IncompatibleGeometryError, match='regridding'):
        prepare({'selected': replace(images['selected'], metadata=rotated)}, labels)
    partial = ImageMetadata('partial', spacing_zyx=(1,1,1))
    with pytest.raises(IncompatibleGeometryError):
        prepare({'selected': replace(images['selected'], metadata=partial)}, labels)
