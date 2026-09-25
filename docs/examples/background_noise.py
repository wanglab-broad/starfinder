"""Analytic structured-background literals and residual-noise isolation."""
from dataclasses import replace

import numpy as np

from starfinder.synthetic import (BackgroundConfig, NoiseConfig, TextureConfig,
                                  formed_scene_preset, generate_formed_scene)


def check_background(depth=3):
    book, base = formed_scene_preset()
    config = replace(base, dataset_version='structured-background-development-v1',
                     shape_zyx=(depth, 7, 9), count=0, dtype='float64')
    # Independent literals: normalized X gradient [1, 2, ..., 9].
    gradient = BackgroundConfig(gradient_enabled=True, gradient_intercept=1,
        gradient_slopes_zyx=(0, 0, 8), tissue_weights=np.ones((3, 4)))
    scene = generate_formed_scene(book, config=replace(config, background=gradient))
    for image in scene.rounds.values():
        np.testing.assert_array_equal(image[0, 0, :, 0], np.arange(1., 10.))
    # A broad Gaussian, untruncated even beyond four sigma.
    region = BackgroundConfig(regions_enabled=True, region_centers=((0, 2, 2),),
        region_sigma_zyx=((1, 1, 1),), region_heights=(8,), tissue_weights=np.ones((3, 4)))
    scene = generate_formed_scene(book, config=replace(config, background=region))
    image = next(iter(scene.rounds.values()))
    np.testing.assert_allclose(image[0, 2, 2:4, 0], [8, 4.852245277701067], rtol=0, atol=1e-12)
    np.testing.assert_allclose(image[0, 2, 7, 0], 0.00002981322537662937, rtol=0, atol=1e-16)
    # Noise toggle cannot redraw the latent blobs or the formed population.
    background = BackgroundConfig(texture_enabled=True, texture=TextureConfig(count=2),
                                  tissue_weights=np.ones((3, 4)))
    clean = generate_formed_scene(book, config=replace(config, background=background))
    noisy = generate_formed_scene(book, config=replace(config, background=background,
        noise=NoiseConfig(dependent_enabled=True, alpha=4, independent_enabled=True, sigma=2)))
    components = lambda s: s.provenance['effective_config']['background']['components']
    assert components(clean) == components(noisy) and len(components(clean)) == 2
    assert clean.provenance['transforms'] == noisy.provenance['transforms']
    # Zero strengths with both residuals enabled reproduce the pre-noise image.
    zero = generate_formed_scene(book, config=replace(config, background=background,
        noise=NoiseConfig(dependent_enabled=True, alpha=0, independent_enabled=True, sigma=0)))
    for label, image in clean.rounds.items():
        np.testing.assert_array_equal(zero.rounds[label], image)
        assert not np.array_equal(noisy.rounds[label], image)
    return noisy


if __name__ == '__main__':
    for depth in (3, 1):
        check_background(depth)
    print('Structured background literals and noise isolation passed for 3D and Z=1.')
