"""The amplicon-scene paths of the cross-process repeatability test (``test_formed_streams.py``).

Each path builds one scene or dataset and returns a JSON-ready document of what it generated. The repeatability
test runs ``main`` in two interpreters with different ``PYTHONHASHSEED`` values and compares the documents path by
path (W-576, TM-26). The paths are those of the six former cross-process tests:

* ``every-effect``: a scene drawing from every registered component (``every_effect``);
* ``scene/<name>``: the presets ``formed-small-v1`` and ``formed-z1-v1`` and the four development fixtures;
* ``singleton-random-geometry``: a Z=1 scene with random translation and local geometry;
* ``background-noise``: background texture with tissue weights and both noise components;
* ``readout-trend-mixing``: dropout, weakening, loss, trend and mixing;
* ``dataset-tiny``: ``generate_amplicon_dataset`` on the bounded ``tiny`` benchmark preset, two FOVs.

``bounded`` and ``digest`` are shared with ``test_synthetic_contracts.py``.
"""
from dataclasses import replace
import hashlib
import json

import numpy as np

from starfinder.synthetic import (DEVELOPMENT_FIXTURES, BackgroundConfig, GeometryConfig, NoiseConfig,
                                  ReadoutEffectsConfig, ScalarDistribution, TextureConfig, amplicon_scene_preset,
                                  benchmark_scene_preset, development_scene_preset, generate_amplicon_dataset,
                                  generate_amplicon_scene)

SCENE_PRESETS = ('formed-small-v1', 'formed-z1-v1')


def every_effect(**changes):
    """A scene drawing from every registered component."""
    book, config = amplicon_scene_preset()
    config = replace(config, shape_zyx=(3, 9, 11), density=.02, dtype='float64',
        brightness=ScalarDistribution('lognormal', (2, .2)),
        axial_width=ScalarDistribution('uniform', (.8, 1.2)),
        lateral_width=ScalarDistribution('uniform', (.8, 1.2)),
        elongation=ScalarDistribution('folded_lognormal', (0, .2)),
        angle=ScalarDistribution('uniform', (0, np.pi)),
        readout=ReadoutEffectsConfig(dropout_enabled=True, dropout_probability=(.3, .3, .3),
            weakening_enabled=True, weakening_probability=(.3, .3, .3), weak_factor=(.5, .5, .5),
            loss_enabled=True, loss_probability=.3),
        background=BackgroundConfig(texture_enabled=True, tissue_weights=np.ones((3, 4)),
            texture=TextureConfig(density=.01, axial_width=ScalarDistribution('uniform', (1, 2)),
                                  brightness=ScalarDistribution('uniform', (2, 4)))),
        noise=NoiseConfig(True, .5, True, .5),
        geometry=GeometryConfig(translation_enabled=True, translation_max_zyx=(.2, .5, .5),
                                local_enabled=True, centers_zyx=((1, 4, 5),), strength=.05,
                                affine_enabled=True, affine_max_zyx=(.01, .05, .05),
                                polynomial_enabled=True, polynomial_max_zyx=(.01, .05, .05)))
    return generate_amplicon_scene(book, config=replace(config, **changes))


def bounded(name='tiny', *, preset_dtype='uint16', **changes):
    """A benchmark preset reduced to a bounded grid; every appearance default kept."""
    book, config = benchmark_scene_preset(name, dtype=preset_dtype)
    return book, replace(config, shape_zyx=(4, 48, 48), count=6, **changes)


def digest(result):
    h = hashlib.sha256()
    for fov in sorted(result.rounds):
        for label, image in result.rounds[fov].items():
            h.update(label.encode())
            h.update(image.tobytes())
    for table in (result.formed, result.round_truth):
        h.update(table.to_json(orient='table').encode())
    return h.hexdigest()


def scene_digest(s, *, arrays=True, tables=True):
    """SHA-256 over the round images, optionally the signal arrays and tables, and the provenance."""
    h = hashlib.sha256()
    for a in [*s.rounds.values(), *((s.intended, s.pre_mix, s.realized) if arrays else ())]:
        h.update(a.tobytes())
    for t in (s.formed, s.round_truth) if tables else ():
        h.update(t.to_json(orient='table').encode())
    h.update(json.dumps(s.provenance, sort_keys=True).encode())
    return h.hexdigest()


def preset_document(book, config):
    s = generate_amplicon_scene(book, config=config)
    return dict(provenance=s.provenance, formed=s.formed.to_json(), rounds=s.round_truth.to_json(),
                images={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in s.rounds.items()},
                signals=hashlib.sha256(s.realized.tobytes()).hexdigest())


def singleton_random_geometry():
    cb, cfg = amplicon_scene_preset('formed-z1-v1')
    s = generate_amplicon_scene(cb, config=replace(cfg, shape_zyx=(1, 7, 9), geometry=GeometryConfig(
        translation_enabled=True, translation_max_zyx=(0, .2, .3), local_enabled=True,
        centers_zyx=((0, 2, 3),), strength=.03)))
    vectors = {k: t['vectors_zyx'][0][0] for k, t in s.provenance['transforms'].items()}
    assert all(z == 0 for z in vectors.values()), f'singleton-random-geometry: a Z vector is not 0: {vectors}'
    return dict(provenance=s.provenance, truth=s.round_truth.to_json(),
                images={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in s.rounds.items()})


def background_noise():
    b, c = amplicon_scene_preset()
    s = generate_amplicon_scene(b, config=replace(c, shape_zyx=(1, 7, 9),
        background=BackgroundConfig(texture_enabled=True, tissue_weights=[[1] * 4] * 3),
        noise=NoiseConfig(True, 4, True, 2)))
    return scene_digest(s, arrays=False, tables=False)


def readout_trend_mixing():
    b, c = amplicon_scene_preset()
    e = ReadoutEffectsConfig(dropout_enabled=True, dropout_probability=(.2, .5, .8),
        weakening_enabled=True, weakening_probability=(.3, .7, .4), weak_factor=(.2, .4, .6),
        loss_enabled=True, loss_probability=.5, trend_enabled=True, trend_base=.9,
        mixing_enabled=True, mixing=[[[1, .1, 0, 0], [.2, 1, 0, 0], [0, 0, 1, .3], [0, 0, 0, 1]]] * 3)
    return scene_digest(generate_amplicon_scene(b, config=replace(c, readout=e)))


def dataset_tiny():
    book, config = bounded()
    return digest(generate_amplicon_dataset(book, config, fov_ids=('FOV_001', 'FOV_002')))


def paths():
    """Every path, in order, as a mapping of its name to a function returning its document."""
    builds = {'every-effect': lambda: scene_digest(every_effect())}
    for name in SCENE_PRESETS:
        builds[f'scene/{name}'] = lambda name=name: preset_document(*amplicon_scene_preset(name))
    for name, (condition, size) in DEVELOPMENT_FIXTURES.items():
        builds[f'scene/{name}'] = lambda c=condition, s=size: preset_document(*development_scene_preset(c, size=s))
    builds.update({'singleton-random-geometry': singleton_random_geometry, 'background-noise': background_noise,
                   'readout-trend-mixing': readout_trend_mixing, 'dataset-tiny': dataset_tiny})
    return builds


def main():
    """Print one JSON line ``[path, document]`` per path, then ``["probe", hash('probe')]``."""
    for path, build in paths().items():
        try:
            document = build()
        except Exception as error:
            raise RuntimeError(f'path {path} failed in the child process') from error
        print(json.dumps([path, document], sort_keys=True), flush=True)
    print(json.dumps(['probe', hash('probe')]))
