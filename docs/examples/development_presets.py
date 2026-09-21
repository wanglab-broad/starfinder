"""Create bounded controlled fixtures, or inspect saved outputs without generation."""
import hashlib
import html
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from starfinder.dataset import RoundState
from starfinder.io import (ImageLayer, ImageLoadResult, ImageProcessingState,
                           load_image_checkpoint, save_image_checkpoint)
from starfinder.reporting import embed_figure, render_table
from saved_synthetic import checksum, read_json, source_identity, write_json


def create(directory):
    from starfinder.synthetic import (DEVELOPMENT_FACTORS, DEVELOPMENT_SIZES,
        development_preset_factors, development_scene_preset, generate_formed_scene)
    directory.mkdir(parents=True, exist_ok=False)
    code = source_identity()
    code['source_sha256']['docs/examples/development_presets.py'] = checksum(__file__)
    code["snapshot_sha256"] = hashlib.sha256(json.dumps(code["source_sha256"], sort_keys=True).encode()).hexdigest()
    cases = []
    for size in DEVELOPMENT_SIZES:
        conditions = ('clean', *DEVELOPMENT_FACTORS, 'combined')
        for condition in conditions:
            start = perf_counter()
            book, config = development_scene_preset(condition, size=size)
            scene = generate_formed_scene(book, config=config)
            name = f'{size}-{condition}'
            root = directory / name
            root.mkdir()
            scene.formed.to_parquet(root / 'formed.parquet', index=False)
            scene.round_truth.to_parquet(root / 'round-truth.parquet', index=False)
            np.savez(root / 'signals.npz', intended=scene.intended, pre_mix=scene.pre_mix, realized=scene.realized)
            write_json(root / 'synthetic.json', scene.provenance)
            source = dict(scene.provenance, uri=str(root / 'synthetic.json'), sha256=checksum(root / 'synthetic.json'),
                unverified_reason=None, catalog='docs/datasets.md#controlled-development-package-v1')
            extension = source['extensions']['starfinder.synthetic']
            checkpoint = save_image_checkpoint(root / 'prepared', tuple(
                ImageLayer(label, ('sequencing', 'registration_reference') if label == scene.round_labels[0] else ('sequencing',), ImageLoadResult(array, scene.round_metadata[label], scene.channel_labels, ()),
                           ImageProcessingState(scene.round_metadata[label]))
                for label, array in scene.rounds.items()),
                rounds=RoundState(list(scene.round_labels), reference_round=scene.round_labels[0]),
                stage='prepared_input', dataset_id=config.dataset_version, sample_id=config.sample_id,
                FOV=config.FOV_id, run_id=name, config=extension['requested_config'], code=code, sources=(source,))
            loaded = load_image_checkpoint(checkpoint)
            for layer in loaded.layers:
                assert layer.loaded.image.tobytes() == scene.rounds[layer.round_label].tobytes()
                assert layer.loaded.metadata == scene.round_metadata[layer.round_label]
            pd.testing.assert_frame_equal(pd.read_parquet(root / 'formed.parquet'), scene.formed)
            pd.testing.assert_frame_equal(pd.read_parquet(root / 'round-truth.parquet'), scene.round_truth)
            with np.load(root / 'signals.npz') as saved:
                for key in saved.files:
                    assert saved[key].tobytes() == getattr(scene, key).tobytes()
            assert read_json(root / 'synthetic.json') == scene.provenance
            cases.append(dict(name=name, factors=development_preset_factors(condition), shape=list(config.shape_zyx),
                config_sha256=extension['config_sha256'], seconds=perf_counter()-start,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())))
    files = {str(p.relative_to(directory)): checksum(p) for p in directory.rglob('*') if p.is_file()}
    write_json(directory / 'manifest.json', dict(version='controlled-development-v1', code=code, cases=cases, files=files,
        seed=42, scene_key='controlled-development-v1', axes='ZYXC; signals NCR',
        owner='Jiahao', retention='thesis/project handoff and associated publication', backup='unverified',
        limitations=['Development only; not calibrated evaluation data, cells or biological RNA truth.',
                     'No processing/detection/decoding is performed; truth is never a detector result.',
                     'RSS is the cumulative process high-water mark, not an isolated per-case peak.',
                     'Different sizes do not promise matching voxel noise or texture positions.']))


def inspect(directory):
    """Verify persisted identities and render channel views using only saved files."""
    import matplotlib.pyplot as plt
    manifest = read_json(directory / 'manifest.json')
    for path, digest in manifest['files'].items():
        assert checksum(directory / path) == digest, path
    sections = []
    for case in manifest['cases']:
        root = directory / case['name']
        source = read_json(root / 'synthetic.json')
        ext = source['extensions']['starfinder.synthetic']
        assert ext['config_sha256'] == case['config_sha256']
        loaded = load_image_checkpoint(root / 'prepared')
        assert loaded.artifact['payload']['config'] == ext['requested_config']
        for layer in loaded.layers:
            assert hashlib.sha256(layer.loaded.image.tobytes()).hexdigest() == ext['observation']['image_sha256'][layer.round_label]
            assert list(layer.loaded.channel_labels) == ext['channel_labels']
        formed = pd.read_parquet(root / 'formed.parquet')
        histories = pd.read_parquet(root / 'round-truth.parquet')
        assert formed.amplicon_id.tolist() == ext['amplicon_ids']
        with np.load(root / 'signals.npz') as signals:
            assert all(signals[key].shape == (2, 4, 3) for key in signals.files)
        # The middle round exposes temporary effects and fractional motion.
        layer = loaded.layers[1]
        array = layer.loaded.image
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for c, ax in enumerate(axes):
            ax.imshow(array[array.shape[0]//2, :, :, c], vmin=-1, vmax=16, cmap='magma', origin='lower')
            ax.set_title(f'{layer.round_label} / {layer.loaded.channel_labels[c]}')
            for row in histories[histories.round_label == layer.round_label].itertuples():
                ax.plot(row.x, row.y, '+', color='cyan')
                ax.text(row.x+1, row.y+1, row.amplicon_id, color='cyan', fontsize=7)
        sections.append(f'<section id="{case["name"]}"><h2>{case["name"]}</h2><p>Enabled: '
            +html.escape(', '.join(case['factors']) or 'none')+'</p>'
            +embed_figure(fig, case['name']+' middle-round channel slices; cyan marks full truth, including absent emission')
            +render_table(histories[['amplicon_id','round_label','z','y','x','dropped','weakened','lost','emitting','center_in_bounds','support_intersects','support_truncated']])
            +'<details><summary>Complete saved config and transforms</summary><pre>'
            +html.escape(json.dumps(ext, indent=2))+'</pre></details></section>')
    destination = directory / 'inspection.html'
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Controlled development presets</title>'
        '<style>body{font:16px system-ui;margin:2em}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #aaa}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Controlled development presets</h1><p>Saved-file integrity and image/config reload checks passed. '
        'Development fixtures only; no calibrated evaluation, cellular truth or detection/decoding claim. '
        'ZYXC images remain 3D; these views show center slices, with shared intensity range −1 to 16. '
        'Cyan labels are independent formed truth, including lost or dropped emission. Physical calibration is unknown.</p>'
        '<nav>'+ ' | '.join(f'<a href="#{c["name"]}">{c["name"]}</a>' for c in manifest['cases'])+'</nav>'
        +render_table(manifest['cases'])+''.join(sections)+'</html>')
    write_json(directory / 'inspection.json', dict(cases=len(manifest['cases']), saved_file_integrity='exact',
        images_config_truth='exact at create; saved hashes and logical images/config checked on inspect',
        report_sha256=checksum(destination), manifest_sha256=checksum(directory / 'manifest.json')))


if __name__ == '__main__':
    command, target = sys.argv[1:]
    {'create': create, 'inspect': inspect}[command](Path(target))
