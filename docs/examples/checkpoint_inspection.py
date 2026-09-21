"""Read-only inspection of the bounded saved-formed-v3 delivery (no processing)."""
import html
import json
from pathlib import Path
import sys

import pandas as pd

from saved_synthetic import verify_files
from saved_synthetic_report import inspection_figure, decoding_rows
from starfinder.io import load_image_checkpoint, load_candidate_checkpoint, load_final_checkpoint
from starfinder.reporting import render_table


def inspect_saved(directory, destination):
    directory, destination = Path(directory), Path(destination)
    delivery = json.loads((directory/'delivery.json').read_text())
    verify_files(directory, delivery)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Saved checkpoint inspection</title>',
        '<style>body{font:16px system-ui;margin:2em}img{max-width:100%}table{display:block;overflow:auto;border-collapse:collapse}td,th{padding:.4em;border:1px solid #bbb}pre{white-space:pre-wrap}</style>',
        '<h1>Saved checkpoint inspection</h1><p>Development fixtures only; no calibrated real-data performance. '
        'Read-only reload: no detection, registration, extraction or filtering runs here. '
        'Coordinates are zero-based ZYX voxel indices; physical calibration is unknown. '
        'Truth IDs gt-A/gt-B and detector displays spot-1/spot-2 use explicit coordinate correspondence.</p>',
        '<nav>'+''.join(f'<a href="#z{c["depth"]}">Z={c["depth"]}</a> ' for c in delivery['cases'])+'</nav>']
    for case in delivery['cases']:
        root = directory/case['root']
        images = load_image_checkpoint(directory/case['registered'])
        candidates = load_candidate_checkpoint(directory/case['candidates'])
        formed = pd.read_parquet(root/'formed.parquet')
        histories = pd.read_parquet(root/'round-truth.parquet')
        final_path = root/'run/final-accepted/artifact.json'
        if final_path.exists():
            final = load_final_checkpoint(final_path)
            decoded, filtered = final.pre_qc.decoded.table, final.filtering.table
            trace = final.pre_qc.source_trace(spot_namespace=candidates.spots.spot_namespace,
                                            spot_id=candidates.intensities.spot_ids[0])
            assert trace['available']
            stages = 'Reloaded decoded pre-QC and final accepted checkpoints; source trace verified.'
        else:
            # Accepted older W-175 deliveries predate molecular persistence.
            decoded = pd.read_parquet(root/'uninterrupted-decoded.parquet')
            filtered = pd.read_parquet(root/'uninterrupted-filtered.parquet')
            stages = 'Molecular checkpoints unavailable; these are saved example comparison tables, not molecular artifacts.'
        rows, identities, signals = decoding_rows(candidates, formed, decoded, filtered, case['settings']['filtering'])
        parts += [f'<section id="z{case["depth"]}"><h2>Z={case["depth"]}</h2><p>'+stages+'</p>',
            '<h3>Explicit truth/detection correspondence</h3>'+render_table(identities),
            '<h3>Observed colors, nucleotide decoding, gene assignment and filtering</h3>'+render_table(rows),
            '<p>Nucleotide decoding uses the saved start base; it is independent of gene assignment. '
            'Exact WTA matching has no correction/distance metric.</p>',
            '<h3>Source traces in saved round/channel order</h3>'+render_table(signals),
            '<h3>Complete formed population</h3>'+render_table(formed),
            '<h3>Per-round histories, loss and visibility</h3>'+render_table(histories)]
        for index, layer in enumerate(images.layers):
            parts += ['<h3>'+html.escape(layer.round_label)+' — saved image and transform</h3>',
                '<pre>'+html.escape(repr(layer.processing))+'</pre>',
                '<pre>'+html.escape(repr(layer.loaded.metadata))+'</pre>',
                inspection_figure(layer.loaded, formed, index, layer.round_label)]
        parts.append('</section>')
    parts += ['<p>Images and tables are embedded for offline access. The source delivery and its checksum manifest '
              'are still required for raw HDF5/Parquet inspection. Owner Jiahao; retain through thesis/publication; '
              'backup and public reproducibility unverified. MATLAB and live Jupyter execution are not implied.</p></html>']
    with destination.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(parts))
    return destination


if __name__ == '__main__':
    inspect_saved(sys.argv[1], sys.argv[2])
