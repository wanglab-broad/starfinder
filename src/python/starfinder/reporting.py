"""Offline presentation of validated saved records; never reruns processing."""
from __future__ import annotations

import base64
from dataclasses import asdict, is_dataclass
import hashlib
import html
from io import BytesIO, StringIO
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from starfinder.provenance import read_run

__all__ = ['embed_figure', 'render_table', 'write_run_summary']


def render_table(rows) -> str:
    """Render records or a DataFrame as an escaped HTML table (no row index)."""
    return pd.DataFrame(rows).to_html(index=False, escape=True, border=0)


def embed_figure(fig, label: str, svg: bool = False) -> str:
    """Embed and close a Matplotlib figure for offline HTML, with an alt label."""
    import matplotlib.pyplot as plt
    output = StringIO() if svg else BytesIO()
    fig.savefig(output, format='svg' if svg else 'png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    if svg:
        content = output.getvalue()
        content = content[content.index('<svg '):]
        return content.replace('<svg ', '<svg role="img" aria-label="'+html.escape(label, quote=True)+'" ', 1)
    return '<img alt="'+html.escape(label, quote=True)+'" src="data:image/png;base64,'+base64.b64encode(output.getvalue()).decode()+'">'


def _display(value):
    if is_dataclass(value):
        return _display(asdict(value))
    if isinstance(value, np.ndarray):
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                    values=value.tolist() if value.size <= 64 else 'omitted from display; retained in source')
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _display(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_display(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _details(value):
    return '<pre>'+html.escape(json.dumps(_display(value), indent=2, default=str))+'</pre>'


def _population(table):
    return (f'<p>{len(table)} rows retained; showing at most 200. '
            'Read the source checkpoint for the complete table.</p>'+render_table(table.head(200)))


def _link(uri, label):
    uri = str(uri)
    parsed = urlsplit(uri)
    if parsed.scheme not in ('', 'file', 'http', 'https'):
        return html.escape(label)+' (locator: '+html.escape(uri)+')'
    target = Path(uri).absolute().as_uri() if not parsed.scheme else uri
    return '<a href="'+html.escape(target, quote=True)+'">'+html.escape(label)+'</a>'


def write_run_summary(path: str | Path, destination: str | Path, *,
                      sha256: str | None = None, title: str = 'Saved run summary') -> Path:
    """Write standalone HTML from a validated run, events and linked checkpoints.

    Run status and stage state are independent of artifact retention. Missing
    metrics remain unavailable; a successful configured subset is not a completed
    scientific workflow. Integrity errors propagate without publishing a report.
    Existing destinations are never overwritten. Source links need their original
    files; summary text is embedded and needs no network or live kernel.
    ``sha256`` optionally pins the run manifest. Component checksums are verified
    by ``read_run``. No image processing or scientific evaluation is performed.
    """
    path = Path(path).resolve()
    if path.is_dir():
        path /= 'run.json'
    destination = Path(destination)
    run = read_run(path, sha256=sha256)
    extension = run.get('extensions', {}).get('starfinder.provenance', {})
    final = extension.get('final_state', {})
    sections = []

    def section(identity, heading, body):
        sections.append((identity, heading, body))

    section('outcome', 'Recorded outcome', render_table([dict(
        run_id=run['run_id'], dataset=run['dataset_id'], sample=run['sample_id'],
        status=run['status'], detected=final.get('detected') if final.get('detected') is not None else 'unavailable',
        counts=final.get('counts') if final.get('counts') is not None else 'unavailable')])+
        '<p>Status describes the requested run only. Absent metrics are unavailable, not zero. '
        'Software execution and synthetic development examples do not establish validated real-data performance.</p>')
    stages = []
    for stage in ('prepared_input', 'registered_images', 'candidates_signals', 'decoded_pre_qc', 'final_accepted'):
        events = [e for e in run['events'] if e['stage'] == stage]
        artifacts = [a for a in run['artifacts'] if a['stage'] == stage]
        stages.append(dict(stage=stage, recorded_state=extension.get('stage_state', {}).get(stage, 'not recorded'),
            events=len(events), artifact_states=', '.join(a['status'] for a in artifacts) or 'none recorded'))
    section('stages', 'Stages and retention states', render_table(stages))
    section('configuration', 'Requested and effective configuration', _details(run['config']))
    section('events', 'Actual operations and diagnostics', ''.join(
        '<details><summary>'+html.escape(f"{e['sequence']}: {e['stage']} / {e['operation']} / {e['outcome']} / {e['round']}")+
        '</summary>'+_details(e)+'</details>' for e in run['events']) or '<p>No events recorded.</p>')
    section('failures', 'Failures and recovery', _details(run['failures']) if run['failures'] else '<p>No failures recorded.</p>')
    section('geometry', 'Geometry, transforms and partial histories', _details(final) if final else '<p>Final snapshot unavailable.</p>')
    artifact_body = []
    for a in run['artifacts']:
        artifact_body.append('<h3>'+html.escape(a['stage']+' — '+a['status'])+'</h3>'+_details(a))
        artifact_body.extend('<p>'+_link(path.parent/c['path'], c['path'])+'</p>' for c in a['components'])
        manifest = a.get('extensions', {}).get('starfinder.checkpoint', {}).get('manifest_path')
        if a['status'] == 'complete' and manifest:
            from starfinder.io import load_decoded_checkpoint, load_final_checkpoint, load_candidate_checkpoint
            loaders = dict(candidates_signals=load_candidate_checkpoint,
                           decoded_pre_qc=load_decoded_checkpoint, final_accepted=load_final_checkpoint)
            loader = loaders.get(a['stage'])
            if loader:
                descriptors = [c for c in a['components'] if c['path'] == manifest]
                if len(descriptors) != 1:
                    raise ValueError('summary: checkpoint manifest is not a verified component')
                saved = loader(path.parent/manifest, sha256=descriptors[0]['sha256'])
                if (saved.artifact['run_id'], saved.artifact['artifact_id']) != (run['run_id'], a['artifact_id']):
                    raise ValueError('summary: checkpoint identity differs from run artifact')
                if a['stage'] == 'candidates_signals':
                    artifact_body.append(_population(saved.spots.spots)+_details(dict(
                        signal_shape=saved.intensities.values.shape, rounds=saved.intensities.round_labels,
                        channels=saved.intensities.channel_labels, namespace=saved.intensities.spot_namespace,
                        geometry=saved.spots.metadata)))
                elif a['stage'] == 'decoded_pre_qc':
                    artifact_body.append('<p>Complete pre-QC population; observed colors and gene assignments are separate.</p>'+_population(saved.decoded.table))
                else:
                    artifact_body.append('<p>QC accounting includes rejected candidates; accepted population is separate.</p>'+_population(saved.filtering.table)+_details(saved.filtering.counts))
    section('artifacts', 'Saved checkpoints and populations', ''.join(artifact_body) or '<p>No artifacts recorded.</p>')
    section('sources', 'Sources and identity', _link(path, 'Validated run.json')+_details(dict(
        run_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), code=run['code'], sources=run['sources']))+
        ''.join('<p>'+_link(s['uri'], s['source_id'])+'</p>' for s in run['sources'] if s['uri']))
    section('retention', 'Environment, retention and limitations', _details(dict(
        environment=run['environment'], saving_policy=run['saving_policy'], owner=run['owner'],
        retention=run['retention'], backup_status=run['backup_status']))+
        '<p>Physical calibration stays unknown where recorded as null. Source links are locators, '
        'not evidence of public availability. Copying this HTML preserves the summary; copying checkpoints '
        'also requires their components and source references. No scientific or human approval is implied.</p>')
    content = '<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(title)+'</title>'
    content += '<style>body{font:16px system-ui;margin:2em;max-width:1200px}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:.4em;border:1px solid #bbb;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}nav a{margin-right:1em}details{margin:.6em 0}</style>'
    content += '<h1>'+html.escape(title)+'</h1><nav>'+''.join('<a href="#'+i+'">'+h+'</a>' for i,h,_ in sections)+'</nav>'
    content += ''.join('<section id="'+i+'"><h2>'+h+'</h2>'+b+'</section>' for i,h,b in sections)+'</html>'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(content)
    return destination
