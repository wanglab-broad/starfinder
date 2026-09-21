"""Saved-only success/failure inspection with independent state expectations."""
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys

import pytest

from starfinder.provenance import read_run
from starfinder.reporting import write_run_summary
from test.test_provenance import fixture, recorder

EXAMPLES = Path(__file__).resolve().parents[3]/'docs/examples'


class Offline(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = set(), set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        assert tag not in ('script', 'iframe', 'link', 'object', 'embed')
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if attrs.get('href', '').startswith('#'):
            self.links.add(attrs['href'][1:])
        if tag == 'img':
            assert attrs['src'].startswith('data:image/png;base64,') and attrs['alt']


def test_summary_integrity_missing_metrics_and_escaping(tmp_path):
    fov, config = fixture(tmp_path)
    rec = recorder(tmp_path/'run')
    fov.run(config, provenance=rec)
    output = write_run_summary(rec.path, tmp_path/'summary.html', title='<script>bad</script>')
    body = output.read_text()
    assert '&lt;script&gt;bad&lt;/script&gt;' in body
    assert 'candidates_signals_disabled' in body and 'gene' in body
    parsed = Offline(); parsed.feed(body)
    assert parsed.links <= parsed.ids
    with pytest.raises(FileExistsError):
        write_run_summary(rec.path, output)
    with pytest.raises(ValueError, match='integrity'):
        write_run_summary(rec.path, tmp_path/'bad.html', sha256='0'*64)
    original = rec.path.read_text()
    changed = json.loads(original)
    saved = next(a for a in changed['artifacts'] if a['status'] == 'complete')
    saved['extensions']['starfinder.checkpoint']['manifest_path'] = '../foreign/artifact.json'
    rec.path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='verified component'):
        write_run_summary(rec.path, tmp_path/'foreign.html')
    assert not (tmp_path/'foreign.html').exists()
    rec.path.write_text(original)
    run = read_run(rec.path)
    component = next(c for a in run['artifacts'] for c in a['components'])
    (rec.path.parent/component['path']).unlink()
    with pytest.raises(FileNotFoundError):
        write_run_summary(rec.path, tmp_path/'missing.html')
    assert not (tmp_path/'missing.html').exists()


def test_running_is_not_promoted_to_success(tmp_path):
    rec = recorder(tmp_path/'running')
    output = write_run_summary(rec.path, tmp_path/'running.html').read_text()
    assert '<td>running</td>' in output and '<td>unavailable</td>' in output
    assert 'No events recorded.' in output and 'No artifacts recorded.' in output
    assert '<td>succeeded</td>' not in output


def test_saved_examples_and_notebook_cells(tmp_path, monkeypatch):
    root = tmp_path/'examples'
    for mode in ('prepare', 'render'):
        result = subprocess.run([sys.executable, str(EXAMPLES/'run_summaries.py'), mode, str(root)],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout+result.stderr
    success, failure = (root/'z9-summary.html').read_text(), (root/'failed-summary.html').read_text()
    assert 'succeeded' in success and 'accepted' in success
    assert 'InsufficientLandmarksError' in failure and 'partial' in failure and 'unavailable' in failure
    run = read_run(root/'failed')
    assert run['status'] == 'failed'
    assert run['extensions']['starfinder.provenance']['final_state']['counts'] is None
    assert not any(a['stage'] in ('decoded_pre_qc', 'final_accepted') for a in run['artifacts'])
    for path in root.glob('*.html'):
        parsed = Offline(); parsed.feed(path.read_text()); assert parsed.links <= parsed.ids
    body = (root/'inspection.html').read_text()
    for term in ('gt-A', 'gt-B', 'spot-1', 'spot-2', 'CAAT', 'CCAG', 'per-round', 'source trace verified'):
        assert term.casefold() in body.casefold()
    monkeypatch.setenv('STARFINDER_INSPECTION_ROOT', str(root))
    monkeypatch.chdir(EXAMPLES.parents[1]/'src/python')
    notebook = json.loads((EXAMPLES/'checkpoint_inspection.ipynb').read_text())
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            exec(compile(''.join(cell['source']), 'checkpoint_inspection.ipynb', 'exec'), namespace)
    assert namespace['trace']['available']
    assert (root/'notebook-inspection.html').is_file()
