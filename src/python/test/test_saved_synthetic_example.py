"""E1: independent-process saved synthetic delivery and offline review integrity."""
from pathlib import Path
import json
from html.parser import HTMLParser
import base64
import subprocess
import sys


EXAMPLE = Path(__file__).resolve().parents[3] / 'docs/examples/saved_synthetic.py'


def test_saved_delivery_fresh_process_and_report(tmp_path, monkeypatch):
    destination = tmp_path / 'delivery'
    for mode in ('create', 'reload', 'report'):
        result = subprocess.run([sys.executable, str(EXAMPLE), mode, str(destination)],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    verification = json.loads(next(destination.glob('reload-*.json')).read_text())
    assert [case['depth'] for case in verification['cases']] == [9, 1]
    assert all(case['independent_truth'] == 'passed' for case in verification['cases'])
    report = (destination / 'review.html').read_text()
    assert 'gene-A' in report and 'gene-B' in report and 'W-173 remains open' in report
    class OfflineHTML(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids, self.anchors, self.images = set(), [], []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs:
                self.ids.add(attrs['id'])
            if tag == 'a' and attrs.get('href', '').startswith('#'):
                self.anchors.append(attrs['href'][1:])
            if tag == 'img':
                assert attrs.get('alt')
                source = attrs['src']
                assert source.startswith('data:image/png;base64,')
                assert base64.b64decode(source.split(',', 1)[1]).startswith(b'\x89PNG')
                self.images.append(source)
            assert tag not in ('script', 'iframe', 'link', 'object', 'embed')
            if tag not in ('img',):
                assert 'src' not in attrs
    parsed = OfflineHTML()
    parsed.feed(report)
    assert parsed.images and set(parsed.anchors) <= parsed.ids
    assert {'datasets','images','reload','execution','validation','reproducibility'} <= parsed.ids
    assert 'gt-A' in report and 'intensity profiles' in report
    assert 'linear 0–8 scale' in report
    delivery = json.loads((destination / 'delivery.json').read_text())
    assert delivery['create_process'] != verification['process']
    assert [case['shape'] for case in delivery['cases']] == [[9,32,32,4],[1,32,32,4]]
    assert all(case['inspection_exports'] == 'exact' for case in verification['cases'])
    # Literal independent expectations, then reorder/reject/unmatch without relabeling.
    import pandas as pd
    from starfinder.io import load_candidate_checkpoint
    monkeypatch.syspath_prepend(str(EXAMPLE.parent))
    from saved_synthetic_report import decoding_rows
    for case in delivery['cases']:
        root = destination / case['root']
        saved = load_candidate_checkpoint(destination / case['candidates'])
        formed = pd.read_parquet(root / 'formed.parquet')
        decoded = pd.read_parquet(root / 'uninterrupted-decoded.parquet')
        filtered = pd.read_parquet(root / 'uninterrupted-filtered.parquet')
        rows, identities, signals = decoding_rows(saved, formed, decoded, filtered, case['settings']['filtering'])
        assert [(r['Detected spot ID'], r['Matched GT ID'], r['Color sequence'], r['Decoded barcode'],
                 r['GT color sequence'], r['GT barcode'], r['Assigned gene'], r['GT gene'], r['Filter status'])
                for r in rows] == [
            ('spot-1', 'gt-B', '214', 'CAAT', '214', 'CAAT', 'gene-B', 'gene-B', 'accepted'),
            ('spot-2', 'gt-A', '123', 'CCAG', '123', 'CCAG', 'gene-A', 'gene-A', 'accepted')]
        assert all(r['simulation_namespace'] != r['detector_namespace'] for r in identities)
        assert [r['Round'] for r in signals[:3]] == ['round10', 'round2', 'round1']
        saved.spots.spots.sort_values('spot_id', ascending=False, inplace=True)
        decoded.loc[decoded.spot_id == '0', 'gene_id'] = 'deliberately-different-assignment'
        filtered.loc[filtered.spot_id == '0', 'accepted'] = False
        filtered.loc[filtered.spot_id == '0', 'rejection_reasons'] = 'score:wta_l2_nll'
        rows, _, reordered_signals = decoding_rows(saved, formed[formed.amplicon_id == 'gt-A'],
            decoded.iloc[::-1], filtered.iloc[::-1], case['settings']['filtering'])
        assert [r['Detected spot ID'] for r in rows] == ['spot-2', 'spot-1']
        assert rows[1]['Matched GT ID'] == 'unmatched' and rows[1]['GT barcode'] == '—'
        assert rows[1]['Decoded barcode'] == 'CAAT' and rows[1]['Color sequence'] == '214'
        assert rows[1]['Filter status'] == 'rejected' and rows[1]['Rejection reason'] == 'score:wta_l2_nll'
        assert reordered_signals[3:] == [dict(r, **{'Matched GT ID': 'unmatched'}) for r in signals[:3]]
    assert 'candidate-1' not in report and 'spot-A' not in report
    assert 'Decoded barcode' in report and 'start_base=C' in report
    original = (destination / 'review.html').read_bytes()
    repeated = subprocess.run([sys.executable, str(EXAMPLE), 'report', str(destination)],
                              capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0 and 'Preserve report versions' in repeated.stderr
    assert (destination / 'review.html').read_bytes() == original
    # Saved truth is part of delivery integrity, not implicitly regenerated.
    path = destination / 'z9' / 'truth-signals.npz'
    path.write_bytes(path.read_bytes() + b'corrupt')
    corrupted = subprocess.run([sys.executable, str(EXAMPLE), 'reload', str(destination)],
                              capture_output=True, text=True, timeout=60)
    assert corrupted.returncode != 0 and 'delivery checksum mismatch' in corrupted.stderr
