"""E1: independent-process saved synthetic delivery and offline review integrity."""
from pathlib import Path
import json
import subprocess
import sys


EXAMPLE = Path(__file__).resolve().parents[3] / 'docs/examples/saved_synthetic.py'


def test_saved_delivery_fresh_process_and_report(tmp_path):
    destination = tmp_path / 'delivery'
    for mode in ('create', 'reload', 'report'):
        result = subprocess.run([sys.executable, str(EXAMPLE), mode, str(destination)],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    verification = json.loads(next(destination.glob('reload-*.json')).read_text())
    assert [case['depth'] for case in verification['cases']] == [3, 1]
    assert all(case['independent_truth'] == 'passed' for case in verification['cases'])
    report = (destination / 'review.html').read_text()
    assert report.count('<svg ') == 6
    assert 'gene-A' in report and 'gene-B' in report and 'W-173 remains open' in report
    assert '<script' not in report and '<img' not in report and 'src=' not in report
    original = (destination / 'review.html').read_bytes()
    repeated = subprocess.run([sys.executable, str(EXAMPLE), 'report', str(destination)],
                              capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0 and 'Preserve report versions' in repeated.stderr
    assert (destination / 'review.html').read_bytes() == original
    # Saved truth is part of delivery integrity, not implicitly regenerated.
    path = destination / 'z3' / 'truth-signals.npz'
    path.write_bytes(path.read_bytes() + b'corrupt')
    corrupted = subprocess.run([sys.executable, str(EXAMPLE), 'reload', str(destination)],
                              capture_output=True, text=True, timeout=60)
    assert corrupted.returncode != 0 and 'delivery checksum mismatch' in corrupted.stderr
