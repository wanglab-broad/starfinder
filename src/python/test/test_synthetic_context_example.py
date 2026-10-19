"""The example docs/examples/context_scene.py runs, as test_background_noise.py runs its example.

The example generates and saves ``context-tissue2d-v1`` and ``appearance-tissue2d-v1`` on its
closed-form bank, so it reads no real data and needs no ``STARFINDER_APPEARANCE_BANKS``.
Generating the two presets takes 5 s or more on one CPU, so the test is ``slow``
(docs/contributing.md, "Test markers").
"""
import os
from pathlib import Path
import runpy

import pytest

pytestmark = pytest.mark.synthetic

EXAMPLE = Path(__file__).resolve().parents[3] / 'docs' / 'examples' / 'context_scene.py'


@pytest.mark.slow
def test_example(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('STARFINDER_APPEARANCE_BANKS', raising=False)
    example = runpy.run_path(str(EXAMPLE))
    summaries = example['run'](tmp_path)
    assert list(summaries) == ['context-tissue2d-v1', 'appearance-tissue2d-v1']
    for name, truth in summaries.items():
        assert (tmp_path / name / 'sample.json').is_file()
        assert truth['nuclei'] > 0 and truth['cells'] >= truth['nuclei'] and truth['molecules'] > 0
        pairs = truth['marker_pairs']
        assert list(pairs['relation']) == ['exclusive', 'coexpressed']
        # The known answer of the exclusive pair (identity 6): no cell holds both markers.
        assert pairs.loc[pairs['relation'] == 'exclusive', 'both'].tolist() == [0]
    assert (tmp_path / 'appearance-tissue2d-v1' / 'appearance.json').is_file()
    example['report'](summaries)
    printed = capsys.readouterr().out
    assert 'context-tissue2d-v1:' in printed and 'appearance-tissue2d-v1:' in printed
    assert os.environ.get('STARFINDER_APPEARANCE_BANKS') is None
