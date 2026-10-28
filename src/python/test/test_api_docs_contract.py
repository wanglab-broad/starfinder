"""Public constant comments survive re-exporting into API namespaces."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.io]
ROOT = Path(__file__).resolve().parents[3]


def test_reexported_constants_render_source_comments(tmp_path):
    """Build both autosummary tables and object pages from public constant names."""
    pytest.importorskip('sphinx', reason='documentation extra required')
    source = tmp_path / 'docs'
    source.mkdir()
    (source / 'conf.py').write_text(
        f"import sys\nsys.path.insert(0, {str(ROOT / 'docs/_ext')!r})\n"
        "extensions = ['public_constants', 'sphinx.ext.autosummary']\n"
        "autosummary_generate = True\nmaster_doc = 'index'\n"
    )
    (source / 'index.rst').write_text(
        'Constants\n=========\n\n.. autosummary::\n   :toctree: generated\n\n'
        '   starfinder.assembly.ASSEMBLY_STATUSES\n'
        '   starfinder.assignment.ASSIGNMENT_STATUSES\n'
        '   starfinder.synthetic.PRESET_VERSION\n'
    )
    # A separate Sphinx process leaves its global docutils registrations isolated
    # from other documentation contract tests in the same pytest session.
    build = subprocess.run(
        [sys.executable, '-m', 'sphinx', '-n', '-W', '--keep-going', '-b', 'html',
         str(source), str(tmp_path / 'html')],
        capture_output=True, text=True, timeout=60,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    expected = {
        'starfinder.assembly.ASSEMBLY_STATUSES': 'Whether assembly retained or excluded an observation.',
        'starfinder.assignment.ASSIGNMENT_STATUSES': 'What assign decided for one molecule.',
        'starfinder.synthetic.PRESET_VERSION': 'Version of the benchmark appearance defaults;',
    }
    for name, comment in expected.items():
        page = (tmp_path / 'html/generated' / f'{name}.html').read_text()
        assert comment in page
        assert 'Built-in immutable sequence.' not in page
        assert 'str(object=' not in page
        assert f'id="{name}"' in page
    index = (tmp_path / 'html/index.html').read_text()
    assert 'What assign decided for one molecule.' in index
    assert 'Version of the benchmark appearance defaults;' in index
