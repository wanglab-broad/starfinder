"""API pages: public constant comments survive re-exporting, and class pages describe each field once."""
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


def test_class_pages_describe_each_field_once(tmp_path):
    """A dataclass field named in Parameters gets no second entry; tables list the rest."""
    pytest.importorskip('sphinx', reason='documentation extra required')
    source = tmp_path / 'docs'
    (source / '_templates/autosummary').mkdir(parents=True)
    (source / '_templates/autosummary/class.rst').write_text(
        (ROOT / 'docs/_templates/autosummary/class.rst').read_text())
    (tmp_path / 'sample_classes.py').write_text(
        'import dataclasses\n\n\n'
        '@dataclasses.dataclass\n'
        'class Sample:\n'
        '    """A sample.\n\n    Parameters\n    ----------\n'
        '    size : int\n        The size.\n    """\n\n'
        '    size: int = 1\n'
        '    extra: int = 2\n'
        '    _cache: dict = dataclasses.field(default_factory=dict)\n\n'
        '    @property\n'
        '    def double(self):\n'
        '        """Twice the size."""\n'
        '        return 2 * self.size\n\n'
        '    def grow(self, step):\n'
        '        """Grow by a step.\n\n        Parameters\n        ----------\n'
        '        step : int\n            The step.\n        """\n'
    )
    (source / 'conf.py').write_text(
        f"import sys\nsys.path[:0] = [{str(ROOT / 'docs/_ext')!r}, {str(tmp_path)!r}]\n"
        "extensions = ['sphinx.ext.autosummary', 'sphinx.ext.napoleon', 'class_pages']\n"
        "autosummary_generate = True\nmaster_doc = 'index'\ntemplates_path = ['_templates']\n"
        "napoleon_use_param = False\nautodoc_typehints = 'none'\n"
    )
    (source / 'index.rst').write_text(
        'Classes\n=======\n\nSee :attr:`sample_classes.Sample.size`.\n\n'
        '.. currentmodule:: sample_classes\n\n'
        '.. autosummary::\n   :toctree: generated\n\n   Sample\n'
    )
    build = subprocess.run(
        [sys.executable, '-m', 'sphinx', '-n', '-W', '--keep-going', '-b', 'html',
         str(source), str(tmp_path / 'html')],
        capture_output=True, text=True, timeout=60,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    page = (tmp_path / 'html/generated/sample_classes.Sample.html').read_text()
    # The field Parameters names has no entry; the field it does not name keeps one.
    assert 'id="sample_classes.Sample.size"' not in page
    assert 'id="sample_classes.Sample.extra"' in page
    assert 'id="sample_classes.Sample.double"' in page
    assert 'id="sample_classes.Sample.grow"' in page
    assert '_cache' not in page
    attributes = page.split('rubric">Attributes', 1)[1].split('rubric">Methods', 1)
    assert 'Twice the size.' in attributes[0] and 'extra' in attributes[0]
    assert 'Grow by a step.' in attributes[1].split('id="sample_classes.Sample.', 1)[0]
    # A reference to the removed entry points at the class.
    index = (tmp_path / 'html/index.html').read_text()
    assert 'href="generated/sample_classes.Sample.html#sample_classes.Sample"' in index
