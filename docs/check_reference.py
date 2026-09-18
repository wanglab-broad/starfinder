"""Check authored API inventory, alphabetical lists and five-section navigation.

Run with the installed checkout: uv run python ../../docs/check_reference.py.
No images, generated stubs or external services are needed.
"""
from importlib import import_module
from pathlib import Path
import re

DOCS = Path(__file__).resolve().parent


def main():
    inventory = (DOCS / 'api/inventory.rst').read_text()
    documented = set(re.findall(r':py:obj:`(starfinder\.[^`]+)`', inventory))
    expected = set()
    summaries = set()
    for path in sorted((DOCS / 'api').glob('*.rst')):
        module = None
        lines = path.read_text().splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('.. currentmodule:: '):
                module = lines[i].split(':: ')[1]
            if lines[i] == '.. autosummary::':
                names = []
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith('   ')):
                    name = lines[i].strip()
                    if name and not name.startswith(':'):
                        names.append(name)
                    i += 1
                assert names == sorted(names, key=str.casefold), (path, names)
                summaries.update(f'{module}.{name}' for name in names)
                continue
            i += 1
    # Discover explicit public namespaces, not a copied list of expected exports.
    package = DOCS.parent / 'src/python/starfinder'
    for path in sorted(package.rglob('*.py')):
        relative = path.relative_to(package).with_suffix('')
        parts = relative.parts
        if any(part.startswith('_') and part != '__init__' for part in parts):
            continue
        if '__all__' not in path.read_text():
            continue
        suffix = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
        if not suffix:
            continue
        module = import_module('starfinder.' + suffix)
        expected.update(f'{module.__name__}.{name}' for name in module.__all__)
    assert expected == summaries, {'missing_pages': expected - summaries, 'extra_pages': summaries - expected}
    assert expected == documented, {'missing_inventory': expected - documented, 'extra_inventory': documented - expected}
    root = import_module('starfinder')
    root_names = set(re.findall(r'^\* ``starfinder\.([^`]+)``', inventory, re.M))
    assert root_names == set(root.__all__)
    index = (DOCS / 'api/python-index.rst').read_text()
    names = re.findall(r':py:obj:`([^`]+)`', index)
    assert set(names) == expected
    assert names == sorted(names, key=lambda n: (n.rsplit('.', 1)[-1].casefold(), n))
    matlab = []
    for path in (DOCS / 'api/matlab').glob('*.rst'):
        names = re.findall(r'mat:autofunction:: (\w+)', path.read_text())
        assert names == sorted(names, key=str.casefold), path
        matlab.extend(names)
    matlab.append('STARMapDataset')
    expected_matlab = {p.stem for p in (DOCS.parent / 'src/matlab').glob('*.m')}
    assert set(matlab) == expected_matlab
    names = re.findall(r':mat:(?:func|class):`([^`]+)`', (DOCS / 'api/matlab-index.rst').read_text())
    assert set(names) == expected_matlab
    assert names == sorted(names, key=str.casefold)
    home = (DOCS / 'index.md').read_text().split('```{toctree}', 1)[1].split('```', 1)[0]
    assert re.findall(r'^([^:\n].*) <([^>]+)>$', home, re.M) == [
        ('Get started', 'getting-started'), ('Workflow', 'workflows'),
        ('Benchmark', 'benchmark'), ('API', 'api/index'), ('Wiki / Convention', 'wiki')]
    assert ':member-order: alphabetical' in (DOCS / '_templates/autosummary/class.rst').read_text()
    assert ':member-order: alphabetical' in (DOCS / 'api/matlab/dataset.rst').read_text()
    print(f'PASS: {len(expected)} Python exports, {len(root_names)} root exports, '
          f'{len(expected_matlab)} MATLAB interfaces; alphabetical references and five sections.')


if __name__ == '__main__':
    main()
