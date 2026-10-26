"""Check the public-name lists, alphabetical references and five-section navigation.

Each public namespace's ``__all__`` equals the union of the sorted autosummary
lists under ``docs/api``; the root bullets of ``api/inventory.rst`` equal the root
``__all__``, and each arrow names an export of its owning namespace that is the
root object. MATLAB references, the home toctree and member order are checked too.

Run with the installed checkout: uv run python ../../docs/check_reference.py.
No images, generated stubs or external services are needed.
"""
from importlib import import_module
from pathlib import Path
import re

DOCS = Path(__file__).resolve().parent


def main():
    inventory = (DOCS / 'api/inventory.rst').read_text()
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
    root = import_module('starfinder')
    root_names = set(re.findall(r'^\* ``starfinder\.([^`]+)``', inventory, re.M))
    assert root_names == set(root.__all__)
    targets = re.findall(r'^\* ``starfinder\.([^`]+)`` → :py:obj:`(starfinder\.[^`]+)`', inventory, re.M)
    for name, target in targets:
        owner, _, attribute = target.rpartition('.')
        assert target in expected, ('root target is not an export of its namespace', name, target)
        assert getattr(root, name) is getattr(import_module(owner), attribute), ('root object differs', name, target)
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
    index = (DOCS / 'index.md').read_text()
    assert index.count('```{toctree}') == 1, 'the home page has one toctree, the five sections'
    home = index.split('```{toctree}', 1)[1].split('```', 1)[0]
    assert re.findall(r'^([^:\n].*) <([^>]+)>$', home, re.M) == [
        ('Get Started', 'get-started/index'), ('Workflow', 'workflow/index'),
        ('API', 'api/index'), ('Wiki', 'wiki'), ('Benchmark', 'benchmark/index')]
    assert ':member-order: alphabetical' in (DOCS / '_templates/autosummary/class.rst').read_text()
    assert ':member-order: alphabetical' in (DOCS / 'api/matlab/dataset.rst').read_text()
    print(f'PASS: {len(expected)} Python exports, {len(root_names)} root exports '
          f'({len(targets)} owner targets), {len(expected_matlab)} MATLAB interfaces; alphabetical references and five sections.')


if __name__ == '__main__':
    main()
