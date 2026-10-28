"""Render public constants from their source comments, including re-exports.

Sphinx's data documenter looks for ``#:`` comments in the public module only.
A constant imported from a private module otherwise inherits its builtin type's
text. Follow explicit imports to the defining comment, without changing the
package's exports or copying their documentation into API pages.
"""
from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

from sphinx.pycode import ModuleAnalyzer


def _comment(module: str, name: str, seen: set[tuple[str, str]], dependencies: set[str]) -> list[str] | None:
    if (module, name) in seen:
        return None
    seen.add((module, name))
    analyzer = ModuleAnalyzer.for_module(module)
    comments = analyzer.find_attr_docs()
    dependencies.add(analyzer.srcname)
    if ('', name) in comments:
        return list(comments['', name])
    source = Path(analyzer.srcname)
    package = module if source.name == '__init__.py' else module.rpartition('.')[0]
    for node in ast.parse(source.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    target = resolve_name('.' * node.level + (node.module or ''), package)
                    return _comment(target, alias.name, seen, dependencies)
    return None


def _public_constant_docstring(app, what, name, obj, options, lines):
    if what != 'data' or not name.startswith('starfinder.'):
        return
    module, _, attribute = name.rpartition('.')
    dependencies: set[str] = set()
    comment = _comment(module, attribute, set(), dependencies)
    for source in dependencies:
        app.env.note_dependency(source)
    if comment:
        lines[:] = comment


def setup(app):
    app.setup_extension('sphinx.ext.autodoc')
    app.connect('autodoc-process-docstring', _public_constant_docstring)
    return {'version': '1', 'parallel_read_safe': True, 'parallel_write_safe': True}
