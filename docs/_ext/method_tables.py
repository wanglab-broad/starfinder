"""The ``method-table`` and ``config-options`` directives: tables read from the method registries.

Each stage registers its methods in a public dict from a frozen config type to the stage's
spec (``docs/method-registry.md``). These directives render that registry, and the fields
of one config, at build time, so a reference page never lists a method or a default that
the code no longer has. Usage, in MyST::

    ```{method-table} starfinder.spot_finding.SPOT_FINDING_METHODS
    ```

    ```{config-options} starfinder.spot_finding.SpotiflowConfig
    ```

The method table shows the first line of each config class's docstring, so that line must
read well to a user.
"""
from __future__ import annotations

import dataclasses
import importlib

from docutils import nodes
from docutils.parsers.rst import Directive, directives


def _resolve(path):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _first_line(obj):
    doc = (obj.__doc__ or "").strip()
    return doc.split("\n\n")[0].replace("\n", " ").strip() if doc else ""


def _dimensions(spec):
    dims = getattr(spec, "dimensions", None)
    if not dims:
        return ""
    return " and ".join(f"{d}D" for d in sorted(dims))


def _extras(spec):
    extras = sorted({d.extra for d in getattr(spec, "requires", ()) if getattr(d, "extra", None)})
    return ", ".join(extras) if extras else "none"


def _default(field):
    if field.default is not dataclasses.MISSING:
        value = field.default
    elif field.default_factory is not dataclasses.MISSING:
        value = field.default_factory()
    else:
        return "required"
    return repr(value)


def _cell(text, literal=False):
    """A table cell; in running text, ``double backticks`` become inline code."""
    entry = nodes.entry()
    paragraph = nodes.paragraph()
    if literal:
        paragraph += nodes.literal(text=text)
    else:
        for i, part in enumerate(text.split("``")):
            if part:
                paragraph += nodes.literal(text=part) if i % 2 else nodes.Text(part)
    entry += paragraph
    return entry


def _table(header, rows, literal_columns):
    table = nodes.table(classes=["sf-generated-table"])
    group = nodes.tgroup(cols=len(header))
    table += group
    group.extend(nodes.colspec(colwidth=1) for _ in header)
    head = nodes.thead()
    line = nodes.row()
    line.extend(_cell(h) for h in header)
    head += line
    body = nodes.tbody()
    for cells in rows:
        line = nodes.row()
        line.extend(_cell(text, i in literal_columns) for i, text in enumerate(cells))
        body += line
    group += head
    group += body
    return table


class MethodTable(Directive):
    """One row per registered method: name, config type, what it does, dimensions, extras."""

    required_arguments = 1
    has_content = False

    def run(self):
        try:
            registry = _resolve(self.arguments[0])
        except (ImportError, AttributeError) as error:
            raise self.error(f"method-table: cannot import {self.arguments[0]!r}: {error}")
        header = ["Method", "Configuration", "What it does", "Dimensions", "Optional extras"]
        rows = [[spec.name, config.__name__, _first_line(config), _dimensions(spec), _extras(spec)]
                for config, spec in registry.items()]
        # A column that no method fills (dimensions of a decoder, for example) is left out.
        keep = [i for i in range(len(header)) if i < 3 or any(row[i] for row in rows)]
        header = [header[i] for i in keep]
        rows = [[row[i] for i in keep] for row in rows]
        return [_table(header, rows, literal_columns={0, 1})]


class ConfigOptions(Directive):
    """One row per field of a config dataclass: name and default."""

    required_arguments = 1
    has_content = False
    option_spec = {"exclude": directives.unchanged}

    def run(self):
        try:
            config = _resolve(self.arguments[0])
        except (ImportError, AttributeError) as error:
            raise self.error(f"config-options: cannot import {self.arguments[0]!r}: {error}")
        if not dataclasses.is_dataclass(config):
            raise self.error(f"config-options: {self.arguments[0]!r} is not a dataclass")
        excluded = {name.strip() for name in self.options.get("exclude", "").split(",") if name.strip()}
        rows = [[field.name, _default(field)] for field in dataclasses.fields(config) if field.name not in excluded]
        return [_table(["Option", "Default"], rows, literal_columns={0, 1})]


def setup(app):
    app.add_directive("method-table", MethodTable)
    app.add_directive("config-options", ConfigOptions)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
