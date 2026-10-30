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
import enum
import importlib
import inspect
import re
from pathlib import Path

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


def _parameter_descriptions(obj):
    """Read numpydoc Parameters once, against public fields or a signature.

    Excluded option-table columns do not exempt fields from checking. A callable
    with no parameters may omit the section, as there are no entries to describe.
    """
    if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
        expected = {field.name for field in dataclasses.fields(obj)
                    if not field.name.startswith("_")}
        kind = "field"
    else:
        expected = set()
        for parameter in inspect.signature(obj).parameters.values():
            if parameter.name in ("self", "cls"):
                continue
            stars = {inspect.Parameter.VAR_POSITIONAL: "*",
                     inspect.Parameter.VAR_KEYWORD: "**"}.get(parameter.kind, "")
            expected.add(stars + parameter.name)
        kind = "parameter"
    label = obj.__qualname__
    lines = inspect.cleandoc(obj.__doc__ or "").splitlines()
    sections = [i for i in range(len(lines) - 1)
                if lines[i] and not lines[i].startswith(" ")
                and re.fullmatch(r"-{3,}", lines[i + 1])]
    starts = [i for i in sections if lines[i] == "Parameters"]
    if not starts:
        if not expected:
            return {}
        missing = ", ".join(f"{kind} {name!r}" for name in sorted(expected))
        raise ValueError(f"{label}: missing Parameters section ({missing})")
    descriptions = {}
    names = []
    text = []

    def save():
        for name in names:
            if name in descriptions:
                raise ValueError(f"{label}: duplicate Parameters {kind} {name!r}")
            descriptions[name] = " ".join(text)

    for start in starts:
        end = next((i for i in sections if i > start), len(lines))
        names, text = [], []
        for line in lines[start + 2:end]:
            if line and not line.startswith(" "):
                save()
                names = [name.strip() for name in line.split(":", 1)[0].split(",")]
                text = []
            elif line.strip():
                text.append(line.strip())
        save()
    problems = [f"missing Parameters {kind} {name!r}"
                for name in sorted(expected - descriptions.keys())]
    problems += [f"unknown Parameters {kind} {name!r}"
                 for name in sorted(descriptions.keys() - expected)]
    problems += [f"empty Parameters description for {kind} {name!r}"
                 for name in sorted(expected & descriptions.keys()) if not descriptions[name]]
    if problems:
        raise ValueError(f"{label}: " + "; ".join(problems))
    return descriptions


def _api_objects(api_dir):
    """Resolve every autosummary entry, following every currentmodule on a page.

    Class templates render :members:, so include public methods defined on the
    class itself, including class/static methods, but not properties or inherited
    methods. Exceptions (including warnings), enums and constants are exempt.
    """
    objects = {}
    for path in sorted(Path(api_dir).glob("*.rst")):
        module = None
        lines = path.read_text().splitlines()
        i = 0
        while i < len(lines):
            match = re.match(r"\.\. (?:current)?module:: (\S+)", lines[i])
            if match:
                module = match.group(1)
            if not lines[i].startswith(".. autosummary::"):
                i += 1
                continue
            i += 1
            while i < len(lines) and (lines[i].startswith("   ") or not lines[i].strip()):
                name = lines[i].strip()
                i += 1
                if not name or name.startswith(":"):
                    continue
                obj = importlib.import_module(module)
                for part in name.split("."):
                    obj = getattr(obj, part)
                if inspect.isclass(obj) and issubclass(obj, (BaseException, enum.Enum)):
                    continue
                qualified = f"{module}.{name}"
                if inspect.isclass(obj):
                    objects[qualified] = obj
                    for member_name, member in sorted(vars(obj).items()):
                        if member_name.startswith("_"):
                            continue
                        func = (member.__func__ if isinstance(member, (classmethod, staticmethod))
                                else member)
                        if inspect.isfunction(func):
                            objects[f"{qualified}.{member_name}"] = func
                elif callable(obj):
                    objects[qualified] = obj
    return objects


def _check_api_object(name, obj):
    """Check the original docstring, before Napoleon transforms it."""
    try:
        _parameter_descriptions(obj)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error
    if inspect.isclass(obj):
        return
    annotation = inspect.signature(obj).return_annotation
    if annotation in (inspect.Signature.empty, None, type(None), "None", "NoneType"):
        return
    if not re.search(r"^(?:Returns|Yields)\n-{3,}\s*$",
                     inspect.cleandoc(obj.__doc__ or ""), re.MULTILINE):
        raise ValueError(f"{name}: missing Returns or Yields section for return annotation "
                         f"{annotation!r}")


def _check_api_docs(app):
    """Validate all API targets on every build, including incremental builds."""
    from sphinx.util import logging

    logger = logging.getLogger(__name__)
    for name, obj in _api_objects(Path(app.srcdir) / "api").items():
        try:
            _check_api_object(name, obj)
        except ValueError as error:
            logger.warning("API parameter documentation: %s", error)


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
    """One row per field of a config dataclass: name, default and description."""

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
        try:
            descriptions = _parameter_descriptions(config)
        except ValueError as error:
            raise self.error(f"config-options: {error}")
        excluded = {name.strip() for name in self.options.get("exclude", "").split(",") if name.strip()}
        rows = [[field.name, _default(field), descriptions.get(field.name, "")]
                for field in dataclasses.fields(config) if field.name not in excluded]
        return [_table(["Option", "Default", "Description"], rows, literal_columns={0, 1})]


def setup(app):
    app.connect("builder-inited", _check_api_docs)
    app.add_directive("method-table", MethodTable)
    app.add_directive("config-options", ConfigOptions)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
