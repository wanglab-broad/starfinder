"""Class pages that describe each parameter once, after scanpy's class pages.

A dataclass's ``Parameters`` section already gives each field its type, default and
description, so the class page does not repeat a field named there as an attribute
entry. Which entries drop comes from ``dataclasses.fields`` and the docstring's own
``Parameters`` names; no class is listed by name. Fields whose names start with ``_``
leave the signature. Before the member entries, the page shows an ``Attributes``
table (properties and other attributes that ``Parameters`` does not name; left out
when empty) and a ``Methods`` table, one line per member. A cross-reference to a
dropped field entry resolves to the class.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import re

from sphinx.ext.autodoc import ALL, ClassDocumenter, MethodDocumenter


def _parameter_names(cls):
    """Names in the numpydoc ``Parameters`` section(s) of a class docstring."""
    lines = inspect.cleandoc(cls.__doc__ or "").splitlines()
    sections = [i for i in range(len(lines) - 1)
                if lines[i] and not lines[i].startswith(" ") and re.fullmatch(r"-{3,}", lines[i + 1])]
    names = set()
    for start in (i for i in sections if lines[i] == "Parameters"):
        end = next((i for i in sections if i > start), len(lines))
        for line in lines[start + 2:end]:
            if line and not line.startswith(" "):
                names.update(name.strip() for name in line.split(":", 1)[0].split(","))
    return names


def described_fields(cls):
    """The dataclass fields of ``cls`` that its ``Parameters`` section describes."""
    if not (inspect.isclass(cls) and dataclasses.is_dataclass(cls)):
        return set()
    return {field.name for field in dataclasses.fields(cls)} & _parameter_names(cls)


class ClassPageDocumenter(ClassDocumenter):
    """``autoclass`` with the class-page rules of this module."""

    def _get_signature(self):
        owner, name, signature = super()._get_signature()
        if signature is not None:
            signature = signature.replace(parameters=[
                parameter for parameter in signature.parameters.values()
                if not parameter.name.startswith("_")])
        return owner, name, signature

    def filter_members(self, members, want_all):
        described = described_fields(self.object)
        return [member for member in super().filter_members(members, want_all)
                if member[0] not in described]

    def document_members(self, all_members=False):
        want_all = all_members or self.options.inherited_members or self.options.members is ALL
        _, members = self.get_object_members(want_all)
        attributes, methods = [], []
        for name, member, isattr in self.filter_members(members, want_all):
            classes = [cls for cls in self.documenters.values()
                       if cls.can_document_member(member, name, isattr, self)]
            if not classes:
                continue
            documenter = max(classes, key=lambda cls: cls.priority)
            (methods if issubclass(documenter, MethodDocumenter) else attributes).append(name)
        source = self.get_sourcename()
        # Relative to the current module, as autosummary resolves names there.
        qualified = ".".join(self.objpath)
        for title, names in (("Attributes", attributes), ("Methods", methods)):
            if not names:
                continue
            self.add_line(f".. rubric:: {title}", source)
            self.add_line("", source)
            self.add_line(".. autosummary::", source)
            self.add_line("", source)
            for name in sorted(names, key=str.casefold):
                self.add_line(f"   ~{qualified}.{name}", source)
            self.add_line("", source)
        super().document_members(all_members)


def _field_reference(app, env, node, contnode):
    """Resolve a reference to a dropped field entry to its class."""
    if node.get("refdomain") != "py":
        return None
    target = node["reftarget"].lstrip("~.")
    module, cls = node.get("py:module"), node.get("py:class")
    candidates = [target]
    if module:
        candidates += [f"{module}.{target}"] + ([f"{module}.{cls}.{target}"] if cls else [])
    domain = env.get_domain("py")
    for candidate in candidates:
        owner, _, field = candidate.rpartition(".")
        module_name, _, class_name = owner.rpartition(".")
        try:
            obj = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError, ValueError):
            continue
        if field in described_fields(obj):
            return domain.resolve_xref(env, node.get("refdoc", env.docname), app.builder,
                                       "class", owner, node, contnode)
    return None


def setup(app):
    app.setup_extension("sphinx.ext.autodoc")
    app.setup_extension("sphinx.ext.autosummary")
    app.add_autodocumenter(ClassPageDocumenter, override=True)
    app.connect("missing-reference", _field_reference)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
