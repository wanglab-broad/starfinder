"""Generate workflow key tables from the checkout's JSON Schema.

An optional local JSON pointer selects an object. Referenced objects get their
own tables; inline objects use dotted paths. Defaults are schema defaults only.
"""
from __future__ import annotations

import json
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
import yaml

from method_tables import _table


def _resolve(schema, pointer):
    if not pointer.startswith("#/"):
        raise ValueError(f"expected a local schema pointer, got {pointer!r}")
    value = schema
    for part in pointer[2:].split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    return value


def _effective(schema, node):
    """Resolve metadata and unconditional inheritance, without inventing defaults."""
    result = {}
    if "$ref" in node:
        result.update(_effective(schema, _resolve(schema, node["$ref"])))
    for branch in node.get("allOf", ()):
        for key, value in _effective(schema, branch).items():
            result[key] = {**result.get(key, {}), **value} if key == "properties" else value
    for key, value in node.items():
        result[key] = {**result.get(key, {}), **value} if key == "properties" else value
    return result


def _kind(node):
    if "$ref" in node:
        return node["$ref"].rsplit("/", 1)[-1]
    if "const" in node:
        return json.dumps(node["const"], ensure_ascii=False)
    if "enum" in node:
        return " | ".join(json.dumps(v, ensure_ascii=False) for v in node["enum"])
    for choice in ("oneOf", "anyOf"):
        if choice in node:
            return " | ".join(_kind(branch) for branch in node[choice])
    kind = node.get("type", "unspecified")
    if isinstance(kind, list):
        kind = " | ".join(kind)
    if "items" in node and isinstance(node["items"], dict):
        kind += f" of {_kind(node['items'])}"
    return kind


def schema_rows(schema, node, prefix=""):
    """Expand fields of inline objects, leaving referenced objects to their own tables."""
    rows = []
    for name, field in _effective(schema, node).get("properties", {}).items():
        path = f"{prefix}.{name}" if prefix else name
        effective = _effective(schema, field)
        default = json.dumps(effective["default"], ensure_ascii=False) if "default" in effective else "—"
        rows.append([path, _kind(field), default, " ".join(effective.get("description", "").split())])
        if "$ref" not in field:
            rows.extend(schema_rows(schema, field, path))
            if isinstance(field.get("items"), dict) and "$ref" not in field["items"]:
                rows.extend(schema_rows(schema, field["items"], path + "[]"))
    return rows


class WorkflowConfig(Directive):
    """A schema object's keys, types, defaults and descriptions."""

    optional_arguments = 1
    has_content = False

    def run(self):
        env = self.state.document.settings.env
        source = Path(env.srcdir).parent / "workflow/schemas/config.schema.yaml"
        env.note_dependency(str(source))
        try:
            schema = yaml.safe_load(source.read_text())
            node = _resolve(schema, self.arguments[0]) if self.arguments else schema
            if self.arguments == ["#/$defs"]:
                result = []
                for name, definition in node.items():
                    section = nodes.section(ids=["schema-" + name.replace("_", "-")])
                    section += nodes.title(text=name)
                    if definition.get("description"):
                        section += nodes.paragraph(text=" ".join(definition["description"].split()))
                    rows = schema_rows(schema, definition)
                    if not rows:
                        rows = [["value", _kind(definition),
                                 json.dumps(definition["default"]) if "default" in definition else "—", ""]]
                    section += _table(["Key", "Type / values", "Schema default", "Description"], rows,
                                      literal_columns={0, 1, 2})
                    result.append(section)
                return result
            rows = schema_rows(schema, node)
        except (OSError, KeyError, ValueError, yaml.YAMLError) as error:
            raise self.error(f"workflow-config: {error}")
        return [_table(["Key", "Type / values", "Schema default", "Description"], rows, literal_columns={0, 1, 2})]


def setup(app):
    app.add_directive("workflow-config", WorkflowConfig)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
