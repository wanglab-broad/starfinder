"""Generate the workflow's reference tables from the code, and the workflow contract's migration table.

``workflow-config`` renders the MATLAB dialect's keys from ``workflow/schemas/config.schema.yaml``.
An optional local JSON pointer selects an object. Referenced objects get their
own tables; inline objects use dotted paths. Defaults are schema defaults only.
``workflow-migration`` renders the migration table of design/specs/orchestration.md
from the record's own labelled table, so the guide's table is generated from it.

The Python workflow's tables (docs/workflow/reference.md) are read from the package:
``workflow-commands`` from the parser of ``starfinder workflow``; ``workflow-tasks`` from the
manifests that the plan writes for a few configurations over a placeholder input tree;
``workflow-keys`` from the key lists, the schema and the translators of a block, with the
defaults that the plan writes into the resolved configuration; ``workflow-left-keys`` from the
keys the plan refuses; ``workflow-profile`` from a profile's ``config.yaml``.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import json
import tempfile
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
import yaml

from method_tables import _parameter_descriptions, _table


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


def labelled_table(text, label):
    """(header, rows) of the Markdown table right after the MyST label ``(label)=`` of a design record."""
    lines = text.splitlines()
    found = [i for i, line in enumerate(lines) if line.strip() == f"({label})="]
    if len(found) != 1:
        raise ValueError(f"({label})= occurs {len(found)} times")
    start = found[0] + 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    if end - start < 3:
        raise ValueError(f"no table follows ({label})=")
    cells = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines[start:end]]
    # Single backticks of the record become the inline code of _table's cells.
    convert = lambda text: text.replace("`", "``")  # noqa: E731
    return [convert(c) for c in cells[0]], [[convert(c) for c in row] for row in cells[2:]]


class WorkflowMigration(Directive):
    """The migration table of the workflow contract, generated from the design record's own table
    (design/specs/orchestration.md, ``(orchestration-migration)=``, or the label given)."""

    optional_arguments = 1
    has_content = False

    def run(self):
        env = self.state.document.settings.env
        source = Path(env.srcdir).parent / "design/specs/orchestration.md"
        env.note_dependency(str(source))
        try:
            header, rows = labelled_table(source.read_text(encoding="utf-8"),
                                          self.arguments[0] if self.arguments else "orchestration-migration")
        except (OSError, ValueError) as error:
            raise self.error(f"workflow-migration: {error}")
        return [_table(header, rows, literal_columns=set())]


# --- the Python workflow: commands, task kinds and keys, read from the package ------------------------------------

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _value(value):
    return json.dumps(value, ensure_ascii=False)


def _described(content, names, what):
    """{name: description} of a directive's content lines ``name: description``, one for each of ``names``."""
    given = {}
    for line in content:
        if line.strip():
            name, _, text = line.partition(":")
            if not text.strip():
                raise ValueError(f"the line {line!r} is not 'name: description'")
            given[name.strip()] = text.strip()
    unknown = sorted(set(given) - set(names))
    missing = [name for name in names if name not in given]
    if unknown or missing:
        raise ValueError(f"{what}: " + "; ".join(
            ([f"no {what} named {', '.join(unknown)}"] if unknown else [])
            + ([f"describe {', '.join(missing)}"] if missing else [])))
    return given


def _subcommands():
    """{command: parser} and {command: help} of ``starfinder workflow``, built by the package's own add_parser."""
    from starfinder.workflow._cli import add_parser
    workflow = add_parser(argparse.ArgumentParser(prog="starfinder").add_subparsers())
    group = next(a for a in workflow._actions if isinstance(a, argparse._SubParsersAction))
    return dict(group.choices), {a.dest: a.help for a in group._choices_actions}


def _signature(action):
    return (tuple(action.option_strings), action.required, action.default, tuple(action.choices or ()),
            action.nargs, action.help)


def command_rows(content=()):
    """(rows of the commands, rows of their arguments). An argument that every task command takes alike is one row;
    an argument without help text takes its description from ``content``."""
    from starfinder.workflow._shape import COMMANDS
    parsers, helps = _subcommands()
    tasks = [name for name in parsers if name in set(COMMANDS.values())]
    commands = [[f"starfinder workflow {name}", helps[name] or ""] for name in parsers]
    owners = {}
    for name, parser in parsers.items():
        for action in parser._actions:
            if not isinstance(action, argparse._HelpAction):
                owners.setdefault(_signature(action), []).append(name)
    undescribed = sorted({s[0][-1] for s in owners if not s[5]})
    extra = _described(content, undescribed, "argument")
    arguments = []
    for signature, names in owners.items():
        flags, required, default, choices, nargs, text = signature
        where = "every task command" if names == tasks else ", ".join(names)
        value = ("required" if required else "flag" if nargs == 0 else
                 "—" if default is None else str(default))
        arguments.append([", ".join(flags), where, value, " | ".join(map(str, choices)) or "—",
                          text or extra[flags[-1]]])
    return commands, arguments


def _placeholder(root, config):
    """The placeholder input tree of docs/examples/prepare_workflow.py for ``config`` under ``root``."""
    import runpy
    runpy.run_path(str(EXAMPLES / "prepare_workflow.py"))["placeholder_inputs"](config, root)


def _probes(save_dir):
    """Configurations whose plans hold every task kind: the full example, and the full example in mode tiled with a
    BigStitcher placement and segmentation runs by a method instead of imported labels."""
    full = yaml.safe_load((EXAMPLES / "workflow-full.yaml").read_text())
    tiled = copy.deepcopy(full)
    tiled["processing"]["mode"] = "tiled"
    tiled["stitching"] = {"method": "bigstitcher", "bigstitcher_save_dir": str(save_dir),
                          "positions": full["stitching"]["positions"]}
    tiled["segmentation"]["runs"] = [
        {"name": "nucleus", "target": "nucleus", "method": "stardist", "model_path": str(save_dir), "scale": 1.0,
         "inputs": [{"role": "nuclear", "round": "reference_stain", "channel": "DAPI"}]},
        {"name": "cell", "target": "cell", "seeds": "nucleus", "method": "seeded_watershed",
         "inputs": [{"role": "amplicon", "reference_merged": True}]}]
    return [full, tiled]


def task_rows():
    """One row per task kind: its command, what one task covers, the blocks it reads, its default threads and its
    declared outputs, the rule group's output templates of the plans of ``_probes`` (relative to the output root)."""
    from starfinder.workflow import plan
    from starfinder.workflow._shape import COMMANDS, EXECUTION_DEFAULTS, ID_FIELDS, KIND_THREADS, KINDS, TASK_BLOCKS
    outputs, identities = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_dir = tmp / "bigstitcher"
        for i, config in enumerate(_probes(save_dir)):
            config.update(root_input_path=str(tmp / f"input{i}"), root_output_path=str(tmp / f"output{i}"))
            _placeholder(tmp / f"input{i}", config)
            path = tmp / f"config{i}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            manifest = plan(path)
            groups = {group["name"]: group for group in manifest["groups"]}
            # A run, an assignment name and another round are literal parts of the templates: written as such.
            names = {f"/{e['round_name']}/": "/<round>/" for e in config.get("additional_round") or []}
            for task in manifest["tasks"]:
                kind, identity, group = task["kind"], task["identity"], groups[task["group"]]
                identities.setdefault(kind, [k for k in ID_FIELDS if k in identity])
                literal = dict(names, **{f"/{identity[k]}/": f"/<{k}>/" for k in ("run", "name") if k in identity})
                for template in group["outputs"]:
                    text = template.replace(str(save_dir), "<bigstitcher_save_dir>")
                    for value, name in literal.items():
                        text = text.replace(value, name)
                    temporary = " (temporary)" if template in group["temporary"] else ""
                    outputs.setdefault(kind, {})[f"``{text}``{temporary}"] = None
    missing = [kind for kind in KINDS if kind not in outputs]
    if missing:
        raise ValueError(f"no probe plans the task kinds {', '.join(missing)}")
    return [[kind, COMMANDS[kind], ", ".join(identities[kind]) or "the run", ", ".join(TASK_BLOCKS[kind]),
             str(KIND_THREADS.get(kind, EXECUTION_DEFAULTS["threads"])), ", ".join(outputs[kind])] for kind in KINDS]


@functools.lru_cache(maxsize=None)
def _defaults_probe():
    """(given, resolved) of the minimal example without its execution section, and of the minimal example with
    every other block present and the fewest keys its translators accept: the keys the plan adds are its defaults."""
    from starfinder.workflow._plan import resolve
    minimal = yaml.safe_load((EXAMPLES / "workflow-minimal.yaml").read_text())
    minimal.pop("execution", None)
    blocks = dict(minimal, reference_stains=[{"channel": "ch04", "name": "DAPI"}],
                  additional_round=[{"round_name": "other", "channel_order": ["ch04"]}], morphology={},
                  segmentation={"runs": [{"name": "cell", "target": "cell", "method": "import",
                                          "path": "labels/{fov_id}.tif"}]},
                  assignment={}, stitching={"bigstitcher_save_dir": "/save", "positions": {"source": "table",
                                                                                          "table": "positions.csv"}},
                  assembly={}, export={"route": "per_fov"}, report={})
    return [(given, resolve(copy.deepcopy(given), None)["configuration"]) for given in (minimal, blocks)]


#: Defaults the plan derives from other keys (starfinder/workflow/_plan.py, resolve), written in words.
DERIVED = {("morphology", "rounds"): "every additional_round",
           # check_tiled writes the minimum in place of null (_plan.py, check_tiled).
           ("processing.tiled", "halo"): "the minimum: the largest footprint of the tile's operations"}


def _schema_node(schema, block):
    node = schema
    for part in [] if block == "dataset" else block.split("."):
        node = node["properties"][part]
    return node


def block_keys(block):
    """[(key, description or None)] of a block: its keys as the package lists them, without the keys the plan
    refuses; a key that a config dataclass or an API call defines takes that object's description."""
    from starfinder.dataset import FOV
    from starfinder.dataset import workflow as translators
    from starfinder.workflow import _shape
    from starfinder.workflow._plan import PlanRefused, check_shape, schema

    def described(obj, names=None):
        descriptions = _parameter_descriptions(obj)
        names = names or [f.name for f in dataclasses.fields(obj) if f.init]
        return [(name, f"{descriptions[name]} ({obj.__qualname__})") for name in names]

    properties = _schema_node(schema(), block).get("properties", {}) if block != "dataset" else {}
    if block == "dataset":
        keys = [(k, None) for k in _shape.DATASET_KEYS]
    elif block == "processing":
        keys = [(k, properties.get(k, {}).get("description")) for k in _shape.PROCESSING_KEYS]
    elif block == "assignment":
        from starfinder.assignment import AssignmentConfig
        keys = described(AssignmentConfig) + described(FOV.assign_molecules, list(translators._ASSIGNMENT_CALL_KEYS))
        keys.append(("previews", None))
    elif block == "stitching":
        keys = [(k, None) for k in (*translators._STITCHING_KEYS, *properties)]
    elif block == "assembly":
        from starfinder.assembly import AssemblyConfig
        keys = described(AssemblyConfig) + [(k, None) for k in translators._ASSEMBLY_CALL_KEYS]
    elif block == "export":
        from starfinder import export
        classes = {None: export.ExportConfig, "fusion": export.FusionConfig, "labels": export.LabelConfig,
                   "spatialdata": export.SpatialDataExportConfig}
        keys = []
        for key, (part, field) in translators._export_keys().items():
            description = _parameter_descriptions(classes[part])[field]
            keys.append((key, f"{description} ({classes[part].__qualname__})"))
        keys.append(("previews", None))
    elif block == "report":
        from starfinder.report import ReportConfig
        keys = described(ReportConfig)
    elif block == "execution":
        keys = [(k, None) for k in _shape.EXECUTION_KEYS]
    elif block == "execution.defaults":
        keys = [(k, None) for k in _shape.RESOURCE_KEYS]
    else:
        keys = [(k, node.get("description")) for k, node in properties.items()]
    # A key both a config class and an API call name (assignment's correspondence) is the class's.
    first = {}
    for key, description in keys:
        first.setdefault(key, description)
    keys = list(first.items())
    if block in _shape.BLOCKS:
        def refused(key):
            try:
                check_shape({block: {key: None}})
            except PlanRefused:
                return True
            return False
        keys = [(k, d) for k, d in keys if not refused(k)]
    return keys


def key_rows(block, content=()):
    """One row per key of ``block``: key, type (the Python schema's, where it names one), the default the plan writes
    and the description (the defining object's, the schema's, or the directive's content)."""
    from starfinder.workflow._plan import schema
    from starfinder.workflow._shape import TILED_DEFAULTS
    keys = block_keys(block)
    extra = _described(content, [k for k, d in keys if d is None], f"{block} key")
    (minimal, resolved_minimal), (given, resolved) = _defaults_probe()
    if block in ("dataset", "execution", "execution.defaults"):
        given, resolved = minimal, resolved_minimal
    node = _schema_node(schema(), block)
    rows = []
    for key, description in keys:
        if (block, key) in DERIVED or (block.split(".")[0], key) in DERIVED:
            default = DERIVED.get((block, key)) or DERIVED[(block.split(".")[0], key)]
        elif block == "processing.tiled":
            default = _value(TILED_DEFAULTS[key])
        else:
            g, r = given, resolved
            for part in [] if block == "dataset" else block.split("."):
                g, r = g.get(part) or {}, r.get(part) or {}
            default = _value(r[key]) if key in r and key not in g else "—"
        field = node.get("properties", {}).get(key)
        rows.append([key, _kind(field) if field else "—", default, description or extra[key]])
    return rows


def left_rows():
    """The keys that left the Python configuration, with where each went, as the plan names them."""
    from starfinder.workflow._shape import LEFT_KEYS, PROCESSING_LEFT
    return [[key, where] for key, where in LEFT_KEYS.items()] + \
        [[f"processing.{key}", where] for key, where in PROCESSING_LEFT.items()]


def _generated(directive, name, rows, header, literal):
    try:
        return [_table(header, rows(), literal_columns=literal)]
    except (ImportError, KeyError, ValueError, OSError) as error:
        raise directive.error(f"{name}: {error}")


class WorkflowCommands(Directive):
    """The commands of ``starfinder workflow`` and their arguments, from the package's parser. The content
    describes, one ``--flag: description`` line each, the arguments that have no help text."""

    has_content = True

    def run(self):
        try:
            commands, arguments = command_rows(self.content)
        except (ImportError, ValueError) as error:
            raise self.error(f"workflow-commands: {error}")
        return [_table(["Command", "What it does"], commands, literal_columns={0}),
                _table(["Argument", "Of", "Default", "Values", "Description"], arguments, literal_columns={0, 2, 3})]


class WorkflowTasks(Directive):
    """The task kinds with their commands, the blocks they read and their declared outputs, from the plan."""

    has_content = False

    def run(self):
        for name in ("workflow-full.yaml", "prepare_workflow.py"):
            self.state.document.settings.env.note_dependency(str(EXAMPLES / name))
        return _generated(self, "workflow-tasks", task_rows,
                          ["Kind", "Command", "One task per", "Blocks it reads", "Threads", "Declared outputs"],
                          {0, 1})


class WorkflowKeys(Directive):
    """The keys of one block of the Python configuration (``dataset`` for the top-level keys), their types, the
    defaults the plan writes and their descriptions. The content describes, one ``key: description`` line each, the
    keys that no config class, API call or schema entry describes."""

    required_arguments = 1
    has_content = True

    def run(self):
        return _generated(self, "workflow-keys", lambda: key_rows(self.arguments[0], self.content),
                          ["Key", "Type / values", "Plan default", "Description"], {0, 1, 2})


class WorkflowLeftKeys(Directive):
    """The keys that left the Python configuration and where each went, as the plan's refusal names them."""

    has_content = False

    def run(self):
        return _generated(self, "workflow-left-keys", left_rows, ["Key", "Where it went"], {0})


class WorkflowProfile(Directive):
    """The settings of a Snakemake profile's ``config.yaml`` (the argument: the profile's folder)."""

    required_arguments = 1
    has_content = False

    def run(self):
        env = self.state.document.settings.env
        source = Path(env.srcdir).parent / self.arguments[0] / "config.yaml"
        env.note_dependency(str(source))

        def rows():
            settings = yaml.safe_load(source.read_text())
            return [[key, value if isinstance(value, str) else _value(value)] for key, value in settings.items()]
        return _generated(self, "workflow-profile", rows, ["Setting", "Value"], {0, 1})


def setup(app):
    app.add_directive("workflow-config", WorkflowConfig)
    app.add_directive("workflow-migration", WorkflowMigration)
    app.add_directive("workflow-commands", WorkflowCommands)
    app.add_directive("workflow-tasks", WorkflowTasks)
    app.add_directive("workflow-keys", WorkflowKeys)
    app.add_directive("workflow-left-keys", WorkflowLeftKeys)
    app.add_directive("workflow-profile", WorkflowProfile)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
