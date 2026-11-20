"""The parameters of the Workflow overview's cards (design/specs/report-algorithms.md, "Workflow overview").

A card opens on the parameters of its step. They are read from ``workflow/resolved.json`` (:func:`read_resolved`), the
file whose hash each ``run.json`` ``workflow_config`` holds, so that the page has no second parameter table of its own.
The resolved configuration's ``api`` writes every block as the dataclass the tasks ran with, each with its ``type``;
:func:`differing` lists the values of such a node that differ from the defaults of that dataclass, in the order of its
fields, and the page shows these first, the whole block behind "all parameters".
"""
from __future__ import annotations

import dataclasses
import json
import sys
from functools import lru_cache

from ._values import Unavailable

#: The modules that define the configuration dataclasses of the resolved ``api``, imported to find a ``type`` by name;
#: a type none of them (or any other imported ``starfinder`` module) holds is shown in full, as no defaults are known.
_MODULES = ("starfinder.preprocessing", "starfinder.registration", "starfinder.spot_finding", "starfinder.readout",
            "starfinder.segmentation", "starfinder.assignment", "starfinder.stitching", "starfinder.assembly",
            "starfinder.export", "starfinder.dataset", "starfinder.dataset.config")
_MISSING = object()
#: Where each card's parameters are in ``configuration.api``: the keys to follow, from the ``api``.
PIPELINE = ("processing", "pipeline")
CARD_NODES = {
    "Preprocessing": PIPELINE + ("preprocessing", "steps"),
    "Registration": PIPELINE + ("registration",),
    "Post-registration preprocessing": PIPELINE + ("preprocessing", "post_registration"),
    "Spot finding": PIPELINE + ("spot_finding",),
    "Read extraction": PIPELINE + ("extraction",),
    "Decoding": PIPELINE + ("decoding",),
    "Scoring": PIPELINE + ("scoring",),
    "Deduplication": PIPELINE + ("deduplication",),
    "Read QC filter": PIPELINE + ("filtering",),
    "Segmentation": ("segmentation", "plan"),
    "Assignment": ("assignment", "config"),
    "Stitching": ("stitching",),
    "Export": ("export", "config"),
}


class Resolved:
    """The resolved configuration a report read: its ``path`` as the run records give it, the ``sha256`` it has now
    (``sha256:<hex>``), the hashes the FOVs ran under (``recorded``), and the configuration."""

    def __init__(self, path, sha256, recorded, configuration):
        self.path, self.sha256, self.recorded, self.configuration = path, sha256, recorded, configuration

    @property
    def changed(self) -> bool:
        """True when a FOV ran under another hash than the file has now (a rerun of a task that is not a FOV's
        changes the file and leaves the hash the FOVs wrote)."""
        return any(digest != self.sha256 for digest in self.recorded)

    def node(self, card):
        """The part of ``configuration.api`` that ``card`` shows, or None when the run has none."""
        found = self.configuration.get("api")
        for key in CARD_NODES[card]:
            found = found.get(key) if isinstance(found, dict) else None
        return found


def read_resolved(inputs, root, records):
    """The resolved configuration of the run: a :class:`Resolved`, None when no run record names one (the Files table
    says so), or :class:`Unavailable` (``input_missing``; ``not_recorded`` when the file is not a resolved
    configuration with an ``api``). The path is the first ``workflow_config.path`` the records hold.

    Parameters
    ----------
    inputs : _Inputs
        The report's reader, which lists the file among the inputs.
    root : Path
        The output root, which a relative ``workflow_config.path`` is joined to.
    records : list of dict
        The run records of the FOVs that have one.

    Returns
    -------
    Resolved or Unavailable or None
        As above.
    """
    from pathlib import Path
    configs = [r["workflow_config"] for r in records if isinstance(r.get("workflow_config"), dict)]
    configs = [c for c in configs if c.get("path")]
    if not configs:
        return None
    path = Path(configs[0]["path"])
    path = path if path.is_absolute() else Path(root) / path
    data = inputs.json(path, "workflow_config", strict=False)
    if isinstance(data, Unavailable):
        return data
    configuration = data.get("configuration") if isinstance(data, dict) else None
    if not isinstance(configuration, dict) or not isinstance(configuration.get("api"), dict):
        return Unavailable("not_recorded", f"{inputs.relative(path)} is not a resolved configuration with an api")
    digest = "sha256:" + inputs.entries[inputs.relative(path)]["sha256"]
    recorded = list(dict.fromkeys(c.get("sha256") for c in configs))
    return Resolved(configs[0]["path"], digest, recorded, configuration)


@lru_cache(maxsize=None)
def _classes():
    import importlib
    found = {}
    for name in _MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        for key, value in vars(module).items():
            if isinstance(value, type) and dataclasses.is_dataclass(value):
                found.setdefault(key, value)
    return found


def dataclass_named(name):
    """The dataclass a ``type`` of the resolved ``api`` names, or None.

    Parameters
    ----------
    name : str
        The ``type`` of a node.

    Returns
    -------
    type or None
        The dataclass.
    """
    found = _classes().get(name)
    if found is None:
        for module in [m for key, m in list(sys.modules.items()) if key.startswith("starfinder.") and m is not None]:
            value = vars(module).get(name)
            if isinstance(value, type) and dataclasses.is_dataclass(value):
                return value
    return found


def field_order(name) -> list:
    """The field names of the dataclass ``name`` in their order; empty for an unknown type."""
    cls = dataclass_named(name)
    return [] if cls is None else [field.name for field in dataclasses.fields(cls)]


def defaults(name) -> dict | None:
    """The defaults of the dataclass ``name`` in the form of the resolved configuration (``_shape.plain``), per field
    that has one; None for an unknown type.

    Parameters
    ----------
    name : str
        The ``type`` of a node.

    Returns
    -------
    dict or None
        Field name to default.
    """
    from starfinder.workflow._shape import plain
    cls = dataclass_named(name)
    if cls is None:
        return None
    out = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            out[field.name] = plain(field.default)
        elif field.default_factory is not dataclasses.MISSING:
            out[field.name] = plain(field.default_factory())
    return out


def typed(value) -> bool:
    """A dataclass of the resolved ``api``: an object with its ``type``."""
    return isinstance(value, dict) and isinstance(value.get("type"), str)


def differing(node, prefix=()) -> list:
    """``(path, value)`` for every value of the dataclass ``node`` that is not its field's default, in the order of
    the dataclass's fields (the resolved configuration writes its keys sorted); a nested dataclass or a list of them contributes its own differing values under ``path`` (``steps[0].method``).
    A field without a default is listed, and every field of a dataclass of unknown type.

    Parameters
    ----------
    node : dict
        A dataclass of the resolved ``api`` (an object with its ``type``).
    prefix : tuple of str
        The path of ``node`` in the block being listed.

    Returns
    -------
    list of tuple
        ``(path, value)`` pairs, ``path`` a tuple of keys.
    """
    known = defaults(node["type"])
    order = {name: index for index, name in enumerate(field_order(node["type"]))}
    rows = []
    for name, value in sorted(node.items(), key=lambda item: order.get(item[0], len(order))):
        if name == "type":
            continue
        default = _MISSING if known is None else known.get(name, _MISSING)
        if default is not _MISSING and value == default:
            continue
        path = prefix + (name,)
        if typed(value):
            rows += differing(value, path)
        elif isinstance(value, list) and value and all(typed(item) for item in value):
            for index, item in enumerate(value):
                rows += differing(item, prefix + (f"{name}[{index}]",))
        else:
            rows.append((path, value))
    return rows


def shown(value) -> str:
    """A value as the panels write it: a string as it is, anything else as compact JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def path_text(path) -> str:
    """A path of :func:`differing` as ``a.b[0].c``."""
    return ".".join(path)
