"""Every public config and generated option-table target documents all fields."""
import ast
from dataclasses import is_dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.preprocessing]
ROOT = Path(__file__).resolve().parents[3]


def _config_classes():
    """Discover public Config classes and table targets without a fixed allowlist."""
    classes = set()
    package = ROOT / "src/python/starfinder"
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text())
        names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)
                 and not node.name.startswith("_") and node.name.endswith("Config")]
        if not names:
            continue
        parts = path.relative_to(package.parent).with_suffix("").parts
        module = import_module(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
        classes.update(getattr(module, name) for name in names
                       if is_dataclass(getattr(module, name)))
    pattern = re.compile(r"(?:\{config-options\}|\.\. config-options::)\s+([\w.]+)")
    for path in sorted((ROOT / "docs").rglob("*")):
        if path.suffix not in (".md", ".rst"):
            continue
        for target in pattern.findall(path.read_text()):
            module, _, name = target.rpartition(".")
            classes.add(getattr(import_module(module), name))
    return sorted(classes, key=lambda cls: (cls.__module__, cls.__qualname__))


def test_config_dataclasses_have_complete_parameters():
    pytest.importorskip("docutils", reason="documentation extra required")
    spec = spec_from_file_location("config_parameters_tables", ROOT / "docs/_ext/method_tables.py")
    tables = module_from_spec(spec)
    spec.loader.exec_module(tables)
    configs = _config_classes()
    assert configs, "No config dataclasses discovered"
    problems = []
    for config in configs:
        try:
            tables._parameter_descriptions(config)
        except ValueError as error:
            problems.append(f"{config.__module__}.{error}")
    assert not problems, "\n".join(problems)
