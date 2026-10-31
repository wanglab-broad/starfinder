"""Public configs, option-table targets and API objects document their parameters."""
import ast
from dataclasses import is_dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.preprocessing]
ROOT = Path(__file__).resolve().parents[3]
GENERATED_DOCS = (("api", "generated"), ("_build",))


def _config_option_targets(docs):
    """Return config-options targets named by hand-written pages, skipping build outputs."""
    pattern = re.compile(r"(?:\{config-options\}|\.\. config-options::)\s+([\w.]+)")
    targets = []
    for path in sorted(docs.rglob("*")):
        parts = path.relative_to(docs).parts
        if path.suffix not in (".md", ".rst") or any(
                parts[:len(prefix)] == prefix for prefix in GENERATED_DOCS):
            continue
        targets.extend(pattern.findall(path.read_text()))
    return targets


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
    for target in _config_option_targets(ROOT / "docs"):
        module, _, name = target.rpartition(".")
        classes.add(getattr(import_module(module), name))
    return sorted(classes, key=lambda cls: (cls.__module__, cls.__qualname__))


def test_config_option_scan_skips_generated_pages(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    (docs / "api/generated").mkdir(parents=True)
    (docs / "_build/html").mkdir(parents=True)
    (docs / "guide").mkdir()
    (docs / "page.md").write_text("```{config-options} pkg.PageConfig\n```\n")
    (docs / "guide/page.rst").write_text(".. config-options:: pkg.GuideConfig\n")
    (docs / "api/page.md").write_text("```{config-options} pkg.ApiConfig\n```\n")
    (docs / "api/generated/pkg.md").write_text("```{config-options} pkg.GeneratedConfig\n```\n")
    (docs / "_build/html/page.rst").write_text(".. config-options:: pkg.BuiltConfig\n")
    (docs / "_build/page.md").write_text("```{config-options} pkg.BuiltConfig\n```\n")
    read = []
    read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text",
                        lambda self, *args, **kwargs: read.append(self) or read_text(self, *args, **kwargs))
    targets = _config_option_targets(docs)
    assert sorted(targets) == ["pkg.ApiConfig", "pkg.GuideConfig", "pkg.PageConfig"]
    assert sorted(path.relative_to(docs).as_posix() for path in read) == [
        "api/page.md", "guide/page.rst", "page.md"]


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


@pytest.fixture
def tables():
    pytest.importorskip("docutils", reason="documentation extra required")
    spec = spec_from_file_location("api_parameters_tables", ROOT / "docs/_ext/method_tables.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_complete(tables, objects):
    problems = []
    for name, obj in objects.items():
        try:
            tables._check_api_object(name, obj)
        except ValueError as error:
            problems.append(str(error))
    assert not problems, "\n".join(problems)


def test_api_objects_have_complete_parameters_and_returns(tables):
    objects = tables._api_objects(ROOT / "docs/api")
    assert objects, "No API objects discovered"
    _assert_complete(tables, objects)


def documented(value, *items, **options) -> int:
    """A callable fixture.

    Parameters
    ----------
    value : int
        Input value.
    *items : tuple
        Additional items.
    **options : dict
        Named options.

    Returns
    -------
    int
        Input value.
    """
    return value


@pytest.mark.parametrize("change, expected", [
    (lambda doc: doc.replace("    value : int\n        Input value.\n", ""), "missing Parameters parameter 'value'"),
    (lambda doc: doc.replace("value : int", "typo : int"), "unknown Parameters parameter 'typo'"),
    (lambda doc: doc.replace("    *items : tuple", "    value : int\n        Again.\n    *items : tuple"), "duplicate Parameters parameter 'value'"),
    (lambda doc: doc.replace("        Input value.\n", ""), "empty Parameters description for parameter 'value'"),
    (lambda doc: doc + "\n    Parameters\n    ----------\n    value : int\n        Again.\n", "duplicate Parameters parameter 'value'"),
    (lambda doc: doc.replace("**options : dict", "options : dict"), "missing Parameters parameter '**options'"),
    (lambda doc: doc[:doc.index("    Returns")], "missing Returns or Yields"),
])
def test_contract_rejects_incomplete_callable(tables, monkeypatch, change, expected):
    monkeypatch.setattr(documented, "__doc__", change(documented.__doc__))
    with pytest.raises(AssertionError) as error:
        _assert_complete(tables, {"fixture.documented": documented})
    assert "fixture.documented" in str(error.value)
    assert expected in str(error.value)
    print(f"Deliberately broken coverage fixture: {error.value}")


def test_public_fields_and_return_annotations(tables):
    from dataclasses import dataclass

    @dataclass
    class Record:
        """Record fixture.

        Parameters
        ----------
        value : int
            Public value.
        """
        value: int
        _cache: int = 0

    tables._check_api_object("fixture.Record", Record)
    tables._check_api_object("fixture.documented", documented)
    for annotation in (None, "None", type(None), "NoneType"):
        def no_result():
            pass
        no_result.__annotations__["return"] = annotation
        tables._check_api_object("fixture.no_result", no_result)
    def unannotated():
        return 1
    tables._check_api_object("fixture.unannotated", unannotated)
    def generator() -> "Iterator[int]":
        """Yield a value.

        Yields
        ------
        int
            Value.
        """
        yield 1
    tables._check_api_object("fixture.generator", generator)


def test_api_discovery_follows_each_module_and_defined_methods(tables, tmp_path, monkeypatch):
    from dataclasses import dataclass
    from enum import Enum
    from types import ModuleType
    import sys

    class Parent:
        def inherited(self):
            pass

    @dataclass
    class Record(Parent):
        value: int

        def method(self, value):
            pass

        @classmethod
        def create(cls, value):
            pass

        @staticmethod
        def static(value):
            pass

        @property
        def prop(self):
            return 1

        def _private(self):
            pass

    class Choice(Enum):
        ONE = 1

    first = ModuleType("api_first")
    first.Record = Record
    first.Error = ValueError
    first.Warning = UserWarning
    first.Choice = Choice
    first.CONSTANT = 1
    second = ModuleType("api_second")
    second.function = documented
    monkeypatch.setitem(sys.modules, first.__name__, first)
    monkeypatch.setitem(sys.modules, second.__name__, second)
    (tmp_path / "api.rst").write_text(""".. currentmodule:: api_first

.. autosummary::
   :toctree: generated

   Record
   Error
   Warning
   Choice
   CONSTANT

.. currentmodule:: api_second

.. autosummary::

   function
""")
    objects = tables._api_objects(tmp_path)
    assert set(objects) == {"api_first.Record", "api_first.Record.method",
                            "api_first.Record.create", "api_first.Record.static",
                            "api_second.function"}
    for name in ("method", "create", "static"):
        assert "value" in tables.inspect.signature(objects[f"api_first.Record.{name}"]).parameters


@pytest.mark.parametrize("kind", ["function", "method", "dataclass", "returns"])
def test_strict_build_rejects_broken_api(tables, tmp_path, kind):
    """A real Sphinx build rejects broken API documentation."""
    import subprocess
    import sys

    pytest.importorskip("sphinx", reason="documentation extra required")
    source = tmp_path / "source"
    (source / "api").mkdir(parents=True)
    fixtures = {
        "function": 'def broken(value):\n    """Broken function."""\n',
        "method": 'class Record:\n    def broken(self, value):\n        """Broken method."""\n',
        "dataclass": 'from dataclasses import dataclass\n@dataclass\nclass Record:\n    """Broken record."""\n    value: int\n',
        "returns": 'def broken() -> int:\n    """Broken return section."""\n    return 1\n',
    }
    (source / "fixture.py").write_text(fixtures[kind])
    (source / "conf.py").write_text(
        f"import sys\nsys.path[:0] = {[str(source), str(ROOT / 'docs/_ext')]!r}\n"
        "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'method_tables']\nmaster_doc = 'index'\n")
    (source / "index.rst").write_text("Fixture\n=======\n\n.. toctree::\n\n   api/fixture\n")
    target = "Record" if kind in ("method", "dataclass") else "broken"
    (source / "api/fixture.rst").write_text(
        "API fixture\n===========\n\n.. currentmodule:: fixture\n\n.. autosummary::\n\n   " + target + "\n\n"
        + (".. autoclass:: Record\n   :members:\n" if target == "Record" else ".. autofunction:: broken\n"))
    result = subprocess.run([sys.executable, "-m", "sphinx", "-n", "-W", "--keep-going",
                             "-b", "html", str(source), str(tmp_path / "html")],
                            capture_output=True, text=True, timeout=60)
    output = result.stdout + result.stderr
    print(f"Strict-build {kind} fixture (exit {result.returncode}):\n{output}")
    assert result.returncode != 0
    qualified = "fixture.Record.broken" if kind == "method" else f"fixture.{target}"
    assert qualified in output
    assert ("missing Returns or Yields" if kind == "returns" else "'value'") in output

    assert output.count("WARNING:") == 1, output
    doc_target = "Record.broken" if kind == "method" else target
    section = ("\n\nReturns\n-------\nint\n    Result.\n" if kind == "returns" else
               "\n\nParameters\n----------\nvalue : int\n    Input value.\n")
    with (source / "fixture.py").open("a") as stream:
        stream.write(f"\n{doc_target}.__doc__ += {section!r}\n")
    repaired = subprocess.run([sys.executable, "-m", "sphinx", "-E", "-n", "-W",
                               "--keep-going", "-b", "html", str(source), str(tmp_path / "html")],
                              capture_output=True, text=True, timeout=60)
    print(f"Repaired strict-build {kind} fixture (exit {repaired.returncode}):\n"
          + repaired.stdout + repaired.stderr)
    assert repaired.returncode == 0, repaired.stdout + repaired.stderr
