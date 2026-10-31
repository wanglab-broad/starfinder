"""Schema-generated workflow reference and incremental rebuild coverage; the workflow package's tables against the
contract's (design/specs/orchestration.md)."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
import yaml

from . import docs_pages

pytestmark = [pytest.mark.workflow, pytest.mark.contract]
ROOT = Path(__file__).resolve().parents[3]


def test_schema_tables_render_fields_and_rebuild_when_the_schema_changes(tmp_path, monkeypatch):
    """Render a small schema, then change only its input and check the rebuilt HTML."""
    sphinx = pytest.importorskip("sphinx.application", reason="documentation extra required")
    monkeypatch.syspath_prepend(str(ROOT / "docs/_ext"))
    source = tmp_path / "docs"
    source.mkdir()
    schema_path = tmp_path / "workflow/schemas/config.schema.yaml"
    schema_path.parent.mkdir(parents=True)
    schema = {
        "properties": {"job": {"$ref": "#/$defs/job"}, "flag": {"type": "boolean", "default": False}},
        "$defs": {
            "base": {"properties": {"threads": {"type": "integer", "default": 1}}},
            "job": {"allOf": [{"$ref": "#/$defs/base"}], "properties": {
                "file": {"type": "string", "description": "Image <path> & name"},
                "view": {"type": "object", "properties": {"axis": {"enum": ["z", "channel"]}}},
            }},
        },
    }
    schema_path.write_text(yaml.safe_dump(schema, sort_keys=False))
    (source / "conf.py").write_text("extensions = ['workflow_config']\nmaster_doc = 'index'\n")
    (source / "index.rst").write_text(
        "Workflow\n========\n\n.. workflow-config::\n\nDefinitions\n-----------\n\n.. workflow-config:: #/$defs\n"
    )
    warnings = io.StringIO()
    app = sphinx.Sphinx(str(source), str(source), str(tmp_path / "html"), str(tmp_path / "doctrees"),
                        "html", status=io.StringIO(), warning=warnings, freshenv=True)
    app.build()
    assert not warnings.getvalue(), warnings.getvalue()
    html = (tmp_path / "html/index.html").read_text()
    assert 'schema-job' in html
    for expected in ("threads", "view.axis", "false", "Image &lt;path&gt; &amp; name"):
        assert expected in html
    assert str(schema_path) in {str(p) for p in app.env.dependencies["index"]}
    schema["properties"]["new_field"] = {"type": "string", "description": "Added without editing the page"}
    schema_path.write_text(yaml.safe_dump(schema, sort_keys=False))
    # Set a later input timestamp without a sleep; incremental Sphinx uses its dependency mtime.
    import os
    import time
    later = time.time() + 2
    os.utime(schema_path, (later, later))
    app.build()
    assert not warnings.getvalue(), warnings.getvalue()
    html = (tmp_path / "html/index.html").read_text()
    assert "new_field" in html
    assert "Added without editing the page" in html


def _cells(row):
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def test_the_blocks_of_each_task_kind_are_the_contracts():
    """The package's map from task kind to blocks equals the contract's table ("The blocks of a task kind")."""
    from starfinder.workflow import TASK_BLOCKS
    rows = docs_pages.table(docs_pages.design("specs/orchestration.md"), "orchestration-task-blocks").splitlines()[2:]
    contract = {}
    for row in rows:
        kinds, blocks = _cells(row)[:2]
        for kind in (k.strip(" `") for k in kinds.split(",")):
            contract[kind] = tuple(b.strip(" `") for b in blocks.split(","))
    assert dict(TASK_BLOCKS) == contract


def test_every_top_level_key_the_contract_removes_is_refused_by_the_plan():
    """Each top-level key the migration table removes (or keeps for the MATLAB dialect only) is a key the plan refuses
    with the place it went."""
    from starfinder.workflow._shape import LEFT_KEYS
    rows = docs_pages.table(docs_pages.design("specs/orchestration.md"), "orchestration-migration").splitlines()[2:]
    import re
    removed = set()
    for row in rows:
        key, proposed = _cells(row)[:2]
        names = re.findall(r"`([^`]+)`", key.split("(")[0])
        if proposed in ("Removed", "MATLAB dialect only") and names and re.fullmatch(r"\w+", names[0]):
            removed |= {name for name in names if re.fullmatch(r"\w+", name)}
    assert removed and removed <= set(LEFT_KEYS)


def test_the_migration_guide_renders_the_contracts_migration_table(monkeypatch):
    """docs/migration.md generates its table from the contract's (the directive workflow-migration): every row, in
    order, as the table cells of the site's generated tables. The page itself is built by the strict build."""
    pytest.importorskip("sphinx", reason="documentation extra required")
    monkeypatch.syspath_prepend(str(ROOT / "docs/_ext"))
    import method_tables
    import workflow_config
    text = docs_pages.design("specs/orchestration.md")
    header, rows = workflow_config.labelled_table(text, "orchestration-migration")
    record = docs_pages.table(text, "orchestration-migration").splitlines()
    assert header == [cell.replace("`", "``") for cell in _cells(record[0])]
    assert rows == [[cell.replace("`", "``") for cell in _cells(line)] for line in record[2:]]
    table = method_tables._table(header, rows, literal_columns=set())
    assert len(list(table.findall(lambda node: node.tagname == "row"))) == len(rows) + 1
    assert "```{workflow-migration}" in docs_pages.section(docs_pages.read("migration.md"), "workflow-python-shape")
