"""Schema-generated workflow reference and incremental rebuild coverage."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
import yaml

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
