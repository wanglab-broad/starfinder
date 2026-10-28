"""Render benchmark profile names from the maintained recipe JSON files."""
from __future__ import annotations

import json
from pathlib import Path

from docutils.parsers.rst import Directive

from method_tables import _table


class BenchmarkProfiles(Directive):
    """List registration methods or pipeline execution modes without copying options."""

    required_arguments = 1
    has_content = False

    def run(self):
        task = self.arguments[0]
        if task not in {"registration", "pipeline"}:
            raise self.error("benchmark-profiles: use registration or pipeline")
        source = Path(self.state.document.settings.env.srcdir).parent / "benchmarks" / "configs" / f"{task}.json"
        self.state.document.settings.env.note_dependency(str(source))
        profiles = json.loads(source.read_text())["profiles"]
        header = ["Profile", "Method" if task == "registration" else "Execution"]
        rows = [[name, profile["registration"]["method"] if task == "registration" else profile["execution"]]
                for name, profile in profiles.items()]
        return [_table(header, rows, literal_columns={0, 1})]


def setup(app):
    app.add_directive("benchmark-profiles", BenchmarkProfiles)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
