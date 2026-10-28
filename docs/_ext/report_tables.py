"""Render processing-report column names from the code's saved-table schemas."""
from __future__ import annotations

from docutils.parsers.rst import Directive

from method_tables import _table


class ReportTables(Directive):
    """The sample and comparison CSV files and their columns."""

    has_content = False

    def run(self):
        from starfinder.report._schema import REPORT_TABLES, SAMPLES_DTYPES, TABLE_DTYPES

        schemas = [(name, TABLE_DTYPES[name]) for name in REPORT_TABLES]
        schemas.append(("samples", SAMPLES_DTYPES))
        rows = [[f"{name}.csv", ", ".join(f"``{column}``" for column in columns)]
                for name, columns in schemas]
        return [_table(["File", "Columns"], rows, literal_columns={0})]


def setup(app):
    app.add_directive("report-tables", ReportTables)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
