"""The processing report (§2.11): what the records of one sample's processing hold.

The report describes a run and judges nothing: no pass or fail verdict, no composite
score, no filtering or correction. It reads the records at their fixed places under
the output root (the per-FOV run records, the saved transforms, the placement, the
per-FOV assignments and the export) and writes its own folder
``<OUTPUT>/report/<sample>/`` with ``tables/`` and the record ``report.json``; a
value whose input is missing is unavailable with one reason of the closed list
:data:`UNAVAILABLE_REASONS`, recorded once in ``tables/unavailable.csv``.
:class:`ReportConfig` (with :class:`MarkerGroup`, :class:`ReportLimits` and
:class:`RuleLimit`) is what the user supplies; :data:`REPORT_PARTS`,
:data:`REPORT_TABLES` and :data:`PROBLEM_RULES` are the parts of the page, the tables
of the folder and the rules of the problematic FOVs; :func:`read_report` reads a
report's record back, checking every file it lists. ``Dataset.report`` writes a sample's
report (the page ``report.html``, ``report.json`` and ``tables/``) and returns a
:class:`ReportRecord`. See docs/report-contract.md and docs/report-algorithms.md.
"""
from ._config import MarkerGroup, ReportConfig, ReportLimits, RuleLimit
from ._schema import PROBLEM_RULES as _PROBLEM_RULES
from ._schema import REPORT_PARTS as _REPORT_PARTS
from ._schema import REPORT_TABLES as _REPORT_TABLES
from ._schema import UNAVAILABLE_REASONS as _UNAVAILABLE_REASONS
from ._schema import ReportRecord, read_report

#: The reasons a value or a part of the report can be unavailable, in the order that decides which one a value
#: gets when several apply (P1): the first of them in this order.
UNAVAILABLE_REASONS = _UNAVAILABLE_REASONS
#: The parts of the report page, in page order; a row of ``unavailable.csv`` names one of them, or
#: ``rule:<rule>`` for a rule of :data:`PROBLEM_RULES` not evaluated for the whole sample.
REPORT_PARTS = _REPORT_PARTS
#: The tables of the report folder's ``tables/``, in folder order (``tables/<name>.csv``).
REPORT_TABLES = _REPORT_TABLES
#: The rules of the problematic FOVs, in the order of their rows in ``problematic_fovs.csv``.
PROBLEM_RULES = _PROBLEM_RULES

__all__ = ["MarkerGroup", "PROBLEM_RULES", "read_report", "REPORT_PARTS", "REPORT_TABLES", "ReportConfig",
           "ReportLimits", "ReportRecord", "RuleLimit", "UNAVAILABLE_REASONS"]
