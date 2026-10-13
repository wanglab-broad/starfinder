starfinder.report
=================

The processing report (§2.11): what the records of one sample's processing hold. See
:doc:`../report-contract` and :doc:`../report-algorithms`.

The report describes a run and judges nothing: it gives no pass or fail verdict and no
composite score, and it filters, corrects or rewrites nothing. It reads the records at their
fixed places under the output root (each FOV's ``checkpoints/<fov>/run.json``, with
``registered/transforms.json`` for a run record written before ``transforms``; the placement
``stitching/<sample>/``; the per-FOV ``segmentation/<run>/`` and ``assignment/<name>/``
records; the export ``export/<sample>/``), checks each file against the SHA-256 a record holds
for it, and writes its own folder ``<OUTPUT>/report/<sample>/``: ``tables/`` (CSV with the
checkpoint table writer) and the record ``report.json``, which lists every file read
(``inputs``) and every file written (``files``, with each table's dtypes). Before anything is
written the checks of "Checks before anything is written" run in order, each raising with the
folder unchanged; a second report of the sample replaces the first as a whole once it is
complete, and a file of the folder that the earlier ``report.json`` does not list stops it
with ``FileExistsError`` (RP1).

A value whose input is missing is never zero: it is null and named by one row of
``tables/unavailable.csv`` (and of ``report.json`` ``unavailable``) with one reason of
:py:data:`~starfinder.report.UNAVAILABLE_REASONS`, the first of them that applies (P1), for a
part of :py:data:`~starfinder.report.REPORT_PARTS`; a value that does not exist in its case is
null with no row. :py:data:`~starfinder.report.REPORT_TABLES` are the tables of the folder and
:py:data:`~starfinder.report.PROBLEM_RULES` the rules of the problematic FOVs.
:py:class:`~starfinder.report.ReportConfig`, with
:py:class:`~starfinder.report.MarkerGroup`, :py:class:`~starfinder.report.ReportLimits` and
:py:class:`~starfinder.report.RuleLimit`, is what the user supplies: the marker groups and
pairs, the positive threshold, the example FOVs, the provisional limits of the problematic-FOV
rules and the size of the whole-sample figures. :py:func:`~starfinder.report.read_report`
reads a report's ``report.json`` back, checking every file it lists against its SHA-256.

``Dataset.report(sample, fov_ids, *, assignment="default", segmentation=None, config=None,
output_dir=None)`` is the entry: it writes the folder (by default ``<OUTPUT>/report/<sample>/``)
and returns a :py:class:`~starfinder.report.ReportRecord`, the folder and its ``report.json``.
It is a reader, not a runner: it calls no processing stage, reads the saved records at the places
of :doc:`../report-contract` ("Where each record is looked for") and the raw input of the main
example for its replay, and writes only its own folder. The folder holds every table of
:py:data:`~starfinder.report.REPORT_TABLES` (the values each definition of
:doc:`../report-algorithms` computes: the read fractions, the shifts and their residuals, the
alignment score, the area a FOV owns and the cell density, the problematic FOVs, the example
FOVs, the intensity distributions of the replayed main example, the reads assigned to cells,
the per-cell values, MECR and the reads along Z), ``report.json`` and the page
``report.html``.

The page is built from the template shipped with the package
(``starfinder/report/templates/report.html``): eight sections (Summary, Workflow overview,
Preprocessing, Registration, Readout, Segmentation and assembly, Sample measurements, Files) in
the layout of :doc:`../report-algorithms` ("Layout of the page"). The tables are embedded as one
JSON block, from which the page's inline script draws the FOV maps (plain SVG), their tooltips
and the paged tables; the static figures are matplotlib PNG and JPEG images embedded as
``data:`` addresses; a part that is unavailable shows "Unavailable:" and its reason in its
place, and a value unavailable for one FOV a dash with its reason in the tooltip. The page has a
light theme only and loads nothing, so it opens offline. ``report.json`` records the page's
bytes and its size budget (6 MiB + 16 KiB per FOV + 3 MiB per marker figure); a page above the
budget is still written.

.. currentmodule:: starfinder.report

.. autosummary::
   :toctree: generated

   MarkerGroup
   PROBLEM_RULES
   read_report
   REPORT_PARTS
   REPORT_TABLES
   ReportConfig
   ReportLimits
   ReportRecord
   RuleLimit
   UNAVAILABLE_REASONS
