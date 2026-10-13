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

The tables written so far are ``fovs``, ``registration``, ``steps``, ``readout_totals``,
``population``, ``problematic_fovs`` (header only) and ``unavailable``, with the columns that
copy record fields; the computed values (the fractions, the shifts and residuals, the
alignment score, the area a FOV owns, the reads in cells), the example FOVs, the replay, the
sample measurements, the page and the entry ``Dataset.report`` are added by the definitions of
:doc:`../report-algorithms` that make them.

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
   RuleLimit
   UNAVAILABLE_REASONS
