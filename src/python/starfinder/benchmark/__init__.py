"""Private entries behind the experiment commands: trials, expansion, evaluate-only adapters and the report.

``starfinder experiment expand|evaluate|report`` and ``starfinder trial run`` (``starfinder.__main__``) are the
public interface; this package exports no names. The old benchmark lifecycle (``BenchmarkCase``,
``BenchmarkTrialResult``, ``run_benchmark``, ``evaluate_benchmark``, ``report_benchmark``) was removed in W-781;
docs/migration.md names each replacement.
"""
