"""Private benchmark report formatting; undefined metrics stay undefined.

The synthetic inspection image moved to ``starfinder.synthetic._storage`` (R7).
"""
from starfinder.evaluation import EvaluationResult


def _print_quality_report(report: EvaluationResult) -> None:
    """Print supplied metrics without adding thresholds or scientific claims."""
    print("REGISTRATION QUALITY REPORT")
    for name, value in report.values.items():
        rendered = "undefined" if value is None else f"{value:.4f}"
        reason = report.reasons.get(name, "")
        print(f"{name}: {rendered} {report.units[name]} {reason}".rstrip())
