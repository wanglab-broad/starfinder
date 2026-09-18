"""Private benchmark report formatting; undefined metrics stay undefined."""
from starfinder.evaluation import EvaluationResult


def _print_quality_report(report: EvaluationResult) -> None:
    """Print supplied metrics without adding thresholds or scientific claims."""
    print("REGISTRATION QUALITY REPORT")
    for name, value in report.values.items():
        rendered = "undefined" if value is None else f"{value:.4f}"
        reason = report.reasons.get(name, "")
        print(f"{name}: {rendered} {report.units[name]} {reason}".rstrip())


def _e2e_summary(shifts, spots, decoding):
    """Select supplied canonical results for a benchmark summary."""
    return {"shift_max_error": shifts.values["max_error"],
            "shift_passed": shifts.values["passed"],
            "spot_recall": spots.values["recall"],
            "spot_precision": spots.values["precision"],
            "spot_mean_distance": spots.values["mean_distance"],
            **decoding.values}
