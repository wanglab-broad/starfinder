"""Detection evaluation against supplied, explicitly eligible truth."""
from .matching import match_points

__all__ = ["evaluate_spots"]


def evaluate_spots(detected, truth, **matching_config):
    """Evaluate supplied (N,3) detections against truth; see match_points.

    All matching policy, threshold, units and geometry arguments are required
    by match_points. Reference means truth, observed means detected.
    """
    return match_points(truth, detected, **matching_config)
