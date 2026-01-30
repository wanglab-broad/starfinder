"""Registration module for image alignment."""

from starfinder.registration.phase_correlation import (
    phase_correlate,
    apply_shift,
    register_volume,
)
from starfinder.registration._skimage_backend import phase_correlate_skimage

__all__ = [
    "phase_correlate",
    "apply_shift",
    "register_volume",
    "phase_correlate_skimage",
]
