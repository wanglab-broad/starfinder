"""Registration module for image alignment."""

from starfinder.registration._skimage_backend import phase_correlate_skimage

# Local registration exports (lazy import - SimpleITK optional)
# Import will fail gracefully with helpful message if SimpleITK missing
from starfinder.registration.demons import (
    apply_deformation,
    demons_register,
    register_volume_local,
)
from starfinder.registration.phase_correlation import (
    apply_shift,
    phase_correlate,
    register_volume,
)

__all__ = [
    # Global (rigid)
    "phase_correlate",
    "apply_shift",
    "register_volume",
    "phase_correlate_skimage",
    # Local (non-rigid)
    "demons_register",
    "apply_deformation",
    "register_volume_local",
]
