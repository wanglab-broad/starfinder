"""Registration module for image alignment."""

from starfinder.registration._skimage_backend import phase_correlate_skimage

# Local registration exports (lazy import - SimpleITK optional)
# Import will fail gracefully with helpful message if SimpleITK missing
from starfinder.registration.demons import (
    apply_deformation,
    demons_register,
    matlab_compatible_config,
    register_volume_local,
)
from starfinder.registration.metrics import (
    detect_spots,
    normalized_cross_correlation,
    print_quality_report,
    registration_quality_report,
    spot_colocalization,
    spot_matching_accuracy,
    structural_similarity,
)
from starfinder.registration.phase_correlation import (
    apply_shift,
    phase_correlate,
    register_volume,
)
from starfinder.registration.pointset import (
    cpd_register,
    register_volume_cpd,
    register_volume_tps,
    tps_register,
)

__all__ = [
    # Global (rigid)
    "phase_correlate",
    "apply_shift",
    "register_volume",
    "phase_correlate_skimage",
    # Local (non-rigid) — demons
    "demons_register",
    "apply_deformation",
    "register_volume_local",
    "matlab_compatible_config",
    # Local (non-rigid) — TPS
    "tps_register",
    "register_volume_tps",
    # Local (non-rigid) — CPD
    "cpd_register",
    "register_volume_cpd",
    # Quality metrics
    "normalized_cross_correlation",
    "structural_similarity",
    "spot_colocalization",
    "spot_matching_accuracy",
    "detect_spots",
    "registration_quality_report",
    "print_quality_report",
]
