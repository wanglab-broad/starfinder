"""Non-rigid registration using demons algorithm (SimpleITK)."""

from __future__ import annotations


def _import_sitk():
    """Lazy import SimpleITK with helpful error message."""
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError:
        raise ImportError(
            "SimpleITK required for local registration. "
            "Install with: uv add 'starfinder[local-registration]'"
        )
